"""Runs the trained policy on-device and publishes /policy/cmd.

This node PROPOSES; it never actuates. safety_arbiter is the sole authority
over whether /policy/cmd ever reaches the motors, and only does so in
ARMED_AUTO with the deadman still held -- see safety_arbiter.py.

Processing is synchronous, single-threaded, one frame at a time. If inference
is slower than the camera's frame period, throughput degrades to 1/inference
time rather than queueing -- that is the correct, observable failure mode for
a bench-test phase whose whole purpose is measuring whether inference is fast
enough, not something to paper over with threading.

The preprocessing_version check at startup is not optional: it is the one
mechanical guarantee that this node's crop/resize/normalisation genuinely
matches what the model was trained against. A silent mismatch here is a domain
shift with no error message anywhere -- the model loads, inference runs,
numbers come out, and the robot just drives badly for a reason nobody can see
in a log.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import CompressedImage

from mobilebot_policy.inference_backend import build_backend
from mobilebot_policy.preprocess import (CONTRACT_VERSION, PreprocessConfig,
                                         decode_jpeg, preprocess_bgr, to_nchw)

# OpenCV defaults to a multi-threaded pool sized for having the machine to
# itself, same problem as ONNX Runtime below. This is a process-level
# performance setting, not a change to any pixel computation, so it does NOT
# touch preprocess.py or its CONTRACT_VERSION -- output is bit-identical,
# only faster under contention. Set before any cv2 call, hence at import time.
import cv2
cv2.setNumThreads(1)


@dataclass
class RollingStats:
    """Same pattern as hardware_bridge's RTT tracker -- min/mean/max over a
    window, logged periodically, reset after each log. Consistency here means
    both nodes' latency numbers are directly comparable in the launch log.
    """
    count: int = 0
    total: float = 0.0
    minimum: float = float('inf')
    maximum: float = 0.0

    def record(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def summary_ms(self) -> str:
        if self.count == 0:
            return 'no frames processed'
        mean = self.total / self.count
        return (f'{self.minimum * 1000:.1f}/{mean * 1000:.1f}/{self.maximum * 1000:.1f} ms '
               f'(min/avg/max), effective {self.count / self.total:.1f} Hz')

    def reset(self) -> None:
        self.count = 0
        self.total = 0.0
        self.minimum = float('inf')
        self.maximum = 0.0


class PolicyNode(Node):

    def __init__(self) -> None:
        super().__init__('policy_node')

        self.declare_parameters('', [
            ('backend', 'onnx'),               # 'onnx' or 'dummy'
            ('model_path', ''),                # required if backend == 'onnx'
            ('max_steer_rad', 0.436),           # must match hardware_bridge/arbiter
            ('speed_setpoints_mps', [0.0, 0.1]),  # index = speed class, per your teleop step
            ('stats_period_s', 5.0),
            ('enforce_contract_version', True),  # refuse to start on a mismatch
            # ONNX Runtime's default pool is sized for having the whole
            # machine to itself; on this shared 4-core graph that causes
            # contention, not speedup. Sweep with benchmark_inference --threads
            # to find the true optimum before trusting these defaults.
            ('onnx_intra_op_threads', 1),
            ('onnx_inter_op_threads', 1),
            # 0 = process every frame. Under full-graph contention, capping
            # this trades control smoothness for CPU headroom elsewhere --
            # worth it at 0.1 m/s, where 20Hz vs 10Hz is a few mm of travel
            # between updates, if it protects the 50Hz actuation path's RTT.
            ('max_inference_hz', 0.0),
        ])

        p = self.get_parameter
        backend_kind = str(p('backend').value)
        model_path_str = str(p('model_path').value)
        self._max_steer = float(p('max_steer_rad').value)
        self._speed_setpoints = list(p('speed_setpoints_mps').value)
        self._stats_period = float(p('stats_period_s').value)
        enforce_contract = bool(p('enforce_contract_version').value)
        intra_op_threads = int(p('onnx_intra_op_threads').value)
        inter_op_threads = int(p('onnx_inter_op_threads').value)
        self._max_inference_hz = float(p('max_inference_hz').value)
        self._min_period_s = (1.0 / self._max_inference_hz
                              if self._max_inference_hz > 0 else 0.0)
        self._last_processed = 0.0

        if backend_kind == 'onnx' and not model_path_str:
            raise SystemExit('backend is "onnx" but model_path is empty -- set it in '
                            'mobilebot.yaml or pass --ros-args -p model_path:=...')

        model_path = Path(model_path_str) if model_path_str else None
        self._backend = build_backend(backend_kind, model_path,
                                      intra_op_threads, inter_op_threads)

        self._check_contract_version(enforce_contract)

        if len(self._speed_setpoints) != self._backend.num_speed_classes:
            raise SystemExit(
                f'speed_setpoints_mps has {len(self._speed_setpoints)} entries but the '
                f'model has {self._backend.num_speed_classes} speed classes -- these '
                f'must be the same length and in the same class order.')

        self._preproc_cfg = PreprocessConfig()  # frozen crop96-640x384-resize160x120-v1

        self._decode_stats = RollingStats()
        self._preprocess_stats = RollingStats()
        self._inference_stats = RollingStats()
        self._total_stats = RollingStats()
        self._last_stats_log = time.monotonic()
        self._frames_received = 0
        self._frames_failed = 0

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
        cmd_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(CompressedImage, 'image_raw/compressed',
                                 self._on_image, sensor_qos)
        self._cmd_pub = self.create_publisher(AckermannDriveStamped, 'policy/cmd', cmd_qos)

        self.get_logger().info(
            f'policy_node up: backend={backend_kind} '
            f'contract={self._backend.preprocessing_version} '
            f'speed_setpoints={self._speed_setpoints} '
            f'max_inference_hz={self._max_inference_hz or "unlimited"}')

    def _check_contract_version(self, enforce: bool) -> None:
        model_version = self._backend.preprocessing_version
        if model_version == CONTRACT_VERSION:
            self.get_logger().info(f'preprocessing contract OK: {model_version}')
            return

        message = (f'preprocessing_version MISMATCH: model was trained with '
                  f'"{model_version}", this node is running "{CONTRACT_VERSION}". '
                  f'Deploying anyway would silently feed the model images cropped/'
                  f'resized differently than it was trained on.')
        if enforce:
            raise SystemExit(message + ' Refusing to start. Set '
                            'enforce_contract_version:=false to override (not recommended).')
        self.get_logger().error(message + ' Continuing ONLY because '
                                'enforce_contract_version is false.')

    def _on_image(self, msg: CompressedImage) -> None:
        now = time.monotonic()
        if self._min_period_s and (now - self._last_processed) < self._min_period_s:
            return  # rate-capped: cheapest possible skip, before any decode work
        self._last_processed = now

        self._frames_received += 1
        t0 = time.perf_counter()

        try:
            bgr = decode_jpeg(bytes(msg.data))
        except ValueError as exc:
            self._frames_failed += 1
            self.get_logger().warn(f'decode failed, skipping frame: {exc}',
                                   throttle_duration_sec=2.0)
            return
        t1 = time.perf_counter()

        rgb = preprocess_bgr(bgr, self._preproc_cfg)
        nchw = to_nchw(rgb)
        t2 = time.perf_counter()

        result = self._backend.infer(nchw)
        t3 = time.perf_counter()

        self._decode_stats.record(t1 - t0)
        self._preprocess_stats.record(t2 - t1)
        self._inference_stats.record(result.inference_seconds)
        self._total_stats.record(t3 - t0)

        speed_class = max(0, min(result.speed_class, len(self._speed_setpoints) - 1))
        speed_mps = self._speed_setpoints[speed_class]
        steering_rad = max(-self._max_steer, min(result.steering_rad, self._max_steer))

        out = AckermannDriveStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'base_link'
        out.drive.steering_angle = steering_rad
        out.drive.speed = speed_mps
        self._cmd_pub.publish(out)

        self._maybe_log_stats()

    def _maybe_log_stats(self) -> None:
        now = time.monotonic()
        if now - self._last_stats_log < self._stats_period:
            return
        self._last_stats_log = now

        self.get_logger().info(
            f'frames {self._frames_received} (failed {self._frames_failed}) | '
            f'decode {self._decode_stats.summary_ms()} | '
            f'preprocess {self._preprocess_stats.summary_ms()} | '
            f'inference {self._inference_stats.summary_ms()} | '
            f'total {self._total_stats.summary_ms()}')

        for stats in (self._decode_stats, self._preprocess_stats,
                     self._inference_stats, self._total_stats):
            stats.reset()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
