"""Adapter between the ROS 2 graph and the STM32 over the Arduino Router Bridge.

Design points worth knowing before editing:

* **The command sender runs on its own thread, not a ROS timer.** `Bridge.call`
  blocks until the MCU responds. Running it inside an executor callback would
  couple actuation latency to whatever else the executor is doing, and a single
  slow round-trip would stall subscriptions. The thread reads the latest command
  from a holder; it never queues, because a backlog of stale drive commands is
  exactly what you do not want.

* **Arming is reconciled, not commanded.** The arbiter publishes what the system
  should be (`RobotState.want_armed`); this node repeatedly nudges the MCU
  toward that state. A dropped arm message self-heals on the next tick, and the
  MCU's latching FAILSAFE is cleared by the natural disarm-then-arm sequence.

* **Unit conversion lives here.** The MCU speaks normalised [-1, 1]. ROS speaks
  radians and m/s. Keeping the mapping in a ROS parameter means recalibrating
  the vehicle never requires reflashing the microcontroller.
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass

import rclpy
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Imu

from mobilebot_msgs.msg import RobotState, McuStatus

# The vendored client, dropped in beside this file. The fallback covers running
# outside the container against a host-installed arduino package.
try:
    from mobilebot_bridge.app_utils import Bridge
except ImportError:  # pragma: no cover
    from arduino.app_utils import Bridge


def clamp(value: float, limit: float) -> float:
    return -limit if value < -limit else (limit if value > limit else value)


@dataclass
class DriveCommand:
    steering_rad: float = 0.0
    speed_mps: float = 0.0


class LatestCommand:
    """Thread-safe single-slot holder. Newest value wins; nothing is queued."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = DriveCommand()

    def set(self, command: DriveCommand) -> None:
        with self._lock:
            self._value = command

    def get(self) -> DriveCommand:
        with self._lock:
            return self._value


class RoundTripStats:
    """Rolling min/mean/max of Bridge round-trip time, in milliseconds."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._count = 0
        self._sum = 0.0
        self._min = float('inf')
        self._max = 0.0
        self.errors = 0
        self.mismatches = 0

    def record(self, rtt_ms: float) -> None:
        self._count += 1
        self._sum += rtt_ms
        self._min = min(self._min, rtt_ms)
        self._max = max(self._max, rtt_ms)

    def summary(self) -> str:
        if self._count == 0:
            return f'no successful calls (errors={self.errors})'
        return (f'rtt min/avg/max {self._min:.1f}/{self._sum / self._count:.1f}/'
                f'{self._max:.1f} ms over {self._count} calls, '
                f'errors={self.errors}, seq_mismatch={self.mismatches}')


class HardwareBridge(LifecycleNode):

    def __init__(self) -> None:
        super().__init__('hardware_bridge')

        self.declare_parameters('', [
            ('command_rate_hz', 50.0),
            ('max_steer_rad', 0.436),
            ('speed_to_throttle', 2.0),
            ('imu_frame_id', 'imu_link'),
            ('handshake_attempts', 10),
            ('arm_retry_period_s', 0.2),
            ('stats_period_s', 5.0),
        ])

        self._latest = LatestCommand()
        self._want_armed = False
        self._mcu_state = McuStatus.STATE_INIT
        self._seq = 0
        self._stats = RoundTripStats()

        self._sender: threading.Thread | None = None
        self._running = threading.Event()
        self._last_arm_attempt = 0.0
        self._last_stats_log = 0.0

        self._imu_pub = None
        self._status_pub = None

    # -- lifecycle ---------------------------------------------------------

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        p = self.get_parameter
        self._period = 1.0 / float(p('command_rate_hz').value)
        self._max_steer = float(p('max_steer_rad').value)
        self._speed_to_throttle = float(p('speed_to_throttle').value)
        self._imu_frame = str(p('imu_frame_id').value)
        self._arm_retry = float(p('arm_retry_period_s').value)
        self._stats_period = float(p('stats_period_s').value)

        reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST, depth=1)
        sensor = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST, depth=10)

        self._imu_pub = self.create_publisher(Imu, 'imu/data', sensor)
        self._status_pub = self.create_publisher(McuStatus, 'mcu/status', reliable)

        self.create_subscription(AckermannDriveStamped, 'cmd_drive',
                                 self._on_cmd_drive, reliable)
        self.create_subscription(RobotState, 'robot/state',
                                 self._on_robot_state, reliable)

        # Register the MCU -> Linux notification handlers before the handshake,
        # so no telemetry is dropped in the window between them.
        Bridge.provide('imu_sample', self._on_imu_sample)
        Bridge.provide('mcu_status', self._on_mcu_status)

        if not self._handshake(int(p('handshake_attempts').value)):
            self.get_logger().error('MCU handshake failed; staying unconfigured')
            return TransitionCallbackReturn.FAILURE

        self.get_logger().info('MCU handshake OK')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self._stats.reset()
        self._running.set()
        self._sender = threading.Thread(target=self._sender_loop,
                                        name='bridge_sender', daemon=True)
        self._sender.start()
        self.get_logger().info('hardware_bridge active')
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self._stop_sender()
        self._safe_shutdown()
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self._stop_sender()
        self._safe_shutdown()
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self._stop_sender()
        self._safe_shutdown()
        return TransitionCallbackReturn.SUCCESS

    def _stop_sender(self) -> None:
        self._running.clear()
        if self._sender is not None:
            self._sender.join(timeout=1.0)
            self._sender = None

    def _safe_shutdown(self) -> None:
        """Leave the MCU disarmed and neutral, whatever else happened."""
        try:
            Bridge.call('set_drive', 0.0, 0.0, self._next_seq())
            Bridge.call('set_armed', False)
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            self.get_logger().warn(f'safe shutdown call failed: {exc}')

    # -- handshake ---------------------------------------------------------

    def _handshake(self, attempts: int) -> bool:
        for attempt in range(attempts):
            token = 0xA5A50000 | (attempt & 0xFFFF)
            try:
                echoed = Bridge.call('ping', token)
                if int(echoed) == token:
                    return True
                self.get_logger().warn(
                    f'ping echo mismatch: sent {token:#x} got {int(echoed):#x}')
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f'ping attempt {attempt} failed: {exc}')
            time.sleep(0.2)
        return False

    # -- ROS callbacks -----------------------------------------------------

    def _on_cmd_drive(self, msg: AckermannDriveStamped) -> None:
        self._latest.set(DriveCommand(msg.drive.steering_angle, msg.drive.speed))

    def _on_robot_state(self, msg: RobotState) -> None:
        self._want_armed = msg.want_armed

    # -- Bridge callbacks (MCU -> Linux, on the Bridge's own thread) --------

    def _on_imu_sample(self, ax: float, ay: float, az: float,
                       gx: float, gy: float, gz: float, seq: int) -> None:
        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._imu_frame

        msg.linear_acceleration.x = float(ax)
        msg.linear_acceleration.y = float(ay)
        msg.linear_acceleration.z = float(az)
        msg.angular_velocity.x = float(gx)
        msg.angular_velocity.y = float(gy)
        msg.angular_velocity.z = float(gz)

        # No orientation estimate is produced. The convention for "this field is
        # not filled in" is a leading -1 in the covariance.
        msg.orientation_covariance[0] = -1.0

        self._imu_pub.publish(msg)

    def _on_mcu_status(self, state: int, last_seq: int, cmd_age_ms: int,
                       servo_deg: float, throttle: float, dropped: int) -> None:
        self._mcu_state = int(state)

        msg = McuStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.state = int(state)
        msg.last_seq = int(last_seq)
        msg.cmd_age_ms = int(cmd_age_ms)
        msg.servo_deg = float(servo_deg)
        msg.throttle = float(throttle)
        msg.dropped = int(dropped)
        self._status_pub.publish(msg)

    # -- sender thread -----------------------------------------------------

    def _next_seq(self) -> int:
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        return self._seq

    def _to_normalised(self, command: DriveCommand) -> tuple[float, float]:
        steering = clamp(command.steering_rad / self._max_steer, 1.0)
        throttle = clamp(command.speed_mps * self._speed_to_throttle, 1.0)
        return steering, throttle

    def _sender_loop(self) -> None:
        self._elevate_priority()
        next_tick = time.monotonic()
        while self._running.is_set():
            next_tick += self._period
            self._send_once()
            self._reconcile_arm_state()
            self._maybe_log_stats()

            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # Overran the budget; resync rather than accumulate debt.
                next_tick = time.monotonic()

    def _elevate_priority(self) -> None:
        """Request SCHED_FIFO for this thread so the 50 Hz set_drive path is
        never starved by policy_node/data_recorder/etc competing for CPU.

        Measured motivation: RTT went from a 2.24 ms baseline to 24-28 ms
        average (spikes to 146 ms) once the full graph -- policy_node included
        -- ran concurrently on 4 shared cores. 146 ms is uncomfortably close to
        the MCU's 150 ms watchdog. This is the actuation path; it should
        preempt best-effort perception/logging work, not compete with it.

        os.sched_setscheduler with pid=0 sets THIS thread specifically (each
        Linux thread has its own scheduling context), not the whole process --
        callback threads and the executor stay on the normal scheduler.
        Requires CAP_SYS_NICE (root, which this container runs as); fails soft
        with a logged warning otherwise, since a missing capability should
        degrade performance, not crash the bridge.
        """
        try:
            priority = os.sched_get_priority_max(os.SCHED_FIFO)
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
            self.get_logger().info(
                f'sender thread elevated to SCHED_FIFO priority {priority}')
        except (PermissionError, OSError) as exc:
            self.get_logger().warn(
                f'could not elevate sender thread priority ({exc}) -- '
                'set_drive RTT will degrade under CPU contention. '
                'Requires CAP_SYS_NICE / root.')

    def _send_once(self) -> None:
        steering, throttle = self._to_normalised(self._latest.get())
        seq = self._next_seq()
        started = time.monotonic()
        try:
            echoed = Bridge.call('set_drive', float(steering), float(throttle), seq)
        except Exception as exc:  # noqa: BLE001
            self._stats.errors += 1
            self.get_logger().warn(f'set_drive failed: {exc}',
                                   throttle_duration_sec=1.0)
            return

        self._stats.record((time.monotonic() - started) * 1000.0)
        if int(echoed) != seq:
            self._stats.mismatches += 1

    def _reconcile_arm_state(self) -> None:
        now = time.monotonic()
        if now - self._last_arm_attempt < self._arm_retry:
            return

        if self._mcu_state == McuStatus.STATE_FAILSAFE:
            # A latched FAILSAFE MUST be cleared with an explicit disarm
            # before any arm attempt can succeed -- the MCU only transitions
            # DISARMED -> ARMED, never FAILSAFE -> ARMED directly, so
            # set_armed(True) sent while latched is a silent no-op.
            #
            # BUG THIS FIXES: the old comparison (`armed = mcu_state ==
            # STATE_ARMED`) treated FAILSAFE and DISARMED identically -- both
            # read as "not armed". If want_armed happened to be False at the
            # exact moment the MCU entered FAILSAFE (e.g. deadman released
            # between engagements), the old code saw armed(False) ==
            # want_armed(False), concluded nothing needed doing, and never
            # sent the disarm that would have cleared the latch. Every
            # subsequent arm attempt then failed silently forever -- /cmd_drive
            # kept streaming and looked healthy, but the MCU was permanently
            # ignoring it. This branch sends disarm unconditionally whenever
            # FAILSAFE is observed, regardless of what want_armed currently is.
            self._last_arm_attempt = now
            try:
                Bridge.call('set_armed', False)
                self.get_logger().warn(
                    'MCU reported FAILSAFE -- sending disarm to clear the latch')
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f'set_armed(False) failed while clearing FAILSAFE: {exc}')
            return

        armed = self._mcu_state == McuStatus.STATE_ARMED
        if armed == self._want_armed:
            return
        self._last_arm_attempt = now
        try:
            Bridge.call('set_armed', bool(self._want_armed))
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'set_armed failed: {exc}')

    def _maybe_log_stats(self) -> None:
        now = time.monotonic()
        if now - self._last_stats_log < self._stats_period:
            return
        self._last_stats_log = now
        self.get_logger().info(self._stats.summary())
        self._stats.reset()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HardwareBridge()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
