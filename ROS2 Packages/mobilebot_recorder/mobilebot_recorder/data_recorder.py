"""Layer A capture: episode-segmented rosbag2/MCAP recording.

One session per node lifetime. One MCAP bag per episode, where an episode is
one contiguous span of RobotState.mode == MODE_ARMED_RECORD (see the sticky
recording latch in safety_arbiter -- releasing the deadman pauses an episode,
re-engaging starts the next one in the same session).

Design choices worth knowing before editing:

* **Raw passthrough subscriptions for everything except /robot/state.** Most
  topics are subscribed with raw=True: the callback receives already-serialized
  CDR bytes and the writer forwards them unchanged. No deserialize, no
  re-serialize. /robot/state is the one exception -- the recorder must read
  `.mode` to know when to open and close episodes, so it is subscribed both
  ways: typed, for that decision, and raw, for the bag.

* **rosbag2_py.SequentialWriter, not `ros2 bag record` subprocesses.** A
  spawned process pays a real startup cost every episode; with a deadman that
  can be released and re-engaged many times per session, that cost adds up and
  risks losing the first ~1 s of short episodes. The writer opens and closes
  in-process in well under a control period.

* **Image decimation is wall-clock, not header-stamp.** The source is ~28-30 Hz
  and the target is 20 Hz. Deciding "keep this frame" from arrival time avoids
  deserializing the CompressedImage just to read its stamp, which would defeat
  the point of the raw subscription for the highest-bandwidth topic.

* **Session metadata is queried, not read from YAML.** Values are pulled from
  the live parameter servers of the nodes that own them at session start, and
  written as the first message on /session/metadata in every episode bag. If a
  node is unreachable the corresponding fields are left at their zero/empty
  default and a warning is logged -- a missing value should be visible, not
  silently wrong.
"""

from __future__ import annotations

import datetime
import getpass
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.serialization import serialize_message
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.msg import ParameterType

from sensor_msgs.msg import CompressedImage, CameraInfo, Joy, Imu
from ackermann_msgs.msg import AckermannDriveStamped
from mobilebot_msgs.msg import TeleopCommand, RobotState, McuStatus, SessionMetadata

import rosbag2_py


# (topic, ros type string, python message class, QoS) for every topic
# captured in Layer A. Order here is also registration order in the bag.
#
# best_effort MUST match what the actual publisher uses, or the subscription
# is silently refused at the DDS layer -- reliability is a QoS policy where
# a RELIABLE subscriber cannot connect to a BEST_EFFORT publisher. This is
# exactly the bug that dropped imu/data entirely: hardware_bridge publishes
# it BEST_EFFORT, this table defaulted every non-image topic to RELIABLE.
@dataclass
class TopicSpec:
    name: str
    type_string: str
    msg_class: type
    best_effort: bool = False


TOPICS: list[TopicSpec] = [
    TopicSpec('session/metadata', 'mobilebot_msgs/msg/SessionMetadata', SessionMetadata),
    # No decimation: record every frame the camera delivers (~28 Hz). A
    # minimum-interval gate can only ever keep every Nth frame for integer
    # N, so gating a 28 Hz source at 50 ms silently produced 14 Hz, not the
    # intended 20 Hz. Decimate offline in `extract` instead, where a
    # fractional accumulator is trivial and no data is lost irreversibly.
    TopicSpec('image_raw/compressed', 'sensor_msgs/msg/CompressedImage', CompressedImage,
             best_effort=True),
    TopicSpec('camera/camera_info', 'sensor_msgs/msg/CameraInfo', CameraInfo),
    TopicSpec('cmd_drive', 'ackermann_msgs/msg/AckermannDriveStamped', AckermannDriveStamped),
    TopicSpec('teleop/cmd', 'mobilebot_msgs/msg/TeleopCommand', TeleopCommand),
    TopicSpec('joy', 'sensor_msgs/msg/Joy', Joy),
    TopicSpec('robot/state', 'mobilebot_msgs/msg/RobotState', RobotState),
    TopicSpec('mcu/status', 'mobilebot_msgs/msg/McuStatus', McuStatus),
    # hardware_bridge publishes this BEST_EFFORT (see hardware_bridge.py) --
    # must match here or every IMU sample is silently dropped.
    TopicSpec('imu/data', 'sensor_msgs/msg/Imu', Imu, best_effort=True),
]


def _git_commit(workspace_src: str = '.') -> str:
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                             cwd=workspace_src, capture_output=True, text=True,
                             timeout=2.0)
        return out.stdout.strip() if out.returncode == 0 else ''
    except Exception:  # noqa: BLE001
        return ''


class ParamFetcher:
    """One-shot cross-node parameter lookup over the get_parameters service.

    A missing node or a service timeout is not fatal -- it leaves the field at
    its default and lets the caller log a warning, because a recording session
    should not fail to start just because one auxiliary node hasn't come up
    yet.
    """

    def __init__(self, node: Node, timeout_s: float = 2.0) -> None:
        self._node = node
        self._timeout = timeout_s

    def fetch(self, remote_node: str, names: list[str]) -> dict:
        client = self._node.create_client(GetParameters, f'/{remote_node}/get_parameters')
        result: dict = {}
        try:
            if not client.wait_for_service(timeout_sec=self._timeout):
                self._node.get_logger().warn(
                    f'{remote_node} not reachable for parameter fetch; '
                    f'session metadata for it will be blank')
                return result

            request = GetParameters.Request(names=names)
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self._node, future, timeout_sec=self._timeout)
            if future.result() is None:
                self._node.get_logger().warn(f'{remote_node} parameter fetch timed out')
                return result

            for name, value in zip(names, future.result().values):
                result[name] = _extract(value)
        finally:
            self._node.destroy_client(client)
        return result


def _extract(param_value):
    t = param_value.type
    if t == ParameterType.PARAMETER_DOUBLE:
        return param_value.double_value
    if t == ParameterType.PARAMETER_INTEGER:
        return param_value.integer_value
    if t == ParameterType.PARAMETER_BOOL:
        return param_value.bool_value
    if t == ParameterType.PARAMETER_STRING:
        return param_value.string_value
    return None


@dataclass
class EpisodeWriter:
    """Wraps one open rosbag2 writer plus the bookkeeping the extractor needs."""

    path: Path
    writer: rosbag2_py.SequentialWriter
    started_at: datetime.datetime
    frame_count: int = 0
    last_kept_image_ns: int = 0


class DataRecorder(Node):

    def __init__(self) -> None:
        super().__init__('data_recorder')

        self.declare_parameters('', [
            ('output_dir', str(Path.home() / 'mobilebot_data')),
            ('session_id', ''),          # empty => derive from timestamp
            ('storage_id', 'mcap'),

            # Optional session tags. All are skippable; blank means "not given".
            ('location', ''),
            ('route_id', ''),
            ('surface', ''),
            ('weather', ''),
            ('sun_direction_deg', float('nan')),
            ('notes', ''),

            # Static vehicle geometry -- source of truth for session metadata,
            # since no single running node owns these.
            ('wheelbase_m', 0.140),
            ('track_m', 0.146),
            ('wheel_diameter_m', 0.064),
            ('servo_centre_deg', 90.0),

            ('preprocessing_version', 'unset'),
            ('crop_x', 0), ('crop_y', 0),
            ('crop_width', 640), ('crop_height', 480),
            ('resize_width', 160), ('resize_height', 120),
        ])

        p = self.get_parameter
        self._output_dir = Path(str(p('output_dir').value)).expanduser()
        self._storage_id = str(p('storage_id').value)

        self._session_id = str(p('session_id').value) or self._make_session_id()
        self._session_dir = self._output_dir / self._session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)

        self._episode: EpisodeWriter | None = None
        self._episode_index = 0
        self._recording_mode_active = False

        self._session_metadata = self._build_session_metadata()
        self._log_session_summary()

        reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST, depth=10)
        sensor = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST, depth=10)

        # Raw passthrough for everything we don't need to inspect.
        for spec in TOPICS:
            if spec.name == 'robot/state':
                continue  # handled specially below, see class docstring
            if spec.name == 'session/metadata':
                continue  # never subscribed; written directly by this node
            qos = sensor if spec.best_effort else reliable
            self.create_subscription(
                spec.msg_class, spec.name,
                self._make_raw_handler(spec), qos, raw=True)

        # Typed, so we can read .mode -- and a second raw subscription so the
        # message still lands in the bag unmodified.
        self.create_subscription(RobotState, 'robot/state', self._on_robot_state_typed, reliable)
        self.create_subscription(RobotState, 'robot/state', self._make_raw_handler(
            TopicSpec('robot/state', 'mobilebot_msgs/msg/RobotState', RobotState)), reliable, raw=True)

        self.get_logger().info(
            f'data_recorder up. session={self._session_id} dir={self._session_dir}')

    # -- session metadata ----------------------------------------------------

    def _make_session_id(self) -> str:
        return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')

    def _build_session_metadata(self) -> SessionMetadata:
        p = self.get_parameter
        fetcher = ParamFetcher(self)

        cam = fetcher.fetch('usb_cam', [
            'video_device', 'image_width', 'image_height', 'pixel_format',
            'framerate', 'exposure', 'auto_white_balance'])
        teleop = fetcher.fetch('teleop_translator', [
            'steer_rate_base', 'steer_rate_max', 'steer_accel_time',
            'steer_centre_rate', 'max_steer_rad', 'speed_step', 'max_speed'])
        bridge = fetcher.fetch('hardware_bridge', ['speed_to_throttle'])

        msg = SessionMetadata()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.session_id = self._session_id

        msg.camera_device = str(cam.get('video_device', ''))
        msg.image_width = int(cam.get('image_width', 0) or 0)
        msg.image_height = int(cam.get('image_height', 0) or 0)
        msg.pixel_format = str(cam.get('pixel_format', ''))
        msg.camera_framerate_hz = float(cam.get('framerate', 0.0) or 0.0)
        msg.exposure = int(cam.get('exposure', 0) or 0)
        msg.auto_white_balance = bool(cam.get('auto_white_balance', False))

        msg.preprocessing_version = str(p('preprocessing_version').value)
        msg.crop_x = int(p('crop_x').value)
        msg.crop_y = int(p('crop_y').value)
        msg.crop_width = int(p('crop_width').value)
        msg.crop_height = int(p('crop_height').value)
        msg.resize_width = int(p('resize_width').value)
        msg.resize_height = int(p('resize_height').value)

        msg.steer_rate_base = float(teleop.get('steer_rate_base', 0.0) or 0.0)
        msg.steer_rate_max = float(teleop.get('steer_rate_max', 0.0) or 0.0)
        msg.steer_accel_time = float(teleop.get('steer_accel_time', 0.0) or 0.0)
        msg.steer_centre_rate = float(teleop.get('steer_centre_rate', 0.0) or 0.0)
        msg.max_steer_rad = float(teleop.get('max_steer_rad', 0.0) or 0.0)
        msg.speed_step = float(teleop.get('speed_step', 0.0) or 0.0)
        msg.max_speed = float(teleop.get('max_speed', 0.0) or 0.0)

        msg.wheelbase_m = float(p('wheelbase_m').value)
        msg.track_m = float(p('track_m').value)
        msg.wheel_diameter_m = float(p('wheel_diameter_m').value)
        msg.speed_to_throttle = float(bridge.get('speed_to_throttle', 0.0) or 0.0)
        msg.servo_centre_deg = float(p('servo_centre_deg').value)

        msg.git_commit = _git_commit()
        msg.ros_distro = os.environ.get('ROS_DISTRO', '')

        msg.location = str(p('location').value)
        msg.route_id = str(p('route_id').value)
        msg.surface = str(p('surface').value)
        msg.weather = str(p('weather').value)
        msg.sun_direction_deg = float(p('sun_direction_deg').value)
        msg.notes = str(p('notes').value)

        return msg

    def _log_session_summary(self) -> None:
        m = self._session_metadata
        missing = []
        if not m.camera_device:
            missing.append('camera (usb_cam unreachable)')
        if m.max_steer_rad == 0.0:
            missing.append('teleop shaping (teleop_translator unreachable)')
        if m.speed_to_throttle == 0.0:
            missing.append('speed_to_throttle (hardware_bridge unreachable)')
        if missing:
            self.get_logger().warn(
                f'session metadata incomplete, missing: {", ".join(missing)}. '
                f'Start the missing node(s) and restart the recorder if this '
                f'session matters for training.')

    # -- episode lifecycle ----------------------------------------------------

    def _on_robot_state_typed(self, msg: RobotState) -> None:
        is_recording = (msg.mode == RobotState.MODE_ARMED_RECORD)
        if is_recording and not self._recording_mode_active:
            self._open_episode()
        elif not is_recording and self._recording_mode_active:
            self._close_episode()
        self._recording_mode_active = is_recording

    def _open_episode(self) -> None:
        self._episode_index += 1
        started = datetime.datetime.now(datetime.timezone.utc)
        # Index only, no timestamp: unique within the session already, and
        # short enough to type by hand into the host tools.
        path = self._session_dir / f'episode_{self._episode_index:04d}'

        writer = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py.StorageOptions(uri=str(path), storage_id=self._storage_id)
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr', output_serialization_format='cdr')
        writer.open(storage_options, converter_options)

        for topic_id, spec in enumerate(TOPICS):
            writer.create_topic(rosbag2_py.TopicMetadata(
                id=topic_id, name=spec.name, type=spec.type_string,
                serialization_format='cdr'))

        self._episode = EpisodeWriter(path=path, writer=writer, started_at=started)

        # First message in every episode bag: the session metadata this
        # episode was recorded under. This is what makes Layer C recoverable
        # from Layer A alone, per-episode, without a side channel.
        now_ns = self.get_clock().now().nanoseconds
        writer.write('session/metadata', serialize_message(self._session_metadata), now_ns)

        self.get_logger().info(f'episode {self._episode_index} opened: {path.name}')

    def _close_episode(self) -> None:
        if self._episode is None:
            return
        ep = self._episode
        self._episode = None
        del ep.writer  # closes the bag; rosbag2_py has no explicit close()
        self.get_logger().info(
            f'episode {self._episode_index} closed: {ep.frame_count} frames, '
            f'{(datetime.datetime.now(datetime.timezone.utc) - ep.started_at).total_seconds():.1f} s')

    # -- message capture ------------------------------------------------------

    def _make_raw_handler(self, spec: TopicSpec):
        def handler(raw_bytes: bytes) -> None:
            ep = self._episode
            if ep is None:
                return

            now_ns = self.get_clock().now().nanoseconds

            if spec.name == 'image_raw/compressed':
                ep.last_kept_image_ns = now_ns
                ep.frame_count += 1

            ep.writer.write(spec.name, raw_bytes, now_ns)

        return handler


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DataRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._close_episode()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
