"""Sole authority over robot motion.

Everything else in the graph is a *proposer*. This node is the *decider*: it is
the only publisher of /cmd_drive, and it holds the mode state machine that
determines which proposer, if any, is currently selected.

Modes
    SAFE          neutral output, MCU disarmed, resting state
    ARMED_TELEOP  teleop proposer selected
    ARMED_RECORD  teleop proposer selected, recorder gated on
    ARMED_AUTO    policy proposer selected, deadman still required
    FAILSAFE      latched fault; requires a deliberate re-arm to clear

/cmd_drive is published continuously at a fixed rate in every mode, including
SAFE. The MCU watchdog must be fed even when disarmed, and the MCU refuses to
arm into a stale command channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import SetBool

from mobilebot_msgs.msg import TeleopCommand, RobotState, McuStatus


def clamp(value: float, limit: float) -> float:
    return -limit if value < -limit else (limit if value > limit else value)


def move_toward(current: float, target: float, max_delta: float) -> float:
    diff = target - current
    if diff > max_delta:
        return current + max_delta
    if diff < -max_delta:
        return current - max_delta
    return target


@dataclass
class Freshness:
    """Age tracker for one input stream."""

    timeout_s: float
    stamp_ns: int = 0
    seen: bool = False

    def touch(self, now_ns: int) -> None:
        self.stamp_ns = now_ns
        self.seen = True

    def ok(self, now_ns: int) -> bool:
        if not self.seen:
            return False
        return (now_ns - self.stamp_ns) * 1e-9 <= self.timeout_s


class SafetyArbiter(Node):

    def __init__(self) -> None:
        super().__init__('safety_arbiter')

        self.declare_parameters('', [
            ('publish_rate_hz', 50.0),
            ('teleop_timeout_s', 0.15),
            # Policy runs an order of magnitude slower than teleop (~17 Hz,
            # 59 ms period, 119 ms max observed on the QRB2210) vs teleop's
            # 50 Hz / 20 ms. Sharing teleop's 150 ms timeout left only ~30 ms
            # of margin -- a single slow inference frame would trip FAILSAFE
            # mid-drive. 0.30 s is ~5 policy periods and ~2.5x the observed
            # worst-case pipeline latency. Raise this if you raise the model
            # size; lower it if inference gets faster.
            ('policy_timeout_s', 0.30),
            ('mcu_timeout_s', 0.30),
            ('image_timeout_s', 0.30),

            ('max_steer_rad', 0.436),
            ('max_speed', 0.5),
            ('max_reverse_speed', 0.2),
            ('steer_slew_rad_s', 3.0),
            ('speed_slew_m_s2', 1.0),

            ('require_estop_for_auto', True),
            ('estop_confirmed', False),
        ])

        p = self.get_parameter
        self._period = 1.0 / float(p('publish_rate_hz').value)
        self._max_steer = float(p('max_steer_rad').value)
        self._max_speed = float(p('max_speed').value)
        self._max_reverse = float(p('max_reverse_speed').value)
        self._steer_slew = float(p('steer_slew_rad_s').value)
        self._speed_slew = float(p('speed_slew_m_s2').value)
        self._require_estop = bool(p('require_estop_for_auto').value)

        self._teleop_fresh = Freshness(float(p('teleop_timeout_s').value))
        self._mcu_fresh = Freshness(float(p('mcu_timeout_s').value))
        self._image_fresh = Freshness(float(p('image_timeout_s').value))
        self._policy_fresh = Freshness(float(p('policy_timeout_s').value))

        self._teleop: TeleopCommand | None = None
        self._policy: AckermannDriveStamped | None = None
        self._mcu: McuStatus | None = None

        self._mode = RobotState.MODE_SAFE
        self._reason = 'startup'
        self._recording = False        # live: are we in ARMED_RECORD right now
        self._recording_armed = False  # sticky: should we resume it after SAFE
        self._auto_requested = False

        self._out_steer = 0.0
        self._out_speed = 0.0

        # State pattern: one handler per mode, dispatched by table. Adding a
        # mode is a new handler and one table entry, not an edit to a
        # conditional chain that already carries safety logic.
        self._handlers: dict[int, Callable[[], None]] = {
            RobotState.MODE_SAFE: self._handle_safe,
            RobotState.MODE_ARMED_TELEOP: self._handle_armed_teleop,
            RobotState.MODE_ARMED_RECORD: self._handle_armed_record,
            RobotState.MODE_ARMED_AUTO: self._handle_armed_auto,
            RobotState.MODE_FAILSAFE: self._handle_failsafe,
        }

        reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST, depth=1)
        sensor = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(TeleopCommand, 'teleop/cmd', self._on_teleop, reliable)
        self.create_subscription(AckermannDriveStamped, 'policy/cmd', self._on_policy, reliable)
        self.create_subscription(McuStatus, 'mcu/status', self._on_mcu, reliable)
        # /image_raw/compressed is the genuine JPEG stream (usb_cam publishes
        # this directly from the MJPEG capture with no decode). /image_raw
        # itself is a decoded preview usb_cam additionally produces and is not
        # subscribed anywhere in this graph -- disable it in the camera config
        # if a parameter exists to skip that decode, since it costs CPU for a
        # topic nothing consumes.
        self.create_subscription(CompressedImage, 'image_raw/compressed',
                                 self._on_image, sensor)

        self._cmd_pub = self.create_publisher(AckermannDriveStamped, 'cmd_drive', reliable)
        self._state_pub = self.create_publisher(RobotState, 'robot/state', reliable)

        # Autonomous entry is deliberately NOT a gamepad button. It requires a
        # service call from a terminal, so it cannot be triggered by a fumble.
        self.create_service(SetBool, 'arbiter/request_auto', self._on_request_auto)

        self.create_timer(self._period, self._on_timer)
        self.get_logger().info('safety_arbiter up in SAFE')

    # -- subscriptions -----------------------------------------------------

    def _now(self) -> int:
        return self.get_clock().now().nanoseconds

    def _on_teleop(self, msg: TeleopCommand) -> None:
        self._teleop = msg
        self._teleop_fresh.touch(self._now())

    def _on_policy(self, msg: AckermannDriveStamped) -> None:
        self._policy = msg
        self._policy_fresh.touch(self._now())

    def _on_mcu(self, msg: McuStatus) -> None:
        # mcu_fresh only tracks whether messages are ARRIVING, not their
        # content -- the MCU keeps publishing status at 20 Hz even while
        # internally latched in FAILSAFE, so a plain staleness check sees a
        # perfectly healthy stream and gives no indication that every arm
        # request is being silently rejected. Surface it here so it's visible
        # in the log rather than only inferable from "the robot isn't moving".
        if self._mcu is None or self._mcu.state != McuStatus.STATE_FAILSAFE:
            if msg.state == McuStatus.STATE_FAILSAFE:
                self.get_logger().warn(
                    'MCU reports FAILSAFE -- actuation is being rejected at the '
                    'firmware level regardless of what /cmd_drive says. '
                    'hardware_bridge should clear this automatically; if the '
                    'robot stays motionless after re-engaging, check its log.')
        self._mcu = msg
        self._mcu_fresh.touch(self._now())

    def _on_image(self, _msg: CompressedImage) -> None:
        self._image_fresh.touch(self._now())

    def _on_request_auto(self, request, response):
        if request.data:
            if self._require_estop and not self.get_parameter('estop_confirmed').value:
                response.success = False
                response.message = 'refused: estop_confirmed parameter is false'
                return response
            self._auto_requested = True
            response.message = 'autonomous requested; hold deadman to engage'
        else:
            self._auto_requested = False
            response.message = 'autonomous request cleared'
        response.success = True
        return response

    # -- state machine -----------------------------------------------------

    def _transition(self, mode: int, reason: str) -> None:
        if mode == self._mode:
            return
        self.get_logger().warn(f'mode {self._mode} -> {mode}: {reason}')
        self._mode = mode
        self._reason = reason
        # FAILSAFE always clears recording intent -- a fault is not a pause.
        # SAFE does NOT clear it: releasing the deadman mid-session should
        # pause recording, not cancel it, so the operator can reposition and
        # continue without re-pressing record every time. _recording_armed is
        # the sticky intent; _recording is the live "currently writing" flag
        # the recorder gates on and is derived from both mode and the latch.
        if mode == RobotState.MODE_FAILSAFE:
            self._recording_armed = False
            self._auto_requested = False
        self._recording = (mode == RobotState.MODE_ARMED_RECORD)

    def _deadman(self) -> bool:
        return self._teleop is not None and self._teleop.deadman

    def _estop_pressed(self) -> bool:
        return (self._teleop is not None
                and self._teleop.request == TeleopCommand.REQUEST_ESTOP)

    def _handle_safe(self) -> None:
        self._target(0.0, 0.0)
        if not self._deadman():
            return
        if not self._mcu_fresh.ok(self._now()):
            return
        # Resume directly into ARMED_RECORD if recording was armed before the
        # deadman was released -- each engagement is a new episode within the
        # same recording session, with no extra button press required.
        if self._recording_armed:
            self._transition(RobotState.MODE_ARMED_RECORD, 'deadman engaged, recording resumed')
        else:
            self._transition(RobotState.MODE_ARMED_TELEOP, 'deadman engaged')

    def _handle_armed_teleop(self) -> None:
        if not self._deadman():
            self._transition(RobotState.MODE_SAFE, 'deadman released')
            self._target(0.0, 0.0)
            return
        if self._teleop.request == TeleopCommand.REQUEST_TOGGLE_RECORD:
            self._recording_armed = True
            self._transition(RobotState.MODE_ARMED_RECORD, 'record toggled on')
        if self._auto_requested:
            self._transition(RobotState.MODE_ARMED_AUTO, 'autonomous engaged')
            return
        self._follow_teleop()

    def _handle_armed_record(self) -> None:
        if not self._deadman():
            # Recording intent is preserved across this transition; only the
            # live _recording flag drops, closing the current episode.
            self._transition(RobotState.MODE_SAFE, 'deadman released, recording paused')
            self._target(0.0, 0.0)
            return
        if self._teleop.request == TeleopCommand.REQUEST_TOGGLE_RECORD:
            self._recording_armed = False
            self._transition(RobotState.MODE_ARMED_TELEOP, 'record toggled off')
        self._follow_teleop()

    def _handle_armed_auto(self) -> None:
        # The deadman still governs autonomous driving. The policy proposes;
        # the human retains the authority to stop.
        if not self._deadman():
            self._transition(RobotState.MODE_SAFE, 'deadman released')
            self._target(0.0, 0.0)
            return
        if not self._auto_requested:
            self._transition(RobotState.MODE_ARMED_TELEOP, 'autonomous cancelled')
            return
        now = self._now()
        if not self._image_fresh.ok(now):
            self._transition(RobotState.MODE_FAILSAFE, 'image stream stale')
            return
        if not self._policy_fresh.ok(now) or self._policy is None:
            self._transition(RobotState.MODE_FAILSAFE, 'policy command stale')
            return
        self._target(self._policy.drive.steering_angle, self._policy.drive.speed)

    def _handle_failsafe(self) -> None:
        self._target(0.0, 0.0)
        # Latched. Clearing requires the deadman released AND an explicit
        # re-arm press, so a fault cannot be cleared by simply holding on.
        if self._deadman():
            return
        if (self._teleop is not None
                and self._teleop.request == TeleopCommand.REQUEST_REARM):
            self._transition(RobotState.MODE_SAFE, 'operator re-arm')

    def _follow_teleop(self) -> None:
        self._target(self._teleop.drive.steering_angle, self._teleop.drive.speed)

    # -- output ------------------------------------------------------------

    def _target(self, steer: float, speed: float) -> None:
        steer = clamp(steer, self._max_steer)
        speed = min(speed, self._max_speed)
        speed = max(speed, -self._max_reverse)
        self._pending = (steer, speed)

    def _on_timer(self) -> None:
        now = self._now()
        self._pending = (0.0, 0.0)

        # Global pre-emptions, evaluated before any mode logic.
        if self._estop_pressed() and self._mode != RobotState.MODE_FAILSAFE:
            self._transition(RobotState.MODE_FAILSAFE, 'operator e-stop')
        elif self._mode not in (RobotState.MODE_SAFE, RobotState.MODE_FAILSAFE):
            if not self._teleop_fresh.ok(now):
                self._transition(RobotState.MODE_FAILSAFE, 'teleop command stale')
            elif not self._mcu_fresh.ok(now):
                self._transition(RobotState.MODE_FAILSAFE, 'MCU telemetry stale')
            elif self._teleop is not None and not self._teleop.link_ok:
                self._transition(RobotState.MODE_FAILSAFE, 'joy link reported down')

        self._handlers[self._mode]()

        target_steer, target_speed = self._pending
        self._out_steer = move_toward(self._out_steer, target_steer,
                                      self._steer_slew * self._period)
        self._out_speed = move_toward(self._out_speed, target_speed,
                                      self._speed_slew * self._period)

        self._publish_cmd()
        self._publish_state(now)

    def _publish_cmd(self) -> None:
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = self._out_steer
        msg.drive.speed = self._out_speed
        self._cmd_pub.publish(msg)

    def _publish_state(self, now: int) -> None:
        msg = RobotState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.mode = self._mode
        msg.want_armed = self._mode in (RobotState.MODE_ARMED_TELEOP,
                                        RobotState.MODE_ARMED_RECORD,
                                        RobotState.MODE_ARMED_AUTO)
        msg.recording = self._recording
        msg.joy_ok = self._teleop_fresh.ok(now)
        msg.mcu_ok = self._mcu_fresh.ok(now)
        msg.image_ok = self._image_fresh.ok(now)
        msg.reason = self._reason
        self._state_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SafetyArbiter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
