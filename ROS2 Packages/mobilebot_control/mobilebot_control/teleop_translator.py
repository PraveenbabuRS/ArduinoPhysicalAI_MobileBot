"""Translate a discrete-axis gamepad into continuous Ackermann intent.

The Manta pad reports ternary axes (-1, 0, +1), which cannot directly label a
steering-regression dataset. Two shaping strategies convert it into continuous,
learnable commands:

  Steering -- rate integration with self-centring.
      D-pad held        : integrate toward lock at an accelerating rate
      D-pad neutral, R1 : hold the current angle (clutch)
      D-pad neutral     : return to centre at a constant rate

  Throttle -- stepped setpoint, cruise-control style.
      Edge-triggered increments. Holding a constant speed across a run is what
      behaviour cloning wants; a constant setpoint delivers it exactly.

This node owns the button map and nothing else. It publishes intent; it never
decides whether the robot may move.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Joy

from mobilebot_msgs.msg import TeleopCommand


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else (high if value > high else value)


def move_toward(current: float, target: float, max_delta: float) -> float:
    diff = target - current
    if diff > max_delta:
        return current + max_delta
    if diff < -max_delta:
        return current - max_delta
    return target


@dataclass
class ButtonMap:
    """Array indices, not the 1-based numbers printed on the pad."""

    axis_steer: int
    axis_drive: int
    deadman: int
    steer_hold: int
    estop: int
    rearm: int
    record: int


class SteeringIntegrator:
    """Ternary D-pad input -> continuous normalised steering in [-1, 1].

    Rate acceleration is the reason this feels drivable: a single fixed rate is
    either too coarse for straight-line trim or too slow for a hairpin. Rate
    ramps from `rate_base` to `rate_max` over `accel_time` of continuous hold,
    so a tap trims and a hold sweeps.
    """

    def __init__(self, rate_base: float, rate_max: float, accel_time: float,
                 centre_rate: float) -> None:
        self._rate_base = rate_base
        self._rate_max = rate_max
        self._accel_time = max(accel_time, 1e-3)
        self._centre_rate = centre_rate

        self._value = 0.0
        self._hold_time = 0.0

    @property
    def value(self) -> float:
        return self._value

    def reset(self) -> None:
        self._value = 0.0
        self._hold_time = 0.0

    def update(self, direction: float, clutch: bool, dt: float) -> float:
        if direction != 0.0:
            self._hold_time += dt
            ramp = min(self._hold_time / self._accel_time, 1.0)
            rate = self._rate_base + ramp * (self._rate_max - self._rate_base)
            self._value = clamp(self._value + math.copysign(rate * dt, direction),
                                -1.0, 1.0)
        else:
            self._hold_time = 0.0
            if not clutch:
                self._value = move_toward(self._value, 0.0, self._centre_rate * dt)
        return self._value


class ThrottleSetpoint:
    """Edge-triggered stepped speed setpoint.

    Continuous integration would make every run a different speed profile. A
    stepped setpoint gives repeatable constant-speed demonstrations, which is
    what the throttle-decoupling recommendation actually asks for.
    """

    def __init__(self, step: float, max_forward: float, max_reverse: float) -> None:
        self._step = step
        self._max_forward = max_forward
        self._max_reverse = max_reverse
        self._value = 0.0
        self._previous = 0.0

    @property
    def value(self) -> float:
        return self._value

    def reset(self) -> None:
        self._value = 0.0
        self._previous = 0.0

    def update(self, direction: float) -> float:
        if direction != 0.0 and self._previous == 0.0:
            self._value = clamp(self._value + math.copysign(self._step, direction),
                                -self._max_reverse, self._max_forward)
        self._previous = direction
        return self._value


class TeleopTranslator(Node):

    def __init__(self) -> None:
        super().__init__('teleop_translator')

        self.declare_parameters('', [
            ('publish_rate_hz', 50.0),
            ('joy_timeout_s', 0.15),

            ('axis_steer', 0),
            ('axis_drive', 1),
            ('btn_deadman', 4),
            ('btn_steer_hold', 5),
            ('btn_estop', 2),
            ('btn_rearm', 0),
            ('btn_record', 8),

            ('invert_steer_axis', False),
            ('invert_drive_axis', False),

            ('steer_rate_base', 0.6),
            ('steer_rate_max', 2.5),
            ('steer_accel_time', 0.5),
            ('steer_centre_rate', 1.8),
            ('max_steer_rad', 0.436),

            ('speed_step', 0.1),
            ('max_speed', 0.5),
            ('max_reverse_speed', 0.2),
        ])

        p = self.get_parameter
        self._period = 1.0 / float(p('publish_rate_hz').value)
        self._joy_timeout = float(p('joy_timeout_s').value)

        self._map = ButtonMap(
            axis_steer=int(p('axis_steer').value),
            axis_drive=int(p('axis_drive').value),
            deadman=int(p('btn_deadman').value),
            steer_hold=int(p('btn_steer_hold').value),
            estop=int(p('btn_estop').value),
            rearm=int(p('btn_rearm').value),
            record=int(p('btn_record').value),
        )
        self._invert_steer = bool(p('invert_steer_axis').value)
        self._invert_drive = bool(p('invert_drive_axis').value)
        self._max_steer_rad = float(p('max_steer_rad').value)

        self._steering = SteeringIntegrator(
            rate_base=float(p('steer_rate_base').value),
            rate_max=float(p('steer_rate_max').value),
            accel_time=float(p('steer_accel_time').value),
            centre_rate=float(p('steer_centre_rate').value),
        )
        self._throttle = ThrottleSetpoint(
            step=float(p('speed_step').value),
            max_forward=float(p('max_speed').value),
            max_reverse=float(p('max_reverse_speed').value),
        )

        self._joy: Joy | None = None
        self._joy_stamp_ns = 0
        self._prev_buttons: list[int] = []
        self._was_linked = False

        # Freshness beats delivery for a continuously re-sent state stream: a
        # retransmitted stale stick position is worse than none at all.
        joy_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(Joy, 'joy', self._on_joy, joy_qos)
        self._pub = self.create_publisher(TeleopCommand, 'teleop/cmd', cmd_qos)
        self.create_timer(self._period, self._on_timer)

        self.get_logger().info(
            f'teleop_translator up: {p("publish_rate_hz").value} Hz, '
            f'max steer {math.degrees(self._max_steer_rad):.1f} deg, '
            f'max speed {p("max_speed").value} m/s'
        )

    # -- callbacks ---------------------------------------------------------

    def _on_joy(self, msg: Joy) -> None:
        # Stamp on arrival using the local clock. The host PC's header stamp is
        # kept only for offline latency auditing; it is never used for control.
        self._joy = msg
        self._joy_stamp_ns = self.get_clock().now().nanoseconds

    def _on_timer(self) -> None:
        link_ok = self._link_is_fresh()

        if not link_ok:
            if self._was_linked:
                self.get_logger().warn('/joy stale -- publishing neutral intent')
            self._was_linked = False
            self._steering.reset()
            self._throttle.reset()
            self._publish(deadman=False, steer_hold=False, link_ok=False,
                          request=TeleopCommand.REQUEST_NONE)
            return

        self._was_linked = True
        joy = self._joy
        assert joy is not None

        deadman = self._button(joy, self._map.deadman)
        steer_hold = self._button(joy, self._map.steer_hold)
        request = self._pending_request(joy)

        if not deadman:
            # Releasing the deadman must not leave a wound-up integrator that
            # would be re-applied the instant it is pressed again.
            self._steering.reset()
            self._throttle.reset()
            self._publish(False, steer_hold, True, request)
            return

        steer_dir = self._axis(joy, self._map.axis_steer, self._invert_steer)
        drive_dir = self._axis(joy, self._map.axis_drive, self._invert_drive)

        self._steering.update(steer_dir, steer_hold, self._period)
        self._throttle.update(drive_dir)

        self._publish(True, steer_hold, True, request)
        self._prev_buttons = list(joy.buttons)

    # -- helpers -----------------------------------------------------------

    def _link_is_fresh(self) -> bool:
        if self._joy is None:
            return False
        age_s = (self.get_clock().now().nanoseconds - self._joy_stamp_ns) * 1e-9
        return age_s <= self._joy_timeout

    @staticmethod
    def _button(joy: Joy, index: int) -> bool:
        return index < len(joy.buttons) and joy.buttons[index] == 1

    @staticmethod
    def _axis(joy: Joy, index: int, invert: bool) -> float:
        if index >= len(joy.axes):
            return 0.0
        value = float(joy.axes[index])
        # Tolerate an analog pad being plugged in later: anything past half
        # deflection counts as a full press, and true analog input still works.
        if abs(value) < 0.5:
            value = 0.0
        return -value if invert else value

    def _pressed_now(self, joy: Joy, index: int) -> bool:
        if index >= len(joy.buttons):
            return False
        was = self._prev_buttons[index] if index < len(self._prev_buttons) else 0
        return joy.buttons[index] == 1 and was == 0

    def _pending_request(self, joy: Joy) -> int:
        # Priority order matters: an e-stop press must win over anything else
        # that happened in the same frame.
        if self._pressed_now(joy, self._map.estop):
            return TeleopCommand.REQUEST_ESTOP
        if self._pressed_now(joy, self._map.rearm):
            return TeleopCommand.REQUEST_REARM
        if self._pressed_now(joy, self._map.record):
            return TeleopCommand.REQUEST_TOGGLE_RECORD
        return TeleopCommand.REQUEST_NONE

    def _publish(self, deadman: bool, steer_hold: bool, link_ok: bool,
                 request: int) -> None:
        msg = TeleopCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = self._steering.value * self._max_steer_rad
        msg.drive.speed = self._throttle.value
        msg.deadman = deadman
        msg.steer_hold = steer_hold
        msg.link_ok = link_ok
        msg.request = request
        self._pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TeleopTranslator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
