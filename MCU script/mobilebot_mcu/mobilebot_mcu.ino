/*
 * mobilebot_mcu.ino
 * ---------------------------------------------------------------------------
 * STM32U585 (Zephyr) firmware for the UNO Q behaviour-cloning robot.
 *
 * Responsibilities (and nothing else):
 *   - Serve set_drive / set_armed RPCs from the Linux side over the Bridge.
 *   - Enforce the L1 safety layer: command-staleness watchdog -> FAILSAFE.
 *   - Slew-limit steering and throttle so the actuators never see a step.
 *   - Stream IMU samples and MCU status upward as typed MessagePack.
 *
 * Explicit non-goals: no control policy, no odometry, no filtering. The MCU is
 * a dumb, fast, trustworthy actuator with a dead-man's switch.
 *
 * UNITS CONTRACT (must match hardware_bridge on the Linux side)
 *   steering : normalised, -1.0 = full left, 0.0 = centre, +1.0 = full right
 *   throttle : normalised, -1.0 = full reverse, 0.0 = coast, +1.0 = full fwd
 *   seq      : monotonically increasing uint32 from Linux; echoed back so the
 *              Linux side can measure RTT and detect dropped commands.
 *
 * The rad <-> normalised steering conversion deliberately lives on the Linux
 * side as a ROS parameter, so vehicle calibration never requires a reflash.
 * ---------------------------------------------------------------------------
 */

#include <Wire.h>
#include <Servo.h>
#include <Adafruit_MPU6050.h>
#include "Arduino_RouterBridge.h"

// ===========================================================================
// Configuration
// ===========================================================================

namespace cfg {

// --- Pin map (UNO Q headers) ---
constexpr uint8_t PIN_SERVO = 9;
constexpr uint8_t PIN_MOTOR_PWM = 10;  // ENA
constexpr uint8_t PIN_MOTOR_DIR = 11;  // DIR1

// --- Steering calibration. Adjust after bench measurement; do not guess. ---
constexpr float SERVO_CENTRE_DEG = 94.5f;   // mechanical centre
constexpr float SERVO_MAX_DEV_DEG = 40.0f;  // safe travel either side
constexpr float SERVO_SLEW_DEG_PER_S = 300.0f;
constexpr bool  STEERING_INVERT = false;    // flip if +steering turns left

// --- Drive calibration ---
// Below MOTOR_PWM_MIN the motor stalls and only heats up, so normalised
// throttle is mapped onto [MIN, MAX] rather than [0, MAX].
constexpr uint8_t MOTOR_PWM_MIN = 45;
constexpr uint8_t MOTOR_PWM_MAX = 200;  // conservative ceiling for v1
constexpr float   THROTTLE_SLEW_PER_S = 2.0f;
constexpr float   THROTTLE_REVERSE_LIMIT = 0.4f;
constexpr bool    MOTOR_DIR_INVERT = false;

// --- Timing (all periods in ms) ---
constexpr uint32_t CONTROL_PERIOD_MS = 20;   //  50 Hz actuation
constexpr uint32_t IMU_PERIOD_MS     = 10;   // 100 Hz telemetry
constexpr uint32_t STATUS_PERIOD_MS  = 50;   //  20 Hz status

// --- L1 safety layer ---
// If no valid set_drive arrives within this window, latch FAILSAFE.
constexpr uint32_t COMMAND_TIMEOUT_MS = 150;

// --- Debug. Serial printing in the control path costs time; keep it off. ---
constexpr bool DEBUG_SERIAL = true;

}  // namespace cfg

// ===========================================================================
// Types
// ===========================================================================

enum class DriveState : uint8_t {
  INIT = 0,      // hardware coming up, RPC not yet served
  DISARMED = 1,  // healthy, actuators held neutral, awaiting set_armed(true)
  ARMED = 2,     // executing commands
  FAILSAFE = 3   // latched fault; requires explicit disarm-then-arm to clear
};

struct DriveCommand {
  float steering;
  float throttle;
  uint32_t seq;
};

// ===========================================================================
// Shared state
// ===========================================================================
//
// The Bridge callback may execute on a different Zephyr thread from loop().
// The handler therefore does the absolute minimum: stage the values, publish a
// ready flag last. loop() consumes it. A torn read is bounded to one stale
// float for one 20 ms tick and cannot produce an out-of-range value, because
// every field is clamped again on the consumer side.
//
// If Arduino_RouterBridge's provide_safe() defers execution into loop context
// on your core version, prefer it -- set USE_PROVIDE_SAFE and the race is gone.

#define USE_PROVIDE_SAFE 0

static volatile DriveCommand g_staged_cmd = {0.0f, 0.0f, 0};
static volatile bool g_cmd_ready = false;
static volatile bool g_arm_request = false;
static volatile bool g_arm_request_pending = false;

static DriveState g_state = DriveState::INIT;
static uint32_t g_last_cmd_ms = 0;
static uint32_t g_last_seq = 0;
static uint32_t g_dropped_seq = 0;
static uint32_t g_imu_seq = 0;

// Actuator state, in physical units, after slew limiting.
static float g_servo_deg = cfg::SERVO_CENTRE_DEG;
static float g_throttle_applied = 0.0f;

// Latest accepted command target (pre-slew).
static float g_target_steering = 0.0f;
static float g_target_throttle = 0.0f;

static Adafruit_MPU6050 mpu;
static Servo steering_servo;

static float g_accel_offset[3] = {0.0f, 0.0f, 0.0f};

// Scheduler timestamps.
static uint32_t g_t_control = 0;
static uint32_t g_t_imu = 0;
static uint32_t g_t_status = 0;

// ===========================================================================
// Utilities
// ===========================================================================

static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Move `current` toward `target` by at most `max_delta`.
static inline float slew(float current, float target, float max_delta) {
  const float diff = target - current;
  if (diff > max_delta) return current + max_delta;
  if (diff < -max_delta) return current - max_delta;
  return target;
}

// Unsigned-safe elapsed-time test. Never compute (deadline - now) on unsigned
// types -- that is the underflow bug in the original IMU sketch.
static inline bool elapsed(uint32_t now, uint32_t last, uint32_t period) {
  return (uint32_t)(now - last) >= period;
}

// ===========================================================================
// Actuator abstraction
// ===========================================================================
//
// Isolated so that swapping Servo.h for the Zephyr PWM API is a two-function
// change if the Arduino core does not support Servo on this board.

static void steeringWriteNormalised(float norm) {
  norm = clampf(norm, -1.0f, 1.0f);
  if (cfg::STEERING_INVERT) norm = -norm;
  const float deg = cfg::SERVO_CENTRE_DEG + norm * cfg::SERVO_MAX_DEV_DEG;
  steering_servo.write((int)(deg + 0.5f));
}

static void steeringWriteDegrees(float deg) {
  const float lo = cfg::SERVO_CENTRE_DEG - cfg::SERVO_MAX_DEV_DEG;
  const float hi = cfg::SERVO_CENTRE_DEG + cfg::SERVO_MAX_DEV_DEG;
  steering_servo.write((int)(clampf(deg, lo, hi) + 0.5f));
}

// The SmartElex 15D is PWM+DIR: throttle 0 coasts, it does not brake.
static void motorWriteNormalised(float throttle) {
  throttle = clampf(throttle, -cfg::THROTTLE_REVERSE_LIMIT, 1.0f);

  const float magnitude = fabsf(throttle);
  if (magnitude < 0.01f) {
    analogWrite(cfg::PIN_MOTOR_PWM, 0);
    return;
  }

  bool forward = (throttle > 0.0f);
  if (cfg::MOTOR_DIR_INVERT) forward = !forward;
  digitalWrite(cfg::PIN_MOTOR_DIR, forward ? LOW : HIGH);

  const float span = (float)(cfg::MOTOR_PWM_MAX - cfg::MOTOR_PWM_MIN);
  const uint8_t duty = (uint8_t)(cfg::MOTOR_PWM_MIN + magnitude * span);
  analogWrite(cfg::PIN_MOTOR_PWM, duty);
}

// Immediate, unconditional neutral. Called on every fault path.
static void actuatorsNeutral() {
  analogWrite(cfg::PIN_MOTOR_PWM, 0);
  g_throttle_applied = 0.0f;
  g_target_throttle = 0.0f;
  g_target_steering = 0.0f;
  g_servo_deg = cfg::SERVO_CENTRE_DEG;
  steeringWriteDegrees(g_servo_deg);
}

// ===========================================================================
// State machine
// ===========================================================================

static void transitionTo(DriveState next) {
  if (next == g_state) return;
  g_state = next;

  if (next == DriveState::DISARMED || next == DriveState::FAILSAFE) {
    actuatorsNeutral();
  }

  if (cfg::DEBUG_SERIAL) {
    Serial.print("[state] -> ");
    Serial.println((int)next);
  }
}

// ===========================================================================
// RPC surface (Linux -> MCU)
// ===========================================================================

// Returns the sequence number it accepted, so the caller can confirm delivery
// and measure round-trip latency without a separate channel.
static uint32_t rpc_set_drive(float steering, float throttle, uint32_t seq) {
  g_staged_cmd.steering = clampf(steering, -1.0f, 1.0f);
  g_staged_cmd.throttle = clampf(throttle, -1.0f, 1.0f);
  g_staged_cmd.seq = seq;
  g_cmd_ready = true;  // publish last
  return seq;
}

static bool rpc_set_armed(bool armed) {
  g_arm_request = armed;
  g_arm_request_pending = true;
  return armed;
}

// Cheap liveness probe for the hardware_bridge lifecycle handshake.
static uint32_t rpc_ping(uint32_t token) {
  return token;
}

// ===========================================================================
// Periodic tasks
// ===========================================================================

static void taskConsumeCommands(uint32_t now) {
  if (g_arm_request_pending) {
    g_arm_request_pending = false;
    const bool want_armed = g_arm_request;

    if (!want_armed) {
      // Disarm always succeeds and clears a latched FAILSAFE.
      transitionTo(DriveState::DISARMED);
    } else if (g_state == DriveState::DISARMED) {
      // Only arm from a known-good disarmed state, and only if a fresh
      // command already exists -- arming into a stale channel is how robots
      // lurch on power-up.
      if (elapsed(now, g_last_cmd_ms, cfg::COMMAND_TIMEOUT_MS)) {
        if (cfg::DEBUG_SERIAL) Serial.println("[arm] refused: stale command channel");
      } else {
        transitionTo(DriveState::ARMED);
      }
    }
  }

  if (!g_cmd_ready) return;
  g_cmd_ready = false;

  const float steering = g_staged_cmd.steering;
  const float throttle = g_staged_cmd.throttle;
  const uint32_t seq = g_staged_cmd.seq;

  if (seq != 0 && seq != g_last_seq + 1) {
    g_dropped_seq++;  // diagnostic only; a gap is not itself a fault
  }
  g_last_seq = seq;
  g_last_cmd_ms = now;

  g_target_steering = steering;
  g_target_throttle = throttle;
}

static void taskWatchdog(uint32_t now) {
  if (g_state != DriveState::ARMED) return;
  if (elapsed(now, g_last_cmd_ms, cfg::COMMAND_TIMEOUT_MS)) {
    if (cfg::DEBUG_SERIAL) Serial.println("[watchdog] command timeout");
    transitionTo(DriveState::FAILSAFE);
  }
}

static void taskControl(uint32_t now) {
  const float dt = (float)cfg::CONTROL_PERIOD_MS / 1000.0f;

  float steering_cmd = 0.0f;
  float throttle_cmd = 0.0f;
  if (g_state == DriveState::ARMED) {
    steering_cmd = g_target_steering;
    throttle_cmd = g_target_throttle;
  }

  // Steering slews in degrees so the rate limit is a physical quantity.
  float target_deg = cfg::SERVO_CENTRE_DEG +
                     (cfg::STEERING_INVERT ? -steering_cmd : steering_cmd) *
                         cfg::SERVO_MAX_DEV_DEG;
  g_servo_deg = slew(g_servo_deg, target_deg, cfg::SERVO_SLEW_DEG_PER_S * dt);
  steeringWriteDegrees(g_servo_deg);

  g_throttle_applied =
      slew(g_throttle_applied, throttle_cmd, cfg::THROTTLE_SLEW_PER_S * dt);
  motorWriteNormalised(g_throttle_applied);
}

static void taskImu(uint32_t now) {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // Typed MessagePack floats -- no Arduino String, no heap churn.
  Bridge.notify("imu_sample",
                a.acceleration.x - g_accel_offset[0],
                a.acceleration.y - g_accel_offset[1],
                a.acceleration.z - g_accel_offset[2],
                g.gyro.x, g.gyro.y, g.gyro.z,
                g_imu_seq++);
}

static void taskStatus(uint32_t now) {
  const uint32_t cmd_age = (uint32_t)(now - g_last_cmd_ms);
  Bridge.notify("mcu_status",
                (uint8_t)g_state,
                g_last_seq,
                cmd_age,
                g_servo_deg,
                g_throttle_applied,
                g_dropped_seq);
}

// ===========================================================================
// Setup
// ===========================================================================

static void calibrateAccelerometer() {
  constexpr int kSamples = 150;
  float sum[3] = {0.0f, 0.0f, 0.0f};

  for (int i = 0; i < kSamples; i++) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    sum[0] += a.acceleration.x;
    sum[1] += a.acceleration.y;
    sum[2] += a.acceleration.z;
    delay(10);
  }

  g_accel_offset[0] = sum[0] / kSamples;
  g_accel_offset[1] = sum[1] / kSamples;
  g_accel_offset[2] = (sum[2] / kSamples) - 9.81f;
}

void setup() {
  if (cfg::DEBUG_SERIAL) {
    Serial.begin(115200);
    const uint32_t t0 = millis();
    while (!Serial && !elapsed(millis(), t0, 2000)) delay(10);
  }

  pinMode(cfg::PIN_MOTOR_PWM, OUTPUT);
  pinMode(cfg::PIN_MOTOR_DIR, OUTPUT);
  analogWrite(cfg::PIN_MOTOR_PWM, 0);

  steering_servo.attach(cfg::PIN_SERVO);
  steeringWriteDegrees(cfg::SERVO_CENTRE_DEG);

  Wire.begin();
  Wire.setClock(400000);

  if (!mpu.begin()) {
    // No IMU is not a reason to let the motors run.
    if (cfg::DEBUG_SERIAL) Serial.println("[fatal] MPU6050 not found");
    while (1) {
      actuatorsNeutral();
      delay(100);
    }
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_44_HZ);

  calibrateAccelerometer();

  Bridge.begin();
#if USE_PROVIDE_SAFE
  Bridge.provide_safe("set_drive", rpc_set_drive);
  Bridge.provide_safe("set_armed", rpc_set_armed);
  Bridge.provide_safe("ping", rpc_ping);
#else
  Bridge.provide("set_drive", rpc_set_drive);
  Bridge.provide("set_armed", rpc_set_armed);
  Bridge.provide("ping", rpc_ping);
#endif

  actuatorsNeutral();
  transitionTo(DriveState::DISARMED);

  const uint32_t now = millis();
  g_t_control = now;
  g_t_imu = now;
  g_t_status = now;
  g_last_cmd_ms = now - cfg::COMMAND_TIMEOUT_MS;  // start stale, by design
}

// ===========================================================================
// Main loop -- cooperative scheduler, never blocks
// ===========================================================================

void loop() {
  const uint32_t now = millis();

  taskConsumeCommands(now);
  taskWatchdog(now);

  if (elapsed(now, g_t_control, cfg::CONTROL_PERIOD_MS)) {
    g_t_control += cfg::CONTROL_PERIOD_MS;
    taskControl(now);
  }

  if (elapsed(now, g_t_imu, cfg::IMU_PERIOD_MS)) {
    g_t_imu += cfg::IMU_PERIOD_MS;
    taskImu(now);
  }

  if (elapsed(now, g_t_status, cfg::STATUS_PERIOD_MS)) {
    g_t_status += cfg::STATUS_PERIOD_MS;
    taskStatus(now);
  }
}
