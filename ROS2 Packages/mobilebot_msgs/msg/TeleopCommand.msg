# Atomic teleoperation intent. Drive command and safety flags travel together
# in one message so the arbiter can never pair a fresh command with a stale
# deadman state.

uint8 REQUEST_NONE=0
uint8 REQUEST_ESTOP=1
uint8 REQUEST_REARM=2
uint8 REQUEST_TOGGLE_RECORD=3

std_msgs/Header header

ackermann_msgs/AckermannDrive drive

bool deadman        # hold-to-run, true only while the button is held
bool steer_hold     # hold-clutch: suppress return-to-centre
bool link_ok        # false when /joy has gone stale
uint8 request       # edge-triggered operator request, see constants above
