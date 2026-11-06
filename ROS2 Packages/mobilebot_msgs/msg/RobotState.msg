# Published by safety_arbiter. Declarative: describes what the system SHOULD
# be doing. hardware_bridge reconciles the MCU toward want_armed, which makes
# arming idempotent and tolerant of a dropped message.

uint8 MODE_SAFE=0
uint8 MODE_ARMED_TELEOP=1
uint8 MODE_ARMED_RECORD=2
uint8 MODE_ARMED_AUTO=3
uint8 MODE_FAILSAFE=4

std_msgs/Header header

uint8 mode
bool want_armed
bool recording

bool joy_ok
bool mcu_ok
bool image_ok

string reason       # human-readable cause of the most recent transition
