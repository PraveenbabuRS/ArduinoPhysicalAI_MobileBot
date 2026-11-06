# Mirror of the MCU's internal state, republished by hardware_bridge from the
# mcu_status Bridge notification.

uint8 STATE_INIT=0
uint8 STATE_DISARMED=1
uint8 STATE_ARMED=2
uint8 STATE_FAILSAFE=3

std_msgs/Header header

uint8 state
uint32 last_seq
uint32 cmd_age_ms
float32 servo_deg
float32 throttle
uint32 dropped
