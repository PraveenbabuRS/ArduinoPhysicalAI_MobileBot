# Layer C. Written as the FIRST message in every episode bag by data_recorder,
# so each episode is self-describing on its own -- no side file, no separate
# bag to keep in sync. Every value is QUERIED LIVE from the running nodes'
# parameter servers at session start, never copied from a YAML file, so it is
# guaranteed to describe what actually ran. A node that could not be reached
# leaves its fields at zero/empty and data_recorder logs a warning naming it.

std_msgs/Header header
string session_id

# --- Camera (queried from usb_cam) ---
string camera_device
int32 image_width
int32 image_height
string pixel_format
float32 camera_framerate_hz
int32 exposure
bool auto_white_balance

# --- Preprocessing contract ---
# "unset" is a valid value for preprocessing_version and MUST be checked by
# the extractor before a session is used to build a training set -- it means
# policy_node's crop/resize contract did not exist yet when this was recorded.
string preprocessing_version
int32 crop_x
int32 crop_y
int32 crop_width
int32 crop_height
int32 resize_width
int32 resize_height

# --- Teleop shaping (queried from teleop_translator) ---
float32 steer_rate_base
float32 steer_rate_max
float32 steer_accel_time
float32 steer_centre_rate
float32 max_steer_rad
float32 speed_step
float32 max_speed

# --- Vehicle geometry and calibration ---
float32 wheelbase_m
float32 track_m
float32 wheel_diameter_m
float32 speed_to_throttle
float32 servo_centre_deg

# --- Provenance ---
string git_commit
string ros_distro

# --- Operator-supplied, optional (launch args, default empty/NaN) ---
string location
string route_id
string surface
string weather
float32 sun_direction_deg
string notes
