cmake_minimum_required(VERSION 3.8)
project(mobilebot_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TeleopCommand.msg"
  "msg/RobotState.msg"
  "msg/McuStatus.msg"
  "msg/SessionMetadata.msg"
  DEPENDENCIES std_msgs ackermann_msgs
)

ament_package()
