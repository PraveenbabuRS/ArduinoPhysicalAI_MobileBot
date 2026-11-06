"""Bring-up for line-of-sight teleoperation and data collection.

The joystick half of the graph runs on the HOST PC, not here. Launch it there
separately -- and note that autorepeat_rate is not optional, because joy_node
defaults to publishing only on change, which would trip the staleness watchdog:

    ros2 run joy joy_node --ros-args -p autorepeat_rate:=50.0 -p deadzone:=0.05

hardware_bridge is a lifecycle node and is deliberately NOT auto-transitioned
here. Drive it explicitly so a failed MCU handshake is visible rather than
silently retried:

    ros2 lifecycle set /hardware_bridge configure
    ros2 lifecycle set /hardware_bridge activate

policy_node is off by default (use_policy:=false) -- it only needs to run when
you are actually testing the model, and there is no reason to pay its CPU cost
during ordinary teleop data collection.

estop_confirmed defaults false, matching safety_arbiter's checked-in default.
This robot has no physical e-stop (weight budget); ARMED_AUTO stays
unreachable unless you explicitly override this per session:

    ros2 launch mobilebot_bringup teleop.launch.py use_policy:=true estop_confirmed:=true

Passing it explicitly, every time, is deliberate -- it is a conscious decision
about a documented safety gate, not something that should ever become a quiet
permanent default in a config file.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, LifecycleNode


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory('mobilebot_bringup'))
    default_params = str(share / 'config' / 'mobilebot.yaml')

    params = LaunchConfiguration('params_file')
    use_camera = LaunchConfiguration('use_camera')
    use_recorder = LaunchConfiguration('use_recorder')
    use_policy = LaunchConfiguration('use_policy')

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('use_camera', default_value='true'),
        DeclareLaunchArgument('use_recorder', default_value='true'),
        DeclareLaunchArgument('use_policy', default_value='false'),
        DeclareLaunchArgument('estop_confirmed', default_value='false'),

        # Optional session tags for data_recorder's Layer C metadata. All
        # default to empty -- skip any or all of them without error.
        DeclareLaunchArgument('location', default_value=''),
        DeclareLaunchArgument('route_id', default_value=''),
        DeclareLaunchArgument('surface', default_value=''),
        DeclareLaunchArgument('weather', default_value=''),
        DeclareLaunchArgument('notes', default_value=''),

        Node(
            package='mobilebot_control',
            executable='teleop_translator',
            name='teleop_translator',
            parameters=[params],
            output='screen',
        ),
        Node(
            package='mobilebot_control',
            executable='safety_arbiter',
            name='safety_arbiter',
            parameters=[params, {
                'estop_confirmed': LaunchConfiguration('estop_confirmed'),
            }],
            output='screen',
        ),
        LifecycleNode(
            package='mobilebot_bridge',
            executable='hardware_bridge',
            name='hardware_bridge',
            namespace='',
            parameters=[params],
            output='screen',
        ),
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam',
            parameters=[params],
            remappings=[('image_raw', 'image_raw')],
            condition=IfCondition(use_camera),
            output='screen',
        ),
        Node(
            package='mobilebot_recorder',
            executable='data_recorder',
            name='data_recorder',
            parameters=[params, {
                'location': LaunchConfiguration('location'),
                'route_id': LaunchConfiguration('route_id'),
                'surface': LaunchConfiguration('surface'),
                'weather': LaunchConfiguration('weather'),
                'notes': LaunchConfiguration('notes'),
            }],
            condition=IfCondition(use_recorder),
            output='screen',
        ),
        Node(
            package='mobilebot_policy',
            executable='policy_node',
            name='policy_node',
            parameters=[params],
            condition=IfCondition(use_policy),
            output='screen',
        ),
    ])
