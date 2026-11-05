from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("realtime", default_value="true"),
        Node(package="g1_arm_rtc_ros", executable="g1_arm_rtc_node", name="g1_arm_rtc", output="screen",
             parameters=[{"realtime": LaunchConfiguration("realtime"), "publish_rate_hz": 100.0}]),
    ])
