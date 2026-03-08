#!/usr/bin/env python3
# encoding: utf-8
"""
Launch file for the Autonomous Explorer system.

This launches:
  1. The autonomous explorer node (LLM brain + motor control + voice)

Prerequisites (launch separately or via bringup):
  - controller node (motor control, odometry, EKF)
  - depth camera node (RGB + depth images)
  - LiDAR node (scan data)

Usage:
    # First, start the robot hardware:
    ros2 launch controller controller.launch.py
    ros2 launch peripherals depth_camera.launch.py
    ros2 launch peripherals lidar.launch.py

    # Then launch the explorer:
    ros2 launch autonomous_explorer explorer.launch.py

    # With custom LLM provider:
    ros2 launch autonomous_explorer explorer.launch.py llm_provider:=openai

    # With all parameters:
    ros2 launch autonomous_explorer explorer.launch.py \
        llm_provider:=claude \
        loop_interval:=2.0 \
        voice_enabled:=true
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('autonomous_explorer')
    params_file = os.path.join(pkg_dir, 'config', 'explorer_params.yaml')

    return LaunchDescription([
        # Launch arguments (override via CLI)
        DeclareLaunchArgument(
            'llm_provider',
            default_value=os.environ.get('LLM_PROVIDER', 'claude'),
            description='LLM provider: claude, openai, or dryrun',
        ),
        DeclareLaunchArgument(
            'loop_interval',
            default_value='3.0',
            description='Seconds between LLM analysis cycles',
        ),
        DeclareLaunchArgument(
            'voice_enabled',
            default_value='true',
            description='Enable voice I/O via WonderEcho Pro',
        ),
        DeclareLaunchArgument(
            'max_linear_speed',
            default_value='0.20',
            description='Maximum linear speed in m/s',
        ),
        DeclareLaunchArgument(
            'max_angular_speed',
            default_value='0.80',
            description='Maximum angular speed in rad/s',
        ),
        DeclareLaunchArgument(
            'camera_type',
            default_value=os.environ.get('DEPTH_CAMERA_TYPE', 'aurora'),
            description='Depth camera type (informational — matches DEPTH_CAMERA_TYPE env var)',
        ),
        DeclareLaunchArgument(
            'agent_mode',
            default_value=os.environ.get('AGENT_MODE', 'false'),
            description='Enable ROSA-style tool-calling agent mode',
        ),

        # Autonomous Explorer node
        Node(
            package='autonomous_explorer',
            executable='explorer_node',
            name='autonomous_explorer',
            output='screen',
            parameters=[
                params_file,
                {
                    'llm_provider': LaunchConfiguration('llm_provider'),
                    'loop_interval': LaunchConfiguration('loop_interval'),
                    'voice_enabled': LaunchConfiguration('voice_enabled'),
                    'max_linear_speed': LaunchConfiguration('max_linear_speed'),
                    'max_angular_speed': LaunchConfiguration('max_angular_speed'),
                    'agent_mode': LaunchConfiguration('agent_mode'),
                },
            ],
        ),
    ])
