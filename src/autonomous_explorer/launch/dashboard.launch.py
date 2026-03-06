#!/usr/bin/env python3
# encoding: utf-8
"""
Launch file for the Explorer Dashboard (curses terminal UI).

Run this in a SEPARATE terminal from the explorer node:

    # Terminal 1: explorer
    ros2 launch autonomous_explorer explorer.launch.py

    # Terminal 2: dashboard
    ros2 launch autonomous_explorer dashboard.launch.py

The dashboard subscribes to /explorer/status and displays real-time
sensor data, LLM reasoning, battery, safety status, etc.

Alternative (no launch file needed):
    ros2 run autonomous_explorer dashboard
"""
from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    return LaunchDescription([
        ExecuteProcess(
            cmd=['ros2', 'run', 'autonomous_explorer', 'dashboard'],
            output='screen',
        ),
    ])
