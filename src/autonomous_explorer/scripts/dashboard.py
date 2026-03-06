#!/usr/bin/env python3
# encoding: utf-8
"""
Real-time terminal dashboard for MentorPi Autonomous Explorer.

Usage:
    python3 scripts/dashboard.py
    # or after colcon build:
    ros2 run autonomous_explorer dashboard
"""
from autonomous_explorer.dashboard import main

if __name__ == '__main__':
    main()
