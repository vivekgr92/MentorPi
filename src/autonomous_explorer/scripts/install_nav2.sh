#!/usr/bin/env bash
# ============================================================================
# install_nav2.sh — Install Nav2 + SLAM packages for hybrid explorer mode
#
# Run this on the Raspberry Pi (Ubuntu 24.04 / ROS2 Jazzy):
#   sudo bash install_nav2.sh
# ============================================================================
set -eo pipefail

echo "=============================================="
echo "  Installing Nav2 + SLAM for MentorPi"
echo "=============================================="

# Ensure ROS2 apt sources are set up
if ! apt-cache policy | grep -q "packages.ros.org"; then
    echo "ERROR: ROS2 apt repository not configured."
    echo "See: https://docs.ros.org/en/jazzy/Installation.html"
    exit 1
fi

echo ">>> Updating package index..."
apt update

echo ""
echo ">>> Installing Nav2 navigation stack..."
apt install -y \
    ros-jazzy-navigation2 \
    ros-jazzy-nav2-bringup \
    ros-jazzy-nav2-simple-commander \
    ros-jazzy-nav2-msgs

echo ""
echo ">>> Installing SLAM (RTAbMap)..."
apt install -y \
    ros-jazzy-rtabmap-ros \
    ros-jazzy-rtabmap-sync \
    ros-jazzy-rtabmap-slam

echo ""
echo ">>> Installing DWB local planner (lighter than TEB for Pi 5)..."
apt install -y \
    ros-jazzy-dwb-core \
    ros-jazzy-dwb-plugins \
    ros-jazzy-dwb-critics

echo ""
echo ">>> Installing TEB local planner (optional, heavier)..."
apt install -y ros-jazzy-teb-local-planner 2>/dev/null || {
    echo "WARNING: TEB planner not available — DWB will be used instead."
}

echo ""
echo ">>> Installing costmap converter (needed by TEB)..."
apt install -y ros-jazzy-costmap-converter 2>/dev/null || true

echo ""
echo ">>> Installing robot_localization (EKF for sensor fusion)..."
apt install -y ros-jazzy-robot-localization 2>/dev/null || true

echo ""
echo ">>> Installing tf2 tools..."
apt install -y \
    ros-jazzy-tf2-ros \
    ros-jazzy-tf2-tools

echo ""
echo "=============================================="
echo "  Installation complete!"
echo ""
echo "  Verify with:"
echo "    source /opt/ros/jazzy/setup.bash"
echo "    ros2 pkg list | grep -E 'nav2|rtabmap|dwb'"
echo "=============================================="
