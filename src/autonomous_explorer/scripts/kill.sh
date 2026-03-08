#!/usr/bin/env bash
# ============================================================================
# kill.sh — Gracefully shut down the entire Jeeves stack
#
# Usage:
#   ./kill.sh           # graceful shutdown (SIGINT first, then SIGKILL)
#   ./kill.sh --force   # immediate SIGKILL
# ============================================================================
set -eo pipefail

FORCE=false
[[ "${1:-}" == "--force" || "${1:-}" == "-f" ]] && FORCE=true

# Node names launched by jeeves_agent.launch.py, in reverse startup order
NODES=(
    # Explorer agent
    autonomous_explorer
    semantic_map_publisher
    # Nav2 stack
    lifecycle_manager_navigation
    velocity_smoother
    smoother_server
    bt_navigator
    behavior_server
    planner_server
    controller_server
    # SLAM + EKF
    slam_toolbox
    ekf_filter_node
    # Support
    rosbridge_websocket
    foxglove_bridge
    twist_mux
    # IMU filter
    imu_filter
    imu_calib
    # Static TFs
    camera_tf
    lidar_tf
    # Hardware drivers
    LD19
    odom_publisher
    ros_robot_controller
)

# Process names that match ros2 executables (for pkill fallback)
# Be specific to avoid killing unrelated processes
PROCESS_PATTERNS=(
    "explorer_node"
    "semantic_map_publisher"
    "lifecycle_manager"
    "velocity_smoother"
    "smoother_server"
    "bt_navigator"
    "behavior_server"
    "planner_server"
    "controller_server"
    "async_slam_toolbox_node"
    "ekf_node"
    "rosbridge_websocket"
    "foxglove_bridge"
    "twist_mux"
    "complementary_filter_node"
    "apply_calib"
    "static_transform_publisher"
    "ldlidar_stl_ros2_node"
    "odom_publisher"
    "ros_robot_controller"
    "aurora930_node"
    "ascamera_node"
)

echo "=== Jeeves Stack Shutdown ==="

# ── Step 1: Stop motors immediately ──
# Send zero velocity to prevent runaway if controller is still alive
if command -v ros2 &>/dev/null; then
    echo "[1/4] Zeroing motors..."
    timeout 3 ros2 topic pub --once /controller/cmd_vel geometry_msgs/msg/Twist \
        '{linear: {x: 0.0}, angular: {z: 0.0}}' 2>/dev/null || true
fi

# ── Step 2: Graceful SIGINT to ros2 launch (kills the whole tree) ──
echo "[2/4] Sending SIGINT to ros2 launch processes..."
pkill -INT -f "ros2.launch.*jeeves_agent" 2>/dev/null || true
pkill -INT -f "ros2.launch.*explorer" 2>/dev/null || true

if [[ "$FORCE" == "false" ]]; then
    # Wait up to 5 seconds for graceful shutdown
    echo "      Waiting up to 5s for graceful shutdown..."
    for i in {1..10}; do
        if ! pgrep -f "ros2.launch.*jeeves_agent" &>/dev/null && \
           ! pgrep -f "ros2.launch.*explorer" &>/dev/null; then
            echo "      Launch processes exited."
            break
        fi
        sleep 0.5
    done
fi

# ── Step 3: Kill any remaining individual processes ──
echo "[3/4] Cleaning up remaining processes..."
KILLED=0
for pattern in "${PROCESS_PATTERNS[@]}"; do
    if pgrep -f "$pattern" &>/dev/null; then
        if [[ "$FORCE" == "true" ]]; then
            pkill -9 -f "$pattern" 2>/dev/null || true
        else
            pkill -INT -f "$pattern" 2>/dev/null || true
        fi
        KILLED=$((KILLED + 1))
    fi
done

if [[ $KILLED -gt 0 ]]; then
    echo "      Sent signal to $KILLED remaining process group(s)."
    if [[ "$FORCE" == "false" ]]; then
        sleep 2
        # SIGKILL any stragglers
        for pattern in "${PROCESS_PATTERNS[@]}"; do
            if pgrep -f "$pattern" &>/dev/null; then
                pkill -9 -f "$pattern" 2>/dev/null || true
            fi
        done
    fi
else
    echo "      No remaining processes found."
fi

# ── Step 4: Final zero velocity (safety) ──
if command -v ros2 &>/dev/null; then
    echo "[4/4] Final motor zero..."
    timeout 3 ros2 topic pub --once /controller/cmd_vel geometry_msgs/msg/Twist \
        '{linear: {x: 0.0}, angular: {z: 0.0}}' 2>/dev/null || true
fi

# ── Step 5: Kill dashboard screen session ──
screen -S jeeves_dash -X quit 2>/dev/null || true

echo "=== Shutdown complete ==="
