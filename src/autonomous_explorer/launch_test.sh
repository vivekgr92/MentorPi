#!/usr/bin/env bash
# Launch integration tests for autonomous_explorer with all hardware.
#
# Usage:
#   ./launch_test.sh              # Full suite
#   ./launch_test.sh -k "sensor"  # Only sensor tests
#   ./launch_test.sh -k "motor"   # Only motor tests
#
# Prerequisites: STM32 (/dev/ttyACM0), LiDAR (/dev/ttyUSB0), depth camera connected.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── ROS2 environment ─────────────────────────────────────────────────────────
source /opt/ros/jazzy/setup.bash
[[ -f "$PROJECT_ROOT/install/setup.bash" ]] && source "$PROJECT_ROOT/install/setup.bash"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export LD_LIBRARY_PATH="/opt/ros/jazzy/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
export MACHINE_TYPE=MentorPi_Tank
export LIDAR_TYPE=LD19
export DEPTH_CAMERA_TYPE=aurora
export need_compile=True

# ── Device symlinks ──────────────────────────────────────────────────────────
[[ -e /dev/ttyACM0 ]] && sudo ln -sf /dev/ttyACM0 /dev/rrc
[[ -e /dev/ttyUSB0 ]] && sudo ln -sf /dev/ttyUSB0 /dev/ldlidar

# ── Track PIDs for cleanup ───────────────────────────────────────────────────
PIDS=()

cleanup() {
    echo ""
    echo "==> Stopping driver nodes..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null && wait "$pid" 2>/dev/null || true
    done

    # Release PID hold via direct serial — prevents motor buzz
    echo "==> Releasing motor PID hold via serial..."
    python3 -c "
import sys, time
sys.path.insert(0, '$PROJECT_ROOT/install/ros_robot_controller/lib/python3.13/site-packages')
sys.path.insert(0, '$PROJECT_ROOT/install/sdk/lib/python3.13/site-packages')
from ros_robot_controller.ros_robot_controller_sdk import Board
board = Board()
board.set_motor_speed([[1, 0], [2, 0], [3, 0], [4, 0]])
time.sleep(0.3)
print('Motors released.')
" 2>/dev/null || echo "(serial release failed — motors may buzz)"

    echo "==> Done."
}

trap cleanup EXIT

# ── Start driver nodes ───────────────────────────────────────────────────────
echo "==> Starting STM32 controller..."
ros2 run ros_robot_controller ros_robot_controller &>/tmp/rrc.log &
PIDS+=($!)

echo "==> Starting odom publisher..."
ros2 run controller odom_publisher \
    --ros-args -p base_frame_id:=base_footprint -p odom_frame_id:=odom \
    -p pub_odom_topic:=true &>/tmp/odom.log &
PIDS+=($!)

echo "==> Starting LiDAR..."
ros2 launch peripherals lidar.launch.py &>/tmp/lidar.log &
PIDS+=($!)

echo "==> Starting depth camera..."
ros2 launch peripherals depth_camera.launch.py &>/tmp/camera.log &
PIDS+=($!)

# ── Wait for topics ──────────────────────────────────────────────────────────
echo "==> Waiting for sensors..."
REQUIRED_TOPICS=(
    "/ascamera/camera_publisher/rgb0/image"
    "/ros_robot_controller/imu_raw"
    "/scan_raw"
)
MAX_WAIT=30
ELAPSED=0

while (( ELAPSED < MAX_WAIT )); do
    TOPICS=$(ros2 topic list 2>/dev/null || true)
    ALL_FOUND=true
    for t in "${REQUIRED_TOPICS[@]}"; do
        if ! echo "$TOPICS" | grep -q "^${t}$"; then
            ALL_FOUND=false
            break
        fi
    done
    if $ALL_FOUND; then
        echo "==> All sensors online."
        break
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done

if ! $ALL_FOUND; then
    echo "WARNING: Not all topics appeared after ${MAX_WAIT}s. Running tests anyway."
    echo "Missing topics — check /tmp/rrc.log, /tmp/lidar.log, /tmp/camera.log"
fi

# ── Run tests ────────────────────────────────────────────────────────────────
echo "==> Running integration tests..."
echo ""
PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}" \
    python3 -m pytest "$SCRIPT_DIR/test/test_integration.py" \
    -v --tb=short "$@"
