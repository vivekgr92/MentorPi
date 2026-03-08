#!/usr/bin/env bash
# ============================================================================
# launch_explorer.sh — One-command launcher for the MentorPi Autonomous Explorer
#
# Usage:
#   ./launch_explorer.sh                          # defaults (Claude, full logging)
#   ./launch_explorer.sh --provider openai         # use OpenAI GPT-4o
#   ./launch_explorer.sh --no-voice                # disable voice I/O
#   ./launch_explorer.sh --no-hardware             # skip hardware node launch
#   ./launch_explorer.sh --dry-run                  # test without API keys
#   ./launch_explorer.sh --log-level minimal       # reduce disk usage
#   ./launch_explorer.sh --help                    # show all options
# ============================================================================
set -eo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
PROVIDER="${LLM_PROVIDER:-claude}"
LOG_LEVEL="${EXPLORER_LOG_LEVEL:-full}"
VOICE_ENABLED="true"
LOOP_INTERVAL="3.0"
MAX_LINEAR_SPEED="0.20"
MAX_ANGULAR_SPEED="0.80"
LAUNCH_HARDWARE="true"
BUILD_FIRST="false"
HYBRID_MODE="false"
FOXGLOVE="true"
DASHBOARD="false"

# Machine/sensor defaults (override via env or flags)
: "${MACHINE_TYPE:=MentorPi_Tank}"
: "${DEPTH_CAMERA_TYPE:=aurora}"
: "${LIDAR_TYPE:=LD19}"
: "${need_compile:=True}"

# ROS2 middleware (Debian Trixie needs explicit CycloneDDS setup)
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

# ── Parse arguments ─────────────────────────────────────────────────────────
usage() {
    cat <<'USAGE'
MentorPi Autonomous Explorer — Launch Script

Usage: launch_explorer.sh [OPTIONS]

Options:
  --provider <claude|openai|dryrun>
                               LLM provider (default: claude)
  --dry-run                    Shortcut for --provider dryrun --no-voice
  --log-level <full|compact|minimal>
                               Logging detail level (default: full)
  --loop-interval <secs>       Seconds between LLM cycles (default: 3.0)
  --max-speed <m/s>            Max linear speed (default: 0.20)
  --max-turn <rad/s>           Max angular speed (default: 0.80)
  --no-voice                   Disable voice I/O
  --no-hardware                Skip launching hardware nodes (camera, LiDAR,
                               controller). Use if they're already running.
  --no-foxglove                Disable Foxglove Bridge (remote visualization)
  --foxglove-port <port>       Foxglove Bridge port (default: 8765)
  --dashboard                  Launch curses dashboard in foreground after startup
  --no-dashboard               Skip dashboard (default: no dashboard)
  --hybrid                     Enable hybrid mode: SLAM + Nav2 + LLM
                               (requires install_nav2.sh first)
  --build                      Run colcon build before launching
  --help                       Show this help message

Environment variables:
  ANTHROPIC_API_KEY            Required for Claude provider
  OPENAI_API_KEY               Required for OpenAI provider (and voice TTS/STT)
  MACHINE_TYPE                 Robot chassis (default: MentorPi_Tank)
  DEPTH_CAMERA_TYPE            Camera model (default: aurora)
  LIDAR_TYPE                   LiDAR model (default: LD19)
  EXPLORER_LOG_DIR             Log directory (default: ~/mentorpi_explorer/logs)

Examples:
  # Basic launch with Claude
  ./launch_explorer.sh

  # OpenAI with compact logging, no voice
  ./launch_explorer.sh --provider openai --log-level compact --no-voice

  # Dry-run: test full pipeline without API keys
  ./launch_explorer.sh --dry-run

  # Build first, then launch with slower cycle
  ./launch_explorer.sh --build --loop-interval 5.0

  # Launch without remote visualization
  ./launch_explorer.sh --no-foxglove
USAGE
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --provider)
            PROVIDER="$2"; shift 2 ;;
        --log-level)
            LOG_LEVEL="$2"; shift 2 ;;
        --loop-interval)
            LOOP_INTERVAL="$2"; shift 2 ;;
        --max-speed)
            MAX_LINEAR_SPEED="$2"; shift 2 ;;
        --max-turn)
            MAX_ANGULAR_SPEED="$2"; shift 2 ;;
        --no-voice)
            VOICE_ENABLED="false"; shift ;;
        --dry-run|--dryrun)
            PROVIDER="dryrun"; VOICE_ENABLED="false"; shift ;;
        --no-hardware)
            LAUNCH_HARDWARE="false"; shift ;;
        --no-foxglove)
            FOXGLOVE="false"; shift ;;
        --dashboard)
            DASHBOARD="true"; shift ;;
        --no-dashboard)
            DASHBOARD="false"; shift ;;
        --foxglove-port)
            FOXGLOVE_PORT="$2"; shift 2 ;;
        --hybrid)
            HYBRID_MODE="true"; LOOP_INTERVAL="5.0"; shift ;;
        --build)
            BUILD_FIRST="true"; shift ;;
        --help|-h)
            usage ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage."
            exit 1 ;;
    esac
done

# ── Source .env file if present ────────────────────────────────────────────
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ENV_FILE="$(cd "$_SCRIPT_DIR/../../.." && pwd)/.env"
if [[ -f "$_ENV_FILE" ]]; then
    set -a
    source "$_ENV_FILE"
    set +a
fi

# ── Export environment ──────────────────────────────────────────────────────
export MACHINE_TYPE
export DEPTH_CAMERA_TYPE
export LIDAR_TYPE
export need_compile
export LLM_PROVIDER="$PROVIDER"
export EXPLORER_LOG_LEVEL="$LOG_LEVEL"
export USE_TWIST_MUX="true"

# ── Validate API keys (skip for dry-run) ──────────────────────────────────
if [[ "$PROVIDER" == "dryrun" ]]; then
    echo ">>> DRY-RUN MODE: No LLM API calls will be made."
    echo ""
elif [[ "$PROVIDER" == "claude" && -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ERROR: ANTHROPIC_API_KEY is not set."
    echo "  export ANTHROPIC_API_KEY=\"sk-ant-...\""
    echo "  Or use --dry-run to test without API keys."
    exit 1
elif [[ "$PROVIDER" == "openai" && -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set."
    echo "  export OPENAI_API_KEY=\"sk-...\""
    echo "  Or use --dry-run to test without API keys."
    exit 1
fi

if [[ "$VOICE_ENABLED" == "true" && -z "${OPENAI_API_KEY:-}" ]]; then
    echo "WARNING: OPENAI_API_KEY is not set. Voice TTS/STT will not work."
    echo "  Set it or use --no-voice to disable voice."
    echo ""
fi

# ── Source ROS2 and workspace ─────────────────────────────────────────────
source /opt/ros/jazzy/setup.bash
# CycloneDDS shared lib path (Debian Trixie aarch64)
export LD_LIBRARY_PATH="/opt/ros/jazzy/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ── Build if requested ─────────────────────────────────────────────────────
if [[ "$BUILD_FIRST" == "true" ]]; then
    echo ">>> Building autonomous_explorer..."
    cd "$WS_ROOT"
    colcon build --packages-select autonomous_explorer
    echo ""
fi

# ── Source the workspace ────────────────────────────────────────────────────
if [[ -f "$WS_ROOT/install/setup.bash" ]]; then
    echo ">>> Sourcing workspace: $WS_ROOT/install/setup.bash"
    source "$WS_ROOT/install/setup.bash"
else
    echo "ERROR: Workspace not built. Run:"
    echo "  cd $WS_ROOT && colcon build --packages-select autonomous_explorer"
    echo "  Or use: ./launch_explorer.sh --build"
    exit 1
fi

echo "=============================================="
echo "  MentorPi Autonomous Explorer"
echo "=============================================="
echo "  Workspace   : $WS_ROOT"
echo "  Provider    : $PROVIDER"
echo "  Log level   : $LOG_LEVEL"
echo "  Voice       : $VOICE_ENABLED"
echo "  Loop interval: ${LOOP_INTERVAL}s"
echo "  Speed limits : ${MAX_LINEAR_SPEED} m/s, ${MAX_ANGULAR_SPEED} rad/s"
echo "  Hardware     : $LAUNCH_HARDWARE"
echo "  Sensor fusion: EKF (odom_raw + IMU → odom)"
echo "  Foxglove     : $FOXGLOVE (port ${FOXGLOVE_PORT:-8765})"
echo "  Dashboard    : $DASHBOARD"
echo "  Hybrid (Nav2): $HYBRID_MODE"
echo "=============================================="
echo ""

# ── Track background PIDs for cleanup ──────────────────────────────────────
PIDS=()

cleanup() {
    echo ""
    echo ">>> Shutting down..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -INT "$pid" 2>/dev/null || true
        fi
    done
    sleep 2
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    # Kill orphan audio processes that may hold the speaker device
    pkill -9 aplay 2>/dev/null || true
    pkill -9 arecord 2>/dev/null || true
    echo ">>> All nodes stopped."
}
trap cleanup EXIT INT TERM

# ── Launch hardware nodes ──────────────────────────────────────────────────
if [[ "$LAUNCH_HARDWARE" == "true" ]]; then

    # --- Auto-detect USB devices and create symlinks ---
    source "$SCRIPT_DIR/detect_devices.sh"
    detect_devices
    create_symlinks

    # --- 1. STM32 controller (motors, IMU, battery, servos) ---
    if [[ -e /dev/rrc ]]; then
        echo ">>> Launching STM32 controller..."
        ros2 run ros_robot_controller ros_robot_controller &
        PIDS+=($!)
        sleep 2
    else
        echo "WARNING: /dev/rrc not found — STM32 controller not started."
        echo "  Motors, IMU, and battery will be unavailable."
    fi

    # --- 2. Odometry publisher (cmd_vel → motor commands + odometry) ---
    echo ">>> Launching odometry publisher..."
    ros2 run controller odom_publisher &
    PIDS+=($!)
    sleep 1

    # --- 2b. IMU calibration + complementary filter ---
    # Pipeline: /ros_robot_controller/imu_raw → imu_calib → imu_corrected → complementary_filter → /imu
    if [[ -e /dev/rrc ]]; then
        # Find the IMU calibration file
        IMU_CALIB_FILE=""
        if [[ "$need_compile" == "True" ]]; then
            # Installed workspace — use share directory
            _CALIB_SHARE="$(ros2 pkg prefix calibration 2>/dev/null)/share/calibration/config/imu_calib.yaml"
            if [[ -f "$_CALIB_SHARE" ]]; then
                IMU_CALIB_FILE="$_CALIB_SHARE"
            fi
        fi
        # Fallback: try source tree
        if [[ -z "$IMU_CALIB_FILE" ]]; then
            _CALIB_SRC="$WS_ROOT/src/calibration/config/imu_calib.yaml"
            if [[ -f "$_CALIB_SRC" ]]; then
                IMU_CALIB_FILE="$_CALIB_SRC"
            fi
        fi

        if [[ -n "$IMU_CALIB_FILE" ]]; then
            echo ">>> Launching IMU calibration (calib file: $IMU_CALIB_FILE)..."
            ros2 run imu_calib apply_calib --ros-args \
                -p calib_file:="$IMU_CALIB_FILE" \
                -r raw:=/ros_robot_controller/imu_raw \
                -r corrected:=imu_corrected &
            PIDS+=($!)
            sleep 1

            echo ">>> Launching IMU complementary filter..."
            ros2 run imu_complementary_filter complementary_filter_node --ros-args \
                -p use_mag:=false \
                -p do_bias_estimation:=true \
                -p do_adaptive_gain:=true \
                -r /imu/data_raw:=imu_corrected \
                -r imu/data:=imu &
            PIDS+=($!)
            sleep 1
        else
            echo "WARNING: IMU calibration file not found — IMU filter not started."
            echo "  EKF will run without IMU data (reduced accuracy)."
        fi

        # --- 2c. EKF sensor fusion (odom_raw + imu → odom) ---
        echo ">>> Launching EKF sensor fusion..."
        _EKF_CONFIG=""
        if [[ "$need_compile" == "True" ]]; then
            _EKF_SHARE="$(ros2 pkg prefix autonomous_explorer 2>/dev/null)/share/autonomous_explorer/config/ekf.yaml"
            if [[ -f "$_EKF_SHARE" ]]; then
                _EKF_CONFIG="$_EKF_SHARE"
            fi
        fi
        if [[ -z "$_EKF_CONFIG" ]]; then
            _EKF_CONFIG="$WS_ROOT/src/autonomous_explorer/config/ekf.yaml"
        fi

        ros2 run robot_localization ekf_node --ros-args \
            --params-file "$_EKF_CONFIG" \
            -r odometry/filtered:=odom \
            -r cmd_vel:=controller/cmd_vel &
        PIDS+=($!)
        sleep 1
        echo ">>> Sensor fusion pipeline active: odom_raw + IMU → /odom"
    else
        echo "WARNING: STM32 not available — skipping IMU filter and EKF."
    fi

    # --- 3. LiDAR ---
    if [[ -e /dev/ldlidar ]]; then
        echo ">>> Launching LiDAR ($LIDAR_TYPE)..."
        ros2 launch peripherals lidar.launch.py &
        PIDS+=($!)
        sleep 2
    else
        echo "WARNING: No LiDAR serial device found — LiDAR not started."
    fi

    # --- 4. Depth camera (if driver is available) ---
    if ros2 pkg list 2>/dev/null | grep -E "deptrum-ros-driver-aurora930|ascamera" >/dev/null 2>&1 || false; then
        echo ">>> Launching depth camera ($DEPTH_CAMERA_TYPE)..."
        ros2 launch peripherals depth_camera.launch.py &
        PIDS+=($!)
        sleep 2
    else
        echo "WARNING: Camera driver not installed — camera not started."
        echo "  Explorer will use dummy frame in dry-run or wait for camera in live mode."
    fi

    echo ">>> Hardware nodes started."
    sleep 1
fi

# ── Launch Foxglove Bridge (remote visualization) ────────────────────────
if [[ "$FOXGLOVE" == "true" ]]; then
    _FG_PORT="${FOXGLOVE_PORT:-8765}"
    if ros2 pkg list 2>/dev/null | grep -q foxglove_bridge; then
        echo ">>> Launching Foxglove Bridge on port $_FG_PORT..."
        echo "    Connect from Mac: Open Foxglove → 'Open connection' → ws://$(hostname -I | awk '{print $1}'):$_FG_PORT"
        ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:="$_FG_PORT" &
        PIDS+=($!)
        sleep 1
    else
        echo "WARNING: foxglove_bridge not installed — remote visualization disabled."
        echo "  Install: sudo apt install ros-jazzy-foxglove-bridge"
    fi
fi

# ── Launch twist_mux (priority-based cmd_vel multiplexer) ─────────────────
if ros2 pkg list 2>/dev/null | grep -q twist_mux; then
    _TMUX_CONFIG=""
    if [[ "$need_compile" == "True" ]]; then
        _TMUX_SHARE="$(ros2 pkg prefix autonomous_explorer 2>/dev/null)/share/autonomous_explorer/config/twist_mux.yaml"
        if [[ -f "$_TMUX_SHARE" ]]; then
            _TMUX_CONFIG="$_TMUX_SHARE"
        fi
    fi
    if [[ -z "$_TMUX_CONFIG" ]]; then
        _TMUX_CONFIG="$WS_ROOT/src/autonomous_explorer/config/twist_mux.yaml"
    fi

    echo ">>> Launching twist_mux (priority cmd_vel multiplexer)..."
    echo "    Priorities: safety(0) > joystick(1) > nav2(2) > autonomous(3)"
    ros2 run twist_mux twist_mux --ros-args \
        --params-file "$_TMUX_CONFIG" \
        -r cmd_vel_out:=/controller/cmd_vel &
    PIDS+=($!)
    sleep 1
else
    echo "WARNING: twist_mux not installed — using direct cmd_vel."
    echo "  Install: sudo apt install ros-jazzy-twist-mux"
    export USE_TWIST_MUX="false"
fi

# ── Launch the explorer ────────────────────────────────────────────────────
if [[ "$HYBRID_MODE" == "true" ]]; then
    echo ">>> Launching HYBRID explorer (SLAM + Nav2 + LLM)..."
    ros2 launch autonomous_explorer hybrid_explorer.launch.py \
        llm_provider:="$PROVIDER" \
        loop_interval:="$LOOP_INTERVAL" \
        voice_enabled:="$VOICE_ENABLED" &
    PIDS+=($!)
else
    echo ">>> Launching autonomous explorer (direct mode)..."
    ros2 launch autonomous_explorer explorer.launch.py \
        llm_provider:="$PROVIDER" \
        loop_interval:="$LOOP_INTERVAL" \
        voice_enabled:="$VOICE_ENABLED" \
        max_linear_speed:="$MAX_LINEAR_SPEED" \
        max_angular_speed:="$MAX_ANGULAR_SPEED" &
    PIDS+=($!)
fi

echo ""
echo ">>> Explorer is running! Press Ctrl+C to stop."
echo ">>> Logs: ${EXPLORER_LOG_DIR:-~/mentorpi_explorer/logs}/"
echo ""

if [[ "$DASHBOARD" == "true" ]]; then
    # Wait for explorer node to publish /explorer/status before launching dashboard
    echo ">>> Waiting 5s for nodes to initialize before launching dashboard..."
    sleep 5
    echo ">>> Launching curses dashboard (Ctrl+C exits dashboard + stops all nodes)..."
    echo ""
    # Run dashboard in foreground — user sees the TUI directly
    # On exit (Ctrl+C or 'q'), the cleanup trap fires and kills everything
    ros2 run autonomous_explorer dashboard
else
    echo ">>> TIP: Monitor with the dashboard in another terminal:"
    echo ">>>   ros2 run autonomous_explorer dashboard"
    echo ">>> Or attach to the screen session (if using jeeves_agent.launch.py):"
    echo ">>>   screen -r jeeves_dash"
    echo ""
    # Wait for all background processes
    wait
fi
