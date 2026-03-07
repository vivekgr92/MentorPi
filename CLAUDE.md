# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MentorPi is a ROS2-based educational robotics platform supporting multiple chassis types (Mecanum, Ackermann, Tank) with vision, navigation, SLAM, and LLM integration. It runs on a Raspberry Pi with a custom STM32 controller board. Primary language is Python 3.

**Current robot**: MentorPi T1 with tracked (tank) chassis, running ROS2 Jazzy on Ubuntu 24.04.

## Build & Run

```bash
# Build all packages
colcon build

# Build a single package
colcon build --packages-select <package_name>

# Source the workspace
source install/setup.bash

# Run tests for a package
colcon test --packages-select <package_name>
colcon test-result --verbose

# Launch the full system
ros2 launch bringup bringup.launch.py

# Launch individual subsystems
ros2 launch app start_app.launch.py
ros2 launch controller controller.launch.py
ros2 launch peripherals depth_camera.launch.py
ros2 launch peripherals lidar.launch.py
ros2 launch slam rtabmap_slam.launch.py
ros2 launch navigation navigation.launch.py

# Launch autonomous explorer (see section below)
ros2 launch autonomous_explorer explorer.launch.py
```

## Required Environment Variables

These must be set before launching nodes:

- `MACHINE_TYPE` — Robot chassis: `MentorPi_Mecanum`, `MentorPi_Ackermann`, or `MentorPi_Tank`
- `DEPTH_CAMERA_TYPE` — Camera model: `ascamera`, `aurora`, `Dabai`, `usb_cam`, `HP60C`
- `LIDAR_TYPE` — LIDAR model: `ydlidar_G4`, `SLAMTEC_A1`, `LD19`, `LD14P`, `MS200`
- `need_compile` — `True` to use installed (colcon build) packages, `False` to use source paths directly (dev shortcut hardcoded to `/home/ubuntu/ros2_ws/src/`)

Autonomous explorer additional env vars:

- `LLM_PROVIDER` — `claude` or `openai` (default: `claude`)
- `ANTHROPIC_API_KEY` — API key for Claude
- `OPENAI_API_KEY` — API key for OpenAI (also used for TTS/STT regardless of LLM provider)
- `VOICE_ENABLED` — `true` or `false` (default: `true`)
- `EXPLORER_LOG_LEVEL` — `full`, `compact`, or `minimal` (default: `full`)
- `EXPLORER_LOG_DIR` — Log directory (default: `~/mentorpi_explorer/logs`)

## Architecture

### Package Dependency Flow

```
bringup (top-level launcher)
├── driver/controller (odometry, EKF fusion, motor control)
│   └── driver/ros_robot_controller (STM32 board: motors, servos, IMU, LEDs, buzzer)
│       └── driver/sdk (PID, utilities, YAML helpers)
├── peripherals (LIDAR, depth camera, IMU filter, joystick, keyboard teleop)
├── app (vision applications: object tracking, line following, hand gestures, AR)
├── yolov5_ros2 (YOLOv5 object detection node)
├── slam (RTAbMap SLAM)
├── navigation (Nav2 autonomous navigation)
├── large_models (LLM integration: voice, TTS, visual patrol, function calling)
└── autonomous_explorer (LLM vision-driven autonomous exploration — custom package)
```

### Custom Message Packages (CMake/ament_cmake)

- **interfaces** — Vision/control msgs & srvs: `ColorDetect`, `ObjectInfo`, `SetPose2D`, `SetColorDetectParam`, etc.
- **ros_robot_controller_msgs** — Hardware msgs: `MotorState`, `PWMServoState`, `BusServoState`, `RGBState`, `BuzzerState`, etc.
- **large_models_msgs** — AI msgs & srvs: `AgentResult`, `Transport`, `SetModel`, `SetContent`, etc.

All other packages are Python (ament_python/setuptools).

### Key Patterns

- **`need_compile` toggle**: Launch files check this env var. When `False`, paths are hardcoded to `/home/ubuntu/ros2_ws/src/` for rapid iteration without rebuilding. When `True`, uses `get_package_share_directory()`.
- **Chassis abstraction**: `odom_publisher_node.py` and `ros_robot_controller_node.py` branch on `MACHINE_TYPE` to handle different kinematics (Mecanum omnidirectional, Ackermann steering, Tank differential).
- **Sensor fusion**: `controller.launch.py` runs EKF via `robot_localization` to fuse wheel odometry + IMU, configured in `controller/config/ekf.yaml`.
- **Color detection pipeline**: Uses LAB color space. Colors are calibrated and stored as YAML. ROI-based processing for performance.
- **Bilingual comments**: Code contains comments in both Chinese and English (e.g., `# 获取相机类型(get camera type)`).

### ROS2 Topics Reference

| Purpose | Topic | Message Type | QoS |
|---------|-------|-------------|-----|
| Motor control (Twist) | `/controller/cmd_vel` | `geometry_msgs/Twist` | default |
| Motor control (raw) | `/ros_robot_controller/set_motor` | `MotorsState` | default |
| Camera RGB | `/ascamera/camera_publisher/rgb0/image` | `sensor_msgs/Image` | default |
| Camera depth | `/ascamera/camera_publisher/depth0/image_raw` | `sensor_msgs/Image` (uint16, mm) | default |
| LiDAR scan | `/scan_raw` | `sensor_msgs/LaserScan` | BEST_EFFORT |
| PWM servos | `ros_robot_controller/pwm_servo/set_state` | `SetPWMServoState` | default |
| Odometry | `/odom` | `nav_msgs/Odometry` | default |

## Key Source Files

- `src/driver/ros_robot_controller/ros_robot_controller/ros_robot_controller_node.py` — Main hardware controller node
- `src/driver/controller/controller/odom_publisher_node.py` — Odometry calculation for all chassis types
- `src/app/app/object_tracking.py` — Color-based object tracking with PID control
- `src/app/app/line_following.py` — Visual line following with LIDAR obstacle detection
- `src/app/app/lidar_controller.py` — LiDAR obstacle avoidance and following modes
- `src/peripherals/launch/lidar.launch.py` — LIDAR type multiplexer
- `src/peripherals/launch/depth_camera.launch.py` — Camera type multiplexer
- `src/large_models/large_models/config.py` — LLM API configuration (Aliyun, OpenAI, StepFun)
- `src/large_models/large_models/llm_control_move.py` — Voice-controlled movement via LLM
- `src/large_models/large_models/vllm_with_camera.py` — Vision-language model with camera
- `src/autonomous_explorer/` — Autonomous exploration system (see below)

## Autonomous Explorer Package (`src/autonomous_explorer/`)

Custom package that uses LLM vision (Claude or OpenAI) as the robot's brain for autonomous exploration.

### Architecture

Single ROS2 node running a continuous **sense -> think -> act -> speak** loop:

1. **SENSE** — Subscribes to RGB camera, depth camera, and LiDAR
2. **THINK** — Encodes camera frame as base64 JPEG, combines with LiDAR sector distances + depth samples + exploration memory, sends to LLM vision API
3. **ACT** — Executes movement commands (Twist on `/controller/cmd_vel`), controls camera servos for look_around
4. **SPEAK** — Announces observations via OpenAI TTS + aplay through WonderEcho Pro
5. **SAFETY** — LiDAR emergency stop (20cm threshold) overrides all LLM decisions in real-time

### Files

```
src/autonomous_explorer/
├── autonomous_explorer/
│   ├── config.py              # All config: env vars, topics, safety thresholds, servo IDs, logging
│   ├── llm_provider.py        # Provider abstraction with token usage tracking (_meta dict)
│   ├── explorer_node.py       # Main ROS2 node: modular __init__ (9 helpers), voice dispatch table, LLM dashboard dataclass
│   ├── joystick_reader.py     # Pygame joystick reader (daemon thread, 50Hz, SHANWAN/WirelessGamepad)
│   ├── data_logger.py         # Async JSONL logger: background thread, frame saving, compression
│   ├── voice_io.py            # VoiceIO (arecord + Whisper STT + OpenAI TTS), WonderEchoDetector
│   └── exploration_memory.py  # Rolling action log, discovery tracker, stuck detection, shared speech_contains_discovery()
├── scripts/
│   ├── analytics_dashboard.py # Post-session analysis: charts, stats, path map, provider comparison
│   └── dataset_export.py      # Export to CSV, imitation learning pairs, HuggingFace format
├── config/explorer_params.yaml
├── launch/explorer.launch.py
├── package.xml, setup.py, setup.cfg
```

### Multi-LLM Support

Switch providers via `LLM_PROVIDER` env var or launch arg:

```bash
# Claude (default)
ros2 launch autonomous_explorer explorer.launch.py llm_provider:=claude

# OpenAI
ros2 launch autonomous_explorer explorer.launch.py llm_provider:=openai
```

Both providers use the same JSON response schema:
```json
{"action": "forward|backward|turn_left|turn_right|spin_left|spin_right|stop|look_around|investigate",
 "speed": 0.0-1.0, "duration": 0.5-3.0,
 "speech": "what the robot says", "reasoning": "why this action"}
```

Adding a new provider: subclass `LLMProvider` in `llm_provider.py`, implement `analyze_scene()`, register in `create_provider()`.

### Safety Mechanisms

- **LiDAR emergency stop**: Angle-based sector detection at scan rate (~10Hz), hard stop at 20cm
- **LiDAR caution zone**: Speed reduced at 40cm, LLM warned in prompt
- **Motor timeout**: Auto-stop if no command in 5 seconds
- **Speed clamping**: Max 0.2 m/s linear, 0.8 rad/s angular
- **Duration clamping**: Max 3 seconds per action (forces frequent re-evaluation)
- **Forward-only blocking**: Emergency stop blocks forward motion only, allows backing up

### Control Modes

The explorer supports two control modes, toggled via Start button on the gamepad, keyboard (`m`/`a`), or voice ("manual mode" / "autonomous mode"):

**Autonomous mode** (default): LLM-driven sense→think→act→speak loop. Press Enter or say "start exploring" to begin.

**Manual mode**: Direct joystick control via SHANWAN Android Gamepad (`/dev/input/js0`):

| Input | Action |
|---|---|
| Left stick Y (axis 1) | Forward / backward |
| Right stick X (axis 2) | Turn left / right |
| D-pad up/down | Camera tilt |
| D-pad left/right | Camera pan |
| Start button | Toggle autonomous / manual mode |
| Select button | Center camera (manual mode) |

**Keyboard commands**: Enter=start, s=stop, m=manual, a=auto, q=quit, status=show state
**Voice** (via WonderEcho Pro): "start exploring", "stop", "manual mode", "autonomous mode", "go left", "what do you see", etc.

### Joystick Integration (`joystick_reader.py`)

- Self-contained pygame reader running as daemon thread at ~50Hz
- Auto-detects SHANWAN Android Gamepad vs USB WirelessGamepad button layouts
- Deadzone filtering (0.10) on analog sticks
- Handles connect/disconnect/reconnect gracefully (zeros axes, stops motors on disconnect)
- All joystick commands go through `_send_twist()` — same speed clamping and emergency stop as autonomous mode
- Switching modes always stops motors first; switching to autonomous does NOT auto-resume exploration

### Data Logging & Dataset Collection

Every exploration cycle is logged to JSONL for analysis, fine-tuning, or imitation learning. Logging runs on a background thread and never blocks the exploration loop.

**Log levels** (set via `EXPLORER_LOG_LEVEL` env var):
- `full` — JSONL records + RGB/depth frames saved to disk
- `compact` — JSONL records only, no frame files
- `minimal` — Only action names and safety overrides

**Storage layout:**
```
~/mentorpi_explorer/logs/
  exploration_YYYYMMDD_HHMMSS.jsonl        <- JSONL, one record per cycle
  exploration_YYYYMMDD_HHMMSS/frames/rgb/  <- cycle_000001.jpg, ...
  exploration_YYYYMMDD_HHMMSS/frames/depth/<- cycle_000001.png (16-bit)
  exploration_*.jsonl.gz                   <- auto-compressed after 24h
```

**Each JSONL record captures:** sensor_data (lidar scan + sectors, IMU orientation/accel/gyro, odometry x/y/theta, depth, battery, frame paths), llm_input (provider, model, prompts, image resolution), llm_output (raw response, parsed action, tokens used, cost, response time), safety_override (triggered, reason, original vs override action), execution (actual action, motor speeds, servo positions, duration), voice (command received, speech output), exploration_memory (distance, discoveries).

**Analytics dashboard** (`scripts/analytics_dashboard.py`):
```bash
# Text summary + matplotlib charts
python3 scripts/analytics_dashboard.py ~/mentorpi_explorer/logs/
# Save charts as PNG
python3 scripts/analytics_dashboard.py ~/mentorpi_explorer/logs/ --save-dir ./charts/
# Text only (no matplotlib needed)
python3 scripts/analytics_dashboard.py ~/mentorpi_explorer/logs/ --text-only
```
Shows: action distribution pie chart, LLM response time histogram, cumulative cost + safety override scatter, path map from odometry, provider comparison (if multiple used).

**Dataset export** (`scripts/dataset_export.py`):
```bash
# All formats
python3 scripts/dataset_export.py ~/mentorpi_explorer/logs/ --format all -o ./dataset/
# Individual: csv, imitation, huggingface
python3 scripts/dataset_export.py ~/mentorpi_explorer/logs/ --format csv
```
Exports to: CSV (flattened, pandas-ready), imitation learning pairs (observation-action JSONL), HuggingFace dataset format (with dataset_info.json + action class labels).

### Dependencies

```bash
pip install anthropic openai
# Optional for analytics charts:
pip install matplotlib
```

## Testing

Tests use pytest. Each package has `test/test_copyright.py`, `test/test_flake8.py`, and `test/test_pep257.py` for basic code quality checks.

## Hardware

- **Compute**: Raspberry Pi 5 (8GB), Debian Trixie / Bookworm, ROS2 Jazzy
- **Controller**: STM32 board via serial (`/dev/rrc`) — manages motors, servos, IMU, LEDs, buzzer, OLED, battery
- **RRC Lite Controller**: Firmware hex file in `RRC Lite Controller/` directory
- **Chassis**: Tracked (tank treads), differential/skid steering, 4x closed-loop encoder motors
- **Actuators**: 4x DC motors (IDs 1-4, PWM), PWM servos (pan=ID 2, tilt=ID 1, center=1500us, range=500-2500us)
- **Sensors**: 3D depth camera (RGB 1920x1080 + depth, 0.2-4m range), TOF LiDAR STL-19P (360°), IMU (on STM32)
- **Voice**: WonderEcho Pro AI Voice Box (USB mic+speaker, `/dev/wonderecho` serial at 115200, wake word "HELLO HIWONDER")
- **Gamepad**: SHANWAN Android Gamepad (USB, `/dev/input/js0`), used for manual control mode in autonomous_explorer

## Development Notes

- ROS2 is installed on the robot (Pi 5, Ubuntu 24.04), NOT on the dev machine at `/home/vivek`
- Code is written on the dev machine and deployed to the Pi for building and testing
- The `autonomous_explorer` package calls Claude/OpenAI APIs directly (does not use the `speech.so` binary from `large_models`)
- LiDAR data uses BEST_EFFORT QoS: `QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)`
- Depth images are uint16 in millimeters; valid range 0-40000mm
- The existing `large_models` package supports Aliyun, OpenAI, and StepFun via a compiled `speech.so` binary
