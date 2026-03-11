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

- `LLM_PROVIDER` — `claude` or `openai` (default: `openai`)
- `ANTHROPIC_API_KEY` — API key for Claude
- `OPENAI_API_KEY` — API key for OpenAI (primary for LLM, TTS, STT, VLM)
- `VOICE_ENABLED` — `true` or `false` (default: `true`)
- `EXPLORER_LOG_LEVEL` — `full`, `compact`, or `minimal` (default: `full`)
- `EXPLORER_LOG_DIR` — Log directory (default: `~/mentorpi_explorer/logs`)
- `YOLO_ENABLED` — `true` or `false` (default: `true`) — local YOLO11n object detection
- `YOLO_MODEL_PATH` — path to ONNX model (default: `models/yolo11n.onnx` in package)

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

Single ROS2 node running a ROSA-style tool-calling agent loop:

1. **SENSE** — Subscribes to RGB camera, depth camera, and LiDAR. Runs YOLO11n locally on each frame (~200ms on Pi 5).
2. **THINK** — Sends sensor context + camera image (cloud) or YOLO text detections (local) + conversation history to LLM with 7 tools via native tool-calling API.
3. **ACT** — LLM calls tools (navigate, identify, label, speak, etc.). Tool handlers bridge to Nav2, VLM, WorldKnowledge, VoiceIO.
4. **SPEAK** — Announces observations via OpenAI TTS + aplay through WonderEcho Pro (gTTS fallback).
5. **SAFETY** — LiDAR emergency stop (20cm threshold) overrides all LLM decisions in real-time.

**Camera is fixed** — no pan/tilt servos. Robot spins on tracks to look around.

### Files

```
src/autonomous_explorer/
├── autonomous_explorer/
│   ├── config.py              # All config: env vars, topics, safety thresholds, YOLO settings, agent prompt, named constants
│   ├── llm_provider.py        # Provider abstraction with token usage tracking (_meta dict)
│   ├── explorer_node.py       # Main ROS2 node: agent loop, sensor snapshot, YOLO integration
│   ├── tool_registry.py       # ToolDefinition dataclass, ToolRegistry (timeout-enforced), 7 tool JSON schemas
│   ├── tool_handlers.py       # ToolHandlers class: 7 bound handlers + unbound legacy handlers
│   ├── conversation_manager.py # 5-turn sliding window, dual Claude/OpenAI rendering, context builders
│   ├── yolo_detector.py       # YOLO11n ONNX inference, depth-fused distance, text/dict formatters
│   ├── joystick_reader.py     # Pygame joystick reader (daemon thread, 50Hz, SHANWAN/WirelessGamepad)
│   ├── data_logger.py         # Async JSONL logger: background thread, frame saving, compression
│   ├── voice_io.py            # VoiceIO (arecord + Whisper STT + OpenAI TTS + gTTS fallback), WonderEchoDetector
│   ├── agent_logger.py        # Synchronous agent logger: ROS2 rosout + /semantic_map/agent_status for Foxglove
│   ├── exploration_memory.py  # Rolling action log, discovery tracker, stuck detection
│   └── semantic_map_publisher.py # Publishes MarkerArray from WorldKnowledge for Foxglove 3D visualization
├── models/
│   └── yolo11n.onnx           # YOLO11n ONNX model (~11MB, exported from ultralytics)
├── scripts/
│   ├── analytics_dashboard.py # Post-session analysis: charts, stats, path map, provider comparison
│   └── dataset_export.py      # Export to CSV, imitation learning pairs, HuggingFace format
├── config/
│   ├── model_config.yaml      # Provider profiles: cloud, local, hybrid, budget, dryrun
│   └── explorer_params.yaml   # Explorer node defaults
├── launch/explorer.launch.py
├── package.xml, setup.py, setup.cfg
```

### Multi-LLM Support & Model Config

Provider switching via `model_config.yaml` profiles or launch args:

```bash
# Cloud (default) — OpenAI for everything
ros2 launch autonomous_explorer jeeves_agent.launch.py

# Local — LM Studio on 10.0.0.176 (qwen2.5-vl-7b) + gTTS + Google STT
ros2 launch autonomous_explorer jeeves_agent.launch.py model_profile:=local

# Hybrid — OpenAI LLM+VLM, free TTS/STT
ros2 launch autonomous_explorer jeeves_agent.launch.py model_profile:=hybrid

# Budget — Claude LLM+VLM, free TTS/STT
ros2 launch autonomous_explorer jeeves_agent.launch.py model_profile:=budget

# Dry-run — no API keys needed
ros2 launch autonomous_explorer jeeves_agent.launch.py model_profile:=dryrun
```

**`config/model_config.yaml`** defines per-service providers (llm, tts, stt, vlm) with primary+fallback chains:
- **LLM**: OpenAI GPT-4o primary, **no fallback** (hard error → stops exploration, tells user to relaunch with `model_profile:=local`)
- **TTS**: OpenAI TTS-1 primary → gTTS fallback → espeak fallback
- **STT**: OpenAI Whisper primary → Google Speech Recognition fallback
- **VLM**: OpenAI GPT-4o primary → Claude Sonnet fallback (for `identify_objects`)

Adding a new provider: subclass `LLMProvider` in `llm_provider.py`, implement `analyze_scene()` + `agent_turn()`, register in `create_provider()`.

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
pip install anthropic openai ultralytics onnxruntime
# Optional for analytics charts:
pip install matplotlib
```

### YOLO11n Local Object Detection

Local YOLO11n runs on every camera frame (~200ms on Pi 5 CPU, ~200MB RAM). Detections are fused with depth camera for distance estimation.

**Perception cascade** (used by `identify_objects` tool):
1. **VLM (cloud)** — primary. Sends camera frame to GPT-4o / Claude for rich object descriptions.
2. **YOLO (local)** — fallback if VLM fails. Returns object labels + confidence + bbox + depth-fused distance.

**Text-only mode for local LLMs**: When `YOLO_TEXT_ONLY_LOCAL=True` (default) and provider is `local`, camera images are NOT sent to the LLM. Instead, YOLO detections are injected as text (~150 tokens vs 2000-4000 token base64 images), preventing context overflow on small models.

**Key files:**
- `yolo_detector.py` — `YoloDetector` class: lazy ONNX load, `detect()`, `detections_to_text()`, `_estimate_distance()`
- `models/yolo11n.onnx` — 11MB ONNX model (exported via `ultralytics` from `yolo11n.pt`)
- `config.py` — `YOLO_ENABLED`, `YOLO_MODEL_PATH`, `YOLO_CONFIDENCE_THRESHOLD=0.4`, `YOLO_INPUT_SIZE=640`

### Jeeves Agent Architecture (ROSA-style Tool-Calling)

ROSA-style tool-calling agent where the LLM invokes typed tools per turn, enabling multi-step reasoning.

#### Architecture Decision Records

| # | Decision | Rationale |
|---|----------|-----------|
| ADR-1 | Single node with tool dispatch (not multi-node) | Pi 5 CPU budget tight with Nav2+SLAM. Zero extra IPC. |
| ADR-2 | Native tool-calling (Claude tool_use / OpenAI functions) | Eliminates brittle JSON parsing. LLM chains multiple tools per turn. |
| ADR-3 | slam_toolbox instead of RTAbMap | 2D LiDAR only, less CPU/RAM, map serialization for persistence. |
| ADR-4 | VLM primary + YOLO11n fallback for perception | Cloud VLM gives rich descriptions; local YOLO (~200ms) provides reliable fallback + text-only mode for local LLMs. |
| ADR-5 | 5-turn sliding conversation window | Multi-turn context for task continuity, <10K tokens/call. |
| ADR-6 | 7 tools for hackathon (reduced from 14) | Eliminates overlap, reduces LLM confusion. `identify_objects` auto-registers. Camera is fixed (no pan/tilt). |

#### Tool API (7 tools — hackathon demo set)

| Category | Tool | Implementation |
|----------|------|----------------|
| Navigation | `navigate_to` | WorldKnowledge room lookup → Nav2Bridge.navigate_to() |
| Navigation | `explore_frontier` | Nav2Bridge.get_frontier_goals() → navigate to nearest. **Waits** for Nav2 (polls 0.5s, returns `in_progress` after 8s for LLM re-evaluation). |
| Navigation | `go_home` | Delegates to `navigate_to(x=0, y=0)` |
| Perception | `identify_objects` | VLM (cloud) primary → YOLO (local) fallback. Auto-registers objects in WorldKnowledge. Returns objects + scene description + room type guess. |
| Knowledge | `label_room` | Updates WorldKnowledge rooms + stores odom position |
| Knowledge | `query_knowledge` | Queries rooms, objects, connections by type |
| Communication | `speak` | Wraps VoiceIO.speak() |

**Removed tools** (handlers still exist in `tool_handlers.py` but NOT registered):
`move_direct` (Nav2 handles all movement), `look_around` (camera fixed), `describe_scene` (merged into `identify_objects`), `check_surroundings` (LiDAR data in prompt context), `register_object` (auto-registration in `identify_objects`), `save_map` (should auto-save), `listen` (voice handled by wake word pipeline).

#### Core Loop Change

Old: `sense -> think -> single JSON action -> speak` (one action per cycle)
New: `sense -> think -> [tool_call, ...] -> tool_results -> optional follow-up` (multi-step per turn)

The agent loop (`_agent_loop()`) implements a full ReAct cycle per turn, decomposed into focused helpers:
1. **`_prepare_agent_context()`** — snapshot sensors (incl. YOLO detections), build conversation messages with sensor+identity+knowledge context, handle voice instruction injection. For local models with `YOLO_TEXT_ONLY_LOCAL`, skips image and injects YOLO text instead.
2. **`_run_agent_turn()`** — call LLM with tools via native tool-calling API, execute tool dispatch loop (up to `AGENT_MAX_TOOL_ROUNDS=3` rounds, `AGENT_TURN_TIMEOUT=30s`). **Movement limiter**: only 1 movement tool (`navigate_to`, `explore_frontier`, `go_home`, `move_direct`, `look_around`) per LLM response — additional movement calls return an error prompting LLM to re-evaluate after sensors refresh. On LLM hard failure: stops motors, stops exploration, speaks error, logs relaunch instructions.
3. **`_finalize_agent_turn()`** — update dashboard, consciousness, stuck watchdog, loop timing

Similarly, action execution is decomposed:
- **`_execute_action()`** — dispatcher that delegates to:
  - `_execute_nav2_action()` — Nav2 navigate-to with timeout and re-evaluation
  - `_execute_motor_action()` — direct motor control (forward/backward/turn/spin/stop/look_around)
  - `_apply_safety_override()` — LiDAR safety check before forward motion

And prompt building:
- **`_build_user_prompt()`** — assembles final prompt from:
  - `_build_sensor_context()` — odom, IMU, Nav2 map/frontier data
  - `_build_identity_and_knowledge_context()` — consciousness identity + world knowledge

Inspired by [ROSA (NASA JPL)](https://github.com/nasa-jpl/rosa) but without LangChain dependency — native API calls for lower overhead on Pi 5.

#### Agent Mode Launch

```bash
# Legacy single-action mode (default)
ros2 launch autonomous_explorer explorer.launch.py

# Agent mode (ROSA-style tool-calling)
ros2 launch autonomous_explorer explorer.launch.py agent_mode:=true

# Or via env var
AGENT_MODE=true ros2 run autonomous_explorer explorer_node
```

#### Key Agent Files

```
src/autonomous_explorer/autonomous_explorer/
  tool_registry.py           # ToolDefinition dataclass, ToolRegistry (timeout-enforced via ThreadPoolExecutor), 7 tool JSON schemas
  tool_handlers.py           # ToolHandlers class: 7 bound handlers + unbound legacy handlers, VLM→YOLO cascade, auto-registration
  conversation_manager.py    # 5-turn sliding window, dual Claude/OpenAI rendering, context builders
  yolo_detector.py           # YoloDetector: ONNX inference, depth-fused distance, text/dict formatters
  llm_provider.py            # agent_turn() on Claude/OpenAI/DryRun providers (native tool-calling)
  explorer_node.py           # _agent_loop() + _init_agent_mode(), YOLO integration in _snapshot_sensors(), LLM hard-fail, movement limiter
  agent_logger.py            # AgentLogger: synchronous structured logging to ROS2 rosout + /semantic_map/agent_status
  config.py                  # AGENT_MODE, AGENT_SYSTEM_PROMPT (7 tools), YOLO_*, model_config.yaml profiles
  semantic_map_publisher.py  # Reads WorldKnowledge JSON, publishes MarkerArray + agent_status
models/
  yolo11n.onnx               # YOLO11n ONNX model (~11MB)
launch/
  explorer.launch.py         # Legacy single-action mode + agent_mode launch arg
  jeeves_agent.launch.py     # Full stack: slam_toolbox + Nav2 + EKF + twist_mux + foxglove + rosbridge + semantic map + agent
  hybrid_explorer.launch.py  # Legacy RTAbMap-based hybrid (pre-agent)
config/
  model_config.yaml          # Per-service provider config: llm, tts, stt, vlm with primary+fallback + named profiles
  slam_toolbox_params.yaml   # slam_toolbox async config (tuned for Pi 5 + LD19)
  nav2_explorer_params.yaml  # Nav2 DWB controller + NavFn planner + costmaps
  ekf.yaml                   # robot_localization EKF (odom_raw + IMU → odom)
  twist_mux.yaml             # Priority mux: safety > joystick > nav2 > autonomous
  explorer_params.yaml       # Explorer node defaults (LLM, safety, speeds)
  foxglove_jeeves.json       # Foxglove Studio layout (10 panels, import via File > Import)
  foxglove_layout.json       # Semantic map demo layout (3D map + rooms/objects, camera, agent status, tool call plot)
```

#### Jeeves Agent Launch (`jeeves_agent.launch.py`)

Full stack launch with staged startup to allow dependencies to initialize:

| Time | Component | Purpose |
|------|-----------|---------|
| t=0s | STM32 controller, odom publisher, LiDAR, camera, static TFs, twist_mux, foxglove, rosbridge | Hardware drivers, spatial frames, cmd_vel routing, visualization, WebSocket bridges |
| t=3s | IMU filter chain (imu_calib + complementary_filter) | Calibrated IMU → `/imu` topic for EKF |
| t=5s | EKF, slam_toolbox (async) | Sensor fusion (odom_raw + imu → /odom), 2D SLAM from LiDAR → /map |
| t=10s | Nav2 (6 nodes + lifecycle mgr) | Path planning, obstacle avoidance, recovery behaviors |
| t=18s | Semantic map publisher, Explorer (agent mode) | Room/object MarkerArray visualization + ROSA-style tool-calling LLM brain |

```bash
# Full Jeeves stack (single command — launches ALL hardware + nav + agent)
ros2 launch autonomous_explorer jeeves_agent.launch.py

# With OpenAI
ros2 launch autonomous_explorer jeeves_agent.launch.py llm_provider:=openai

# Dry-run (no API keys)
ros2 launch autonomous_explorer jeeves_agent.launch.py llm_provider:=dryrun

# Without hardware (launch drivers separately)
ros2 launch autonomous_explorer jeeves_agent.launch.py hardware:=false

# With ascamera instead of aurora
ros2 launch autonomous_explorer jeeves_agent.launch.py camera_type:=ascamera

# Without foxglove
ros2 launch autonomous_explorer jeeves_agent.launch.py foxglove:=false

# Without EKF (use raw odom)
ros2 launch autonomous_explorer jeeves_agent.launch.py use_ekf:=false
```

#### Foxglove Visualization

Two layouts available. Import in Foxglove Studio via **Layout > Import from file**. Connect to `ws://<PI_IP>:8765`.

**`foxglove_jeeves.json`** — Full telemetry dashboard (10 panels):

| Panel | Type | Shows |
|-------|------|-------|
| Map | 3D | Occupancy grid (`/map`), LiDAR points (`/scan_raw`), robot pose, TF tree |
| Camera | Image | Live RGB feed (`/ascamera/.../rgb0/image`) |
| LiDAR Distances | Plot | Front/left/right/back sector distances over time |
| Motor Velocity | Plot | Linear + angular velocity over time |
| Session Cost | Plot | Cumulative API cost ($) over time |
| Actions | StateTransitions | Control mode + last LLM action |
| E-Stop | Indicator | Green=clear, Red=emergency stop |
| Battery | Indicator | Green=OK, Yellow=low, Red=critical |
| Status JSON | RawMessages | Full `/explorer/status` JSON |
| Logs | Log | Filtered rosout (explorer, slam, nav2) |

**`foxglove_layout.json`** — Semantic map demo layout (4 panels, best for hackathon judges):

| Panel | Type | Shows |
|-------|------|-------|
| Semantic Map | 3D | `/map` + `/semantic_map/rooms` (cyan labels) + `/semantic_map/objects` (yellow labels) + `/scan_raw` + robot TF |
| Camera | Image | Live RGB feed |
| Agent Status | RawMessages | `/semantic_map/agent_status` — LLM reasoning stream |
| Tool Calls | Plot | Tool call count + cumulative cost over time |

#### Semantic Map Visualization

Real-time visualization of Jeeves's spatial knowledge. Room and object labels appear on the 3D map as Jeeves explores. Objects are auto-registered by `identify_objects()`, rooms by `label_room()`.

**Data flow:**
```
label_room() / identify_objects() (auto-register) in tool_handlers.py
  → world_knowledge.save() → ~/mentorpi_explorer/knowledge/*.json
    → semantic_map_publisher.py reads (1Hz file mtime check)
      → publishes MarkerArray on /semantic_map/rooms + /semantic_map/objects
      → saves ~/.jeeves/semantic_map.json (simplified view)
    → subscribes to /explorer/status → republishes reasoning on /semantic_map/agent_status
```

**Published topics:**

| Topic | Type | Description |
|-------|------|-------------|
| `/semantic_map/rooms` | `MarkerArray` | TEXT_VIEW_FACING markers (cyan, z=0.5m) at room coordinates |
| `/semantic_map/objects` | `MarkerArray` | TEXT_VIEW_FACING markers (yellow, z=0.2m) at object coordinates |
| `/semantic_map/agent_status` | `String` | Current LLM reasoning step (extracted from `/explorer/status`) |

**Persistence:** `~/.jeeves/semantic_map.json` — simplified JSON loaded on startup:
```json
{
  "rooms": [{"name": "kitchen", "x": 2.1, "y": -0.5, "objects": ["cup", "bottle"], "labeled_at": "..."}],
  "home_position": {"x": 0.0, "y": 0.0},
  "last_updated": "..."
}
```

**WebSocket bridges:**
- Foxglove Studio: `ws://<PI_IP>:8765` (foxglove_bridge, binary protocol)
- Web dashboards / roslibjs: `ws://<PI_IP>:9090` (rosbridge_websocket, JSON protocol)

#### Pi 5 Resource Budget

| Component | CPU | RAM |
|-----------|-----|-----|
| Hardware drivers (STM32, LiDAR, camera) | 12-19% | 140 MB |
| slam_toolbox | 10-15% | 200-400 MB |
| Nav2 (6 nodes) | 15-25% | 400-800 MB |
| Support (EKF, twist_mux, foxglove) | 6-10% | 75 MB |
| Jeeves agent node (incl. YOLO11n ONNX) | 10-20% | 300-400 MB |
| **Total** | **53-89%** | **~1.1-1.8 GB** |

#### Hackathon Demo Script (5 min)

1. **Explore** (0:00-1:30) — frontier exploration, label rooms, build knowledge graph
2. **Query** (1:30-2:30) — "Where is the refrigerator?" -> knowledge lookup -> speaks answer
3. **Navigate** (2:30-3:30) — "Go to the kitchen" -> Nav2 path planning -> arrives
4. **Return** (3:30-4:15) — "Come back" -> navigates home
5. **Persistence** (4:15-5:00) — restart node -> recalls rooms/objects from previous session

#### End-to-End Demo Verification (2026-03-08)

Full hackathon demo verified across 3 areas. **Status: DEMO-READY, no blocking issues.**

##### Voice-to-LLM Pipeline — VERIFIED

Complete path: WonderEcho wake word → serial packet → `_voice_listener_loop()` → `arecord` 5s → OpenAI Whisper STT (Google fallback) → `_process_voice_command()` keyword dispatch → free-form fallback sets `_pending_voice_instruction` → `_agent_loop()` injects as `** VOICE INSTRUCTION FROM MASTER: "..." **` into user prompt → added to `ConversationManager` with sensor context + identity context + camera image → sent to `agent_turn()` with 7 tools.

No gaps: unknown voice commands (e.g., "find me something to drink") correctly bypass the keyword dispatch table and reach the LLM in agent mode.

##### Agent Reasoning & Tool Chain — VERIFIED

- 7 tools registered with correct JSON schemas (Claude `tool_use` + OpenAI functions format)
- 7 handlers bound in `ToolHandlers.bind_to_registry()` (legacy handlers still exist but unbound)
- ToolRegistry enforces per-tool `timeout_s` via `concurrent.futures.ThreadPoolExecutor`
- ReAct loop: up to `AGENT_MAX_TOOL_ROUNDS=3` rounds per turn, `AGENT_TURN_TIMEOUT=30s`
- Multi-tool chain verified: `explore_frontier()` → `identify_objects()` → `label_room()` → `query_knowledge()` → `navigate_to()` → `speak()` can execute sequentially across rounds
- `identify_objects` auto-registers detected objects in WorldKnowledge (VLM primary → YOLO fallback)
- LLM hard failure: if GPT-4o fails, stops motors, speaks error, tells user to relaunch with `model_profile:=local`
- `ConversationManager`: 5-turn sliding window, supports images, tool call/result pairs, dual Claude/OpenAI rendering
- `WorldKnowledge`: persists rooms/objects/behaviors to `~/mentorpi_explorer/knowledge/*.json`, loads on startup, survives restart
- `Consciousness`: persists lifetime stats to `~/mentorpi_explorer/jeeves_lifetime_stats.json`, journals to `~/mentorpi_explorer/journals/`

##### Navigation, Mapping & Visualization — VERIFIED

- Launch staging: Hardware/TF/mux/foxglove/rosbridge (t=0) → IMU filter (t=3) → EKF+SLAM (t=5) → Nav2 (t=10) → semantic map + agent (t=18)
- slam_toolbox: async mode, `/scan_raw`, 0.05m resolution, map→odom TF at 20Hz, loop closure enabled
- Nav2: NavfnPlanner + DWB controller, obstacle+inflation costmap layers, 0.15m robot radius, spin/backup/wait recovery
- twist_mux: safety(0) > joystick(1) > nav2(2) > autonomous(3) → `/controller/cmd_vel`
- Nav2Bridge: `navigate_to()`, `get_frontier_goals()`, `cancel_navigation()`, `render_map_image()` all implemented
- TF tree complete: `map` → `odom` → `base_footprint` → `{lidar_frame, camera_link}`
- Foxglove: 10-panel telemetry layout (`foxglove_jeeves.json`) + 4-panel semantic map demo layout (`foxglove_layout.json`)
- Semantic map: `semantic_map_publisher` node reads WorldKnowledge JSON at 1Hz, publishes MarkerArray on `/semantic_map/rooms` + `/semantic_map/objects`, persists to `~/.jeeves/semantic_map.json`
- rosbridge: JSON WebSocket on port 9090 for web dashboard / roslibjs clients

##### Known Limitations (non-blocking)

- First `navigate_to()` call may silently queue if Nav2 lifecycle manager hasn't finished configuring (tool returns error, LLM retries — wastes one tool round)
- Voice instruction cleared immediately after injection — if LLM turn fails, instruction is lost (low risk, same-cycle synchronous injection)
- Legacy mode (non-agent) ignores free-form voice commands (not relevant for agent demo)
- YOLO11n ONNX inference takes ~200ms on Pi 5 CPU — runs every sensor snapshot, not parallelized with LLM call yet
- Map auto-save on timer/shutdown not yet implemented (was `save_map` tool, now removed)

##### Agent Debug Logging (`agent_logger.py`)

`AgentLogger` provides structured, synchronous logging for the agent loop. Each log line is formatted as `[AGENT] HH:MM:SS TAG: detail` and published to both ROS2 rosout and `/semantic_map/agent_status` (for Foxglove).

**9 public methods**: `voice_received`, `turn_start`, `llm_request`, `llm_response`, `tool_start`, `tool_result`, `tool_error`, `turn_complete`, `status`

**Integration points** in `explorer_node.py` (all guarded with `if self._agent_log:`):
- `_init_agent_mode()`: creates AgentLogger with publish callback
- `_voice_listener_loop()`: logs voice commands after STT
- `_agent_loop()`: logs turn_start with voice instruction
- `_run_agent_turn()`: logs LLM request/response, tool start/result/error, turn_complete

**Per-tool timing stats**: `get_tool_stats()` returns cumulative call count + total time per tool name.

##### Demo Reliability Features (2026-03-11)

Changes to prevent tool call flooding and improve demo reliability:

1. **Movement limiter**: Only 1 movement tool (`navigate_to`, `explore_frontier`, `go_home`, `move_direct`, `look_around`) per LLM response. Additional movement calls return `"Only one movement tool per turn. Wait for next sensor cycle."` — forces LLM to re-evaluate with fresh sensor data.

2. **`explore_frontier` wait**: Now polls `nav2.is_navigating` every 0.5s instead of fire-and-forget. Returns `in_progress` after `NAV2_TOOL_REEVAL_INTERVAL=8s` to let LLM re-evaluate, or `arrived`/error on completion. Handles e-stop cancellation.

3. **Config tuning**: `AGENT_MAX_TOOL_ROUNDS` 5→3, `AGENT_TURN_TIMEOUT` 120s→30s — prevents runaway tool chains.

4. **Agent does NOT auto-start**: `self.exploring = False` on init. Waits for voice command ("start exploring") before beginning autonomous loop.

#### Migration Path (8 steps — ALL DONE)

1. ~~Add `tool_registry.py` + `conversation_manager.py` (new files, no existing changes)~~ **DONE**
2. ~~Add `agent_turn()` to `LLMProvider` alongside existing `analyze_scene()`~~ **DONE**
3. ~~Implement tool handlers (most wrap existing methods on the node)~~ **DONE**
4. ~~Add `_agent_loop()` alongside `_exploration_loop()` (feature-flagged `agent_mode` param)~~ **DONE**
5. ~~Switch hybrid launch from RTAbMap to slam_toolbox (create `jeeves_agent.launch.py`)~~ **DONE**
6. ~~Add Foxglove visualization panels for demo~~ **DONE**
7. ~~Add YOLO11n local detection + VLM→YOLO cascade + text-only mode for local LLMs~~ **DONE**
8. ~~Reduce 14→7 tools for hackathon demo, update system prompt, add tool timeout enforcement~~ **DONE**

#### Built-from-Source Packages (Debian Trixie)

Nav2, slam_toolbox, and supporting packages are NOT in rospian apt repo. Built from source:

```
src/navigation2/          # --branch jazzy, patched: removed -Werror from nav2_common
src/slam_toolbox/         # --branch jazzy, patched: rviz plugin made optional
src/BehaviorTree.CPP/     # tag 4.6.2
src/bond_core/            # --branch ros2
src/robot_localization/   # --branch jazzy-devel
src/twist_mux/            # --branch rolling
src/imu_tools/            # --branch jazzy
src/angles/               # --branch ros2
src/diagnostics/          # --branch ros2
src/interactive_markers/  # --branch jazzy
src/geographic_info/      # --branch jazzy
src/foxglove_bridge/      # already in tree
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
