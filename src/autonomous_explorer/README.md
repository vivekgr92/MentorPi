# Autonomous Explorer

LLM-powered autonomous exploration for the MentorPi T1 tank robot. Uses Claude or OpenAI vision APIs as the robot's "brain" in a continuous sense-think-act-speak loop.

## Prerequisites

### Hardware
- MentorPi T1 (tank chassis) with STM32 controller board
- Depth camera (ASCamera or compatible)
- 2D LiDAR (any supported model)
- WonderEcho Pro (optional, for voice I/O)
- Raspberry Pi 5 running ROS2 Jazzy

### Software
```bash
# Python dependencies (on the Pi)
pip install anthropic openai

# Optional (for analytics dashboard charts)
pip install matplotlib
```

### API Keys
You need at least one LLM provider API key:
```bash
# Required — set the provider you want to use
export ANTHROPIC_API_KEY="sk-ant-..."   # for Claude (default)
export OPENAI_API_KEY="sk-..."          # for OpenAI (also needed for voice TTS/STT)

# Choose your provider
export LLM_PROVIDER=claude              # "claude" or "openai"
```

## Quick Start

### Option 1: Launch Script (recommended)
```bash
# Make the script executable (one time)
chmod +x src/autonomous_explorer/scripts/launch_explorer.sh

# Run with defaults (Claude provider, full logging)
./src/autonomous_explorer/scripts/launch_explorer.sh

# Run with OpenAI
./src/autonomous_explorer/scripts/launch_explorer.sh --provider openai

# Run without voice I/O
./src/autonomous_explorer/scripts/launch_explorer.sh --no-voice

# Skip hardware launch (if already running)
./src/autonomous_explorer/scripts/launch_explorer.sh --no-hardware

# Minimal logging (less disk usage)
./src/autonomous_explorer/scripts/launch_explorer.sh --log-level minimal

# See all options
./src/autonomous_explorer/scripts/launch_explorer.sh --help
```

### Option 2: Manual Launch
```bash
# 1. Build the package
colcon build --packages-select autonomous_explorer
source install/setup.bash

# 2. Set environment variables
export MACHINE_TYPE=MentorPi_Tank
export DEPTH_CAMERA_TYPE=ascamera
export LIDAR_TYPE=ydlidar_G4
export need_compile=True
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
export LLM_PROVIDER=claude

# 3. Launch hardware nodes (each in a separate terminal)
ros2 launch controller controller.launch.py
ros2 launch peripherals depth_camera.launch.py
ros2 launch peripherals lidar.launch.py

# 4. Launch the explorer
ros2 launch autonomous_explorer explorer.launch.py
```

## Environment Variables

### Required
| Variable | Description | Example |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key (for Claude) | `sk-ant-...` |
| `OPENAI_API_KEY` | OpenAI API key (for GPT-4o and voice) | `sk-...` |
| `MACHINE_TYPE` | Robot chassis type | `MentorPi_Tank` |
| `DEPTH_CAMERA_TYPE` | Camera model | `ascamera` |
| `LIDAR_TYPE` | LiDAR model | `ydlidar_G4` |

### Optional
| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `claude` | LLM backend: `claude` or `openai` |
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Claude model to use |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model to use |
| `EXPLORER_LOG_LEVEL` | `full` | Logging detail: `full`, `compact`, `minimal` |
| `EXPLORER_LOG_DIR` | `~/mentorpi_explorer/logs` | Log output directory |
| `VOICE_ENABLED` | `true` | Enable/disable voice I/O |
| `WONDERECHO_PORT` | `/dev/wonderecho` | WonderEcho serial device |
| `AUDIO_DEVICE` | `plughw:1,0` | ALSA audio device |

## Launch Parameters

Override via command line when launching:
```bash
ros2 launch autonomous_explorer explorer.launch.py \
    llm_provider:=openai \
    loop_interval:=2.0 \
    voice_enabled:=false \
    max_linear_speed:=0.15 \
    max_angular_speed:=0.60
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `llm_provider` | `claude` | LLM provider |
| `loop_interval` | `3.0` | Seconds between LLM analysis cycles |
| `voice_enabled` | `true` | Voice I/O toggle |
| `max_linear_speed` | `0.20` | Max forward/back speed (m/s) |
| `max_angular_speed` | `0.80` | Max rotation speed (rad/s) |

## Data Logging

Every exploration cycle is logged to `~/mentorpi_explorer/logs/`.

### Log Levels

- **`full`** — JSONL records + RGB/depth frame images (highest disk usage)
- **`compact`** — JSONL records only, no images
- **`minimal`** — Only action and safety override data

### Log Structure
```
~/mentorpi_explorer/logs/
  exploration_20260302_143000.jsonl       # cycle-by-cycle data
  exploration_20260302_143000/
    frames/
      rgb/cycle_000001.jpg                # camera frames (full mode)
      depth/cycle_000001.png
  exploration_20260301_120000.jsonl.gz     # auto-compressed old sessions
```

### Analytics Dashboard

Visualize exploration session data:
```bash
# Analyze a single session
python3 scripts/analytics_dashboard.py ~/mentorpi_explorer/logs/exploration_*.jsonl

# Analyze all sessions in a directory
python3 scripts/analytics_dashboard.py ~/mentorpi_explorer/logs/

# Save charts as PNG files
python3 scripts/analytics_dashboard.py ~/mentorpi_explorer/logs/ --save-dir ./charts/

# Text-only mode (no matplotlib needed)
python3 scripts/analytics_dashboard.py ~/mentorpi_explorer/logs/ --text-only
```

Charts generated:
- Action distribution (pie chart)
- LLM response time histogram
- Cumulative cost + safety override markers
- Robot path map from odometry
- Provider comparison (if multiple providers used)

### Dataset Export

Export logs to ML-ready formats:
```bash
# Export to all formats
python3 scripts/dataset_export.py ~/mentorpi_explorer/logs/ --format all -o ./dataset/

# Export to CSV only (for pandas/Excel)
python3 scripts/dataset_export.py ~/mentorpi_explorer/logs/ --format csv

# Export observation-action pairs (for imitation learning)
python3 scripts/dataset_export.py ~/mentorpi_explorer/logs/ --format imitation

# Export as HuggingFace dataset
python3 scripts/dataset_export.py ~/mentorpi_explorer/logs/ --format huggingface
```

## Safety

- **Emergency stop**: LiDAR detects obstacle within 20cm — motors halt immediately
- **Caution zone**: Obstacles within 40cm — speed reduced, LLM warned
- **Motor timeout**: Motors stop after 5s without a new command
- **Speed limits**: Capped at 0.20 m/s linear, 0.80 rad/s angular
- **Ctrl+C**: Graceful shutdown — stops motors, closes logs, flushes data

## ROS2 Topics Used

| Topic | Type | Direction |
|-------|------|-----------|
| `/ascamera/camera_publisher/rgb0/image` | `sensor_msgs/Image` | Subscribe |
| `/ascamera/camera_publisher/depth0/image_raw` | `sensor_msgs/Image` | Subscribe |
| `/scan_raw` | `sensor_msgs/LaserScan` | Subscribe |
| `/imu/data_raw` | `sensor_msgs/Imu` | Subscribe |
| `/odom` | `nav_msgs/Odometry` | Subscribe |
| `/controller/cmd_vel` | `geometry_msgs/Twist` | Publish |
| `ros_robot_controller/pwm_servo/set_state` | `SetPWMServoState` | Publish |

## Troubleshooting

**"No camera data received"** — Ensure the depth camera node is running:
```bash
ros2 topic echo /ascamera/camera_publisher/rgb0/image --once
```

**"No LiDAR data"** — Check the LiDAR node:
```bash
ros2 topic echo /scan_raw --once
```

**"API key not set"** — Export the required key:
```bash
export ANTHROPIC_API_KEY="your-key"
```

**"Voice not working"** — Check the WonderEcho device exists:
```bash
ls -la /dev/wonderecho
```

**High disk usage from logging** — Switch to compact or minimal logging:
```bash
export EXPLORER_LOG_LEVEL=compact
```
