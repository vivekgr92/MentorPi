# Autonomous Explorer — Launch Instructions

All commands assume you're on the Pi in `~/Projects/MentorPi`.

## Prerequisites (every terminal)

```bash
cd ~/Projects/MentorPi
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export LD_LIBRARY_PATH="/opt/ros/jazzy/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH"
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

## Build

```bash
# Build everything
colcon build

# Build just the explorer
colcon build --packages-select autonomous_explorer

# Build foxglove bridge
colcon build --packages-select foxglove_bridge
```

---

## Launch (4 terminals)

### Terminal 1: STM32 Controller

```bash
sudo ln -sf /dev/ttyACM0 /dev/rrc
ros2 run ros_robot_controller ros_robot_controller
```

### Terminal 2: Odometry Publisher

```bash
ros2 run controller odom_publisher \
  --ros-args \
  -p base_frame_id:=base_footprint \
  -p odom_frame_id:=odom \
  -p pub_odom_topic:=true
```

### Terminal 3: Foxglove Bridge

```bash
ros2 run foxglove_bridge foxglove_bridge --ros-args -p port:=8765
```

Connect from Foxglove Studio (desktop app) at: `ws://<pi-ip>:8765`

### Terminal 4: Explorer Node

```bash
# With Claude
export LLM_PROVIDER=claude
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...       # needed for TTS/STT
ros2 run autonomous_explorer explorer_node

# With OpenAI
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
ros2 run autonomous_explorer explorer_node

# Dry run (no API keys needed)
export LLM_PROVIDER=dryrun
ros2 run autonomous_explorer explorer_node
```

---

## Keyboard Controls (in explorer terminal)

| Key       | Action                          |
|-----------|---------------------------------|
| Enter     | Start exploration               |
| `s`       | Stop / pause exploration        |
| `m`       | Switch to manual (joystick) mode|
| `a`       | Switch to autonomous mode       |
| `status`  | Show current state              |
| `q`       | Quit                            |

---

## Gamepad Controls (manual mode)

| Input             | Action               |
|-------------------|----------------------|
| Left stick Y      | Forward / backward   |
| Right stick X     | Turn left / right    |
| D-pad up/down     | Camera tilt          |
| D-pad left/right  | Camera pan           |
| Start button      | Toggle auto/manual   |
| Select button     | Center camera        |

---

## Foxglove Studio

### Connection
- Download desktop app from https://foxglove.dev/download
- Open connection -> Foxglove WebSocket -> `ws://<pi-ip>:8765`
- Browser (`app.foxglove.dev`) won't work with `ws://` — use the desktop app

### Useful Panels

| Panel Type    | Topic                                          | Shows               |
|---------------|------------------------------------------------|----------------------|
| Image         | `/ascamera/camera_publisher/rgb0/image`         | Camera feed          |
| Plot          | `/controller/cmd_vel`                           | Motor velocities     |
| Raw Messages  | `/explorer/status`                              | LLM action, cost, reasoning |
| 3D / Map      | `/scan_raw`                                     | LiDAR scan           |
| Plot          | `/odom`                                         | Robot position x, y  |
| Plot          | `/ros_robot_controller/battery`                 | Battery voltage      |

### Kill Foxglove (if port stuck)

```bash
fuser -k 8765/tcp
```

---

## Jeeves Knowledge Browser (no ROS2 needed)

```bash
python3 scripts/jeeves_knowledge.py              # summary dashboard
python3 scripts/jeeves_knowledge.py stats         # lifetime stats
python3 scripts/jeeves_knowledge.py rooms         # spatial knowledge
python3 scripts/jeeves_knowledge.py objects        # known objects
python3 scripts/jeeves_knowledge.py lessons        # learned behaviors
python3 scripts/jeeves_knowledge.py journal        # today's journal
python3 scripts/jeeves_knowledge.py reflections    # today's reflections
```

---

## Analytics Dashboard

```bash
# Text + charts
python3 scripts/analytics_dashboard.py ~/mentorpi_explorer/logs/

# Save charts as PNG
python3 scripts/analytics_dashboard.py ~/mentorpi_explorer/logs/ --save-dir ./charts/

# Text only (no matplotlib)
python3 scripts/analytics_dashboard.py ~/mentorpi_explorer/logs/ --text-only
```

---

## Dataset Export

```bash
# All formats
python3 scripts/dataset_export.py ~/mentorpi_explorer/logs/ --format all -o ./dataset/

# Individual formats
python3 scripts/dataset_export.py ~/mentorpi_explorer/logs/ --format csv
python3 scripts/dataset_export.py ~/mentorpi_explorer/logs/ --format imitation
python3 scripts/dataset_export.py ~/mentorpi_explorer/logs/ --format huggingface
```

---

## Data Locations

| What                  | Path                                              |
|-----------------------|---------------------------------------------------|
| Lifetime stats        | `~/mentorpi_explorer/jeeves_lifetime_stats.json`   |
| Session logs          | `~/mentorpi_explorer/logs/exploration_*.jsonl`     |
| RGB frames            | `~/mentorpi_explorer/logs/exploration_*/frames/rgb/` |
| Depth frames          | `~/mentorpi_explorer/logs/exploration_*/frames/depth/` |
| Reflections           | `~/mentorpi_explorer/logs/reflections_*.txt`       |
| Journals              | `~/mentorpi_explorer/journals/journal_*.md`        |
| World knowledge       | `~/mentorpi_explorer/knowledge/world_map.json`     |
| Known objects         | `~/mentorpi_explorer/knowledge/known_objects.json` |
| Learned behaviors     | `~/mentorpi_explorer/knowledge/learned_behaviors.json` |
| Exploration memory    | `/tmp/explorer_memory.json`                        |

---

## Environment Variables Reference

| Variable              | Default         | Description                    |
|-----------------------|-----------------|--------------------------------|
| `LLM_PROVIDER`        | `claude`        | `claude`, `openai`, or `dryrun`|
| `ANTHROPIC_API_KEY`   |                 | Claude API key                 |
| `OPENAI_API_KEY`      |                 | OpenAI key (also for TTS/STT) |
| `VOICE_ENABLED`       | `true`          | Enable/disable TTS/STT        |
| `EXPLORER_LOG_LEVEL`  | `full`          | `full`, `compact`, `minimal`   |
| `EXPLORER_LOG_DIR`    | `~/mentorpi_explorer/logs` | Log directory       |
| `JEEVES_DATA_DIR`     | `~/mentorpi_explorer`      | Consciousness data  |
| `USE_TWIST_MUX`       | `true`          | Use twist_mux priority muxer  |
| `USE_NAV2`            | `false`         | Enable hybrid Nav2+SLAM mode  |
