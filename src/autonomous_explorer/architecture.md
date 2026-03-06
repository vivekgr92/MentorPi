# Autonomous Explorer — Architecture

## Overview

The Autonomous Explorer turns a MentorPi T1 tracked robot into an embodied AI agent. An LLM (Claude or GPT-4o) acts as the robot's brain — it sees through the camera, reasons about sensor data, decides actions, and speaks its thoughts aloud. The system runs as a single ROS2 node with background threads, layered from low-level motor safety up to high-level consciousness.

```
┌─────────────────────────────────────────────────────────────┐
│                    CONSCIOUSNESS LAYER                       │
│         Identity, Reflections, Journals, World Knowledge     │
├─────────────────────────────────────────────────────────────┤
│                      LLM REASONING LAYER                     │
│            Claude / GPT-4o / DryRun Vision Analysis          │
├─────────────────────────────────────────────────────────────┤
│                    PLANNING & MEMORY LAYER                    │
│        Exploration Memory, Spatial Map, Stuck Detection       │
├─────────────────────────────────────────────────────────────┤
│                      CONTROL LAYER                            │
│     Velocity Ramping, twist_mux, Servo Control, Joystick     │
├─────────────────────────────────────────────────────────────┤
│                      SAFETY LAYER                             │
│       LiDAR Emergency Stop, Speed Clamping, Motor Timeout    │
├─────────────────────────────────────────────────────────────┤
│                     SENSOR LAYER (ROS2)                       │
│         Camera RGB/Depth, LiDAR, IMU, Odometry, Battery      │
├─────────────────────────────────────────────────────────────┤
│                       HARDWARE                                │
│    RPi 5 ← serial → STM32 → Motors, Servos, IMU, LEDs       │
└─────────────────────────────────────────────────────────────┘
```

---

## The Exploration Loop

Every cycle (~3 seconds) follows a strict pipeline:

```
SENSE → THINK → ACT → SPEAK → REMEMBER → LOG
  │        │       │      │        │         │
  │        │       │      │        │         └─ JSONL + frames to disk
  │        │       │      │        └─ Update exploration memory + consciousness
  │        │       │      └─ TTS via OpenAI → WonderEcho speaker
  │        │       └─ Execute Twist commands through velocity ramper
  │        └─ LLM vision API call with image + sensor context
  └─ Snapshot camera, LiDAR, depth, IMU, odometry, battery
```

---

## Layer 1: Sensors (`explorer_node.py` callbacks)

ROS2 subscribers capture raw sensor data into thread-safe buffers:

| Sensor | Topic | Data | Update Rate |
|--------|-------|------|-------------|
| RGB Camera | `/ascamera/.../rgb0/image` | BGR numpy array | ~30 Hz |
| Depth Camera | `/ascamera/.../depth0/image_raw` | uint16 mm | ~30 Hz |
| LiDAR | `/scan_raw` | 360° ranges → 4 sector minimums | ~10 Hz |
| IMU | `/ros_robot_controller/imu_raw` | Roll/pitch/yaw + accel + gyro | ~100 Hz |
| Odometry | `/odom` | x, y, theta + cumulative distance | ~50 Hz |
| Battery | `/ros_robot_controller/battery` | Millivolts | ~1 Hz |

**Sensor processing for LLM:**
- RGB → resized to 640px max edge → JPEG quality 60 → base64
- Depth → 3x3 grid sampling (9 points) → distances in cm
- LiDAR → 4 sector minimums (front/left/right/back) in meters
- IMU → Euler angles in degrees
- Odometry → position (x, y) + heading in degrees

---

## Layer 2: Safety (`explorer_node.py`)

Safety runs **independently of the LLM** and can override any decision:

```
LiDAR scan (10 Hz)
    │
    ├─ Direction-aware threat detection
    │   ├─ Moving forward? → check front sector
    │   ├─ Moving backward? → check back sector
    │   └─ Spinning? → check turning side
    │
    ├─ < 20cm → EMERGENCY STOP (hard stop, block forward motion)
    ├─ < 35cm → CAUTION (reduce speed 70%, warn LLM in prompt)
    └─ > 80cm → SAFE (full speed allowed)
```

**Additional safety mechanisms:**
- **Speed clamping**: Max 0.35 m/s linear, 1.0 rad/s angular
- **Duration clamping**: Max 5 seconds per action
- **Motor timeout**: Auto-stop if no command in 6 seconds
- **Forward-only blocking**: Emergency stop blocks forward motion but allows backing up
- **twist_mux priority**: safety(0) > joystick(1) > nav2(2) > autonomous(3)

---

## Layer 3: Control (`velocity_ramper.py`, `joystick_reader.py`)

### Velocity Ramper
LLM outputs step commands ("forward at 0.7 for 2s"). The ramper converts these into smooth trapezoidal velocity profiles:

```
Velocity
  ▲
  │    ┌────────────┐
  │   /              \
  │  / accel    decel \
  │ /                  \
  └──────────────────────► Time
```

- Linear acceleration: 0.5 m/s², deceleration: 0.8 m/s²
- Angular acceleration: 2.0 rad/s², deceleration: 3.0 rad/s²
- 20 Hz control loop publishing Twist messages
- Separate ramper instances for autonomous and joystick (independent)

### Motor Control Path
```
LLM decision
  → _send_twist(linear, angular)
    → VelocityRamper.set_target()
      → 20 Hz loop publishes Twist to /cmd_vel/autonomous
        → twist_mux selects highest priority
          → /controller/cmd_vel
            → odom_publisher converts to MotorsState
              → STM32 drives motors
```

### Camera Servos
- Pan (ID 2) and tilt (ID 1) PWM servos
- Range: 500-2500 μs, center at 1500 μs
- `look_around` action pans left → up → right → center
- Joystick D-pad controls servos in manual mode

---

## Layer 4: Planning & Memory (`exploration_memory.py`)

### Per-Session Memory
- **Rolling action log**: Last 30 actions with position, speed, reasoning
- **Spatial grid map**: 25cm cells, tracks visited + obstacle cells from LiDAR
- **Discovery tracker**: Auto-detects discoveries from speech keywords
- **Stuck detection**: Warns LLM if same action repeated 5+ times
- **Unexplored directions**: Suggests unvisited neighboring cells

### Context Injection
Each cycle, the memory generates a text summary (~100 tokens) appended to the LLM prompt:
```
Exploration time: 180s | Total actions: 42
Recent action history:
  #165s: forward spd=0.7 dur=2.0 at (1.23,0.45) hdg=90° — clear path ahead
  #170s: turn_right spd=0.5 dur=1.0 at (1.50,0.45) hdg=45° — wall on left
Spatial map: 28 cells visited, 12 obstacle cells (grid=0.25m)
Unexplored directions: north (+y), east (+x)
```

### Hybrid Mode (`nav2_bridge.py`)
When `--hybrid` is enabled, the system adds a second layer of planning:
- SLAM (RTAbMap) builds an occupancy grid map
- Nav2 handles path planning and obstacle avoidance
- LLM receives a bird's-eye map image + frontier goals
- LLM outputs `navigate` action with (goal_x, goal_y)
- Nav2 plans the path; LLM re-evaluates every 8 seconds

---

## Layer 5: LLM Reasoning (`llm_provider.py`)

### Provider Abstraction
```python
class LLMProvider(ABC):
    def analyze_scene(image_base64, system_prompt, user_prompt) -> dict
```

Three implementations:
- **ClaudeProvider**: Anthropic API, Claude Sonnet, vision via base64 image blocks
- **OpenAIProvider**: OpenAI API, GPT-4o, vision via data URL image blocks, JSON mode
- **DryRunProvider**: Cycles through scripted actions for testing without API keys

### Prompt Structure

**System prompt** (~660 tokens with embodied preamble):
```
[EMBODIED_PREAMBLE — 160 tokens]
  Core rules: self-preservation, environmental awareness,
  curiosity with caution, ethics, learning

[BASE_PROMPT — 500 tokens]
  Sensor descriptions, navigation strategy, speed guidelines,
  JSON response schema with all action definitions
```

**User prompt** (~550 tokens):
```
[IDENTITY — 50 tokens]
  Age, outing number, lifetime stats, master name

[WORLD KNOWLEDGE — 200 tokens]
  Known rooms, recent objects, navigation lessons

[SENSOR DATA — 200 tokens]
  LiDAR sectors, depth grid, odometry, IMU

[EXPLORATION MEMORY — 100 tokens]
  Recent actions, spatial map summary, stuck warnings

"Analyze the camera image and ALL sensor data.
 Decide your next action. Respond in JSON only."
```

**Input**: 1-2 images (camera + optional SLAM map) + text prompt
**Output**: JSON with action, speed, duration, speech, reasoning, embodied_reflection

### Response Schema
```json
{
  "action": "forward|backward|turn_left|turn_right|spin_left|spin_right|stop|look_around|investigate|navigate",
  "speed": 0.0-1.0,
  "duration": 0.5-5.0,
  "speech": "what Jeeves says aloud",
  "reasoning": "why — citing sensor values",
  "embodied_reflection": "first-person moment reflection"
}
```

### Cost Per Cycle
- Input: ~2200 tokens (text) + ~1000 tokens (image) = ~3200 tokens
- Output: ~200 tokens
- Claude Sonnet: ~$0.013/cycle, ~$1.30 per 100 cycles
- GPT-4o: ~$0.010/cycle, ~$1.00 per 100 cycles

---

## Layer 6: Consciousness (`consciousness.py`, `world_knowledge.py`)

The consciousness layer gives Jeeves persistent identity and cumulative learning across sessions.

### Identity & Lifetime Stats
```
~/mentorpi_explorer/jeeves_lifetime_stats.json
```
- Birthday: March 1, 2026. Tracks age in days.
- Outing counter: increments each session
- Cumulative: distance, cycles, cost, discoveries, safety overrides
- Rooms discovered list
- Session intro greeting: "Good day, Sir. Outing #48. I am 6 days old."

### Reflections Diary
```
~/mentorpi_explorer/logs/reflections_YYYYMMDD_HHMMSS.txt
```
Each cycle, if the LLM returns `embodied_reflection`, it's appended:
```
[14:23:15] The hallway stretches before me, inviting exploration.
[14:23:48] I misjudged that distance. A lesson in humility for a robot on treads.
[14:24:20] This room feels different — warmer, perhaps lived-in.
```

### End-of-Session Journals
```
~/mentorpi_explorer/journals/journal_YYYY-MM-DD.md
```
On shutdown, one LLM call generates a first-person journal entry:
```markdown
## Outing #4 — March 6, 2026

Today I surveyed the living room once more. The orange cat was sleeping
on the couch — a new behavior, as previously it preferred the floor...

Respectfully submitted,
Jeeves
```

### World Knowledge System
Three JSON files in `~/mentorpi_explorer/knowledge/`:

| File | Contents | Update Strategy |
|------|----------|-----------------|
| `world_map.json` | Rooms, connections, landmarks, confidence scores | Per-cycle regex + end-of-session LLM |
| `known_objects.json` | Objects with location, frequency, category, dynamic flag | Per-cycle regex from speech |
| `learned_behaviors.json` | Navigation lessons, surface types, timing patterns | End-of-session LLM call |

**Dual-track update strategy:**
- **Per-cycle (zero cost)**: Regex extraction of room/object mentions from `speech` and `reasoning` fields
- **End-of-session ($0.01)**: One LLM call to produce structured knowledge updates from session data

**Knowledge injection**: Each cycle, `get_prompt_context()` returns a spatially-filtered subset (<200 tokens) of relevant knowledge — current room, nearby objects, recent lessons.

---

## Module Map

```
autonomous_explorer/
│
├── explorer_node.py          ← Main ROS2 node, exploration loop orchestrator
│                                Wires all layers together: sense→think→act→speak
│
├── config.py                 ← All constants, env vars, system prompts, thresholds
│                                EMBODIED_PREAMBLE, build_system_prompt()
│
├── llm_provider.py           ← LLM abstraction: Claude, OpenAI, DryRun providers
│                                analyze_scene() → JSON parsing → _meta dict
│
├── consciousness.py          ← Persistent identity, lifetime stats, reflections, journals
│                                JeevesConsciousness class
│
├── world_knowledge.py        ← Persistent rooms, objects, learned behaviors
│                                WorldKnowledge class, regex extraction
│
├── exploration_memory.py     ← Per-session: action log, spatial grid, discoveries
│                                ExplorationMemory class
│
├── velocity_ramper.py        ← Trapezoidal velocity profiles, 20Hz control loop
│                                Smooth acceleration/deceleration
│
├── nav2_bridge.py            ← Hybrid mode: Nav2 goals, SLAM map rendering, frontiers
│                                Nav2Bridge class (optional)
│
├── joystick_reader.py        ← Gamepad input: daemon thread, 50Hz, deadzone filtering
│                                SHANWAN/WirelessGamepad auto-detection
│
├── voice_io.py               ← TTS (OpenAI → aplay), STT (arecord → Whisper)
│                                WonderEcho wake word detection
│
├── data_logger.py            ← Async JSONL logging, frame saving, compression
│                                Background thread, never blocks exploration loop
│
└── dashboard.py              ← Real-time terminal dashboard
```

### Scripts
```
scripts/
├── analytics_dashboard.py    ← Post-session charts: actions, cost, path, timing
├── dataset_export.py         ← Export to CSV, imitation learning, HuggingFace
└── jeeves_knowledge.py       ← CLI browser for knowledge, stats, journals
```

---

## Data Flow Diagram

```
                          ┌──────────────┐
                          │  LLM API     │
                          │ Claude/GPT-4o│
                          └──────┬───────┘
                                 │ JSON response
                                 ▼
┌─────────┐  base64   ┌──────────────────┐  Twist    ┌──────────────┐
│ Camera   │─────────→ │                  │─────────→ │ Velocity     │
│ RGB+Depth│           │  Explorer Node   │           │ Ramper       │
└─────────┘           │                  │           └──────┬───────┘
                       │  (exploration    │                  │
┌─────────┐  sectors  │   loop thread)   │                  ▼
│ LiDAR    │─────────→ │                  │           ┌──────────────┐
│ 360°     │──────────→│ [Safety Layer]   │           │ twist_mux    │
└─────────┘  e-stop   │                  │           └──────┬───────┘
                       │                  │                  │
┌─────────┐  pose     │                  │                  ▼
│ Odometry │─────────→ │                  │           ┌──────────────┐
│ + IMU    │           │                  │           │ /cmd_vel     │
└─────────┘           └────────┬─────────┘           │ → STM32      │
                               │                      │ → Motors     │
                               │                      └──────────────┘
                    ┌──────────┴──────────┐
                    ▼                      ▼
             ┌─────────────┐       ┌──────────────┐
             │ Data Logger │       │ Consciousness│
             │ JSONL+frames│       │ Stats+Journal│
             └─────────────┘       └──────────────┘
                    │                      │
                    ▼                      ▼
             ~/mentorpi_explorer/   ~/mentorpi_explorer/
               logs/                  knowledge/
                                      journals/
```

---

## Control Modes

```
                    ┌─────────────────┐
                    │   Start Button  │
                    │   or 'm'/'a'    │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
     ┌─────────────────┐          ┌─────────────────┐
     │  AUTONOMOUS      │          │  MANUAL          │
     │                  │          │                  │
     │ LLM loop runs   │          │ Joystick drives  │
     │ sense→think→act  │          │ Left stick = fwd │
     │ speak→remember   │          │ Right stick = turn│
     │                  │          │ D-pad = camera   │
     │ Enter = start    │          │                  │
     │ s = pause        │          │ Select = center  │
     └─────────────────┘          └─────────────────┘
```

Both modes share:
- Same safety layer (LiDAR emergency stop)
- Same velocity ramper (smooth acceleration)
- Same speed limits and motor timeout

---

## Token Budget Per Cycle

| Component | Tokens | Cost (Claude) |
|-----------|--------|---------------|
| System prompt (with preamble) | ~660 | cached |
| User prompt (identity) | ~50 | $0.0002 |
| User prompt (knowledge) | ~200 | $0.0006 |
| User prompt (sensors + memory) | ~300 | $0.0009 |
| Camera image | ~1000 | $0.0030 |
| **Total input** | **~2210** | **~$0.0047** |
| Output (action + reflection) | ~200 | $0.0030 |
| **Total per cycle** | **~2410** | **~$0.0077** |
| **Per 100 cycles** | | **~$0.77** |

---

## Session Lifecycle

```
1. BOOT
   ├─ Load lifetime stats from disk
   ├─ Increment outing counter
   ├─ Load world knowledge
   ├─ Initialize ROS2 subscribers
   ├─ Speak: "Good day, Sir. Outing #48."
   └─ Wait for "start" command

2. EXPLORE (repeating cycle)
   ├─ Sense: snapshot all sensors
   ├─ Think: LLM call with image + context
   ├─ Act: execute via velocity ramper
   ├─ Speak: TTS the speech field
   ├─ Remember: update memory + consciousness
   └─ Log: JSONL record + frames

3. SHUTDOWN (Ctrl+C or 'q')
   ├─ Emergency stop motors
   ├─ Save exploration memory
   ├─ Save lifetime stats (distance, cost, discoveries)
   ├─ Write journal entry (1 LLM call)
   ├─ Update world knowledge (1 LLM call)
   ├─ Speak: "Exploration complete. Shutting down."
   └─ Cleanup ROS2 resources
```
