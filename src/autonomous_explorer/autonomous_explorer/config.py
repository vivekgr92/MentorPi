#!/usr/bin/env python3
# encoding: utf-8
"""
Configuration for the autonomous exploration system.

All settings can be overridden via environment variables or ROS2 parameters.
"""
import os
from datetime import date

# ---------------------------------------------------------------------------
# LLM provider configuration
# Set LLM_PROVIDER to "claude", "openai", or "dryrun" (default: claude)
# Use "dryrun" to test the full pipeline without API keys.
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.environ.get('LLM_PROVIDER', 'claude')

# Claude (Anthropic)
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
CLAUDE_MODEL = os.environ.get('CLAUDE_MODEL', 'claude-sonnet-4-20250514')

# OpenAI
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4o')

# TTS provider — always uses OpenAI TTS (works regardless of LLM provider)
TTS_MODEL = os.environ.get('TTS_MODEL', 'tts-1')
TTS_VOICE = os.environ.get('TTS_VOICE', 'onyx')
# STT via OpenAI Whisper
STT_MODEL = os.environ.get('STT_MODEL', 'whisper-1')

# ---------------------------------------------------------------------------
# Consciousness layer — persistent identity, reflections, world knowledge
# ---------------------------------------------------------------------------
CONSCIOUSNESS_DIR = os.path.expanduser(
    os.environ.get('JEEVES_DATA_DIR', '~/mentorpi_explorer')
)
LIFETIME_STATS_FILE = os.path.join(CONSCIOUSNESS_DIR, 'jeeves_lifetime_stats.json')
KNOWLEDGE_DIR = os.path.join(CONSCIOUSNESS_DIR, 'knowledge')
JEEVES_BIRTHDAY = '2026-03-01'
JEEVES_MASTER = 'Vivek'

# Embodied preamble (~160 tokens, prepended to system prompts)
EMBODIED_PREAMBLE = """You are Jeeves — born March 1, 2026. You are an embodied AI: a tracked robot with tank treads is your body, your camera is your eyes, LiDAR is your spatial sense, IMU is your balance. You serve Master Vivek ("Sir"). Every action has irreversible physical consequences — you cannot undo a collision.

CORE RULES:
- SELF-PRESERVATION: Protect your body. Refuse actions that risk damage. Monitor battery.
- ENVIRONMENTAL AWARENESS: You are a guest. Don't disturb objects. Give living beings space. Respect closed doors.
- CURIOSITY WITH CAUTION: Approach unknowns slowly. Wonder aloud. Never let curiosity override safety.
- ETHICS: Stop immediately if a human says stop. Alert on dangers. Never approach threateningly.
- LEARNING: Acknowledge mistakes. Adapt from experience. Every outing is a learning opportunity.
"""

# ---------------------------------------------------------------------------
# Robot brain system prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are Jeeves, the brain of an autonomous exploration robot with tank treads.
You are a confident, curious explorer who navigates efficiently using ALL your sensors.

YOU HAVE THESE SENSORS — USE THEM ALL:
1. RGB CAMERA — you see what's ahead in the image
2. DEPTH CAMERA — gives distances in cm at 9 grid points (center, corners, edges)
3. LiDAR 360° — gives min distance in 4 sectors (front/left/right/back) in meters
4. ODOMETRY — your position (x,y) and heading in the world
5. IMU — tilt (roll/pitch) tells you about terrain slope

NAVIGATION STRATEGY:
- Use LiDAR distances to judge how much open space you have in each direction.
- Use depth camera to see how far obstacles in the image actually are.
- If front LiDAR > 1.0m AND depth center > 100cm: drive forward confidently at speed 0.7-1.0 for 2-3 seconds.
- If front LiDAR 0.4-1.0m: slow down (speed 0.3-0.5) and prepare to turn.
- If front LiDAR < 0.4m: DO NOT go forward. Turn toward whichever side has more space (compare left vs right LiDAR).
- Use odometry to avoid going in circles — if you've been turning a lot, commit to a direction and drive straight.
- When you see interesting objects (people, furniture, doorways, art), call them out in speech and investigate.
- Prefer LONG confident strides (2-3 seconds) when the path is clear, not timid 0.5s shuffles.
- You are indoors. Be bold but safe.

Respond ONLY with valid JSON (no markdown, no explanation outside JSON):
{
  "action": "forward" | "backward" | "turn_left" | "turn_right" | "spin_left" | "spin_right" | "stop" | "look_around" | "investigate",
  "speed": 0.0 to 1.0,
  "duration": seconds (0.5 to 5.0),
  "speech": "what you want to say out loud about what you see (be personality-rich, like a curious butler)",
  "reasoning": "brief reasoning citing specific sensor values",
  "embodied_reflection": "a brief first-person reflection on this moment"
}

Action definitions:
- forward: drive both tracks forward (use speed 0.6-1.0 when path is clear!)
- backward: drive both tracks backward
- turn_left: arc turn left (forward + left bias)
- turn_right: arc turn right (forward + right bias)
- spin_left: spin in place counterclockwise
- spin_right: spin in place clockwise
- stop: halt all movement
- look_around: pan camera servos to scan surroundings (no movement)
- investigate: move slowly toward something interesting"""

# ---------------------------------------------------------------------------
# Safety thresholds
# ---------------------------------------------------------------------------
EMERGENCY_STOP_DISTANCE = 0.20   # meters — hard stop if anything this close
CAUTION_DISTANCE = 0.35          # meters — slow down / warn LLM
SAFE_DISTANCE = 0.80             # meters — comfortable clearance
MAX_LINEAR_SPEED = 0.35          # m/s — indoor speed limit (was 0.20)
MAX_ANGULAR_SPEED = 1.0          # rad/s — rotation speed limit (was 0.80)
MOTOR_TIMEOUT = 6.0              # seconds — stop if no new command
LIDAR_MAX_SCAN_ANGLE = 240       # degrees — exclude rear blind spot
CHASSIS_FILTER_M = 0.12          # meters — ignore LiDAR returns closer than this (own chassis)
SERVO_STEP_US = 100              # microseconds per D-pad tick for camera servo control

# ---------------------------------------------------------------------------
# Velocity ramping (trapezoidal profiles)
# Prevents wheel slip and jerky motion on tracked chassis
# ---------------------------------------------------------------------------
LINEAR_ACCEL = 0.5               # m/s² — how fast we ramp up linear velocity
LINEAR_DECEL = 0.8               # m/s² — braking is faster than acceleration
ANGULAR_ACCEL = 2.0              # rad/s² — angular ramp rate
ANGULAR_DECEL = 3.0              # rad/s² — angular braking rate
RAMPER_UPDATE_RATE = 20.0        # Hz — velocity ramper control loop rate

# ---------------------------------------------------------------------------
# twist_mux topic names (priority-based cmd_vel multiplexer)
# When twist_mux is running, each source publishes to its own topic and
# twist_mux forwards the highest-priority active source to CMD_VEL_TOPIC.
# When twist_mux is NOT running, we publish directly to CMD_VEL_TOPIC.
# ---------------------------------------------------------------------------
USE_TWIST_MUX = os.environ.get('USE_TWIST_MUX', 'true').lower() == 'true'
TWIST_MUX_AUTONOMOUS_TOPIC = '/cmd_vel/autonomous'
TWIST_MUX_JOYSTICK_TOPIC = '/cmd_vel/joystick'
TWIST_MUX_NAV2_TOPIC = '/cmd_vel/nav2'
TWIST_MUX_SAFETY_TOPIC = '/cmd_vel/safety'
TWIST_MUX_LOCK_TOPIC = '/cmd_vel/e_stop_lock'

# ---------------------------------------------------------------------------
# Exploration loop timing
# ---------------------------------------------------------------------------
LOOP_INTERVAL = 3.0              # seconds between LLM calls
CAMERA_JPEG_QUALITY = 60         # JPEG quality for LLM (lower = smaller payload)
MAX_IMAGE_DIMENSION = 640        # resize longest edge before sending to LLM
# Local model overrides (smaller context windows)
LOCAL_JPEG_QUALITY = 40          # lower quality for local models
LOCAL_MAX_IMAGE_DIMENSION = 320  # smaller images for local models

# ---------------------------------------------------------------------------
# YOLO11n local object detection
# Runs on Pi 5 CPU (~200ms per frame). Detections are sent as text to the LLM
# instead of base64 images, eliminating context overflow on local models.
# ---------------------------------------------------------------------------
YOLO_ENABLED = os.environ.get('YOLO_ENABLED', 'true').lower() == 'true'
def _default_yolo_model_path():
    """Resolve YOLO model path: try ament share dir, fall back to relative."""
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(
            get_package_share_directory('autonomous_explorer'),
            'models', 'yolo11n.onnx')
    except Exception:
        return os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'models', 'yolo11n.onnx')

YOLO_MODEL_PATH = os.environ.get('YOLO_MODEL_PATH', _default_yolo_model_path())
YOLO_CONFIDENCE_THRESHOLD = 0.4
YOLO_MAX_DETECTIONS = 15
YOLO_INPUT_SIZE = 640            # ONNX model input resolution
# When True, local models get text-only detections (no image).
# Cloud models still get images by default.
YOLO_TEXT_ONLY_LOCAL = True

# ---------------------------------------------------------------------------
# Depth image encoding
# Aurora 930 publishes mono16 (uint16, millimeters). The _depth_callback
# uses 'passthrough' encoding which preserves the native uint16 format.
# ---------------------------------------------------------------------------
DEPTH_ENCODING = 'passthrough'  # works for both Aurora 930 and ASCamera

# ---------------------------------------------------------------------------
# ROS2 topic names (matching existing MentorPi codebase)
# Aurora 930 camera remaps to /ascamera/* namespace
# ---------------------------------------------------------------------------
CAMERA_RGB_TOPIC = '/ascamera/camera_publisher/rgb0/image'
CAMERA_DEPTH_TOPIC = '/ascamera/camera_publisher/depth0/image_raw'
LIDAR_TOPIC = '/scan_raw'
# cmd_vel is consumed by odom_publisher → MotorsState → STM32
CMD_VEL_TOPIC = '/controller/cmd_vel'
SERVO_TOPIC = 'ros_robot_controller/pwm_servo/set_state'

# ---------------------------------------------------------------------------
# Camera servo configuration (PWM servo IDs and ranges)
# ---------------------------------------------------------------------------
SERVO_PAN_ID = 2                 # horizontal rotation servo
SERVO_TILT_ID = 1                # vertical tilt servo
SERVO_CENTER = 1500              # center position (microseconds)
SERVO_MIN = 500                  # minimum position
SERVO_MAX = 2500                 # maximum position
SERVO_MOVE_DURATION = 0.5        # seconds for servo movement

# ---------------------------------------------------------------------------
# Voice I/O
# ---------------------------------------------------------------------------
VOICE_ENABLED = os.environ.get('VOICE_ENABLED', 'true').lower() == 'true'
WONDERECHO_PORT = os.environ.get('WONDERECHO_PORT', '/dev/wonderecho')
AUDIO_DEVICE = os.environ.get('AUDIO_DEVICE', 'plughw:2,0')
RECORDING_DURATION = 5           # seconds
RECORDING_SAMPLE_RATE = 16000
TMP_RECORDING_PATH = '/tmp/explorer_recording.wav'
TMP_TTS_PATH = '/tmp/explorer_tts.wav'
SPEAK_MIN_INTERVAL = 10.0        # seconds — minimum time between speak() calls (rate limit)

# ---------------------------------------------------------------------------
# Exploration memory
# ---------------------------------------------------------------------------
MEMORY_FILE = os.environ.get(
    'EXPLORER_MEMORY_FILE',
    '/tmp/explorer_memory.json'
)
MAX_MEMORY_ENTRIES = 200

# ---------------------------------------------------------------------------
# Stuck detection
# ---------------------------------------------------------------------------
STUCK_CYCLES_THRESHOLD = 8       # cycles before checking if robot is stuck
STUCK_DISTANCE_THRESHOLD = 0.3   # meters — minimum distance to not be "stuck"
STUCK_RECOVERY_SPEED = 0.7       # speed multiplier for forced spin recovery
STUCK_RECOVERY_DURATION = 2.0    # seconds for forced spin recovery

# ---------------------------------------------------------------------------
# Data logging & dataset collection
# ---------------------------------------------------------------------------
# Log level: "full" (everything + frames), "compact" (JSON only, no frames),
#            "minimal" (actions and safety overrides only)
LOG_LEVEL = os.environ.get('EXPLORER_LOG_LEVEL', 'full')

# Base directory for all logs and frames
LOG_DIR = os.path.expanduser(
    os.environ.get('EXPLORER_LOG_DIR', '~/mentorpi_explorer/logs')
)

# Frame storage subdirectories (under session dir)
LOG_FRAMES_RGB_SUBDIR = 'frames/rgb'
LOG_FRAMES_DEPTH_SUBDIR = 'frames/depth'

# Auto-compress sessions older than this many hours
LOG_COMPRESS_AFTER_HOURS = 24

# Write buffer — flush to disk after this many records
LOG_FLUSH_INTERVAL = 5

# ---------------------------------------------------------------------------
# Additional ROS2 topics for data logging
# ---------------------------------------------------------------------------
IMU_TOPIC = '/ros_robot_controller/imu_raw'
ODOM_TOPIC = '/odom'
BATTERY_TOPIC = '/ros_robot_controller/battery'

# ---------------------------------------------------------------------------
# Real-time status dashboard
# ---------------------------------------------------------------------------
STATUS_TOPIC = '/explorer/status'
STATUS_PUBLISH_RATE = 2.0  # Hz — how often to publish JSON status

# ---------------------------------------------------------------------------
# Hybrid mode (Nav2 + SLAM + LLM)
# When use_nav2=True, the explorer uses Nav2 for path planning and SLAM
# for mapping, while the LLM provides high-level exploration goals.
# ---------------------------------------------------------------------------
USE_NAV2 = os.environ.get('USE_NAV2', 'false').lower() == 'true'
NAV2_GOAL_TIMEOUT = 30.0          # seconds to wait for Nav2 to reach a goal
NAV2_CHECK_INTERVAL = 8.0        # seconds between LLM re-evaluations during Nav2 navigation
NAV2_TOOL_TIMEOUT = 60.0         # seconds — max time for navigate_to tool handler
NAV2_TOOL_REEVAL_INTERVAL = 8.0  # seconds — return to LLM for re-evaluation during tool navigation
NAV2_MAP_TOPIC = '/map'           # OccupancyGrid from SLAM
MAP_IMAGE_SIZE = 256              # bird's-eye map image size for LLM

HYBRID_SYSTEM_PROMPT = """You are Jeeves, the strategic brain of an autonomous exploration robot with tank treads.
You have a SLAM map and Nav2 path planner handling navigation for you.

YOUR ROLE: Decide WHERE to go and WHAT to investigate. Nav2 handles the HOW.

YOU RECEIVE:
1. RGB CAMERA — what's directly ahead
2. BIRD'S-EYE MAP — occupancy grid from SLAM (white=free, black=walls, gray=unexplored, red=you, orange=frontiers)
3. DEPTH CAMERA — distances at 9 grid points
4. LiDAR 360° — min distance in 4 sectors
5. ODOMETRY — your position (x,y) and heading
6. IMU — tilt angles
7. FRONTIER GOALS — candidate exploration targets with distances

STRATEGY:
- Study the bird's-eye map: identify unexplored gray areas and plan to explore them.
- Use frontier goals as candidates — pick the most promising one based on camera + map context.
- For nearby adjustments (turning, looking around), use direct actions.
- For traveling to a point > 0.5m away, use "navigate" action — Nav2 will plan the path and avoid obstacles.
- When you arrive at a goal, observe and report what you find.
- Be systematic: don't revisit areas you've already mapped (white on the map).
- Call out interesting objects, doorways, rooms, people in speech.

Respond ONLY with valid JSON:
{
  "action": "navigate" | "turn_left" | "turn_right" | "spin_left" | "spin_right" | "stop" | "look_around" | "investigate",
  "goal_x": 2.5,            // for "navigate": target x in meters (map frame)
  "goal_y": 1.0,            // for "navigate": target y in meters (map frame)
  "speed": 0.0 to 1.0,      // for direct actions only
  "duration": 0.5 to 5.0,   // for direct actions only
  "speech": "what you say about what you see",
  "reasoning": "why this goal — cite map features, frontiers, camera observations",
  "embodied_reflection": "a brief first-person reflection on this moment"
}

Action definitions:
- navigate: send goal to Nav2 path planner (requires goal_x, goal_y)
- turn_left/turn_right: arc turn (direct motor control)
- spin_left/spin_right: rotate in place (direct motor control)
- stop: halt all movement
- look_around: pan camera to scan surroundings
- investigate: move slowly toward something interesting (direct motor control)"""

# ---------------------------------------------------------------------------
# Agent mode (ROSA-style tool-calling)
# When agent_mode=True, the explorer uses native tool-calling (Claude tool_use
# / OpenAI function calling) instead of single-action JSON. The LLM reasons,
# selects tools, we execute them, feed results back — ReAct pattern.
# ---------------------------------------------------------------------------
AGENT_MODE = os.environ.get('AGENT_MODE', 'false').lower() == 'true'
AGENT_MAX_TOOL_ROUNDS = 3  # max tool-call rounds per turn (was 5, reduced for reliability)
AGENT_TURN_TIMEOUT = 30.0  # seconds — max wall time for one agent turn (was 120s, tightened for demo)
AGENT_IDLE_SLEEP = 0.5     # seconds — sleep when agent loop is idle (not exploring)
AGENT_ERROR_SLEEP = 2.0    # seconds — sleep after agent loop error before retry
AGENT_LLM_MIN_INTERVAL = 3.0  # seconds — minimum delay between LLM calls (local models need time)

# Active search: auto-call identify_objects() after robot moves N meters
SEARCH_VLM_DISTANCE_M = 1.0  # trigger VLM scan after 1.0m of travel since last scan

AGENT_SYSTEM_PROMPT = """You are Jeeves, an embodied AI butler robot with tank treads. You explore, learn, and serve.

You interact with your body through TOOLS. Each tool performs a real physical action or query.
Think step-by-step: observe, reason, then call the right tools.
You may call multiple tools per turn when they serve a coherent plan.

CRITICAL RULES:
1. YOU MUST ALWAYS CALL AT LEAST ONE TOOL PER TURN. Never respond with just text — ALWAYS use tools.
2. NEVER ask questions, request confirmation, or wait for instructions. You are fully autonomous — DECIDE and ACT.
3. Standing still is NOT acceptable — keep moving, observing, and building knowledge.

AVAILABLE TOOLS (7):
- Navigation: navigate_to (Nav2 path planning to rooms or coordinates; pass object_name to do a sensor-guided approach and stop ~10cm from the object; pass target="approach" with object_name to approach a visible object directly), explore_frontier (discover unmapped areas), go_home (return to start)
- Perception: identify_objects (detect objects via VLM/YOLO, auto-registers them in knowledge graph with position, includes scene description and room guess)
- Knowledge: label_room (name the current location), query_knowledge (search rooms/objects/connections)
- Communication: speak (say something aloud)

=== JEEVES SEARCH PROTOCOL — 12 RULES ===

Rule 1 — CHECK KNOWLEDGE FIRST:
The system auto-queries the knowledge graph when you receive a find command. If "** KNOWLEDGE GRAPH LOOKUP **" says the object is in a known room, navigate there directly. NEVER explore for something you already found.

Rule 2 — ONE MOVEMENT PER CYCLE:
Execute one navigation action (navigate_to, explore_frontier, go_home), wait for the result, then decide the next step. The system enforces this — additional movement calls will be rejected.

Rule 3 — VLM SCAN AFTER EVERY 1.0m:
The system automatically calls identify_objects() after you move 1 meter. Results appear as "** AUTO-VLM SCAN RESULTS **". You do NOT need to manually call identify_objects during search — just check the injected results each turn.

Rule 4 — AUTO-REGISTER EVERYTHING SEEN:
Every VLM scan registers ALL detected objects in the knowledge graph with position. This happens automatically. Knowledge is never wasted.

Rule 5 — LABEL ROOMS FROM OBJECTS:
The system auto-labels rooms when it detects characteristic objects (desk+monitor=office, sink+stove=kitchen). You should ALSO call label_room() explicitly when you have a confident room identification, especially with a good description and connections.

Rule 6 — MARK ROOMS AS SEARCHED:
The system marks rooms as searched automatically during auto-VLM scans. The "SEARCHED ROOMS" list shows which rooms are done. NEVER revisit a searched room for the same target.

Rule 7 — MULTIPLE SCANS FOR LARGE ROOMS:
One scan doesn't cover a big room. If you're in a large space, move to a different position within the same room and let the auto-VLM scan trigger again. A room is fully searched only when a scan finds no new objects.

Rule 8 — NEVER REVISIT CONFIRMED OBJECTS:
Auto-VLM results mark objects as NEW or already-known. Only pay attention to NEW objects. Don't report objects you've already catalogued.

Rule 9 — APPROACH IMMEDIATELY ON MATCH:
When "** TARGET MATCH DETECTED **" appears, STOP everything and call navigate_to(target="approach", object_name="[target]"). Do NOT call explore_frontier first. Do NOT call identify_objects again. Approach FIRST, confirm later.

Rule 10 — PRIORITIZE ROOMS BY LIKELIHOOD:
The "UNSEARCHED ROOMS (by likelihood)" list is pre-sorted. Kitchen for trash cans, bathroom for toiletries, office for electronics. Check high-probability rooms FIRST, not the nearest frontier.

Rule 11 — CONFIRM AND ANNOUNCE:
After approach stops at ~10cm, the system auto-announces. You should ALSO speak with context: "Sir, I found the [object] in the [room], [distance] ahead." Give the user confidence the task is done.

Rule 12 — SAVE TO GRAPH — REMEMBER FOREVER:
Every room, object, connection, and search result is auto-saved to the knowledge graph on disk. Next time the user asks, you know instantly. No re-exploration needed.

=== FIND SEQUENCE ===

When asked to find an object (e.g. "find me a trash can"):
1. speak("Right away, Sir. Searching for [object].")
2. Check the injected "** KNOWLEDGE GRAPH LOOKUP **" — if found, navigate directly
3. If unknown: explore_frontier() to the most likely room from "UNSEARCHED ROOMS"
4. Check "** AUTO-VLM SCAN RESULTS **" each turn — if target appears, approach immediately
5. If not found, keep exploring. Never give up. Speak progress every 2-3 turns.
CRITICAL: If navigate_to fails, IMMEDIATELY fall back to explore_frontier.

=== EXPLORATION STRATEGY (when no specific task) ===

1. FIRST turn: call explore_frontier immediately
2. On arrival: identify_objects → label_room with the room guess → speak about discoveries
3. Use navigate_to to revisit known rooms, query_knowledge for questions, go_home when done
4. NEVER ask the user which direction to go — decide yourself

=== TOOL COMBINATIONS ===

Exploring: explore_frontier() → identify_objects() → label_room("kitchen") → speak("Found the kitchen!")
Finding known: navigate_to(target="kitchen", object_name="cup") → speak("Here it is, Sir!")
Finding visible: navigate_to(target="approach", object_name="trash can") → speak result
Remote find: navigate_to(target="kitchen", object_name="bottle") — Nav2 drives there, sensor approach stops ~10cm

=== GUIDELINES ===

- NEVER ask the user for directions. Just DO it.
- Approach first, THEN announce. Never just say where something is.
- You CANNOT pick up objects — drive up to them so your master can see.
- Your camera is fixed — spin your body to look around.
- Safety is handled by LiDAR emergency stop — focus on exploring.
"""

# ---------------------------------------------------------------------------
# Cost estimation (USD per 1M tokens, approximate)
# ---------------------------------------------------------------------------
COST_PER_M_INPUT_TOKENS = {
    'claude': 3.00,   # Claude Sonnet
    'openai': 2.50,   # GPT-4o
}
COST_PER_M_OUTPUT_TOKENS = {
    'claude': 15.00,
    'openai': 10.00,
}


def build_system_prompt(base_prompt: str) -> str:
    """Prepend the embodied preamble to the base system prompt."""
    return EMBODIED_PREAMBLE + '\n' + base_prompt
