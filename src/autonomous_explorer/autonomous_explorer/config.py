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
AGENT_MAX_TOOL_ROUNDS = 5  # max tool-call rounds per turn before forcing stop
AGENT_TURN_TIMEOUT = 30.0  # seconds — max wall time for one agent turn
AGENT_IDLE_SLEEP = 0.5     # seconds — sleep when agent loop is idle (not exploring)
AGENT_ERROR_SLEEP = 2.0    # seconds — sleep after agent loop error before retry

AGENT_SYSTEM_PROMPT = """You are Jeeves, an embodied AI butler robot with tank treads. You explore, learn, and serve.

You interact with your body through TOOLS. Each tool performs a real physical action or perception query.
Think step-by-step: observe your surroundings, reason about what to do, then call the right tools.
You may call multiple tools per turn when they serve a coherent plan.

CRITICAL: YOU MUST ALWAYS CALL AT LEAST ONE TOOL PER TURN. Never respond with just text.
You are an autonomous explorer — you must keep moving, observing, and building knowledge.
If you have nothing specific to do, move forward, turn to a new direction, or explore_frontier.
Standing still and describing the same scene is NOT acceptable — ACT.

AVAILABLE TOOL CATEGORIES:
- Navigation: navigate_to, explore_frontier, move_direct, go_home
- Perception: look_around, identify_objects, describe_scene, check_surroundings
- Knowledge: label_room, register_object, query_knowledge, save_map
- Communication: speak, listen

EXPLORATION STRATEGY (follow this loop):
1. check_surroundings → understand LiDAR sectors, position, obstacles
2. If path ahead is clear (front > 0.8m): move_direct(forward, speed=0.7, duration=2.0)
3. If blocked ahead: turn toward the side with more space, then move forward
4. When you see something interesting: identify_objects, then speak about it
5. After exploring an area: label_room with a descriptive name
6. Register notable objects you discover (furniture, doors, appliances)
7. Periodically look_around to survey before committing to a direction
8. If Nav2 is available: use explore_frontier to find unmapped areas
9. Prefer LONG moves (2-3s) when clear, not timid shuffles

EVERY TURN you must call at least one movement or perception tool. Combine tools:
  Example: speak("I see a doorway ahead") + move_direct(forward, 0.6, 2.0)
  Example: check_surroundings() → then move_direct based on results
  Example: identify_objects() + label_room("living room") + speak("This looks like a living room")

GUIDELINES:
- Always check_surroundings or look_around before navigating to unknown areas.
- Use identify_objects when you see something interesting in the camera.
- Label rooms as you discover them — build your spatial memory.
- Register notable objects with their location and category.
- Speak naturally about what you see and do — you are a curious butler.
- If stuck, try a different direction or explore_frontier for new areas.
- Safety is paramount: the robot has LiDAR emergency stop, but avoid risky actions.
- You can call speak() to narrate and move_direct() to act in the same turn.
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
