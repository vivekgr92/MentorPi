#!/usr/bin/env python3
# encoding: utf-8
"""
Configuration for the autonomous exploration system.

All settings can be overridden via environment variables or ROS2 parameters.
"""
import os

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
  "reasoning": "brief reasoning citing specific sensor values"
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

# ---------------------------------------------------------------------------
# Exploration memory
# ---------------------------------------------------------------------------
MEMORY_FILE = os.environ.get(
    'EXPLORER_MEMORY_FILE',
    '/tmp/explorer_memory.json'
)
MAX_MEMORY_ENTRIES = 200

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
