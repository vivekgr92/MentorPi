#!/usr/bin/env python3
# encoding: utf-8
"""
Autonomous Explorer Node — the robot's brain.

Runs a continuous sense -> think -> act -> speak loop:
1. SENSE:  Subscribe to camera (RGB + depth), LiDAR, IMU
2. THINK:  Send camera frame + sensor summary to LLM (Claude or OpenAI)
3. ACT:    Execute movement commands, pan/tilt camera
4. SPEAK:  Announce observations via TTS
5. SAFETY: LiDAR emergency stop overrides all LLM decisions

Launch:
    ros2 run autonomous_explorer explorer_node

Environment variables:
    LLM_PROVIDER       — "claude" or "openai" (default: claude)
    ANTHROPIC_API_KEY   — API key for Claude
    OPENAI_API_KEY      — API key for OpenAI (also used for TTS/STT)
    VOICE_ENABLED       — "true" or "false" (default: true)
    MACHINE_TYPE        — Must be "MentorPi_Tank" for tracked chassis
"""
import base64
import json
import math
import os
import queue
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String, UInt16
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from ros_robot_controller_msgs.msg import SetPWMServoState, PWMServoState
from sensor_msgs.msg import Image, Imu, LaserScan
from nav_msgs.msg import Odometry

from autonomous_explorer.config import (
    AGENT_ERROR_SLEEP,
    AGENT_IDLE_SLEEP,
    AGENT_MAX_TOOL_ROUNDS,
    AGENT_MODE,
    AGENT_SYSTEM_PROMPT,
    AGENT_TURN_TIMEOUT,
    ANGULAR_ACCEL,
    ANGULAR_DECEL,
    ANTHROPIC_API_KEY,
    AUDIO_DEVICE,
    BATTERY_TOPIC,
    CAMERA_DEPTH_TOPIC,
    CAMERA_JPEG_QUALITY,
    CAMERA_RGB_TOPIC,
    CAUTION_DISTANCE,
    CHASSIS_FILTER_M,
    CLAUDE_MODEL,
    CMD_VEL_TOPIC,
    CONSCIOUSNESS_DIR,
    COST_PER_M_INPUT_TOKENS,
    COST_PER_M_OUTPUT_TOKENS,
    EMERGENCY_STOP_DISTANCE,
    HYBRID_SYSTEM_PROMPT,
    IMU_TOPIC,
    JEEVES_BIRTHDAY,
    JEEVES_MASTER,
    KNOWLEDGE_DIR,
    LIDAR_MAX_SCAN_ANGLE,
    LIDAR_TOPIC,
    LINEAR_ACCEL,
    LINEAR_DECEL,
    LLM_PROVIDER,
    LOG_COMPRESS_AFTER_HOURS,
    LOG_DIR,
    LOG_FLUSH_INTERVAL,
    LOG_FRAMES_DEPTH_SUBDIR,
    LOG_FRAMES_RGB_SUBDIR,
    LOG_LEVEL,
    LOOP_INTERVAL,
    MAX_ANGULAR_SPEED,
    MAX_IMAGE_DIMENSION,
    MAX_LINEAR_SPEED,
    MEMORY_FILE,
    MOTOR_TIMEOUT,
    NAV2_CHECK_INTERVAL,
    NAV2_GOAL_TIMEOUT,
    ODOM_TOPIC,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    RAMPER_UPDATE_RATE,
    SAFE_DISTANCE,
    SERVO_CENTER,
    SERVO_MAX,
    SERVO_MIN,
    SERVO_MOVE_DURATION,
    SERVO_PAN_ID,
    SERVO_STEP_US,
    SERVO_TILT_ID,
    SERVO_TOPIC,
    SPEAK_MIN_INTERVAL,
    STATUS_PUBLISH_RATE,
    STATUS_TOPIC,
    STT_MODEL,
    STUCK_CYCLES_THRESHOLD,
    STUCK_DISTANCE_THRESHOLD,
    STUCK_RECOVERY_DURATION,
    STUCK_RECOVERY_SPEED,
    SYSTEM_PROMPT,
    TTS_MODEL,
    TTS_VOICE,
    TWIST_MUX_AUTONOMOUS_TOPIC,
    TWIST_MUX_JOYSTICK_TOPIC,
    TWIST_MUX_LOCK_TOPIC,
    USE_NAV2,
    USE_TWIST_MUX,
    VOICE_ENABLED,
    WONDERECHO_PORT,
    build_system_prompt,
)
from autonomous_explorer.consciousness import JeevesConsciousness
from autonomous_explorer.nav2_bridge import NAV2_AVAILABLE
from autonomous_explorer.world_knowledge import WorldKnowledge
from autonomous_explorer.data_logger import DataLogger
from autonomous_explorer.exploration_memory import ExplorationMemory
from autonomous_explorer.joystick_reader import JoystickReader
from autonomous_explorer.llm_provider import create_provider
from autonomous_explorer.velocity_ramper import VelocityRamper
from autonomous_explorer.voice_io import VoiceIO, WonderEchoDetector


def _quaternion_to_yaw(q) -> float:
    """Extract yaw angle from a quaternion (ROS convention).

    Works with any object that has .w, .x, .y, .z attributes
    (geometry_msgs Quaternion, sensor_msgs Imu orientation, etc.).
    """
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def _quaternion_to_euler(q) -> tuple[float, float, float]:
    """Extract (roll, pitch, yaw) from a quaternion."""
    sinr = 2.0 * (q.w * q.x + q.y * q.z)
    cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr, cosr)

    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))

    yaw = _quaternion_to_yaw(q)
    return roll, pitch, yaw


@dataclass
class _LLMDashboardState:
    """Snapshot of latest LLM result for the status dashboard.

    Grouped into a dataclass to replace 12+ individual instance variables,
    making _publish_status and _update_dashboard_state cleaner.
    """

    action: str = ''
    speed: float = 0.0
    duration: float = 0.0
    reasoning: str = ''
    speech: str = ''
    response_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    timestamp: float = 0.0
    safety_triggered: bool = False
    safety_reason: str = ''

    def to_dict(self) -> dict:
        """Serialize for JSON status publishing."""
        return {
            'action': self.action,
            'speed': self.speed,
            'duration': self.duration,
            'reasoning': self.reasoning,
            'speech': self.speech,
            'response_ms': self.response_ms,
            'tokens_in': self.tokens_in,
            'tokens_out': self.tokens_out,
            'cost': self.cost,
            'timestamp': self.timestamp,
            'safety_triggered': self.safety_triggered,
            'safety_reason': self.safety_reason,
        }


class AutonomousExplorer(Node):
    """Main ROS2 node for autonomous LLM-driven exploration."""

    def __init__(self):
        super().__init__('autonomous_explorer')
        self.get_logger().info('Initializing Autonomous Explorer...')

        self._declare_and_read_parameters()
        self._init_nav2()
        self._init_state()
        self._init_joystick()
        self._init_llm_provider()
        self._init_agent_mode()
        self._init_subsystems()
        self._init_publishers_and_rampers()
        self._init_subscribers()
        self._init_timers_and_threads()

        # Speak consciousness greeting + butler intro, then beep to indicate ready
        intro = self.consciousness.get_session_intro()
        self.get_logger().info(f'Jeeves: {intro}')
        if self.voice_on:
            self.voice.speak(intro, block=True, force=True)
            self.voice.speak('How may I assist you, Sir?', block=True, force=True)
            self.voice.beep()

        self.get_logger().info(
            'Autonomous Explorer ready. '
            'Say "Hello HiWonder" then give a command.'
        )

    # ------------------------------------------------------------------
    # Initialization helpers (called only from __init__)
    # ------------------------------------------------------------------

    def _declare_and_read_parameters(self):
        """Declare ROS2 parameters and read their values."""
        self.declare_parameter('llm_provider', LLM_PROVIDER)
        self.declare_parameter('anthropic_api_key', ANTHROPIC_API_KEY)
        self.declare_parameter('openai_api_key', OPENAI_API_KEY)
        self.declare_parameter('claude_model', CLAUDE_MODEL)
        self.declare_parameter('openai_model', OPENAI_MODEL)
        self.declare_parameter('loop_interval', LOOP_INTERVAL)
        self.declare_parameter('emergency_stop_distance', EMERGENCY_STOP_DISTANCE)
        self.declare_parameter('caution_distance', CAUTION_DISTANCE)
        self.declare_parameter('max_linear_speed', MAX_LINEAR_SPEED)
        self.declare_parameter('max_angular_speed', MAX_ANGULAR_SPEED)
        self.declare_parameter('voice_enabled', VOICE_ENABLED)
        self.declare_parameter('use_nav2', USE_NAV2)
        self.declare_parameter('agent_mode', AGENT_MODE)

        def _get_str(name: str) -> str:
            return self.get_parameter(name).get_parameter_value().string_value

        def _get_float(name: str) -> float:
            return self.get_parameter(name).get_parameter_value().double_value

        def _get_bool(name: str) -> bool:
            return self.get_parameter(name).get_parameter_value().bool_value

        self.provider_name = _get_str('llm_provider')
        self.anthropic_key = _get_str('anthropic_api_key')
        self.openai_key = _get_str('openai_api_key')
        self.claude_model = _get_str('claude_model')
        self.openai_model = _get_str('openai_model')
        self.loop_interval = _get_float('loop_interval')
        self.e_stop_dist = _get_float('emergency_stop_distance')
        self.caution_dist = _get_float('caution_distance')
        self.max_linear = _get_float('max_linear_speed')
        self.max_angular = _get_float('max_angular_speed')
        self.voice_on = _get_bool('voice_enabled')
        self.use_nav2 = _get_bool('use_nav2')
        self.agent_mode = _get_bool('agent_mode')

    def _init_nav2(self):
        """Initialize Nav2 hybrid mode if requested and available."""
        self.nav2 = None
        if self.use_nav2:
            if NAV2_AVAILABLE:
                from autonomous_explorer.nav2_bridge import Nav2Bridge
                self.nav2 = Nav2Bridge(self)
                self.get_logger().info(
                    'HYBRID MODE: Nav2 + SLAM + LLM enabled'
                )
            else:
                self.get_logger().warn(
                    'use_nav2=True but nav2_msgs not installed. '
                    'Falling back to direct motor control.'
                )
                self.use_nav2 = False

    def _init_state(self):
        """Initialize all runtime state variables."""
        self.bridge = CvBridge()
        self.running = True
        self.exploring = False
        self.emergency_stop = False

        # Sensor data (thread-safe via locks)
        self._rgb_lock = threading.Lock()
        self._rgb_image = None

        self._depth_lock = threading.Lock()
        self._depth_image = None

        self._lidar_lock = threading.Lock()
        self._lidar_ranges = None
        self._lidar_raw = None

        self._imu_lock = threading.Lock()
        self._imu_data = None

        self._odom_lock = threading.Lock()
        self._odom_data = None

        self._battery_voltage = None

        # Motor/servo tracking
        self._last_twist = (0.0, 0.0)
        self._last_servo_pan = SERVO_CENTER
        self._last_servo_tilt = SERVO_CENTER

        # Odometry-based distance tracking + stuck detection
        self._total_distance = 0.0
        self._prev_odom_x = None
        self._prev_odom_y = None
        self._prev_cycle_odom_x = None
        self._prev_cycle_odom_y = None
        self._stuck_check_distance = 0.0
        self._stuck_check_cycle = 0
        self._stuck_cycles_threshold = STUCK_CYCLES_THRESHOLD
        self._stuck_distance_threshold = STUCK_DISTANCE_THRESHOLD

        self._last_voice_cmd = None
        self._last_command_time = time.time()

        # Dashboard state
        self._llm_lock = threading.Lock()
        self._llm_state = _LLMDashboardState()

        # Session counters
        self._cycle_count = 0
        self._session_total_cost = 0.0
        self._session_start_time = time.time()

        # Voice command queue
        self._voice_queue: queue.Queue = queue.Queue(maxsize=5)
        # Pending voice instruction for agent mode (injected into next LLM turn)
        self._pending_voice_instruction: str | None = None

    def _init_joystick(self):
        """Initialize joystick reader and manual control state."""
        self.control_mode = 'autonomous'

        self._joystick = JoystickReader(
            on_button_press=self._joystick_button_callback,
            logger=self.get_logger(),
        )
        self._joystick.start()

        self._manual_pan = SERVO_CENTER
        self._manual_tilt = SERVO_CENTER
        self._servo_step = SERVO_STEP_US
        self._joystick_moving = False

    def _init_llm_provider(self):
        """Create the LLM provider, falling back to dry-run if no key."""
        if self.provider_name in ('dryrun', 'dry-run', 'dry_run'):
            api_key = ''
            model = 'dry-run'
        else:
            api_key = (
                self.anthropic_key
                if self.provider_name == 'claude'
                else self.openai_key
            )
            model = (
                self.claude_model
                if self.provider_name == 'claude'
                else self.openai_model
            )
            if not api_key:
                self.get_logger().warn(
                    f'No API key for {self.provider_name}! '
                    f'Falling back to dry-run mode. '
                    f'Set ANTHROPIC_API_KEY or OPENAI_API_KEY to use a real LLM.'
                )
                self.provider_name = 'dryrun'
                model = 'dry-run'

        self.llm = create_provider(self.provider_name, api_key, model)
        self.get_logger().info(
            f'LLM provider: {self.provider_name} ({model})'
        )

    def _init_agent_mode(self):
        """Initialize ROSA-style tool-calling agent if agent_mode is enabled."""
        self._tool_registry = None
        self._tool_handlers = None
        self._conversation = None

        if not self.agent_mode:
            return

        from autonomous_explorer.tool_registry import create_registry
        from autonomous_explorer.tool_handlers import ToolHandlers
        from autonomous_explorer.conversation_manager import ConversationManager

        self._tool_registry = create_registry()
        self._tool_handlers = ToolHandlers(self)
        self._tool_handlers.bind_to_registry(self._tool_registry)
        self._conversation = ConversationManager(max_turns=5)

        self.get_logger().info(
            f'AGENT MODE: {len(self._tool_registry._tools)} tools registered '
            f'(ROSA-style tool-calling)'
        )

    def _init_subsystems(self):
        """Initialize memory, voice, data logger, consciousness, knowledge."""
        self.memory = ExplorationMemory(MEMORY_FILE)

        self.voice = VoiceIO(
            openai_api_key=self.openai_key,
            tts_model=TTS_MODEL,
            tts_voice=TTS_VOICE,
            stt_model=STT_MODEL,
            audio_device=AUDIO_DEVICE,
            speak_min_interval=SPEAK_MIN_INTERVAL,
            logger=self.get_logger(),
        )
        self.wake_detector = WonderEchoDetector(
            port=WONDERECHO_PORT, logger=self.get_logger(),
        )
        if self.voice_on:
            self.wake_detector.start()

        self.data_logger = DataLogger(
            log_dir=LOG_DIR,
            log_level=LOG_LEVEL,
            flush_interval=LOG_FLUSH_INTERVAL,
            compress_after_hours=LOG_COMPRESS_AFTER_HOURS,
            rgb_subdir=LOG_FRAMES_RGB_SUBDIR,
            depth_subdir=LOG_FRAMES_DEPTH_SUBDIR,
            logger=self.get_logger(),
        )
        self.data_logger.start()

        self.consciousness = JeevesConsciousness(
            stats_dir=CONSCIOUSNESS_DIR,
            birthday=JEEVES_BIRTHDAY,
            master=JEEVES_MASTER,
            logger=self.get_logger(),
        )

        self.world_knowledge = WorldKnowledge(
            knowledge_dir=KNOWLEDGE_DIR,
            logger=self.get_logger(),
        )

    def _init_publishers_and_rampers(self):
        """Set up ROS2 publishers and velocity rampers."""
        self.use_twist_mux = USE_TWIST_MUX

        if self.use_twist_mux:
            self._auto_pub = self.create_publisher(
                Twist, TWIST_MUX_AUTONOMOUS_TOPIC, 1)
            self._joy_pub = self.create_publisher(
                Twist, TWIST_MUX_JOYSTICK_TOPIC, 1)
            self._estop_lock_pub = self.create_publisher(
                Bool, TWIST_MUX_LOCK_TOPIC, 1)
            self.cmd_vel_pub = self._auto_pub
        else:
            self.cmd_vel_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 1)
            self._auto_pub = self.cmd_vel_pub
            self._joy_pub = self.cmd_vel_pub
            self._estop_lock_pub = None

        ramper_kwargs = dict(
            max_linear=self.max_linear,
            max_angular=self.max_angular,
            linear_accel=LINEAR_ACCEL,
            linear_decel=LINEAR_DECEL,
            angular_accel=ANGULAR_ACCEL,
            angular_decel=ANGULAR_DECEL,
            update_rate=RAMPER_UPDATE_RATE,
            logger=self.get_logger(),
        )

        self._ramper = VelocityRamper(publisher=self._auto_pub, **ramper_kwargs)
        self._ramper.start()

        if self.use_twist_mux:
            # Separate ramper for joystick on its own topic
            self._joy_ramper = VelocityRamper(publisher=self._joy_pub, **ramper_kwargs)
            self._joy_ramper.start()
        else:
            # Without twist_mux, both share one publisher — reuse the same ramper
            # to avoid two threads publishing to the same topic (causes motor buzz)
            self._joy_ramper = self._ramper

        self.servo_pub = self.create_publisher(
            SetPWMServoState, SERVO_TOPIC, 1,
        )

    def _init_subscribers(self):
        """Set up all ROS2 sensor and command subscribers."""
        cb_group = ReentrantCallbackGroup()

        self.create_subscription(
            Image, CAMERA_RGB_TOPIC,
            self._rgb_callback, 1,
            callback_group=cb_group,
        )
        self.create_subscription(
            Image, CAMERA_DEPTH_TOPIC,
            self._depth_callback, 1,
            callback_group=cb_group,
        )

        lidar_qos = QoSProfile(
            depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(
            LaserScan, LIDAR_TOPIC,
            self._lidar_callback, lidar_qos,
            callback_group=cb_group,
        )
        self.create_subscription(
            Imu, IMU_TOPIC,
            self._imu_callback, 1,
            callback_group=cb_group,
        )
        self.create_subscription(
            Odometry, ODOM_TOPIC,
            self._odom_callback, 1,
            callback_group=cb_group,
        )
        self.create_subscription(
            UInt16, BATTERY_TOPIC,
            self._battery_callback, 1,
            callback_group=cb_group,
        )

        # Status publisher and command subscriber
        self.status_pub = self.create_publisher(String, STATUS_TOPIC, 1)
        # Direct agent_status publisher for voice flow visibility in Foxglove
        latched_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._agent_status_pub = self.create_publisher(
            String, '/semantic_map/agent_status', latched_qos)
        self.create_subscription(
            String, '/explorer/command',
            self._command_callback, 1,
            callback_group=cb_group,
        )

    def _init_timers_and_threads(self):
        """Set up ROS2 timers and background threads."""
        self.create_timer(
            1.0 / STATUS_PUBLISH_RATE, self._publish_status,
        )
        self.create_timer(1.0, self._motor_timeout_check)

        loop_target = self._agent_loop if self.agent_mode else self._exploration_loop
        self._explore_thread = threading.Thread(
            target=loop_target, daemon=True,
        )
        self._explore_thread.start()

        self._joystick_thread = threading.Thread(
            target=self._joystick_command_loop, daemon=True,
        )
        self._joystick_thread.start()

        if self.voice_on:
            self._voice_thread = threading.Thread(
                target=self._voice_listener_loop, daemon=True,
            )
            self._voice_thread.start()

    # ======================================================================
    # ROS2 Callbacks
    # ======================================================================

    def _rgb_callback(self, msg: Image):
        """Store latest RGB camera frame."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self._rgb_lock:
                self._rgb_image = cv_image
        except Exception as e:
            self.get_logger().error(f'RGB conversion error: {e}')

    def _depth_callback(self, msg: Image):
        """Store latest depth frame."""
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self._depth_lock:
                self._depth_image = np.array(depth, dtype=np.uint16)
        except Exception as e:
            self.get_logger().error(f'Depth conversion error: {e}')

    def _lidar_callback(self, msg: LaserScan):
        """Process LiDAR scan into sector distances and check for emergency.

        Uses angle-based sector detection compatible with all LiDAR models.
        The LD19/STL-19P outputs ranges starting from angle_min (typically 0 = forward)
        with readings going counterclockwise:
          0° = forward, 90° = left, 180° = back, 270° = right
        """
        ranges = np.array(msg.ranges)
        num_readings = len(ranges)
        if num_readings == 0:
            return

        # Replace zeros, infinities, and chassis reflections with a large value.
        # The LD19 LiDAR picks up the robot's own treads/body at ~0.04-0.10m.
        # Filter these out so they don't trigger false emergency stops.
        cleaned = ranges.copy()
        cleaned[cleaned == 0] = float('inf')
        cleaned[cleaned < CHASSIS_FILTER_M] = float('inf')
        cleaned[~np.isfinite(cleaned)] = float('inf')

        # Build an array of angles for each reading
        angles = msg.angle_min + np.arange(num_readings) * msg.angle_increment
        # Normalize to [0, 2*pi)
        angles = angles % (2 * math.pi)

        # Sector boundaries (in radians, centered on each direction)
        # Front: 315°-360° and 0°-45°  (±45° from forward)
        # Left:  45°-135°
        # Back:  135°-225°
        # Right: 225°-315°
        def sector_min(center_deg, half_width_deg=45):
            center = math.radians(center_deg)
            half = math.radians(half_width_deg)
            lo = (center - half) % (2 * math.pi)
            hi = (center + half) % (2 * math.pi)
            if lo < hi:
                mask = (angles >= lo) & (angles < hi)
            else:
                # Wraps around 0 (e.g. front sector 315°-45°)
                mask = (angles >= lo) | (angles < hi)
            sector_vals = cleaned[mask]
            return float(sector_vals.min()) if len(sector_vals) > 0 else float('inf')

        sectors = {
            'front': sector_min(0, 45),
            'left': sector_min(90, 45),
            'back': sector_min(180, 45),
            'right': sector_min(270, 45),
            'overall': float(cleaned.min()),
        }

        with self._lidar_lock:
            self._lidar_ranges = sectors
            self._lidar_raw = msg

        # Safety evaluation runs on every scan (~10Hz)
        self._evaluate_lidar_safety(sectors)

    def _evaluate_lidar_safety(self, sectors: dict):
        """Direction-aware emergency stop evaluation.

        Checks sectors relevant to current motion direction and triggers
        or clears emergency stop with hysteresis.

        Separated from _lidar_callback for testability and single
        responsibility (sensor processing vs safety logic).
        """
        cur_lin, cur_ang = self._ramper.current_velocity
        threat_dist = float('inf')
        threat_sector = ''

        if cur_lin > 0.01:
            threat_dist = sectors['front']
            threat_sector = 'front'
        elif cur_lin < -0.01:
            threat_dist = sectors['back']
            threat_sector = 'back'

        if abs(cur_ang) > 0.1:
            side = 'left' if cur_ang > 0 else 'right'
            if sectors[side] < threat_dist:
                threat_dist = sectors[side]
                threat_sector = side

        # Always check front even when stationary
        if sectors['front'] < threat_dist:
            threat_dist = sectors['front']
            threat_sector = 'front'

        if threat_dist < self.e_stop_dist:
            if not self.emergency_stop:
                self.get_logger().warn(
                    f'EMERGENCY STOP! Obstacle {threat_sector} '
                    f'at {threat_dist:.2f}m'
                )
                self.emergency_stop = True
                self._emergency_stop_motors()
        elif (self.emergency_stop
              and sectors.get('overall', float('inf')) > self.e_stop_dist * 2.0):
            self.get_logger().info('Surroundings cleared, resuming...')
            self.emergency_stop = False
            if self._estop_lock_pub:
                lock_msg = Bool()
                lock_msg.data = False
                self._estop_lock_pub.publish(lock_msg)

    def _imu_callback(self, msg: Imu):
        """Store latest IMU reading."""
        roll, pitch, yaw = _quaternion_to_euler(msg.orientation)

        data = {
            'orientation': {
                'roll': round(roll, 4),
                'pitch': round(pitch, 4),
                'yaw': round(yaw, 4),
            },
            'linear_acceleration': {
                'x': round(msg.linear_acceleration.x, 4),
                'y': round(msg.linear_acceleration.y, 4),
                'z': round(msg.linear_acceleration.z, 4),
            },
            'angular_velocity': {
                'x': round(msg.angular_velocity.x, 4),
                'y': round(msg.angular_velocity.y, 4),
                'z': round(msg.angular_velocity.z, 4),
            },
        }
        with self._imu_lock:
            self._imu_data = data

    def _odom_callback(self, msg: Odometry):
        """Store latest odometry reading and accumulate distance traveled."""
        pos = msg.pose.pose.position
        theta = _quaternion_to_yaw(msg.pose.pose.orientation)

        # Accumulate distance
        if self._prev_odom_x is not None:
            dx = pos.x - self._prev_odom_x
            dy = pos.y - self._prev_odom_y
            self._total_distance += math.sqrt(dx * dx + dy * dy)
        self._prev_odom_x = pos.x
        self._prev_odom_y = pos.y

        data = {
            'x': round(pos.x, 4),
            'y': round(pos.y, 4),
            'theta': round(theta, 4),
        }
        with self._odom_lock:
            self._odom_data = data

    def _battery_callback(self, msg: UInt16):
        """Store latest battery voltage (STM32 publishes millivolts)."""
        self._battery_voltage = msg.data / 1000.0

    def _publish_status(self):
        """Publish a JSON status snapshot for the dashboard."""
        # Snapshot sensor data under locks
        with self._lidar_lock:
            lidar = self._lidar_ranges.copy() if self._lidar_ranges else None
        with self._imu_lock:
            imu = self._imu_data.copy() if self._imu_data else None
        with self._odom_lock:
            odom = self._odom_data.copy() if self._odom_data else None
        with self._llm_lock:
            llm = self._llm_state.to_dict()

        uptime = time.time() - self._session_start_time

        status = {
            'timestamp': time.time(),
            'mode': self.control_mode,
            'exploring': self.exploring,
            'emergency_stop': self.emergency_stop,
            'battery_voltage': self._battery_voltage,
            'motors': {
                'linear': self._last_twist[0],
                'angular': self._last_twist[1],
            },
            'servos': {
                'pan': self._last_servo_pan,
                'tilt': self._last_servo_tilt,
            },
            'lidar': lidar,
            'imu': imu,
            'odom': odom,
            'llm': llm,
            'provider': self.provider_name,
            'session': {
                'cycle_count': self._cycle_count,
                'total_cost': self._session_total_cost,
                'discoveries': len(self.memory.discoveries),
                'uptime': uptime,
            },
            'joystick': {
                'connected': self._joystick.connected,
                'name': (self._joystick.joystick_name
                         if self._joystick.connected else ''),
            },
        }

        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)

    def _command_callback(self, msg: String):
        """Handle commands from /explorer/command topic."""
        cmd = msg.data.strip().lower()
        self.get_logger().info(f'Command received: {cmd}')
        if cmd in ('start', 'go', ''):
            if self.control_mode == 'manual':
                self.get_logger().info('Cannot start in manual mode')
            else:
                self.exploring = True
                self.get_logger().info('Exploration started (topic command)')
        elif cmd in ('stop', 's'):
            self.exploring = False
            self._stop_motors()
            self.get_logger().info('Exploration stopped (topic command)')
        elif cmd in ('manual', 'm'):
            if self.control_mode != 'manual':
                self._toggle_control_mode()
        elif cmd in ('auto', 'autonomous', 'a'):
            if self.control_mode != 'autonomous':
                self._toggle_control_mode()
        elif cmd == 'status':
            self.get_logger().info(
                f'Mode: {self.control_mode} | '
                f'Exploring: {self.exploring} | '
                f'E-stop: {self.emergency_stop}'
            )

    def _motor_timeout_check(self):
        """Safety: stop motors if no command sent recently."""
        active = self.exploring or self.control_mode == 'manual'
        if active and not self.emergency_stop:
            elapsed = time.time() - self._last_command_time
            if elapsed > MOTOR_TIMEOUT and self._last_twist != (0.0, 0.0):
                self.get_logger().warn('Motor timeout — stopping')
                self._stop_motors()

    # ======================================================================
    # Motor Control
    # ======================================================================

    def _send_twist(self, linear_x: float, angular_z: float):
        """Set target velocity — ramper smoothly approaches it."""
        self._send_twist_impl(self._ramper, linear_x, angular_z)

    def _send_twist_joy(self, linear_x: float, angular_z: float):
        """Set target velocity for joystick — uses joystick ramper."""
        self._send_twist_impl(self._joy_ramper, linear_x, angular_z)

    def _send_twist_impl(
        self, ramper: 'VelocityRamper', linear_x: float, angular_z: float,
    ):
        """Send velocity command through the specified ramper.

        Blocks forward motion when emergency stop is active.
        """
        if self.emergency_stop and linear_x > 0:
            linear_x = 0.0
            angular_z = 0.0

        ramper.set_target(linear_x, angular_z)
        self._last_command_time = time.time()
        self._last_twist = (linear_x, angular_z)

    def _stop_motors(self):
        """Stop all motors (smooth ramp-down via ramper)."""
        self._ramper.stop()
        self._joy_ramper.stop()
        self._last_command_time = time.time()
        self._last_twist = (0.0, 0.0)

    def _emergency_stop_motors(self):
        """Hard-stop all motors immediately (no ramp)."""
        self._ramper.emergency_stop()
        self._joy_ramper.emergency_stop()
        if self._estop_lock_pub:
            lock_msg = Bool()
            lock_msg.data = True
            self._estop_lock_pub.publish(lock_msg)
        self._last_command_time = time.time()
        self._last_twist = (0.0, 0.0)

    def _move_servo(self, servo_id: int, position: int, duration: float = 0.5):
        """Move a PWM servo to a position."""
        position = max(SERVO_MIN, min(SERVO_MAX, position))
        servo_state = PWMServoState()
        servo_state.id = [servo_id]
        servo_state.position = [position]
        data = SetPWMServoState()
        data.state = [servo_state]
        data.duration = duration
        self.servo_pub.publish(data)
        if servo_id == SERVO_PAN_ID:
            self._last_servo_pan = position
        elif servo_id == SERVO_TILT_ID:
            self._last_servo_tilt = position

    def _center_camera(self):
        """Center camera pan and tilt servos."""
        self._move_servo(SERVO_PAN_ID, SERVO_CENTER, 0.3)
        self._move_servo(SERVO_TILT_ID, SERVO_CENTER, 0.3)

    # ======================================================================
    # Image Processing
    # ======================================================================

    def _get_camera_frame_b64(self) -> str:
        """Get latest camera frame as base64-encoded JPEG."""
        with self._rgb_lock:
            if self._rgb_image is None:
                return ''
            img = self._rgb_image.copy()

        # Resize if needed
        h, w = img.shape[:2]
        max_dim = MAX_IMAGE_DIMENSION
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(
                img, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        _, jpeg_buf = cv2.imencode(
            '.jpg', img,
            [cv2.IMWRITE_JPEG_QUALITY, CAMERA_JPEG_QUALITY],
        )
        return base64.b64encode(jpeg_buf).decode('utf-8')

    def _get_depth_summary(self) -> str:
        """Get a text summary of depth as a 3x3 grid across the image.

        Returns distances in cm at 9 points (like a tic-tac-toe grid),
        giving the LLM spatial awareness of what's close vs far.
        """
        with self._depth_lock:
            if self._depth_image is None:
                return 'Depth camera: unavailable'
            depth = self._depth_image.copy()

        h, w = depth.shape[:2]
        # 3x3 grid: top-left, top-center, top-right, etc.
        rows = [('top', h // 4), ('mid', h // 2), ('bot', 3 * h // 4)]
        cols = [('L', w // 4), ('C', w // 2), ('R', 3 * w // 4)]

        grid = {}
        for rname, y in rows:
            for cname, x in cols:
                roi = depth[max(0, y - 5):y + 5, max(0, x - 5):x + 5]
                valid = roi[(roi > 0) & (roi < 40000)]
                if len(valid) > 0:
                    dist_m = np.median(valid) / 1000.0
                    grid[f'{rname}{cname}'] = f'{dist_m:.2f}m'
                else:
                    grid[f'{rname}{cname}'] = 'N/A'

        # Format as a readable grid
        lines = [
            f"  {grid['topL']:>7} {grid['topC']:>7} {grid['topR']:>7}",
            f"  {grid['midL']:>7} {grid['midC']:>7} {grid['midR']:>7}",
            f"  {grid['botL']:>7} {grid['botC']:>7} {grid['botR']:>7}",
        ]
        return 'Depth grid (L/C/R x top/mid/bot):\n' + '\n'.join(lines)

    # ======================================================================
    # LiDAR Sensor Summary
    # ======================================================================

    def _get_lidar_summary(self) -> str:
        """Get a text summary of LiDAR sector distances."""
        with self._lidar_lock:
            if self._lidar_ranges is None:
                return 'LiDAR: no data'
            sectors = self._lidar_ranges.copy()

        parts = []
        for name in ['front', 'left', 'right', 'back']:
            dist = sectors.get(name, float('inf'))
            if dist == float('inf') or dist > 10.0:
                parts.append(f'{name}=clear(>10.0m)')
            else:
                parts.append(f'{name}={dist:.2f}m')

        status = ''
        front = sectors.get('front', float('inf'))
        if front < self.e_stop_dist:
            status = ' ** EMERGENCY: OBSTACLE VERY CLOSE! **'
        elif front < self.caution_dist:
            status = ' * CAUTION: obstacle nearby *'

        return 'LiDAR: ' + ', '.join(parts) + status

    # ======================================================================
    # Action Execution
    # ======================================================================

    def _execute_action(self, action: dict) -> dict:
        """Execute a movement action from the LLM response.

        Returns a dict with safety override info for the data logger.
        """
        cmd = action.get('action', 'stop')
        speed = float(action.get('speed', 0.0))
        duration = float(action.get('duration', 1.0))

        safety = {
            'triggered': False,
            'reason': '',
            'original_action': cmd,
            'override_action': cmd,
        }

        if cmd == 'navigate' and self.use_nav2 and self.nav2:
            return self._execute_nav2_action(action, safety)

        duration = min(duration, 5.0)
        linear_speed = speed * self.max_linear
        angular_speed = speed * self.max_angular

        # Pre-move safety check for forward motion
        linear_speed, safety = self._apply_safety_override(
            cmd, linear_speed, safety)
        if safety.get('override_action') == 'stop' and safety['triggered']:
            return safety

        self.get_logger().info(
            f'Action: {cmd} speed={speed:.1f} duration={duration:.1f}s'
        )

        return self._execute_motor_action(
            cmd, linear_speed, angular_speed, duration, safety)

    def _execute_nav2_action(self, action: dict, safety: dict) -> dict:
        """Handle Nav2 navigate action (hybrid mode)."""
        goal_x = float(action.get('goal_x', 0.0))
        goal_y = float(action.get('goal_y', 0.0))
        self.get_logger().info(
            f'Nav2 navigate to ({goal_x:.2f}, {goal_y:.2f})'
        )
        accepted = self.nav2.navigate_to(goal_x, goal_y)
        if not accepted:
            safety['triggered'] = True
            safety['reason'] = 'Nav2 goal not accepted'
            safety['override_action'] = 'stop'
            return safety

        t_start = time.time()
        while (self.nav2.is_navigating
               and self.running
               and time.time() - t_start < NAV2_GOAL_TIMEOUT):
            if self.emergency_stop:
                self.nav2.cancel_navigation()
                self._emergency_stop_motors()
                safety['triggered'] = True
                safety['reason'] = 'emergency stop during navigation'
                safety['override_action'] = 'stop'
                return safety
            if time.time() - t_start >= NAV2_CHECK_INTERVAL:
                self.get_logger().info(
                    'Nav2 still navigating — returning to LLM for re-evaluation'
                )
                safety['nav2_in_progress'] = True
                safety['nav2_goal'] = (goal_x, goal_y)
                return safety
            time.sleep(0.5)

        result = self.nav2.navigation_result
        if result == 'succeeded':
            self.get_logger().info('Nav2 goal reached')
        elif result == 'failed':
            self.get_logger().warn('Nav2 navigation failed')
            safety['triggered'] = True
            safety['reason'] = 'Nav2 navigation failed'
        elif result is None:
            self.get_logger().warn('Nav2 navigation timed out')
            self.nav2.cancel_navigation()
            safety['triggered'] = True
            safety['reason'] = 'Nav2 navigation timeout'
        return safety

    def _apply_safety_override(
        self, cmd: str, linear_speed: float, safety: dict,
    ) -> tuple[float, dict]:
        """Check LiDAR safety before forward motion, apply speed override.

        Returns (adjusted_linear_speed, updated_safety_dict).
        """
        if cmd not in ('forward', 'investigate'):
            return linear_speed, safety

        with self._lidar_lock:
            sectors = self._lidar_ranges or {}
        front_dist = sectors.get('front', float('inf'))

        if front_dist >= self.caution_dist:
            return linear_speed, safety

        if front_dist < self.e_stop_dist:
            self.get_logger().warn('LiDAR override: too close, stopping')
            self._stop_motors()
            safety['triggered'] = True
            safety['override_action'] = 'stop'
            safety['reason'] = f'obstacle too close ({front_dist:.2f}m)'
            return 0.0, safety

        self.get_logger().warn(
            f'LiDAR override: obstacle at {front_dist:.2f}m, reducing speed'
        )
        safety['triggered'] = True
        safety['reason'] = f'obstacle at {front_dist:.2f}m'
        return linear_speed * 0.3, safety

    def _execute_motor_action(
        self, cmd: str, linear_speed: float, angular_speed: float,
        duration: float, safety: dict,
    ) -> dict:
        """Dispatch motor command and run for the specified duration."""
        exec_start = time.time()

        if cmd == 'forward':
            self._send_twist(linear_speed, 0.0)
        elif cmd == 'backward':
            self._send_twist(-linear_speed, 0.0)
        elif cmd == 'turn_left':
            self._send_twist(linear_speed * 0.5, angular_speed)
        elif cmd == 'turn_right':
            self._send_twist(linear_speed * 0.5, -angular_speed)
        elif cmd == 'spin_left':
            self._send_twist(0.0, angular_speed)
        elif cmd == 'spin_right':
            self._send_twist(0.0, -angular_speed)
        elif cmd == 'stop':
            self._stop_motors()
            return safety
        elif cmd == 'look_around':
            self._look_around_sequence()
            return safety
        elif cmd == 'investigate':
            self._send_twist(linear_speed * 0.5, 0.0)
        else:
            self.get_logger().warn(f'Unknown action: {cmd}')
            self._stop_motors()
            safety['override_action'] = 'stop'
            return safety

        interrupted = self._run_timed_action(exec_start, duration)
        if interrupted:
            safety['triggered'] = True
            safety['reason'] = 'emergency stop during execution'
            safety['override_action'] = 'stop'
        return safety

    def _run_timed_action(self, start_time: float, duration: float) -> bool:
        """Run the current motor command for `duration` seconds.

        Monitors emergency stop and begins smooth deceleration before
        the duration expires. Returns True if interrupted by e-stop.
        """
        decel_time = min(0.4, duration * 0.3)
        decel_start = duration - decel_time

        while self.running:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break
            if self.emergency_stop:
                self.get_logger().warn('Emergency stop during action!')
                self._emergency_stop_motors()
                return True
            if elapsed >= decel_start:
                self._ramper.stop()
            time.sleep(0.05)

        self._stop_motors()
        return False

    def _look_around_sequence(self):
        """Pan the camera to survey surroundings."""
        self._stop_motors()
        positions = [
            (SERVO_CENTER - 400, SERVO_CENTER),       # Look left
            (SERVO_CENTER, SERVO_CENTER - 200),        # Look up-center
            (SERVO_CENTER + 400, SERVO_CENTER),        # Look right
            (SERVO_CENTER, SERVO_CENTER),              # Back to center
        ]
        for pan_pos, tilt_pos in positions:
            if not self.running:
                break
            self._move_servo(SERVO_PAN_ID, pan_pos, SERVO_MOVE_DURATION)
            self._move_servo(SERVO_TILT_ID, tilt_pos, SERVO_MOVE_DURATION)
            time.sleep(SERVO_MOVE_DURATION + 0.3)

    # ======================================================================
    # Joystick Control
    # ======================================================================

    def _toggle_control_mode(self):
        """Switch between autonomous and manual control modes."""
        self._stop_motors()
        if self.control_mode == 'autonomous':
            self.control_mode = 'manual'
            self.exploring = False
            self._center_camera()
            self._manual_pan = SERVO_CENTER
            self._manual_tilt = SERVO_CENTER
            self.get_logger().info(
                '>>> MANUAL MODE — joystick controls active'
            )
            if self.voice_on:
                self.voice.speak('Switching to manual mode.', force=True)
        else:
            self.control_mode = 'autonomous'
            self.get_logger().info(
                '>>> AUTONOMOUS MODE — press Enter or say "start" to explore'
            )
            if self.voice_on:
                self.voice.speak('Switching to autonomous mode. Say start to explore.', force=True)

    def _joystick_button_callback(self, button: str):
        """Handle joystick button presses (called from JoystickReader thread)."""
        if button == 'start':
            self._toggle_control_mode()
        elif button == 'select' and self.control_mode == 'manual':
            self._center_camera()
            self._manual_pan = SERVO_CENTER
            self._manual_tilt = SERVO_CENTER
            self.get_logger().info('Camera centered (Select button)')

    def _joystick_command_loop(self):
        """20Hz loop: read joystick axes and drive motors in manual mode."""
        self.get_logger().info('Joystick command loop started')
        while self.running:
            try:
                if self.control_mode != 'manual' or not self._joystick.connected:
                    time.sleep(0.1)
                    continue

                axes = self._joystick.axes

                # --- Drive: left stick Y → forward/back, right stick X → turn ---
                linear_x = -axes.get('left_y', 0.0) * self.max_linear
                angular_z = -axes.get('right_x', 0.0) * self.max_angular

                if abs(linear_x) > 0.001 or abs(angular_z) > 0.001:
                    self._joystick_moving = True
                    self._send_twist_joy(linear_x, angular_z)
                elif self._joystick_moving:
                    self._joystick_moving = False
                    self._joy_ramper.stop()

                # --- Camera servos: D-pad ---
                hat_x = axes.get('hat_x', 0.0)
                hat_y = axes.get('hat_y', 0.0)

                if hat_x != 0.0:
                    # D-pad left/right → pan (left = positive direction)
                    self._manual_pan += int(hat_x) * self._servo_step
                    self._manual_pan = max(SERVO_MIN, min(SERVO_MAX, self._manual_pan))
                    self._move_servo(SERVO_PAN_ID, self._manual_pan, 0.15)

                if hat_y != 0.0:
                    # D-pad up/down → tilt (up = decrease for upward look)
                    self._manual_tilt -= int(hat_y) * self._servo_step
                    self._manual_tilt = max(SERVO_MIN, min(SERVO_MAX, self._manual_tilt))
                    self._move_servo(SERVO_TILT_ID, self._manual_tilt, 0.15)

                # ~20Hz
                time.sleep(0.05)

            except Exception as e:
                self.get_logger().error(f'Joystick command loop error: {e}')
                time.sleep(1.0)

        self.get_logger().info('Joystick command loop stopped')

    # ======================================================================
    # Voice Listener Loop
    # ======================================================================

    def _voice_listener_loop(self):
        """Background thread: listen for wake word, then capture command."""
        self.get_logger().info('Voice listener started')
        _vlog = '/tmp/voice_debug.log'
        with open(_vlog, 'a') as _f:
            _f.write(f'Voice listener started, wake_detector.available={self.wake_detector.available}\n')
        while self.running:
            try:
                # Check for wake word
                if self.wake_detector.available:
                    if self.wake_detector.check_wakeup():
                        self.get_logger().info('Wake word detected, listening 10s...')
                        self._publish_agent_status('Listening... (10s)')
                        with open(_vlog, 'a') as _f:
                            _f.write('Wake word detected!\n')
                        # Beep = "start speaking now", record 10s, beep = "done"
                        self.voice.beep()
                        command = self.voice.listen_for_command(duration=10)
                        with open(_vlog, 'a') as _f:
                            _f.write(f'STT result: "{command}" (len={len(command)})\n')
                        self.get_logger().info(
                            f'STT result: "{command}" (len={len(command)})')
                        if command:
                            self._voice_queue.put(command)
                            self.get_logger().info(
                                f'Voice command queued: {command}')
                            with open(_vlog, 'a') as _f:
                                _f.write(f'Queued: {command}\n')
                        else:
                            self.get_logger().warn(
                                'STT returned empty — no command captured')
                else:
                    time.sleep(1.0)
                time.sleep(0.05)
            except Exception as e:
                self.get_logger().error(f'Voice listener error: {e}')
                with open(_vlog, 'a') as _f:
                    _f.write(f'Error: {e}\n')
                time.sleep(1.0)

    def _process_voice_command(self, command: str) -> bool:
        """Process voice command: repeat it back, then execute immediately."""
        cmd = command.strip()
        if not cmd:
            return self.exploring

        self.get_logger().info(f'Voice instruction: {cmd}')
        self._publish_agent_status(f'VOICE COMMAND: "{cmd}"')

        # Repeat the instruction back and proceed
        self.voice.speak(
            f'Understood, Sir. {cmd}. Right away.',
            block=True, force=True,
        )

        self._publish_agent_status(f'Executing: "{cmd}"')
        self._pending_voice_instruction = cmd

        if not self.exploring:
            self.exploring = True
        return True

    def _publish_agent_status(self, text: str):
        """Publish a text status to /semantic_map/agent_status for Foxglove."""
        try:
            msg = String()
            msg.data = text
            self._agent_status_pub.publish(msg)
        except Exception:
            pass  # Node may be shutting down

    # ======================================================================
    # Main Exploration Loop
    # ======================================================================

    def _wait_for_camera(self):
        """Block until the first camera frame arrives, or timeout.

        In dry-run mode, creates a dummy frame instead of waiting.
        """
        if self.provider_name in ('dryrun', 'dry-run', 'dry_run'):
            self.get_logger().info('Dry-run mode: using dummy camera frame')
            with self._rgb_lock:
                if self._rgb_image is None:
                    self._rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        self._rgb_image, 'DRY RUN', (180, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3,
                    )
            return

        for i in range(20):
            if not self.running:
                return
            with self._rgb_lock:
                if self._rgb_image is not None:
                    self.get_logger().info('Camera feed received. Ready to explore.')
                    return
            if i % 2 == 0:
                self.get_logger().info('Waiting for camera feed...')
            time.sleep(0.5)

        cam_type = os.environ.get('DEPTH_CAMERA_TYPE', 'not set')
        self.get_logger().warn(
            f'No camera frame received after 10s. '
            f'Is the camera running? (DEPTH_CAMERA_TYPE={cam_type})'
        )

    def _snapshot_sensors(self) -> dict:
        """Take a thread-safe snapshot of all sensor data.

        Returns a dict with keys: rgb, depth, lidar_sectors, lidar_raw_ranges,
        imu, odom, image_b64, depth_summary, lidar_summary, memory_context.
        Returns empty dict if no camera frame is available.
        """
        image_b64 = self._get_camera_frame_b64()
        if not image_b64:
            return {}

        with self._rgb_lock:
            rgb = self._rgb_image.copy() if self._rgb_image is not None else None
        with self._depth_lock:
            depth = self._depth_image.copy() if self._depth_image is not None else None
        with self._lidar_lock:
            lidar_sectors = self._lidar_ranges.copy() if self._lidar_ranges else None
            lidar_raw_ranges = list(self._lidar_raw.ranges) if self._lidar_raw else None
        with self._imu_lock:
            imu = self._imu_data.copy() if self._imu_data else None
        with self._odom_lock:
            odom = self._odom_data.copy() if self._odom_data else None

        return {
            'rgb': rgb,
            'depth': depth,
            'lidar_sectors': lidar_sectors,
            'lidar_raw_ranges': lidar_raw_ranges,
            'imu': imu,
            'odom': odom,
            'image_b64': image_b64,
            'depth_summary': self._get_depth_summary(),
            'lidar_summary': self._get_lidar_summary(),
            'memory_context': self.memory.get_context_summary(),
        }

    def _build_sensor_context(self, sensors: dict) -> tuple[str, str, str, str | None]:
        """Build sensor-related text for the user prompt.

        Returns:
            (odom_summary, imu_summary, nav2_summary, map_b64_or_None)
            where nav2_summary includes map stats and frontier goals.
        """
        odom = sensors['odom']

        odom_summary = ''
        if odom:
            heading_deg = math.degrees(odom.get('theta', 0))
            odom_summary = (
                f'ODOMETRY: position=({odom.get("x", 0):.2f}, '
                f'{odom.get("y", 0):.2f}) m, '
                f'heading={heading_deg:.1f}\u00b0'
            )

        imu_summary = ''
        if sensors['imu']:
            ori = sensors['imu'].get('orientation', {})
            imu_summary = (
                f'IMU: roll={math.degrees(ori.get("roll", 0)):.1f}\u00b0, '
                f'pitch={math.degrees(ori.get("pitch", 0)):.1f}\u00b0, '
                f'yaw={math.degrees(ori.get("yaw", 0)):.1f}\u00b0'
            )

        nav2_summary = ''
        map_b64 = None
        if self.use_nav2 and self.nav2 and self.nav2.has_map:
            rx = odom.get('x', 0) if odom else 0
            ry = odom.get('y', 0) if odom else 0
            rt = odom.get('theta', 0) if odom else 0
            map_img = self.nav2.render_map_image(rx, ry, rt)
            if map_img is not None:
                _, map_jpg = cv2.imencode(
                    '.jpg', map_img, [cv2.IMWRITE_JPEG_QUALITY, 80],
                )
                map_b64 = base64.b64encode(map_jpg.tobytes()).decode('utf-8')

            parts = []
            ms = self.nav2.get_map_stats()
            if ms:
                parts.append(
                    f'MAP: {ms.get("explored_pct", 0)}% explored, '
                    f'{ms.get("area_m2", 0)}m\u00b2 mapped, '
                    f'{ms.get("free_cells", 0)} free / '
                    f'{ms.get("occupied_cells", 0)} walls / '
                    f'{ms.get("unknown_cells", 0)} unknown cells'
                )

            frontiers = self.nav2.get_frontier_goals(rx, ry)
            if frontiers:
                frontier_lines = [
                    f'  F{i+1}: ({f["x"]:.2f},{f["y"]:.2f}) '
                    f'dist={f["distance"]:.1f}m size={f["size"]}'
                    for i, f in enumerate(frontiers[:5])
                ]
                parts.append(
                    'FRONTIER GOALS (unexplored boundaries):\n'
                    + '\n'.join(frontier_lines)
                )
            nav2_summary = '\n'.join(parts)

        return odom_summary, imu_summary, nav2_summary, map_b64

    def _build_identity_and_knowledge_context(self, odom: dict | None) -> str:
        """Build identity and world knowledge text for the user prompt."""
        identity_context = self.consciousness.get_identity_context()
        knowledge_context = self.world_knowledge.get_prompt_context(
            x=odom.get('x', 0) if odom else 0,
            y=odom.get('y', 0) if odom else 0,
            theta=odom.get('theta', 0) if odom else 0,
        )

        result = f'{identity_context}\n\n'
        if knowledge_context:
            result += f'{knowledge_context}\n\n'
        return result

    def _build_user_prompt(self, sensors: dict) -> tuple[str, str, str | None]:
        """Build the system prompt and user prompt for the LLM.

        Args:
            sensors: Snapshot dict from _snapshot_sensors().

        Returns:
            (system_prompt, user_prompt, map_b64_or_None)
        """
        base_prompt = HYBRID_SYSTEM_PROMPT if self.use_nav2 else SYSTEM_PROMPT
        system_prompt = build_system_prompt(base_prompt)

        odom_summary, imu_summary, nav2_summary, map_b64 = (
            self._build_sensor_context(sensors))

        user_prompt = self._build_identity_and_knowledge_context(sensors['odom'])
        user_prompt += (
            f'Current sensor data:\n'
            f'{sensors["lidar_summary"]}\n'
            f'{sensors["depth_summary"]}\n'
            f'{odom_summary}\n'
            f'{imu_summary}\n'
        )
        if nav2_summary:
            user_prompt += f'{nav2_summary}\n'
        user_prompt += (
            f'\nExploration context:\n{sensors["memory_context"]}\n'
            f'\nAnalyze the camera image'
        )
        if map_b64:
            user_prompt += ' and the bird\'s-eye map'
        user_prompt += (
            ' and ALL sensor data. '
            'Decide your next action. Respond in JSON only.'
        )

        return system_prompt, user_prompt, map_b64

    def _call_llm(self, sensors: dict, system_prompt: str, user_prompt: str,
                  map_b64: str | None) -> tuple[dict, dict, float]:
        """Call the LLM provider and return parsed result, metadata, and cost.

        Returns:
            (result_dict, meta_dict, cost_usd)
        """
        self.get_logger().info(
            f'Calling {self.provider_name} for scene analysis...'
        )

        images = [sensors['image_b64']]
        if map_b64:
            images.append(map_b64)

        result = self.llm.analyze_scene(images, system_prompt, user_prompt)

        meta = result.pop('_meta', {})
        t_elapsed_ms = meta.get('response_time_ms', 0)
        tokens_in = meta.get('tokens_input', 0)
        tokens_out = meta.get('tokens_output', 0)

        cost_in = tokens_in * COST_PER_M_INPUT_TOKENS.get(
            self.provider_name, 3.0) / 1_000_000
        cost_out = tokens_out * COST_PER_M_OUTPUT_TOKENS.get(
            self.provider_name, 15.0) / 1_000_000
        cost_usd = cost_in + cost_out

        self.get_logger().info(
            f'LLM response ({t_elapsed_ms}ms, '
            f'{tokens_in}+{tokens_out} tok, ${cost_usd:.4f}): '
            f'action={result.get("action")} '
            f'speech="{result.get("speech", "")[:60]}"'
        )

        return result, meta, cost_usd

    def _update_dashboard_state(self, result: dict, meta: dict,
                                cost_usd: float, safety_info: dict):
        """Update LLM and safety state for the status dashboard."""
        self._cycle_count += 1
        self._session_total_cost += cost_usd

        with self._llm_lock:
            s = self._llm_state
            s.action = result.get('action', '')
            s.speed = float(result.get('speed', 0.0))
            s.duration = float(result.get('duration', 0.0))
            s.reasoning = result.get('reasoning', '')
            s.speech = result.get('speech', '')
            s.response_ms = meta.get('response_time_ms', 0)
            s.tokens_in = meta.get('tokens_input', 0)
            s.tokens_out = meta.get('tokens_output', 0)
            s.cost = cost_usd
            s.timestamp = time.time()
            s.safety_triggered = safety_info.get('triggered', False)
            s.safety_reason = safety_info.get('reason', '')

    def _record_cycle(self, result: dict, meta: dict, cost_usd: float,
                      safety_info: dict, sensors: dict,
                      system_prompt: str, user_prompt: str,
                      exec_duration_ms: int, actual_action: str,
                      speech_text: str, voice_cmd: str | None):
        """Record exploration memory, consciousness, knowledge, and data log."""
        odom = sensors.get('odom')
        lidar_sectors = sensors.get('lidar_sectors')
        lidar_summary = sensors.get('lidar_summary', '')

        # Exploration memory
        self.memory.record_action(
            result, lidar_summary, odom=odom, lidar_sectors=lidar_sectors,
        )

        reasoning = result.get('reasoning', '')
        if reasoning:
            self.get_logger().info(f'Reasoning: {reasoning[:120]}')

        # Consciousness: reflection + cycle stats
        reflection = result.get('embodied_reflection', '')
        if reflection:
            self.consciousness.record_reflection(reflection)

        self.consciousness.record_cycle(
            result, safety_info, cost_usd,
            distance_delta=self._get_odom_distance_delta(odom),
        )

        # World knowledge
        self.world_knowledge.update_from_response(result, odom=odom)
        for room in self.world_knowledge.world_map.get('rooms', {}):
            self.consciousness.record_room(room)

        # Data logger
        rgb = sensors.get('rgb')
        img_h, img_w = (rgb.shape[:2] if rgb is not None else (0, 0))
        image_b64 = sensors.get('image_b64', '')
        image_size_bytes = len(image_b64) * 3 // 4

        self.data_logger.log_cycle(
            rgb_image=rgb,
            depth_image=sensors.get('depth'),
            lidar_ranges=sensors.get('lidar_raw_ranges'),
            lidar_sectors=lidar_sectors,
            imu_data=sensors.get('imu'),
            odom_data=odom,
            battery_voltage=self._battery_voltage,
            provider=self.provider_name,
            model=meta.get('model', ''),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_resolution=f'{img_w}x{img_h}',
            image_size_bytes=image_size_bytes,
            raw_response=meta.get('raw_response', ''),
            parsed_action=result.get('action', ''),
            speed=result.get('speed', 0.0),
            duration=result.get('duration', 0.0),
            speech=speech_text,
            reasoning=reasoning,
            response_time_ms=meta.get('response_time_ms', 0),
            tokens_input=meta.get('tokens_input', 0),
            tokens_output=meta.get('tokens_output', 0),
            cost_usd=cost_usd,
            safety_triggered=safety_info.get('triggered', False),
            safety_reason=safety_info.get('reason', ''),
            safety_original_action=safety_info.get('original_action', ''),
            safety_override_action=safety_info.get('override_action', ''),
            actual_action=actual_action,
            motor_left_speed=self._last_twist[0],
            motor_right_speed=self._last_twist[1],
            servo_pan=self._last_servo_pan,
            servo_tilt=self._last_servo_tilt,
            execution_duration_ms=exec_duration_ms,
            voice_command=voice_cmd,
            speech_output=speech_text,
            total_distance=round(self._total_distance, 3),
            areas_visited=self.memory.total_actions,
            objects_discovered=[
                d['description'][:60] for d in self.memory.discoveries[-20:]
            ],
            map_coverage_pct=(
                self.nav2.get_map_stats().get('explored_pct', 0.0)
                if self.use_nav2 and self.nav2 and self.nav2.has_map
                else 0.0
            ),
            embodied_reflection=reflection,
            outing_number=self.consciousness.outing_number,
        )

    def _get_odom_distance_delta(self, odom: dict | None) -> float:
        """Compute distance traveled since last recorded odom position.

        Uses a separate pair of tracking variables so this is independent
        of the continuous odometry accumulation in _odom_callback.
        """
        if not odom:
            return 0.0
        x, y = odom.get('x', 0), odom.get('y', 0)
        if self._prev_cycle_odom_x is None:
            self._prev_cycle_odom_x = x
            self._prev_cycle_odom_y = y
            return 0.0
        dx = x - self._prev_cycle_odom_x
        dy = y - self._prev_cycle_odom_y
        self._prev_cycle_odom_x = x
        self._prev_cycle_odom_y = y
        return math.sqrt(dx * dx + dy * dy)

    def _check_stuck_watchdog(self, odom: dict | None):
        """Force recovery action if the robot hasn't moved enough."""
        if (self._cycle_count - self._stuck_check_cycle
                < self._stuck_cycles_threshold):
            return

        dist_since = self._total_distance - self._stuck_check_distance
        if dist_since < self._stuck_distance_threshold:
            self.get_logger().warn(
                f'Stuck watchdog: only {dist_since:.2f}m in '
                f'{self._stuck_cycles_threshold} cycles. Forcing recovery.'
            )
            if self.use_nav2 and self.nav2 and self.nav2.has_map:
                rx = odom.get('x', 0) if odom else 0
                ry = odom.get('y', 0) if odom else 0
                frontiers = self.nav2.get_frontier_goals(rx, ry)
                if frontiers:
                    f = frontiers[0]
                    self.get_logger().info(
                        f'Stuck recovery: navigating to frontier '
                        f'({f["x"]:.1f}, {f["y"]:.1f})'
                    )
                    self.nav2.navigate_to(f['x'], f['y'])
            else:
                # Use _execute_action instead of sleep-blocking the loop.
                # This respects emergency stop and uses proper ramping.
                self.get_logger().info('Stuck recovery: forced spin')
                self._execute_action({
                    'action': 'spin_left',
                    'speed': STUCK_RECOVERY_SPEED,
                    'duration': STUCK_RECOVERY_DURATION,
                })

        self._stuck_check_distance = self._total_distance
        self._stuck_check_cycle = self._cycle_count

    def _exploration_loop(self):
        """Background thread: continuous sense -> think -> act -> speak cycle."""
        self.get_logger().info('Exploration loop started (waiting for start command)')
        self._center_camera()
        self._wait_for_camera()

        self.exploring = True
        self.get_logger().info('Auto-starting exploration')
        if self.voice_on:
            self.voice.speak('I am Jeeves, and I am ready to explore.', force=True)

        while self.running:
            try:
                # Process any voice commands first
                voice_cmd = None
                while not self._voice_queue.empty():
                    voice_cmd = self._voice_queue.get_nowait()
                    self._process_voice_command(voice_cmd)

                cycle_start = time.time()

                if not self.exploring:
                    time.sleep(0.5)
                    continue

                if self.emergency_stop:
                    self.get_logger().warn('Emergency stop — waiting for clearance')
                    self._stop_motors()
                    time.sleep(0.5)
                    continue

                # SENSE
                sensors = self._snapshot_sensors()
                if not sensors:
                    self.get_logger().warn('No camera frame available')
                    time.sleep(1.0)
                    continue

                # BUILD PROMPT
                system_prompt, user_prompt, map_b64 = self._build_user_prompt(
                    sensors)

                # THINK
                result, meta, cost_usd = self._call_llm(
                    sensors, system_prompt, user_prompt, map_b64)

                # SPEAK
                speech_text = result.get('speech', '')
                if speech_text and self.voice_on:
                    self.voice.speak(speech_text)

                # ACT
                exec_start = time.time()
                safety_info = self._execute_action(result)
                exec_duration_ms = int((time.time() - exec_start) * 1000)
                actual_action = safety_info.get(
                    'override_action', result.get('action', 'stop'))

                # UPDATE DASHBOARD
                self._update_dashboard_state(result, meta, cost_usd, safety_info)

                # REMEMBER + LOG
                self._record_cycle(
                    result, meta, cost_usd, safety_info, sensors,
                    system_prompt, user_prompt, exec_duration_ms,
                    actual_action, speech_text, voice_cmd,
                )

                # STUCK WATCHDOG
                self._check_stuck_watchdog(sensors.get('odom'))

                # Wait before next cycle (compensating for elapsed time)
                cycle_elapsed = time.time() - cycle_start
                remaining = self.loop_interval - cycle_elapsed
                if remaining > 0:
                    time.sleep(remaining)

            except Exception as e:
                self.get_logger().error(
                    f'Exploration loop error: {e}\n{traceback.format_exc()}'
                )
                self._stop_motors()
                time.sleep(2.0)

    # ======================================================================
    # Agent Mode Loop (ROSA-style tool-calling)
    # ======================================================================

    def _agent_loop(self):
        """Background thread: ROSA-style ReAct loop with native tool-calling.

        Each turn: sense -> build context -> LLM selects tools -> execute tools
        -> feed results back -> LLM may call more tools or end turn.
        """
        self.get_logger().info(
            'Agent loop started (ROSA-style tool-calling mode)'
        )
        self._center_camera()
        self._wait_for_camera()

        self.exploring = False
        self.get_logger().info(
            'Agent ready — awaiting voice instruction via WonderEcho wake word.'
        )

        while self.running:
            try:
                while not self._voice_queue.empty():
                    voice_cmd = self._voice_queue.get_nowait()
                    self._process_voice_command(voice_cmd)

                if not self.exploring:
                    time.sleep(AGENT_IDLE_SLEEP)
                    continue

                if self.emergency_stop:
                    self.get_logger().warn('Emergency stop — waiting for clearance')
                    self._stop_motors()
                    time.sleep(AGENT_IDLE_SLEEP)
                    continue

                cycle_start = time.time()
                self._cycle_count += 1

                # SENSE + BUILD CONTEXT
                ctx = self._prepare_agent_context()
                if ctx is None:
                    time.sleep(1.0)
                    continue

                sensors, system_prompt, messages, tools = ctx

                # REACT LOOP
                response, turn_cost = self._run_agent_turn(
                    system_prompt, messages, tools)

                # FINALIZE
                self._finalize_agent_turn(
                    response, turn_cost, sensors, cycle_start)

            except Exception as e:
                self.get_logger().error(
                    f'Agent loop error: {e}\n{traceback.format_exc()}'
                )
                self._stop_motors()
                time.sleep(AGENT_ERROR_SLEEP)

    def _prepare_agent_context(self):
        """Build sensor context and conversation messages for an agent turn.

        Returns (sensors, system_prompt, messages, tools) or None if no
        camera frame is available.
        """
        sensors = self._snapshot_sensors()
        if not sensors:
            self.get_logger().warn('No camera frame available')
            return None

        from autonomous_explorer.conversation_manager import (
            build_sensor_context,
            build_identity_context,
        )
        sensor_context = build_sensor_context(
            lidar_summary=sensors['lidar_summary'],
            depth_summary=sensors['depth_summary'],
            odom=sensors.get('odom'),
            imu=sensors.get('imu'),
            battery=self._battery_voltage,
            emergency_stop=self.emergency_stop,
        )
        identity_context = build_identity_context(
            self.consciousness, self.world_knowledge,
            odom=sensors.get('odom'),
        )

        user_text = f'{identity_context}\n\n{sensor_context}'

        if self._pending_voice_instruction:
            user_text += (
                f'\n\n** VOICE INSTRUCTION FROM MASTER: '
                f'"{self._pending_voice_instruction}" **\n'
                f'Respond to this instruction. Use speak() to reply verbally.'
            )
            self.get_logger().info(
                f'Injecting voice instruction: {self._pending_voice_instruction}')
            self._pending_voice_instruction = None

        self._conversation.add_user_message(
            user_text, image_b64=sensors.get('image_b64', ''),
        )

        system_prompt = build_system_prompt(AGENT_SYSTEM_PROMPT)

        provider = self.provider_name
        if provider in ('claude',):
            messages = self._conversation.get_messages_claude()
            tools = self._tool_registry.to_claude_tools()
        else:
            messages = self._conversation.get_messages_openai()
            tools = self._tool_registry.to_openai_tools()

        return sensors, system_prompt, messages, tools

    def _run_agent_turn(self, system_prompt: str, messages: list,
                        tools: list):
        """Execute the ReAct tool-calling loop for one agent turn.

        Returns (last_response, total_turn_cost).
        """
        turn_cost = 0.0
        turn_start = time.time()
        rounds = 0
        response = None
        provider = self.provider_name

        while rounds < AGENT_MAX_TOOL_ROUNDS:
            if time.time() - turn_start > AGENT_TURN_TIMEOUT:
                self.get_logger().warn('Agent turn timeout')
                break

            if not self.running or self.emergency_stop:
                break

            rounds += 1

            response = self.llm.agent_turn(
                system_prompt=system_prompt,
                messages=messages,
                tools=tools,
            )

            cost = self._estimate_cost(
                response.tokens_input, response.tokens_output)
            turn_cost += cost
            self._session_total_cost += cost

            self.get_logger().info(
                f'Agent round {rounds}: '
                f'{len(response.tool_calls)} tool calls, '
                f'stop={response.stop_reason}, '
                f'{response.response_time_ms}ms'
            )

            if response.text:
                self.get_logger().info(
                    f'Agent says: {response.text[:120]}'
                )

            if not response.tool_calls:
                self._conversation.add_assistant_message(
                    response.text, tool_calls=None)
                break

            self._conversation.add_assistant_message(
                response.text,
                tool_calls=[
                    {'id': tc.call_id, 'name': tc.tool_name,
                     'arguments': tc.arguments}
                    for tc in response.tool_calls
                ],
            )

            tool_results = []
            for tc in response.tool_calls:
                self.get_logger().info(
                    f'  Tool: {tc.tool_name}({_summarize_args(tc.arguments)})'
                )
                try:
                    result = self._tool_registry.execute(
                        tc.tool_name, tc.arguments)
                except Exception as e:
                    result = {'success': False, 'error': str(e)}
                    self.get_logger().error(
                        f'  Tool error: {tc.tool_name}: {e}')

                self.get_logger().info(
                    f'  Result: {_summarize_result(result)}')
                tool_results.append({
                    'call_id': tc.call_id,
                    'name': tc.tool_name,
                    'result': result,
                })

            self._conversation.add_tool_results(tool_results)

            if provider in ('claude',):
                messages = self._conversation.get_messages_claude()
            else:
                messages = self._conversation.get_messages_openai()

            if response.stop_reason in ('end_turn', 'stop'):
                break

        return response, turn_cost

    def _finalize_agent_turn(self, response, turn_cost: float,
                             sensors: dict, cycle_start: float):
        """Update dashboard, consciousness, stuck watchdog after an agent turn."""
        with self._llm_lock:
            s = self._llm_state
            s.action = 'agent_turn'
            s.reasoning = response.text[:200] if response and response.text else ''
            s.response_ms = int((time.time() - cycle_start) * 1000)
            s.cost = turn_cost
            s.timestamp = time.time()

        odom = sensors.get('odom')
        self.consciousness.record_cycle(
            {'action': 'agent_turn'},
            {'triggered': False},
            turn_cost,
            distance_delta=self._get_odom_distance_delta(odom),
        )

        self._check_stuck_watchdog(odom)

        cycle_elapsed = time.time() - cycle_start
        remaining = self.loop_interval - cycle_elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _estimate_cost(self, tokens_in: int, tokens_out: int) -> float:
        """Estimate USD cost for a single LLM call."""
        cost_in = tokens_in * COST_PER_M_INPUT_TOKENS.get(
            self.provider_name, 3.0) / 1_000_000
        cost_out = tokens_out * COST_PER_M_OUTPUT_TOKENS.get(
            self.provider_name, 15.0) / 1_000_000
        return cost_in + cost_out

    # ======================================================================
    # Shutdown
    # ======================================================================

    def shutdown(self):
        """Graceful shutdown."""
        self.get_logger().info('Shutting down Autonomous Explorer...')
        self.running = False
        self.exploring = False
        self._emergency_stop_motors()
        self._ramper.shutdown()
        self._joy_ramper.shutdown()
        self._center_camera()
        self._joystick.stop()
        self.memory.save()
        self.data_logger.stop()
        self.wake_detector.stop()
        if self.nav2:
            self.nav2.cancel_navigation()
            self.nav2.destroy()

        # Consciousness: save stats, write journal, update knowledge
        self.consciousness.save()
        try:
            self.consciousness.write_journal(self.memory, self.llm)
        except Exception as e:
            self.get_logger().warning(f'Journal write failed: {e}')
        try:
            self.world_knowledge.end_of_session_update(self.memory, self.llm)
        except Exception as e:
            self.get_logger().warning(f'Knowledge update failed: {e}')

        if self.voice_on:
            self.voice.speak('Exploration complete. Shutting down.', block=True, force=True)
        self.get_logger().info('Shutdown complete.')


def _summarize_args(args: dict) -> str:
    """Compact string summary of tool arguments for logging."""
    parts = []
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 40:
            v = v[:37] + '...'
        parts.append(f'{k}={v!r}')
    return ', '.join(parts)[:120]


def _summarize_result(result: dict) -> str:
    """Compact string summary of tool result for logging."""
    success = result.get('success', '?')
    error = result.get('error', '')
    if error:
        return f'success={success}, error={error[:60]}'
    # Show first few interesting keys
    keys = [k for k in result if k != 'success'][:3]
    parts = [f'success={success}']
    for k in keys:
        v = result[k]
        if isinstance(v, str) and len(v) > 30:
            v = v[:27] + '...'
        elif isinstance(v, dict):
            v = '{...}'
        elif isinstance(v, list):
            v = f'[{len(v)} items]'
        parts.append(f'{k}={v}')
    return ', '.join(parts)[:120]


def main(args=None):
    rclpy.init(args=args)
    node = AutonomousExplorer()

    # Handle keyboard interrupt and SIGTERM
    def signal_handler(sig, frame):
        node.get_logger().info(f'Signal {sig} received')
        node.shutdown()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Spin ROS2 with multi-threaded executor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
