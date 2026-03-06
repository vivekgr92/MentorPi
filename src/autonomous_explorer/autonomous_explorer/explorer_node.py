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

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from std_msgs.msg import String, UInt16
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from ros_robot_controller_msgs.msg import SetPWMServoState, PWMServoState
from sensor_msgs.msg import Image, LaserScan

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

from autonomous_explorer.config import (
    ANTHROPIC_API_KEY,
    AUDIO_DEVICE,
    BATTERY_TOPIC,
    CAMERA_DEPTH_TOPIC,
    CAMERA_JPEG_QUALITY,
    CAMERA_RGB_TOPIC,
    CAUTION_DISTANCE,
    CLAUDE_MODEL,
    CMD_VEL_TOPIC,
    COST_PER_M_INPUT_TOKENS,
    COST_PER_M_OUTPUT_TOKENS,
    EMERGENCY_STOP_DISTANCE,
    IMU_TOPIC,
    LIDAR_MAX_SCAN_ANGLE,
    LIDAR_TOPIC,
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
    ODOM_TOPIC,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    SAFE_DISTANCE,
    SERVO_CENTER,
    SERVO_MAX,
    SERVO_MIN,
    SERVO_MOVE_DURATION,
    SERVO_PAN_ID,
    SERVO_TILT_ID,
    SERVO_TOPIC,
    STATUS_PUBLISH_RATE,
    STATUS_TOPIC,
    STT_MODEL,
    SYSTEM_PROMPT,
    TTS_MODEL,
    TTS_VOICE,
    VOICE_ENABLED,
    WONDERECHO_PORT,
)
from autonomous_explorer.data_logger import DataLogger
from autonomous_explorer.exploration_memory import ExplorationMemory
from autonomous_explorer.joystick_reader import JoystickReader
from autonomous_explorer.llm_provider import create_provider
from autonomous_explorer.voice_io import VoiceIO, WonderEchoDetector


class AutonomousExplorer(Node):
    """Main ROS2 node for autonomous LLM-driven exploration."""

    def __init__(self):
        super().__init__('autonomous_explorer')
        self.get_logger().info('Initializing Autonomous Explorer...')

        # ------------------------------------------------------------------
        # Declare ROS2 parameters (overridable via launch file / CLI)
        # ------------------------------------------------------------------
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

        # Read parameters
        self.provider_name = self.get_parameter(
            'llm_provider').get_parameter_value().string_value
        self.anthropic_key = self.get_parameter(
            'anthropic_api_key').get_parameter_value().string_value
        self.openai_key = self.get_parameter(
            'openai_api_key').get_parameter_value().string_value
        self.claude_model = self.get_parameter(
            'claude_model').get_parameter_value().string_value
        self.openai_model = self.get_parameter(
            'openai_model').get_parameter_value().string_value
        self.loop_interval = self.get_parameter(
            'loop_interval').get_parameter_value().double_value
        self.e_stop_dist = self.get_parameter(
            'emergency_stop_distance').get_parameter_value().double_value
        self.caution_dist = self.get_parameter(
            'caution_distance').get_parameter_value().double_value
        self.max_linear = self.get_parameter(
            'max_linear_speed').get_parameter_value().double_value
        self.max_angular = self.get_parameter(
            'max_angular_speed').get_parameter_value().double_value
        self.voice_on = self.get_parameter(
            'voice_enabled').get_parameter_value().bool_value

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.bridge = CvBridge()
        self.running = True
        self.exploring = False  # Start paused — user must say go or press Enter
        self.emergency_stop = False

        # Latest sensor data (thread-safe via locks)
        self._rgb_lock = threading.Lock()
        self._rgb_image = None  # numpy BGR

        self._depth_lock = threading.Lock()
        self._depth_image = None  # numpy uint16 (mm)

        self._lidar_lock = threading.Lock()
        self._lidar_ranges = None  # dict with front/left/right/back min distances
        self._lidar_raw = None  # raw LaserScan for detailed analysis

        self._imu_lock = threading.Lock()
        self._imu_data = None  # dict with orientation, accel, gyro

        self._odom_lock = threading.Lock()
        self._odom_data = None  # dict with position {x, y, theta}

        self._battery_voltage = None  # float or None

        # Track last twist and servo commands for logging
        self._last_twist = (0.0, 0.0)  # (linear_x, angular_z)
        self._last_servo_pan = SERVO_CENTER
        self._last_servo_tilt = SERVO_CENTER
        # Latest voice command text (consumed per cycle)
        self._last_voice_cmd = None

        self._last_command_time = time.time()

        # LLM state tracking (for status dashboard)
        self._llm_lock = threading.Lock()
        self._last_llm_action = ''
        self._last_llm_speed = 0.0
        self._last_llm_duration = 0.0
        self._last_llm_reasoning = ''
        self._last_llm_speech = ''
        self._last_llm_response_ms = 0
        self._last_llm_tokens_in = 0
        self._last_llm_tokens_out = 0
        self._last_llm_cost = 0.0
        self._last_llm_time = 0.0  # time.time() of last LLM result
        self._last_safety_triggered = False
        self._last_safety_reason = ''

        # Session-level counters
        self._cycle_count = 0
        self._session_total_cost = 0.0
        self._session_start_time = time.time()

        # Voice command queue (from wake word + STT)
        self._voice_queue: queue.Queue = queue.Queue(maxsize=5)

        # ------------------------------------------------------------------
        # Control mode: "autonomous" (LLM-driven) or "manual" (joystick)
        # ------------------------------------------------------------------
        self.control_mode = 'autonomous'

        # Joystick reader (daemon thread)
        self._joystick = JoystickReader(
            on_button_press=self._joystick_button_callback,
            logger=self.get_logger(),
        )
        self._joystick.start()

        # Camera servo positions for D-pad control (start centered)
        self._manual_pan = SERVO_CENTER
        self._manual_tilt = SERVO_CENTER
        _SERVO_STEP = 100  # microseconds per D-pad press tick
        self._servo_step = _SERVO_STEP
        self._joystick_moving = False

        # ------------------------------------------------------------------
        # Initialize LLM provider (auto-fallback to dry-run if no API key)
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Initialize exploration memory
        # ------------------------------------------------------------------
        self.memory = ExplorationMemory(MEMORY_FILE)

        # ------------------------------------------------------------------
        # Initialize voice I/O
        # ------------------------------------------------------------------
        # Use openai key for TTS/STT regardless of LLM provider
        tts_key = self.openai_key or self.anthropic_key
        self.voice = VoiceIO(
            openai_api_key=self.openai_key,
            tts_model=TTS_MODEL,
            tts_voice=TTS_VOICE,
            stt_model=STT_MODEL,
            audio_device=AUDIO_DEVICE,
            logger=self.get_logger(),
        )
        self.wake_detector = WonderEchoDetector(
            port=WONDERECHO_PORT, logger=self.get_logger(),
        )
        if self.voice_on:
            self.wake_detector.start()

        # ------------------------------------------------------------------
        # Initialize data logger
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # ROS2 publishers
        # ------------------------------------------------------------------
        self.cmd_vel_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 1)
        self.servo_pub = self.create_publisher(
            SetPWMServoState, SERVO_TOPIC, 1,
        )

        # ------------------------------------------------------------------
        # ROS2 subscribers
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Status publisher (for real-time dashboard)
        # ------------------------------------------------------------------
        self.status_pub = self.create_publisher(String, STATUS_TOPIC, 1)
        self.create_subscription(
            String, '/explorer/command',
            self._command_callback, 1,
            callback_group=cb_group,
        )
        self.create_timer(
            1.0 / STATUS_PUBLISH_RATE, self._publish_status,
        )

        # ------------------------------------------------------------------
        # Motor timeout safety timer
        # ------------------------------------------------------------------
        self.create_timer(1.0, self._motor_timeout_check)

        # ------------------------------------------------------------------
        # Start background threads
        # ------------------------------------------------------------------
        self._explore_thread = threading.Thread(
            target=self._exploration_loop, daemon=True,
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

        self.get_logger().info(
            'Autonomous Explorer ready. '
            'Say "start exploring" or press Enter in terminal to begin.'
        )

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

        # Replace zeros and infinities with a large value
        cleaned = ranges.copy()
        cleaned[cleaned == 0] = float('inf')
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

        # Emergency stop check — front sector
        if sectors['front'] < self.e_stop_dist:
            if not self.emergency_stop:
                self.get_logger().warn(
                    f"EMERGENCY STOP! Obstacle at {sectors['front']:.2f}m"
                )
                self.emergency_stop = True
                self._stop_motors()
        elif self.emergency_stop and sectors['front'] > self.e_stop_dist * 1.5:
            self.get_logger().info('Front cleared, resuming...')
            self.emergency_stop = False

    def _imu_callback(self, msg: Imu):
        """Store latest IMU reading."""
        q = msg.orientation
        # Convert quaternion to Euler (simplified roll/pitch/yaw)
        sinr = 2.0 * (q.w * q.x + q.y * q.z)
        cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr, cosr)
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)

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
        """Store latest odometry reading."""
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        theta = math.atan2(siny, cosy)
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
            llm = {
                'action': self._last_llm_action,
                'speed': self._last_llm_speed,
                'duration': self._last_llm_duration,
                'reasoning': self._last_llm_reasoning,
                'speech': self._last_llm_speech,
                'response_ms': self._last_llm_response_ms,
                'tokens_in': self._last_llm_tokens_in,
                'tokens_out': self._last_llm_tokens_out,
                'cost': self._last_llm_cost,
                'timestamp': self._last_llm_time,
                'safety_triggered': self._last_safety_triggered,
                'safety_reason': self._last_safety_reason,
            }

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
        """Send a Twist command with speed limiting."""
        if self.emergency_stop and linear_x > 0:
            self.get_logger().warn('Emergency stop active — blocking forward')
            linear_x = 0.0
            angular_z = 0.0

        # Clamp speeds
        linear_x = max(-self.max_linear, min(self.max_linear, linear_x))
        angular_z = max(-self.max_angular, min(self.max_angular, angular_z))

        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.cmd_vel_pub.publish(msg)
        self._last_command_time = time.time()
        self._last_twist = (linear_x, angular_z)

    def _stop_motors(self):
        """Immediately stop all motors."""
        msg = Twist()
        self.cmd_vel_pub.publish(msg)
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
                    dist_cm = int(np.median(valid) / 10)
                    grid[f'{rname}{cname}'] = f'{dist_cm}cm'
                else:
                    grid[f'{rname}{cname}'] = 'N/A'

        # Format as a readable grid
        lines = [
            f"  {grid['topL']:>6} {grid['topC']:>6} {grid['topR']:>6}",
            f"  {grid['midL']:>6} {grid['midC']:>6} {grid['midR']:>6}",
            f"  {grid['botL']:>6} {grid['botC']:>6} {grid['botR']:>6}",
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
                parts.append(f'{name}=clear(>10m)')
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

        # Clamp duration for safety
        duration = min(duration, 5.0)
        # Scale speed to actual velocity limits
        linear_speed = speed * self.max_linear
        angular_speed = speed * self.max_angular

        # Check LiDAR safety before moving forward
        with self._lidar_lock:
            sectors = self._lidar_ranges or {}
        front_dist = sectors.get('front', float('inf'))

        if cmd in ('forward', 'investigate') and front_dist < self.caution_dist:
            self.get_logger().warn(
                f'LiDAR override: obstacle at {front_dist:.2f}m, '
                f'reducing speed'
            )
            linear_speed *= 0.3
            safety['triggered'] = True
            safety['reason'] = f'obstacle at {front_dist:.2f}m'
            if front_dist < self.e_stop_dist:
                self.get_logger().warn('LiDAR override: too close, stopping')
                self._stop_motors()
                safety['override_action'] = 'stop'
                safety['reason'] = f'obstacle too close ({front_dist:.2f}m)'
                return safety

        self.get_logger().info(
            f'Action: {cmd} speed={speed:.1f} duration={duration:.1f}s'
        )

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

        # Execute for the specified duration, checking safety continuously
        while time.time() - exec_start < duration and self.running:
            if self.emergency_stop:
                self.get_logger().warn('Emergency stop during action!')
                self._stop_motors()
                safety['triggered'] = True
                safety['reason'] = 'emergency stop during execution'
                safety['override_action'] = 'stop'
                return safety
            time.sleep(0.1)

        self._stop_motors()
        return safety

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
                self.voice.speak('Switching to manual mode.')
        else:
            self.control_mode = 'autonomous'
            self.get_logger().info(
                '>>> AUTONOMOUS MODE — press Enter or say "start" to explore'
            )
            if self.voice_on:
                self.voice.speak('Switching to autonomous mode. Say start to explore.')

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
                    self._send_twist(linear_x, angular_z)
                elif self._joystick_moving:
                    self._joystick_moving = False
                    self._stop_motors()

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
        while self.running:
            try:
                # Check for wake word
                if self.wake_detector.available:
                    if self.wake_detector.check_wakeup():
                        self.get_logger().info('Wake word detected, listening...')
                        self.voice.speak('I am listening.', block=True)
                        command = self.voice.listen_for_command(duration=5)
                        if command:
                            self._voice_queue.put(command)
                else:
                    time.sleep(1.0)
                time.sleep(0.05)
            except Exception as e:
                self.get_logger().error(f'Voice listener error: {e}')
                time.sleep(1.0)

    def _process_voice_command(self, command: str) -> bool:
        """Handle a voice command. Returns True if exploration should continue."""
        cmd = command.lower().strip()
        self.get_logger().info(f'Voice command: {cmd}')

        if any(w in cmd for w in ['manual mode', 'manual control',
                                     'take control', 'joystick']):
            if self.control_mode != 'manual':
                self._toggle_control_mode()
            else:
                self.voice.speak('Already in manual mode.')
            return False
        elif any(w in cmd for w in ['autonomous mode', 'auto mode',
                                      'automatic mode', 'autopilot']):
            if self.control_mode != 'autonomous':
                self._toggle_control_mode()
            else:
                self.voice.speak('Already in autonomous mode.')
            return self.exploring
        elif any(w in cmd for w in ['stop exploring', 'stop', 'halt', 'freeze']):
            self.exploring = False
            self._stop_motors()
            self.voice.speak('Stopping exploration. Standing by.')
            return False
        elif any(w in cmd for w in ['start exploring', 'start', 'go', 'explore']):
            if self.control_mode == 'manual':
                self.voice.speak(
                    'Cannot start exploration in manual mode. '
                    'Switch to autonomous mode first.'
                )
                return False
            self.exploring = True
            self.voice.speak('Starting exploration! Let me see what is around.')
            return True
        elif any(w in cmd for w in ['what do you see', 'what is around',
                                     'describe', 'look']):
            self.voice.speak('Let me look around and tell you what I see.')
            # The next LLM call will describe the scene
            return self.exploring
        elif any(w in cmd for w in ['go left', 'turn left']):
            self.voice.speak('Turning left.')
            self._execute_action({
                'action': 'spin_left', 'speed': 0.3, 'duration': 1.0,
            })
            return self.exploring
        elif any(w in cmd for w in ['go right', 'turn right']):
            self.voice.speak('Turning right.')
            self._execute_action({
                'action': 'spin_right', 'speed': 0.3, 'duration': 1.0,
            })
            return self.exploring
        elif any(w in cmd for w in ['go forward', 'move forward', 'go ahead']):
            self.voice.speak('Moving forward.')
            self._execute_action({
                'action': 'forward', 'speed': 0.3, 'duration': 1.5,
            })
            return self.exploring
        elif any(w in cmd for w in ['go back', 'reverse', 'back up']):
            self.voice.speak('Backing up.')
            self._execute_action({
                'action': 'backward', 'speed': 0.3, 'duration': 1.0,
            })
            return self.exploring
        else:
            self.voice.speak(f'I heard: {command}')
            return self.exploring

    # ======================================================================
    # Main Exploration Loop
    # ======================================================================

    def _exploration_loop(self):
        """Background thread: continuous sense -> think -> act -> speak cycle."""
        self.get_logger().info('Exploration loop started (waiting for start command)')
        self._center_camera()

        # Wait for first camera frame (skip in dry-run mode)
        if self.provider_name in ('dryrun', 'dry-run', 'dry_run'):
            self.get_logger().info(
                'Dry-run mode: using dummy camera frame'
            )
            # Create a small dummy image so the pipeline has something to encode
            with self._rgb_lock:
                if self._rgb_image is None:
                    self._rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        self._rgb_image, 'DRY RUN', (180, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3,
                    )
        else:
            for i in range(20):
                if not self.running:
                    return
                with self._rgb_lock:
                    has_image = self._rgb_image is not None
                if has_image:
                    break
                if i % 2 == 0:
                    self.get_logger().info('Waiting for camera feed...')
                time.sleep(0.5)
            else:
                cam_type = os.environ.get('DEPTH_CAMERA_TYPE', 'not set')
                self.get_logger().warn(
                    f'No camera frame received after 10s. '
                    f'Is the camera running? (DEPTH_CAMERA_TYPE={cam_type})'
                )

        if self._rgb_image is not None:
            self.get_logger().info('Camera feed received. Ready to explore.')
        if self.provider_name in ('dryrun', 'dry-run', 'dry_run'):
            self.get_logger().info('Dry-run mode: auto-starting exploration')
        self.exploring = True
        self.get_logger().info('Auto-starting exploration')
        if self.voice_on:
            self.voice.speak(
                'I am Jeeves, and I am ready to explore.',
            )

        while self.running:
            try:
                # Process any voice commands first
                voice_cmd = None
                while not self._voice_queue.empty():
                    voice_cmd = self._voice_queue.get_nowait()
                    self._process_voice_command(voice_cmd)

                if not self.exploring:
                    time.sleep(0.5)
                    continue

                if self.emergency_stop:
                    self.get_logger().warn('Emergency stop — waiting for clearance')
                    self._stop_motors()
                    time.sleep(0.5)
                    continue

                # ----------------------------------------------------------
                # SENSE: Gather sensor data
                # ----------------------------------------------------------
                image_b64 = self._get_camera_frame_b64()
                if not image_b64:
                    self.get_logger().warn('No camera frame available')
                    time.sleep(1.0)
                    continue

                # Snapshot raw sensor data for logging
                with self._rgb_lock:
                    rgb_snapshot = self._rgb_image.copy() if self._rgb_image is not None else None
                with self._depth_lock:
                    depth_snapshot = self._depth_image.copy() if self._depth_image is not None else None
                with self._lidar_lock:
                    lidar_sectors = self._lidar_ranges.copy() if self._lidar_ranges else None
                    lidar_raw_ranges = list(self._lidar_raw.ranges) if self._lidar_raw else None
                with self._imu_lock:
                    imu_snapshot = self._imu_data.copy() if self._imu_data else None
                with self._odom_lock:
                    odom_snapshot = self._odom_data.copy() if self._odom_data else None

                depth_summary = self._get_depth_summary()
                lidar_summary = self._get_lidar_summary()
                memory_context = self.memory.get_context_summary()

                # Build odometry summary (theta stored in radians)
                odom_summary = ''
                if odom_snapshot:
                    heading_deg = math.degrees(odom_snapshot.get('theta', 0))
                    odom_summary = (
                        f'ODOMETRY: position=({odom_snapshot.get("x", 0):.2f}, '
                        f'{odom_snapshot.get("y", 0):.2f}) m, '
                        f'heading={heading_deg:.1f}°'
                    )

                # Build IMU summary (orientation stored in radians)
                imu_summary = ''
                if imu_snapshot:
                    ori = imu_snapshot.get('orientation', {})
                    imu_summary = (
                        f'IMU: roll={math.degrees(ori.get("roll", 0)):.1f}°, '
                        f'pitch={math.degrees(ori.get("pitch", 0)):.1f}°, '
                        f'yaw={math.degrees(ori.get("yaw", 0)):.1f}°'
                    )

                user_prompt = (
                    f'Current sensor data:\n'
                    f'{lidar_summary}\n'
                    f'{depth_summary}\n'
                    f'{odom_summary}\n'
                    f'{imu_summary}\n'
                    f'\nExploration context:\n{memory_context}\n'
                    f'\nAnalyze the camera image and ALL sensor data. '
                    f'Decide your next action. Respond in JSON only.'
                )

                # Image metadata for logging
                img_h, img_w = (rgb_snapshot.shape[:2] if rgb_snapshot is not None
                                else (0, 0))
                image_size_bytes = len(image_b64) * 3 // 4  # approx decoded size

                # ----------------------------------------------------------
                # THINK: Call LLM with vision
                # ----------------------------------------------------------
                self.get_logger().info(
                    f'Calling {self.provider_name} for scene analysis...'
                )
                result = self.llm.analyze_scene(
                    image_b64, SYSTEM_PROMPT, user_prompt,
                )

                # Extract metadata from provider
                meta = result.pop('_meta', {})
                t_elapsed_ms = meta.get('response_time_ms', 0)
                tokens_in = meta.get('tokens_input', 0)
                tokens_out = meta.get('tokens_output', 0)
                raw_response = meta.get('raw_response', '')

                # Estimate cost
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

                # Update LLM state for dashboard
                self._cycle_count += 1
                self._session_total_cost += cost_usd
                with self._llm_lock:
                    self._last_llm_action = result.get('action', '')
                    self._last_llm_speed = float(result.get('speed', 0.0))
                    self._last_llm_duration = float(result.get('duration', 0.0))
                    self._last_llm_reasoning = result.get('reasoning', '')
                    self._last_llm_speech = result.get('speech', '')
                    self._last_llm_response_ms = t_elapsed_ms
                    self._last_llm_tokens_in = tokens_in
                    self._last_llm_tokens_out = tokens_out
                    self._last_llm_cost = cost_usd
                    self._last_llm_time = time.time()

                # ----------------------------------------------------------
                # SPEAK: Announce what the robot is thinking
                # ----------------------------------------------------------
                speech_text = result.get('speech', '')
                if speech_text and self.voice_on:
                    self.voice.speak(speech_text)

                # ----------------------------------------------------------
                # ACT: Execute the chosen action
                # ----------------------------------------------------------
                exec_start = time.time()
                safety_info = self._execute_action(result)
                exec_duration_ms = int((time.time() - exec_start) * 1000)

                actual_action = safety_info.get('override_action',
                                                result.get('action', 'stop'))

                # Update safety state for dashboard
                with self._llm_lock:
                    self._last_safety_triggered = safety_info.get(
                        'triggered', False)
                    self._last_safety_reason = safety_info.get('reason', '')

                # ----------------------------------------------------------
                # REMEMBER: Log the decision
                # ----------------------------------------------------------
                self.memory.record_action(
                    result, lidar_summary,
                    odom=odom_snapshot,
                    lidar_sectors=lidar_sectors,
                )

                # Log reasoning
                reasoning = result.get('reasoning', '')
                if reasoning:
                    self.get_logger().info(f'Reasoning: {reasoning[:120]}')

                # ----------------------------------------------------------
                # LOG: Full cycle record for dataset collection
                # ----------------------------------------------------------
                self.data_logger.log_cycle(
                    # Sensor data
                    rgb_image=rgb_snapshot,
                    depth_image=depth_snapshot,
                    lidar_ranges=lidar_raw_ranges,
                    lidar_sectors=lidar_sectors,
                    imu_data=imu_snapshot,
                    odom_data=odom_snapshot,
                    battery_voltage=self._battery_voltage,
                    # LLM I/O
                    provider=self.provider_name,
                    model=meta.get('model', ''),
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    image_resolution=f'{img_w}x{img_h}',
                    image_size_bytes=image_size_bytes,
                    raw_response=raw_response,
                    parsed_action=result.get('action', ''),
                    speed=result.get('speed', 0.0),
                    duration=result.get('duration', 0.0),
                    speech=speech_text,
                    reasoning=reasoning,
                    response_time_ms=t_elapsed_ms,
                    tokens_input=tokens_in,
                    tokens_output=tokens_out,
                    cost_usd=cost_usd,
                    # Safety
                    safety_triggered=safety_info.get('triggered', False),
                    safety_reason=safety_info.get('reason', ''),
                    safety_original_action=safety_info.get('original_action', ''),
                    safety_override_action=safety_info.get('override_action', ''),
                    # Execution
                    actual_action=actual_action,
                    motor_left_speed=self._last_twist[0],
                    motor_right_speed=self._last_twist[1],
                    servo_pan=self._last_servo_pan,
                    servo_tilt=self._last_servo_tilt,
                    execution_duration_ms=exec_duration_ms,
                    # Voice
                    voice_command=voice_cmd,
                    speech_output=speech_text,
                    # Exploration memory
                    total_distance=0.0,  # TODO: integrate odometry distance
                    areas_visited=self.memory.total_actions,
                    objects_discovered=[
                        d['description'][:60]
                        for d in self.memory.discoveries[-20:]
                    ],
                    map_coverage_pct=0.0,
                )

                # ----------------------------------------------------------
                # Wait before next cycle
                # ----------------------------------------------------------
                time.sleep(self.loop_interval)

            except Exception as e:
                self.get_logger().error(f'Exploration loop error: {e}')
                self._stop_motors()
                time.sleep(2.0)

    # ======================================================================
    # Shutdown
    # ======================================================================

    def shutdown(self):
        """Graceful shutdown."""
        self.get_logger().info('Shutting down Autonomous Explorer...')
        self.running = False
        self.exploring = False
        self._stop_motors()
        self._center_camera()
        self._joystick.stop()
        self.memory.save()
        self.data_logger.stop()
        self.wake_detector.stop()
        if self.voice_on:
            self.voice.speak('Exploration complete. Shutting down.', block=True)
        self.get_logger().info('Shutdown complete.')


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

    # Start a thread for terminal input (start/stop via keyboard)
    def stdin_listener():
        while node.running:
            try:
                line = input()  # Blocks until Enter
                line = line.strip().lower()
                if line in ('q', 'quit', 'exit'):
                    node.shutdown()
                    rclpy.shutdown()
                    return
                elif line in ('s', 'stop'):
                    node.exploring = False
                    node._stop_motors()
                    node.get_logger().info('Exploration paused (keyboard)')
                elif line in ('m', 'manual'):
                    if node.control_mode != 'manual':
                        node._toggle_control_mode()
                    else:
                        node.get_logger().info('Already in manual mode')
                elif line in ('a', 'auto', 'autonomous'):
                    if node.control_mode != 'autonomous':
                        node._toggle_control_mode()
                    else:
                        node.get_logger().info('Already in autonomous mode')
                elif line in ('', 'go', 'start'):
                    if node.control_mode == 'manual':
                        node.get_logger().info(
                            'Cannot start exploration in manual mode. '
                            'Press "a" to switch to autonomous first.'
                        )
                    else:
                        node.exploring = True
                        node.get_logger().info('Exploration started (keyboard)')
                elif line == 'status':
                    js_status = (
                        f'connected ({node._joystick.joystick_name})'
                        if node._joystick.connected
                        else 'disconnected'
                    )
                    node.get_logger().info(
                        f'Mode: {node.control_mode} | '
                        f'Exploring: {node.exploring} | '
                        f'E-stop: {node.emergency_stop} | '
                        f'Joystick: {js_status} | '
                        f'Provider: {node.provider_name} | '
                        f'Actions: {node.memory.total_actions}'
                    )
                else:
                    node.get_logger().info(
                        'Commands: Enter=start, s=stop, m=manual, '
                        'a=auto, q=quit, status'
                    )
            except EOFError:
                break

    stdin_thread = threading.Thread(target=stdin_listener, daemon=True)
    stdin_thread.start()

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
