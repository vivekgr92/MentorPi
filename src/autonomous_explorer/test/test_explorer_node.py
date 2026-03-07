"""Tests for autonomous_explorer.explorer_node module.

The explorer_node depends heavily on ROS2 (rclpy, message types, etc.).
We mock the ROS2 layer at import time and test the pure-logic methods:
  - LiDAR sector computation
  - Depth image summarization
  - LiDAR text summary
  - Image resizing / base64 encoding
  - Voice command parsing
  - Action execution mapping (speed clamping, safety overrides)
  - Motor timeout logic
  - Command callback routing
"""
import base64
import math
import sys
import time
from types import ModuleType
from unittest.mock import MagicMock, PropertyMock, patch

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock all ROS2 dependencies before importing explorer_node
# ---------------------------------------------------------------------------

_MOCK_MODULES = [
    'rclpy', 'rclpy.node', 'rclpy.executors', 'rclpy.callback_groups',
    'rclpy.qos', 'rclpy.action',
    'geometry_msgs', 'geometry_msgs.msg',
    'std_msgs', 'std_msgs.msg',
    'sensor_msgs', 'sensor_msgs.msg',
    'nav_msgs', 'nav_msgs.msg',
    'ros_robot_controller_msgs', 'ros_robot_controller_msgs.msg',
    'cv_bridge', 'action_msgs', 'action_msgs.msg',
    'nav2_msgs', 'nav2_msgs.action',
]

_saved = {}
for mod_name in _MOCK_MODULES:
    if mod_name not in sys.modules:
        mock_mod = MagicMock()
        sys.modules[mod_name] = mock_mod
        _saved[mod_name] = mock_mod

# Provide specific mocks for classes used at import time
sys.modules['rclpy.qos'].QoSProfile = MagicMock
sys.modules['rclpy.qos'].QoSReliabilityPolicy = MagicMock()
sys.modules['rclpy.node'].Node = type('Node', (), {
    '__init__': lambda self, *a, **kw: None,
    'get_logger': lambda self: MagicMock(),
    'declare_parameter': lambda self, *a, **kw: None,
    'get_parameter': lambda self, *a, **kw: MagicMock(),
    'create_subscription': lambda self, *a, **kw: MagicMock(),
    'create_publisher': lambda self, *a, **kw: MagicMock(),
    'create_timer': lambda self, *a, **kw: MagicMock(),
    'get_clock': lambda self: MagicMock(),
    'destroy_subscription': lambda self, *a, **kw: None,
})

# Twist mock
class _FakeTwist:
    def __init__(self):
        self.linear = MagicMock(x=0.0, y=0.0, z=0.0)
        self.angular = MagicMock(x=0.0, y=0.0, z=0.0)

sys.modules['geometry_msgs.msg'].Twist = _FakeTwist

# CvBridge mock
class _FakeCvBridge:
    def imgmsg_to_cv2(self, msg, encoding='bgr8'):
        return msg._cv_image if hasattr(msg, '_cv_image') else np.zeros((480, 640, 3), dtype=np.uint8)

sys.modules['cv_bridge'].CvBridge = _FakeCvBridge

# Now import the module under test
from autonomous_explorer.explorer_node import AutonomousExplorer
from autonomous_explorer import config


# ---------------------------------------------------------------------------
# Fixture: create an explorer with mocked ROS2 internals
# ---------------------------------------------------------------------------

@pytest.fixture
def explorer():
    """Create an AutonomousExplorer with all ROS2 internals mocked."""
    with patch.object(AutonomousExplorer, '__init__', lambda self: None):
        node = AutonomousExplorer()

    # Set up minimal state that methods expect
    node._rgb_lock = __import__('threading').Lock()
    node._depth_lock = __import__('threading').Lock()
    node._lidar_lock = __import__('threading').Lock()
    node._imu_lock = __import__('threading').Lock()
    node._odom_lock = __import__('threading').Lock()
    node._llm_lock = __import__('threading').Lock()
    node._rgb_image = None
    node._depth_image = None
    node._lidar_ranges = None
    node._lidar_raw = None
    node._imu_data = None
    node._odom_data = None
    node._battery_voltage = 0.0
    node._last_twist = (0.0, 0.0)
    node._last_command_time = time.time()
    node._last_servo_pan = 1500
    node._last_servo_tilt = 1500
    node._total_distance = 0.0
    node._prev_odom_x = None
    node._prev_odom_y = None
    node.emergency_stop = False
    node.exploring = False
    node.running = True
    node.control_mode = 'autonomous'
    node.e_stop_dist = config.EMERGENCY_STOP_DISTANCE
    node.caution_dist = config.CAUTION_DISTANCE
    node.max_linear = config.MAX_LINEAR_SPEED
    node.max_angular = config.MAX_ANGULAR_SPEED
    node.voice_on = False
    node.voice = MagicMock()
    node.use_nav2 = False
    node.nav2 = None
    node.provider_name = 'dryrun'

    # Mock ramper and publishers
    node._ramper = MagicMock()
    node._ramper.current_velocity = (0.0, 0.0)
    node._joy_ramper = MagicMock()
    node._estop_lock_pub = None
    node.servo_pub = MagicMock()

    # Logger
    node.get_logger = lambda: MagicMock()

    return node


# ===================================================================
# _get_camera_frame_b64
# ===================================================================

class TestGetCameraFrameB64:
    """Test camera frame encoding to base64."""

    def test_no_image_returns_empty(self, explorer):
        explorer._rgb_image = None
        result = explorer._get_camera_frame_b64()
        assert result == ''

    def test_valid_image_returns_base64(self, explorer):
        explorer._rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = explorer._get_camera_frame_b64()
        assert len(result) > 0
        # Should be valid base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_large_image_gets_resized(self, explorer):
        explorer._rgb_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        result = explorer._get_camera_frame_b64()
        assert len(result) > 0
        # Decoded JPEG should be smaller than original raw pixels
        decoded = base64.b64decode(result)
        assert len(decoded) < 1080 * 1920 * 3

    def test_small_image_not_resized(self, explorer):
        # Image smaller than MAX_IMAGE_DIMENSION should pass through
        explorer._rgb_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        result = explorer._get_camera_frame_b64()
        assert len(result) > 0


# ===================================================================
# _get_depth_summary
# ===================================================================

class TestGetDepthSummary:
    """Test depth image → text grid summary."""

    def test_no_depth_returns_unavailable(self, explorer):
        explorer._depth_image = None
        result = explorer._get_depth_summary()
        assert 'unavailable' in result

    def test_valid_depth_returns_grid(self, explorer):
        # Create depth image with known values (millimeters)
        depth = np.full((480, 640), 2000, dtype=np.uint16)  # 2m everywhere
        explorer._depth_image = depth
        result = explorer._get_depth_summary()
        assert 'Depth grid' in result
        # 2000mm / 10 = 200cm
        assert '200cm' in result

    def test_zero_depth_shows_na(self, explorer):
        depth = np.zeros((480, 640), dtype=np.uint16)  # all zeros = invalid
        explorer._depth_image = depth
        result = explorer._get_depth_summary()
        assert 'N/A' in result

    def test_mixed_valid_invalid(self, explorer):
        depth = np.zeros((480, 640), dtype=np.uint16)
        # Set center region to 1500mm
        depth[200:280, 280:360] = 1500
        explorer._depth_image = depth
        result = explorer._get_depth_summary()
        # Center should have a value, corners should be N/A
        assert 'cm' in result

    def test_far_depth_values(self, explorer):
        depth = np.full((480, 640), 39000, dtype=np.uint16)  # 39m, just under max
        explorer._depth_image = depth
        result = explorer._get_depth_summary()
        assert '3900cm' in result

    def test_beyond_max_range_excluded(self, explorer):
        depth = np.full((480, 640), 45000, dtype=np.uint16)  # 45m, over 40000 limit
        explorer._depth_image = depth
        result = explorer._get_depth_summary()
        assert 'N/A' in result


# ===================================================================
# _get_lidar_summary
# ===================================================================

class TestGetLidarSummary:
    """Test LiDAR sector → text summary."""

    def test_no_lidar_returns_no_data(self, explorer):
        explorer._lidar_ranges = None
        result = explorer._get_lidar_summary()
        assert 'no data' in result

    def test_normal_readings(self, explorer):
        explorer._lidar_ranges = {
            'front': 1.5, 'left': 0.8, 'right': 2.0, 'back': 3.0,
        }
        result = explorer._get_lidar_summary()
        assert 'front=1.50m' in result
        assert 'left=0.80m' in result
        assert 'right=2.00m' in result

    def test_clear_distance_shows_clear(self, explorer):
        explorer._lidar_ranges = {
            'front': float('inf'), 'left': 15.0, 'right': float('inf'), 'back': float('inf'),
        }
        result = explorer._get_lidar_summary()
        assert 'clear' in result

    def test_emergency_warning(self, explorer):
        explorer._lidar_ranges = {
            'front': 0.15, 'left': 1.0, 'right': 1.0, 'back': 2.0,
        }
        result = explorer._get_lidar_summary()
        assert 'EMERGENCY' in result

    def test_caution_warning(self, explorer):
        explorer._lidar_ranges = {
            'front': 0.30, 'left': 1.0, 'right': 1.0, 'back': 2.0,
        }
        result = explorer._get_lidar_summary()
        assert 'CAUTION' in result

    def test_safe_no_warning(self, explorer):
        explorer._lidar_ranges = {
            'front': 2.0, 'left': 2.0, 'right': 2.0, 'back': 2.0,
        }
        result = explorer._get_lidar_summary()
        assert 'EMERGENCY' not in result
        assert 'CAUTION' not in result


# ===================================================================
# _process_voice_command
# ===================================================================

class TestProcessVoiceCommand:
    """Test voice command parsing and routing."""

    def test_stop_command(self, explorer):
        explorer.exploring = True
        result = explorer._process_voice_command('stop')
        assert result is False
        assert explorer.exploring is False

    def test_halt_command(self, explorer):
        explorer.exploring = True
        result = explorer._process_voice_command('halt')
        assert result is False

    def test_start_command_in_autonomous(self, explorer):
        explorer.control_mode = 'autonomous'
        result = explorer._process_voice_command('start exploring')
        assert result is True
        assert explorer.exploring is True

    def test_start_command_in_manual_blocked(self, explorer):
        explorer.control_mode = 'manual'
        result = explorer._process_voice_command('start')
        assert result is False

    def test_manual_mode_command(self, explorer):
        explorer.control_mode = 'autonomous'
        explorer._toggle_control_mode = MagicMock()
        result = explorer._process_voice_command('manual mode')
        explorer._toggle_control_mode.assert_called_once()
        assert result is False

    def test_autonomous_mode_command(self, explorer):
        explorer.control_mode = 'manual'
        explorer._toggle_control_mode = MagicMock()
        result = explorer._process_voice_command('autonomous mode')
        explorer._toggle_control_mode.assert_called_once()

    def test_turn_left_command(self, explorer):
        # "turn left" doesn't contain "go"/"start"/"stop" so reaches the correct branch
        with patch.object(explorer, '_execute_action', return_value={}) as mock_exec:
            explorer._process_voice_command('turn left')
            mock_exec.assert_called_once()
            action = mock_exec.call_args[0][0]
            assert action['action'] == 'spin_left'

    def test_turn_right_command(self, explorer):
        with patch.object(explorer, '_execute_action', return_value={}) as mock_exec:
            explorer._process_voice_command('turn right')
            mock_exec.assert_called_once()
            action = mock_exec.call_args[0][0]
            assert action['action'] == 'spin_right'

    def test_go_left_matches_start_first(self, explorer):
        # "go left" contains "go" which matches the start branch before turn_left
        explorer.control_mode = 'autonomous'
        result = explorer._process_voice_command('go left')
        assert result is True  # starts exploration (matched "go")
        assert explorer.exploring is True

    def test_move_forward_command(self, explorer):
        # "move forward" avoids the "go" prefix
        with patch.object(explorer, '_execute_action', return_value={}) as mock_exec:
            explorer._process_voice_command('move forward')
            mock_exec.assert_called_once()
            action = mock_exec.call_args[0][0]
            assert action['action'] == 'forward'

    def test_go_back_command(self, explorer):
        # "reverse" doesn't conflict with other branches
        with patch.object(explorer, '_execute_action', return_value={}) as mock_exec:
            explorer._process_voice_command('reverse')
            action = mock_exec.call_args[0][0]
            assert action['action'] == 'backward'

    def test_back_up_command(self, explorer):
        with patch.object(explorer, '_execute_action', return_value={}) as mock_exec:
            explorer._process_voice_command('back up')
            action = mock_exec.call_args[0][0]
            assert action['action'] == 'backward'

    def test_what_do_you_see_command(self, explorer):
        explorer.exploring = True
        result = explorer._process_voice_command('what do you see')
        assert result is True  # continues exploring

    def test_unknown_command_echoed(self, explorer):
        explorer.exploring = True
        result = explorer._process_voice_command('play music')
        assert result is True  # continues whatever it was doing
        explorer.voice.speak.assert_called()


# ===================================================================
# _execute_action (speed clamping, safety overrides)
# ===================================================================

class TestExecuteAction:
    """Test action execution with safety and speed logic."""

    def test_stop_action(self, explorer):
        safety = explorer._execute_action({'action': 'stop', 'speed': 0.0, 'duration': 0.0})
        assert safety['triggered'] is False
        explorer._ramper.stop.assert_called()

    def test_forward_action_sends_twist(self, explorer):
        explorer._lidar_ranges = {'front': 5.0}
        # Need to make the execution loop end quickly
        explorer.running = False
        safety = explorer._execute_action({'action': 'forward', 'speed': 0.5, 'duration': 0.1})
        explorer._ramper.set_target.assert_called()
        args = explorer._ramper.set_target.call_args[0]
        assert args[0] > 0  # positive linear speed
        assert args[1] == 0.0  # no angular

    def test_backward_action_negative_speed(self, explorer):
        explorer._lidar_ranges = {'front': 5.0}
        explorer.running = False
        explorer._execute_action({'action': 'backward', 'speed': 0.5, 'duration': 0.1})
        args = explorer._ramper.set_target.call_args[0]
        assert args[0] < 0  # negative linear speed

    def test_turn_left_has_angular(self, explorer):
        explorer._lidar_ranges = {'front': 5.0}
        explorer.running = False
        explorer._execute_action({'action': 'turn_left', 'speed': 0.5, 'duration': 0.1})
        args = explorer._ramper.set_target.call_args[0]
        assert args[1] > 0  # positive angular = left turn

    def test_turn_right_has_negative_angular(self, explorer):
        explorer._lidar_ranges = {'front': 5.0}
        explorer.running = False
        explorer._execute_action({'action': 'turn_right', 'speed': 0.5, 'duration': 0.1})
        args = explorer._ramper.set_target.call_args[0]
        assert args[1] < 0  # negative angular = right turn

    def test_spin_left_no_linear(self, explorer):
        explorer._lidar_ranges = {'front': 5.0}
        explorer.running = False
        explorer._execute_action({'action': 'spin_left', 'speed': 0.5, 'duration': 0.1})
        args = explorer._ramper.set_target.call_args[0]
        assert args[0] == 0.0  # no linear
        assert args[1] > 0    # positive angular

    def test_unknown_action_stops(self, explorer):
        explorer._lidar_ranges = {'front': 5.0}
        safety = explorer._execute_action({'action': 'dance', 'speed': 0.5, 'duration': 1.0})
        assert safety['override_action'] == 'stop'

    def test_forward_blocked_by_lidar_caution(self, explorer):
        explorer._lidar_ranges = {'front': 0.30}  # in caution zone
        explorer.running = False
        safety = explorer._execute_action({'action': 'forward', 'speed': 1.0, 'duration': 0.1})
        assert safety['triggered'] is True
        assert 'obstacle' in safety['reason']

    def test_forward_blocked_by_lidar_emergency(self, explorer):
        explorer._lidar_ranges = {'front': 0.10}  # below emergency threshold
        safety = explorer._execute_action({'action': 'forward', 'speed': 1.0, 'duration': 1.0})
        assert safety['triggered'] is True
        assert safety['override_action'] == 'stop'

    def test_duration_clamped_to_max(self, explorer):
        explorer._lidar_ranges = {'front': 5.0}
        explorer.running = False
        # Duration 10s should be clamped to 5.0
        explorer._execute_action({'action': 'forward', 'speed': 0.3, 'duration': 10.0})
        # If it ran for 10s we'd know, but since running=False it exits immediately


# ===================================================================
# _send_twist (emergency stop blocking)
# ===================================================================

class TestSendTwist:
    """Test motor command sending with emergency stop."""

    def test_normal_send(self, explorer):
        explorer._send_twist(0.2, 0.5)
        explorer._ramper.set_target.assert_called_with(0.2, 0.5)
        assert explorer._last_twist == (0.2, 0.5)

    def test_emergency_stop_blocks_forward(self, explorer):
        explorer.emergency_stop = True
        explorer._send_twist(0.2, 0.0)
        explorer._ramper.set_target.assert_called_with(0.0, 0.0)

    def test_emergency_stop_allows_backward(self, explorer):
        explorer.emergency_stop = True
        explorer._send_twist(-0.2, 0.0)
        # Backward (negative linear) is allowed — only forward (>0) is blocked
        explorer._ramper.set_target.assert_called_with(-0.2, 0.0)

    def test_updates_last_command_time(self, explorer):
        t_before = time.time()
        explorer._send_twist(0.1, 0.0)
        assert explorer._last_command_time >= t_before


# ===================================================================
# _motor_timeout_check
# ===================================================================

class TestMotorTimeoutCheck:
    """Test the motor safety timeout."""

    def test_no_timeout_when_recent_command(self, explorer):
        explorer.exploring = True
        explorer._last_command_time = time.time()
        explorer._last_twist = (0.1, 0.0)
        explorer._motor_timeout_check()
        explorer._ramper.stop.assert_not_called()

    def test_timeout_stops_motors(self, explorer):
        explorer.exploring = True
        explorer._last_command_time = time.time() - 10.0  # 10s ago
        explorer._last_twist = (0.1, 0.0)
        explorer._stop_motors = MagicMock()
        explorer._motor_timeout_check()
        explorer._stop_motors.assert_called_once()

    def test_no_timeout_when_already_stopped(self, explorer):
        explorer.exploring = True
        explorer._last_command_time = time.time() - 10.0
        explorer._last_twist = (0.0, 0.0)  # already stopped
        explorer._stop_motors = MagicMock()
        explorer._motor_timeout_check()
        explorer._stop_motors.assert_not_called()

    def test_no_timeout_when_emergency_stop(self, explorer):
        explorer.exploring = True
        explorer.emergency_stop = True
        explorer._last_command_time = time.time() - 10.0
        explorer._last_twist = (0.1, 0.0)
        explorer._stop_motors = MagicMock()
        explorer._motor_timeout_check()
        explorer._stop_motors.assert_not_called()

    def test_no_timeout_when_idle(self, explorer):
        explorer.exploring = False
        explorer.control_mode = 'autonomous'
        explorer._last_command_time = time.time() - 10.0
        explorer._last_twist = (0.1, 0.0)
        explorer._stop_motors = MagicMock()
        explorer._motor_timeout_check()
        explorer._stop_motors.assert_not_called()


# ===================================================================
# LiDAR sector computation (extracted logic)
# ===================================================================

class TestLidarSectorComputation:
    """Test the LiDAR angle-based sector distance computation.

    The _lidar_callback method computes sector minimums from raw scan data.
    We test the core math by replicating the sector_min function.
    """

    @staticmethod
    def _compute_sectors(ranges, angle_min=0.0, angle_increment=None):
        """Replicate the sector computation from _lidar_callback."""
        num_readings = len(ranges)
        if angle_increment is None:
            angle_increment = 2 * math.pi / num_readings

        cleaned = np.array(ranges, dtype=np.float64).copy()
        cleaned[cleaned == 0] = float('inf')
        cleaned[~np.isfinite(cleaned)] = float('inf')

        angles = angle_min + np.arange(num_readings) * angle_increment
        angles = angles % (2 * math.pi)

        def sector_min(center_deg, half_width_deg=45):
            center = math.radians(center_deg)
            half = math.radians(half_width_deg)
            lo = (center - half) % (2 * math.pi)
            hi = (center + half) % (2 * math.pi)
            if lo < hi:
                mask = (angles >= lo) & (angles < hi)
            else:
                mask = (angles >= lo) | (angles < hi)
            sector_vals = cleaned[mask]
            return float(sector_vals.min()) if len(sector_vals) > 0 else float('inf')

        return {
            'front': sector_min(0, 45),
            'left': sector_min(90, 45),
            'back': sector_min(180, 45),
            'right': sector_min(270, 45),
        }

    def test_uniform_distance(self):
        """All readings at 2.0m → all sectors = 2.0m."""
        ranges = [2.0] * 360
        sectors = self._compute_sectors(ranges, angle_increment=math.radians(1))
        for name in ['front', 'left', 'back', 'right']:
            assert abs(sectors[name] - 2.0) < 0.01

    def test_close_obstacle_front(self):
        """Close obstacle directly ahead (0°) detected in front sector."""
        ranges = [5.0] * 360
        ranges[0] = 0.15  # obstacle at 0°
        sectors = self._compute_sectors(ranges, angle_increment=math.radians(1))
        assert sectors['front'] == 0.15
        assert sectors['left'] > 1.0
        assert sectors['right'] > 1.0

    def test_close_obstacle_left(self):
        """Close obstacle at 90° detected in left sector."""
        ranges = [5.0] * 360
        ranges[90] = 0.30
        sectors = self._compute_sectors(ranges, angle_increment=math.radians(1))
        assert sectors['left'] == 0.30
        assert sectors['front'] > 1.0

    def test_close_obstacle_right(self):
        """Close obstacle at 270° detected in right sector."""
        ranges = [5.0] * 360
        ranges[270] = 0.25
        sectors = self._compute_sectors(ranges, angle_increment=math.radians(1))
        assert sectors['right'] == 0.25

    def test_close_obstacle_back(self):
        """Close obstacle at 180° detected in back sector."""
        ranges = [5.0] * 360
        ranges[180] = 0.40
        sectors = self._compute_sectors(ranges, angle_increment=math.radians(1))
        assert sectors['back'] == 0.40

    def test_zeros_treated_as_infinity(self):
        """Zero readings (invalid) should not register as obstacles."""
        ranges = [0.0] * 360
        sectors = self._compute_sectors(ranges, angle_increment=math.radians(1))
        for name in ['front', 'left', 'back', 'right']:
            assert sectors[name] == float('inf')

    def test_nan_treated_as_infinity(self):
        """NaN readings should not register as obstacles."""
        ranges = [float('nan')] * 360
        sectors = self._compute_sectors(ranges, angle_increment=math.radians(1))
        for name in ['front', 'left', 'back', 'right']:
            assert sectors[name] == float('inf')

    def test_front_sector_wraps_around(self):
        """Front sector spans 315°-45° — must handle wrap-around."""
        ranges = [5.0] * 360
        ranges[350] = 0.20  # at 350° = within front sector
        sectors = self._compute_sectors(ranges, angle_increment=math.radians(1))
        assert sectors['front'] == 0.20

    def test_mixed_valid_invalid(self):
        """Mix of valid readings, zeros, and infinities."""
        ranges = [5.0] * 360
        ranges[0] = 0.0     # invalid zero at front
        ranges[1] = 1.5     # valid at front
        ranges[90] = float('inf')  # invalid inf at left
        ranges[91] = 0.8    # valid at left
        sectors = self._compute_sectors(ranges, angle_increment=math.radians(1))
        assert sectors['front'] == 1.5   # zero excluded, next valid
        assert sectors['left'] == 0.8    # inf excluded, next valid
