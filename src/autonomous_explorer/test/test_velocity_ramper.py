"""Tests for autonomous_explorer.velocity_ramper module."""
import sys
import time
from unittest.mock import MagicMock, call
from types import ModuleType

import pytest

# Mock geometry_msgs if not available (dev machine without ROS2)
if 'geometry_msgs' not in sys.modules:
    _geo = ModuleType('geometry_msgs')
    _msg = ModuleType('geometry_msgs.msg')

    class _FakeTwist:
        def __init__(self):
            self.linear = MagicMock(x=0.0)
            self.angular = MagicMock(z=0.0)

    _msg.Twist = _FakeTwist
    _geo.msg = _msg
    sys.modules['geometry_msgs'] = _geo
    sys.modules['geometry_msgs.msg'] = _msg

from autonomous_explorer.velocity_ramper import VelocityRamper


@pytest.fixture
def mock_publisher():
    """Mock ROS2 publisher for Twist messages."""
    return MagicMock()


@pytest.fixture
def ramper(mock_publisher):
    """Create a VelocityRamper with fast update rate for testing."""
    return VelocityRamper(
        publisher=mock_publisher,
        max_linear=0.35,
        max_angular=1.0,
        linear_accel=0.5,
        linear_decel=0.8,
        angular_accel=2.0,
        angular_decel=3.0,
        update_rate=100.0,  # fast for tests
    )


# ===================================================================
# _ramp function (core algorithm)
# ===================================================================

class TestRampFunction:
    """Test the trapezoidal velocity ramping algorithm."""

    def test_ramp_up_from_zero(self, ramper):
        result = ramper._ramp(0.0, 0.2, accel=0.5, decel=0.8)
        assert result > 0.0
        assert result <= 0.2

    def test_ramp_down_to_zero(self, ramper):
        result = ramper._ramp(0.2, 0.0, accel=0.5, decel=0.8)
        assert result < 0.2
        assert result >= 0.0

    def test_at_target_returns_target(self, ramper):
        result = ramper._ramp(0.2, 0.2, accel=0.5, decel=0.8)
        assert abs(result - 0.2) < 0.002

    def test_small_diff_snaps_to_target(self, ramper):
        result = ramper._ramp(0.1999, 0.2, accel=0.5, decel=0.8)
        assert result == 0.2

    def test_decel_when_approaching_zero(self, ramper):
        # Moving toward zero uses decel rate (faster)
        decel_step = ramper._ramp(0.2, 0.0, accel=0.5, decel=0.8)
        accel_step = ramper._ramp(0.0, 0.2, accel=0.5, decel=0.8)
        # Decel makes bigger steps than accel at same dt
        assert (0.2 - decel_step) >= accel_step - 0.001

    def test_negative_velocity(self, ramper):
        result = ramper._ramp(0.0, -0.2, accel=0.5, decel=0.8)
        assert result < 0.0

    def test_direction_reversal_uses_decel(self, ramper):
        # Going from positive to negative should use decel rate
        result = ramper._ramp(0.1, -0.1, accel=0.5, decel=0.8)
        assert result < 0.1


# ===================================================================
# set_target and clamping
# ===================================================================

class TestSetTarget:
    """Test target velocity setting and speed clamping."""

    def test_set_target(self, ramper):
        ramper.set_target(0.2, 0.5)
        assert ramper.target_velocity == (0.2, 0.5)

    def test_clamps_linear_max(self, ramper):
        ramper.set_target(1.0, 0.0)  # exceeds max_linear=0.35
        target_lin, _ = ramper.target_velocity
        assert target_lin <= 0.35

    def test_clamps_linear_min(self, ramper):
        ramper.set_target(-1.0, 0.0)
        target_lin, _ = ramper.target_velocity
        assert target_lin >= -0.35

    def test_clamps_angular_max(self, ramper):
        ramper.set_target(0.0, 5.0)  # exceeds max_angular=1.0
        _, target_ang = ramper.target_velocity
        assert target_ang <= 1.0

    def test_clamps_angular_min(self, ramper):
        ramper.set_target(0.0, -5.0)
        _, target_ang = ramper.target_velocity
        assert target_ang >= -1.0

    def test_clears_estop_flag(self, ramper):
        ramper.emergency_stop()
        ramper.set_target(0.1, 0.0)
        assert ramper._e_stopped is False


# ===================================================================
# stop and emergency_stop
# ===================================================================

class TestStopMethods:
    """Test smooth and emergency stop."""

    def test_stop_sets_zero_target(self, ramper):
        ramper.set_target(0.2, 0.5)
        ramper.stop()
        assert ramper.target_velocity == (0.0, 0.0)

    def test_emergency_stop_zeros_immediately(self, ramper, mock_publisher):
        ramper.set_target(0.3, 0.5)
        ramper._current_linear = 0.3
        ramper._current_angular = 0.5
        ramper.emergency_stop()
        assert ramper.current_velocity == (0.0, 0.0)
        assert ramper.target_velocity == (0.0, 0.0)
        assert ramper._e_stopped is True
        # Should have published a zero Twist
        mock_publisher.publish.assert_called()

    def test_block_forward_zeros_positive_linear(self, ramper):
        ramper._target_linear = 0.3
        ramper._current_linear = 0.2
        ramper.block_forward()
        assert ramper._target_linear == 0.0
        assert ramper._current_linear == 0.0

    def test_block_forward_preserves_backward(self, ramper):
        ramper._target_linear = -0.2
        ramper._current_linear = -0.1
        ramper.block_forward()
        assert ramper._target_linear == -0.2
        assert ramper._current_linear == -0.1


# ===================================================================
# Properties
# ===================================================================

class TestProperties:
    """Test velocity ramper properties."""

    def test_is_moving_when_nonzero(self, ramper):
        ramper._current_linear = 0.1
        assert ramper.is_moving is True

    def test_not_moving_when_zero(self, ramper):
        assert ramper.is_moving is False

    def test_current_velocity_initially_zero(self, ramper):
        assert ramper.current_velocity == (0.0, 0.0)


# ===================================================================
# Background thread lifecycle
# ===================================================================

class TestLifecycle:
    """Test start/shutdown of the background ramping thread."""

    def test_start_creates_thread(self, ramper):
        ramper.start()
        assert ramper._running is True
        assert ramper._thread is not None
        assert ramper._thread.is_alive()
        ramper.shutdown()

    def test_shutdown_stops_thread(self, ramper):
        ramper.start()
        ramper.shutdown()
        assert ramper._running is False

    def test_shutdown_publishes_zero(self, ramper, mock_publisher):
        ramper.start()
        ramper.shutdown()
        # Last call should be zero Twist
        last_msg = mock_publisher.publish.call_args[0][0]
        assert last_msg.linear.x == 0.0
        assert last_msg.angular.z == 0.0

    def test_double_start_safe(self, ramper):
        ramper.start()
        ramper.start()  # Should not create second thread
        ramper.shutdown()

    def test_ramping_publishes_messages(self, ramper, mock_publisher):
        ramper.start()
        ramper.set_target(0.2, 0.0)
        time.sleep(0.1)  # let the loop run a few cycles
        ramper.shutdown()
        assert mock_publisher.publish.call_count > 0
