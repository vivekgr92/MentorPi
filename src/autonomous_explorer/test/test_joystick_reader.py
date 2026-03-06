"""Tests for autonomous_explorer.joystick_reader module."""
from unittest.mock import MagicMock, patch

import pytest

from autonomous_explorer.joystick_reader import (
    JoystickReader,
    DEADZONE,
    _BUTTONS_SHANWAN,
    _BUTTONS_WIRELESS,
    _AXIS_NAMES,
)


# ===================================================================
# Button maps
# ===================================================================

class TestButtonMaps:
    """Verify button map integrity."""

    def test_shanwan_has_start_and_select(self):
        assert 'start' in _BUTTONS_SHANWAN
        assert 'select' in _BUTTONS_SHANWAN

    def test_wireless_has_start_and_select(self):
        assert 'start' in _BUTTONS_WIRELESS
        assert 'select' in _BUTTONS_WIRELESS

    def test_axis_names_count(self):
        assert len(_AXIS_NAMES) == 4
        assert 'left_x' in _AXIS_NAMES
        assert 'left_y' in _AXIS_NAMES
        assert 'right_x' in _AXIS_NAMES
        assert 'right_y' in _AXIS_NAMES


# ===================================================================
# JoystickReader initialization
# ===================================================================

class TestJoystickReaderInit:
    """Test JoystickReader construction."""

    def test_initial_state(self):
        reader = JoystickReader()
        assert reader.connected is False
        assert reader.joystick_name == ''

    def test_initial_axes_zero(self):
        reader = JoystickReader()
        axes = reader.axes
        for name in _AXIS_NAMES:
            assert axes[name] == 0.0
        assert axes['hat_x'] == 0.0
        assert axes['hat_y'] == 0.0

    def test_axes_returns_copy(self):
        reader = JoystickReader()
        a1 = reader.axes
        a2 = reader.axes
        assert a1 is not a2  # distinct objects
        assert a1 == a2      # same values

    def test_callbacks_stored(self):
        press_cb = MagicMock()
        release_cb = MagicMock()
        reader = JoystickReader(
            on_button_press=press_cb,
            on_button_release=release_cb,
        )
        assert reader._on_press is press_cb
        assert reader._on_release is release_cb

    def test_custom_deadzone(self):
        reader = JoystickReader(deadzone=0.2)
        assert reader._deadzone == 0.2


# ===================================================================
# Deadzone filtering
# ===================================================================

class TestDeadzone:
    """Test deadzone filtering on analog sticks."""

    def test_within_deadzone_returns_zero(self):
        reader = JoystickReader(deadzone=0.10)
        assert reader._apply_deadzone(0.05) == 0.0
        assert reader._apply_deadzone(-0.05) == 0.0
        assert reader._apply_deadzone(0.09) == 0.0

    def test_outside_deadzone_passes_through(self):
        reader = JoystickReader(deadzone=0.10)
        assert reader._apply_deadzone(0.5) == 0.5
        assert reader._apply_deadzone(-0.8) == -0.8

    def test_at_boundary(self):
        reader = JoystickReader(deadzone=0.10)
        # Code uses abs(value) < deadzone, so exactly at threshold passes through
        assert reader._apply_deadzone(0.10) == 0.10
        assert reader._apply_deadzone(0.09) == 0.0

    def test_zero_deadzone(self):
        reader = JoystickReader(deadzone=0.0)
        assert reader._apply_deadzone(0.001) == 0.001

    def test_full_range(self):
        reader = JoystickReader(deadzone=0.10)
        assert reader._apply_deadzone(1.0) == 1.0
        assert reader._apply_deadzone(-1.0) == -1.0


# ===================================================================
# Start / Stop lifecycle
# ===================================================================

class TestLifecycle:
    """Test thread start/stop."""

    @patch.object(JoystickReader, '_run')
    def test_start_creates_thread(self, mock_run):
        reader = JoystickReader()
        reader.start()
        assert reader._running is True
        assert reader._thread is not None
        reader.stop()

    @patch.object(JoystickReader, '_run')
    def test_stop_clears_thread(self, mock_run):
        reader = JoystickReader()
        reader.start()
        reader.stop()
        assert reader._running is False
        assert reader._thread is None

    @patch.object(JoystickReader, '_run')
    def test_double_start_safe(self, mock_run):
        reader = JoystickReader()
        reader.start()
        reader.start()  # Should not create second thread
        reader.stop()

    def test_stop_without_start_safe(self):
        reader = JoystickReader()
        reader.stop()  # Should not raise


# ===================================================================
# Logger methods
# ===================================================================

class TestLogging:
    """Test log helper methods."""

    def test_info_with_logger(self):
        logger = MagicMock()
        reader = JoystickReader(logger=logger)
        reader._info('test message')
        logger.info.assert_called_once_with('test message')

    def test_warn_with_logger(self):
        logger = MagicMock()
        reader = JoystickReader(logger=logger)
        reader._warn('warning message')
        logger.warn.assert_called_once_with('warning message')

    def test_no_logger_no_crash(self):
        reader = JoystickReader(logger=None)
        reader._info('test')
        reader._warn('test')
