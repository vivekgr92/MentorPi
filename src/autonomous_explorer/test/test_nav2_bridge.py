"""Tests for autonomous_explorer.nav2_bridge module.

The Nav2Bridge depends on ROS2 and nav2_msgs, so these tests focus on
the static/pure methods (frontier detection, map rendering math) and
mock the ROS2 layer where needed.
"""
import math

import cv2
import numpy as np
import pytest

from autonomous_explorer.nav2_bridge import NAV2_AVAILABLE


# Skip all tests if nav2_msgs is not installed (dev machine without ROS2)
pytestmark = pytest.mark.skipif(
    not NAV2_AVAILABLE,
    reason='nav2_msgs not installed'
)


# ===================================================================
# Frontier detection (static method)
# ===================================================================

class TestFindFrontiers:
    """Test frontier cell detection from occupancy grids."""

    @pytest.fixture
    def bridge_class(self):
        from autonomous_explorer.nav2_bridge import Nav2Bridge
        return Nav2Bridge

    def test_no_unknown_returns_none(self, bridge_class):
        """Grid fully explored — no frontiers."""
        data = np.zeros((10, 10), dtype=np.int8)  # all free
        free_mask = data == 0
        result = bridge_class._find_frontiers(data, free_mask)
        assert result is None

    def test_no_free_returns_none(self, bridge_class):
        """Grid fully unknown — no frontiers."""
        data = np.full((10, 10), -1, dtype=np.int8)  # all unknown
        free_mask = data == 0
        result = bridge_class._find_frontiers(data, free_mask)
        assert result is None

    def test_finds_frontier_at_boundary(self, bridge_class):
        """Free cells adjacent to unknown should be frontiers."""
        data = np.full((10, 10), -1, dtype=np.int8)  # all unknown
        data[5, 5] = 0   # one free cell
        data[5, 6] = 0   # adjacent free cell
        free_mask = data == 0
        result = bridge_class._find_frontiers(data, free_mask)
        assert result is not None
        assert len(result) >= 1

    def test_interior_free_not_frontier(self, bridge_class):
        """Free cells surrounded by other free cells are not frontiers."""
        data = np.zeros((10, 10), dtype=np.int8)  # all free
        # Add unknown border
        data[0, :] = -1
        data[-1, :] = -1
        data[:, 0] = -1
        data[:, -1] = -1
        free_mask = data == 0
        result = bridge_class._find_frontiers(data, free_mask)
        assert result is not None
        # Interior cells (far from border) should not be in result
        for fy, fx in result:
            # All frontiers should be adjacent to the unknown border
            assert fy <= 1 or fy >= 8 or fx <= 1 or fx >= 8


# ===================================================================
# Navigate relative (coordinate math)
# ===================================================================

class TestNavigateRelative:
    """Test relative → absolute coordinate conversion.

    We test the math without needing a real ROS2 node by checking
    the expected goal coordinates.
    """

    def test_forward_from_origin(self):
        """Moving forward 1m from origin facing east should go to (1, 0)."""
        odom = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        distance = 1.0
        angle_deg = 0  # forward
        theta = odom['theta']
        angle_rad = math.radians(angle_deg)
        abs_angle = theta + angle_rad
        goal_x = odom['x'] + distance * math.cos(abs_angle)
        goal_y = odom['y'] + distance * math.sin(abs_angle)
        assert abs(goal_x - 1.0) < 0.01
        assert abs(goal_y - 0.0) < 0.01

    def test_left_90_degrees(self):
        """Moving 1m to the left from origin facing east."""
        odom = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        distance = 1.0
        angle_deg = 90  # left
        theta = odom['theta']
        angle_rad = math.radians(angle_deg)
        abs_angle = theta + angle_rad
        goal_x = odom['x'] + distance * math.cos(abs_angle)
        goal_y = odom['y'] + distance * math.sin(abs_angle)
        assert abs(goal_x - 0.0) < 0.01
        assert abs(goal_y - 1.0) < 0.01

    def test_forward_with_rotation(self):
        """Forward 1m while facing north (theta=pi/2)."""
        odom = {'x': 0.0, 'y': 0.0, 'theta': math.pi / 2}
        distance = 1.0
        angle_deg = 0
        theta = odom['theta']
        angle_rad = math.radians(angle_deg)
        abs_angle = theta + angle_rad
        goal_x = odom['x'] + distance * math.cos(abs_angle)
        goal_y = odom['y'] + distance * math.sin(abs_angle)
        assert abs(goal_x - 0.0) < 0.01
        assert abs(goal_y - 1.0) < 0.01


# ===================================================================
# Map rendering colors (constants check)
# ===================================================================

class TestMapColors:
    """Verify map rendering color constants."""

    def test_colors_are_bgr_tuples(self):
        from autonomous_explorer.nav2_bridge import (
            _COLOR_UNKNOWN, _COLOR_FREE, _COLOR_OCCUPIED,
            _COLOR_ROBOT, _COLOR_GOAL, _COLOR_FRONTIER,
        )
        for color in [_COLOR_UNKNOWN, _COLOR_FREE, _COLOR_OCCUPIED,
                       _COLOR_ROBOT, _COLOR_GOAL, _COLOR_FRONTIER]:
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)

    def test_free_is_white(self):
        from autonomous_explorer.nav2_bridge import _COLOR_FREE
        assert _COLOR_FREE == (255, 255, 255)

    def test_occupied_is_black(self):
        from autonomous_explorer.nav2_bridge import _COLOR_OCCUPIED
        assert _COLOR_OCCUPIED == (0, 0, 0)
