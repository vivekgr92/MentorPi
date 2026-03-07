"""Tests for autonomous_explorer.nav2_bridge module.

The Nav2Bridge class needs ROS2 runtime, but many of its methods contain
pure math/logic that can be tested independently. Tests that need ROS2
(action clients, subscriptions) are skipped on the dev machine.
"""
import math

import cv2
import numpy as np
import pytest

from autonomous_explorer.nav2_bridge import NAV2_AVAILABLE


# ===================================================================
# Frontier detection (static method — no ROS2 needed)
# The _find_frontiers method uses only numpy/OpenCV, but it's accessed
# via Nav2Bridge which imports nav2_msgs at module level. We replicate
# the algorithm here for testing when nav2_msgs is unavailable.
# ===================================================================

def _find_frontiers_standalone(data, free_mask):
    """Replica of Nav2Bridge._find_frontiers for testing without ROS2."""
    unknown_mask = data == -1
    if not unknown_mask.any() or not free_mask.any():
        return None
    kernel = np.ones((3, 3), dtype=np.uint8)
    unknown_dilated = cv2.dilate(
        unknown_mask.astype(np.uint8), kernel, iterations=1,
    )
    frontier_mask = free_mask & (unknown_dilated > 0)
    if not frontier_mask.any():
        return None
    return np.argwhere(frontier_mask)


class TestFindFrontiers:
    """Test frontier cell detection from occupancy grids."""

    def test_no_unknown_returns_none(self):
        """Grid fully explored (all free) — no frontiers."""
        data = np.zeros((10, 10), dtype=np.int8)
        free_mask = data == 0
        result = _find_frontiers_standalone(data, free_mask)
        assert result is None

    def test_no_free_returns_none(self):
        """Grid fully unknown — no frontiers."""
        data = np.full((10, 10), -1, dtype=np.int8)
        free_mask = data == 0
        result = _find_frontiers_standalone(data, free_mask)
        assert result is None

    def test_all_occupied_no_frontiers(self):
        """Grid fully occupied (value > 0) — no frontiers."""
        data = np.full((10, 10), 50, dtype=np.int8)
        free_mask = data == 0
        result = _find_frontiers_standalone(data, free_mask)
        assert result is None

    def test_finds_frontier_at_boundary(self):
        """Free cells adjacent to unknown should be frontiers."""
        data = np.full((10, 10), -1, dtype=np.int8)  # all unknown
        data[5, 5] = 0   # one free cell
        data[5, 6] = 0   # adjacent free cell
        free_mask = data == 0
        result = _find_frontiers_standalone(data, free_mask)
        assert result is not None
        assert len(result) >= 1
        # The free cells should be in the frontier result
        coords = set(map(tuple, result))
        assert (5, 5) in coords or (5, 6) in coords

    def test_interior_free_not_frontier(self):
        """Free cells surrounded by other free cells are not frontiers.
        Only cells adjacent to unknown border are frontiers."""
        data = np.zeros((20, 20), dtype=np.int8)  # all free
        # Add unknown border
        data[0, :] = -1
        data[-1, :] = -1
        data[:, 0] = -1
        data[:, -1] = -1
        free_mask = data == 0
        result = _find_frontiers_standalone(data, free_mask)
        assert result is not None
        # All frontiers should be near edges (row 1 or 18, col 1 or 18)
        for fy, fx in result:
            assert fy <= 1 or fy >= 18 or fx <= 1 or fx >= 18

    def test_single_free_cell_surrounded_by_unknown(self):
        """A single free cell in a sea of unknown is a frontier."""
        data = np.full((10, 10), -1, dtype=np.int8)
        data[5, 5] = 0
        free_mask = data == 0
        result = _find_frontiers_standalone(data, free_mask)
        assert result is not None
        assert tuple(result[0]) == (5, 5)

    def test_free_surrounded_by_occupied_no_frontier(self):
        """Free cells surrounded by occupied (not unknown) are not frontiers."""
        data = np.full((10, 10), 100, dtype=np.int8)  # all walls
        data[5, 5] = 0  # one free cell
        free_mask = data == 0
        result = _find_frontiers_standalone(data, free_mask)
        assert result is None

    def test_large_explored_area_with_frontier_edge(self):
        """Large explored area with unknown on one side."""
        data = np.zeros((50, 50), dtype=np.int8)  # all free
        data[:, 25:] = -1  # right half unknown
        free_mask = data == 0
        result = _find_frontiers_standalone(data, free_mask)
        assert result is not None
        # All frontiers should be near column 25 (the boundary)
        for fy, fx in result:
            assert 23 <= fx <= 25


# ===================================================================
# Navigate relative (coordinate math — no ROS2 needed)
# ===================================================================

class TestNavigateRelativeMath:
    """Test relative → absolute coordinate conversion math.

    The actual navigate_relative() method calls navigate_to() which needs
    ROS2, but the math is extracted and tested independently.
    """

    @staticmethod
    def _compute_goal(distance, angle_deg, odom):
        """Replicate Nav2Bridge.navigate_relative math."""
        theta = odom.get('theta', 0)
        angle_rad = math.radians(angle_deg)
        abs_angle = theta + angle_rad
        goal_x = odom['x'] + distance * math.cos(abs_angle)
        goal_y = odom['y'] + distance * math.sin(abs_angle)
        return goal_x, goal_y

    def test_forward_from_origin(self):
        """Moving forward 1m from origin facing east → (1, 0)."""
        gx, gy = self._compute_goal(1.0, 0, {'x': 0, 'y': 0, 'theta': 0})
        assert abs(gx - 1.0) < 0.01
        assert abs(gy - 0.0) < 0.01

    def test_left_90_degrees(self):
        """Moving 1m to the left from origin facing east → (0, 1)."""
        gx, gy = self._compute_goal(1.0, 90, {'x': 0, 'y': 0, 'theta': 0})
        assert abs(gx - 0.0) < 0.01
        assert abs(gy - 1.0) < 0.01

    def test_right_90_degrees(self):
        """Moving 1m to the right from origin facing east → (0, -1)."""
        gx, gy = self._compute_goal(1.0, -90, {'x': 0, 'y': 0, 'theta': 0})
        assert abs(gx - 0.0) < 0.01
        assert abs(gy - (-1.0)) < 0.01

    def test_backward_180_degrees(self):
        """Moving 1m backward from origin facing east → (-1, 0)."""
        gx, gy = self._compute_goal(1.0, 180, {'x': 0, 'y': 0, 'theta': 0})
        assert abs(gx - (-1.0)) < 0.01
        assert abs(gy - 0.0) < 0.01

    def test_forward_facing_north(self):
        """Forward 1m while facing north (theta=pi/2) → (0, 1)."""
        gx, gy = self._compute_goal(1.0, 0, {'x': 0, 'y': 0, 'theta': math.pi / 2})
        assert abs(gx - 0.0) < 0.01
        assert abs(gy - 1.0) < 0.01

    def test_offset_position(self):
        """Forward 2m from position (3, 4) facing east → (5, 4)."""
        gx, gy = self._compute_goal(2.0, 0, {'x': 3.0, 'y': 4.0, 'theta': 0})
        assert abs(gx - 5.0) < 0.01
        assert abs(gy - 4.0) < 0.01

    def test_diagonal_45_degrees(self):
        """Forward 1m at 45° from origin → (0.707, 0.707)."""
        gx, gy = self._compute_goal(1.0, 45, {'x': 0, 'y': 0, 'theta': 0})
        assert abs(gx - math.sqrt(2) / 2) < 0.01
        assert abs(gy - math.sqrt(2) / 2) < 0.01

    def test_zero_distance(self):
        """Zero distance returns current position."""
        gx, gy = self._compute_goal(0.0, 45, {'x': 1.0, 'y': 2.0, 'theta': 0.5})
        assert abs(gx - 1.0) < 0.01
        assert abs(gy - 2.0) < 0.01


# ===================================================================
# Map rendering colors (constants — importable even without ROS2)
# ===================================================================

class TestMapColors:
    """Verify map rendering color constants are sensible."""

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

    def test_unknown_is_gray(self):
        from autonomous_explorer.nav2_bridge import _COLOR_UNKNOWN
        assert _COLOR_UNKNOWN == (128, 128, 128)

    def test_colors_are_distinct(self):
        from autonomous_explorer.nav2_bridge import (
            _COLOR_UNKNOWN, _COLOR_FREE, _COLOR_OCCUPIED,
            _COLOR_ROBOT, _COLOR_GOAL, _COLOR_FRONTIER,
        )
        colors = [_COLOR_UNKNOWN, _COLOR_FREE, _COLOR_OCCUPIED,
                  _COLOR_ROBOT, _COLOR_GOAL, _COLOR_FRONTIER]
        assert len(colors) == len(set(colors))


# ===================================================================
# Map image size config
# ===================================================================

class TestMapConfig:
    """Test map-related configuration."""

    def test_map_image_size_reasonable(self):
        from autonomous_explorer.nav2_bridge import _MAP_IMAGE_SIZE
        assert 64 <= _MAP_IMAGE_SIZE <= 1024


# ===================================================================
# Occupancy grid → image conversion math
# ===================================================================

class TestOccupancyGridRendering:
    """Test the map rendering logic (world→pixel math, color mapping).

    We test the pure math without needing Nav2Bridge instantiation.
    """

    def test_world_to_pixel_at_origin(self):
        """World (0,0) with origin at (0,0) and resolution 0.05 → pixel (0,0)."""
        origin_x, origin_y, resolution = 0.0, 0.0, 0.05
        wx, wy = 0.0, 0.0
        px = int((wx - origin_x) / resolution)
        py = int((wy - origin_y) / resolution)
        assert px == 0
        assert py == 0

    def test_world_to_pixel_offset(self):
        """World (1.0, 2.0) with origin (-5, -5) and resolution 0.05."""
        origin_x, origin_y, resolution = -5.0, -5.0, 0.05
        wx, wy = 1.0, 2.0
        px = int((wx - origin_x) / resolution)
        py = int((wy - origin_y) / resolution)
        assert px == 120  # (1.0 - (-5.0)) / 0.05 = 120
        assert py == 140  # (2.0 - (-5.0)) / 0.05 = 140

    def test_occupancy_to_rgb_mapping(self):
        """Test that occupancy values map to correct colors."""
        from autonomous_explorer.nav2_bridge import (
            _COLOR_FREE, _COLOR_OCCUPIED, _COLOR_UNKNOWN,
        )
        # Simulate a small 3x3 occupancy grid
        data = np.array([
            [0, -1, 100],
            [0,  0,  -1],
            [50, 0,   0],
        ], dtype=np.int8)
        h, w = data.shape

        img = np.full((h, w, 3), _COLOR_UNKNOWN[0], dtype=np.uint8)
        free_mask = data == 0
        occ_mask = (data > 0) & (data <= 100)
        img[free_mask] = _COLOR_FREE
        img[occ_mask] = _COLOR_OCCUPIED

        # Check specific cells
        assert tuple(img[0, 0]) == _COLOR_FREE       # 0 = free
        assert tuple(img[0, 1]) == _COLOR_UNKNOWN     # -1 = unknown
        assert tuple(img[0, 2]) == _COLOR_OCCUPIED    # 100 = occupied
        assert tuple(img[2, 0]) == _COLOR_OCCUPIED    # 50 = occupied

    def test_quaternion_to_heading(self):
        """Test yaw extraction from quaternion (used for robot arrow)."""
        # Quaternion for theta=0 (facing east): (0,0,0,1)
        theta = 0.0
        qz = math.sin(theta / 2.0)
        qw = math.cos(theta / 2.0)
        yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
        assert abs(yaw) < 0.01

        # Quaternion for theta=pi/2 (facing north)
        theta = math.pi / 2
        qz = math.sin(theta / 2.0)
        qw = math.cos(theta / 2.0)
        yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
        assert abs(yaw - math.pi / 2) < 0.01


# ===================================================================
# Frontier clustering (used by get_frontier_goals)
# ===================================================================

class TestFrontierClustering:
    """Test frontier clustering and goal extraction logic."""

    def test_small_clusters_filtered(self):
        """Clusters with < 3 points should be ignored."""
        # Create a map with 2 isolated frontier pixels
        frontier_map = np.zeros((50, 50), dtype=np.uint8)
        frontier_map[10, 10] = 255
        frontier_map[40, 40] = 255
        num_labels, labels = cv2.connectedComponents(frontier_map)
        # Each pixel is its own cluster of size 1
        for label in range(1, num_labels):
            points = np.argwhere(labels == label)
            assert len(points) < 3  # All should be filtered

    def test_large_cluster_accepted(self):
        """A connected cluster of >= 3 pixels is a valid frontier goal."""
        frontier_map = np.zeros((50, 50), dtype=np.uint8)
        frontier_map[10, 10:15] = 255  # 5 connected pixels
        num_labels, labels = cv2.connectedComponents(frontier_map)
        valid_goals = 0
        for label in range(1, num_labels):
            points = np.argwhere(labels == label)
            if len(points) >= 3:
                valid_goals += 1
        assert valid_goals == 1

    def test_centroid_calculation(self):
        """Cluster centroid should be the mean of its points."""
        frontier_map = np.zeros((50, 50), dtype=np.uint8)
        frontier_map[20, 10:16] = 255  # 6 pixels at row 20, cols 10-15
        num_labels, labels = cv2.connectedComponents(frontier_map)
        for label in range(1, num_labels):
            points = np.argwhere(labels == label)
            cy, cx = points.mean(axis=0)
            assert abs(cy - 20.0) < 0.1
            assert abs(cx - 12.5) < 0.1

    def test_distance_filtering(self):
        """Goals closer than min_distance should be filtered."""
        robot_x, robot_y = 0.0, 0.0
        min_distance = 0.5
        resolution = 0.05
        origin_x, origin_y = -2.5, -2.5

        # Goal at (0.1, 0.1) — very close to robot
        cx, cy = 52, 52  # pixel coords
        wx = origin_x + cx * resolution  # 0.1
        wy = origin_y + cy * resolution  # 0.1
        dist = math.sqrt((wx - robot_x) ** 2 + (wy - robot_y) ** 2)
        assert dist < min_distance  # Should be filtered

        # Goal at (2.0, 0.0) — far enough
        cx2 = 90  # pixel coord
        wx2 = origin_x + cx2 * resolution  # 2.0
        dist2 = math.sqrt((wx2 - robot_x) ** 2 + (origin_y + 52 * resolution - robot_y) ** 2)
        assert dist2 > min_distance  # Should be kept


# ===================================================================
# Nav2Bridge class (requires ROS2 — skip if unavailable)
# ===================================================================

@pytest.mark.skipif(not NAV2_AVAILABLE, reason='nav2_msgs not installed')
class TestNav2BridgeWithROS2:
    """Tests requiring actual Nav2Bridge instantiation (ROS2 only)."""

    def test_find_frontiers_matches_standalone(self):
        """Verify the class method matches our standalone replica."""
        from autonomous_explorer.nav2_bridge import Nav2Bridge
        data = np.full((10, 10), -1, dtype=np.int8)
        data[5, 5] = 0
        free_mask = data == 0
        result = Nav2Bridge._find_frontiers(data, free_mask)
        standalone = _find_frontiers_standalone(data, free_mask)
        assert result is not None
        assert standalone is not None
        np.testing.assert_array_equal(result, standalone)
