#!/usr/bin/env python3
# encoding: utf-8
"""
Nav2 bridge for the autonomous explorer.

Provides:
  - Nav2 goal sending via NavigateToPose action client
  - Occupancy grid map rendering as a bird's-eye image for the LLM
  - Frontier detection (boundary between explored and unexplored)
  - Goal conversion: relative directions → absolute map coordinates

This module is optional — if nav2_msgs is not installed, the explorer
falls back to direct motor control.
"""
import math
import threading
import time

import cv2
import numpy as np

try:
    from nav2_msgs.action import NavigateToPose
    from nav_msgs.msg import OccupancyGrid
    from geometry_msgs.msg import PoseStamped
    from rclpy.action import ActionClient
    from action_msgs.msg import GoalStatus
    NAV2_AVAILABLE = True
except ImportError:
    NAV2_AVAILABLE = False


# Map rendering colors (BGR for OpenCV)
_COLOR_UNKNOWN = (128, 128, 128)   # gray
_COLOR_FREE = (255, 255, 255)      # white
_COLOR_OCCUPIED = (0, 0, 0)        # black
_COLOR_ROBOT = (0, 0, 255)         # red
_COLOR_GOAL = (0, 255, 0)          # green
_COLOR_FRONTIER = (255, 165, 0)    # orange
_COLOR_PATH = (255, 0, 0)          # blue

# Map image size sent to LLM (imported from config)
from autonomous_explorer.config import MAP_IMAGE_SIZE as _MAP_IMAGE_SIZE


class Nav2Bridge:
    """Interface between the explorer and the Nav2 navigation stack.

    Handles:
      - Sending navigation goals (absolute or relative)
      - Monitoring navigation progress and results
      - Subscribing to /map and rendering it as an image
      - Detecting frontier cells for exploration
    """

    def __init__(self, node):
        if not NAV2_AVAILABLE:
            raise ImportError(
                'nav2_msgs not installed. Run install_nav2.sh first.'
            )

        self._node = node
        self._logger = node.get_logger()

        # Nav2 action client
        self._nav_client = ActionClient(
            node, NavigateToPose, 'navigate_to_pose',
        )
        self._goal_handle = None
        self._navigating = False
        self._nav_result = None
        self._nav_feedback = None
        self._nav_lock = threading.Lock()

        # Map subscriber
        self._map_msg = None
        self._map_lock = threading.Lock()
        from autonomous_explorer.config import NAV2_MAP_TOPIC
        self._map_sub = node.create_subscription(
            OccupancyGrid, NAV2_MAP_TOPIC, self._map_callback, 10,
        )

        # Rendered map cache
        self._map_image = None
        self._map_render_time = 0

        self._logger.info('Nav2Bridge initialized')

    # ------------------------------------------------------------------
    # Navigation goal API
    # ------------------------------------------------------------------

    def wait_for_nav2(self, timeout_sec: float = 10.0) -> bool:
        """Wait for the Nav2 action server to become available."""
        self._logger.info('Waiting for Nav2 action server...')
        ready = self._nav_client.wait_for_server(timeout_sec=timeout_sec)
        if ready:
            self._logger.info('Nav2 action server is ready')
        else:
            self._logger.warn('Nav2 action server not available')
        return ready

    def navigate_to(
        self, x: float, y: float, theta: float = 0.0,
        frame_id: str = 'map',
    ) -> bool:
        """Send a goal pose to Nav2.

        Returns True if the goal was accepted, False otherwise.
        """
        if self._navigating:
            self._logger.warn('Already navigating — cancel first')
            return False

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = frame_id
        goal.pose.header.stamp = self._node.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.z = math.sin(theta / 2.0)
        goal.pose.pose.orientation.w = math.cos(theta / 2.0)

        self._logger.info(
            f'Sending Nav2 goal: ({x:.2f}, {y:.2f}) theta={math.degrees(theta):.1f}°'
        )

        future = self._nav_client.send_goal_async(
            goal, feedback_callback=self._nav_feedback_cb,
        )
        future.add_done_callback(self._nav_goal_response_cb)

        with self._nav_lock:
            self._navigating = True
            self._nav_result = None
            self._nav_feedback = None

        return True

    def navigate_relative(
        self, distance: float, angle_deg: float,
        current_odom: dict,
    ) -> bool:
        """Navigate to a point relative to the robot's current position.

        Args:
            distance: meters to travel
            angle_deg: direction in degrees (0=forward, 90=left, -90=right)
            current_odom: dict with keys 'x', 'y', 'theta' (radians)
        """
        theta = current_odom.get('theta', 0)
        angle_rad = math.radians(angle_deg)
        abs_angle = theta + angle_rad

        goal_x = current_odom['x'] + distance * math.cos(abs_angle)
        goal_y = current_odom['y'] + distance * math.sin(abs_angle)
        goal_theta = abs_angle  # face the direction of travel

        return self.navigate_to(goal_x, goal_y, goal_theta)

    def cancel_navigation(self):
        """Cancel the current navigation goal."""
        with self._nav_lock:
            if self._goal_handle is not None:
                self._logger.info('Canceling Nav2 goal')
                self._goal_handle.cancel_goal_async()
                self._navigating = False
                self._goal_handle = None

    @property
    def is_navigating(self) -> bool:
        with self._nav_lock:
            return self._navigating

    @property
    def navigation_result(self) -> str | None:
        """Return 'succeeded', 'failed', 'canceled', or None if still navigating."""
        with self._nav_lock:
            return self._nav_result

    @property
    def navigation_feedback(self) -> dict | None:
        """Return latest feedback: distance_remaining, ETA, etc."""
        with self._nav_lock:
            return self._nav_feedback

    # ------------------------------------------------------------------
    # Nav2 action callbacks
    # ------------------------------------------------------------------

    def _nav_goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._logger.warn('Nav2 goal rejected')
            with self._nav_lock:
                self._navigating = False
                self._nav_result = 'rejected'
            return

        self._logger.info('Nav2 goal accepted')
        with self._nav_lock:
            self._goal_handle = goal_handle

        # Wait for result
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._nav_result_cb)

    def _nav_result_cb(self, future):
        result = future.result()
        status = result.status

        with self._nav_lock:
            self._navigating = False
            self._goal_handle = None
            if status == GoalStatus.STATUS_SUCCEEDED:
                self._nav_result = 'succeeded'
                self._logger.info('Nav2 goal reached!')
            elif status == GoalStatus.STATUS_CANCELED:
                self._nav_result = 'canceled'
                self._logger.info('Nav2 goal canceled')
            else:
                self._nav_result = 'failed'
                self._logger.warn(f'Nav2 goal failed (status={status})')

    def _nav_feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        remaining = fb.distance_remaining
        with self._nav_lock:
            self._nav_feedback = {
                'distance_remaining': round(remaining, 2),
            }

    # ------------------------------------------------------------------
    # Map subscription and rendering
    # ------------------------------------------------------------------

    def _map_callback(self, msg: 'OccupancyGrid'):
        with self._map_lock:
            self._map_msg = msg

    @property
    def has_map(self) -> bool:
        with self._map_lock:
            return self._map_msg is not None

    def get_map_stats(self) -> dict:
        """Return basic stats about the current map."""
        with self._map_lock:
            if self._map_msg is None:
                return {}
            info = self._map_msg.info
            data = np.array(self._map_msg.data, dtype=np.int8)
            total = len(data)
            free = int(np.sum(data == 0))
            occupied = int(np.sum((data > 0) & (data <= 100)))
            unknown = int(np.sum(data == -1))
            return {
                'width': info.width,
                'height': info.height,
                'resolution': info.resolution,
                'total_cells': total,
                'free_cells': free,
                'occupied_cells': occupied,
                'unknown_cells': unknown,
                'explored_pct': round(
                    100.0 * (free + occupied) / max(total, 1), 1,
                ),
                'area_m2': round(
                    (free + occupied) * info.resolution ** 2, 2,
                ),
            }

    def render_map_image(
        self,
        robot_x: float = 0.0,
        robot_y: float = 0.0,
        robot_theta: float = 0.0,
        goal_x: float | None = None,
        goal_y: float | None = None,
    ) -> np.ndarray | None:
        """Render the occupancy grid as a bird's-eye RGB image.

        Returns a small (_MAP_IMAGE_SIZE x _MAP_IMAGE_SIZE) RGB image
        showing walls, free space, unknown areas, robot position,
        and frontier cells. Returns None if no map is available.
        """
        with self._map_lock:
            if self._map_msg is None:
                return None
            msg = self._map_msg

        info = msg.info
        w, h = info.width, info.height
        resolution = info.resolution
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y

        # Convert occupancy grid to numpy
        data = np.array(msg.data, dtype=np.int8).reshape(h, w)

        # Create RGB image
        img = np.full((h, w, 3), _COLOR_UNKNOWN[0], dtype=np.uint8)
        # Free space
        free_mask = data == 0
        img[free_mask] = _COLOR_FREE
        # Occupied
        occ_mask = (data > 0) & (data <= 100)
        img[occ_mask] = _COLOR_OCCUPIED

        # Detect and mark frontiers (free cells adjacent to unknown)
        frontiers = self._find_frontiers(data, free_mask)
        if frontiers is not None and len(frontiers) > 0:
            for fy, fx in frontiers:
                cv2.circle(img, (fx, fy), 1, _COLOR_FRONTIER, -1)

        # World → pixel coordinate conversion
        def world_to_pixel(wx, wy):
            px = int((wx - origin_x) / resolution)
            py = int((wy - origin_y) / resolution)
            return px, py

        # Draw robot position and heading
        rx, ry = world_to_pixel(robot_x, robot_y)
        if 0 <= rx < w and 0 <= ry < h:
            cv2.circle(img, (rx, ry), max(3, w // 80), _COLOR_ROBOT, -1)
            arrow_len = max(8, w // 40)
            dx = int(arrow_len * math.cos(robot_theta))
            dy = int(arrow_len * math.sin(robot_theta))
            cv2.arrowedLine(
                img, (rx, ry), (rx + dx, ry - dy),
                _COLOR_ROBOT, max(1, w // 200),
            )

        # Draw goal if provided
        if goal_x is not None and goal_y is not None:
            gx, gy = world_to_pixel(goal_x, goal_y)
            if 0 <= gx < w and 0 <= gy < h:
                cv2.drawMarker(
                    img, (gx, gy), _COLOR_GOAL,
                    cv2.MARKER_CROSS, max(5, w // 60), max(1, w // 200),
                )

        # Flip vertically (map origin is bottom-left, image top-left)
        img = cv2.flip(img, 0)

        # Resize to target size
        scale = _MAP_IMAGE_SIZE / max(w, h)
        img = cv2.resize(
            img, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )

        # Convert BGR to RGB for the LLM
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self._map_image = img
        self._map_render_time = time.time()
        return img

    def get_frontier_goals(
        self,
        robot_x: float,
        robot_y: float,
        min_distance: float = 0.5,
        max_goals: int = 5,
    ) -> list[dict]:
        """Find frontier clusters and return them as potential goals.

        Returns a list of dicts: [{'x': float, 'y': float, 'size': int}, ...]
        sorted by distance from robot (nearest first).
        """
        with self._map_lock:
            if self._map_msg is None:
                return []
            msg = self._map_msg

        info = msg.info
        w, h = info.width, info.height
        resolution = info.resolution
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y

        data = np.array(msg.data, dtype=np.int8).reshape(h, w)
        free_mask = data == 0
        frontiers = self._find_frontiers(data, free_mask)

        if frontiers is None or len(frontiers) == 0:
            return []

        # Cluster frontiers using simple connected components
        frontier_map = np.zeros((h, w), dtype=np.uint8)
        for fy, fx in frontiers:
            frontier_map[fy, fx] = 255

        num_labels, labels = cv2.connectedComponents(frontier_map)

        goals = []
        for label in range(1, num_labels):
            points = np.argwhere(labels == label)
            if len(points) < 3:
                continue  # skip tiny clusters

            # Centroid in world coordinates
            cy, cx = points.mean(axis=0)
            wx = origin_x + cx * resolution
            wy = origin_y + cy * resolution

            dist = math.sqrt((wx - robot_x) ** 2 + (wy - robot_y) ** 2)
            if dist < min_distance:
                continue

            goals.append({
                'x': round(wx, 2),
                'y': round(wy, 2),
                'size': len(points),
                'distance': round(dist, 2),
            })

        # Sort by distance
        goals.sort(key=lambda g: g['distance'])
        return goals[:max_goals]

    # ------------------------------------------------------------------
    # Frontier detection
    # ------------------------------------------------------------------

    @staticmethod
    def _find_frontiers(
        data: np.ndarray, free_mask: np.ndarray,
    ) -> np.ndarray | None:
        """Find frontier cells: free cells adjacent to unknown cells."""
        unknown_mask = data == -1
        if not unknown_mask.any() or not free_mask.any():
            return None

        # Dilate unknown region by 1 pixel
        kernel = np.ones((3, 3), dtype=np.uint8)
        unknown_dilated = cv2.dilate(
            unknown_mask.astype(np.uint8), kernel, iterations=1,
        )

        # Frontier = free AND adjacent-to-unknown
        frontier_mask = free_mask & (unknown_dilated > 0)

        if not frontier_mask.any():
            return None

        return np.argwhere(frontier_mask)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy(self):
        """Cancel any in-progress navigation and clean up."""
        self.cancel_navigation()
        self._node.destroy_subscription(self._map_sub)
