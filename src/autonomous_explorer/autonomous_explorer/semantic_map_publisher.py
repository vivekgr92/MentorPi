#!/usr/bin/env python3
"""
Semantic Map Publisher Node.

Reads from WorldKnowledge JSON files on disk and publishes ROS2 MarkerArray
messages for Foxglove visualization. Judges see room labels appearing on the
map in real-time as Jeeves explores and labels new rooms.

Published topics:
  /semantic_map/rooms    — visualization_msgs/MarkerArray (TEXT_VIEW_FACING)
  /semantic_map/objects  — visualization_msgs/MarkerArray (TEXT_VIEW_FACING)
  /semantic_map/agent_status — std_msgs/String (current LLM reasoning)

Persistence:
  ~/.jeeves/semantic_map.json — simplified view for external tools
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

# Where WorldKnowledge stores its files
_DEFAULT_KNOWLEDGE_DIR = os.path.expanduser('~/mentorpi_explorer/knowledge')
# Where we write the simplified semantic_map.json
_DEFAULT_SEMANTIC_MAP_PATH = os.path.expanduser('~/.jeeves/semantic_map.json')


class SemanticMapPublisher(Node):

    def __init__(self):
        super().__init__('semantic_map_publisher')

        # Parameters
        self.declare_parameter(
            'knowledge_dir', _DEFAULT_KNOWLEDGE_DIR)
        self.declare_parameter(
            'semantic_map_path', _DEFAULT_SEMANTIC_MAP_PATH)
        self.declare_parameter('publish_rate', 1.0)

        self._knowledge_dir = self.get_parameter(
            'knowledge_dir').get_parameter_value().string_value
        self._semantic_map_path = self.get_parameter(
            'semantic_map_path').get_parameter_value().string_value
        self._publish_rate = self.get_parameter(
            'publish_rate').get_parameter_value().double_value

        # Ensure output directories exist
        Path(self._semantic_map_path).parent.mkdir(parents=True, exist_ok=True)

        # File paths for WorldKnowledge data
        self._world_map_file = os.path.join(
            self._knowledge_dir, 'world_map.json')
        self._known_objects_file = os.path.join(
            self._knowledge_dir, 'known_objects.json')

        # Cache: avoid re-publishing if nothing changed
        self._last_world_map_mtime = 0.0
        self._last_objects_mtime = 0.0
        self._rooms_data = {}
        self._objects_data = {}

        # Transient latched QoS so Foxglove sees markers immediately on connect
        latched_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )

        # Publishers
        self._rooms_pub = self.create_publisher(
            MarkerArray, '/semantic_map/rooms', latched_qos)
        self._objects_pub = self.create_publisher(
            MarkerArray, '/semantic_map/objects', latched_qos)
        self._status_pub = self.create_publisher(
            String, '/semantic_map/agent_status', 10)

        # Subscribe to explorer status for agent reasoning
        self._latest_reasoning = ''
        self.create_subscription(
            String, '/explorer/status', self._status_cb, 10)

        # Load persisted semantic map on startup
        self._load_semantic_map()

        # 1Hz timer
        period = 1.0 / self._publish_rate
        self.create_timer(period, self._timer_cb)

        self.get_logger().info(
            f'Semantic map publisher started '
            f'(knowledge_dir={self._knowledge_dir}, '
            f'rate={self._publish_rate}Hz)')

    # ── Subscriber callback ──

    def _status_cb(self, msg: String):
        """Extract LLM reasoning from explorer status JSON."""
        try:
            status = json.loads(msg.data)
            reasoning = status.get('llm', {}).get('reasoning', '')
            if not reasoning:
                reasoning = status.get('llm', {}).get('action', '')
            if reasoning and reasoning != self._latest_reasoning:
                self._latest_reasoning = reasoning
                out = String()
                out.data = reasoning
                self._status_pub.publish(out)
        except (json.JSONDecodeError, AttributeError):
            pass

    # ── Timer callback ──

    def _timer_cb(self):
        """Read knowledge files, publish markers, save semantic map."""
        changed = self._reload_knowledge_files()
        # Always publish (markers are latched, but new subscriptions need them)
        self._publish_room_markers()
        self._publish_object_markers()
        if changed:
            self._save_semantic_map()

    # ── Knowledge file reading ──

    def _reload_knowledge_files(self) -> bool:
        """Re-read WorldKnowledge JSON files if modified. Returns True if changed."""
        changed = False

        # world_map.json
        try:
            mtime = os.path.getmtime(self._world_map_file)
            if mtime != self._last_world_map_mtime:
                with open(self._world_map_file, 'r') as f:
                    self._rooms_data = json.load(f)
                self._last_world_map_mtime = mtime
                changed = True
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        # known_objects.json
        try:
            mtime = os.path.getmtime(self._known_objects_file)
            if mtime != self._last_objects_mtime:
                with open(self._known_objects_file, 'r') as f:
                    self._objects_data = json.load(f)
                self._last_objects_mtime = mtime
                changed = True
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        return changed

    # ── MarkerArray publishing ──

    def _publish_room_markers(self):
        """Publish TEXT_VIEW_FACING markers for each labeled room."""
        rooms = self._rooms_data.get('rooms', {})
        if not rooms:
            return

        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        for idx, (name, info) in enumerate(rooms.items()):
            # Rooms need coordinates — use position if stored
            x = float(info.get('x', info.get('position', {}).get('x', 0.0)))
            y = float(info.get('y', info.get('position', {}).get('y', 0.0)))

            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = now
            marker.ns = 'rooms'
            marker.id = idx
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD

            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.5  # Float above map plane
            marker.pose.orientation.w = 1.0

            # scale.z = text height — MUST be > 0 or Foxglove won't render
            marker.scale.z = 0.4

            # Bright green, fully opaque
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.5
            marker.color.a = 1.0

            # Plain text — no emoji (Foxglove font may not render them)
            visits = info.get('times_visited', info.get('visit_count', 1))
            label = name.title()
            if isinstance(visits, int) and visits > 1:
                label += f' ({visits}x)'
            marker.text = label
            marker.lifetime.sec = 0  # Never expire

            marker_array.markers.append(marker)

        self._rooms_pub.publish(marker_array)

    def _publish_object_markers(self):
        """Publish TEXT_VIEW_FACING markers for each registered object."""
        objects = self._objects_data.get('objects', self._objects_data)
        if not objects or not isinstance(objects, dict):
            return

        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        idx = 0
        for name, info in objects.items():
            if not isinstance(info, dict):
                continue

            # Objects may have direct x/y or locations list
            locations = info.get('locations', [])
            if locations and isinstance(locations, list):
                loc = locations[-1]  # Most recent location
                if isinstance(loc, dict):
                    x = float(loc.get('x', 0.0))
                    y = float(loc.get('y', 0.0))
                elif isinstance(loc, (list, tuple)) and len(loc) >= 2:
                    x, y = float(loc[0]), float(loc[1])
                else:
                    continue
            elif 'x' in info and 'y' in info:
                x = float(info['x'])
                y = float(info['y'])
            elif 'position' in info:
                x = float(info['position'].get('x', 0.0))
                y = float(info['position'].get('y', 0.0))
            else:
                # No position data — skip
                continue

            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = now
            marker.ns = 'objects'
            marker.id = idx
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD

            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.3  # Below room labels
            marker.pose.orientation.w = 1.0

            # scale.z = text height — MUST be > 0 or Foxglove won't render
            marker.scale.z = 0.25

            # Yellow, fully opaque
            marker.color.r = 1.0
            marker.color.g = 0.8
            marker.color.b = 0.0
            marker.color.a = 1.0

            # Plain text — no emoji
            category = info.get('category', '')
            label = name
            if category:
                label += f' [{category}]'
            marker.text = label
            marker.lifetime.sec = 0

            marker_array.markers.append(marker)
            idx += 1

        self._objects_pub.publish(marker_array)

    # ── Semantic map persistence ──

    def _load_semantic_map(self):
        """Load previously saved semantic_map.json to bootstrap rooms/objects."""
        try:
            with open(self._semantic_map_path, 'r') as f:
                data = json.load(f)
            self.get_logger().info(
                f'Loaded semantic map: {len(data.get("rooms", []))} rooms')
        except (FileNotFoundError, json.JSONDecodeError):
            self.get_logger().info('No previous semantic map found, starting fresh')

    def _save_semantic_map(self):
        """Save simplified semantic map JSON for external consumption."""
        rooms_list = []
        rooms = self._rooms_data.get('rooms', {})
        objects = self._objects_data.get('objects', self._objects_data)

        for name, info in rooms.items():
            x = float(info.get('x', info.get('position', {}).get('x', 0.0)))
            y = float(info.get('y', info.get('position', {}).get('y', 0.0)))

            # Collect objects associated with this room
            room_objects = []
            if isinstance(objects, dict):
                for obj_name, obj_info in objects.items():
                    if isinstance(obj_info, dict):
                        obj_room = obj_info.get('room', '')
                        if obj_room and obj_room.lower() == name.lower():
                            room_objects.append(obj_name)

            labeled_at = info.get('discovered', info.get('labeled_at', ''))
            if not labeled_at:
                labeled_at = datetime.now(timezone.utc).isoformat()

            rooms_list.append({
                'name': name,
                'x': x,
                'y': y,
                'objects': room_objects,
                'labeled_at': labeled_at,
            })

        semantic_map = {
            'rooms': rooms_list,
            'home_position': {'x': 0.0, 'y': 0.0},
            'last_updated': datetime.now(timezone.utc).isoformat(),
        }

        try:
            with open(self._semantic_map_path, 'w') as f:
                json.dump(semantic_map, f, indent=2)
        except OSError as e:
            self.get_logger().warning(f'Failed to save semantic map: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = SemanticMapPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
