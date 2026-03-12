#!/usr/bin/env python3
"""
Semantic Map Publisher Node.

Reads from the WorldKnowledge NetworkX knowledge graph on disk and publishes
ROS2 MarkerArray messages for Foxglove visualization. Judges see room labels
appearing on the map in real-time as Jeeves explores and labels new rooms.

Published topics:
  /semantic_map/rooms    — visualization_msgs/MarkerArray (TEXT_VIEW_FACING)
  /semantic_map/objects  — visualization_msgs/MarkerArray (TEXT_VIEW_FACING)

Note: /semantic_map/agent_status is published by explorer_node directly (not here).

Persistence:
  ~/.jeeves/knowledge_graph.json — NetworkX node_link_data graph
"""
import json
import os
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

from autonomous_explorer.world_knowledge import _DEFAULT_GRAPH_PATH


class SemanticMapPublisher(Node):

    def __init__(self):
        super().__init__('semantic_map_publisher')

        # Parameters
        self.declare_parameter('graph_path', _DEFAULT_GRAPH_PATH)
        self.declare_parameter('publish_rate', 1.0)

        self._graph_path = self.get_parameter(
            'graph_path').get_parameter_value().string_value
        self._publish_rate = self.get_parameter(
            'publish_rate').get_parameter_value().double_value

        # Cache: avoid re-publishing if nothing changed
        self._last_mtime = 0.0
        self._rooms = {}   # {name: {x, y, times_visited, description, ...}}
        self._objects = {}  # {name: {room, confidence, ...}}

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
        # NOTE: /semantic_map/agent_status is published directly by
        # explorer_node via AgentLogger (TRANSIENT_LOCAL QoS).
        # Do NOT create a second publisher here — dual publishers with
        # different QoS on the same topic confuses Foxglove's subscriber.

        # Load on startup
        self._reload_graph()

        # 1Hz timer
        period = 1.0 / self._publish_rate
        self.create_timer(period, self._timer_cb)

        self.get_logger().info(
            f'Semantic map publisher started '
            f'(graph_path={self._graph_path}, '
            f'rate={self._publish_rate}Hz)')

    # ── Timer callback ──

    def _timer_cb(self):
        """Read knowledge graph, publish markers."""
        self._reload_graph()
        self._publish_room_markers()
        self._publish_object_markers()

    # ── Knowledge graph reading ──

    def _reload_graph(self) -> bool:
        """Re-read knowledge graph JSON if modified. Returns True if changed."""
        try:
            mtime = os.path.getmtime(self._graph_path)
            if mtime == self._last_mtime:
                return False
        except FileNotFoundError:
            return False

        try:
            with open(self._graph_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return False

        self._last_mtime = mtime

        # Parse node_link_data format: nodes have 'id' and attributes
        rooms = {}
        objects = {}
        nodes_by_id = {}

        for node in data.get('nodes', []):
            node_id = node.get('id', '')
            node_type = node.get('type', '')
            nodes_by_id[node_id] = node

            if node_type == 'room':
                rooms[node_id] = {
                    'x': float(node.get('x', 0.0)),
                    'y': float(node.get('y', 0.0)),
                    'times_visited': node.get('times_visited', 1),
                    'description': node.get('description', ''),
                    'first_discovered': node.get('first_discovered', ''),
                }
            elif node_type == 'object':
                obj_name = node.get('name', node_id.removeprefix('obj:'))
                objects[obj_name] = {
                    'confidence': node.get('confidence', 0.0),
                    'last_seen': node.get('last_seen', ''),
                    'node_id': node_id,
                }

        # Find room for each object via CONTAINS edges
        for link in data.get('links', []):
            source = link.get('source', '')
            target = link.get('target', '')
            relation = link.get('relation', '')
            if relation == 'CONTAINS':
                # source=room, target=obj:name
                src_node = nodes_by_id.get(source, {})
                tgt_node = nodes_by_id.get(target, {})
                if src_node.get('type') == 'room':
                    obj_name = tgt_node.get(
                        'name', target.removeprefix('obj:'))
                    if obj_name in objects:
                        objects[obj_name]['room'] = source
                        # Use room coords for object position
                        objects[obj_name]['x'] = float(
                            src_node.get('x', 0.0))
                        objects[obj_name]['y'] = float(
                            src_node.get('y', 0.0))

        self._rooms = rooms
        self._objects = objects
        return True

    # ── MarkerArray publishing ──

    def _publish_room_markers(self):
        """Publish TEXT_VIEW_FACING markers for each labeled room."""
        if not self._rooms:
            return

        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        for idx, (name, info) in enumerate(self._rooms.items()):
            x = info.get('x', 0.0)
            y = info.get('y', 0.0)

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
            visits = info.get('times_visited', 1)
            label = name.title()
            if isinstance(visits, int) and visits > 1:
                label += f' ({visits}x)'
            marker.text = label
            marker.lifetime.sec = 0  # Never expire

            marker_array.markers.append(marker)

        self._rooms_pub.publish(marker_array)

    def _publish_object_markers(self):
        """Publish TEXT_VIEW_FACING markers for each registered object."""
        if not self._objects:
            return

        marker_array = MarkerArray()
        now = self.get_clock().now().to_msg()

        idx = 0
        for name, info in self._objects.items():
            # Only show objects that have a position (via room)
            if 'x' not in info or 'y' not in info:
                continue

            x = info['x']
            y = info['y']

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
            room = info.get('room', '')
            label = name
            if room:
                label += f' [{room}]'
            marker.text = label
            marker.lifetime.sec = 0

            marker_array.markers.append(marker)
            idx += 1

        self._objects_pub.publish(marker_array)


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
