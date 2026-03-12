#!/usr/bin/env python3
# encoding: utf-8
"""
World Knowledge System — NetworkX-backed knowledge graph.

Jeeves builds a cumulative understanding of his world across sessions.
Knowledge is stored as a directed graph with two node types:
  - room:   name, x, y, description, first_discovered, times_visited,
            last_visited, searched_for
  - object: name, last_seen, confidence

And two edge types:
  - CONNECTS_TO: between rooms (bidirectional)
  - CONTAINS:    room → object

Persisted to ~/.jeeves/knowledge_graph.json via nx.node_link_data.
Auto-saves on every mutation.
"""
import json
import os
import re
import tempfile
from datetime import datetime

import networkx as nx


_DEFAULT_GRAPH_PATH = os.path.expanduser('~/.jeeves/knowledge_graph.json')

# Rule 10: room-object likelihood mapping for search prioritization
ROOM_OBJECT_LIKELIHOOD: dict[str, set[str]] = {
    'kitchen': {'trash can', 'trash', 'bin', 'refrigerator', 'fridge',
                'microwave', 'oven', 'stove', 'sink', 'cup', 'mug',
                'utensil', 'fork', 'knife', 'spoon', 'bottle', 'food',
                'plate', 'bowl', 'toaster', 'kettle', 'pot', 'pan',
                'dishwasher', 'counter', 'cabinet'},
    'bathroom': {'toilet', 'sink', 'mirror', 'towel', 'soap',
                 'toothbrush', 'shampoo', 'shower', 'bathtub', 'toilet paper',
                 'toiletries', 'trash can', 'bin'},
    'bedroom': {'bed', 'pillow', 'lamp', 'nightstand', 'closet',
                'clothes', 'alarm clock', 'dresser', 'blanket', 'wardrobe'},
    'living room': {'couch', 'sofa', 'tv', 'television', 'remote',
                    'coffee table', 'bookshelf', 'rug', 'cushion', 'speaker'},
    'office': {'desk', 'laptop', 'computer', 'keyboard', 'mouse',
               'monitor', 'chair', 'printer', 'book', 'pen', 'paper',
               'trash can', 'bin'},
    'hallway': {'shoe', 'coat rack', 'umbrella', 'door mat', 'key',
                'coat', 'jacket', 'mirror', 'trash can'},
    'garage': {'tool', 'broom', 'mop', 'bucket', 'cleaning', 'wrench',
               'drill', 'trash can', 'bin', 'recycling'},
    'dining room': {'dining table', 'chair', 'wine glass', 'fork',
                    'knife', 'spoon', 'napkin', 'candle', 'vase'},
    'laundry': {'washing machine', 'dryer', 'detergent', 'basket',
                'iron', 'ironing board', 'clothes'},
    'entrance': {'shoe', 'coat', 'umbrella', 'key', 'door mat',
                 'trash can', 'bin'},
}


class WorldKnowledge:
    """Persistent world knowledge backed by a NetworkX directed graph."""

    def __init__(self, knowledge_dir: str = '', logger=None, *,
                 graph_path: str = ''):
        self.logger = logger
        self._graph_path = graph_path or _DEFAULT_GRAPH_PATH
        os.makedirs(os.path.dirname(self._graph_path), exist_ok=True)
        self._graph: nx.DiGraph = nx.DiGraph()
        self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self):
        """Load graph from JSON. Start empty if file missing or corrupt."""
        if os.path.exists(self._graph_path):
            try:
                with open(self._graph_path, 'r') as f:
                    data = json.load(f)
                self._graph = nx.node_link_graph(data)
                if self.logger:
                    rooms = [n for n, d in self._graph.nodes(data=True)
                             if d.get('type') == 'room']
                    objs = [n for n, d in self._graph.nodes(data=True)
                            if d.get('type') == 'object']
                    self.logger.info(
                        f'Knowledge graph loaded: {len(rooms)} rooms, '
                        f'{len(objs)} objects')
            except (json.JSONDecodeError, OSError, nx.NetworkXError) as e:
                if self.logger:
                    self.logger.warning(
                        f'Failed to load knowledge graph: {e}')
                self._graph = nx.DiGraph()
        else:
            self._graph = nx.DiGraph()

    def save(self):
        """Atomic-write graph to JSON."""
        try:
            data = nx.node_link_data(self._graph)
            dir_name = os.path.dirname(self._graph_path)
            fd, tmp_path = tempfile.mkstemp(
                dir=dir_name, suffix='.tmp', prefix='kg_')
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp_path, self._graph_path)
            except BaseException:
                os.unlink(tmp_path)
                raise
        except OSError as e:
            if self.logger:
                self.logger.warning(f'Failed to save knowledge graph: {e}')

    # ------------------------------------------------------------------
    # Room operations
    # ------------------------------------------------------------------

    def add_room(self, name: str, x: float = 0.0, y: float = 0.0,
                 description: str = '') -> dict:
        """Add or update a room node. Returns the room's attributes."""
        key = name.strip().lower()
        now = datetime.now().isoformat(timespec='seconds')

        if self._graph.has_node(key) and self._graph.nodes[key].get('type') == 'room':
            node = self._graph.nodes[key]
            node['times_visited'] = node.get('times_visited', 0) + 1
            node['last_visited'] = now
            if x != 0.0 or y != 0.0:
                node['x'] = x
                node['y'] = y
            if description:
                node['description'] = description
        else:
            self._graph.add_node(key, **{
                'type': 'room',
                'x': x,
                'y': y,
                'description': description,
                'first_discovered': now,
                'times_visited': 1,
                'last_visited': now,
                'searched_for': {},  # {target_object: timestamp}
            })

        self.save()
        return dict(self._graph.nodes[key])

    def add_object(self, name: str, room: str = '',
                   confidence: float = 0.7,
                   x: float = 0.0, y: float = 0.0) -> dict:
        """Add or update an object node. Optionally link to a room.

        Rule 4: stores position (x, y) so objects can be relocated later.
        Returns True for 'is_new' if this is the first time seeing this object.
        """
        obj_key = f'obj:{name.strip().lower()}'
        room_key = room.strip().lower() if room else ''
        now = datetime.now().isoformat(timespec='seconds')
        is_new = not self._graph.has_node(obj_key)

        if not is_new:
            node = self._graph.nodes[obj_key]
            node['last_seen'] = now
            node['confidence'] = max(node.get('confidence', 0), confidence)
            if x != 0.0 or y != 0.0:
                node['x'] = x
                node['y'] = y
        else:
            self._graph.add_node(obj_key, **{
                'type': 'object',
                'name': name.strip().lower(),
                'last_seen': now,
                'confidence': confidence,
                'x': x,
                'y': y,
            })

        # Create CONTAINS edge from room → object
        if room_key:
            # Remove old CONTAINS edges to this object (object moves rooms)
            old_rooms = [
                u for u, _, d in self._graph.in_edges(obj_key, data=True)
                if d.get('relation') == 'CONTAINS'
            ]
            for old_room in old_rooms:
                self._graph.remove_edge(old_room, obj_key)

            # Ensure room exists
            if not self._graph.has_node(room_key):
                self.add_room(room_key)

            self._graph.add_edge(
                room_key, obj_key, relation='CONTAINS')

        self.save()
        result = dict(self._graph.nodes[obj_key])
        result['is_new'] = is_new
        return result

    def add_connection(self, room_a: str, room_b: str):
        """Add bidirectional CONNECTS_TO edge between two rooms."""
        a = room_a.strip().lower()
        b = room_b.strip().lower()
        if not self._graph.has_node(a):
            self.add_room(a)
        if not self._graph.has_node(b):
            self.add_room(b)
        self._graph.add_edge(a, b, relation='CONNECTS_TO')
        self._graph.add_edge(b, a, relation='CONNECTS_TO')
        self.save()

    # ------------------------------------------------------------------
    # Search tracking
    # ------------------------------------------------------------------

    def mark_room_searched(self, room_name: str, target_object: str,
                           scan_x: float = 0.0, scan_y: float = 0.0,
                           new_objects_found: int = 0):
        """Record that we searched this room for target_object.

        Rule 6: marks room as searched so we never revisit.
        Rule 7: tracks scan positions so large rooms get multiple scans.
        """
        key = room_name.strip().lower()
        if not self._graph.has_node(key):
            return
        node = self._graph.nodes[key]
        if node.get('type') != 'room':
            return
        now = datetime.now().isoformat(timespec='seconds')
        # Mark the target as searched
        searched = node.setdefault('searched_for', {})
        searched[target_object.lower()] = now
        # Track scan positions within the room (Rule 7)
        scans = node.setdefault('scan_positions', [])
        scans.append({
            'x': scan_x, 'y': scan_y,
            'target': target_object.lower(),
            'new_objects': new_objects_found,
            'timestamp': now,
        })
        self.save()

    def is_room_fully_searched(self, room_name: str,
                                target_object: str) -> bool:
        """Check if a room is fully searched for a target.

        Rule 7: A room is fully searched when the last scan found no new objects,
        meaning there's nothing left to discover from the current vantage.
        """
        key = room_name.strip().lower()
        if not self._graph.has_node(key):
            return False
        node = self._graph.nodes[key]
        scans = node.get('scan_positions', [])
        target = target_object.lower()
        target_scans = [s for s in scans if s.get('target') == target]
        if not target_scans:
            return False
        # Fully searched if the last scan found 0 new objects
        return target_scans[-1].get('new_objects', 1) == 0

    def get_search_summary(self, target_object: str) -> dict:
        """Get search status for a target object across all rooms.

        Returns:
            {
                'searched_rooms': ['kitchen', 'hallway'],
                'unsearched_rooms': ['bedroom', 'bathroom'],
                'objects_by_room': {'kitchen': ['cup', 'plate'], ...},
                'suggested_next': 'bedroom' or None,
                'found_in': 'kitchen' or None,
            }
        """
        target = target_object.lower()
        searched = []
        unsearched = []
        objects_by_room = {}
        found_in = None

        for node_id, data in self._graph.nodes(data=True):
            if data.get('type') != 'room':
                continue

            # Check if this room was searched for this target
            room_searched = target in data.get('searched_for', {})
            if room_searched:
                searched.append(node_id)
            else:
                unsearched.append(node_id)

            # Collect objects in this room
            room_objs = []
            for _, obj_id, edge_data in self._graph.out_edges(
                    node_id, data=True):
                if edge_data.get('relation') != 'CONTAINS':
                    continue
                obj_data = self._graph.nodes.get(obj_id, {})
                obj_name = obj_data.get('name', obj_id.removeprefix('obj:'))
                room_objs.append(obj_name)
                # Check if target is here
                if target in obj_name or obj_name in target:
                    found_in = node_id
            if room_objs:
                objects_by_room[node_id] = room_objs

        # Rule 10: prioritize unsearched rooms by object likelihood
        def _room_likelihood(room_name: str) -> int:
            """Lower = higher priority. Rooms likely to contain target sort first."""
            for room_type, likely_objects in ROOM_OBJECT_LIKELIHOOD.items():
                if room_type in room_name:
                    if any(target in obj or obj in target
                           for obj in likely_objects):
                        return 0  # high priority
            return 1  # default
        unsearched.sort(key=_room_likelihood)
        suggested = unsearched[0] if unsearched else None

        return {
            'searched_rooms': searched,
            'unsearched_rooms': unsearched,
            'objects_by_room': objects_by_room,
            'suggested_next': suggested,
            'found_in': found_in,
        }

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    def get_room_for_object(self, object_name: str) -> str | None:
        """Find which room CONTAINS this object. Returns room name or None."""
        obj_key = f'obj:{object_name.strip().lower()}'
        if not self._graph.has_node(obj_key):
            return None
        for u, _, d in self._graph.in_edges(obj_key, data=True):
            if d.get('relation') == 'CONTAINS':
                return u
        return None

    def get_unexplored_connections(self, room_name: str) -> list[str]:
        """Get rooms connected to room_name that haven't been visited."""
        key = room_name.strip().lower()
        if not self._graph.has_node(key):
            return []
        result = []
        for _, neighbor, d in self._graph.out_edges(key, data=True):
            if d.get('relation') != 'CONNECTS_TO':
                continue
            n_data = self._graph.nodes.get(neighbor, {})
            if n_data.get('times_visited', 0) == 0:
                result.append(neighbor)
        return result

    def get_rooms(self) -> dict[str, dict]:
        """Get all rooms as {name: attributes}."""
        return {
            n: dict(d)
            for n, d in self._graph.nodes(data=True)
            if d.get('type') == 'room'
        }

    def get_objects(self) -> dict[str, dict]:
        """Get all objects as {name: attributes}."""
        result = {}
        for n, d in self._graph.nodes(data=True):
            if d.get('type') != 'object':
                continue
            obj_name = d.get('name', n.removeprefix('obj:'))
            obj_data = dict(d)
            # Add room info
            room = self.get_room_for_object(obj_name)
            if room:
                obj_data['room'] = room
            result[obj_name] = obj_data
        return result

    def get_room_objects(self, room_name: str) -> list[str]:
        """Get names of objects in a specific room."""
        key = room_name.strip().lower()
        if not self._graph.has_node(key):
            return []
        result = []
        for _, obj_id, d in self._graph.out_edges(key, data=True):
            if d.get('relation') == 'CONTAINS':
                obj_data = self._graph.nodes.get(obj_id, {})
                result.append(
                    obj_data.get('name', obj_id.removeprefix('obj:')))
        return result

    def get_known_objects_in_room(self, room_name: str) -> set[str]:
        """Get set of known object names in a room (for Rule 8 dedup)."""
        return set(self.get_room_objects(room_name))

    def get_room_connections(self, room_name: str) -> list[str]:
        """Get rooms connected to this room."""
        key = room_name.strip().lower()
        if not self._graph.has_node(key):
            return []
        return [
            v for _, v, d in self._graph.out_edges(key, data=True)
            if d.get('relation') == 'CONNECTS_TO'
        ]

    # ------------------------------------------------------------------
    # Compatibility properties for callers using old dict API
    # ------------------------------------------------------------------

    @property
    def world_map(self) -> dict:
        """Compatibility: returns {'rooms': {...}} dict view.

        Callers using world_map.get('rooms', {}) will get a dict of
        room_name → room_attributes (including 'position' and 'connections').
        """
        rooms = {}
        for node_id, data in self._graph.nodes(data=True):
            if data.get('type') != 'room':
                continue
            room = dict(data)
            room['position'] = {'x': room.get('x', 0.0),
                                'y': room.get('y', 0.0)}
            room['connections'] = self.get_room_connections(node_id)
            rooms[node_id] = room
        return {'rooms': rooms}

    @property
    def known_objects(self) -> dict:
        """Compatibility: returns {'objects': {...}} dict view."""
        return {'objects': self.get_objects()}

    @property
    def learned_behaviors(self) -> dict:
        """Compatibility: returns empty learned behaviors."""
        return {'navigation_lessons': [], 'timing_patterns': [],
                'surface_types': {}}

    # ------------------------------------------------------------------
    # Legacy compatibility methods
    # ------------------------------------------------------------------

    def _update_object(self, name: str, timestamp: str,
                       odom: dict | None = None,
                       category: str | None = None):
        """Legacy: called by tool_handlers.py identify_objects auto-register."""
        # Determine room from nearby rooms based on odom
        room = ''
        if odom:
            room = self.nearest_room(
                odom.get('x', 0.0), odom.get('y', 0.0))
        self.add_object(name, room=room, confidence=0.7)

    def nearest_room(self, x: float, y: float,
                     max_dist: float = 3.0) -> str:
        """Find the nearest room to (x, y) within max_dist.

        Public API. Legacy alias ``_nearest_room`` is kept for
        backward compatibility with existing mocks and callers.
        """
        best_room = ''
        best_dist = max_dist
        for node_id, data in self._graph.nodes(data=True):
            if data.get('type') != 'room':
                continue
            rx = data.get('x', 0.0)
            ry = data.get('y', 0.0)
            dist = ((x - rx) ** 2 + (y - ry) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_room = node_id
        return best_room

    # Backward-compatible alias (mocks and older callers use the private name)
    _nearest_room = nearest_room

    def update_from_response(self, result: dict, odom: dict | None = None):
        """Legacy: cheap per-cycle update via keyword extraction.

        Still useful for the non-agent exploration loop.
        """
        speech = result.get('speech', '')
        reasoning = result.get('reasoning', '')
        text = f'{speech} {reasoning}'.lower()
        if not text.strip():
            return

        now = datetime.now().isoformat(timespec='seconds')
        x = odom.get('x', 0.0) if odom else 0.0
        y = odom.get('y', 0.0) if odom else 0.0

        room_patterns = [
            r'\b(kitchen|bathroom|bedroom|living room|hallway|corridor|'
            r'closet|garage|office|dining room|laundry|basement|attic|'
            r'foyer|entryway|pantry|study|den|nursery|guest room|'
            r'storage room|utility room)\b'
        ]
        for pattern in room_patterns:
            for room in re.findall(pattern, text):
                self.add_room(room, x=x, y=y)

    def end_of_session_update(self, memory, llm_provider):
        """Legacy: end-of-session LLM knowledge update."""
        self.save()

    # ------------------------------------------------------------------
    # Prompt context for LLM
    # ------------------------------------------------------------------

    def get_prompt_context(self, x: float = 0.0, y: float = 0.0,
                           theta: float = 0.0) -> str:
        """Spatially-filtered knowledge summary for the user prompt.

        Returns only known rooms + their objects + connections.
        Target: <200 tokens.
        """
        lines = []
        rooms = self.get_rooms()
        if rooms:
            room_names = list(rooms.keys())[:8]
            lines.append(f'KNOWN ROOMS: {", ".join(room_names)}')

            # Find nearest room to current position
            nearest = self.nearest_room(x, y, max_dist=50.0)
            if nearest and nearest in rooms:
                info = rooms[nearest]
                desc = info.get('description', '')[:80]
                conns = self.get_room_connections(nearest)
                objs = self.get_room_objects(nearest)
                lines.append(f'NEAREST ROOM: {nearest} — {desc}')
                if conns:
                    lines.append(f'  Connects to: {", ".join(conns[:4])}')
                if objs:
                    lines.append(f'  Contains: {", ".join(objs[:6])}')

        objects = self.get_objects()
        if objects:
            recent = sorted(
                objects.items(),
                key=lambda kv: kv[1].get('last_seen', ''),
                reverse=True,
            )[:5]
            obj_parts = []
            for name, info in recent:
                room = info.get('room', '')
                obj_parts.append(
                    f'{name} (in {room})' if room else name)
            lines.append(f'KNOWN OBJECTS: {"; ".join(obj_parts)}')

        return '\n'.join(lines) if lines else ''

    # ------------------------------------------------------------------
    # Summary methods for CLI tool
    # ------------------------------------------------------------------

    def get_rooms_summary(self) -> str:
        rooms = self.get_rooms()
        if not rooms:
            return 'No rooms discovered yet.'
        lines = []
        for name, info in sorted(rooms.items()):
            visits = info.get('times_visited', 0)
            desc = info.get('description', 'No description')[:60]
            conns = self.get_room_connections(name)
            objs = self.get_room_objects(name)
            lines.append(f'  {name}: {visits} visits — {desc}')
            if conns:
                lines.append(f'    Connects to: {", ".join(conns)}')
            if objs:
                lines.append(f'    Objects: {", ".join(objs)}')
        return '\n'.join(lines)

    def get_objects_summary(self) -> str:
        objects = self.get_objects()
        if not objects:
            return 'No objects catalogued yet.'
        lines = []
        for name, info in sorted(objects.items()):
            conf = info.get('confidence', 0)
            room = info.get('room', 'unknown')
            lines.append(f'  {name}: conf={conf:.0%}, room={room}')
        return '\n'.join(lines)

    def get_lessons_summary(self) -> str:
        return 'No navigation lessons (deprecated).'
