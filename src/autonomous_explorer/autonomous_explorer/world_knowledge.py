#!/usr/bin/env python3
# encoding: utf-8
"""
World Knowledge System — persistent rooms, objects, and learned behaviors.

Jeeves builds a cumulative understanding of his world across sessions.
Knowledge is stored in three JSON files under ~/mentorpi_explorer/knowledge/:
  - world_map.json    — rooms, corridors, doors, connections, landmarks
  - known_objects.json — objects with locations, frequency, categories
  - learned_behaviors.json — navigation lessons, surface types, timing patterns

Update strategy (dual-track):
  - Per-cycle: cheap string matching on LLM speech for room/object mentions (zero cost)
  - End-of-session: one LLM call to produce structured knowledge update (~$0.01)
"""
import json
import os
import re
import time
from datetime import datetime


class WorldKnowledge:
    """Persistent world knowledge that grows across exploration sessions."""

    def __init__(self, knowledge_dir: str, logger=None):
        self.knowledge_dir = os.path.expanduser(knowledge_dir)
        self.logger = logger

        os.makedirs(self.knowledge_dir, exist_ok=True)

        self._map_path = os.path.join(self.knowledge_dir, 'world_map.json')
        self._objects_path = os.path.join(self.knowledge_dir, 'known_objects.json')
        self._behaviors_path = os.path.join(self.knowledge_dir, 'learned_behaviors.json')

        self.world_map = self._load_json(self._map_path, {
            'rooms': {},
            'corridors': [],
            'doors': [],
        })
        self.known_objects = self._load_json(self._objects_path, {
            'objects': {},
        })
        self.learned_behaviors = self._load_json(self._behaviors_path, {
            'navigation_lessons': [],
            'timing_patterns': [],
            'surface_types': {},
        })

    def _load_json(self, path: str, default: dict) -> dict:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                # Merge defaults for forward compat
                for key, val in default.items():
                    if key not in data:
                        data[key] = val
                return data
            except (json.JSONDecodeError, OSError):
                pass
        return default.copy()

    def get_prompt_context(self, x: float = 0.0, y: float = 0.0,
                           theta: float = 0.0) -> str:
        """Spatially-filtered knowledge summary for the user prompt.

        Returns only the current room + adjacent areas + nearby objects.
        Target: <200 tokens to keep prompt costs low.
        """
        lines = []

        # Rooms summary
        rooms = self.world_map.get('rooms', {})
        if rooms:
            room_names = list(rooms.keys())[:6]
            lines.append(f'KNOWN ROOMS: {", ".join(room_names)}')

            # Find "current" room — the one with the most recent visit
            # (positional matching would require map coordinates, which we
            # don't have precisely — so use most-recently-visited heuristic)
            latest_room = None
            latest_time = ''
            for name, info in rooms.items():
                lv = info.get('last_visited', '')
                if lv > latest_time:
                    latest_time = lv
                    latest_room = name
            if latest_room:
                room_info = rooms[latest_room]
                desc = room_info.get('description', '')[:80]
                connections = room_info.get('connections', [])
                lines.append(
                    f'LAST ROOM: {latest_room} — {desc}'
                )
                if connections:
                    lines.append(f'  Connects to: {", ".join(connections[:4])}')

        # Recent objects (last 5 seen)
        objects = self.known_objects.get('objects', {})
        if objects:
            recent_objs = sorted(
                objects.items(),
                key=lambda kv: kv[1].get('last_seen', ''),
                reverse=True,
            )[:5]
            obj_parts = []
            for name, info in recent_objs:
                loc = info.get('usual_location', '')
                cat = info.get('category', '')
                obj_parts.append(f'{name} ({cat}, {loc})' if loc else f'{name} ({cat})')
            if obj_parts:
                lines.append(f'KNOWN OBJECTS: {"; ".join(obj_parts)}')

        # Relevant navigation lessons (last 3)
        lessons = self.learned_behaviors.get('navigation_lessons', [])
        if lessons:
            recent = lessons[-3:]
            lesson_parts = [l.get('lesson', '')[:60] for l in recent]
            lines.append('LESSONS: ' + ' | '.join(lesson_parts))

        return '\n'.join(lines) if lines else ''

    def update_from_response(self, result: dict, odom: dict | None = None):
        """Cheap per-cycle update via keyword extraction from LLM response.

        Extracts room and object mentions from speech/reasoning fields.
        No LLM call — pure string matching.
        """
        speech = result.get('speech', '')
        reasoning = result.get('reasoning', '')
        text = f'{speech} {reasoning}'.lower()

        now = datetime.now().isoformat(timespec='seconds')

        # Detect room mentions
        room_patterns = [
            r'\b(kitchen|bathroom|bedroom|living room|hallway|corridor|'
            r'closet|garage|office|dining room|laundry|basement|attic|'
            r'foyer|entryway|pantry|study|den|nursery|guest room|'
            r'storage room|utility room)\b'
        ]
        for pattern in room_patterns:
            matches = re.findall(pattern, text)
            for room in matches:
                self._update_room(room, now, odom)

        # Detect object mentions — simple heuristic
        object_patterns = [
            (r'\b(chair|table|desk|couch|sofa|shelf|bookshelf|refrigerator|'
             r'tv|television|monitor|lamp|plant|rug|carpet|door|window|'
             r'bed|cabinet|mirror|painting|clock|fan|box|bag|shoe|'
             r'cat|dog|person|human)\b', None),
        ]
        for pattern, category in object_patterns:
            matches = re.findall(pattern, text)
            for obj in set(matches):
                self._update_object(obj, now, odom, category)

    def _update_room(self, room: str, timestamp: str,
                     odom: dict | None = None):
        rooms = self.world_map.setdefault('rooms', {})
        if room in rooms:
            rooms[room]['times_visited'] = rooms[room].get('times_visited', 0) + 1
            rooms[room]['last_visited'] = timestamp
        else:
            rooms[room] = {
                'first_discovered': timestamp,
                'times_visited': 1,
                'last_visited': timestamp,
                'description': '',
                'connections': [],
                'landmarks': [],
                'notes': '',
                'confidence': 0.3,
            }

    def _update_object(self, obj: str, timestamp: str,
                       odom: dict | None = None,
                       category: str | None = None):
        objects = self.known_objects.setdefault('objects', {})
        location = ''
        if odom:
            location = f'({odom.get("x", 0):.1f}, {odom.get("y", 0):.1f})'

        # Auto-categorize
        if not category:
            living = {'cat', 'dog', 'person', 'human'}
            category = 'living_being' if obj in living else 'object'

        if obj in objects:
            objects[obj]['times_seen'] = objects[obj].get('times_seen', 0) + 1
            objects[obj]['last_seen'] = timestamp
            if location:
                objects[obj]['last_location'] = location
        else:
            objects[obj] = {
                'first_seen': timestamp,
                'times_seen': 1,
                'last_seen': timestamp,
                'usual_location': location,
                'last_location': location,
                'description': '',
                'is_dynamic': category == 'living_being',
                'category': category,
            }

    def end_of_session_update(self, memory, llm_provider):
        """One LLM call at shutdown for structured knowledge summary.

        Asks the LLM to review the session and produce knowledge updates.
        """
        if llm_provider.provider_name == 'dryrun':
            self.save()
            return

        # Build a summary of the session for the LLM
        recent_actions = list(memory.action_log)[-10:]
        action_text = '\n'.join(
            f'- {a.get("action", "?")} at ({a.get("x", 0):.1f},{a.get("y", 0):.1f}): '
            f'{a.get("speech", "")[:60]}'
            for a in recent_actions
        )

        discoveries_text = '\n'.join(
            f'- {d["description"][:80]}'
            for d in memory.discoveries[-10:]
        ) if memory.discoveries else 'None'

        prompt = (
            f'You are Jeeves. After this outing, summarize what you learned.\n\n'
            f'Recent actions:\n{action_text}\n\n'
            f'Discoveries:\n{discoveries_text}\n\n'
            f'Current known rooms: {list(self.world_map.get("rooms", {}).keys())}\n\n'
            f'Respond in JSON:\n'
            f'{{"rooms_visited": ["room names"], '
            f'"new_objects": [{{"name": "x", "category": "furniture/living_being/object", '
            f'"location": "where"}}], '
            f'"lessons": ["navigation lessons learned"], '
            f'"room_connections": [{{"from": "room1", "to": "room2"}}]}}'
        )

        try:
            result = llm_provider.analyze_scene(
                '',
                'You are Jeeves. Summarize exploration knowledge. Respond in JSON only.',
                prompt,
            )
            meta = result.pop('_meta', {})
            raw = meta.get('raw_response', '')

            # Try parsing the response as knowledge update
            import re as _re
            brace_match = _re.search(r'\{[\s\S]*\}', raw)
            if brace_match:
                try:
                    update = json.loads(brace_match.group())
                    self._apply_knowledge_update(update)
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            if self.logger:
                self.logger.warning(f'Knowledge update LLM call failed: {e}')

        self.save()

    def _apply_knowledge_update(self, update: dict):
        """Apply a structured knowledge update from the LLM."""
        now = datetime.now().isoformat(timespec='seconds')

        # Rooms
        for room in update.get('rooms_visited', []):
            if isinstance(room, str):
                self._update_room(room.lower(), now)

        # Objects
        for obj in update.get('new_objects', []):
            if isinstance(obj, dict):
                name = obj.get('name', '').lower()
                if name:
                    self._update_object(
                        name, now, category=obj.get('category', 'object')
                    )

        # Lessons
        for lesson in update.get('lessons', []):
            if isinstance(lesson, str) and lesson.strip():
                self.learned_behaviors.setdefault('navigation_lessons', []).append({
                    'learned_on': now[:10],
                    'lesson': lesson.strip(),
                    'confidence': 0.5,
                    'times_confirmed': 1,
                })

        # Room connections
        for conn in update.get('room_connections', []):
            if isinstance(conn, dict):
                from_room = conn.get('from', '').lower()
                to_room = conn.get('to', '').lower()
                if from_room and to_room:
                    rooms = self.world_map.get('rooms', {})
                    if from_room in rooms:
                        conns = rooms[from_room].setdefault('connections', [])
                        if to_room not in conns:
                            conns.append(to_room)
                    if to_room in rooms:
                        conns = rooms[to_room].setdefault('connections', [])
                        if from_room not in conns:
                            conns.append(from_room)

    def save(self):
        """Persist all knowledge files to disk."""
        self._save_json(self._map_path, self.world_map)
        self._save_json(self._objects_path, self.known_objects)
        self._save_json(self._behaviors_path, self.learned_behaviors)

    def _save_json(self, path: str, data: dict):
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            if self.logger:
                self.logger.warning(f'Failed to save {path}: {e}')

    # ------------------------------------------------------------------
    # Summary methods for CLI tool
    # ------------------------------------------------------------------

    def get_rooms_summary(self) -> str:
        rooms = self.world_map.get('rooms', {})
        if not rooms:
            return 'No rooms discovered yet.'
        lines = []
        for name, info in sorted(rooms.items()):
            visits = info.get('times_visited', 0)
            desc = info.get('description', 'No description')[:60]
            conns = info.get('connections', [])
            lines.append(f'  {name}: {visits} visits — {desc}')
            if conns:
                lines.append(f'    Connects to: {", ".join(conns)}')
        return '\n'.join(lines)

    def get_objects_summary(self) -> str:
        objects = self.known_objects.get('objects', {})
        if not objects:
            return 'No objects catalogued yet.'
        lines = []
        for name, info in sorted(objects.items()):
            cat = info.get('category', 'unknown')
            seen = info.get('times_seen', 0)
            loc = info.get('usual_location', 'unknown')
            lines.append(f'  {name} [{cat}]: seen {seen}x, location: {loc}')
        return '\n'.join(lines)

    def get_lessons_summary(self) -> str:
        lessons = self.learned_behaviors.get('navigation_lessons', [])
        if not lessons:
            return 'No navigation lessons learned yet.'
        lines = []
        for l in lessons:
            conf = l.get('confidence', 0)
            lines.append(f'  [{conf:.0%}] {l.get("lesson", "")}')
        return '\n'.join(lines)
