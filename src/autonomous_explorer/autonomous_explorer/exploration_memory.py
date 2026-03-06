#!/usr/bin/env python3
# encoding: utf-8
"""
Exploration memory for the autonomous explorer.

Maintains a rolling log of decisions, observations, visited areas,
and a lightweight spatial map built from odometry + LiDAR data.
This context is included in each LLM prompt so the robot builds
situational awareness over time.
"""
import json
import math
import os
import time
from collections import deque


# Spatial map grid resolution in meters
_GRID_CELL = 0.25


def _grid_key(x: float, y: float) -> str:
    """Quantize a world coordinate to a grid cell key."""
    gx = round(x / _GRID_CELL) * _GRID_CELL
    gy = round(y / _GRID_CELL) * _GRID_CELL
    return f'{gx:.2f},{gy:.2f}'


class ExplorationMemory:
    """Lightweight memory of exploration history.

    Stores recent actions, observations, a list of notable objects
    or areas discovered, and a spatial map of visited/obstacle cells.
    """

    def __init__(self, filepath: str, max_entries: int = 200):
        self.filepath = filepath
        self.max_entries = max_entries
        # Rolling log of recent decisions (kept in RAM, last N entries)
        self.action_log: deque = deque(maxlen=30)
        # Notable objects / discoveries
        self.discoveries: list[dict] = []
        # Simple movement history for direction tracking
        self.movement_history: deque = deque(maxlen=50)
        # Cumulative stats
        self.total_actions = 0
        self.start_time = time.time()

        # Spatial map: sets of grid cell keys
        self.visited_cells: set[str] = set()
        self.obstacle_cells: set[str] = set()

        self._load()

    def _load(self):
        """Load existing memory from disk if available."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                self.discoveries = data.get('discoveries', [])
                self.total_actions = data.get('total_actions', 0)
                self.start_time = data.get('start_time', time.time())
                for entry in data.get('action_log', []):
                    self.action_log.append(entry)
                self.visited_cells = set(data.get('visited_cells', []))
                self.obstacle_cells = set(data.get('obstacle_cells', []))
            except (json.JSONDecodeError, KeyError):
                pass  # Start fresh if file is corrupted

    def save(self):
        """Persist current memory to disk."""
        data = {
            'start_time': self.start_time,
            'total_actions': self.total_actions,
            'discoveries': self.discoveries[-self.max_entries:],
            'action_log': list(self.action_log),
            'visited_cells': list(self.visited_cells),
            'obstacle_cells': list(self.obstacle_cells),
        }
        try:
            with open(self.filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass  # Non-critical — memory is primarily in RAM

    def record_action(
        self,
        llm_response: dict,
        sensor_summary: str,
        odom: dict | None = None,
        lidar_sectors: dict | None = None,
    ):
        """Record an exploration action and its context.

        Args:
            llm_response: Parsed LLM JSON response.
            sensor_summary: Text summary of sensors (for legacy compat).
            odom: Odometry dict with keys 'x', 'y', 'theta' (radians).
            lidar_sectors: Dict with keys 'front', 'left', 'right', 'back'
                           (distances in meters).
        """
        entry = {
            'time': time.time(),
            'elapsed': round(time.time() - self.start_time, 1),
            'action': llm_response.get('action', 'unknown'),
            'speed': llm_response.get('speed', 0),
            'duration': llm_response.get('duration', 0),
            'speech': llm_response.get('speech', ''),
            'reasoning': llm_response.get('reasoning', ''),
        }

        # Attach position if available
        if odom:
            entry['x'] = odom.get('x', 0)
            entry['y'] = odom.get('y', 0)
            entry['heading'] = round(math.degrees(odom.get('theta', 0)), 1)

            # Update spatial map — mark current cell as visited
            self.visited_cells.add(_grid_key(entry['x'], entry['y']))

            # Mark obstacle cells from LiDAR
            if lidar_sectors:
                theta = odom.get('theta', 0)
                self._mark_obstacles(
                    entry['x'], entry['y'], theta, lidar_sectors,
                )

        self.action_log.append(entry)
        self.total_actions += 1
        self.movement_history.append(entry['action'])

        # Auto-detect discoveries from speech content
        speech = entry['speech'].lower()
        discovery_keywords = [
            'see', 'found', 'notice', 'spot', 'discover',
            'interesting', 'detect', 'observe',
        ]
        if any(kw in speech for kw in discovery_keywords):
            disc = {
                'time': entry['time'],
                'description': entry['speech'],
                'action_number': self.total_actions,
            }
            if odom:
                disc['x'] = odom.get('x', 0)
                disc['y'] = odom.get('y', 0)
            self.discoveries.append(disc)

        # Save every 10 actions
        if self.total_actions % 10 == 0:
            self.save()

    def _mark_obstacles(
        self, rx: float, ry: float, theta: float,
        sectors: dict,
    ):
        """Project LiDAR sector distances into obstacle grid cells."""
        # Sector angles relative to robot heading (radians)
        sector_angles = {
            'front': 0.0,
            'left': math.pi / 2,
            'right': -math.pi / 2,
            'back': math.pi,
        }
        for sector, angle_offset in sector_angles.items():
            dist = sectors.get(sector, 999)
            if dist < 2.0:  # Only mark nearby obstacles
                abs_angle = theta + angle_offset
                ox = rx + dist * math.cos(abs_angle)
                oy = ry + dist * math.sin(abs_angle)
                self.obstacle_cells.add(_grid_key(ox, oy))

    def get_context_summary(self) -> str:
        """Generate a text summary for inclusion in LLM prompts."""
        lines = []
        elapsed = round(time.time() - self.start_time)
        lines.append(
            f"Exploration time: {elapsed}s | Total actions: {self.total_actions}"
        )

        # Recent action history with reasoning and position
        if self.action_log:
            recent = list(self.action_log)[-5:]
            lines.append("Recent action history:")
            for e in recent:
                pos_str = ''
                if 'x' in e:
                    pos_str = f' at ({e["x"]:.2f},{e["y"]:.2f}) hdg={e.get("heading", 0)}°'
                reason = e.get('reasoning', '')
                if len(reason) > 80:
                    reason = reason[:77] + '...'
                lines.append(
                    f"  #{e.get('elapsed', 0)}s: {e['action']} "
                    f"spd={e.get('speed', 0)} dur={e.get('duration', 0)}"
                    f"{pos_str} — {reason}"
                )

        # Spatial map summary
        if self.visited_cells:
            lines.append(
                f"Spatial map: {len(self.visited_cells)} cells visited, "
                f"{len(self.obstacle_cells)} obstacle cells detected "
                f"(grid={_GRID_CELL}m)"
            )
            # Show bounding box of explored area
            xs, ys = [], []
            for cell in self.visited_cells:
                cx, cy = cell.split(',')
                xs.append(float(cx))
                ys.append(float(cy))
            lines.append(
                f"Explored area: x=[{min(xs):.2f},{max(xs):.2f}] "
                f"y=[{min(ys):.2f},{max(ys):.2f}] m"
            )

            # Suggest unexplored directions from current position
            if self.action_log:
                last = list(self.action_log)[-1]
                if 'x' in last:
                    unexplored = self._get_unexplored_directions(
                        last['x'], last['y'],
                    )
                    if unexplored:
                        lines.append(
                            f"Unexplored directions: {', '.join(unexplored)}"
                        )

        # Recent discoveries with location
        if self.discoveries:
            recent_disc = self.discoveries[-3:]
            lines.append("Discoveries:")
            for d in recent_disc:
                loc = ''
                if 'x' in d:
                    loc = f' at ({d["x"]:.2f},{d["y"]:.2f})'
                desc = d['description'][:60]
                lines.append(f"  - {desc}{loc}")

        # Detect if stuck (same action repeated many times)
        if len(self.movement_history) >= 5:
            last_5 = list(self.movement_history)[-5:]
            if len(set(last_5)) == 1 and last_5[0] != 'stop':
                lines.append(
                    f"WARNING: You seem stuck repeating '{last_5[0]}'. "
                    f"Try a different approach!"
                )

        return '\n'.join(lines)

    def _get_unexplored_directions(self, x: float, y: float) -> list[str]:
        """Check which neighboring cells haven't been visited."""
        directions = []
        checks = [
            ('north (+y)', 0, _GRID_CELL * 2),
            ('south (-y)', 0, -_GRID_CELL * 2),
            ('east (+x)', _GRID_CELL * 2, 0),
            ('west (-x)', -_GRID_CELL * 2, 0),
        ]
        for name, dx, dy in checks:
            key = _grid_key(x + dx, y + dy)
            if key not in self.visited_cells and key not in self.obstacle_cells:
                directions.append(name)
        return directions

    def reset(self):
        """Clear all memory and start fresh."""
        self.action_log.clear()
        self.discoveries.clear()
        self.movement_history.clear()
        self.visited_cells.clear()
        self.obstacle_cells.clear()
        self.total_actions = 0
        self.start_time = time.time()
        self.save()
