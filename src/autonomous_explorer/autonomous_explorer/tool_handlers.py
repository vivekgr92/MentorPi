#!/usr/bin/env python3
# encoding: utf-8
"""
Tool Handlers — implements the 7 registered Jeeves tool functions.

Each handler method takes keyword arguments matching the tool's JSON schema
and returns a dict with at least {'success': bool, ...}.

These handlers bridge between the LLM's tool calls and the robot's
actual capabilities (Nav2, motor control, VLM, knowledge graph, voice).

Usage:
    handlers = ToolHandlers(node)
    handlers.bind_to_registry(registry)
    # Now registry.execute('navigate_to', {'target': 'kitchen'}) works
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime

from autonomous_explorer.config import (
    NAV2_TOOL_REEVAL_INTERVAL,
    NAV2_TOOL_TIMEOUT,
)


class ToolHandlers:
    """Implements the 7 registered Jeeves tool handler methods.

    Takes a reference to the AutonomousExplorer node to access
    sensors, actuators, Nav2, world knowledge, voice, and LLM.

    Legacy handlers (move_direct, look_around, describe_scene, etc.)
    remain as methods but are NOT bound to the registry.
    """

    def __init__(self, node):
        self._node = node
        self._log = node.get_logger()

    def bind_to_registry(self, registry) -> None:
        """Bind the 7 demo tool handlers to the registry.

        Removed tools (move_direct, look_around, describe_scene,
        check_surroundings, register_object, save_map, listen) still
        exist as methods but are not registered — the LLM cannot call them.
        """
        bindings = {
            'navigate_to': self.navigate_to,
            'explore_frontier': self.explore_frontier,
            'go_home': self.go_home,
            'identify_objects': self.identify_objects,
            'label_room': self.label_room,
            'query_knowledge': self.query_knowledge,
            'speak': self.speak,
        }
        for name, handler in bindings.items():
            tool = registry.get_tool(name)
            if tool:
                tool.handler = handler
            else:
                self._log.warning(f'Tool {name} not found in registry')

    # ------------------------------------------------------------------
    # Navigation tools
    # ------------------------------------------------------------------

    def navigate_to(self, target: str, x: float = 0.0, y: float = 0.0,
                    object_name: str = '', speech: str = '') -> dict:
        """Navigate to a named location or coordinates.

        If object_name is set, after Nav2 arrives at the area the robot
        does a LiDAR+depth guided final approach and stops ~10cm from the
        object.
        """
        if speech:
            self._speak_async(speech)

        # If object_name is set and target is "approach" (or object is nearby),
        # skip Nav2 and go straight to YOLO-guided approach
        if object_name and target == 'approach':
            approach = self._approach_object(object_name)
            return {
                'success': approach.get('reached', False),
                'status': 'reached' if approach.get('reached') else 'approach_failed',
                'approach': approach,
            }

        # Resolve named location from world knowledge
        if target != 'coordinates':
            room = self._node.world_knowledge.world_map.get('rooms', {}).get(
                target.lower(), None)
            if room and room.get('position'):
                pos = room['position']
                x = pos.get('x', 0.0)
                y = pos.get('y', 0.0)
                self._log.info(f'Resolved "{target}" to ({x:.2f}, {y:.2f})')
            else:
                return {
                    'success': False,
                    'error': f'Unknown location: {target}. '
                             f'Known rooms: {list(self._node.world_knowledge.world_map.get("rooms", {}).keys())}',
                }

        # Use Nav2 if available
        if self._node.use_nav2 and self._node.nav2:
            accepted = self._node.nav2.navigate_to(x, y)
            if not accepted:
                return {'success': False, 'error': 'Nav2 goal not accepted'}

            # Wait for navigation with timeout, checking e-stop
            t_start = time.time()
            while (self._node.nav2.is_navigating
                   and self._node.running
                   and time.time() - t_start < NAV2_TOOL_TIMEOUT):
                if self._node.emergency_stop:
                    self._node.nav2.cancel_navigation()
                    return {'success': False, 'error': 'Emergency stop during navigation'}
                if self._node._voice_interrupt:
                    self._node.nav2.cancel_navigation()
                    return {'success': False, 'error': 'Interrupted by new voice command'}
                # Return after interval so LLM can re-evaluate
                if time.time() - t_start >= NAV2_TOOL_REEVAL_INTERVAL:
                    fb = self._node.nav2.navigation_feedback or {}
                    return {
                        'success': True,
                        'status': 'in_progress',
                        'distance_remaining': fb.get('distance_remaining', '?'),
                    }
                time.sleep(0.5)

            result = self._node.nav2.navigation_result
            if result != 'succeeded':
                return {
                    'success': False,
                    'status': result or 'timeout',
                }

            # Nav2 arrived — do final object approach if requested
            if object_name:
                approach = self._approach_object(object_name)
                return {
                    'success': True,
                    'status': 'arrived',
                    'approach': approach,
                }

            return {'success': True, 'status': 'arrived'}

        # Fallback: no Nav2 available
        return {
            'success': False,
            'error': 'Nav2 not available. Use move_direct for short movements.',
        }

    def _approach_object(self, object_name: str) -> dict:
        """Sensor-guided final approach — stop ~10cm from the object.

        Uses LiDAR front sector (primary, 10Hz) + depth camera center ROI
        (secondary) to measure distance. No YOLO needed — the VLM already
        confirmed the object is visible and the robot is pointed at it.

        Called after Nav2 brings the robot to the general area, or directly
        when target="approach".
        """
        import numpy as np

        STOP_DISTANCE_M = 0.10   # stop ~10cm from target object
        SLOW_DISTANCE_M = 0.40   # slow down within 40cm
        MAX_STEPS = 40           # ~20 seconds max
        STEP_DURATION = 0.3      # seconds per step
        FAST_SPEED = 0.10        # m/s — normal approach
        SLOW_SPEED = 0.05        # m/s — final creep
        NO_PROGRESS_LIMIT = 8    # steps with no distance change = stuck

        self._log.info(f'Approaching "{object_name}" using LiDAR + depth')

        last_dist = float('inf')
        no_progress_count = 0

        for step in range(MAX_STEPS):
            if self._node.emergency_stop or not self._node.running:
                self._node._stop_motors()
                return {'reached': False, 'error': 'emergency stop'}

            if self._node._voice_interrupt:
                self._node._stop_motors()
                return {'reached': False, 'error': 'interrupted'}

            # --- Get distance from LiDAR front sector (primary) ---
            lidar_dist = float('inf')
            if self._node._lidar_ranges:
                lidar_dist = self._node._lidar_ranges.get('front', float('inf'))

            # --- Get distance from depth camera center ROI (secondary) ---
            depth_dist = float('inf')
            with self._node._depth_lock:
                depth_img = self._node._depth_image
                if depth_img is not None:
                    h, w = depth_img.shape[:2]
                    # Sample center 20% of frame
                    cy, cx = h // 2, w // 2
                    r = max(10, min(h, w) // 10)
                    roi = depth_img[cy - r:cy + r, cx - r:cx + r]
                    valid = roi[(roi > 0) & (roi < 40000)]
                    if len(valid) > 0:
                        depth_dist = float(np.median(valid)) / 1000.0

            # Use the smaller of the two (most conservative)
            dist = min(lidar_dist, depth_dist)

            self._log.info(
                f'Approach step {step}: lidar={lidar_dist:.2f}m, '
                f'depth={depth_dist:.2f}m, using={dist:.2f}m')

            # Close enough — stop
            if dist <= STOP_DISTANCE_M:
                self._node._stop_motors()
                self._log.info(
                    f'Reached "{object_name}" at {dist:.2f}m '
                    f'(lidar={lidar_dist:.2f}, depth={depth_dist:.2f})')
                return {
                    'reached': True,
                    'final_distance_m': round(dist, 2),
                    'lidar_m': round(lidar_dist, 2),
                    'depth_m': round(depth_dist, 2),
                }

            # No valid readings from either sensor
            if dist == float('inf'):
                self._node._stop_motors()
                if step == 0:
                    return {'reached': False, 'error': 'no distance readings from LiDAR or depth'}
                # Already moved some — stop where we are
                return {
                    'reached': True,
                    'final_distance_m': -1,
                    'note': 'lost sensor readings, stopped at current position',
                }

            # Stuck detection — if distance isn't decreasing, abort
            if abs(dist - last_dist) < 0.02:
                no_progress_count += 1
                if no_progress_count >= NO_PROGRESS_LIMIT:
                    self._node._stop_motors()
                    return {
                        'reached': True,
                        'final_distance_m': round(dist, 2),
                        'note': f'stopped — no progress after {no_progress_count} steps',
                    }
            else:
                no_progress_count = 0
            last_dist = dist

            # Speed: slow creep when close, faster when far
            speed = SLOW_SPEED if dist < SLOW_DISTANCE_M else FAST_SPEED
            self._node._ramper.set_target(speed, 0.0)
            time.sleep(STEP_DURATION)

        self._node._stop_motors()
        return {
            'reached': True,
            'final_distance_m': round(last_dist, 2),
            'note': 'max steps reached',
        }

    def explore_frontier(self, preference: str = 'nearest',
                         speech: str = '') -> dict:
        """Navigate to the nearest unexplored frontier."""
        if speech:
            self._speak_async(speech)

        if not (self._node.use_nav2 and self._node.nav2 and self._node.nav2.has_map):
            return {'success': False, 'error': 'No SLAM map available for frontier detection'}

        odom = self._get_odom()
        rx = odom.get('x', 0) if odom else 0
        ry = odom.get('y', 0) if odom else 0

        frontiers = self._node.nav2.get_frontier_goals(rx, ry)
        if not frontiers:
            return {'success': True, 'status': 'no_frontiers', 'message': 'No unexplored frontiers found — exploration may be complete'}

        # Pick frontier based on preference
        if preference == 'largest':
            frontiers.sort(key=lambda f: f['size'], reverse=True)

        goal = frontiers[0]
        accepted = self._node.nav2.navigate_to(goal['x'], goal['y'])
        if not accepted:
            return {'success': False, 'error': 'Nav2 goal not accepted'}

        # Wait for navigation with timeout, return for LLM re-evaluation periodically
        t_start = time.time()
        while (self._node.nav2.is_navigating
               and self._node.running
               and time.time() - t_start < NAV2_TOOL_TIMEOUT):
            if self._node.emergency_stop:
                self._node.nav2.cancel_navigation()
                return {'success': False, 'error': 'Emergency stop during frontier exploration'}
            # Return after interval so LLM can re-evaluate
            if time.time() - t_start >= NAV2_TOOL_REEVAL_INTERVAL:
                # Mark visited even for in-progress — we've committed to this frontier
                self._node.nav2.mark_frontier_visited(goal['x'], goal['y'])
                fb = self._node.nav2.navigation_feedback or {}
                return {
                    'success': True,
                    'status': 'in_progress',
                    'goal': goal,
                    'distance_remaining': fb.get('distance_remaining', '?'),
                    'alternatives': len(frontiers) - 1,
                }
            time.sleep(0.5)

        # Navigation completed or timed out
        result = self._node.nav2.navigation_result
        if result == 'succeeded':
            # Mark this frontier as visited so we don't return to it
            self._node.nav2.mark_frontier_visited(goal['x'], goal['y'])
            return {
                'success': True,
                'status': 'arrived',
                'goal': goal,
                'alternatives': len(frontiers) - 1,
            }
        else:
            return {
                'success': False,
                'status': result or 'timeout',
                'goal': goal,
                'error': f'Frontier navigation {result or "timed out"}',
            }

    def move_direct(self, action: str, speed: float, duration: float) -> dict:
        """Direct motor control for short movements."""
        safety_info = self._node._execute_action({
            'action': action,
            'speed': speed,
            'duration': duration,
        })
        return {
            'success': not safety_info.get('triggered', False),
            'action_executed': safety_info.get('override_action', action),
            'safety_triggered': safety_info.get('triggered', False),
            'safety_reason': safety_info.get('reason', ''),
        }

    def go_home(self, speech: str = '') -> dict:
        """Navigate back to map origin (0, 0)."""
        if speech:
            self._speak_async(speech)
        return self.navigate_to(target='coordinates', x=0.0, y=0.0)

    # ------------------------------------------------------------------
    # Perception tools
    # ------------------------------------------------------------------

    def look_around(self, speech: str = '') -> dict:
        """Pan the camera to survey surroundings."""
        if speech:
            self._speak_async(speech)

        self._node._look_around_sequence()

        # Get sensor summary after looking around
        lidar = self._get_lidar_sectors()
        odom = self._get_odom()

        return {
            'success': True,
            'lidar': lidar,
            'position': odom,
        }

    def identify_objects(self, focus_area: str = 'all') -> dict:
        """Detect objects, auto-register in knowledge graph, and describe scene.

        Tries VLM cloud first, falls back to local YOLO.
        Auto-registers detected objects in the current room's knowledge graph.
        Returns objects list + scene description + room guess.
        """
        objects = []
        source = 'none'
        room_guess = 'unknown'
        description = ''

        # Try VLM first (cloud API call)
        image_b64 = self._node._get_camera_frame_b64()
        if image_b64:
            prompt = (
                'Analyze this image from a robot camera. Respond in JSON:\n'
                '{"objects": [{"name": "...", "distance": "...", "position": "left|center|right"}],\n'
                ' "description": "2-3 sentence scene description",\n'
                ' "room_guess": "kitchen|bedroom|bathroom|living room|office|hallway|unknown"}'
            )
            if focus_area != 'all':
                prompt += f'\nFocus on the {focus_area} portion of the image.'
            try:
                result = self._node.llm.analyze_scene(
                    image_b64,
                    'You are a robot vision system. Identify objects and describe the scene. Respond in JSON only.',
                    prompt,
                )
                result.pop('_meta', {})
                objects = result.get('objects', [])
                description = result.get('description', '')
                room_guess = result.get('room_guess', 'unknown')
                source = 'vlm_cloud'
            except Exception as e:
                self._log.warn(f'VLM failed, falling back to YOLO: {e}')

        # Fallback: YOLO local (~200ms, free)
        if not objects and self._node.yolo:
            with self._node._rgb_lock:
                rgb = self._node._rgb_image.copy() if self._node._rgb_image is not None else None
            with self._node._depth_lock:
                depth = self._node._depth_image.copy() if self._node._depth_image is not None else None
            if rgb is None:
                return {'success': False, 'error': 'No camera frame available'}
            try:
                from autonomous_explorer.yolo_detector import YoloDetector
                detections = self._node.yolo.detect(rgb, depth)
                if focus_area != 'all' and focus_area in ('left', 'center', 'right'):
                    detections = [d for d in detections if d.position == focus_area]
                objects = YoloDetector.detections_to_dict(detections)
                obj_names = [d.label for d in detections]
                unique_objs = list(dict.fromkeys(obj_names))
                room_guess = self._guess_room(unique_objs)
                if detections:
                    description = f'I see {len(detections)} objects: {", ".join(unique_objs)}.'
                else:
                    description = 'No distinct objects detected in view.'
                source = 'yolo_local'
                self._log.info(
                    f'YOLO identify_objects: {len(objects)} objects '
                    f'({self._node.yolo.last_inference_ms:.0f}ms)')
            except Exception as e:
                self._log.error(f'YOLO fallback also failed: {e}')
                return {'success': False, 'error': str(e)}

        if not objects and source == 'none':
            return {'success': False, 'error': 'No camera frame and no YOLO detector available'}

        # Rule 4: Auto-register ALL detected objects with position
        odom = self._get_odom()
        ox = odom.get('x', 0.0) if odom else 0.0
        oy = odom.get('y', 0.0) if odom else 0.0
        # Determine current room from odom position
        current_room = ''
        if odom:
            current_room = self._node.world_knowledge._nearest_room(ox, oy)
        if not current_room and room_guess and room_guess != 'unknown':
            current_room = room_guess

        # Rule 8: track which objects are NEW vs already known
        known_in_room = set()
        if current_room:
            known_in_room = self._node.world_knowledge.get_known_objects_in_room(
                current_room)

        registered = []
        new_objects = []
        for obj in objects:
            name = (obj.get('name') or '').lower().strip()
            if name and name not in ('unknown', 'object'):
                try:
                    result = self._node.world_knowledge.add_object(
                        name, room=current_room, confidence=0.7,
                        x=ox, y=oy)
                    registered.append(name)
                    if result.get('is_new') and name not in known_in_room:
                        new_objects.append(name)
                        obj['is_new'] = True
                    else:
                        obj['is_new'] = False
                except Exception as e:
                    self._log.debug(
                        f'Auto-register skipped for {name}: {e}')
        if registered:
            self._log.info(
                f'Auto-registered objects: {registered} '
                f'(new: {new_objects})')

        return {
            'success': True,
            'objects': objects,
            'description': description,
            'room_guess': room_guess,
            'registered': registered,
            'new_objects': new_objects,
            'source': source,
        }

    def describe_scene(self) -> dict:
        """Describe the scene using VLM cloud, falling back to local YOLO."""
        # Try VLM first (cloud API call)
        image_b64 = self._node._get_camera_frame_b64()
        if image_b64:
            try:
                result = self._node.llm.analyze_scene(
                    image_b64,
                    'You are a robot describing what you see. Be concise and specific.',
                    'Describe this scene in 2-3 sentences. What room might this be? '
                    'What notable objects, features, or hazards do you see? '
                    'Respond in JSON: {"description": "...", "room_guess": "...", '
                    '"hazards": []}',
                )
                meta = result.pop('_meta', {})
                return {
                    'success': True,
                    'description': result.get('description', ''),
                    'room_guess': result.get('room_guess', ''),
                    'hazards': result.get('hazards', []),
                    'source': 'vlm_cloud',
                }
            except Exception as e:
                self._log.warn(f'VLM failed, falling back to YOLO: {e}')

        # Fallback: YOLO-based description (local, free)
        if self._node.yolo:
            with self._node._rgb_lock:
                rgb = self._node._rgb_image.copy() if self._node._rgb_image is not None else None
            with self._node._depth_lock:
                depth = self._node._depth_image.copy() if self._node._depth_image is not None else None
            if rgb is not None:
                try:
                    detections = self._node.yolo.detect(rgb, depth)
                    lidar = self._get_lidar_sectors()
                    obj_names = [d.label for d in detections]
                    unique_objs = list(dict.fromkeys(obj_names))
                    if detections:
                        desc = f'I can see {len(detections)} objects: {", ".join(unique_objs)}.'
                        close = [d for d in detections if 0 < d.distance_m < 1.0]
                        if close:
                            desc += f' Nearby: {", ".join(d.label for d in close)}.'
                    else:
                        desc = 'No distinct objects detected in view.'
                    if lidar:
                        front = lidar.get('front', 0)
                        desc += f' Front clearance: {front:.1f}m.'

                    room_guess = self._guess_room(unique_objs)
                    hazards = [d.label for d in detections
                               if d.label in ('knife', 'scissors', 'fire hydrant')
                               or (0 < d.distance_m < 0.3)]

                    return {
                        'success': True,
                        'description': desc,
                        'room_guess': room_guess,
                        'hazards': hazards,
                        'source': 'yolo_local',
                    }
                except Exception as e:
                    self._log.error(f'YOLO fallback also failed: {e}')
                    return {'success': False, 'error': str(e)}

        return {'success': False, 'error': 'No camera frame and no YOLO detector available'}

    def check_surroundings(self) -> dict:
        """Get a sensor summary without moving."""
        lidar = self._get_lidar_sectors()
        odom = self._get_odom()
        depth_summary = self._node._get_depth_summary()

        map_stats = None
        if self._node.use_nav2 and self._node.nav2 and self._node.nav2.has_map:
            map_stats = self._node.nav2.get_map_stats()

        return {
            'success': True,
            'lidar': lidar or {'status': 'no data'},
            'position': odom or {'status': 'no odometry'},
            'depth': depth_summary,
            'map': map_stats,
            'battery_voltage': self._node._battery_voltage,
            'emergency_stop': self._node.emergency_stop,
        }

    # ------------------------------------------------------------------
    # Knowledge tools
    # ------------------------------------------------------------------

    def label_room(self, room_name: str, description: str = '',
                   connections: list | None = None,
                   speech: str = '') -> dict:
        """Label the current location as a named room."""
        if speech:
            self._speak_async(speech)

        odom = self._get_odom()
        wk = self._node.world_knowledge
        name = room_name.strip().lower()

        x = round(odom.get('x', 0), 2) if odom else 0.0
        y = round(odom.get('y', 0), 2) if odom else 0.0

        wk.add_room(name, x=x, y=y, description=description)

        # Add connections bidirectionally
        if connections:
            for conn in connections:
                wk.add_connection(name, conn)

        # Record in consciousness
        self._node.consciousness.record_room(name)

        rooms = wk.get_rooms()
        return {
            'success': True,
            'room': name,
            'position': odom,
            'total_rooms': len(rooms),
        }

    def register_object(self, object_name: str, category: str = 'object',
                        room: str = '', description: str = '') -> dict:
        """Register an object in the knowledge graph."""
        odom = self._get_odom()
        wk = self._node.world_knowledge

        # If no room specified, find nearest room from odom
        if not room and odom:
            room = wk._nearest_room(
                odom.get('x', 0.0), odom.get('y', 0.0))

        wk.add_object(object_name.lower(), room=room, confidence=0.7)

        return {
            'success': True,
            'object': object_name.lower(),
            'category': category,
            'room': room,
            'position': odom,
        }

    def query_knowledge(self, query_type: str, query: str) -> dict:
        """Search the knowledge graph."""
        wk = self._node.world_knowledge
        query_lower = query.lower().strip()

        if query_type == 'find_object':
            # Check graph for object and return its room
            room = wk.get_room_for_object(query_lower)
            if room:
                objects = wk.get_objects()
                match = objects.get(query_lower, {})
                return {
                    'success': True,
                    'results': {query_lower: {**match, 'room': room}},
                }
            # Fuzzy match
            objects = wk.get_objects()
            matches = {k: v for k, v in objects.items()
                       if query_lower in k or k in query_lower}
            if matches:
                return {'success': True, 'results': matches}
            return {'success': True, 'results': {},
                    'message': f'Object "{query}" not found in knowledge graph'}

        elif query_type == 'describe_room':
            rooms = wk.get_rooms()
            if query_lower in rooms:
                info = rooms[query_lower]
                info['objects'] = wk.get_room_objects(query_lower)
                info['connections'] = wk.get_room_connections(query_lower)
                return {'success': True, 'room': query_lower, 'info': info}
            return {'success': True, 'room': query_lower,
                    'message': f'Room "{query}" not found'}

        elif query_type == 'list_rooms':
            rooms = wk.get_rooms()
            return {
                'success': True,
                'rooms': list(rooms.keys()),
                'count': len(rooms),
            }

        elif query_type == 'list_objects':
            objects = wk.get_objects()
            summary = {k: {'confidence': v.get('confidence', 0),
                           'room': v.get('room', '?')}
                       for k, v in objects.items()}
            return {
                'success': True,
                'objects': summary,
                'count': len(objects),
            }

        elif query_type == 'room_connections':
            conns = wk.get_room_connections(query_lower)
            return {
                'success': True,
                'room': query_lower,
                'connections': conns,
            }

        return {'success': False, 'error': f'Unknown query_type: {query_type}'}

    def save_map(self, map_name: str) -> dict:
        """Save the current SLAM map and knowledge to disk."""
        # Save world knowledge
        self._node.world_knowledge.save()
        self._node.consciousness.save()

        # If slam_toolbox is running, call its save service
        saved_slam = False
        if self._node.use_nav2 and self._node.nav2 and self._node.nav2.has_map:
            try:
                from slam_toolbox.srv import SaveMap
                client = self._node.create_client(SaveMap, '/slam_toolbox/save_map')
                if client.wait_for_service(timeout_sec=2.0):
                    req = SaveMap.Request()
                    req.name.data = map_name
                    future = client.call_async(req)
                    # Don't block indefinitely
                    saved_slam = True
                    self._log.info(f'SLAM map save requested: {map_name}')
                else:
                    self._log.warning('slam_toolbox save_map service not available')
            except ImportError as e:
                self._log.warning(f'slam_toolbox not available: {e}')
            except Exception as e:
                self._log.error(f'SLAM map save failed: {e}')

        return {
            'success': True,
            'knowledge_saved': True,
            'slam_map_saved': saved_slam,
            'map_name': map_name,
        }

    # ------------------------------------------------------------------
    # Communication tools
    # ------------------------------------------------------------------

    def speak(self, text: str, wait: bool = True) -> dict:
        """Say something through the speaker."""
        if not self._node.voice_on:
            self._log.info(f'[SPEECH OFF] {text}')
            return {'success': True, 'spoken': False, 'reason': 'voice disabled'}

        try:
            self._node.voice.speak(text, block=wait, force=True)
            return {'success': True, 'spoken': True, 'text': text}
        except OSError as e:
            self._log.error(f'speak audio device error: {e}')
            return {'success': False, 'error': str(e)}
        except Exception as e:
            self._log.error(f'speak failed: {e}')
            return {'success': False, 'error': str(e)}

    def listen(self, duration_seconds: int = 5) -> dict:
        """Record audio and transcribe it."""
        if not self._node.voice_on:
            return {'success': False, 'error': 'Voice I/O disabled'}

        try:
            transcript = self._node.voice.listen_for_command(
                duration=duration_seconds)
            if transcript:
                return {'success': True, 'transcript': transcript}
            return {'success': True, 'transcript': '', 'message': 'No speech detected'}
        except OSError as e:
            self._log.error(f'listen audio device error: {e}')
            return {'success': False, 'error': str(e)}
        except Exception as e:
            self._log.error(f'listen failed: {e}')
            return {'success': False, 'error': str(e)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_odom(self) -> dict | None:
        with self._node._odom_lock:
            return self._node._odom_data.copy() if self._node._odom_data else None

    def _get_lidar_sectors(self) -> dict | None:
        with self._node._lidar_lock:
            return self._node._lidar_ranges.copy() if self._node._lidar_ranges else None

    @staticmethod
    def _guess_room(object_names: list[str]) -> str:
        """Guess room type from detected object names (high-confidence only).

        Uses a narrow set of strong-signal objects for confident room labeling.
        For broad search prioritization, see ROOM_OBJECT_LIKELIHOOD in
        world_knowledge.py instead.
        """
        names = set(o.lower() for o in object_names)
        if names & {'oven', 'microwave', 'refrigerator', 'sink', 'toaster'}:
            return 'kitchen'
        if names & {'toilet', 'sink'} and not names & {'oven', 'refrigerator'}:
            return 'bathroom'
        if names & {'bed', 'pillow'}:
            return 'bedroom'
        if names & {'couch', 'tv', 'remote', 'sofa'}:
            return 'living room'
        if names & {'laptop', 'keyboard', 'mouse', 'monitor'}:
            return 'office'
        if names & {'dining table', 'wine glass', 'fork', 'knife', 'spoon'}:
            return 'dining room'
        return 'unknown'

    def _speak_async(self, text: str):
        """Speak without blocking the tool execution."""
        if self._node.voice_on and text:
            try:
                self._node.voice.speak(text, block=False, force=True)
            except OSError as e:
                self._log.warning(f'Async speak audio error: {e}')
            except Exception as e:
                self._log.warning(f'Async speak failed: {e}')
