#!/usr/bin/env python3
# encoding: utf-8
"""
Tool Handlers — implements the 14 Jeeves tool functions.

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
    """Implements all 14 Jeeves tool handler methods.

    Takes a reference to the AutonomousExplorer node to access
    sensors, actuators, Nav2, world knowledge, voice, and LLM.
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
                    speech: str = '') -> dict:
        """Navigate to a named location or coordinates."""
        if speech:
            self._speak_async(speech)

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
            return {
                'success': result == 'succeeded',
                'status': result or 'timeout',
            }

        # Fallback: no Nav2 available
        return {
            'success': False,
            'error': 'Nav2 not available. Use move_direct for short movements.',
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

        # Auto-register detected objects in the knowledge graph
        odom = self._get_odom()
        now = datetime.now().isoformat(timespec='seconds')
        registered = []
        for obj in objects:
            name = (obj.get('name') or '').lower().strip()
            if name and name not in ('unknown', 'object'):
                try:
                    self._node.world_knowledge._update_object(
                        name, now, odom, category='detected')
                    registered.append(name)
                except Exception:
                    pass
        if registered:
            self._node.world_knowledge.save()
            self._log.info(f'Auto-registered objects: {registered}')

        return {
            'success': True,
            'objects': objects,
            'description': description,
            'room_guess': room_guess,
            'registered': registered,
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
        now = datetime.now().isoformat(timespec='seconds')

        rooms = self._node.world_knowledge.world_map.setdefault('rooms', {})
        name = room_name.strip().lower()

        if name in rooms:
            rooms[name]['times_visited'] = rooms[name].get('times_visited', 0) + 1
            rooms[name]['last_visited'] = now
            if description:
                rooms[name]['description'] = description
        else:
            rooms[name] = {
                'first_discovered': now,
                'times_visited': 1,
                'last_visited': now,
                'description': description,
                'connections': connections or [],
                'landmarks': [],
                'notes': '',
                'confidence': 0.7,
            }

        # Store position if we have odometry
        if odom:
            rooms[name]['position'] = {
                'x': round(odom.get('x', 0), 2),
                'y': round(odom.get('y', 0), 2),
            }

        # Add connections bidirectionally
        if connections:
            for conn in connections:
                conn_lower = conn.strip().lower()
                if conn_lower in rooms:
                    existing_conns = rooms[conn_lower].setdefault('connections', [])
                    if name not in existing_conns:
                        existing_conns.append(name)
                rooms[name].setdefault('connections', [])
                if conn_lower not in rooms[name]['connections']:
                    rooms[name]['connections'].append(conn_lower)

        # Record in consciousness
        self._node.consciousness.record_room(name)

        # Persist to disk so semantic_map_publisher picks up changes
        self._node.world_knowledge.save()

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
        now = datetime.now().isoformat(timespec='seconds')

        self._node.world_knowledge._update_object(
            object_name.lower(), now, odom, category)

        obj_entry = self._node.world_knowledge.known_objects.get(
            'objects', {}).get(object_name.lower(), {})

        if room:
            obj_entry['usual_location'] = room
        if description:
            obj_entry['description'] = description

        # Persist to disk so semantic_map_publisher picks up changes
        self._node.world_knowledge.save()

        return {
            'success': True,
            'object': object_name.lower(),
            'category': category,
            'position': odom,
        }

    def query_knowledge(self, query_type: str, query: str) -> dict:
        """Search the knowledge graph."""
        wk = self._node.world_knowledge
        query_lower = query.lower().strip()

        if query_type == 'find_object':
            objects = wk.known_objects.get('objects', {})
            matches = {k: v for k, v in objects.items() if query_lower in k}
            if matches:
                return {'success': True, 'results': matches}
            return {'success': True, 'results': {}, 'message': f'Object "{query}" not found'}

        elif query_type == 'describe_room':
            rooms = wk.world_map.get('rooms', {})
            room_info = rooms.get(query_lower)
            if room_info:
                return {'success': True, 'room': query_lower, 'info': room_info}
            return {'success': True, 'room': query_lower, 'message': f'Room "{query}" not found'}

        elif query_type == 'list_rooms':
            rooms = wk.world_map.get('rooms', {})
            return {
                'success': True,
                'rooms': list(rooms.keys()),
                'count': len(rooms),
            }

        elif query_type == 'list_objects':
            objects = wk.known_objects.get('objects', {})
            summary = {k: {'category': v.get('category', '?'),
                           'location': v.get('usual_location', '?')}
                       for k, v in objects.items()}
            return {
                'success': True,
                'objects': summary,
                'count': len(objects),
            }

        elif query_type == 'room_connections':
            rooms = wk.world_map.get('rooms', {})
            room_info = rooms.get(query_lower)
            if room_info:
                return {
                    'success': True,
                    'room': query_lower,
                    'connections': room_info.get('connections', []),
                }
            return {'success': True, 'room': query_lower, 'connections': [],
                    'message': f'Room "{query}" not found'}

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
        """Guess room type from detected object names."""
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
