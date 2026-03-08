"""Tests for autonomous_explorer.tool_handlers module."""
import json
import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from autonomous_explorer.tool_handlers import ToolHandlers
from autonomous_explorer.tool_registry import create_registry


# ===================================================================
# Mock node fixture
# ===================================================================

class MockLogger:
    def info(self, *a, **kw): pass
    def warning(self, *a, **kw): pass
    def error(self, *a, **kw): pass
    def debug(self, *a, **kw): pass


class MockVoice:
    def __init__(self):
        self.spoken = []

    def speak(self, text, block=True):
        self.spoken.append(text)

    def listen_for_command(self, duration=5):
        return 'test command'


class MockWorldKnowledge:
    def __init__(self, tmp_dir):
        self.knowledge_dir = tmp_dir
        self.world_map = {'rooms': {}, 'corridors': [], 'doors': []}
        self.known_objects = {'objects': {}}
        self.learned_behaviors = {'navigation_lessons': []}

    def get_prompt_context(self, x=0, y=0, theta=0):
        return ''

    def _update_object(self, name, timestamp, odom=None, category=None):
        self.known_objects.setdefault('objects', {})[name] = {
            'first_seen': timestamp,
            'times_seen': 1,
            'last_seen': timestamp,
            'usual_location': '',
            'category': category or 'object',
        }

    def save(self):
        self.save_called = True


class MockConsciousness:
    def __init__(self):
        self.rooms_recorded = []
        self.save_called = False

    def get_identity_context(self):
        return 'Test identity'

    def record_room(self, name):
        self.rooms_recorded.append(name)

    def save(self):
        self.save_called = True


@pytest.fixture
def mock_node(tmp_dir):
    """Create a mock explorer node with all needed attributes."""
    node = MagicMock()
    node.get_logger.return_value = MockLogger()

    # Voice
    node.voice_on = True
    node.voice = MockVoice()

    # Sensors
    node._odom_lock = MagicMock()
    node._odom_data = {'x': 1.0, 'y': 2.0, 'theta': 0.5}
    node._lidar_lock = MagicMock()
    node._lidar_ranges = {'front': 1.5, 'left': 2.0, 'right': 0.8, 'back': 3.0}
    node._battery_voltage = 12.1
    node.emergency_stop = False

    # Nav2
    node.use_nav2 = False
    node.nav2 = None

    # World knowledge
    node.world_knowledge = MockWorldKnowledge(tmp_dir)
    node.consciousness = MockConsciousness()

    # LLM
    node.llm = MagicMock()

    # Camera
    node._get_camera_frame_b64.return_value = 'fake_b64_image'
    node._get_depth_summary.return_value = 'Depth: center=150cm'

    # Motor control
    node._execute_action.return_value = {
        'triggered': False,
        'reason': '',
        'original_action': 'forward',
        'override_action': 'forward',
    }
    node._look_around_sequence.return_value = None

    # Node internals
    node.running = True
    node.create_client = MagicMock()

    return node


@pytest.fixture
def handlers(mock_node):
    return ToolHandlers(mock_node)


# ===================================================================
# bind_to_registry
# ===================================================================

class TestBindToRegistry:

    def test_binds_all_14_handlers(self, mock_node):
        reg = create_registry()
        h = ToolHandlers(mock_node)
        h.bind_to_registry(reg)
        for name in reg.tool_names:
            tool = reg.get_tool(name)
            assert tool.handler is not None, f'{name} has no handler after binding'

    def test_execute_after_binding(self, mock_node):
        reg = create_registry()
        h = ToolHandlers(mock_node)
        h.bind_to_registry(reg)
        result = reg.execute('check_surroundings', {})
        assert result['success'] is True


# ===================================================================
# Navigation tools
# ===================================================================

class TestNavigateTo:

    def test_unknown_location_fails(self, handlers):
        result = handlers.navigate_to(target='nonexistent')
        assert result['success'] is False
        assert 'Unknown location' in result['error']

    def test_no_nav2_fails(self, handlers):
        result = handlers.navigate_to(target='coordinates', x=1.0, y=2.0)
        assert result['success'] is False
        assert 'Nav2 not available' in result['error']

    def test_with_speech(self, handlers, mock_node):
        handlers.navigate_to(target='coordinates', x=0, y=0, speech='Going home')
        assert 'Going home' in mock_node.voice.spoken


class TestExploreFrontier:

    def test_no_nav2_fails(self, handlers):
        result = handlers.explore_frontier()
        assert result['success'] is False

    def test_no_frontiers(self, handlers, mock_node):
        mock_node.use_nav2 = True
        mock_node.nav2 = MagicMock()
        mock_node.nav2.has_map = True
        mock_node.nav2.get_frontier_goals.return_value = []
        result = handlers.explore_frontier()
        assert result['success'] is True
        assert result['status'] == 'no_frontiers'


class TestMoveDirect:

    def test_forward(self, handlers, mock_node):
        result = handlers.move_direct(action='forward', speed=0.5, duration=1.0)
        assert result['success'] is True
        mock_node._execute_action.assert_called_once_with({
            'action': 'forward', 'speed': 0.5, 'duration': 1.0,
        })

    def test_safety_override(self, handlers, mock_node):
        mock_node._execute_action.return_value = {
            'triggered': True,
            'reason': 'obstacle at 0.15m',
            'original_action': 'forward',
            'override_action': 'stop',
        }
        result = handlers.move_direct(action='forward', speed=0.8, duration=2.0)
        assert result['success'] is False
        assert result['safety_triggered'] is True


class TestGoHome:

    def test_delegates_to_navigate_to(self, handlers, mock_node):
        # Without Nav2, should fail gracefully
        result = handlers.go_home()
        assert result['success'] is False


# ===================================================================
# Perception tools
# ===================================================================

class TestLookAround:

    def test_calls_look_around_sequence(self, handlers, mock_node):
        result = handlers.look_around()
        assert result['success'] is True
        mock_node._look_around_sequence.assert_called_once()

    def test_returns_sensor_data(self, handlers):
        result = handlers.look_around()
        assert 'lidar' in result
        assert 'position' in result


class TestIdentifyObjects:

    def test_calls_llm(self, handlers, mock_node):
        mock_node.llm.analyze_scene.return_value = {
            'objects': [{'name': 'chair', 'distance': '2m', 'position': 'center'}],
            '_meta': {},
        }
        result = handlers.identify_objects()
        assert result['success'] is True
        assert len(result['objects']) == 1

    def test_no_camera(self, handlers, mock_node):
        mock_node._get_camera_frame_b64.return_value = ''
        result = handlers.identify_objects()
        assert result['success'] is False

    def test_llm_error(self, handlers, mock_node):
        mock_node.llm.analyze_scene.side_effect = Exception('API error')
        result = handlers.identify_objects()
        assert result['success'] is False


class TestDescribeScene:

    def test_calls_llm(self, handlers, mock_node):
        mock_node.llm.analyze_scene.return_value = {
            'description': 'A kitchen',
            'room_guess': 'kitchen',
            'hazards': [],
            '_meta': {},
        }
        result = handlers.describe_scene()
        assert result['success'] is True
        assert result['room_guess'] == 'kitchen'


class TestCheckSurroundings:

    def test_returns_all_sensor_data(self, handlers):
        result = handlers.check_surroundings()
        assert result['success'] is True
        assert 'lidar' in result
        assert 'position' in result
        assert 'depth' in result
        assert 'battery_voltage' in result
        assert result['emergency_stop'] is False


# ===================================================================
# Knowledge tools
# ===================================================================

class TestLabelRoom:

    def test_creates_new_room(self, handlers, mock_node):
        result = handlers.label_room(room_name='Kitchen', description='Has a fridge')
        assert result['success'] is True
        assert result['room'] == 'kitchen'
        rooms = mock_node.world_knowledge.world_map['rooms']
        assert 'kitchen' in rooms
        assert rooms['kitchen']['description'] == 'Has a fridge'

    def test_stores_position(self, handlers, mock_node):
        result = handlers.label_room(room_name='Office')
        rooms = mock_node.world_knowledge.world_map['rooms']
        assert 'position' in rooms['office']
        assert rooms['office']['position']['x'] == 1.0

    def test_increments_visits(self, handlers, mock_node):
        handlers.label_room(room_name='Hall')
        handlers.label_room(room_name='Hall')
        rooms = mock_node.world_knowledge.world_map['rooms']
        assert rooms['hall']['times_visited'] == 2

    def test_bidirectional_connections(self, handlers, mock_node):
        handlers.label_room(room_name='Kitchen', connections=['hallway'])
        handlers.label_room(room_name='Hallway')
        handlers.label_room(room_name='Kitchen', connections=['hallway'])
        rooms = mock_node.world_knowledge.world_map['rooms']
        assert 'hallway' in rooms['kitchen']['connections']
        assert 'kitchen' in rooms['hallway']['connections']

    def test_records_in_consciousness(self, handlers, mock_node):
        handlers.label_room(room_name='Lab')
        assert 'lab' in mock_node.consciousness.rooms_recorded


class TestRegisterObject:

    def test_registers_object(self, handlers, mock_node):
        result = handlers.register_object(
            object_name='Fridge', category='furniture',
        )
        assert result['success'] is True
        assert result['object'] == 'fridge'
        objs = mock_node.world_knowledge.known_objects['objects']
        assert 'fridge' in objs


class TestQueryKnowledge:

    def test_find_object_found(self, handlers, mock_node):
        mock_node.world_knowledge.known_objects = {
            'objects': {'chair': {'category': 'furniture'}},
        }
        result = handlers.query_knowledge(query_type='find_object', query='chair')
        assert result['success'] is True
        assert 'chair' in result['results']

    def test_find_object_not_found(self, handlers):
        result = handlers.query_knowledge(query_type='find_object', query='spaceship')
        assert result['success'] is True
        assert len(result['results']) == 0

    def test_describe_room(self, handlers, mock_node):
        mock_node.world_knowledge.world_map = {
            'rooms': {'kitchen': {'description': 'Big kitchen', 'connections': []}},
        }
        result = handlers.query_knowledge(query_type='describe_room', query='kitchen')
        assert result['success'] is True
        assert result['info']['description'] == 'Big kitchen'

    def test_list_rooms(self, handlers, mock_node):
        mock_node.world_knowledge.world_map = {
            'rooms': {'a': {}, 'b': {}},
        }
        result = handlers.query_knowledge(query_type='list_rooms', query='')
        assert result['count'] == 2

    def test_list_objects(self, handlers, mock_node):
        mock_node.world_knowledge.known_objects = {
            'objects': {
                'chair': {'category': 'furniture', 'usual_location': 'office'},
            },
        }
        result = handlers.query_knowledge(query_type='list_objects', query='')
        assert result['count'] == 1

    def test_room_connections(self, handlers, mock_node):
        mock_node.world_knowledge.world_map = {
            'rooms': {'hall': {'connections': ['kitchen', 'bedroom']}},
        }
        result = handlers.query_knowledge(
            query_type='room_connections', query='hall',
        )
        assert 'kitchen' in result['connections']

    def test_unknown_query_type(self, handlers):
        result = handlers.query_knowledge(query_type='magic', query='x')
        assert result['success'] is False


class TestSaveMap:

    def test_saves_knowledge(self, handlers, mock_node):
        result = handlers.save_map(map_name='test_map')
        assert result['success'] is True
        assert result['knowledge_saved'] is True
        assert mock_node.world_knowledge.save_called is True
        assert mock_node.consciousness.save_called is True


# ===================================================================
# Communication tools
# ===================================================================

class TestSpeak:

    def test_speaks_text(self, handlers, mock_node):
        result = handlers.speak(text='Hello world')
        assert result['success'] is True
        assert result['spoken'] is True
        assert 'Hello world' in mock_node.voice.spoken

    def test_voice_disabled(self, handlers, mock_node):
        mock_node.voice_on = False
        result = handlers.speak(text='Silent')
        assert result['success'] is True
        assert result['spoken'] is False


class TestListen:

    def test_listens(self, handlers, mock_node):
        result = handlers.listen(duration_seconds=3)
        assert result['success'] is True
        assert result['transcript'] == 'test command'

    def test_voice_disabled(self, handlers, mock_node):
        mock_node.voice_on = False
        result = handlers.listen()
        assert result['success'] is False


# ===================================================================
# End-to-end: registry + handlers
# ===================================================================

class TestEndToEnd:
    """Test full registry → handler → execute pipeline."""

    def test_execute_check_surroundings(self, mock_node):
        reg = create_registry()
        h = ToolHandlers(mock_node)
        h.bind_to_registry(reg)
        result = reg.execute('check_surroundings', {})
        assert result['success'] is True
        assert 'lidar' in result

    def test_execute_speak(self, mock_node):
        reg = create_registry()
        h = ToolHandlers(mock_node)
        h.bind_to_registry(reg)
        result = reg.execute('speak', {'text': 'Testing', 'wait': True})
        assert result['success'] is True

    def test_execute_move_direct(self, mock_node):
        reg = create_registry()
        h = ToolHandlers(mock_node)
        h.bind_to_registry(reg)
        result = reg.execute('move_direct', {
            'action': 'forward', 'speed': 0.5, 'duration': 1.0,
        })
        assert result['success'] is True
        mock_node._execute_action.assert_called_once()

    def test_execute_label_room(self, mock_node):
        reg = create_registry()
        h = ToolHandlers(mock_node)
        h.bind_to_registry(reg)
        result = reg.execute('label_room', {
            'room_name': 'Lab', 'description': 'Electronics lab',
        })
        assert result['success'] is True
        assert result['room'] == 'lab'
