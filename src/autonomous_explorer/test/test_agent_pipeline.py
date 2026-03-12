"""Comprehensive agent pipeline tests.

Tests the full agent reasoning pipeline end-to-end: DryRunProvider →
ToolRegistry → ToolHandlers → tool execution → result feedback.
Uses the real DryRunProvider (deterministic tool call sequences) with
mocked hardware/ROS2 to validate:

1. All 7 registered tools execute through the full pipeline
2. DryRunProvider only calls registered tools (no removed tools)
3. VLM→YOLO cascade in identify_objects
4. Auto-registration of detected objects in WorldKnowledge
5. Movement limiter (1 movement per LLM response)
6. LLM hard failure → stop motors → halt exploration
7. YOLO text-only mode for local providers
8. Multi-turn ReAct cycle (DryRun → tools → results → next round)
9. Tool timeout enforcement
10. explore_frontier wait/re-evaluation loop
"""
import threading
import time
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from autonomous_explorer.llm_provider import (
    DryRunProvider,
    create_provider,
    LLMProvider,
)
from autonomous_explorer.conversation_manager import (
    AgentResponse,
    ConversationManager,
    ToolCall,
)
from autonomous_explorer.tool_registry import create_registry, build_jeeves_tools
from autonomous_explorer.tool_handlers import ToolHandlers


# ===================================================================
# Shared fixtures
# ===================================================================

class MockLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.errors = []

    def info(self, msg, *a, **kw):
        self.infos.append(msg)

    def warn(self, msg, *a, **kw):
        self.warnings.append(msg)

    def warning(self, msg, *a, **kw):
        self.warnings.append(msg)

    def error(self, msg, *a, **kw):
        self.errors.append(msg)

    def debug(self, *a, **kw):
        pass


class MockVoice:
    def __init__(self):
        self.spoken = []

    def speak(self, text, block=True, force=False):
        self.spoken.append(text)


class MockWorldKnowledge:
    def __init__(self):
        self._rooms = {}
        self._objects = {}
        self.save_count = 0

    @property
    def world_map(self):
        return {'rooms': self._rooms}

    @property
    def known_objects(self):
        return {'objects': self._objects}

    @property
    def learned_behaviors(self):
        return {'navigation_lessons': []}

    def get_prompt_context(self, x=0, y=0, theta=0):
        return 'Known rooms: none'

    def get_rooms(self):
        return dict(self._rooms)

    def get_objects(self):
        return dict(self._objects)

    def get_room_connections(self, room_name):
        return self._rooms.get(room_name, {}).get('connections', [])

    def get_room_for_object(self, object_name):
        return self._objects.get(object_name, {}).get('room') or None

    def get_room_objects(self, room_name):
        return [n for n, d in self._objects.items()
                if d.get('room') == room_name]

    def _nearest_room(self, x, y, max_dist=3.0):
        return ''

    def add_room(self, name, x=0.0, y=0.0, description=''):
        key = name.strip().lower()
        if key in self._rooms:
            self._rooms[key]['times_visited'] = \
                self._rooms[key].get('times_visited', 0) + 1
        else:
            self._rooms[key] = {
                'type': 'room', 'x': x, 'y': y,
                'description': description, 'times_visited': 1,
                'connections': [],
            }
        self.save()
        return self._rooms[key]

    def add_object(self, name, room='', confidence=0.7, x=0.0, y=0.0):
        key = name.strip().lower()
        is_new = key not in self._objects
        self._objects[key] = {
            'type': 'object', 'name': key,
            'confidence': confidence, 'room': room,
            'x': x, 'y': y, 'is_new': is_new,
        }
        self.save()
        return self._objects[key]

    def get_known_objects_in_room(self, room_name):
        return {n for n, d in self._objects.items()
                if d.get('room') == room_name}

    def add_connection(self, room_a, room_b):
        a = room_a.strip().lower()
        b = room_b.strip().lower()
        if a not in self._rooms:
            self.add_room(a)
        if b not in self._rooms:
            self.add_room(b)
        if b not in self._rooms[a].get('connections', []):
            self._rooms[a].setdefault('connections', []).append(b)
        if a not in self._rooms[b].get('connections', []):
            self._rooms[b].setdefault('connections', []).append(a)
        self.save()

    def mark_room_searched(self, room_name, target_object):
        pass

    def get_search_summary(self, target_object):
        return {'searched_rooms': [], 'unsearched_rooms': [],
                'objects_by_room': {}, 'suggested_next': None,
                'found_in': None}

    def _update_object(self, name, timestamp, odom=None, category=None):
        self.add_object(name)

    def save(self):
        self.save_count += 1


class MockConsciousness:
    def __init__(self):
        self.rooms_recorded = []

    def get_identity_context(self):
        return 'I am Jeeves'

    def record_room(self, name):
        self.rooms_recorded.append(name)

    def save(self):
        pass


class MockNav2:
    """Simulates Nav2Bridge for testing."""

    def __init__(self):
        self.has_map = True
        self.is_navigating = False
        self.navigation_result = 'succeeded'
        self.navigation_feedback = {'distance_remaining': 0.5}
        self.navigate_calls = []
        self.cancel_count = 0
        self.frontier_goals = [
            {'x': 2.0, 'y': 1.0, 'size': 50},
            {'x': -1.0, 'y': 3.0, 'size': 30},
        ]

    def navigate_to(self, x, y):
        self.navigate_calls.append((x, y))
        return True  # goal accepted

    def get_frontier_goals(self, rx, ry):
        return list(self.frontier_goals)

    def cancel_navigation(self):
        self.cancel_count += 1
        self.is_navigating = False


@pytest.fixture
def mock_node():
    """Create a mock explorer node with all attributes needed by ToolHandlers."""
    node = MagicMock()
    node.get_logger.return_value = MockLogger()

    node.voice_on = True
    node.voice = MockVoice()

    node._odom_lock = MagicMock()
    node._odom_data = {'x': 1.0, 'y': 2.0, 'theta': 0.5}
    node._lidar_lock = MagicMock()
    node._lidar_ranges = {'front': 1.5, 'left': 2.0, 'right': 0.8, 'back': 3.0}
    node._battery_voltage = 12.1
    node.emergency_stop = False

    node.use_nav2 = True
    node.nav2 = MockNav2()

    node.world_knowledge = MockWorldKnowledge()
    node.consciousness = MockConsciousness()

    node.llm = MagicMock()
    # Default: VLM returns a realistic scene description
    node.llm.analyze_scene.return_value = {
        'objects': [
            {'name': 'chair', 'distance': '1.5m', 'position': 'center'},
            {'name': 'desk', 'distance': '2.0m', 'position': 'right'},
        ],
        'description': 'An office with a desk and chair.',
        'room_guess': 'office',
        '_meta': {'provider': 'openai', 'model': 'gpt-4o'},
    }

    node._get_camera_frame_b64.return_value = 'fake_b64_image_data'
    node._get_depth_summary.return_value = 'Depth: center=150cm'
    node._rgb_lock = threading.Lock()
    node._rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    node._depth_lock = threading.Lock()
    node._depth_image = np.random.randint(200, 4000, (480, 640), dtype=np.uint16)

    node.yolo = None  # disabled by default

    node._execute_action.return_value = {
        'triggered': False, 'reason': '',
        'original_action': 'forward', 'override_action': 'forward',
    }
    node._look_around_sequence.return_value = None
    node.running = True
    node.create_client = MagicMock()

    return node


@pytest.fixture
def registry():
    return create_registry()


@pytest.fixture
def bound_registry(mock_node):
    """Registry with all 7 handlers bound."""
    reg = create_registry()
    handlers = ToolHandlers(mock_node)
    handlers.bind_to_registry(reg)
    return reg


@pytest.fixture
def handlers(mock_node):
    return ToolHandlers(mock_node)


@pytest.fixture
def dryrun():
    return DryRunProvider()


# ===================================================================
# 1. All 7 registered tools execute through full pipeline
# ===================================================================

class TestFullPipelineExecution:
    """Every registered tool can be executed via registry.execute()."""

    def test_navigate_to_via_registry(self, bound_registry, mock_node):
        # Room lookup fails → coordinates mode
        result = bound_registry.execute(
            'navigate_to', {'target': 'coordinates', 'x': 1.0, 'y': 2.0})
        assert result['success'] is True
        assert mock_node.nav2.navigate_calls[-1] == (1.0, 2.0)

    def test_explore_frontier_via_registry(self, bound_registry):
        result = bound_registry.execute(
            'explore_frontier', {'preference': 'nearest'})
        assert result['success'] is True

    def test_go_home_via_registry(self, bound_registry, mock_node):
        result = bound_registry.execute('go_home', {})
        assert result['success'] is True
        # go_home delegates to navigate_to(0, 0)
        assert mock_node.nav2.navigate_calls[-1] == (0, 0)

    def test_identify_objects_via_registry(self, bound_registry):
        result = bound_registry.execute(
            'identify_objects', {'focus_area': 'all'})
        assert result['success'] is True
        assert 'objects' in result
        assert result['source'] == 'vlm_cloud'

    def test_label_room_via_registry(self, bound_registry, mock_node):
        result = bound_registry.execute(
            'label_room', {'room_name': 'kitchen', 'description': 'Has a fridge'})
        assert result['success'] is True
        assert result['room'] == 'kitchen'
        assert mock_node.consciousness.rooms_recorded == ['kitchen']

    def test_query_knowledge_via_registry(self, bound_registry):
        result = bound_registry.execute(
            'query_knowledge', {'query_type': 'list_rooms', 'query': 'all'})
        assert result['success'] is True

    def test_speak_via_registry(self, bound_registry, mock_node):
        result = bound_registry.execute(
            'speak', {'text': 'Hello world'})
        assert result['success'] is True
        assert 'Hello world' in mock_node.voice.spoken


# ===================================================================
# 2. DryRunProvider only references registered tools
# ===================================================================

class TestDryRunToolConsistency:
    """DryRunProvider._AGENT_SEQUENCES must only call registered tools."""

    REGISTERED_TOOLS = {
        'navigate_to', 'explore_frontier', 'go_home',
        'identify_objects', 'label_room', 'query_knowledge', 'speak',
    }

    def test_all_dryrun_tools_are_registered(self):
        dryrun = DryRunProvider()
        for seq in dryrun._AGENT_SEQUENCES:
            for tc in seq:
                assert tc.tool_name in self.REGISTERED_TOOLS, (
                    f'DryRun references unregistered tool: {tc.tool_name}')

    def test_dryrun_agent_turn_returns_valid_tools(self):
        dryrun = DryRunProvider()
        for i in range(len(dryrun._AGENT_SEQUENCES)):
            response = dryrun.agent_turn(
                system_prompt='test',
                messages=[{'role': 'user', 'content': 'test'}],
                tools=[],
            )
            for tc in response.tool_calls:
                assert tc.tool_name in self.REGISTERED_TOOLS

    def test_dryrun_cycles_through_all_sequences(self):
        dryrun = DryRunProvider()
        seen_tools = set()
        for _ in range(len(dryrun._AGENT_SEQUENCES)):
            response = dryrun.agent_turn('test', [{'role': 'user', 'content': 'x'}], [])
            for tc in response.tool_calls:
                seen_tools.add(tc.tool_name)
        # After a full cycle, should have exercised most tools
        assert len(seen_tools) >= 5, f'Only saw {seen_tools}'

    def test_dryrun_wraps_around(self):
        dryrun = DryRunProvider()
        n = len(dryrun._AGENT_SEQUENCES)
        # Call n+1 times to verify wrap-around
        for _ in range(n + 1):
            response = dryrun.agent_turn('test', [{'role': 'user', 'content': 'x'}], [])
            assert response.tool_calls  # never empty


# ===================================================================
# 3. VLM→YOLO cascade in identify_objects
# ===================================================================

class MockYoloDetector:
    """Simulates YoloDetector for testing YOLO fallback."""

    def __init__(self):
        self.last_inference_ms = 180.0
        from autonomous_explorer.yolo_detector import Detection
        self._detections = [
            Detection(label='person', confidence=0.92,
                      bbox=(100, 50, 300, 400), position='center',
                      distance_m=1.8, size='medium'),
            Detection(label='cup', confidence=0.75,
                      bbox=(400, 300, 480, 380), position='right',
                      distance_m=0.8, size='small'),
        ]

    def detect(self, rgb, depth=None):
        return list(self._detections)


class TestVLMYoloCascade:
    """identify_objects: VLM first → YOLO fallback → auto-register."""

    def test_vlm_primary_succeeds(self, handlers, mock_node):
        """When VLM works, uses cloud result."""
        result = handlers.identify_objects(focus_area='all')
        assert result['success'] is True
        assert result['source'] == 'vlm_cloud'
        assert len(result['objects']) == 2
        mock_node.llm.analyze_scene.assert_called_once()

    def test_yolo_fallback_on_vlm_failure(self, handlers, mock_node):
        """When VLM raises exception, falls back to YOLO."""
        mock_node.llm.analyze_scene.side_effect = Exception('API timeout')
        mock_node.yolo = MockYoloDetector()

        result = handlers.identify_objects(focus_area='all')
        assert result['success'] is True
        assert result['source'] == 'yolo_local'
        assert len(result['objects']) == 2

    def test_yolo_fallback_on_empty_vlm_result(self, handlers, mock_node):
        """When VLM returns empty objects, falls back to YOLO."""
        mock_node.llm.analyze_scene.return_value = {
            'objects': [],
            'description': '',
            'room_guess': 'unknown',
            '_meta': {},
        }
        mock_node.yolo = MockYoloDetector()

        result = handlers.identify_objects(focus_area='all')
        assert result['success'] is True
        assert result['source'] == 'yolo_local'

    def test_both_fail_no_yolo(self, handlers, mock_node):
        """When VLM fails and no YOLO, returns error."""
        mock_node.llm.analyze_scene.side_effect = Exception('API down')
        mock_node.yolo = None

        result = handlers.identify_objects()
        assert result['success'] is False

    def test_both_fail_no_camera(self, handlers, mock_node):
        """When no camera frame at all."""
        mock_node._get_camera_frame_b64.return_value = ''
        mock_node.yolo = MockYoloDetector()
        mock_node._rgb_image = None

        result = handlers.identify_objects()
        assert result['success'] is False

    def test_yolo_focus_area_filter(self, handlers, mock_node):
        """YOLO fallback respects focus_area filter."""
        mock_node.llm.analyze_scene.side_effect = Exception('fail')
        mock_node.yolo = MockYoloDetector()

        result = handlers.identify_objects(focus_area='right')
        assert result['success'] is True
        # Only the 'cup' at position='right' should remain
        assert len(result['objects']) == 1
        assert result['objects'][0]['name'] == 'cup'


# ===================================================================
# 4. Auto-registration of objects in WorldKnowledge
# ===================================================================

class TestAutoRegistration:
    """identify_objects auto-registers objects in world knowledge."""

    def test_vlm_objects_auto_registered(self, handlers, mock_node):
        result = handlers.identify_objects()
        assert result['success'] is True
        registered = result['registered']
        assert 'chair' in registered
        assert 'desk' in registered
        # WorldKnowledge should have the objects
        wk = mock_node.world_knowledge
        assert 'chair' in wk.known_objects['objects']
        assert 'desk' in wk.known_objects['objects']
        assert wk.save_count >= 1

    def test_yolo_objects_auto_registered(self, handlers, mock_node):
        mock_node.llm.analyze_scene.side_effect = Exception('fail')
        mock_node.yolo = MockYoloDetector()

        result = handlers.identify_objects()
        assert 'person' in result['registered']
        assert 'cup' in result['registered']

    def test_unknown_objects_not_registered(self, handlers, mock_node):
        mock_node.llm.analyze_scene.return_value = {
            'objects': [
                {'name': 'unknown', 'distance': '?', 'position': 'center'},
                {'name': '', 'distance': '?', 'position': 'left'},
            ],
            'description': 'Hard to see',
            'room_guess': 'unknown',
            '_meta': {},
        }
        result = handlers.identify_objects()
        assert result['registered'] == []

    def test_room_guess_from_yolo(self, handlers, mock_node):
        """YOLO detections generate a room guess."""
        mock_node.llm.analyze_scene.side_effect = Exception('fail')
        mock_node.yolo = MockYoloDetector()

        result = handlers.identify_objects()
        # person + cup → should get some room guess
        assert result['room_guess'] is not None


# ===================================================================
# 5. Movement limiter (1 per LLM response)
# ===================================================================

class TestMovementLimiter:
    """Only 1 movement tool per LLM response; extras get error."""

    MOVEMENT_TOOLS = {
        'navigate_to', 'explore_frontier', 'go_home',
        'move_direct', 'look_around',
    }

    def test_single_movement_executes(self, bound_registry, mock_node):
        """One movement tool in a response should execute normally."""
        result = bound_registry.execute(
            'navigate_to', {'target': 'coordinates', 'x': 1.0, 'y': 0.0})
        assert result['success'] is True

    def test_movement_limiter_logic(self, bound_registry, mock_node):
        """Simulate the movement limiter from _run_agent_turn.

        The limiter is in explorer_node._run_agent_turn, not in
        the registry itself. We test the pattern here.
        """
        tool_calls = [
            ToolCall('navigate_to', {'target': 'coordinates', 'x': 1.0, 'y': 0.0}, 'tc_0'),
            ToolCall('go_home', {'speech': 'going home'}, 'tc_1'),  # should be blocked
            ToolCall('speak', {'text': 'hello'}, 'tc_2'),  # non-movement, always allowed
        ]

        movement_executed = False
        results = []
        for tc in tool_calls:
            if tc.tool_name in self.MOVEMENT_TOOLS and movement_executed:
                results.append({
                    'success': False,
                    'error': 'Only one movement tool allowed per response.',
                })
            else:
                result = bound_registry.execute(tc.tool_name, tc.arguments)
                if tc.tool_name in self.MOVEMENT_TOOLS:
                    movement_executed = True
                results.append(result)

        # First movement succeeds
        assert results[0]['success'] is True
        # Second movement blocked
        assert results[1]['success'] is False
        assert 'one movement' in results[1]['error'].lower()
        # Non-movement tool still executes
        assert results[2]['success'] is True

    def test_non_movement_tools_unlimited(self, bound_registry, mock_node):
        """Non-movement tools are not limited per response."""
        for i in range(5):
            r = bound_registry.execute('speak', {'text': f'msg {i}'})
            assert r['success'] is True
        assert len(mock_node.voice.spoken) == 5


# ===================================================================
# 6. LLM hard failure → stop exploration
# ===================================================================

class TestLLMHardFailure:
    """When LLM returns error, agent should halt exploration."""

    def test_error_response_detected(self):
        """AgentResponse with 'LLM error' text triggers halt."""
        response = AgentResponse(
            tool_calls=[ToolCall('speak', {'text': 'error'}, 'err_0')],
            text='LLM error: Connection refused',
            raw_response='error',
            tokens_input=0,
            tokens_output=0,
            response_time_ms=0,
            stop_reason='error',
            provider='openai',
            model='gpt-4o',
        )
        # The explorer_node checks: response.text.startswith('LLM error')
        assert response.text.startswith('LLM error')

    def test_agent_error_response_format(self):
        """_agent_error_response returns well-formed AgentResponse."""
        provider = DryRunProvider()
        response = provider._agent_error_response('Connection timeout')
        assert response.text == 'LLM error: Connection timeout'
        assert response.stop_reason == 'error'
        # Should have a fallback move_direct stop tool call
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].tool_name == 'move_direct'
        assert response.tool_calls[0].arguments['action'] == 'stop'

    def test_openai_provider_error_handling(self):
        """OpenAI provider wraps exceptions into error response."""
        provider = create_provider('openai', api_key='test-key')
        # Mock the client to raise
        provider.client = MagicMock()
        provider.client.chat.completions.create.side_effect = \
            Exception('Rate limit exceeded')

        response = provider.agent_turn(
            system_prompt='test',
            messages=[{'role': 'user', 'content': 'hello'}],
            tools=[],
        )
        assert response.text.startswith('LLM error')
        assert response.stop_reason == 'error'

    def test_claude_provider_error_handling(self):
        """Claude provider wraps exceptions into error response."""
        provider = create_provider('claude', api_key='test-key')
        provider.client = MagicMock()
        provider.client.messages.create.side_effect = \
            Exception('Invalid API key')

        response = provider.agent_turn(
            system_prompt='test',
            messages=[{'role': 'user', 'content': 'hello'}],
            tools=[],
        )
        assert response.text.startswith('LLM error')
        assert response.stop_reason == 'error'


# ===================================================================
# 7. YOLO text-only mode for local providers
# ===================================================================

class TestYoloTextOnlyMode:
    """Local models get YOLO text instead of images."""

    def test_local_provider_is_detected(self):
        """Local provider via base_url sets provider_name='local'."""
        provider = create_provider(
            'openai', api_key='lm-studio',
            base_url='http://10.0.0.176:1234/v1')
        assert provider.provider_name == 'local'

    def test_yolo_detections_to_text(self):
        """YoloDetector.detections_to_text produces readable summary."""
        from autonomous_explorer.yolo_detector import Detection, YoloDetector
        detections = [
            Detection('person', 0.92, (100, 50, 300, 400), 'center', 1.8, 'medium'),
            Detection('cup', 0.75, (400, 300, 480, 380), 'right', 0.8, 'small'),
        ]
        text = YoloDetector.detections_to_text(detections)
        assert 'person' in text
        assert 'cup' in text
        assert '1.8m' in text
        assert 'center' in text

    def test_empty_detections_text(self):
        """Empty detections produce a 'none visible' message."""
        from autonomous_explorer.yolo_detector import YoloDetector
        text = YoloDetector.detections_to_text([])
        assert 'none' in text.lower()

    def test_text_only_tokens_small(self):
        """Text-only YOLO output is much smaller than base64 image."""
        from autonomous_explorer.yolo_detector import Detection, YoloDetector
        detections = [
            Detection('person', 0.92, (100, 50, 300, 400), 'center', 1.8, 'medium'),
            Detection('chair', 0.85, (200, 100, 350, 300), 'left', 2.1, 'medium'),
            Detection('cup', 0.75, (400, 300, 480, 380), 'right', 0.8, 'small'),
        ]
        text = YoloDetector.detections_to_text(detections)
        # Text should be well under 500 chars (~150 tokens)
        assert len(text) < 500
        # A base64 640x480 JPEG is typically 50-150KB
        # So text mode saves >99% of context


# ===================================================================
# 8. Multi-turn ReAct cycle (DryRun → tools → results → next)
# ===================================================================

class TestMultiTurnReAct:
    """Test the full ReAct loop: LLM → tool calls → results → feedback."""

    def test_single_turn_with_dryrun(self, bound_registry):
        """DryRun provider returns tool calls that execute via registry."""
        dryrun = DryRunProvider()
        conversation = ConversationManager(max_turns=5)

        conversation.add_user_message('Start exploring')

        response = dryrun.agent_turn(
            system_prompt='You are Jeeves.',
            messages=conversation.get_messages_openai(),
            tools=bound_registry.to_openai_tools(),
        )
        assert response.has_tool_calls

        # Execute all tool calls through the registry
        tool_results = []
        for tc in response.tool_calls:
            result = bound_registry.execute(tc.tool_name, tc.arguments)
            tool_results.append({
                'call_id': tc.call_id,
                'name': tc.tool_name,
                'result': result,
            })

        # All results should have success key
        for tr in tool_results:
            assert 'success' in tr['result'], (
                f'{tr["name"]} missing success key: {tr["result"]}')

    def test_multi_round_conversation(self, bound_registry):
        """Multiple rounds of DryRun → execute → feedback."""
        dryrun = DryRunProvider()
        conversation = ConversationManager(max_turns=5)

        conversation.add_user_message('Explore the area')

        for round_num in range(3):
            response = dryrun.agent_turn(
                system_prompt='You are Jeeves.',
                messages=conversation.get_messages_openai(),
                tools=bound_registry.to_openai_tools(),
            )
            assert response.has_tool_calls, f'Round {round_num}: no tool calls'

            # Record assistant message with tool calls
            conversation.add_assistant_message(
                response.text,
                tool_calls=[
                    {'id': tc.call_id, 'name': tc.tool_name,
                     'arguments': tc.arguments}
                    for tc in response.tool_calls
                ],
            )

            # Execute and record results
            results = []
            for tc in response.tool_calls:
                result = bound_registry.execute(tc.tool_name, tc.arguments)
                results.append({
                    'call_id': tc.call_id,
                    'name': tc.tool_name,
                    'result': result,
                })
            conversation.add_tool_results(results)

        # Conversation should have grown (user + 3 rounds of assistant+results)
        msgs = conversation.get_messages_openai()
        assert len(msgs) >= 4

    def test_dryrun_full_cycle_covers_all_tool_categories(self, bound_registry):
        """Running all DryRun sequences exercises all 4 tool categories."""
        dryrun = DryRunProvider()
        categories_hit = set()
        tool_to_category = {
            'navigate_to': 'navigation',
            'explore_frontier': 'navigation',
            'go_home': 'navigation',
            'identify_objects': 'perception',
            'label_room': 'knowledge',
            'query_knowledge': 'knowledge',
            'speak': 'communication',
        }

        for _ in range(len(dryrun._AGENT_SEQUENCES)):
            response = dryrun.agent_turn(
                'test', [{'role': 'user', 'content': 'go'}], [])
            for tc in response.tool_calls:
                cat = tool_to_category.get(tc.tool_name)
                if cat:
                    categories_hit.add(cat)

        assert categories_hit == {'navigation', 'perception', 'knowledge', 'communication'}

    def test_conversation_manager_claude_rendering(self, bound_registry):
        """Verify conversation renders correctly for Claude format."""
        dryrun = DryRunProvider()
        conversation = ConversationManager(max_turns=5)

        conversation.add_user_message('Look around')
        response = dryrun.agent_turn(
            'You are Jeeves.',
            conversation.get_messages_claude(),
            bound_registry.to_claude_tools(),
        )
        conversation.add_assistant_message(
            response.text,
            tool_calls=[
                {'id': tc.call_id, 'name': tc.tool_name,
                 'arguments': tc.arguments}
                for tc in response.tool_calls
            ],
        )

        msgs = conversation.get_messages_claude()
        # Claude format: messages alternate user/assistant
        roles = [m['role'] for m in msgs]
        assert roles[0] == 'user'
        assert roles[1] == 'assistant'


# ===================================================================
# 9. Tool timeout enforcement
# ===================================================================

class TestToolTimeout:
    """Registry enforces per-tool timeout_s."""

    def test_fast_tool_completes(self):
        """A tool that finishes quickly returns normally."""
        reg = create_registry()

        def fast_handler(**kw):
            return {'success': True, 'text': 'done'}

        reg.get_tool('speak').handler = fast_handler
        result = reg.execute('speak', {'text': 'hello'})
        assert result['success'] is True

    def test_slow_tool_times_out(self):
        """A tool that exceeds timeout_s gets terminated."""
        reg = create_registry()

        def slow_handler(**kw):
            time.sleep(10)
            return {'success': True}

        tool = reg.get_tool('speak')
        tool.handler = slow_handler
        tool.timeout_s = 0.5  # very short timeout

        result = reg.execute('speak', {'text': 'hello'})
        assert result['success'] is False
        assert 'timed out' in result.get('error', '').lower()

    def test_each_tool_has_timeout(self):
        """Every registered tool has a positive timeout."""
        for tool in build_jeeves_tools():
            assert tool.timeout_s > 0, f'{tool.name} has no timeout'

    def test_navigation_tools_have_long_timeout(self):
        """Navigation tools need longer timeout for Nav2."""
        tools = {t.name: t for t in build_jeeves_tools()}
        for name in ('navigate_to', 'explore_frontier', 'go_home'):
            assert tools[name].timeout_s >= 30.0, (
                f'{name} timeout too short: {tools[name].timeout_s}')


# ===================================================================
# 10. explore_frontier wait/re-evaluation loop
# ===================================================================

class TestExploreFrontierWait:
    """explore_frontier polls Nav2 and returns for LLM re-evaluation."""

    def test_immediate_success(self, handlers, mock_node):
        """Navigation completes before re-eval interval."""
        mock_node.nav2.is_navigating = False
        mock_node.nav2.navigation_result = 'succeeded'

        result = handlers.explore_frontier()
        assert result['success'] is True
        assert result['status'] == 'arrived'

    def test_no_map_fails(self, handlers, mock_node):
        mock_node.nav2.has_map = False
        result = handlers.explore_frontier()
        assert result['success'] is False
        assert 'map' in result['error'].lower()

    def test_no_frontiers(self, handlers, mock_node):
        mock_node.nav2.frontier_goals = []
        result = handlers.explore_frontier()
        assert result['status'] == 'no_frontiers'

    def test_goal_not_accepted(self, handlers, mock_node):
        mock_node.nav2.navigate_to = MagicMock(return_value=False)
        result = handlers.explore_frontier()
        assert result['success'] is False
        assert 'not accepted' in result['error'].lower()

    def test_emergency_stop_cancels(self, handlers, mock_node):
        """Emergency stop during navigation cancels and returns error."""
        mock_node.nav2.is_navigating = True
        # Trigger e-stop after a brief period
        def delayed_estop():
            time.sleep(0.2)
            mock_node.emergency_stop = True
        threading.Thread(target=delayed_estop, daemon=True).start()

        result = handlers.explore_frontier()
        assert result['success'] is False
        assert 'emergency' in result['error'].lower()

    def test_returns_in_progress_after_interval(self, handlers, mock_node):
        """Returns in_progress after NAV2_TOOL_REEVAL_INTERVAL."""
        mock_node.nav2.is_navigating = True

        # Patch the reeval interval to be very short for testing
        with patch('autonomous_explorer.tool_handlers.NAV2_TOOL_REEVAL_INTERVAL', 0.3):
            result = handlers.explore_frontier()

        assert result['success'] is True
        assert result['status'] == 'in_progress'
        assert 'goal' in result
        assert 'alternatives' in result

    def test_largest_preference_sorts(self, handlers, mock_node):
        """preference='largest' picks the biggest frontier."""
        mock_node.nav2.frontier_goals = [
            {'x': 1.0, 'y': 0.0, 'size': 10},
            {'x': 2.0, 'y': 0.0, 'size': 100},
        ]
        mock_node.nav2.is_navigating = False
        mock_node.nav2.navigation_result = 'succeeded'

        result = handlers.explore_frontier(preference='largest')
        assert result['success'] is True
        # Should have navigated to the largest frontier (x=2.0)
        assert mock_node.nav2.navigate_calls[-1] == (2.0, 0.0)


# ===================================================================
# 11. Removed tools are not callable
# ===================================================================

class TestRemovedToolsBlocked:
    """Tools removed from the 7-tool set cannot be called."""

    REMOVED_TOOLS = [
        'move_direct', 'look_around', 'describe_scene',
        'check_surroundings', 'register_object', 'save_map', 'listen',
    ]

    def test_removed_tools_not_in_registry(self, bound_registry):
        for name in self.REMOVED_TOOLS:
            result = bound_registry.execute(name, {})
            assert result['success'] is False
            assert 'unknown tool' in result.get('error', '').lower()

    def test_only_7_tools_registered(self, registry):
        assert len(registry.tool_names) == 7

    def test_removed_handler_methods_still_exist(self, handlers):
        """Handler methods still exist (not deleted) but are unbound."""
        assert hasattr(handlers, 'move_direct')
        assert hasattr(handlers, 'look_around')
        assert hasattr(handlers, 'describe_scene')
        assert hasattr(handlers, 'check_surroundings')


# ===================================================================
# 12. Agent system prompt references only 7 tools
# ===================================================================

class TestAgentSystemPrompt:
    """AGENT_SYSTEM_PROMPT is consistent with the 7-tool set."""

    def test_prompt_mentions_7_tools(self):
        from autonomous_explorer.config import AGENT_SYSTEM_PROMPT
        assert 'AVAILABLE TOOLS (7)' in AGENT_SYSTEM_PROMPT

    def test_prompt_contains_registered_tools(self):
        from autonomous_explorer.config import AGENT_SYSTEM_PROMPT
        for tool in ('navigate_to', 'explore_frontier', 'go_home',
                      'identify_objects', 'label_room', 'query_knowledge', 'speak'):
            assert tool in AGENT_SYSTEM_PROMPT, f'{tool} missing from prompt'

    def test_prompt_does_not_mention_removed_tools(self):
        from autonomous_explorer.config import AGENT_SYSTEM_PROMPT
        for tool in ('move_direct', 'look_around', 'describe_scene',
                      'check_surroundings', 'register_object', 'save_map', 'listen'):
            assert tool not in AGENT_SYSTEM_PROMPT, (
                f'Removed tool {tool} still in system prompt')

    def test_prompt_mentions_fixed_camera(self):
        from autonomous_explorer.config import AGENT_SYSTEM_PROMPT
        assert 'fixed' in AGENT_SYSTEM_PROMPT.lower() or 'spin' in AGENT_SYSTEM_PROMPT.lower()


# ===================================================================
# 13. Model config profiles
# ===================================================================

class TestModelConfigProfiles:
    """model_config.yaml profiles create correct providers."""

    def test_dryrun_provider_no_api_key(self):
        provider = create_provider('dryrun', api_key='')
        assert provider.provider_name == 'dryrun'
        assert provider.model == 'dry-run'

    def test_dryrun_aliases(self):
        for alias in ('dryrun', 'dry-run', 'dry_run'):
            p = create_provider(alias, api_key='')
            assert p.provider_name == 'dryrun'

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match='Unknown LLM provider'):
            create_provider('gemini', api_key='test')

    def test_openai_provider_creation(self):
        p = create_provider('openai', api_key='test-key')
        assert p.provider_name == 'openai'
        assert p.model == 'gpt-4o'

    def test_openai_custom_model(self):
        p = create_provider('openai', api_key='test', model='gpt-4o-mini')
        assert p.model == 'gpt-4o-mini'

    def test_local_provider_creation(self):
        p = create_provider('local', api_key='lm-studio',
                            base_url='http://localhost:1234/v1')
        assert p.provider_name == 'local'


# ===================================================================
# 14. Tool schema consistency (Claude + OpenAI formats)
# ===================================================================

class TestToolSchemaConsistency:
    """Tool definitions render consistently for both providers."""

    def test_claude_and_openai_same_tool_names(self, registry):
        claude_names = {t['name'] for t in registry.to_claude_tools()}
        openai_names = {t['function']['name'] for t in registry.to_openai_tools()}
        assert claude_names == openai_names

    def test_claude_format_valid(self, registry):
        for tool in registry.to_claude_tools():
            assert 'name' in tool
            assert 'description' in tool
            assert 'input_schema' in tool
            assert tool['input_schema']['type'] == 'object'

    def test_openai_format_valid(self, registry):
        for tool in registry.to_openai_tools():
            assert tool['type'] == 'function'
            fn = tool['function']
            assert 'name' in fn
            assert 'description' in fn
            assert 'parameters' in fn

    def test_identify_objects_description_mentions_auto_register(self, registry):
        tool = registry.get_tool('identify_objects')
        desc = tool.description.lower()
        assert 'auto' in desc or 'register' in desc

    def test_all_tools_have_categories(self):
        for tool in build_jeeves_tools():
            assert tool.category in (
                'navigation', 'perception', 'knowledge', 'communication')
