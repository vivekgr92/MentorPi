"""Tests for autonomous_explorer.conversation_manager module."""
import json
import math

import pytest

from autonomous_explorer.conversation_manager import (
    AgentResponse,
    ConversationManager,
    ToolCall,
    build_identity_context,
    build_sensor_context,
)


# ===================================================================
# ToolCall dataclass
# ===================================================================

class TestToolCall:

    def test_fields(self):
        tc = ToolCall(tool_name='speak', arguments={'text': 'hi'}, call_id='tc_1')
        assert tc.tool_name == 'speak'
        assert tc.arguments == {'text': 'hi'}
        assert tc.call_id == 'tc_1'


# ===================================================================
# AgentResponse dataclass
# ===================================================================

class TestAgentResponse:

    def test_defaults(self):
        r = AgentResponse()
        assert r.tool_calls == []
        assert r.text is None
        assert r.stop_reason == 'end_turn'

    def test_has_tool_calls_false(self):
        assert AgentResponse().has_tool_calls is False

    def test_has_tool_calls_true(self):
        r = AgentResponse(tool_calls=[
            ToolCall('speak', {'text': 'hi'}, 'tc_1'),
        ])
        assert r.has_tool_calls is True

    def test_cost_estimate(self):
        r = AgentResponse(tokens_input=1_000_000, tokens_output=100_000)
        cost = r.cost_estimate_usd
        assert cost > 0
        assert cost < 100  # sanity check


# ===================================================================
# ConversationManager — message management
# ===================================================================

class TestConversationManagerMessages:

    def test_empty(self):
        cm = ConversationManager(max_turns=5)
        assert cm.message_count == 0

    def test_add_user_message_text_only(self):
        cm = ConversationManager()
        cm.add_user_message('Hello')
        assert cm.message_count == 1
        msgs = cm.get_messages_claude()
        assert msgs[0]['role'] == 'user'
        assert msgs[0]['content'] == 'Hello'

    def test_add_user_message_with_images_b64(self):
        cm = ConversationManager()
        cm.add_user_message('Describe', images_b64=['abc123'])
        msgs = cm.get_messages_claude()
        assert isinstance(msgs[0]['content'], list)
        assert msgs[0]['content'][0]['type'] == 'image'
        assert msgs[0]['content'][1]['type'] == 'text'

    def test_add_user_message_with_image_b64_singular(self):
        cm = ConversationManager()
        cm.add_user_message('Describe', image_b64='abc123')
        msgs = cm.get_messages_claude()
        assert isinstance(msgs[0]['content'], list)
        assert len(msgs[0]['content']) == 2

    def test_add_assistant_message_text_only(self):
        cm = ConversationManager()
        cm.add_user_message('Hi')
        cm.add_assistant_message('Hello back')
        assert cm.message_count == 2
        msgs = cm.get_messages_claude()
        assert msgs[1]['role'] == 'assistant'
        assert msgs[1]['content'] == 'Hello back'

    def test_add_assistant_message_none_text_no_tools(self):
        cm = ConversationManager()
        cm.add_user_message('Hi')
        cm.add_assistant_message(None, tool_calls=None)
        # Nothing should be added
        assert cm.message_count == 1

    def test_add_assistant_message_with_tool_calls(self):
        cm = ConversationManager()
        cm.add_user_message('Explore')
        cm.add_assistant_message('Let me check', tool_calls=[
            {'id': 'tc_1', 'name': 'check_surroundings', 'arguments': {}},
            {'id': 'tc_2', 'name': 'speak', 'arguments': {'text': 'Looking around'}},
        ])
        assert cm.message_count == 2

    def test_add_tool_result(self):
        cm = ConversationManager()
        cm.add_user_message('Go')
        cm.add_assistant_message(None, tool_calls=[
            {'id': 'tc_1', 'name': 'check_surroundings', 'arguments': {}},
        ])
        cm.add_tool_result('tc_1', 'check_surroundings', {'success': True, 'lidar': {}})
        assert cm.message_count == 3

    def test_add_tool_results_bulk(self):
        cm = ConversationManager()
        cm.add_user_message('Go')
        cm.add_assistant_message(None, tool_calls=[
            {'id': 'tc_1', 'name': 'speak', 'arguments': {'text': 'hi'}},
            {'id': 'tc_2', 'name': 'look_around', 'arguments': {}},
        ])
        cm.add_tool_results([
            {'call_id': 'tc_1', 'name': 'speak', 'result': {'success': True}},
            {'call_id': 'tc_2', 'name': 'look_around', 'result': {'success': True}},
        ])
        assert cm.message_count == 4  # user + assistant + 2 tool results

    def test_tool_result_truncation(self):
        cm = ConversationManager()
        cm.add_user_message('Go')
        cm.add_assistant_message(None, tool_calls=[
            {'id': 'tc_1', 'name': 'test', 'arguments': {}},
        ])
        big_result = {'success': True, 'data': 'x' * 1000}
        cm.add_tool_result('tc_1', 'test', big_result)
        # The stored content should be truncated
        stored = cm._messages[-1]['content']
        assert len(stored) <= 500

    def test_clear(self):
        cm = ConversationManager()
        cm.add_user_message('Hi')
        cm.add_assistant_message('Hello')
        cm.clear()
        assert cm.message_count == 0


# ===================================================================
# ConversationManager — sliding window trimming
# ===================================================================

class TestSlidingWindow:

    def test_trims_at_max_turns(self):
        cm = ConversationManager(max_turns=2)
        cm.add_user_message('Turn 1')
        cm.add_assistant_message('Reply 1')
        cm.add_user_message('Turn 2')
        cm.add_assistant_message('Reply 2')
        cm.add_user_message('Turn 3')  # triggers trim
        # Turn 1 should be removed
        msgs = cm.get_messages_claude()
        assert msgs[0]['content'] == 'Turn 2'

    def test_preserves_recent_turns(self):
        cm = ConversationManager(max_turns=3)
        for i in range(5):
            cm.add_user_message(f'User {i}')
            cm.add_assistant_message(f'Bot {i}')
        # Should have turns 2, 3, 4
        msgs = cm.get_messages_claude()
        user_msgs = [m for m in msgs if m['role'] == 'user']
        assert user_msgs[0]['content'] == 'User 2'

    def test_single_turn_window(self):
        cm = ConversationManager(max_turns=1)
        cm.add_user_message('First')
        cm.add_user_message('Second')
        msgs = cm.get_messages_claude()
        assert len(msgs) == 1
        assert msgs[0]['content'] == 'Second'


# ===================================================================
# ConversationManager — Claude format rendering
# ===================================================================

class TestClaudeRendering:

    def test_simple_conversation(self):
        cm = ConversationManager()
        cm.add_user_message('Hi')
        cm.add_assistant_message('Hello')
        msgs = cm.get_messages_claude()
        assert len(msgs) == 2
        assert msgs[0]['role'] == 'user'
        assert msgs[1]['role'] == 'assistant'

    def test_tool_calls_rendered_as_content_blocks(self):
        cm = ConversationManager()
        cm.add_user_message('Explore')
        cm.add_assistant_message('Checking', tool_calls=[
            {'id': 'tc_1', 'name': 'check_surroundings', 'arguments': {}},
        ])
        msgs = cm.get_messages_claude()
        assistant_msg = msgs[1]
        assert assistant_msg['role'] == 'assistant'
        assert isinstance(assistant_msg['content'], list)
        # Should have text block + tool_use block
        types = [block['type'] for block in assistant_msg['content']]
        assert 'text' in types
        assert 'tool_use' in types

    def test_tool_results_in_user_message(self):
        cm = ConversationManager()
        cm.add_user_message('Go')
        cm.add_assistant_message(None, tool_calls=[
            {'id': 'tc_1', 'name': 'speak', 'arguments': {'text': 'hi'}},
        ])
        cm.add_tool_result('tc_1', 'speak', {'success': True})
        msgs = cm.get_messages_claude()
        # Tool result should be in a user message with tool_result blocks
        tool_msg = msgs[2]
        assert tool_msg['role'] == 'user'
        assert isinstance(tool_msg['content'], list)
        assert tool_msg['content'][0]['type'] == 'tool_result'
        assert tool_msg['content'][0]['tool_use_id'] == 'tc_1'

    def test_multiple_tool_results_grouped(self):
        cm = ConversationManager()
        cm.add_user_message('Go')
        cm.add_assistant_message(None, tool_calls=[
            {'id': 'tc_1', 'name': 'speak', 'arguments': {}},
            {'id': 'tc_2', 'name': 'look_around', 'arguments': {}},
        ])
        cm.add_tool_results([
            {'call_id': 'tc_1', 'name': 'speak', 'result': {'success': True}},
            {'call_id': 'tc_2', 'name': 'look_around', 'result': {'success': True}},
        ])
        msgs = cm.get_messages_claude()
        # Both tool results should be in a single user message
        tool_msg = msgs[2]
        assert tool_msg['role'] == 'user'
        assert len(tool_msg['content']) == 2
        assert all(b['type'] == 'tool_result' for b in tool_msg['content'])


# ===================================================================
# ConversationManager — OpenAI format rendering
# ===================================================================

class TestOpenAIRendering:

    def test_simple_conversation(self):
        cm = ConversationManager()
        cm.add_user_message('Hi')
        cm.add_assistant_message('Hello')
        msgs = cm.get_messages_openai()
        assert len(msgs) == 2
        assert msgs[0]['role'] == 'user'
        assert msgs[1]['role'] == 'assistant'

    def test_tool_calls_rendered_as_function_calls(self):
        cm = ConversationManager()
        cm.add_user_message('Explore')
        cm.add_assistant_message('Checking', tool_calls=[
            {'id': 'tc_1', 'name': 'check_surroundings', 'arguments': {'foo': 'bar'}},
        ])
        msgs = cm.get_messages_openai()
        assistant_msg = msgs[1]
        assert assistant_msg['role'] == 'assistant'
        assert 'tool_calls' in assistant_msg
        tc = assistant_msg['tool_calls'][0]
        assert tc['type'] == 'function'
        assert tc['function']['name'] == 'check_surroundings'
        assert json.loads(tc['function']['arguments']) == {'foo': 'bar'}

    def test_tool_results_as_tool_role(self):
        cm = ConversationManager()
        cm.add_user_message('Go')
        cm.add_assistant_message(None, tool_calls=[
            {'id': 'tc_1', 'name': 'speak', 'arguments': {}},
        ])
        cm.add_tool_result('tc_1', 'speak', {'success': True})
        msgs = cm.get_messages_openai()
        tool_msg = msgs[2]
        assert tool_msg['role'] == 'tool'
        assert tool_msg['tool_call_id'] == 'tc_1'

    def test_get_messages_provider_dispatch(self):
        cm = ConversationManager()
        cm.add_user_message('Test')
        assert len(cm.get_messages('claude')) == 1
        assert len(cm.get_messages('openai')) == 1


# ===================================================================
# ConversationManager — add_assistant_tool_calls (legacy)
# ===================================================================

class TestLegacyAssistantToolCalls:

    def test_from_agent_response(self):
        cm = ConversationManager()
        cm.add_user_message('Go')
        response = AgentResponse(
            tool_calls=[ToolCall('speak', {'text': 'hi'}, 'tc_1')],
            text='Thinking...',
        )
        cm.add_assistant_tool_calls(response)
        msgs = cm.get_messages_claude()
        assert len(msgs) == 2
        assert msgs[1]['role'] == 'assistant'

    def test_text_only_response(self):
        cm = ConversationManager()
        cm.add_user_message('Hello')
        response = AgentResponse(text='Just text, no tools')
        cm.add_assistant_tool_calls(response)
        msgs = cm.get_messages_claude()
        assert msgs[1]['content'] == 'Just text, no tools'


# ===================================================================
# build_sensor_context (module-level function)
# ===================================================================

class TestBuildSensorContext:

    def test_empty(self):
        ctx = build_sensor_context()
        assert ctx == ''

    def test_with_lidar(self):
        ctx = build_sensor_context(
            lidar_summary='LiDAR: front=1.5m, left=2.0m',
        )
        assert 'LiDAR' in ctx
        assert 'front=1.5m' in ctx

    def test_with_depth(self):
        ctx = build_sensor_context(depth_summary='Depth: center=150cm')
        assert 'Depth' in ctx

    def test_with_odom(self):
        ctx = build_sensor_context(odom={'x': 1.5, 'y': 2.0, 'theta': 0.5})
        assert 'POSITION' in ctx
        assert '1.50' in ctx

    def test_with_imu(self):
        ctx = build_sensor_context(
            imu={'orientation': {'roll': 0.1, 'pitch': -0.05, 'yaw': 1.2}},
        )
        assert 'IMU' in ctx
        assert 'roll' in ctx

    def test_with_battery(self):
        ctx = build_sensor_context(battery=11.8)
        assert 'BATTERY' in ctx
        assert '11.8' in ctx

    def test_emergency_stop(self):
        ctx = build_sensor_context(emergency_stop=True)
        assert 'EMERGENCY STOP' in ctx

    def test_all_fields(self):
        ctx = build_sensor_context(
            lidar_summary='LiDAR: front=1.0m',
            depth_summary='Depth: center=80cm',
            odom={'x': 0, 'y': 0, 'theta': 0},
            imu={'orientation': {'roll': 0, 'pitch': 0, 'yaw': 0}},
            battery=12.0,
            emergency_stop=False,
        )
        lines = ctx.strip().split('\n')
        assert len(lines) >= 4


# ===================================================================
# build_identity_context (module-level function)
# ===================================================================

class TestBuildIdentityContext:

    def test_with_none(self):
        ctx = build_identity_context(None, None)
        assert ctx == ''

    def test_with_consciousness(self):
        class MockConsciousness:
            def get_identity_context(self):
                return 'Outing #5, 42 rooms discovered'

        ctx = build_identity_context(MockConsciousness(), None)
        assert 'Outing #5' in ctx

    def test_with_world_knowledge(self):
        class MockKnowledge:
            def get_prompt_context(self, x=0, y=0, theta=0):
                return 'KNOWN ROOMS: kitchen, hallway'

        ctx = build_identity_context(None, MockKnowledge())
        assert 'kitchen' in ctx

    def test_with_odom(self):
        class MockKnowledge:
            def get_prompt_context(self, x=0, y=0, theta=0):
                return f'pos=({x:.1f},{y:.1f})'

        ctx = build_identity_context(
            None, MockKnowledge(),
            odom={'x': 1.5, 'y': 2.0, 'theta': 0.3},
        )
        assert '1.5' in ctx
        assert '2.0' in ctx


# ===================================================================
# Full ReAct cycle simulation
# ===================================================================

class TestReActCycle:
    """Test a complete ReAct cycle through ConversationManager."""

    def test_full_cycle(self):
        cm = ConversationManager(max_turns=5)

        # User turn with sensor context
        cm.add_user_message('Sensors: front=2.0m', image_b64='fake_img')

        # Assistant responds with tool calls
        cm.add_assistant_message('Let me check', tool_calls=[
            {'id': 'tc_1', 'name': 'check_surroundings', 'arguments': {}},
            {'id': 'tc_2', 'name': 'speak', 'arguments': {'text': 'Looking around'}},
        ])

        # Tool results
        cm.add_tool_results([
            {'call_id': 'tc_1', 'name': 'check_surroundings',
             'result': {'success': True, 'lidar': {'front': 2.0}}},
            {'call_id': 'tc_2', 'name': 'speak',
             'result': {'success': True, 'spoken': True}},
        ])

        # Assistant follows up with action
        cm.add_assistant_message('Path is clear', tool_calls=[
            {'id': 'tc_3', 'name': 'move_direct',
             'arguments': {'action': 'forward', 'speed': 0.7, 'duration': 2.0}},
        ])

        # Tool result
        cm.add_tool_result('tc_3', 'move_direct', {'success': True})

        # Final text response
        cm.add_assistant_message('Moved forward successfully.')

        # Verify rendering for both providers
        claude_msgs = cm.get_messages_claude()
        openai_msgs = cm.get_messages_openai()
        assert len(claude_msgs) >= 4
        assert len(openai_msgs) >= 4

        # Verify roles alternate correctly for Claude
        roles = [m['role'] for m in claude_msgs]
        for i in range(len(roles) - 1):
            # Claude requires user/assistant alternation
            # tool_results are rendered as user messages
            if roles[i] == 'assistant':
                assert roles[i + 1] in ('user',), \
                    f'After assistant, got {roles[i+1]} at index {i+1}'
