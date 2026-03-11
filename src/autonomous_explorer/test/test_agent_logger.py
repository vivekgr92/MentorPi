"""Tests for autonomous_explorer.agent_logger module."""
import logging
import re

import pytest

from autonomous_explorer.agent_logger import (
    AgentLogger,
    _format_args,
    _format_result_summary,
    _truncate,
)


# ===================================================================
# Helper utilities
# ===================================================================

class TestTruncate:

    def test_short_string_unchanged(self):
        assert _truncate('hello', 10) == 'hello'

    def test_exact_length_unchanged(self):
        assert _truncate('hello', 5) == 'hello'

    def test_long_string_truncated(self):
        result = _truncate('a' * 50, 10)
        assert len(result) == 10
        assert result.endswith('...')

    def test_empty_string(self):
        assert _truncate('', 10) == ''


class TestFormatArgs:

    def test_empty_args(self):
        assert _format_args({}) == ''

    def test_simple_args(self):
        result = _format_args({'speed': 0.5, 'duration': 2.0})
        assert 'speed=0.5' in result
        assert 'duration=2.0' in result

    def test_string_arg_truncated(self):
        result = _format_args({'text': 'x' * 100})
        assert '...' in result
        assert len(result) <= 120

    def test_total_length_capped(self):
        args = {f'key_{i}': f'value_{i}' for i in range(20)}
        result = _format_args(args)
        assert len(result) <= 120


class TestFormatResultSummary:

    def test_short_summary(self):
        assert _format_result_summary('ok') == 'ok'

    def test_long_summary_truncated(self):
        result = _format_result_summary('z' * 100)
        assert len(result) <= 60
        assert result.endswith('...')


# ===================================================================
# AgentLogger
# ===================================================================

@pytest.fixture
def published():
    """Collect strings published via publish_fn."""
    return []


@pytest.fixture
def logger():
    """A standard Python logger for testing."""
    log = logging.getLogger('test_agent_logger')
    log.setLevel(logging.DEBUG)
    return log


@pytest.fixture
def agent_log(logger, published):
    return AgentLogger(
        logger=logger,
        publish_fn=lambda text: published.append(text),
    )


class TestAgentLoggerMethods:

    def test_voice_received(self, agent_log, published):
        agent_log.voice_received('find me something to drink')
        assert len(published) == 1
        assert 'VOICE' in published[0]
        assert 'find me something to drink' in published[0]

    def test_turn_start(self, agent_log, published):
        agent_log.turn_start(5)
        assert 'TURN_START' in published[0]
        assert 'cycle #5' in published[0]

    def test_llm_request_basic(self, agent_log, published):
        agent_log.llm_request(
            provider='openai',
            num_tools=14,
            num_messages=8,
            has_image=True,
        )
        line = published[0]
        assert 'LLM_REQUEST' in line
        assert 'openai' in line
        assert '14 tools' in line
        assert '8 messages' in line
        assert 'image=yes' in line

    def test_llm_request_no_image(self, agent_log, published):
        agent_log.llm_request(
            provider='claude', num_tools=14,
            num_messages=3, has_image=False,
        )
        assert 'image=no' in published[0]

    def test_llm_request_with_voice(self, agent_log, published):
        agent_log.llm_request(
            provider='claude', num_tools=14,
            num_messages=3, has_image=False,
            voice_instruction='go to the kitchen',
        )
        assert 'voice=' in published[0]
        assert 'go to the kitchen' in published[0]

    def test_llm_response(self, agent_log, published):
        agent_log.llm_response(
            num_tool_calls=2,
            stop_reason='tool_use',
            response_ms=1823,
            tokens_in=4521,
            tokens_out=312,
        )
        line = published[0]
        assert 'LLM_RESPONSE' in line
        assert '2 tool calls' in line
        assert 'stop=tool_use' in line
        assert '1823ms' in line
        assert '4521+312 tokens' in line

    def test_tool_start(self, agent_log, published):
        agent_log.tool_start(
            'explore_frontier',
            {'preference': 'nearest'},
            timeout_s=60.0,
        )
        line = published[0]
        assert 'TOOL_START' in line
        assert 'explore_frontier' in line
        assert "preference='nearest'" in line
        assert 'timeout=60s' in line

    def test_tool_start_empty_args(self, agent_log, published):
        agent_log.tool_start('check_surroundings', {}, timeout_s=2.0)
        assert 'check_surroundings()' in published[0]

    def test_tool_result_success(self, agent_log, published):
        agent_log.tool_result(
            'explore_frontier',
            success=True,
            duration_ms=8012,
            summary='navigating to (2.1, -0.5)',
        )
        line = published[0]
        assert 'TOOL_RESULT' in line
        assert 'explore_frontier' in line
        assert 'success' in line
        assert '8012ms' in line
        assert 'navigating to (2.1, -0.5)' in line

    def test_tool_result_failure(self, agent_log, published):
        agent_log.tool_result(
            'navigate_to',
            success=False,
            duration_ms=50,
            summary='no path found',
        )
        assert 'failed' in published[0]

    def test_tool_error(self, agent_log, published):
        agent_log.tool_error(
            'navigate_to', 'Nav2 not available', 12,
        )
        line = published[0]
        assert 'TOOL_ERROR' in line
        assert 'navigate_to' in line
        assert 'Nav2 not available' in line
        assert '12ms' in line

    def test_turn_complete(self, agent_log, published):
        agent_log.turn_complete(
            num_tools_executed=3,
            total_duration_s=14.2,
            total_cost=0.0082,
        )
        line = published[0]
        assert 'TURN_COMPLETE' in line
        assert '3 tools' in line
        assert '14.2s total' in line
        assert '$0.0082' in line

    def test_status(self, agent_log, published):
        agent_log.status('waiting for camera frame')
        assert 'STATUS' in published[0]
        assert 'waiting for camera frame' in published[0]


class TestAgentLoggerTimestamp:

    def test_all_lines_have_timestamp(self, agent_log, published):
        agent_log.voice_received('test')
        agent_log.turn_start(1)
        agent_log.status('hi')
        # HH:MM:SS pattern
        ts_pattern = re.compile(r'\d{2}:\d{2}:\d{2}')
        for line in published:
            assert ts_pattern.search(line), f'No timestamp in: {line}'

    def test_all_lines_have_agent_prefix(self, agent_log, published):
        agent_log.voice_received('test')
        agent_log.turn_start(1)
        agent_log.llm_request('claude', 14, 5, True)
        agent_log.llm_response(1, 'end_turn', 100, 500, 50)
        agent_log.tool_start('speak', {'text': 'hi'}, 10.0)
        agent_log.tool_result('speak', True, 50, 'spoken')
        agent_log.turn_complete(1, 1.5, 0.001)
        agent_log.status('done')
        for line in published:
            assert line.startswith('[AGENT]'), f'Missing prefix: {line}'


class TestAgentLoggerStats:

    def test_tool_stats_empty(self, agent_log):
        assert agent_log.get_tool_stats() == {}

    def test_tool_stats_tracked(self, agent_log):
        agent_log.tool_result('speak', True, 100, 'ok')
        agent_log.tool_result('speak', True, 200, 'ok')
        agent_log.tool_result('move_direct', True, 500, 'ok')
        agent_log.tool_error('navigate_to', 'timeout', 3000)

        stats = agent_log.get_tool_stats()
        assert stats['speak']['calls'] == 2
        assert stats['speak']['total_ms'] == 300
        assert stats['speak']['avg_ms'] == 150.0
        assert stats['move_direct']['calls'] == 1
        assert stats['navigate_to']['calls'] == 1
        assert stats['navigate_to']['total_ms'] == 3000


class TestAgentLoggerPublishFailure:

    def test_publish_failure_does_not_raise(self, logger):
        """If publish_fn raises, the logger should not crash."""
        def bad_publish(text):
            raise RuntimeError('node shutting down')

        agent_log = AgentLogger(logger=logger, publish_fn=bad_publish)
        # Should not raise
        agent_log.voice_received('test')
        agent_log.turn_start(1)
        agent_log.status('hello')
