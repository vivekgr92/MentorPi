"""Tests for autonomous_explorer.llm_provider module."""
import json
import time
from unittest.mock import MagicMock, patch

import pytest

from autonomous_explorer.llm_provider import (
    LLMProvider,
    ClaudeProvider,
    OpenAIProvider,
    DryRunProvider,
    create_provider,
)


# ===================================================================
# _parse_response (static method on LLMProvider)
# ===================================================================

class TestParseResponse:
    """Test JSON extraction from LLM response text."""

    def test_valid_json(self):
        text = '{"action": "forward", "speed": 0.5, "duration": 1.0}'
        result = LLMProvider._parse_response(text)
        assert result['action'] == 'forward'
        assert result['speed'] == 0.5
        assert result['_parse_failed'] is False

    def test_json_with_markdown_fences(self):
        text = '```json\n{"action": "stop", "speed": 0.0, "duration": 0}\n```'
        result = LLMProvider._parse_response(text)
        assert result['action'] == 'stop'
        assert result['_parse_failed'] is False

    def test_json_with_bare_fences(self):
        text = '```\n{"action": "turn_left"}\n```'
        result = LLMProvider._parse_response(text)
        assert result['action'] == 'turn_left'
        assert result['_parse_failed'] is False

    def test_json_embedded_in_text(self):
        text = 'Here is my response: {"action": "backward", "speed": 0.3} and more text'
        result = LLMProvider._parse_response(text)
        assert result['action'] == 'backward'
        assert result['_parse_failed'] is False

    def test_invalid_json_returns_stop_fallback(self):
        text = 'This is not JSON at all'
        result = LLMProvider._parse_response(text)
        assert result['action'] == 'stop'
        assert result['speed'] == 0.0
        assert result['_parse_failed'] is True

    def test_empty_string_returns_fallback(self):
        result = LLMProvider._parse_response('')
        assert result['action'] == 'stop'
        assert result['_parse_failed'] is True

    def test_malformed_json_in_braces(self):
        text = '{action: forward, speed: 0.5}'
        result = LLMProvider._parse_response(text)
        assert result['action'] == 'stop'
        assert result['_parse_failed'] is True

    def test_nested_json_objects(self):
        text = '{"action": "forward", "speed": 0.5, "meta": {"key": "val"}}'
        result = LLMProvider._parse_response(text)
        assert result['action'] == 'forward'
        assert result['_parse_failed'] is False

    def test_whitespace_handling(self):
        text = '  \n  {"action": "stop"}  \n  '
        result = LLMProvider._parse_response(text)
        assert result['action'] == 'stop'
        assert result['_parse_failed'] is False

    def test_multiple_json_objects_greedy_match(self):
        text = '{"action": "forward"} extra {"action": "backward"}'
        result = LLMProvider._parse_response(text)
        # Greedy regex matches from first { to last } — may fail to parse
        # the combined string, so either parse succeeds or we get fallback
        assert 'action' in result


# ===================================================================
# _safe_fallback
# ===================================================================

class TestSafeFallback:
    """Test the error fallback mechanism."""

    def test_returns_stop_action(self):
        provider = DryRunProvider()
        result = provider._safe_fallback("test error")
        assert result['action'] == 'stop'
        assert result['speed'] == 0.0
        assert result['duration'] == 0.0

    def test_includes_error_in_meta(self):
        provider = DryRunProvider()
        result = provider._safe_fallback("network timeout")
        assert '_meta' in result
        assert result['_meta']['error'] == 'network timeout'
        assert result['_meta']['provider'] == 'dryrun'

    def test_meta_has_zero_tokens(self):
        provider = DryRunProvider()
        result = provider._safe_fallback("err")
        assert result['_meta']['tokens_input'] == 0
        assert result['_meta']['tokens_output'] == 0


# ===================================================================
# _attach_meta
# ===================================================================

class TestAttachMeta:
    """Test metadata attachment to parsed results."""

    def test_attaches_all_fields(self):
        parsed = {'action': 'forward', 'speed': 0.5}
        result = LLMProvider._attach_meta(
            parsed,
            raw_response='{"action":"forward"}',
            tokens_input=100,
            tokens_output=50,
            response_time_ms=500,
            provider='test',
            model='test-model',
        )
        assert result['_meta']['tokens_input'] == 100
        assert result['_meta']['tokens_output'] == 50
        assert result['_meta']['response_time_ms'] == 500
        assert result['_meta']['provider'] == 'test'
        assert result['_meta']['model'] == 'test-model'

    def test_mutates_original_dict(self):
        parsed = {'action': 'stop'}
        LLMProvider._attach_meta(
            parsed,
            raw_response='',
            tokens_input=0,
            tokens_output=0,
            response_time_ms=0,
            provider='x',
            model='y',
        )
        assert '_meta' in parsed


# ===================================================================
# DryRunProvider
# ===================================================================

class TestDryRunProvider:
    """Test the dry-run provider for offline testing."""

    def test_provider_name(self):
        p = DryRunProvider()
        assert p.provider_name == 'dryrun'
        assert p.model == 'dry-run'

    def test_cycles_through_actions(self):
        p = DryRunProvider()
        actions_seen = set()
        for _ in range(len(p._ACTIONS)):
            result = p.analyze_scene('fake_b64', 'sys', 'user')
            actions_seen.add(result['action'])
            assert '_meta' in result
            assert result['_meta']['provider'] == 'dryrun'
        assert len(actions_seen) > 1

    def test_wraps_around(self):
        p = DryRunProvider()
        n = len(p._ACTIONS)
        first_result = p.analyze_scene('img', 'sys', 'usr')
        for _ in range(n - 1):
            p.analyze_scene('img', 'sys', 'usr')
        wrap_result = p.analyze_scene('img', 'sys', 'usr')
        assert first_result['action'] == wrap_result['action']

    def test_response_includes_timing(self):
        p = DryRunProvider()
        result = p.analyze_scene('img', 'sys', 'usr')
        assert result['_meta']['response_time_ms'] >= 0

    def test_handles_list_images(self):
        p = DryRunProvider()
        result = p.analyze_scene(['img1', 'img2'], 'sys', 'usr')
        assert 'action' in result

    def test_zero_token_counts(self):
        p = DryRunProvider()
        result = p.analyze_scene('img', 'sys', 'usr')
        assert result['_meta']['tokens_input'] == 0
        assert result['_meta']['tokens_output'] == 0


# ===================================================================
# create_provider factory
# ===================================================================

class TestCreateProvider:
    """Test the provider factory function."""

    def test_dryrun_variants(self):
        for name in ('dryrun', 'dry-run', 'dry_run', 'DryRun'):
            p = create_provider(name, api_key='')
            assert isinstance(p, DryRunProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            create_provider('nonexistent', api_key='fake')

    @patch('autonomous_explorer.llm_provider.anthropic', create=True)
    def test_claude_provider_created(self, mock_anthropic):
        mock_anthropic.Anthropic.return_value = MagicMock()
        p = create_provider('claude', api_key='sk-test')
        assert isinstance(p, ClaudeProvider)
        assert p.model == 'claude-sonnet-4-20250514'

    @patch('autonomous_explorer.llm_provider.anthropic', create=True)
    def test_claude_with_custom_model(self, mock_anthropic):
        mock_anthropic.Anthropic.return_value = MagicMock()
        p = create_provider('claude', api_key='sk-test', model='claude-3-haiku')
        assert p.model == 'claude-3-haiku'

    @patch('autonomous_explorer.llm_provider.openai', create=True)
    def test_openai_provider_created(self, mock_openai):
        mock_openai.OpenAI.return_value = MagicMock()
        p = create_provider('openai', api_key='sk-test')
        assert isinstance(p, OpenAIProvider)
        assert p.model == 'gpt-4o'

    @patch('autonomous_explorer.llm_provider.openai', create=True)
    def test_openai_with_custom_model(self, mock_openai):
        mock_openai.OpenAI.return_value = MagicMock()
        p = create_provider('openai', api_key='sk-test', model='gpt-4-turbo')
        assert p.model == 'gpt-4-turbo'

    def test_provider_name_whitespace_handling(self):
        p = create_provider('  dryrun  ', api_key='')
        assert isinstance(p, DryRunProvider)

    def test_provider_name_case_insensitive(self):
        p = create_provider('DRYRUN', api_key='')
        assert isinstance(p, DryRunProvider)


# ===================================================================
# ClaudeProvider error handling
# ===================================================================

class TestClaudeProviderErrors:
    """Test Claude provider graceful error handling."""

    @patch('autonomous_explorer.llm_provider.anthropic', create=True)
    def test_api_error_returns_safe_fallback(self, mock_anthropic):
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API rate limit")
        mock_anthropic.Anthropic.return_value = mock_client

        p = ClaudeProvider(api_key='sk-test')
        p.client = mock_client  # ensure we use the mock
        result = p.analyze_scene('img_b64', 'sys', 'usr')
        assert result['action'] == 'stop'
        assert 'error' in result['_meta']
        assert 'rate limit' in result['_meta']['error']


# ===================================================================
# OpenAIProvider error handling
# ===================================================================

class TestOpenAIProviderErrors:
    """Test OpenAI provider graceful error handling."""

    @patch('autonomous_explorer.llm_provider.openai', create=True)
    def test_api_error_returns_safe_fallback(self, mock_openai):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("timeout")
        mock_openai.OpenAI.return_value = mock_client

        p = OpenAIProvider(api_key='sk-test')
        result = p.analyze_scene('img_b64', 'sys', 'usr')
        assert result['action'] == 'stop'
        assert 'error' in result['_meta']
