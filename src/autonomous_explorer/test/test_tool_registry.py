"""Tests for autonomous_explorer.tool_registry module."""
import time

import pytest

from autonomous_explorer.tool_registry import (
    ToolDefinition,
    ToolRegistry,
    build_jeeves_tools,
    create_registry,
)


# ===================================================================
# ToolDefinition
# ===================================================================

class TestToolDefinition:

    def test_dataclass_fields(self):
        td = ToolDefinition(
            name='test_tool',
            description='A test tool',
            parameters={'type': 'object', 'properties': {}},
        )
        assert td.name == 'test_tool'
        assert td.handler is None
        assert td.timeout_s == 30.0
        assert td.category == 'general'

    def test_with_handler(self):
        def handler(**kwargs):
            return {'success': True}

        td = ToolDefinition(
            name='t', description='d',
            parameters={'type': 'object', 'properties': {}},
            handler=handler,
        )
        assert td.handler is handler


# ===================================================================
# ToolRegistry
# ===================================================================

class TestToolRegistry:

    def _make_registry(self):
        reg = ToolRegistry()
        reg.register(ToolDefinition(
            name='greet',
            description='Say hello',
            parameters={
                'type': 'object',
                'properties': {
                    'name': {'type': 'string', 'description': 'Who to greet'},
                },
                'required': ['name'],
            },
            handler=lambda name='world': {'success': True, 'msg': f'Hello {name}'},
            category='test',
        ))
        return reg

    def test_register_and_get(self):
        reg = self._make_registry()
        assert reg.get_tool('greet') is not None
        assert reg.get_tool('nonexistent') is None

    def test_tool_names(self):
        reg = self._make_registry()
        assert 'greet' in reg.tool_names

    def test_execute_success(self):
        reg = self._make_registry()
        result = reg.execute('greet', {'name': 'Alice'})
        assert result['success'] is True
        assert result['msg'] == 'Hello Alice'

    def test_execute_unknown_tool(self):
        reg = self._make_registry()
        result = reg.execute('no_such_tool', {})
        assert result['success'] is False
        assert 'Unknown tool' in result['error']

    def test_execute_no_handler(self):
        reg = ToolRegistry()
        reg.register(ToolDefinition(
            name='empty',
            description='No handler',
            parameters={'type': 'object', 'properties': {}},
        ))
        result = reg.execute('empty', {})
        assert result['success'] is False
        assert 'no handler' in result['error']

    def test_execute_bad_params(self):
        reg = ToolRegistry()
        reg.register(ToolDefinition(
            name='strict',
            description='Strict params',
            parameters={'type': 'object', 'properties': {}},
            handler=lambda required_arg: {'success': True},
        ))
        result = reg.execute('strict', {'wrong_arg': 1})
        assert result['success'] is False
        assert 'Bad parameters' in result['error']

    def test_execute_handler_exception(self):
        reg = ToolRegistry()
        reg.register(ToolDefinition(
            name='boom',
            description='Explodes',
            parameters={'type': 'object', 'properties': {}},
            handler=lambda: (_ for _ in ()).throw(RuntimeError('kaboom')),
        ))
        result = reg.execute('boom', {})
        assert result['success'] is False
        assert 'kaboom' in result['error']


# ===================================================================
# Timeout enforcement in execute()
# ===================================================================

class TestExecuteTimeout:
    """Verify that ToolRegistry.execute() enforces per-tool timeout_s."""

    def test_fast_tool_returns_normally(self):
        """A handler that completes within timeout_s returns its result."""
        reg = ToolRegistry()
        reg.register(ToolDefinition(
            name='fast',
            description='Completes quickly',
            parameters={'type': 'object', 'properties': {}},
            handler=lambda: {'success': True, 'value': 42},
            timeout_s=5.0,
        ))
        result = reg.execute('fast', {})
        assert result == {'success': True, 'value': 42}

    def test_slow_tool_returns_timeout_error(self):
        """A handler that exceeds timeout_s returns a timeout error dict."""
        def slow_handler():
            time.sleep(10)
            return {'success': True}

        reg = ToolRegistry()
        reg.register(ToolDefinition(
            name='slow',
            description='Takes too long',
            parameters={'type': 'object', 'properties': {}},
            handler=slow_handler,
            timeout_s=0.2,
        ))
        start = time.monotonic()
        result = reg.execute('slow', {})
        elapsed = time.monotonic() - start

        assert result['success'] is False
        assert result['timed_out'] is True
        assert 'timed out' in result['error']
        assert 'slow' in result['error']
        # Should return after ~0.2s, not 10s
        assert elapsed < 2.0

    def test_exception_in_handler_returns_error(self):
        """A handler that raises still returns an error dict (not a timeout)."""
        def exploding_handler():
            raise ValueError('sensor disconnected')

        reg = ToolRegistry()
        reg.register(ToolDefinition(
            name='explode',
            description='Raises an exception',
            parameters={'type': 'object', 'properties': {}},
            handler=exploding_handler,
            timeout_s=5.0,
        ))
        result = reg.execute('explode', {})
        assert result['success'] is False
        assert 'sensor disconnected' in result['error']
        assert 'timed_out' not in result


# ===================================================================
# Claude/OpenAI format conversion
# ===================================================================

class TestFormatConversion:

    def test_to_claude_tools(self):
        reg = create_registry()
        tools = reg.to_claude_tools()
        assert len(tools) == 14
        for t in tools:
            assert 'name' in t
            assert 'description' in t
            assert 'input_schema' in t
            assert t['input_schema']['type'] == 'object'

    def test_to_openai_tools(self):
        reg = create_registry()
        tools = reg.to_openai_tools()
        assert len(tools) == 14
        for t in tools:
            assert t['type'] == 'function'
            assert 'name' in t['function']
            assert 'description' in t['function']
            assert 'parameters' in t['function']

    def test_tool_names_match(self):
        reg = create_registry()
        claude_names = {t['name'] for t in reg.to_claude_tools()}
        openai_names = {t['function']['name'] for t in reg.to_openai_tools()}
        assert claude_names == openai_names

    def test_to_provider_tools(self):
        reg = create_registry()
        assert len(reg.to_provider_tools('claude')) == 14
        assert len(reg.to_provider_tools('openai')) == 14


# ===================================================================
# build_jeeves_tools — all 14 tools
# ===================================================================

class TestBuildJeevesTools:

    EXPECTED_TOOLS = {
        'navigate_to', 'explore_frontier', 'move_direct', 'go_home',
        'look_around', 'identify_objects', 'describe_scene', 'check_surroundings',
        'label_room', 'register_object', 'query_knowledge', 'save_map',
        'speak', 'listen',
    }

    def test_returns_14_tools(self):
        tools = build_jeeves_tools()
        assert len(tools) == 14

    def test_all_expected_tools_present(self):
        tools = build_jeeves_tools()
        names = {t.name for t in tools}
        assert names == self.EXPECTED_TOOLS

    def test_all_have_descriptions(self):
        for t in build_jeeves_tools():
            assert len(t.description) > 10, f'{t.name} has short description'

    def test_all_have_valid_schemas(self):
        for t in build_jeeves_tools():
            assert t.parameters['type'] == 'object'
            assert 'properties' in t.parameters

    def test_navigate_to_has_required_params(self):
        tools = {t.name: t for t in build_jeeves_tools()}
        nav = tools['navigate_to']
        assert 'target' in nav.parameters['properties']
        assert 'target' in nav.parameters.get('required', [])

    def test_move_direct_has_required_params(self):
        tools = {t.name: t for t in build_jeeves_tools()}
        move = tools['move_direct']
        props = move.parameters['properties']
        assert 'action' in props
        assert 'speed' in props
        assert 'duration' in props

    def test_speak_has_text_param(self):
        tools = {t.name: t for t in build_jeeves_tools()}
        speak = tools['speak']
        assert 'text' in speak.parameters['properties']

    def test_query_knowledge_has_enum(self):
        tools = {t.name: t for t in build_jeeves_tools()}
        qk = tools['query_knowledge']
        qt = qk.parameters['properties']['query_type']
        assert 'enum' in qt
        assert 'find_object' in qt['enum']
        assert 'list_rooms' in qt['enum']

    def test_handlers_default_to_none(self):
        for t in build_jeeves_tools():
            assert t.handler is None, f'{t.name} should have no handler by default'


# ===================================================================
# create_registry factory
# ===================================================================

class TestCreateRegistry:

    def test_returns_registry_with_14_tools(self):
        reg = create_registry()
        assert len(reg.tool_names) == 14

    def test_all_tools_have_no_handlers(self):
        reg = create_registry()
        for name in reg.tool_names:
            tool = reg.get_tool(name)
            assert tool.handler is None
