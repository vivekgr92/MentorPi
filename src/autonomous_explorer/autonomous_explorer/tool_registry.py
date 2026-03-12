#!/usr/bin/env python3
# encoding: utf-8
"""
Tool Registry — defines tools available to the Jeeves LLM agent.

Each tool has a name, description, JSON schema for parameters, and a handler
callable. The registry converts tool definitions to both Claude and OpenAI
native tool-calling formats and dispatches tool calls to their handlers.

Usage:
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name='navigate_to',
        description='Navigate to a named location or coordinates.',
        parameters={...},
        handler=node.tool_navigate_to,
    ))

    # Get tools for LLM API call
    tools = registry.to_claude_tools()   # or .to_openai_tools()

    # Execute a tool call from the LLM response
    result = registry.execute('navigate_to', {'target': 'kitchen'})
"""
from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """A tool available to the LLM agent."""

    name: str
    description: str
    parameters: dict  # JSON Schema for the parameters object
    handler: Callable[..., dict] | None = None
    timeout_s: float = 30.0
    category: str = 'general'  # navigation, perception, knowledge, communication


class ToolRegistry:
    """Registry of tools available to the LLM agent.

    Provides tool definitions in both Claude and OpenAI format,
    and dispatches tool calls to their handler methods.
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def execute(self, name: str, params: dict) -> dict:
        """Execute a tool call, returning the result dict.

        Enforces the tool's timeout_s -- if the handler takes longer,
        returns a timeout error without killing the thread (Python
        limitation), but the agent loop can continue.
        """
        tool = self._tools.get(name)
        if not tool:
            return {'success': False, 'error': f'Unknown tool: {name}'}
        if tool.handler is None:
            return {'success': False, 'error': f'Tool {name} has no handler'}
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(tool.handler, **params)
                return future.result(timeout=tool.timeout_s)
        except concurrent.futures.TimeoutError:
            logger.warning('Tool %s timed out after %.0fs', name, tool.timeout_s)
            return {
                'success': False,
                'error': f'Tool {name} timed out after {tool.timeout_s:.0f}s',
                'timed_out': True,
            }
        except TypeError as e:
            return {'success': False, 'error': f'Bad parameters for {name}: {e}'}
        except Exception as e:
            logger.exception('Tool %s raised an exception', name)
            return {'success': False, 'error': f'Tool error: {e}'}

    # ------------------------------------------------------------------
    # Provider-specific format converters
    # ------------------------------------------------------------------

    def to_claude_tools(self) -> list[dict]:
        """Convert to Claude API tools format (Anthropic tool_use)."""
        return [
            {
                'name': t.name,
                'description': t.description,
                'input_schema': t.parameters,
            }
            for t in self._tools.values()
        ]

    def to_openai_tools(self) -> list[dict]:
        """Convert to OpenAI API tools format (function calling)."""
        return [
            {
                'type': 'function',
                'function': {
                    'name': t.name,
                    'description': t.description,
                    'parameters': t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def to_provider_tools(self, provider_name: str) -> list[dict]:
        """Convert to the correct format for the given provider."""
        if provider_name == 'claude':
            return self.to_claude_tools()
        return self.to_openai_tools()


# ======================================================================
# Tool definitions — 7 Jeeves demo tools (hackathon set)
# ======================================================================

def build_jeeves_tools() -> list[ToolDefinition]:
    """Return the 7 Jeeves demo tool definitions (no handlers bound).

    Handlers are bound later by the agent node via registry.get_tool(name).handler = ...
    """
    return [
        # --- Navigation ---
        ToolDefinition(
            name='navigate_to',
            description=(
                'Navigate to a named location or map coordinates using Nav2 '
                'path planning. The robot plans a path avoiding obstacles and '
                'drives autonomously. Use for destinations > 0.5m away.'
            ),
            category='navigation',
            timeout_s=60.0,
            parameters={
                'type': 'object',
                'properties': {
                    'target': {
                        'type': 'string',
                        'description': (
                            'Room name (e.g. "kitchen"), "coordinates" '
                            'for raw x/y, or "approach" to drive directly '
                            'toward a visible object (requires object_name).'
                        ),
                    },
                    'x': {
                        'type': 'number',
                        'description': 'Map x in meters (required if target="coordinates").',
                    },
                    'y': {
                        'type': 'number',
                        'description': 'Map y in meters (required if target="coordinates").',
                    },
                    'object_name': {
                        'type': 'string',
                        'description': (
                            'If set, after arriving at the target area the robot '
                            'does a sensor-guided final approach (LiDAR + depth) '
                            'and stops ~10cm in front of the object. Use '
                            'target="approach" to approach a visible object directly.'
                        ),
                    },
                    'speech': {
                        'type': 'string',
                        'description': 'What to say while navigating.',
                    },
                },
                'required': ['target'],
            },
        ),

        ToolDefinition(
            name='explore_frontier',
            description=(
                'Navigate to the nearest unexplored area on the SLAM map. '
                'Picks the best frontier goal based on size and distance. '
                'Use when you want to discover new areas.'
            ),
            category='navigation',
            timeout_s=60.0,
            parameters={
                'type': 'object',
                'properties': {
                    'preference': {
                        'type': 'string',
                        'enum': ['nearest', 'largest'],
                        'description': 'How to pick the frontier.',
                    },
                    'speech': {
                        'type': 'string',
                        'description': 'What to say while exploring.',
                    },
                },
                'required': [],
            },
        ),

        ToolDefinition(
            name='go_home',
            description=(
                'Navigate back to the starting position (map origin 0,0). '
                'Use when the user asks you to return or at end of exploration.'
            ),
            category='navigation',
            timeout_s=60.0,
            parameters={
                'type': 'object',
                'properties': {
                    'speech': {
                        'type': 'string',
                        'description': 'What to say while returning.',
                    },
                },
                'required': [],
            },
        ),

        # --- Perception ---
        ToolDefinition(
            name='identify_objects',
            description=(
                'Detect and list objects in the current camera frame with '
                'distances and positions. Uses VLM (cloud) with YOLO fallback. '
                'Automatically registers discovered objects in the knowledge '
                'graph. Also provides a scene description and room type guess.'
            ),
            category='perception',
            timeout_s=15.0,
            parameters={
                'type': 'object',
                'properties': {
                    'focus_area': {
                        'type': 'string',
                        'enum': ['left', 'center', 'right', 'all'],
                        'description': 'Which part of the frame to focus on.',
                    },
                },
                'required': [],
            },
        ),

        # --- Knowledge ---
        ToolDefinition(
            name='label_room',
            description=(
                'Label the current location as a named room in the knowledge '
                'graph. Associates current map coordinates with a room name, '
                'description, and connections to adjacent rooms.'
            ),
            category='knowledge',
            timeout_s=2.0,
            parameters={
                'type': 'object',
                'properties': {
                    'room_name': {
                        'type': 'string',
                        'description': 'Room name (e.g. "kitchen").',
                    },
                    'description': {
                        'type': 'string',
                        'description': 'Brief room description.',
                    },
                    'connections': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'Adjacent room names.',
                    },
                    'speech': {
                        'type': 'string',
                        'description': 'What to say about labeling this room.',
                    },
                },
                'required': ['room_name'],
            },
        ),

        ToolDefinition(
            name='query_knowledge',
            description=(
                'Search the knowledge graph for rooms, objects, or spatial '
                'relationships. Use when asked "where is the [object]", '
                '"what is in the [room]", "list rooms", etc.'
            ),
            category='knowledge',
            timeout_s=2.0,
            parameters={
                'type': 'object',
                'properties': {
                    'query_type': {
                        'type': 'string',
                        'enum': [
                            'find_object', 'describe_room', 'list_rooms',
                            'list_objects', 'room_connections',
                        ],
                        'description': 'Type of knowledge query.',
                    },
                    'query': {
                        'type': 'string',
                        'description': 'Object name, room name, or search term.',
                    },
                },
                'required': ['query_type', 'query'],
            },
        ),

        # --- Communication ---
        ToolDefinition(
            name='speak',
            description=(
                'Say something out loud through the speaker. Use to respond '
                'to the user, announce discoveries, or narrate actions.'
            ),
            category='communication',
            timeout_s=10.0,
            parameters={
                'type': 'object',
                'properties': {
                    'text': {
                        'type': 'string',
                        'description': 'What to say.',
                    },
                    'wait': {
                        'type': 'boolean',
                        'description': 'Block until speech finishes (default true).',
                    },
                },
                'required': ['text'],
            },
        ),
    ]


def create_registry() -> ToolRegistry:
    """Create a ToolRegistry pre-loaded with 7 Jeeves demo tool definitions.

    Handlers are None — the agent node binds them after construction:
        registry = create_registry()
        registry.get_tool('navigate_to').handler = self.tool_navigate_to
    """
    registry = ToolRegistry()
    for tool in build_jeeves_tools():
        registry.register(tool)
    return registry
