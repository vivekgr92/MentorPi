#!/usr/bin/env python3
# encoding: utf-8
"""
Conversation Manager — sliding window conversation history for the Jeeves agent.

Maintains a bounded conversation history that feeds into LLM tool-calling turns.
Each turn can contain user messages (with sensor context + images), assistant
messages (with tool calls), and tool results.

The conversation is assembled as:
  1. System prompt (static, ~300 tokens)
  2. Identity + knowledge context (dynamic, ~200 tokens per cycle)
  3. Conversation history (last N turns, ~500-1000 tokens)
  4. Latest sensor snapshot + camera image

Token budget target: <2200 input tokens per cycle.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """A single tool invocation from the LLM."""

    tool_name: str
    arguments: dict
    call_id: str  # Provider-assigned ID for matching results


@dataclass
class AgentResponse:
    """Response from one LLM agent turn."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    text: str | None = None
    raw_response: str = ''
    tokens_input: int = 0
    tokens_output: int = 0
    response_time_ms: int = 0
    stop_reason: str = 'end_turn'  # 'tool_use', 'end_turn', 'max_tokens'
    provider: str = ''
    model: str = ''

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def cost_estimate_usd(self) -> float:
        """Rough cost estimate based on token counts."""
        # Approximate rates (Claude Sonnet / GPT-4o)
        input_rate = 3.0 / 1_000_000   # $3/M input tokens
        output_rate = 15.0 / 1_000_000  # $15/M output tokens
        return (self.tokens_input * input_rate
                + self.tokens_output * output_rate)


class ConversationManager:
    """Manages sliding-window conversation history for the agent loop.

    Keeps the last `max_turns` exchange rounds (user + assistant + tool_results).
    The system prompt and per-cycle context are assembled separately.
    """

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self._messages: list[dict] = []
        self._turn_boundaries: list[int] = []

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def clear(self) -> None:
        self._messages.clear()
        self._turn_boundaries.clear()

    # ------------------------------------------------------------------
    # Adding messages
    # ------------------------------------------------------------------

    def add_user_message(
        self,
        text: str,
        images_b64: list[str] | None = None,
        image_b64: str | None = None,
    ) -> None:
        """Add a user message with optional images.

        This marks the start of a new turn for window-trimming purposes.
        Accepts either images_b64 (list) or image_b64 (single string).
        """
        self._turn_boundaries.append(len(self._messages))
        self._trim_window()

        if image_b64 and not images_b64:
            images_b64 = [image_b64]
        msg = self._build_user_message(text, images_b64)
        self._messages.append(msg)

    def add_assistant_message(
        self,
        text: str | None,
        tool_calls: list[dict] | None = None,
    ) -> None:
        """Add an assistant message, optionally with tool calls.

        Args:
            text: Assistant's text response (reasoning, speech, etc.)
            tool_calls: List of dicts with 'id', 'name', 'arguments' keys.
                        None for text-only responses.
        """
        if not tool_calls:
            if text:
                self._messages.append({
                    'role': 'assistant',
                    'content': text,
                })
            return

        self._messages.append({
            'role': 'assistant',
            '_tool_calls': tool_calls,
            '_text': text,
        })

    def add_assistant_tool_calls(
        self,
        response: AgentResponse,
    ) -> None:
        """Add an assistant message from an AgentResponse.

        Legacy method — prefer add_assistant_message() for new code.
        """
        tc_list = None
        if response.has_tool_calls:
            tc_list = [
                {'id': tc.call_id, 'name': tc.tool_name,
                 'arguments': tc.arguments}
                for tc in response.tool_calls
            ]
        self.add_assistant_message(response.text, tool_calls=tc_list)

    def add_tool_result(
        self,
        call_id: str,
        tool_name: str,
        result: dict,
    ) -> None:
        """Add a tool result message."""
        # Truncate large results to keep token budget
        result_str = json.dumps(result, default=str)
        if len(result_str) > 500:
            result_str = result_str[:497] + '...'

        self._messages.append({
            'role': 'tool',
            '_call_id': call_id,
            '_tool_name': tool_name,
            'content': result_str,
        })

    def add_tool_results(self, results: list[dict]) -> None:
        """Add multiple tool results at once.

        Args:
            results: List of dicts with 'call_id', 'name', 'result' keys.
        """
        for r in results:
            self.add_tool_result(
                call_id=r['call_id'],
                tool_name=r['name'],
                result=r['result'],
            )

    # ------------------------------------------------------------------
    # Rendering messages for LLM API calls
    # ------------------------------------------------------------------

    def get_messages_claude(self) -> list[dict]:
        """Render conversation history in Claude API format.

        Claude uses:
          - user messages with content blocks (text + image)
          - assistant messages with tool_use content blocks
          - user messages with tool_result content blocks
        """
        rendered = []
        for msg in self._messages:
            role = msg['role']

            if role == 'user':
                rendered.append(msg)

            elif role == 'assistant':
                if '_tool_calls' in msg:
                    content = []
                    if msg.get('_text'):
                        content.append({'type': 'text', 'text': msg['_text']})
                    for tc in msg['_tool_calls']:
                        content.append({
                            'type': 'tool_use',
                            'id': tc['id'],
                            'name': tc['name'],
                            'input': tc['arguments'],
                        })
                    rendered.append({'role': 'assistant', 'content': content})
                else:
                    rendered.append(msg)

            elif role == 'tool':
                # Claude: tool results go in a user message with tool_result blocks
                # Check if previous rendered message is a user tool_result group
                tool_result_block = {
                    'type': 'tool_result',
                    'tool_use_id': msg['_call_id'],
                    'content': msg['content'],
                }
                if (rendered and rendered[-1]['role'] == 'user'
                        and isinstance(rendered[-1].get('content'), list)
                        and rendered[-1]['content']
                        and rendered[-1]['content'][0].get('type') == 'tool_result'):
                    rendered[-1]['content'].append(tool_result_block)
                else:
                    rendered.append({
                        'role': 'user',
                        'content': [tool_result_block],
                    })

        return rendered

    def get_messages_openai(self) -> list[dict]:
        """Render conversation history in OpenAI API format.

        OpenAI uses:
          - user messages with content array (text + image_url)
          - assistant messages with tool_calls array
          - tool messages with tool_call_id
        """
        rendered = []
        for msg in self._messages:
            role = msg['role']

            if role == 'user':
                rendered.append(self._convert_user_msg_openai(msg))

            elif role == 'assistant':
                if '_tool_calls' in msg:
                    oai_msg = {'role': 'assistant'}
                    if msg.get('_text'):
                        oai_msg['content'] = msg['_text']
                    oai_msg['tool_calls'] = [
                        {
                            'id': tc['id'],
                            'type': 'function',
                            'function': {
                                'name': tc['name'],
                                'arguments': json.dumps(tc['arguments']),
                            },
                        }
                        for tc in msg['_tool_calls']
                    ]
                    rendered.append(oai_msg)
                else:
                    rendered.append(msg)

            elif role == 'tool':
                rendered.append({
                    'role': 'tool',
                    'tool_call_id': msg['_call_id'],
                    'content': msg['content'],
                })

        return rendered

    def get_messages(self, provider: str) -> list[dict]:
        """Get messages in the correct format for the given provider."""
        if provider == 'claude':
            return self.get_messages_claude()
        return self.get_messages_openai()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_user_msg_openai(msg: dict) -> dict:
        """Convert a user message from Claude image format to OpenAI format.

        Claude uses {'type': 'image', 'source': {'data': ..., 'media_type': ...}}
        OpenAI uses {'type': 'image_url', 'image_url': {'url': 'data:...;base64,...'}}
        """
        content = msg.get('content')
        if not isinstance(content, list):
            return msg  # plain text, no conversion needed
        converted = []
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'image':
                src = block.get('source', {})
                b64 = src.get('data', '')
                media = src.get('media_type', 'image/jpeg')
                converted.append({
                    'type': 'image_url',
                    'image_url': {'url': f'data:{media};base64,{b64}'},
                })
            elif isinstance(block, dict) and block.get('type') == 'tool_result':
                continue  # OpenAI uses separate tool role messages
            else:
                converted.append(block)
        return {'role': 'user', 'content': converted} if converted else msg

    def _trim_window(self) -> None:
        """Remove old turns to keep within max_turns."""
        while len(self._turn_boundaries) > self.max_turns:
            # Remove messages from the oldest turn
            oldest_start = self._turn_boundaries[0]
            if len(self._turn_boundaries) > 1:
                next_start = self._turn_boundaries[1]
            else:
                next_start = len(self._messages)

            # Delete messages in the oldest turn
            del self._messages[oldest_start:next_start]

            # Adjust all boundaries
            removed = next_start - oldest_start
            self._turn_boundaries.pop(0)
            self._turn_boundaries = [
                b - removed for b in self._turn_boundaries
            ]

    @staticmethod
    def _build_user_message(
        text: str,
        images_b64: list[str] | None = None,
    ) -> dict:
        """Build a user message dict with optional images.

        Uses the Claude content-block format which is also compatible
        with OpenAI's content array format after minor conversion.
        """
        if not images_b64:
            return {'role': 'user', 'content': text}

        content = []
        for img in images_b64:
            content.append({
                'type': 'image',
                'source': {
                    'type': 'base64',
                    'media_type': 'image/jpeg',
                    'data': img,
                },
            })
        content.append({'type': 'text', 'text': text})

        return {'role': 'user', 'content': content}


# ======================================================================
# Module-level context assembly functions
# ======================================================================

def build_sensor_context(
    lidar_summary: str = '',
    depth_summary: str = '',
    odom: dict | None = None,
    imu: dict | None = None,
    battery: float | None = None,
    emergency_stop: bool = False,
) -> str:
    """Format sensor readings into a compact text block for the agent prompt.

    Accepts the pre-formatted summary strings from the explorer node
    (unlike the old static method which took raw dicts).
    """
    import math
    lines = []

    if lidar_summary:
        lines.append(lidar_summary)
    if depth_summary:
        lines.append(depth_summary)

    if odom:
        heading_deg = math.degrees(odom.get('theta', 0))
        lines.append(
            f'POSITION: ({odom.get("x", 0):.2f}, {odom.get("y", 0):.2f})m, '
            f'heading={heading_deg:.0f}deg'
        )

    if imu:
        ori = imu.get('orientation', {})
        lines.append(
            f'IMU: roll={math.degrees(ori.get("roll", 0)):.1f}deg, '
            f'pitch={math.degrees(ori.get("pitch", 0)):.1f}deg'
        )

    if battery is not None:
        lines.append(f'BATTERY: {battery:.1f}V')

    if emergency_stop:
        lines.append('** EMERGENCY STOP ACTIVE — obstacle too close! **')

    return '\n'.join(lines)


def build_identity_context(
    consciousness,
    world_knowledge,
    odom: dict | None = None,
) -> str:
    """Build identity + knowledge context block for the agent prompt."""
    parts = []

    if consciousness:
        identity = consciousness.get_identity_context()
        if identity:
            parts.append(identity)

    if world_knowledge:
        x = odom.get('x', 0) if odom else 0
        y = odom.get('y', 0) if odom else 0
        theta = odom.get('theta', 0) if odom else 0
        knowledge = world_knowledge.get_prompt_context(x=x, y=y, theta=theta)
        if knowledge:
            parts.append(knowledge)

    return '\n'.join(parts)
