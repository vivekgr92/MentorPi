#!/usr/bin/env python3
# encoding: utf-8
"""
Provider-agnostic LLM interface for the autonomous explorer.

Supports two modes:
  1. analyze_scene() — legacy single-action JSON mode (sense-think-act loop)
  2. agent_turn() — ROSA-style tool-calling mode (multi-step reasoning)

Both Claude and OpenAI providers implement native tool-calling via their
respective APIs (Claude tool_use, OpenAI function calling), eliminating
brittle JSON parsing. The LLM selects from predefined tool functions
following the ROSA (NASA JPL) pattern.

Usage (legacy):
    provider = create_provider("claude", api_key="sk-...")
    result = provider.analyze_scene(image_b64, system_prompt, user_prompt)

Usage (agent mode):
    provider = create_provider("openai", api_key="sk-...")
    response = provider.agent_turn(
        system_prompt="You are Jeeves...",
        messages=[...],           # conversation history
        tools=[...],              # tool definitions (from ToolRegistry)
    )
    for tc in response.tool_calls:
        result = registry.execute(tc.tool_name, tc.arguments)
"""
import json
import re
import time
from abc import ABC, abstractmethod

from autonomous_explorer.conversation_manager import AgentResponse, ToolCall


class LLMProvider(ABC):
    """Abstract base class for LLM vision providers."""

    provider_name: str = 'unknown'

    @abstractmethod
    def analyze_scene(
        self,
        image_base64: 'str | list[str]',
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        """Send image(s) + prompt to the LLM and return a parsed JSON dict.

        Args:
            image_base64: Single JPEG base64 string, or list of base64 strings
                (e.g. [camera_image, map_image] for hybrid mode).
            system_prompt: The robot brain system prompt.
            user_prompt: Sensor context and instructions for this frame.

        Returns:
            Parsed JSON dict with keys: action, speed, duration, speech,
            reasoning.  Also includes a '_meta' key with:
              raw_response, tokens_input, tokens_output, response_time_ms,
              provider, model
            Returns a safe fallback on any error.
        """

    @staticmethod
    def _parse_response(text: str) -> dict:
        """Extract JSON from LLM response text, handling markdown fences.

        Sets '_parse_failed' key in the result dict when parsing falls back
        to the default stop action. This is critical for benchmarking —
        it distinguishes "LLM chose to stop" from "LLM returned garbage".
        """
        text = text.strip()
        # Strip markdown code fences if present
        fence_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if fence_match:
            text = fence_match.group(1).strip()
        try:
            result = json.loads(text)
            result['_parse_failed'] = False
            return result
        except json.JSONDecodeError:
            # Try to find a JSON object in the text
            brace_match = re.search(r'\{[\s\S]*\}', text)
            if brace_match:
                try:
                    result = json.loads(brace_match.group())
                    result['_parse_failed'] = False
                    return result
                except json.JSONDecodeError:
                    pass
        return {
            'action': 'stop',
            'speed': 0.0,
            'duration': 0.0,
            'speech': 'I had trouble thinking. Stopping to be safe.',
            'reasoning': f'Failed to parse LLM response: {text[:200]}',
            '_parse_failed': True,
        }

    def _safe_fallback(self, error_msg: str) -> dict:
        """Return a safe stop command when the LLM call fails."""
        return {
            'action': 'stop',
            'speed': 0.0,
            'duration': 0.0,
            'speech': 'My brain had a hiccup. Stopping to be safe.',
            'reasoning': f'LLM error: {error_msg}',
            '_meta': {
                'raw_response': '',
                'tokens_input': 0,
                'tokens_output': 0,
                'response_time_ms': 0,
                'provider': self.provider_name,
                'model': getattr(self, 'model', ''),
                'error': error_msg,
            },
        }

    def agent_turn(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 1024,
    ) -> AgentResponse:
        """Execute one agent turn with native tool-calling.

        This is the ROSA-style interface. The LLM receives conversation
        history (with images) and tool definitions, and returns tool
        invocations rather than free-form JSON.

        Args:
            system_prompt: System instructions for the agent.
            messages: Conversation history in provider-specific format
                (use ConversationManager.get_messages()).
            tools: Tool definitions in provider-specific format
                (use ToolRegistry.to_claude_tools() or .to_openai_tools()).
            max_tokens: Maximum response tokens.

        Returns:
            AgentResponse with tool_calls and/or text.
        """
        # Default implementation falls back to analyze_scene for providers
        # that don't override this method.
        return self._agent_turn_fallback(system_prompt, messages, tools)

    def _agent_turn_fallback(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
    ) -> AgentResponse:
        """Fallback: extract text from latest user message and use analyze_scene."""
        user_text = ''
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                if isinstance(content, str):
                    user_text = content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get('type') == 'text':
                            user_text = block['text']
                            break
                break
        result = self.analyze_scene('', system_prompt, user_text)
        meta = result.pop('_meta', {})
        # Convert the legacy action into a move_direct tool call
        action = result.get('action', 'stop')
        return AgentResponse(
            tool_calls=[ToolCall(
                tool_name='move_direct',
                arguments={
                    'action': action,
                    'speed': result.get('speed', 0.0),
                    'duration': result.get('duration', 0.0),
                },
                call_id='fallback_0',
            )],
            text=result.get('speech'),
            raw_response=meta.get('raw_response', ''),
            tokens_input=meta.get('tokens_input', 0),
            tokens_output=meta.get('tokens_output', 0),
            response_time_ms=meta.get('response_time_ms', 0),
            stop_reason='tool_use',
            provider=self.provider_name,
            model=getattr(self, 'model', ''),
        )

    def _agent_error_response(self, error_msg: str) -> AgentResponse:
        """Return a safe AgentResponse when the LLM call fails."""
        return AgentResponse(
            tool_calls=[ToolCall(
                tool_name='move_direct',
                arguments={'action': 'stop', 'speed': 0.0, 'duration': 0.0},
                call_id='error_0',
            )],
            text=f'LLM error: {error_msg}',
            raw_response=error_msg,
            tokens_input=0,
            tokens_output=0,
            response_time_ms=0,
            stop_reason='error',
            provider=self.provider_name,
            model=getattr(self, 'model', ''),
        )

    @staticmethod
    def _attach_meta(
        parsed: dict,
        *,
        raw_response: str,
        tokens_input: int,
        tokens_output: int,
        response_time_ms: int,
        provider: str,
        model: str,
    ) -> dict:
        """Attach logging metadata to the parsed result dict."""
        parsed['_meta'] = {
            'raw_response': raw_response,
            'tokens_input': tokens_input,
            'tokens_output': tokens_output,
            'response_time_ms': response_time_ms,
            'provider': provider,
            'model': model,
        }
        return parsed


class ClaudeProvider(LLMProvider):
    """Claude (Anthropic) vision provider."""

    provider_name = 'claude'

    def __init__(self, api_key: str, model: str = 'claude-sonnet-4-20250514'):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key, timeout=20.0)
        self.model = model

    def analyze_scene(
        self,
        image_base64: 'str | list[str]',
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        t_start = time.monotonic()
        # Normalize to list
        images = image_base64 if isinstance(image_base64, list) else [image_base64]
        try:
            content = []
            for img in images:
                content.append({
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': 'image/jpeg',
                        'data': img,
                    },
                })
            content.append({
                'type': 'text',
                'text': user_prompt,
            })

            response = self.client.messages.create(
                model=self.model,
                max_tokens=600,
                system=system_prompt,
                messages=[{'role': 'user', 'content': content}],
            )
            elapsed_ms = int((time.monotonic() - t_start) * 1000)
            raw_text = response.content[0].text
            tokens_in = getattr(response.usage, 'input_tokens', 0)
            tokens_out = getattr(response.usage, 'output_tokens', 0)
            parsed = self._parse_response(raw_text)
            return self._attach_meta(
                parsed,
                raw_response=raw_text,
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                response_time_ms=elapsed_ms,
                provider=self.provider_name,
                model=self.model,
            )
        except Exception as e:
            return self._safe_fallback(str(e))


    def agent_turn(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 1024,
    ) -> AgentResponse:
        """Claude native tool-calling via the Messages API."""
        t_start = time.monotonic()
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages,
                tools=tools,
                tool_choice={'type': 'auto'},
            )
            elapsed_ms = int((time.monotonic() - t_start) * 1000)

            tokens_in = getattr(response.usage, 'input_tokens', 0)
            tokens_out = getattr(response.usage, 'output_tokens', 0)

            # Parse content blocks: text and tool_use
            tool_calls = []
            text_parts = []
            for block in response.content:
                if block.type == 'text':
                    text_parts.append(block.text)
                elif block.type == 'tool_use':
                    tool_calls.append(ToolCall(
                        tool_name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                        call_id=block.id,
                    ))

            stop_reason = response.stop_reason  # 'tool_use' or 'end_turn'

            return AgentResponse(
                tool_calls=tool_calls,
                text='\n'.join(text_parts) if text_parts else None,
                raw_response=str(response.content),
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                response_time_ms=elapsed_ms,
                stop_reason=stop_reason or 'end_turn',
                provider=self.provider_name,
                model=self.model,
            )
        except Exception as e:
            return self._agent_error_response(str(e))


class OpenAIProvider(LLMProvider):
    """OpenAI GPT-4o vision provider."""

    provider_name = 'openai'

    def __init__(self, api_key: str, model: str = 'gpt-4o'):
        import openai
        self.client = openai.OpenAI(api_key=api_key, timeout=20.0)
        self.model = model

    def analyze_scene(
        self,
        image_base64: 'str | list[str]',
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        t_start = time.monotonic()
        # Normalize to list
        images = image_base64 if isinstance(image_base64, list) else [image_base64]
        try:
            content = []
            for img in images:
                content.append({
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/jpeg;base64,{img}',
                    },
                })
            content.append({
                'type': 'text',
                'text': user_prompt,
            })

            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=600,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': content},
                ],
                response_format={'type': 'json_object'},
            )
            elapsed_ms = int((time.monotonic() - t_start) * 1000)
            raw_text = response.choices[0].message.content
            usage = response.usage
            tokens_in = getattr(usage, 'prompt_tokens', 0) if usage else 0
            tokens_out = getattr(usage, 'completion_tokens', 0) if usage else 0
            parsed = self._parse_response(raw_text)
            return self._attach_meta(
                parsed,
                raw_response=raw_text,
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                response_time_ms=elapsed_ms,
                provider=self.provider_name,
                model=self.model,
            )
        except Exception as e:
            return self._safe_fallback(str(e))


    def agent_turn(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 1024,
    ) -> AgentResponse:
        """OpenAI native function-calling via the Chat Completions API."""
        t_start = time.monotonic()
        try:
            # Build the messages list with system prompt prepended
            api_messages = [{'role': 'system', 'content': system_prompt}]

            # Convert image blocks from Claude format to OpenAI format
            for msg in messages:
                if msg['role'] == 'user' and isinstance(msg.get('content'), list):
                    converted = []
                    for block in msg['content']:
                        if isinstance(block, dict):
                            if block.get('type') == 'image':
                                # Claude image → OpenAI image_url
                                src = block.get('source', {})
                                b64 = src.get('data', '')
                                media = src.get('media_type', 'image/jpeg')
                                converted.append({
                                    'type': 'image_url',
                                    'image_url': {
                                        'url': f'data:{media};base64,{b64}',
                                    },
                                })
                            elif block.get('type') == 'tool_result':
                                # Skip tool_results in user messages (OpenAI
                                # uses separate tool role messages)
                                continue
                            else:
                                converted.append(block)
                        else:
                            converted.append(block)
                    if converted:
                        api_messages.append({
                            'role': 'user', 'content': converted,
                        })
                else:
                    api_messages.append(msg)

            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=api_messages,
                tools=tools,
                tool_choice='auto',
            )
            elapsed_ms = int((time.monotonic() - t_start) * 1000)

            usage = response.usage
            tokens_in = getattr(usage, 'prompt_tokens', 0) if usage else 0
            tokens_out = getattr(usage, 'completion_tokens', 0) if usage else 0

            choice = response.choices[0]
            message = choice.message

            # Parse tool calls
            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    tool_calls.append(ToolCall(
                        tool_name=tc.function.name,
                        arguments=args,
                        call_id=tc.id,
                    ))

            # Determine stop reason
            finish = choice.finish_reason  # 'tool_calls', 'stop', 'length'
            if finish == 'tool_calls':
                stop_reason = 'tool_use'
            elif finish == 'length':
                stop_reason = 'max_tokens'
            else:
                stop_reason = 'end_turn'

            return AgentResponse(
                tool_calls=tool_calls,
                text=message.content,
                raw_response=str(message),
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                response_time_ms=elapsed_ms,
                stop_reason=stop_reason,
                provider=self.provider_name,
                model=self.model,
            )
        except Exception as e:
            return self._agent_error_response(str(e))


class DryRunProvider(LLMProvider):
    """Fake provider for testing without API keys.

    Cycles through a set of actions so the full pipeline
    (sensors, logging, safety, execution) can be verified.
    """

    provider_name = 'dryrun'

    # Actions to cycle through for realistic-looking exploration
    _ACTIONS = [
        {'action': 'forward', 'speed': 0.3, 'duration': 1.0,
         'speech': 'Dry run: moving forward to test motors.',
         'reasoning': 'Dry-run mode — cycling through actions.'},
        {'action': 'turn_left', 'speed': 0.3, 'duration': 0.8,
         'speech': 'Dry run: turning left.',
         'reasoning': 'Dry-run mode — testing turn left.'},
        {'action': 'forward', 'speed': 0.4, 'duration': 1.5,
         'speech': 'Dry run: driving forward again.',
         'reasoning': 'Dry-run mode — forward movement test.'},
        {'action': 'look_around', 'speed': 0.0, 'duration': 0.0,
         'speech': 'Dry run: scanning surroundings.',
         'reasoning': 'Dry-run mode — testing camera servos.'},
        {'action': 'turn_right', 'speed': 0.3, 'duration': 0.8,
         'speech': 'Dry run: turning right.',
         'reasoning': 'Dry-run mode — testing turn right.'},
        {'action': 'stop', 'speed': 0.0, 'duration': 0.0,
         'speech': 'Dry run: stopping briefly.',
         'reasoning': 'Dry-run mode — testing stop command.'},
    ]

    def __init__(self):
        self.model = 'dry-run'
        self._cycle = 0

    # Tool call sequences for agent mode dry-run
    _AGENT_SEQUENCES = [
        [
            ToolCall('check_surroundings', {}, 'dry_0'),
            ToolCall('speak', {'text': 'Dry run: checking my surroundings.'}, 'dry_1'),
        ],
        [
            ToolCall('move_direct', {'action': 'forward', 'speed': 0.3, 'duration': 1.0}, 'dry_2'),
            ToolCall('speak', {'text': 'Dry run: moving forward to explore.'}, 'dry_3'),
        ],
        [
            ToolCall('look_around', {'speech': 'Dry run: scanning the area.'}, 'dry_4'),
        ],
        [
            ToolCall('move_direct', {'action': 'turn_left', 'speed': 0.3, 'duration': 0.8}, 'dry_5'),
        ],
        [
            ToolCall('explore_frontier', {'preference': 'nearest', 'speech': 'Dry run: exploring frontier.'}, 'dry_6'),
        ],
        [
            ToolCall('move_direct', {'action': 'stop', 'speed': 0.0, 'duration': 0.0}, 'dry_7'),
            ToolCall('speak', {'text': 'Dry run: pausing briefly.'}, 'dry_8'),
        ],
    ]

    def analyze_scene(
        self,
        image_base64: 'str | list[str]',
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        t_start = time.monotonic()
        # Simulate some LLM thinking time
        time.sleep(0.1)
        elapsed_ms = int((time.monotonic() - t_start) * 1000)

        action = self._ACTIONS[self._cycle % len(self._ACTIONS)].copy()
        self._cycle += 1

        return self._attach_meta(
            action,
            raw_response=json.dumps(action),
            tokens_input=0,
            tokens_output=0,
            response_time_ms=elapsed_ms,
            provider=self.provider_name,
            model=self.model,
        )

    def agent_turn(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 1024,
    ) -> AgentResponse:
        """Dry-run agent turn — cycles through predefined tool call sequences."""
        time.sleep(0.1)  # Simulate thinking
        seq = self._AGENT_SEQUENCES[self._cycle % len(self._AGENT_SEQUENCES)]
        self._cycle += 1
        return AgentResponse(
            tool_calls=list(seq),
            text=None,
            raw_response=json.dumps([
                {'tool': tc.tool_name, 'args': tc.arguments} for tc in seq
            ]),
            tokens_input=0,
            tokens_output=0,
            response_time_ms=100,
            stop_reason='tool_use',
            provider=self.provider_name,
            model=self.model,
        )


def create_provider(
    provider_name: str,
    api_key: str,
    model: str | None = None,
) -> LLMProvider:
    """Factory function to create an LLM provider.

    Args:
        provider_name: "claude" or "openai"
        api_key: API key for the chosen provider.
        model: Optional model override.

    Returns:
        An LLMProvider instance.

    Raises:
        ValueError: If provider_name is not recognized.
    """
    name = provider_name.lower().strip()

    if name in ('dryrun', 'dry-run', 'dry_run'):
        return DryRunProvider()
    elif name == 'claude':
        return ClaudeProvider(
            api_key=api_key,
            model=model or 'claude-sonnet-4-20250514',
        )
    elif name == 'openai':
        return OpenAIProvider(
            api_key=api_key,
            model=model or 'gpt-4o',
        )
    else:
        raise ValueError(
            f"Unknown LLM provider '{provider_name}'. "
            f"Supported: 'claude', 'openai', 'dryrun'"
        )
