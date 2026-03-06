#!/usr/bin/env python3
# encoding: utf-8
"""
Provider-agnostic LLM interface for the autonomous explorer.

Supports Claude (Anthropic) and OpenAI with vision capabilities.
Both providers receive camera images + sensor context and return
JSON action commands using the same schema.

Usage:
    provider = create_provider("claude", api_key="sk-...")
    result = provider.analyze_scene(image_b64, system_prompt, user_prompt)
    # result['action'], result['speed'], ...    <- parsed action
    # result['_meta']['tokens_input'], ...      <- logging metadata
"""
import json
import re
import time
from abc import ABC, abstractmethod


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
                max_tokens=512,
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
                max_tokens=512,
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
