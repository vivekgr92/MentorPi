#!/usr/bin/env python3
# encoding: utf-8
"""
Agent debug logger for step-by-step tracing of the agent's behavior.

Logs to both the ROS2 logger (with [AGENT] prefix) and publishes to the
/semantic_map/agent_status topic for Foxglove visualization.

All log lines include wall-clock timestamps in HH:MM:SS format.
"""
import time
from collections import defaultdict
from typing import Callable


# Maximum length for argument values and result summaries in log output.
_MAX_ARG_VALUE_LEN = 40
_MAX_SUMMARY_LEN = 60


def _ts() -> str:
    """Return wall-clock timestamp as HH:MM:SS."""
    return time.strftime('%H:%M:%S')


def _truncate(s: str, maxlen: int) -> str:
    """Truncate a string, adding ellipsis if needed."""
    if len(s) <= maxlen:
        return s
    return s[:maxlen - 3] + '...'


def _format_args(arguments: dict) -> str:
    """Compact argument summary for log output.

    Keeps each value short and the total line readable.
    """
    if not arguments:
        return ''
    parts = []
    for k, v in arguments.items():
        v_str = repr(v) if not isinstance(v, str) else f"'{_truncate(v, _MAX_ARG_VALUE_LEN)}'"
        parts.append(f'{k}={v_str}')
    return ', '.join(parts)[:120]


def _format_result_summary(summary: str) -> str:
    """Truncate a result summary for log output."""
    return _truncate(summary, _MAX_SUMMARY_LEN)


class AgentLogger:
    """Synchronous agent debug logger.

    Logs to a ROS2 logger (via a logging.Logger-like object) with [AGENT]
    prefix, and publishes the same text to the agent_status topic via a
    callback.

    Parameters
    ----------
    logger
        A ROS2 logger (from ``node.get_logger()``).
    publish_fn
        Callable that accepts a single string argument and publishes it
        to ``/semantic_map/agent_status``. Typically
        ``node._publish_agent_status``.
    """

    def __init__(self, logger, publish_fn: Callable[[str], None]):
        self._logger = logger
        self._publish = publish_fn
        # Per-tool cumulative execution time (tool_name -> total_ms)
        self._tool_times: dict[str, int] = defaultdict(int)
        # Per-tool call counts
        self._tool_counts: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self, tag: str, detail: str) -> None:
        """Log a formatted line to both ROS2 logger and agent_status topic."""
        line = f'[AGENT] {_ts()} {tag}: {detail}'
        self._logger.info(line)
        try:
            self._publish(line)
        except Exception:
            pass  # Node may be shutting down

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def voice_received(self, transcript: str) -> None:
        """Called when a voice command arrives from STT."""
        self._emit('VOICE', f'"{_truncate(transcript, 80)}"')

    def turn_start(self, cycle_num: int) -> None:
        """Called at the start of each agent turn."""
        self._emit('TURN_START', f'cycle #{cycle_num}')

    def llm_request(
        self,
        provider: str,
        num_tools: int,
        num_messages: int,
        has_image: bool,
        voice_instruction: str | None = None,
    ) -> None:
        """Called before sending a request to the LLM."""
        img = 'yes' if has_image else 'no'
        detail = f'{provider}, {num_tools} tools, {num_messages} messages, image={img}'
        if voice_instruction:
            detail += f', voice="{_truncate(voice_instruction, 40)}"'
        self._emit('LLM_REQUEST', detail)

    def llm_response(
        self,
        num_tool_calls: int,
        stop_reason: str,
        response_ms: int,
        tokens_in: int,
        tokens_out: int,
    ) -> None:
        """Called after receiving a response from the LLM."""
        self._emit(
            'LLM_RESPONSE',
            f'{num_tool_calls} tool calls, stop={stop_reason}, '
            f'{response_ms}ms, {tokens_in}+{tokens_out} tokens',
        )

    def tool_start(
        self,
        tool_name: str,
        arguments: dict,
        timeout_s: float,
    ) -> None:
        """Called before executing a tool."""
        args_str = _format_args(arguments)
        call = f'{tool_name}({args_str})' if args_str else f'{tool_name}()'
        self._emit('TOOL_START', f'{call} timeout={timeout_s:.0f}s')

    def tool_result(
        self,
        tool_name: str,
        success: bool,
        duration_ms: int,
        summary: str,
    ) -> None:
        """Called after a tool executes successfully (or returns a result)."""
        status = 'success' if success else 'failed'
        self._emit(
            'TOOL_RESULT',
            f'{tool_name} -- {status}, {duration_ms}ms, '
            f'"{_format_result_summary(summary)}"',
        )
        self._tool_times[tool_name] += duration_ms
        self._tool_counts[tool_name] += 1

    def tool_error(
        self,
        tool_name: str,
        error: str,
        duration_ms: int,
    ) -> None:
        """Called when a tool raises an exception."""
        self._emit(
            'TOOL_ERROR',
            f'{tool_name} -- {duration_ms}ms, {_truncate(error, 80)}',
        )
        self._tool_times[tool_name] += duration_ms
        self._tool_counts[tool_name] += 1

    def turn_complete(
        self,
        num_tools_executed: int,
        total_duration_s: float,
        total_cost: float,
    ) -> None:
        """Called at the end of an agent turn."""
        self._emit(
            'TURN_COMPLETE',
            f'{num_tools_executed} tools, {total_duration_s:.1f}s total, '
            f'${total_cost:.4f}',
        )

    def status(self, message: str) -> None:
        """Log a generic status message."""
        self._emit('STATUS', message)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_tool_stats(self) -> dict[str, dict]:
        """Return per-tool cumulative stats.

        Returns a dict mapping tool_name to
        ``{'total_ms': int, 'calls': int, 'avg_ms': float}``.
        """
        stats = {}
        for name in self._tool_times:
            total = self._tool_times[name]
            count = self._tool_counts[name]
            stats[name] = {
                'total_ms': total,
                'calls': count,
                'avg_ms': total / count if count else 0.0,
            }
        return stats
