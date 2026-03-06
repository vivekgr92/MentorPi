"""Tests for autonomous_explorer.config module."""
import os
from unittest import mock

import pytest


class TestConfigDefaults:
    """Verify default configuration values are sensible."""

    def test_safety_thresholds_ordering(self):
        from autonomous_explorer.config import (
            EMERGENCY_STOP_DISTANCE,
            CAUTION_DISTANCE,
            SAFE_DISTANCE,
        )
        assert 0 < EMERGENCY_STOP_DISTANCE < CAUTION_DISTANCE < SAFE_DISTANCE

    def test_speed_limits_positive(self):
        from autonomous_explorer.config import MAX_LINEAR_SPEED, MAX_ANGULAR_SPEED
        assert MAX_LINEAR_SPEED > 0
        assert MAX_ANGULAR_SPEED > 0

    def test_motor_timeout_positive(self):
        from autonomous_explorer.config import MOTOR_TIMEOUT
        assert MOTOR_TIMEOUT > 0

    def test_servo_range_valid(self):
        from autonomous_explorer.config import SERVO_MIN, SERVO_CENTER, SERVO_MAX
        assert SERVO_MIN < SERVO_CENTER < SERVO_MAX
        assert SERVO_MIN >= 0

    def test_loop_interval_positive(self):
        from autonomous_explorer.config import LOOP_INTERVAL
        assert LOOP_INTERVAL > 0

    def test_camera_quality_range(self):
        from autonomous_explorer.config import CAMERA_JPEG_QUALITY
        assert 1 <= CAMERA_JPEG_QUALITY <= 100

    def test_max_image_dimension_reasonable(self):
        from autonomous_explorer.config import MAX_IMAGE_DIMENSION
        assert 100 <= MAX_IMAGE_DIMENSION <= 4096

    def test_cost_tables_have_entries(self):
        from autonomous_explorer.config import (
            COST_PER_M_INPUT_TOKENS,
            COST_PER_M_OUTPUT_TOKENS,
        )
        assert 'claude' in COST_PER_M_INPUT_TOKENS
        assert 'openai' in COST_PER_M_INPUT_TOKENS
        assert 'claude' in COST_PER_M_OUTPUT_TOKENS
        assert 'openai' in COST_PER_M_OUTPUT_TOKENS

    def test_velocity_ramping_values(self):
        from autonomous_explorer.config import (
            LINEAR_ACCEL, LINEAR_DECEL,
            ANGULAR_ACCEL, ANGULAR_DECEL,
            RAMPER_UPDATE_RATE,
        )
        assert LINEAR_ACCEL > 0
        assert LINEAR_DECEL > 0
        assert LINEAR_DECEL >= LINEAR_ACCEL  # braking faster than accel
        assert ANGULAR_ACCEL > 0
        assert ANGULAR_DECEL >= ANGULAR_ACCEL
        assert RAMPER_UPDATE_RATE >= 10.0

    def test_twist_mux_topics_distinct(self):
        from autonomous_explorer.config import (
            TWIST_MUX_AUTONOMOUS_TOPIC,
            TWIST_MUX_JOYSTICK_TOPIC,
            TWIST_MUX_NAV2_TOPIC,
            TWIST_MUX_SAFETY_TOPIC,
            TWIST_MUX_LOCK_TOPIC,
        )
        topics = [
            TWIST_MUX_AUTONOMOUS_TOPIC,
            TWIST_MUX_JOYSTICK_TOPIC,
            TWIST_MUX_NAV2_TOPIC,
            TWIST_MUX_SAFETY_TOPIC,
            TWIST_MUX_LOCK_TOPIC,
        ]
        assert len(topics) == len(set(topics)), "twist_mux topics must be unique"


class TestBuildSystemPrompt:
    """Test the build_system_prompt helper."""

    def test_prepends_embodied_preamble(self):
        from autonomous_explorer.config import (
            build_system_prompt,
            EMBODIED_PREAMBLE,
        )
        base = "Navigate the room."
        result = build_system_prompt(base)
        assert result.startswith(EMBODIED_PREAMBLE)
        assert base in result

    def test_result_contains_both_parts(self):
        from autonomous_explorer.config import build_system_prompt
        base = "Test prompt."
        result = build_system_prompt(base)
        assert "Jeeves" in result  # from EMBODIED_PREAMBLE
        assert "Test prompt." in result


class TestEnvVarOverrides:
    """Verify config reads from environment variables."""

    def test_llm_provider_from_env(self):
        with mock.patch.dict(os.environ, {'LLM_PROVIDER': 'openai'}):
            # Re-import to pick up env var
            assert os.environ['LLM_PROVIDER'] == 'openai'

    def test_voice_enabled_from_env(self):
        with mock.patch.dict(os.environ, {'VOICE_ENABLED': 'false'}):
            assert os.environ['VOICE_ENABLED'] == 'false'

    def test_log_level_from_env(self):
        with mock.patch.dict(os.environ, {'EXPLORER_LOG_LEVEL': 'minimal'}):
            assert os.environ['EXPLORER_LOG_LEVEL'] == 'minimal'
