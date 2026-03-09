"""Tests for autonomous_explorer.model_config module."""
import os
import tempfile

import pytest
import yaml

from autonomous_explorer.model_config import ModelConfigManager


SAMPLE_CONFIG = {
    'llm': {
        'primary': 'openai',
        'fallback': 'local',
        'providers': {
            'openai': {'model': 'gpt-4o', 'api_key_env': 'OPENAI_API_KEY'},
            'claude': {'model': 'claude-sonnet-4-20250514', 'api_key_env': 'ANTHROPIC_API_KEY'},
            'local': {'base_url': 'http://10.0.0.10:1234/v1', 'model': 'llama-3.2-8b', 'api_key': 'lm-studio'},
        },
    },
    'tts': {
        'primary': 'openai',
        'fallback': 'gtts',
        'providers': {
            'openai': {'model': 'tts-1', 'voice': 'onyx', 'api_key_env': 'OPENAI_API_KEY'},
            'gtts': {},
        },
    },
    'stt': {
        'primary': 'openai',
        'fallback': 'google',
        'providers': {
            'openai': {'model': 'whisper-1', 'api_key_env': 'OPENAI_API_KEY'},
            'google': {},
        },
    },
    'vlm': {
        'primary': 'openai',
        'fallback': 'claude',
        'providers': {
            'openai': {'model': 'gpt-4o', 'api_key_env': 'OPENAI_API_KEY'},
            'claude': {'model': 'claude-sonnet-4-20250514', 'api_key_env': 'ANTHROPIC_API_KEY'},
        },
    },
    'profiles': {
        'local': {'llm': 'local', 'tts': 'gtts', 'stt': 'google', 'vlm': 'local'},
        'cloud': {'llm': 'openai', 'tts': 'openai', 'stt': 'openai', 'vlm': 'openai'},
        'dryrun': {'llm': 'dryrun', 'tts': 'gtts', 'stt': 'google', 'vlm': 'dryrun'},
    },
}


def _write_config(config: dict) -> str:
    """Write config to a temp YAML file and return the path."""
    fd, path = tempfile.mkstemp(suffix='.yaml')
    with os.fdopen(fd, 'w') as f:
        yaml.dump(config, f)
    return path


class TestModelConfigInit:
    def test_loads_yaml_file(self):
        path = _write_config(SAMPLE_CONFIG)
        try:
            mc = ModelConfigManager(config_path=path)
            assert mc.get_provider_name('llm') == 'openai'
            assert mc.get_provider_name('tts') == 'openai'
        finally:
            os.unlink(path)

    def test_missing_file_uses_defaults(self):
        mc = ModelConfigManager(config_path='/tmp/nonexistent_model_config.yaml')
        assert mc.get_provider_name('llm') in ('openai', 'claude', 'dryrun')

    def test_empty_path_uses_defaults(self):
        mc = ModelConfigManager(config_path='')
        assert mc.get_provider_name('tts') in ('openai', 'gtts')


class TestProfiles:
    def test_local_profile(self):
        path = _write_config(SAMPLE_CONFIG)
        try:
            mc = ModelConfigManager(config_path=path, profile='local')
            assert mc.get_provider_name('llm') == 'local'
            assert mc.get_provider_name('tts') == 'gtts'
            assert mc.get_provider_name('stt') == 'google'
            assert mc.get_provider_name('vlm') == 'local'
        finally:
            os.unlink(path)

    def test_cloud_profile(self):
        path = _write_config(SAMPLE_CONFIG)
        try:
            mc = ModelConfigManager(config_path=path, profile='cloud')
            assert mc.get_provider_name('llm') == 'openai'
            assert mc.get_provider_name('tts') == 'openai'
        finally:
            os.unlink(path)

    def test_unknown_profile_warns(self):
        path = _write_config(SAMPLE_CONFIG)
        try:
            mc = ModelConfigManager(config_path=path, profile='nonexistent')
            # Should still work, just use defaults
            assert mc.get_provider_name('llm') == 'openai'
        finally:
            os.unlink(path)


class TestProviderConfig:
    def test_get_model(self):
        path = _write_config(SAMPLE_CONFIG)
        try:
            mc = ModelConfigManager(config_path=path)
            assert mc.get_model('llm') == 'gpt-4o'
            assert mc.get_model('llm', 'claude') == 'claude-sonnet-4-20250514'
        finally:
            os.unlink(path)

    def test_get_base_url(self):
        path = _write_config(SAMPLE_CONFIG)
        try:
            mc = ModelConfigManager(config_path=path, profile='local')
            assert mc.get_base_url('llm') == 'http://10.0.0.10:1234/v1'
        finally:
            os.unlink(path)

    def test_get_api_key_literal(self):
        path = _write_config(SAMPLE_CONFIG)
        try:
            mc = ModelConfigManager(config_path=path, profile='local')
            assert mc.get_api_key('llm') == 'lm-studio'
        finally:
            os.unlink(path)

    def test_get_api_key_from_env(self):
        path = _write_config(SAMPLE_CONFIG)
        try:
            os.environ['OPENAI_API_KEY'] = 'sk-test-123'
            mc = ModelConfigManager(config_path=path)
            assert mc.get_api_key('llm') == 'sk-test-123'
        finally:
            os.environ.pop('OPENAI_API_KEY', None)
            os.unlink(path)

    def test_get_provider_config_dict(self):
        path = _write_config(SAMPLE_CONFIG)
        try:
            mc = ModelConfigManager(config_path=path)
            cfg = mc.get_provider_config('tts')
            assert cfg['model'] == 'tts-1'
            assert cfg['voice'] == 'onyx'
        finally:
            os.unlink(path)


class TestFallback:
    def test_should_fallback_on_429(self):
        path = _write_config(SAMPLE_CONFIG)
        try:
            mc = ModelConfigManager(config_path=path)
            assert mc.should_fallback('llm', 'Error 429 insufficient_quota')
        finally:
            os.unlink(path)

    def test_should_fallback_on_timeout(self):
        path = _write_config(SAMPLE_CONFIG)
        try:
            mc = ModelConfigManager(config_path=path)
            assert mc.should_fallback('llm', 'Connection timed out')
        finally:
            os.unlink(path)

    def test_should_not_fallback_on_normal_error(self):
        path = _write_config(SAMPLE_CONFIG)
        try:
            mc = ModelConfigManager(config_path=path)
            assert not mc.should_fallback('llm', 'Invalid JSON in response')
        finally:
            os.unlink(path)

    def test_do_fallback_switches_provider(self):
        path = _write_config(SAMPLE_CONFIG)
        try:
            mc = ModelConfigManager(config_path=path)
            assert mc.get_provider_name('llm') == 'openai'
            result = mc.do_fallback('llm', '429 quota exceeded')
            assert result == 'local'
            assert mc.get_provider_name('llm') == 'local'
        finally:
            os.unlink(path)

    def test_do_fallback_returns_none_when_exhausted(self):
        path = _write_config(SAMPLE_CONFIG)
        try:
            mc = ModelConfigManager(config_path=path)
            mc.do_fallback('llm', '429')
            # Now on local, no more fallbacks
            result = mc.do_fallback('llm', 'connection refused')
            assert result is None
        finally:
            os.unlink(path)

    def test_reset_fallbacks(self):
        path = _write_config(SAMPLE_CONFIG)
        try:
            mc = ModelConfigManager(config_path=path)
            mc.do_fallback('llm', '429')
            assert mc.get_provider_name('llm') == 'local'
            mc.reset_fallbacks('llm')
            assert mc.get_provider_name('llm') == 'openai'
        finally:
            os.unlink(path)


class TestSummary:
    def test_summary_format(self):
        path = _write_config(SAMPLE_CONFIG)
        try:
            mc = ModelConfigManager(config_path=path)
            s = mc.summary()
            assert set(s.keys()) == {'llm', 'tts', 'stt', 'vlm'}
            assert s['llm']['active'] == 'openai'
            assert s['llm']['fallback'] == 'local'
        finally:
            os.unlink(path)
