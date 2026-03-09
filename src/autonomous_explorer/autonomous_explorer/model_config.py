#!/usr/bin/env python3
# encoding: utf-8
"""
Model configuration manager for the autonomous explorer.

Loads provider config from model_config.yaml, supports named profiles,
and handles automatic fallback on errors (429, timeout, connection refused).
"""
import os
import yaml


class ModelConfigManager:
    """Loads model_config.yaml and resolves provider settings per service.

    Services: llm, tts, stt, vlm
    Each service has a primary provider, a fallback, and provider-specific config.
    Named profiles override the primary provider for each service.
    """

    # Errors that trigger automatic fallback
    FALLBACK_ERRORS = (
        'insufficient_quota', '429', 'rate_limit',
        'timeout', 'timed out',
        'connection refused', 'connection error', 'connect',
    )

    def __init__(self, config_path: str = '', profile: str = '', logger=None):
        self._logger = logger
        self._config: dict = {}
        self._profile = profile
        self._active: dict[str, str] = {}  # service -> active provider name
        self._failed: dict[str, set[str]] = {}  # service -> set of failed providers

        if config_path and os.path.exists(config_path):
            self._load(config_path)
        else:
            self._log_warn(f"Model config not found: {config_path}, using defaults")
            self._config = self._defaults()

        self._apply_profile(profile)

    def _load(self, path: str):
        with open(path) as f:
            self._config = yaml.safe_load(f) or {}
        self._log_info(f"Loaded model config: {path}")

    @staticmethod
    def _defaults() -> dict:
        return {
            'llm': {
                'primary': os.environ.get('LLM_PROVIDER', 'openai'),
                'fallback': 'dryrun',
                'providers': {
                    'openai': {'model': 'gpt-4o', 'api_key_env': 'OPENAI_API_KEY'},
                    'claude': {'model': 'claude-sonnet-4-20250514', 'api_key_env': 'ANTHROPIC_API_KEY'},
                },
            },
            'tts': {
                'primary': 'openai',
                'fallback': 'gtts',
                'providers': {
                    'openai': {'model': 'tts-1', 'voice': 'onyx', 'api_key_env': 'OPENAI_API_KEY'},
                    'gtts': {},
                    'espeak': {},
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
                'primary': os.environ.get('LLM_PROVIDER', 'openai'),
                'fallback': 'dryrun',
                'providers': {
                    'openai': {'model': 'gpt-4o', 'api_key_env': 'OPENAI_API_KEY'},
                    'claude': {'model': 'claude-sonnet-4-20250514', 'api_key_env': 'ANTHROPIC_API_KEY'},
                },
            },
            'profiles': {},
        }

    def _apply_profile(self, profile: str):
        """Apply a named profile, overriding primary providers."""
        profiles = self._config.get('profiles', {})
        if profile and profile in profiles:
            overrides = profiles[profile]
            for service, provider in overrides.items():
                svc = self._config.get(service, {})
                svc['primary'] = provider
                self._config[service] = svc
            self._log_info(f"Applied model profile: {profile} -> {overrides}")
        elif profile:
            self._log_warn(f"Unknown profile '{profile}', available: {list(profiles.keys())}")

        # Initialize active providers
        for service in ('llm', 'tts', 'stt', 'vlm'):
            svc = self._config.get(service, {})
            self._active[service] = svc.get('primary', 'dryrun')
            self._failed[service] = set()

    @property
    def profile(self) -> str:
        return self._profile

    def get_provider_name(self, service: str) -> str:
        """Get the currently active provider name for a service."""
        return self._active.get(service, 'dryrun')

    def get_provider_config(self, service: str, provider: str = '') -> dict:
        """Get provider-specific config dict for a service.

        Returns keys like model, api_key_env, base_url, voice, etc.
        """
        if not provider:
            provider = self.get_provider_name(service)
        svc = self._config.get(service, {})
        providers = svc.get('providers', {})
        return dict(providers.get(provider, {}))

    def get_api_key(self, service: str, provider: str = '') -> str:
        """Resolve the API key for a provider from env var or literal."""
        cfg = self.get_provider_config(service, provider)
        # Check for literal key first
        literal = cfg.get('api_key', '')
        if literal:
            return literal
        # Then check env var
        env_var = cfg.get('api_key_env', '')
        if env_var:
            return os.environ.get(env_var, '')
        return ''

    def get_model(self, service: str, provider: str = '') -> str:
        """Get the model name for a provider."""
        cfg = self.get_provider_config(service, provider)
        return cfg.get('model', '')

    def get_base_url(self, service: str, provider: str = '') -> str:
        """Get the base_url for a provider (for local/LM Studio)."""
        cfg = self.get_provider_config(service, provider)
        return cfg.get('base_url', '')

    def should_fallback(self, service: str, error_msg: str) -> bool:
        """Check if an error should trigger a fallback to the next provider.

        Returns True if the error matches known fallback patterns and
        a fallback provider is available.
        """
        error_lower = str(error_msg).lower()
        if not any(e in error_lower for e in self.FALLBACK_ERRORS):
            return False

        svc = self._config.get(service, {})
        fallback = svc.get('fallback', '')
        current = self._active.get(service, '')
        return bool(fallback and fallback != current and fallback not in self._failed.get(service, set()))

    def do_fallback(self, service: str, error_msg: str) -> str | None:
        """Switch to the fallback provider for a service.

        Returns the new provider name, or None if no fallback available.
        """
        current = self._active.get(service, '')
        svc = self._config.get(service, {})
        fallback = svc.get('fallback', '')

        if not fallback or fallback == current:
            return None

        failed = self._failed.setdefault(service, set())
        if fallback in failed:
            return None

        failed.add(current)
        self._active[service] = fallback
        self._log_warn(
            f"[{service}] Falling back: {current} -> {fallback} "
            f"(reason: {str(error_msg)[:80]})"
        )
        return fallback

    def reset_fallbacks(self, service: str = ''):
        """Reset failed providers, re-enabling primary. Call on success or periodically."""
        if service:
            self._failed[service] = set()
            svc = self._config.get(service, {})
            self._active[service] = svc.get('primary', self._active.get(service, 'dryrun'))
        else:
            for s in ('llm', 'tts', 'stt', 'vlm'):
                self.reset_fallbacks(s)

    def summary(self) -> dict:
        """Return a summary of active providers for logging/status."""
        return {
            service: {
                'active': self._active.get(service, '?'),
                'model': self.get_model(service),
                'fallback': self._config.get(service, {}).get('fallback', ''),
                'failed': list(self._failed.get(service, [])),
            }
            for service in ('llm', 'tts', 'stt', 'vlm')
        }

    def _log_info(self, msg: str):
        if self._logger:
            self._logger.info(msg)

    def _log_warn(self, msg: str):
        if self._logger:
            self._logger.warning(msg)
