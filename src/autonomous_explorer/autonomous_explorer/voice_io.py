#!/usr/bin/env python3
# encoding: utf-8
"""
Voice input/output for the autonomous explorer.

Uses WonderEcho Pro USB mic+speaker via arecord/aplay,
with OpenAI Whisper for STT and OpenAI TTS for speech synthesis.
Falls back to espeak for offline TTS.
"""
import os
import subprocess
import tempfile
import threading
import time


class VoiceIO:
    """Handles voice recording, speech-to-text, and text-to-speech.

    Uses the WonderEcho Pro hardware via standard ALSA commands.
    STT: OpenAI Whisper API
    TTS: OpenAI TTS API with aplay for playback, espeak fallback
    """

    def __init__(
        self,
        openai_api_key: str = '',
        tts_model: str = 'tts-1',
        tts_voice: str = 'onyx',
        stt_model: str = 'whisper-1',
        audio_device: str = 'plughw:1,0',
        sample_rate: int = 16000,
        logger=None,
    ):
        self.openai_api_key = openai_api_key
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        self.stt_model = stt_model
        self.audio_device = audio_device
        self.sample_rate = sample_rate
        self.logger = logger
        self._speaking = False
        self._speak_lock = threading.Lock()

        # Cache the detected audio device so we don't shell out on every call.
        # _find_audio_device() runs arecord -l which is slow on Pi 5.
        self._cached_device: str | None = None

        # Determine which TTS to use
        self._openai_available = bool(openai_api_key)
        if self._openai_available:
            try:
                import openai
                self._openai_client = openai.OpenAI(api_key=openai_api_key)
            except ImportError:
                self._openai_available = False
                self._log_warn("openai package not installed, using espeak")

    def _log_info(self, msg: str):
        if self.logger:
            self.logger.info(msg)

    def _log_warn(self, msg: str):
        if self.logger:
            self.logger.warning(msg)

    def _log_error(self, msg: str):
        if self.logger:
            self.logger.error(msg)

    def _find_audio_device(self) -> str:
        """Return the audio device, auto-detecting on first call.

        Caches the result to avoid running `arecord -l` subprocess on
        every speak/record call -- meaningful on Pi 5 where subprocess
        overhead is noticeable.
        """
        if self._cached_device is not None:
            return self._cached_device

        try:
            result = subprocess.run(
                ['arecord', '-l'],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                if 'USB' in line or 'Wonder' in line.lower():
                    parts = line.split(':')
                    if parts and 'card' in parts[0].lower():
                        card_num = ''.join(
                            c for c in parts[0] if c.isdigit()
                        )
                        if card_num:
                            self._cached_device = f'plughw:{card_num},0'
                            return self._cached_device
        except subprocess.TimeoutExpired:
            self._log_warn("Audio device detection timed out")
        except OSError as e:
            self._log_warn(f"Audio device detection failed: {e}")

        self._cached_device = self.audio_device
        return self._cached_device

    def record(self, duration: int = 5, output_path: str = '') -> str:
        """Record audio from WonderEcho Pro mic.

        Args:
            duration: Recording duration in seconds.
            output_path: Path to save WAV file. Auto-generated if empty.

        Returns:
            Path to the recorded WAV file, or empty string on failure.
        """
        if not output_path:
            output_path = tempfile.mktemp(suffix='.wav', prefix='explorer_rec_')

        device = self._find_audio_device()
        cmd = [
            'arecord',
            '-D', device,
            '-f', 'S16_LE',
            '-r', str(self.sample_rate),
            '-c', '1',
            '-d', str(duration),
            output_path,
        ]
        self._log_info(f"Recording {duration}s from {device}...")
        try:
            subprocess.run(
                cmd, capture_output=True, timeout=duration + 5,
            )
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                return output_path
            self._log_warn("Recording file too small or missing")
        except subprocess.TimeoutExpired:
            self._log_warn("Recording timed out")
        except Exception as e:
            self._log_error(f"Recording failed: {e}")
        return ''

    def speech_to_text(self, audio_path: str) -> str:
        """Transcribe audio file using OpenAI Whisper.

        Args:
            audio_path: Path to WAV file.

        Returns:
            Transcribed text, or empty string on failure.
        """
        if not self._openai_available:
            self._log_warn("STT unavailable: no OpenAI API key")
            return ''
        if not os.path.exists(audio_path):
            return ''

        try:
            with open(audio_path, 'rb') as f:
                response = self._openai_client.audio.transcriptions.create(
                    model=self.stt_model,
                    file=f,
                    language='en',
                )
            text = response.text.strip()
            self._log_info(f"STT result: {text}")
            return text
        except Exception as e:
            self._log_error(f"STT failed: {e}")
            return ''

    def speak(self, text: str, block: bool = False):
        """Speak text out loud via TTS + aplay.

        Args:
            text: Text to speak.
            block: If True, wait for speech to finish.
        """
        if not text:
            return

        if block:
            self._speak_impl(text)
        else:
            thread = threading.Thread(
                target=self._speak_impl, args=(text,), daemon=True,
            )
            thread.start()

    def _speak_impl(self, text: str):
        """Internal TTS implementation."""
        with self._speak_lock:
            self._speaking = True
            try:
                if self._openai_available:
                    self._speak_openai(text)
                else:
                    self._speak_espeak(text)
            finally:
                self._speaking = False

    def _speak_openai(self, text: str):
        """Generate speech via OpenAI TTS and play with aplay."""
        output_path = tempfile.mktemp(suffix='.wav', prefix='explorer_tts_')
        try:
            # OpenAI TTS returns audio data
            response = self._openai_client.audio.speech.create(
                model=self.tts_model,
                voice=self.tts_voice,
                input=text[:4096],  # OpenAI TTS limit
                response_format='wav',
            )
            response.write_to_file(output_path)
            self._play_audio(output_path)
        except Exception as e:
            self._log_error(f"OpenAI TTS failed: {e}, falling back to espeak")
            self._speak_espeak(text)
        finally:
            try:
                os.unlink(output_path)
            except OSError as e:
                self._log_warn(f"Could not remove TTS temp file {output_path}: {e}")

    def _speak_espeak(self, text: str):
        """Fallback TTS using espeak."""
        try:
            subprocess.run(
                ['espeak', '-s', '150', text[:500]],
                capture_output=True, timeout=30,
            )
        except FileNotFoundError:
            self._log_warn("espeak not installed")
        except Exception as e:
            self._log_error(f"espeak failed: {e}")

    def _play_audio(self, path: str):
        """Play a WAV file through the speaker."""
        device = self._find_audio_device()
        try:
            subprocess.run(
                ['aplay', '-D', device, path],
                capture_output=True, timeout=30,
            )
        except Exception as e:
            self._log_error(f"aplay failed: {e}")

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def beep(self):
        """Play a short beep to indicate end of listening."""
        try:
            # Generate a short 200ms 880Hz beep WAV
            path = tempfile.mktemp(suffix='.wav', prefix='beep_')
            import struct, wave
            sr = 16000
            dur = 0.2
            n = int(sr * dur)
            with wave.open(path, 'w') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                import math
                samples = [int(16000 * math.sin(2 * math.pi * 880 * i / sr))
                           for i in range(n)]
                w.writeframes(struct.pack(f'<{n}h', *samples))
            self._play_audio(path)
            try:
                os.unlink(path)
            except OSError:
                pass  # Beep temp file cleanup is non-critical
        except Exception as e:
            self._log_warn(f"Beep generation failed: {e}")

    def listen_for_command(self, duration: int = 5) -> str:
        """Record and transcribe a voice command.

        Returns:
            Transcribed command text, or empty string.
        """
        audio_path = self.record(duration)
        self.beep()  # Indicate recording is done
        if audio_path:
            text = self.speech_to_text(audio_path)
            try:
                os.unlink(audio_path)
            except OSError as e:
                self._log_warn(f"Could not remove recording {audio_path}: {e}")
            return text
        return ''


class WonderEchoDetector:
    """Wake word detector for WonderEcho Pro via serial protocol."""

    WAKEUP_PACKET = b'\xaa\x55\x03\x00\xfb'

    def __init__(self, port: str = '/dev/wonderecho', logger=None):
        self.port = port
        self.logger = logger
        self._serial = None
        self._running = False

    def start(self):
        """Open serial connection to WonderEcho Pro."""
        try:
            import serial
            self._serial = serial.Serial(self.port, 115200, timeout=0.1)
            self._running = True
            if self.logger:
                self.logger.info(f"WonderEcho connected on {self.port}")
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"WonderEcho not available on {self.port}: {e}. "
                    f"Wake word detection disabled."
                )
            self._serial = None

    def stop(self):
        """Close serial connection."""
        self._running = False
        if self._serial:
            try:
                self._serial.close()
            except OSError:
                pass  # Serial port may already be closed
            self._serial = None

    def check_wakeup(self) -> bool:
        """Non-blocking check for wake word detection.

        Returns:
            True if wake word was detected.
        """
        if not self._serial:
            return False
        try:
            data = self._serial.read(5)
            if data == self.WAKEUP_PACKET:
                if self.logger:
                    self.logger.info("Wake word detected!")
                return True
        except OSError:
            pass  # Transient serial read errors are expected
        return False

    @property
    def available(self) -> bool:
        return self._serial is not None
