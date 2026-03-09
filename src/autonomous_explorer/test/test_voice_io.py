"""Tests for autonomous_explorer.voice_io module."""
import os
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from autonomous_explorer.voice_io import VoiceIO, WonderEchoDetector


# ===================================================================
# VoiceIO initialization
# ===================================================================

class TestVoiceIOInit:
    """Test VoiceIO construction."""

    def test_init_without_api_key(self):
        vio = VoiceIO(openai_api_key='')
        assert vio._openai_available is False

    @patch('autonomous_explorer.voice_io.openai', create=True)
    def test_init_with_api_key(self, mock_openai):
        mock_openai.OpenAI.return_value = MagicMock()
        vio = VoiceIO(openai_api_key='sk-test')
        assert vio._openai_available is True

    def test_init_without_openai_package(self):
        vio = VoiceIO(openai_api_key='')
        assert vio._openai_available is False

    def test_is_speaking_initially_false(self):
        vio = VoiceIO()
        assert vio.is_speaking is False


# ===================================================================
# speak method
# ===================================================================

class TestSpeak:
    """Test the speak method routing."""

    def test_empty_text_returns_immediately(self):
        vio = VoiceIO()
        vio.speak('')  # Should not raise

    @patch.object(VoiceIO, '_speak_impl')
    def test_blocking_speak_calls_impl(self, mock_impl):
        vio = VoiceIO()
        vio.speak('Hello', block=True, force=True)
        mock_impl.assert_called_once_with('Hello')

    @patch.object(VoiceIO, '_speak_impl')
    def test_nonblocking_speak_starts_thread(self, mock_impl):
        vio = VoiceIO()
        vio.speak('Hello', block=False, force=True)
        # Give the thread a moment
        import time
        time.sleep(0.1)
        mock_impl.assert_called_once_with('Hello')

    @patch.object(VoiceIO, '_speak_impl')
    def test_rate_limiting_skips_rapid_calls(self, mock_impl):
        vio = VoiceIO(speak_min_interval=10.0)
        vio.speak('First', block=True, force=True)
        vio.speak('Second', block=True)  # Should be rate-limited
        mock_impl.assert_called_once_with('First')

    @patch.object(VoiceIO, '_speak_impl')
    def test_force_bypasses_rate_limiting(self, mock_impl):
        vio = VoiceIO(speak_min_interval=10.0)
        vio.speak('First', block=True, force=True)
        vio.speak('Second', block=True, force=True)
        assert mock_impl.call_count == 2


# ===================================================================
# _speak_impl
# ===================================================================

class TestSpeakImpl:
    """Test internal TTS implementation."""

    @patch.object(VoiceIO, '_speak_gtts')
    def test_fallback_to_gtts_when_no_openai(self, mock_gtts):
        vio = VoiceIO(openai_api_key='')
        vio._gtts_available = True
        vio._speak_impl('Test')
        mock_gtts.assert_called_once_with('Test')

    @patch.object(VoiceIO, '_speak_espeak')
    def test_fallback_to_espeak_when_no_openai_no_gtts(self, mock_espeak):
        vio = VoiceIO(openai_api_key='')
        vio._gtts_available = False
        vio._speak_impl('Test')
        mock_espeak.assert_called_once_with('Test')

    def test_speaking_flag_set_during_speech(self):
        vio = VoiceIO(openai_api_key='')
        with patch.object(VoiceIO, '_speak_espeak'):
            assert vio.is_speaking is False
            # Can't easily test during, but after should be False
            vio._speak_impl('Test')
            assert vio.is_speaking is False


# ===================================================================
# _find_audio_device
# ===================================================================

class TestFindAudioDevice:
    """Test audio device auto-detection."""

    @patch('autonomous_explorer.voice_io.subprocess.run')
    def test_finds_usb_device(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout='card 2: USB Audio [USB Audio], device 0: USB Audio\n',
        )
        vio = VoiceIO(audio_device='plughw:1,0')
        device = vio._find_audio_device()
        assert device == 'plughw:2,0'

    @patch('autonomous_explorer.voice_io.subprocess.run')
    def test_finds_wonderecho_device(self, mock_run):
        # WonderEcho devices show "USB PnP" in arecord -l output
        mock_run.return_value = MagicMock(
            stdout='card 3: USB PnP WonderEcho [WonderEcho Pro], device 0:\n',
        )
        vio = VoiceIO(audio_device='plughw:1,0')
        device = vio._find_audio_device()
        assert device == 'plughw:3,0'

    @patch('autonomous_explorer.voice_io.subprocess.run')
    def test_no_usb_device_returns_default(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout='card 0: Internal [Internal Audio]\n',
        )
        vio = VoiceIO(audio_device='plughw:1,0')
        device = vio._find_audio_device()
        assert device == 'plughw:1,0'

    @patch('autonomous_explorer.voice_io.subprocess.run', side_effect=OSError("no arecord"))
    def test_error_returns_default(self, mock_run):
        vio = VoiceIO(audio_device='plughw:1,0')
        device = vio._find_audio_device()
        assert device == 'plughw:1,0'


# ===================================================================
# record
# ===================================================================

class TestRecord:
    """Test audio recording."""

    @patch('subprocess.run')
    @patch('os.path.exists', return_value=True)
    @patch('os.path.getsize', return_value=5000)
    def test_successful_recording(self, mock_size, mock_exists, mock_run):
        vio = VoiceIO(audio_device='plughw:2,0')
        with patch.object(vio, '_find_audio_device', return_value='plughw:2,0'):
            result = vio.record(duration=3, output_path='/tmp/test.wav')
        assert result == '/tmp/test.wav'

    @patch('subprocess.run', side_effect=Exception("device busy"))
    def test_recording_failure_returns_empty(self, mock_run):
        vio = VoiceIO(audio_device='plughw:2,0')
        with patch.object(vio, '_find_audio_device', return_value='plughw:2,0'):
            result = vio.record(duration=3)
        assert result == ''

    @patch('subprocess.run')
    @patch('os.path.exists', return_value=True)
    @patch('os.path.getsize', return_value=100)  # too small
    def test_small_recording_returns_empty(self, mock_size, mock_exists, mock_run):
        vio = VoiceIO(audio_device='plughw:2,0')
        with patch.object(vio, '_find_audio_device', return_value='plughw:2,0'):
            result = vio.record(duration=3, output_path='/tmp/test.wav')
        assert result == ''


# ===================================================================
# speech_to_text
# ===================================================================

class TestSpeechToText:
    """Test STT cascade: OpenAI Whisper → Google Speech Recognition."""

    def test_stt_missing_file_returns_empty(self):
        vio = VoiceIO(openai_api_key='')
        result = vio.speech_to_text('/tmp/nonexistent_file.wav')
        assert result == ''

    @patch.object(VoiceIO, '_stt_google', return_value='hello')
    def test_stt_falls_back_to_google_without_openai(self, mock_google):
        vio = VoiceIO(openai_api_key='')
        # Create a minimal WAV file for os.path.exists check
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(b'\x00' * 100)
            path = f.name
        try:
            result = vio.speech_to_text(path)
            assert result == 'hello'
            mock_google.assert_called_once_with(path)
        finally:
            os.unlink(path)

    @patch.object(VoiceIO, '_stt_google', return_value='fallback text')
    def test_stt_falls_back_to_google_on_openai_error(self, mock_google):
        vio = VoiceIO(openai_api_key='')
        vio._openai_available = True
        vio._openai_client = MagicMock()
        vio._openai_client.audio.transcriptions.create.side_effect = Exception("429 quota")
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(b'\x00' * 100)
            path = f.name
        try:
            result = vio.speech_to_text(path)
            assert result == 'fallback text'
        finally:
            os.unlink(path)


# ===================================================================
# listen_for_command
# ===================================================================

class TestListenForCommand:
    """Test the combined record + transcribe flow."""

    @patch.object(VoiceIO, 'speech_to_text', return_value='start exploring')
    @patch.object(VoiceIO, 'record', return_value='/tmp/test.wav')
    @patch.object(VoiceIO, 'beep')
    @patch('os.unlink')
    def test_returns_transcription(self, mock_unlink, mock_beep, mock_record, mock_stt):
        vio = VoiceIO()
        result = vio.listen_for_command(duration=5)
        assert result == 'start exploring'
        mock_unlink.assert_called_once_with('/tmp/test.wav')

    @patch.object(VoiceIO, 'record', return_value='')
    def test_failed_recording_returns_empty(self, mock_record):
        vio = VoiceIO()
        result = vio.listen_for_command()
        assert result == ''


# ===================================================================
# WonderEchoDetector
# ===================================================================

class TestWonderEchoDetector:
    """Test the wake word detector."""

    def test_init(self):
        det = WonderEchoDetector(port='/dev/wonderecho')
        assert det.port == '/dev/wonderecho'
        assert det.available is False

    def test_start_connects(self):
        mock_serial = MagicMock()
        mock_serial_module = MagicMock()
        mock_serial_module.Serial.return_value = mock_serial

        det = WonderEchoDetector(port='/dev/wonderecho')
        with patch.dict('sys.modules', {'serial': mock_serial_module}):
            det.start()
        assert det.available is True

    def test_start_handles_missing_device(self):
        det = WonderEchoDetector(port='/dev/nonexistent')
        det.start()  # Should not raise
        assert det.available is False

    def test_check_wakeup_no_serial(self):
        det = WonderEchoDetector()
        assert det.check_wakeup() is False

    def test_check_wakeup_detects_packet(self):
        mock_serial = MagicMock()
        mock_serial.read.return_value = WonderEchoDetector.WAKEUP_PACKET

        det = WonderEchoDetector()
        det._serial = mock_serial  # bypass start()
        assert det.check_wakeup() is True

    def test_check_wakeup_no_data(self):
        mock_serial = MagicMock()
        mock_serial.read.return_value = b''

        det = WonderEchoDetector()
        det._serial = mock_serial
        assert det.check_wakeup() is False

    def test_stop_closes_serial(self):
        mock_serial = MagicMock()
        det = WonderEchoDetector()
        det._serial = mock_serial
        det._running = True
        det.stop()
        mock_serial.close.assert_called_once()
        assert det.available is False

    def test_double_stop_safe(self):
        det = WonderEchoDetector()
        det.stop()
        det.stop()  # Should not raise
