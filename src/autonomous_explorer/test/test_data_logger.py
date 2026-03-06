"""Tests for autonomous_explorer.data_logger module."""
import gzip
import json
import os
import time

import numpy as np
import pytest

from autonomous_explorer.data_logger import DataLogger, _json_default


# ===================================================================
# _json_default serializer
# ===================================================================

class TestJsonDefault:
    """Test the custom JSON serializer for numpy types and edge cases."""

    def test_nan_becomes_none(self):
        assert _json_default(float('nan')) is None

    def test_inf_becomes_none(self):
        assert _json_default(float('inf')) is None
        assert _json_default(float('-inf')) is None

    def test_numpy_float(self):
        result = _json_default(np.float64(3.14))
        assert isinstance(result, float)
        assert abs(result - 3.14) < 1e-10

    def test_numpy_int(self):
        result = _json_default(np.int32(42))
        assert isinstance(result, int)
        assert result == 42

    def test_numpy_array(self):
        arr = np.array([1, 2, 3])
        result = _json_default(arr)
        assert result == [1, 2, 3]

    def test_unknown_type_becomes_string(self):
        result = _json_default(object())
        assert isinstance(result, str)

    def test_regular_float_passthrough(self):
        # Regular floats should not match the special cases
        result = _json_default(3.14)
        # Returns str(3.14) since it's not NaN/Inf and not a numpy type
        assert isinstance(result, str)


# ===================================================================
# DataLogger initialization
# ===================================================================

class TestDataLoggerInit:
    """Test DataLogger creation and directory setup."""

    def test_creates_log_directory(self, tmp_dir):
        log_dir = os.path.join(tmp_dir, 'logs')
        dl = DataLogger(log_dir=log_dir, log_level='compact')
        assert os.path.isdir(log_dir)

    def test_full_mode_creates_frame_dirs(self, tmp_dir):
        log_dir = os.path.join(tmp_dir, 'logs')
        dl = DataLogger(log_dir=log_dir, log_level='full')
        # Frame dirs exist under session dir
        assert dl._rgb_dir != ''
        assert dl._depth_dir != ''
        assert os.path.isdir(dl._rgb_dir)
        assert os.path.isdir(dl._depth_dir)

    def test_compact_mode_no_frame_dirs(self, tmp_dir):
        log_dir = os.path.join(tmp_dir, 'logs')
        dl = DataLogger(log_dir=log_dir, log_level='compact')
        assert dl._rgb_dir == ''
        assert dl._depth_dir == ''

    def test_session_id_set(self, tmp_dir):
        dl = DataLogger(log_dir=tmp_dir, log_level='minimal')
        assert len(dl.session_id) > 0

    def test_initial_counters_zero(self, tmp_dir):
        dl = DataLogger(log_dir=tmp_dir, log_level='minimal')
        assert dl.cycle_count == 0
        assert dl.total_input_tokens == 0
        assert dl.total_output_tokens == 0
        assert dl.total_cost_usd == 0.0


# ===================================================================
# DataLogger lifecycle (start/stop)
# ===================================================================

class TestDataLoggerLifecycle:
    """Test start/stop and background thread management."""

    def test_start_creates_jsonl_file(self, tmp_dir):
        dl = DataLogger(log_dir=tmp_dir, log_level='compact')
        dl.start()
        assert os.path.exists(dl.jsonl_path)
        dl.stop()

    def test_stop_flushes_and_closes(self, tmp_dir):
        dl = DataLogger(log_dir=tmp_dir, log_level='compact')
        dl.start()
        dl.log_cycle(parsed_action='forward', speed=0.5, duration=1.0)
        dl.stop()
        # File should have content
        assert os.path.getsize(dl.jsonl_path) > 0

    def test_double_stop_safe(self, tmp_dir):
        dl = DataLogger(log_dir=tmp_dir, log_level='compact')
        dl.start()
        dl.stop()
        dl.stop()  # Should not raise


# ===================================================================
# log_cycle
# ===================================================================

class TestLogCycle:
    """Test cycle logging at various log levels."""

    def test_minimal_record_structure(self, tmp_dir):
        dl = DataLogger(log_dir=tmp_dir, log_level='minimal')
        dl.start()
        dl.log_cycle(
            parsed_action='forward',
            speed=0.5,
            duration=1.0,
            safety_triggered=True,
            safety_reason='obstacle',
            response_time_ms=200,
        )
        dl.stop()
        with open(dl.jsonl_path) as f:
            record = json.loads(f.readline())
        assert record['parsed_action'] == 'forward'
        assert record['safety_triggered'] is True
        assert 'sensor_data' not in record  # minimal has no sensor_data

    def test_compact_record_structure(self, tmp_dir):
        dl = DataLogger(log_dir=tmp_dir, log_level='compact')
        dl.start()
        dl.log_cycle(
            provider='claude',
            model='claude-sonnet',
            parsed_action='turn_left',
            speed=0.3,
            duration=0.8,
            tokens_input=100,
            tokens_output=50,
            cost_usd=0.001,
        )
        dl.stop()
        with open(dl.jsonl_path) as f:
            record = json.loads(f.readline())
        assert 'sensor_data' in record
        assert 'llm_input' in record
        assert 'llm_output' in record
        assert 'safety_override' in record
        assert 'execution' in record
        assert record['llm_output']['parsed_action'] == 'turn_left'

    def test_full_record_saves_frames(self, tmp_dir, sample_rgb_image, sample_depth_image):
        dl = DataLogger(log_dir=tmp_dir, log_level='full')
        dl.start()
        dl.log_cycle(
            rgb_image=sample_rgb_image,
            depth_image=sample_depth_image,
            parsed_action='forward',
            speed=0.5,
            duration=1.0,
        )
        # Give background thread time to write
        time.sleep(1.0)
        dl.stop()
        # Check frame files were created
        rgb_files = os.listdir(dl._rgb_dir)
        depth_files = os.listdir(dl._depth_dir)
        assert len(rgb_files) >= 1
        assert len(depth_files) >= 1

    def test_cycle_counter_increments(self, tmp_dir):
        dl = DataLogger(log_dir=tmp_dir, log_level='minimal')
        dl.start()
        for _ in range(5):
            dl.log_cycle(parsed_action='forward', speed=0.5, duration=1.0)
        dl.stop()
        assert dl.cycle_count == 5

    def test_token_accumulation(self, tmp_dir):
        dl = DataLogger(log_dir=tmp_dir, log_level='minimal')
        dl.start()
        dl.log_cycle(tokens_input=100, tokens_output=50, cost_usd=0.01)
        dl.log_cycle(tokens_input=200, tokens_output=80, cost_usd=0.02)
        dl.stop()
        assert dl.total_input_tokens == 300
        assert dl.total_output_tokens == 130
        assert abs(dl.total_cost_usd - 0.03) < 1e-10

    def test_consciousness_fields_in_record(self, tmp_dir):
        dl = DataLogger(log_dir=tmp_dir, log_level='compact')
        dl.start()
        dl.log_cycle(
            parsed_action='forward',
            embodied_reflection='The path is clear.',
            outing_number=5,
        )
        dl.stop()
        with open(dl.jsonl_path) as f:
            record = json.loads(f.readline())
        assert record['consciousness']['embodied_reflection'] == 'The path is clear.'
        assert record['consciousness']['outing_number'] == 5

    def test_lidar_min_distance_computed(self, tmp_dir):
        dl = DataLogger(log_dir=tmp_dir, log_level='compact')
        dl.start()
        dl.log_cycle(
            lidar_ranges=[0.5, 0.3, 1.0, 0.8, float('inf'), 0.0],
            parsed_action='forward',
        )
        dl.stop()
        with open(dl.jsonl_path) as f:
            record = json.loads(f.readline())
        sensor = record['sensor_data']
        assert sensor['lidar_min_distance'] == 0.3

    def test_queue_full_does_not_block(self, tmp_dir):
        dl = DataLogger(log_dir=tmp_dir, log_level='minimal')
        # Don't start the writer — queue will fill up
        dl._running = False
        for _ in range(600):  # exceed maxsize=500
            dl.log_cycle(parsed_action='forward')
        # Should not hang


# ===================================================================
# Old session compression
# ===================================================================

class TestCompression:
    """Test auto-compression of old JSONL files."""

    def test_compresses_old_files(self, tmp_dir):
        dl = DataLogger(
            log_dir=tmp_dir,
            log_level='minimal',
            compress_after_hours=0,  # compress everything
        )
        # Create a fake old JSONL file
        old_path = os.path.join(tmp_dir, 'exploration_20240101_000000.jsonl')
        with open(old_path, 'w') as f:
            f.write('{"test": true}\n')
        # Set mtime to 2 hours ago
        old_time = time.time() - 7200
        os.utime(old_path, (old_time, old_time))

        dl._compress_old_sessions()

        assert os.path.exists(old_path + '.gz')
        assert not os.path.exists(old_path)

    def test_skips_current_session(self, tmp_dir):
        dl = DataLogger(
            log_dir=tmp_dir,
            log_level='minimal',
            compress_after_hours=0,
        )
        dl.start()
        dl._compress_old_sessions()
        dl.stop()
        # Current session file should NOT be compressed
        assert os.path.exists(dl.jsonl_path)
        assert not os.path.exists(dl.jsonl_path + '.gz')

    def test_skips_already_compressed(self, tmp_dir):
        dl = DataLogger(
            log_dir=tmp_dir,
            log_level='minimal',
            compress_after_hours=0,
        )
        old_path = os.path.join(tmp_dir, 'exploration_20240101_000000.jsonl')
        gz_path = old_path + '.gz'
        with open(old_path, 'w') as f:
            f.write('{"test": true}\n')
        with open(gz_path, 'w') as f:
            f.write('already compressed')
        old_time = time.time() - 7200
        os.utime(old_path, (old_time, old_time))

        dl._compress_old_sessions()
        # Original should still exist (gz already exists, skip)
        assert os.path.exists(old_path)
