"""Tests for autonomous_explorer.exploration_memory module."""
import json
import math
import os
import time

import pytest

from autonomous_explorer.exploration_memory import (
    ExplorationMemory,
    _grid_key,
    _GRID_CELL,
)


# ===================================================================
# _grid_key helper
# ===================================================================

class TestGridKey:
    """Test the spatial quantization function."""

    def test_origin(self):
        assert _grid_key(0.0, 0.0) == '0.00,0.00'

    def test_quantizes_to_grid(self):
        key = _grid_key(0.1, 0.1)
        # Should round to nearest grid cell
        parts = key.split(',')
        x, y = float(parts[0]), float(parts[1])
        assert abs(x) <= _GRID_CELL
        assert abs(y) <= _GRID_CELL

    def test_negative_coords(self):
        key = _grid_key(-1.0, -2.0)
        parts = key.split(',')
        assert float(parts[0]) < 0
        assert float(parts[1]) < 0

    def test_nearby_coords_same_cell(self):
        k1 = _grid_key(0.26, 0.0)
        k2 = _grid_key(0.24, 0.0)
        # Both round to 0.25 grid cell
        assert k1 == k2

    def test_adjacent_cells_differ(self):
        k1 = _grid_key(0.0, 0.0)
        k2 = _grid_key(_GRID_CELL, 0.0)
        assert k1 != k2


# ===================================================================
# ExplorationMemory initialization and persistence
# ===================================================================

class TestExplorationMemoryInit:
    """Test memory creation and loading."""

    def test_fresh_memory(self, tmp_dir):
        filepath = os.path.join(tmp_dir, 'memory.json')
        mem = ExplorationMemory(filepath)
        assert mem.total_actions == 0
        assert len(mem.action_log) == 0
        assert len(mem.discoveries) == 0
        assert len(mem.visited_cells) == 0

    def test_save_and_reload(self, tmp_dir):
        filepath = os.path.join(tmp_dir, 'memory.json')
        mem = ExplorationMemory(filepath)
        mem.total_actions = 42
        mem.visited_cells.add('1.00,2.00')
        mem.discoveries.append({'description': 'a chair', 'time': time.time()})
        mem.save()

        mem2 = ExplorationMemory(filepath)
        assert mem2.total_actions == 42
        assert '1.00,2.00' in mem2.visited_cells
        assert len(mem2.discoveries) == 1

    def test_corrupted_file_starts_fresh(self, tmp_dir):
        filepath = os.path.join(tmp_dir, 'memory.json')
        with open(filepath, 'w') as f:
            f.write('not valid json{{{')
        mem = ExplorationMemory(filepath)
        assert mem.total_actions == 0

    def test_max_entries_respected(self, tmp_dir):
        filepath = os.path.join(tmp_dir, 'memory.json')
        mem = ExplorationMemory(filepath, max_entries=5)
        for i in range(10):
            mem.discoveries.append({'description': f'item {i}', 'time': time.time()})
        mem.save()
        # After save, discoveries trimmed to max_entries
        mem2 = ExplorationMemory(filepath, max_entries=5)
        assert len(mem2.discoveries) <= 5


# ===================================================================
# record_action
# ===================================================================

class TestRecordAction:
    """Test action recording and side effects."""

    def test_basic_record(self, tmp_dir, sample_llm_response):
        mem = ExplorationMemory(os.path.join(tmp_dir, 'mem.json'))
        mem.record_action(sample_llm_response, "sensor data")
        assert mem.total_actions == 1
        assert len(mem.action_log) == 1
        assert mem.action_log[0]['action'] == 'forward'

    def test_records_position_from_odom(self, tmp_dir, sample_llm_response, sample_odom):
        mem = ExplorationMemory(os.path.join(tmp_dir, 'mem.json'))
        mem.record_action(sample_llm_response, "data", odom=sample_odom)
        entry = mem.action_log[0]
        assert 'x' in entry
        assert 'y' in entry
        assert 'heading' in entry
        assert entry['x'] == sample_odom['x']

    def test_marks_visited_cell(self, tmp_dir, sample_llm_response, sample_odom):
        mem = ExplorationMemory(os.path.join(tmp_dir, 'mem.json'))
        mem.record_action(sample_llm_response, "data", odom=sample_odom)
        assert len(mem.visited_cells) >= 1

    def test_marks_obstacles_from_lidar(
        self, tmp_dir, sample_llm_response, sample_odom, sample_lidar_sectors
    ):
        mem = ExplorationMemory(os.path.join(tmp_dir, 'mem.json'))
        mem.record_action(
            sample_llm_response, "data",
            odom=sample_odom,
            lidar_sectors=sample_lidar_sectors,
        )
        assert len(mem.obstacle_cells) > 0

    def test_discovery_detection_from_speech(self, tmp_dir):
        mem = ExplorationMemory(os.path.join(tmp_dir, 'mem.json'))
        response = {
            'action': 'stop',
            'speed': 0.0,
            'duration': 0.0,
            'speech': 'I see a beautiful painting on the wall!',
            'reasoning': 'Object detected.',
        }
        mem.record_action(response, "data")
        assert len(mem.discoveries) == 1
        assert 'painting' in mem.discoveries[0]['description'].lower()

    def test_no_discovery_without_keywords(self, tmp_dir):
        mem = ExplorationMemory(os.path.join(tmp_dir, 'mem.json'))
        response = {
            'action': 'forward',
            'speed': 0.5,
            'duration': 1.0,
            'speech': 'Moving ahead through the corridor.',
            'reasoning': 'Clear path.',
        }
        mem.record_action(response, "data")
        assert len(mem.discoveries) == 0

    def test_movement_history_tracks_actions(self, tmp_dir):
        mem = ExplorationMemory(os.path.join(tmp_dir, 'mem.json'))
        for action_name in ['forward', 'turn_left', 'forward']:
            mem.record_action(
                {'action': action_name, 'speed': 0.3, 'duration': 1.0,
                 'speech': '', 'reasoning': ''},
                "data",
            )
        assert list(mem.movement_history) == ['forward', 'turn_left', 'forward']

    def test_auto_save_every_10_actions(self, tmp_dir):
        filepath = os.path.join(tmp_dir, 'mem.json')
        mem = ExplorationMemory(filepath)
        for i in range(10):
            mem.record_action(
                {'action': 'forward', 'speed': 0.3, 'duration': 1.0,
                 'speech': '', 'reasoning': ''},
                "data",
            )
        assert os.path.exists(filepath)
        with open(filepath) as f:
            data = json.load(f)
        assert data['total_actions'] == 10


# ===================================================================
# get_context_summary
# ===================================================================

class TestContextSummary:
    """Test the text summary generated for LLM prompts."""

    def test_empty_memory(self, tmp_dir):
        mem = ExplorationMemory(os.path.join(tmp_dir, 'mem.json'))
        summary = mem.get_context_summary()
        assert 'Exploration time' in summary
        assert 'Total actions: 0' in summary

    def test_includes_recent_actions(self, tmp_dir, sample_odom):
        mem = ExplorationMemory(os.path.join(tmp_dir, 'mem.json'))
        for _ in range(3):
            mem.record_action(
                {'action': 'forward', 'speed': 0.5, 'duration': 1.0,
                 'speech': '', 'reasoning': 'clear path'},
                "data",
                odom=sample_odom,
            )
        summary = mem.get_context_summary()
        assert 'Recent action history' in summary
        assert 'forward' in summary

    def test_stuck_detection(self, tmp_dir):
        mem = ExplorationMemory(os.path.join(tmp_dir, 'mem.json'))
        for _ in range(6):
            mem.record_action(
                {'action': 'turn_left', 'speed': 0.3, 'duration': 0.8,
                 'speech': '', 'reasoning': ''},
                "data",
            )
        summary = mem.get_context_summary()
        assert 'stuck' in summary.lower() or 'WARNING' in summary

    def test_stop_repetition_not_flagged_as_stuck(self, tmp_dir):
        mem = ExplorationMemory(os.path.join(tmp_dir, 'mem.json'))
        for _ in range(6):
            mem.record_action(
                {'action': 'stop', 'speed': 0.0, 'duration': 0.0,
                 'speech': '', 'reasoning': ''},
                "data",
            )
        summary = mem.get_context_summary()
        assert 'stuck' not in summary.lower()

    def test_includes_spatial_map_info(self, tmp_dir, sample_odom):
        mem = ExplorationMemory(os.path.join(tmp_dir, 'mem.json'))
        mem.record_action(
            {'action': 'forward', 'speed': 0.5, 'duration': 1.0,
             'speech': '', 'reasoning': ''},
            "data",
            odom=sample_odom,
        )
        summary = mem.get_context_summary()
        assert 'Spatial map' in summary or 'cells visited' in summary

    def test_includes_discoveries(self, tmp_dir):
        mem = ExplorationMemory(os.path.join(tmp_dir, 'mem.json'))
        mem.record_action(
            {'action': 'stop', 'speed': 0.0, 'duration': 0.0,
             'speech': 'I see a red chair!', 'reasoning': ''},
            "data",
        )
        summary = mem.get_context_summary()
        assert 'Discoveries' in summary

    def test_unexplored_directions(self, tmp_dir, sample_odom):
        mem = ExplorationMemory(os.path.join(tmp_dir, 'mem.json'))
        mem.record_action(
            {'action': 'forward', 'speed': 0.5, 'duration': 1.0,
             'speech': '', 'reasoning': ''},
            "data",
            odom=sample_odom,
        )
        summary = mem.get_context_summary()
        assert 'Unexplored' in summary


# ===================================================================
# reset
# ===================================================================

class TestReset:
    """Test memory reset."""

    def test_clears_all_state(self, tmp_dir, sample_llm_response, sample_odom):
        mem = ExplorationMemory(os.path.join(tmp_dir, 'mem.json'))
        mem.record_action(sample_llm_response, "data", odom=sample_odom)
        mem.reset()
        assert mem.total_actions == 0
        assert len(mem.action_log) == 0
        assert len(mem.discoveries) == 0
        assert len(mem.visited_cells) == 0
        assert len(mem.obstacle_cells) == 0
        assert len(mem.movement_history) == 0
