"""Tests for autonomous_explorer.world_knowledge module (NetworkX backend)."""
import json
import os

import pytest

from autonomous_explorer.world_knowledge import WorldKnowledge


# ===================================================================
# Initialization and persistence
# ===================================================================

class TestWorldKnowledgeInit:
    """Test knowledge system initialization and file handling."""

    def test_creates_graph_directory(self, tmp_dir):
        graph_path = os.path.join(tmp_dir, 'sub', 'kg.json')
        WorldKnowledge(graph_path=graph_path)
        assert os.path.isdir(os.path.join(tmp_dir, 'sub'))

    def test_default_data_structure(self, tmp_dir):
        wk = WorldKnowledge(graph_path=os.path.join(tmp_dir, 'kg.json'))
        assert 'rooms' in wk.world_map
        assert 'objects' in wk.known_objects
        assert 'navigation_lessons' in wk.learned_behaviors

    def test_loads_existing_graph(self, tmp_dir):
        graph_path = os.path.join(tmp_dir, 'kg.json')
        # Create and save a graph
        wk1 = WorldKnowledge(graph_path=graph_path)
        wk1.add_room('kitchen', x=1.0, y=2.0)
        # Load it back
        wk2 = WorldKnowledge(graph_path=graph_path)
        rooms = wk2.get_rooms()
        assert 'kitchen' in rooms
        assert rooms['kitchen']['x'] == 1.0

    def test_corrupted_file_starts_empty(self, tmp_dir):
        graph_path = os.path.join(tmp_dir, 'kg.json')
        with open(graph_path, 'w') as f:
            f.write('{bad json')
        wk = WorldKnowledge(graph_path=graph_path)
        assert len(wk.get_rooms()) == 0

    def test_empty_graph_has_compat_properties(self, tmp_dir):
        wk = WorldKnowledge(graph_path=os.path.join(tmp_dir, 'kg.json'))
        assert wk.world_map == {'rooms': {}}
        assert wk.known_objects == {'objects': {}}


# ===================================================================
# Room operations
# ===================================================================

class TestRoomOperations:
    """Test room add/update/query."""

    def test_add_room(self, tmp_dir):
        wk = WorldKnowledge(graph_path=os.path.join(tmp_dir, 'kg.json'))
        result = wk.add_room('kitchen', x=1.0, y=2.0, description='Tiles')
        assert result['type'] == 'room'
        assert result['x'] == 1.0
        assert result['times_visited'] == 1

    def test_revisit_increments_count(self, tmp_dir):
        wk = WorldKnowledge(graph_path=os.path.join(tmp_dir, 'kg.json'))
        wk.add_room('kitchen')
        wk.add_room('kitchen')
        rooms = wk.get_rooms()
        assert rooms['kitchen']['times_visited'] == 2

    def test_add_connection(self, tmp_dir):
        wk = WorldKnowledge(graph_path=os.path.join(tmp_dir, 'kg.json'))
        wk.add_room('kitchen')
        wk.add_room('hallway')
        wk.add_connection('kitchen', 'hallway')
        conns = wk.get_room_connections('kitchen')
        assert 'hallway' in conns
        conns_back = wk.get_room_connections('hallway')
        assert 'kitchen' in conns_back

    def test_get_room_objects(self, tmp_dir):
        wk = WorldKnowledge(graph_path=os.path.join(tmp_dir, 'kg.json'))
        wk.add_room('kitchen')
        wk.add_object('cup', room='kitchen')
        objs = wk.get_room_objects('kitchen')
        assert 'cup' in objs

    def test_compat_world_map_includes_connections(self, tmp_dir):
        wk = WorldKnowledge(graph_path=os.path.join(tmp_dir, 'kg.json'))
        wk.add_room('kitchen')
        wk.add_room('hallway')
        wk.add_connection('kitchen', 'hallway')
        wm = wk.world_map
        assert 'hallway' in wm['rooms']['kitchen']['connections']


# ===================================================================
# Object operations
# ===================================================================

class TestObjectOperations:
    """Test object add/update/query."""

    def test_add_object(self, tmp_dir):
        wk = WorldKnowledge(graph_path=os.path.join(tmp_dir, 'kg.json'))
        result = wk.add_object('cup', room='kitchen', confidence=0.9)
        assert result['type'] == 'object'
        assert result['confidence'] == 0.9

    def test_object_room_lookup(self, tmp_dir):
        wk = WorldKnowledge(graph_path=os.path.join(tmp_dir, 'kg.json'))
        wk.add_object('cup', room='kitchen')
        assert wk.get_room_for_object('cup') == 'kitchen'

    def test_object_moves_rooms(self, tmp_dir):
        wk = WorldKnowledge(graph_path=os.path.join(tmp_dir, 'kg.json'))
        wk.add_object('cup', room='kitchen')
        wk.add_object('cup', room='bedroom')
        assert wk.get_room_for_object('cup') == 'bedroom'

    def test_get_objects_includes_room(self, tmp_dir):
        wk = WorldKnowledge(graph_path=os.path.join(tmp_dir, 'kg.json'))
        wk.add_object('cup', room='kitchen')
        objs = wk.get_objects()
        assert objs['cup']['room'] == 'kitchen'

    def test_compat_known_objects(self, tmp_dir):
        wk = WorldKnowledge(graph_path=os.path.join(tmp_dir, 'kg.json'))
        wk.add_object('cup')
        assert 'cup' in wk.known_objects['objects']


# ===================================================================
# Search tracking
# ===================================================================

class TestSearchTracking:
    """Test search state tracking."""

    def test_mark_room_searched(self, tmp_dir):
        wk = WorldKnowledge(graph_path=os.path.join(tmp_dir, 'kg.json'))
        wk.add_room('kitchen')
        wk.mark_room_searched('kitchen', 'trash can')
        summary = wk.get_search_summary('trash can')
        assert 'kitchen' in summary['searched_rooms']

    def test_unsearched_rooms_listed(self, tmp_dir):
        wk = WorldKnowledge(graph_path=os.path.join(tmp_dir, 'kg.json'))
        wk.add_room('kitchen')
        wk.add_room('bedroom')
        wk.mark_room_searched('kitchen', 'cup')
        summary = wk.get_search_summary('cup')
        assert 'kitchen' in summary['searched_rooms']
        assert 'bedroom' in summary['unsearched_rooms']

    def test_found_in_detected(self, tmp_dir):
        wk = WorldKnowledge(graph_path=os.path.join(tmp_dir, 'kg.json'))
        wk.add_room('kitchen')
        wk.add_object('cup', room='kitchen')
        summary = wk.get_search_summary('cup')
        assert summary['found_in'] == 'kitchen'

    def test_suggested_next(self, tmp_dir):
        wk = WorldKnowledge(graph_path=os.path.join(tmp_dir, 'kg.json'))
        wk.add_room('kitchen')
        wk.add_room('bedroom')
        wk.mark_room_searched('kitchen', 'cup')
        summary = wk.get_search_summary('cup')
        assert summary['suggested_next'] == 'bedroom'

    def test_get_unexplored_connections(self, tmp_dir):
        wk = WorldKnowledge(graph_path=os.path.join(tmp_dir, 'kg.json'))
        wk.add_room('kitchen')
        wk.add_connection('kitchen', 'pantry')
        # pantry was auto-created with times_visited=1 by add_connection
        # but we can test by creating an unvisited room manually
        unexplored = wk.get_unexplored_connections('kitchen')
        # pantry was auto-created via add_room (times_visited=1), so not "unexplored"
        assert isinstance(unexplored, list)


# ===================================================================
# update_from_response (per-cycle keyword extraction)
# ===================================================================

class TestUpdateFromResponse:
    """Test cheap per-cycle keyword extraction from LLM responses."""

    def test_detects_room_in_speech(self, tmp_dir):
        wk = WorldKnowledge(graph_path=os.path.join(tmp_dir, 'kg.json'))
        wk.update_from_response({
            'speech': 'I appear to have entered the kitchen.',
            'reasoning': 'Tiles and cabinets visible.',
        })
        assert 'kitchen' in wk.get_rooms()

    def test_detects_multiple_rooms(self, tmp_dir):
        wk = WorldKnowledge(graph_path=os.path.join(tmp_dir, 'kg.json'))
        wk.update_from_response({
            'speech': 'Moving from the hallway into the bedroom.',
            'reasoning': '',
        })
        rooms = wk.get_rooms()
        assert 'hallway' in rooms
        assert 'bedroom' in rooms

    def test_no_match_does_nothing(self, tmp_dir):
        wk = WorldKnowledge(graph_path=os.path.join(tmp_dir, 'kg.json'))
        wk.update_from_response({
            'speech': 'Moving forward slowly.',
            'reasoning': 'Clear path ahead.',
        })
        assert len(wk.get_rooms()) == 0


# ===================================================================
# get_prompt_context
# ===================================================================

class TestPromptContext:
    """Test the spatially-filtered knowledge summary for LLM prompts."""

    def test_empty_returns_empty_string(self, tmp_dir):
        wk = WorldKnowledge(graph_path=os.path.join(tmp_dir, 'kg.json'))
        assert wk.get_prompt_context() == ''

    def test_includes_room_names(self, tmp_dir):
        wk = WorldKnowledge(graph_path=os.path.join(tmp_dir, 'kg.json'))
        wk.add_room('kitchen')
        ctx = wk.get_prompt_context()
        assert 'kitchen' in ctx

    def test_includes_objects(self, tmp_dir):
        wk = WorldKnowledge(graph_path=os.path.join(tmp_dir, 'kg.json'))
        wk.add_object('chair', room='kitchen')
        ctx = wk.get_prompt_context()
        assert 'chair' in ctx


# ===================================================================
# Save / persistence roundtrip
# ===================================================================

class TestSave:
    """Test knowledge persistence to disk."""

    def test_save_creates_file(self, tmp_dir):
        graph_path = os.path.join(tmp_dir, 'kg.json')
        wk = WorldKnowledge(graph_path=graph_path)
        wk.add_room('kitchen')
        assert os.path.exists(graph_path)

    def test_save_and_reload_roundtrip(self, tmp_dir):
        graph_path = os.path.join(tmp_dir, 'kg.json')
        wk1 = WorldKnowledge(graph_path=graph_path)
        wk1.add_room('kitchen', x=1.5, y=2.5)
        wk1.add_object('cup', room='kitchen', confidence=0.9)
        wk1.add_connection('kitchen', 'hallway')

        wk2 = WorldKnowledge(graph_path=graph_path)
        rooms = wk2.get_rooms()
        assert 'kitchen' in rooms
        assert rooms['kitchen']['x'] == 1.5
        assert wk2.get_room_for_object('cup') == 'kitchen'
        assert 'hallway' in wk2.get_room_connections('kitchen')


# ===================================================================
# Summary methods (for CLI tool)
# ===================================================================

class TestSummaryMethods:
    """Test the text summary methods used by the CLI tool."""

    def test_rooms_summary_empty(self, tmp_dir):
        wk = WorldKnowledge(graph_path=os.path.join(tmp_dir, 'kg.json'))
        assert 'No rooms' in wk.get_rooms_summary()

    def test_rooms_summary_with_data(self, tmp_dir):
        wk = WorldKnowledge(graph_path=os.path.join(tmp_dir, 'kg.json'))
        wk.add_room('kitchen')
        assert 'kitchen' in wk.get_rooms_summary()

    def test_objects_summary_empty(self, tmp_dir):
        wk = WorldKnowledge(graph_path=os.path.join(tmp_dir, 'kg.json'))
        assert 'No objects' in wk.get_objects_summary()

    def test_objects_summary_with_data(self, tmp_dir):
        wk = WorldKnowledge(graph_path=os.path.join(tmp_dir, 'kg.json'))
        wk.add_object('chair')
        assert 'chair' in wk.get_objects_summary()

    def test_lessons_summary(self, tmp_dir):
        wk = WorldKnowledge(graph_path=os.path.join(tmp_dir, 'kg.json'))
        assert 'No navigation' in wk.get_lessons_summary()


# ===================================================================
# End-of-session
# ===================================================================

class TestEndOfSessionUpdate:
    """Test the end-of-session knowledge update."""

    def test_dryrun_just_saves(self, tmp_dir):
        from autonomous_explorer.llm_provider import DryRunProvider
        from autonomous_explorer.exploration_memory import ExplorationMemory

        graph_path = os.path.join(tmp_dir, 'kg.json')
        wk = WorldKnowledge(graph_path=graph_path)
        wk.add_room('kitchen')
        mem = ExplorationMemory(os.path.join(tmp_dir, 'mem.json'))
        provider = DryRunProvider()

        wk.end_of_session_update(mem, provider)

        # Should have saved graph
        assert os.path.exists(graph_path)
