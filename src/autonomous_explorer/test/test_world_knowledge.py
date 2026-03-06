"""Tests for autonomous_explorer.world_knowledge module."""
import json
import os

import pytest

from autonomous_explorer.world_knowledge import WorldKnowledge


# ===================================================================
# Initialization and persistence
# ===================================================================

class TestWorldKnowledgeInit:
    """Test knowledge system initialization and file handling."""

    def test_creates_knowledge_directory(self, tmp_dir):
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        assert os.path.isdir(tmp_dir)

    def test_default_data_structure(self, tmp_dir):
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        assert 'rooms' in wk.world_map
        assert 'objects' in wk.known_objects
        assert 'navigation_lessons' in wk.learned_behaviors

    def test_loads_existing_data(self, tmp_dir):
        # Pre-populate a knowledge file
        map_path = os.path.join(tmp_dir, 'world_map.json')
        with open(map_path, 'w') as f:
            json.dump({
                'rooms': {'kitchen': {'times_visited': 3, 'last_visited': '2026-01-01'}},
                'corridors': [],
                'doors': [],
            }, f)
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        assert 'kitchen' in wk.world_map['rooms']
        assert wk.world_map['rooms']['kitchen']['times_visited'] == 3

    def test_corrupted_file_uses_defaults(self, tmp_dir):
        map_path = os.path.join(tmp_dir, 'world_map.json')
        with open(map_path, 'w') as f:
            f.write('{bad json')
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        assert 'rooms' in wk.world_map  # got defaults

    def test_forward_compat_merges_defaults(self, tmp_dir):
        map_path = os.path.join(tmp_dir, 'world_map.json')
        with open(map_path, 'w') as f:
            json.dump({'rooms': {}}, f)  # missing 'corridors', 'doors'
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        assert 'corridors' in wk.world_map
        assert 'doors' in wk.world_map


# ===================================================================
# update_from_response (per-cycle keyword extraction)
# ===================================================================

class TestUpdateFromResponse:
    """Test cheap per-cycle keyword extraction from LLM responses."""

    def test_detects_room_in_speech(self, tmp_dir):
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        wk.update_from_response({
            'speech': 'I appear to have entered the kitchen.',
            'reasoning': 'Tiles and cabinets visible.',
        })
        assert 'kitchen' in wk.world_map['rooms']

    def test_detects_multiple_rooms(self, tmp_dir):
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        wk.update_from_response({
            'speech': 'Moving from the hallway into the bedroom.',
            'reasoning': '',
        })
        assert 'hallway' in wk.world_map['rooms']
        assert 'bedroom' in wk.world_map['rooms']

    def test_detects_objects(self, tmp_dir):
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        wk.update_from_response({
            'speech': 'I see a chair and a table ahead.',
            'reasoning': '',
        })
        assert 'chair' in wk.known_objects['objects']
        assert 'table' in wk.known_objects['objects']

    def test_categorizes_living_beings(self, tmp_dir):
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        wk.update_from_response({
            'speech': 'I notice a cat on the couch.',
            'reasoning': '',
        })
        assert wk.known_objects['objects']['cat']['category'] == 'living_being'
        assert wk.known_objects['objects']['cat']['is_dynamic'] is True

    def test_increments_times_visited(self, tmp_dir):
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        wk.update_from_response({'speech': 'In the kitchen.', 'reasoning': ''})
        wk.update_from_response({'speech': 'Back in the kitchen.', 'reasoning': ''})
        assert wk.world_map['rooms']['kitchen']['times_visited'] == 2

    def test_increments_times_seen(self, tmp_dir):
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        wk.update_from_response({'speech': 'A chair.', 'reasoning': ''})
        wk.update_from_response({'speech': 'That chair again.', 'reasoning': ''})
        assert wk.known_objects['objects']['chair']['times_seen'] == 2

    def test_records_location_from_odom(self, tmp_dir):
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        wk.update_from_response(
            {'speech': 'A lamp here.', 'reasoning': ''},
            odom={'x': 1.5, 'y': 2.3},
        )
        obj = wk.known_objects['objects']['lamp']
        assert '1.5' in obj['usual_location']
        assert '2.3' in obj['usual_location']

    def test_no_match_does_nothing(self, tmp_dir):
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        wk.update_from_response({
            'speech': 'Moving forward slowly.',
            'reasoning': 'Clear path ahead.',
        })
        assert len(wk.world_map['rooms']) == 0
        assert len(wk.known_objects['objects']) == 0


# ===================================================================
# get_prompt_context
# ===================================================================

class TestPromptContext:
    """Test the spatially-filtered knowledge summary for LLM prompts."""

    def test_empty_returns_empty_string(self, tmp_dir):
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        ctx = wk.get_prompt_context()
        assert ctx == ''

    def test_includes_room_names(self, tmp_dir):
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        wk.update_from_response({'speech': 'In the kitchen.', 'reasoning': ''})
        ctx = wk.get_prompt_context()
        assert 'kitchen' in ctx

    def test_includes_last_room(self, tmp_dir):
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        wk.update_from_response({'speech': 'Kitchen area.', 'reasoning': ''})
        ctx = wk.get_prompt_context()
        assert 'LAST ROOM' in ctx

    def test_includes_objects(self, tmp_dir):
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        wk.update_from_response({'speech': 'A chair nearby.', 'reasoning': ''})
        ctx = wk.get_prompt_context()
        assert 'chair' in ctx

    def test_includes_lessons(self, tmp_dir):
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        wk.learned_behaviors['navigation_lessons'].append({
            'learned_on': '2026-03-01',
            'lesson': 'Avoid the narrow corridor.',
            'confidence': 0.8,
            'times_confirmed': 2,
        })
        ctx = wk.get_prompt_context()
        assert 'LESSONS' in ctx
        assert 'corridor' in ctx


# ===================================================================
# _apply_knowledge_update (from LLM end-of-session)
# ===================================================================

class TestApplyKnowledgeUpdate:
    """Test structured knowledge updates from LLM."""

    def test_adds_rooms(self, tmp_dir):
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        wk._apply_knowledge_update({
            'rooms_visited': ['kitchen', 'hallway'],
        })
        assert 'kitchen' in wk.world_map['rooms']
        assert 'hallway' in wk.world_map['rooms']

    def test_adds_objects(self, tmp_dir):
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        wk._apply_knowledge_update({
            'new_objects': [
                {'name': 'Bookshelf', 'category': 'furniture'},
            ],
        })
        assert 'bookshelf' in wk.known_objects['objects']

    def test_adds_lessons(self, tmp_dir):
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        wk._apply_knowledge_update({
            'lessons': ['Turn slowly near walls.'],
        })
        lessons = wk.learned_behaviors['navigation_lessons']
        assert len(lessons) == 1
        assert 'walls' in lessons[0]['lesson']

    def test_adds_room_connections(self, tmp_dir):
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        # First create the rooms
        wk._apply_knowledge_update({'rooms_visited': ['kitchen', 'hallway']})
        # Then connect them
        wk._apply_knowledge_update({
            'room_connections': [{'from': 'kitchen', 'to': 'hallway'}],
        })
        assert 'hallway' in wk.world_map['rooms']['kitchen']['connections']
        assert 'kitchen' in wk.world_map['rooms']['hallway']['connections']

    def test_ignores_invalid_types(self, tmp_dir):
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        wk._apply_knowledge_update({
            'rooms_visited': [123, None],
            'new_objects': ['not a dict', None],
            'lessons': [None, 42, ''],
        })
        # Should not crash; non-string rooms and empty lessons are handled
        # (empty string '' is technically a valid string, so it gets added)
        assert 123 not in wk.world_map['rooms']

    def test_no_duplicate_connections(self, tmp_dir):
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        wk._apply_knowledge_update({'rooms_visited': ['a', 'b']})
        wk._apply_knowledge_update({'room_connections': [{'from': 'a', 'to': 'b'}]})
        wk._apply_knowledge_update({'room_connections': [{'from': 'a', 'to': 'b'}]})
        assert wk.world_map['rooms']['a']['connections'].count('b') == 1


# ===================================================================
# Save / persistence
# ===================================================================

class TestSave:
    """Test knowledge persistence to disk."""

    def test_save_creates_all_files(self, tmp_dir):
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        wk.update_from_response({'speech': 'A kitchen with a chair.', 'reasoning': ''})
        wk.save()

        assert os.path.exists(os.path.join(tmp_dir, 'world_map.json'))
        assert os.path.exists(os.path.join(tmp_dir, 'known_objects.json'))
        assert os.path.exists(os.path.join(tmp_dir, 'learned_behaviors.json'))

    def test_save_and_reload_roundtrip(self, tmp_dir):
        wk1 = WorldKnowledge(knowledge_dir=tmp_dir)
        wk1.update_from_response({'speech': 'A kitchen.', 'reasoning': ''})
        wk1.save()

        wk2 = WorldKnowledge(knowledge_dir=tmp_dir)
        assert 'kitchen' in wk2.world_map['rooms']


# ===================================================================
# Summary methods (for CLI tool)
# ===================================================================

class TestSummaryMethods:
    """Test the text summary methods used by the CLI tool."""

    def test_rooms_summary_empty(self, tmp_dir):
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        assert 'No rooms' in wk.get_rooms_summary()

    def test_rooms_summary_with_data(self, tmp_dir):
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        wk.update_from_response({'speech': 'kitchen visit.', 'reasoning': ''})
        summary = wk.get_rooms_summary()
        assert 'kitchen' in summary

    def test_objects_summary_empty(self, tmp_dir):
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        assert 'No objects' in wk.get_objects_summary()

    def test_objects_summary_with_data(self, tmp_dir):
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        wk.update_from_response({'speech': 'A chair.', 'reasoning': ''})
        summary = wk.get_objects_summary()
        assert 'chair' in summary

    def test_lessons_summary_empty(self, tmp_dir):
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        assert 'No navigation' in wk.get_lessons_summary()

    def test_lessons_summary_with_data(self, tmp_dir):
        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        wk.learned_behaviors['navigation_lessons'].append({
            'lesson': 'Go slow near walls.',
            'confidence': 0.7,
        })
        summary = wk.get_lessons_summary()
        assert 'walls' in summary


# ===================================================================
# End-of-session LLM update
# ===================================================================

class TestEndOfSessionUpdate:
    """Test the end-of-session knowledge update."""

    def test_dryrun_just_saves(self, tmp_dir):
        from autonomous_explorer.llm_provider import DryRunProvider
        from autonomous_explorer.exploration_memory import ExplorationMemory

        wk = WorldKnowledge(knowledge_dir=tmp_dir)
        wk.update_from_response({'speech': 'A kitchen.', 'reasoning': ''})
        mem = ExplorationMemory(os.path.join(tmp_dir, 'mem.json'))
        provider = DryRunProvider()

        wk.end_of_session_update(mem, provider)

        # Should have saved files
        assert os.path.exists(os.path.join(tmp_dir, 'world_map.json'))
