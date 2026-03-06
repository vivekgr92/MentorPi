"""Tests for autonomous_explorer.consciousness module."""
import json
import os
import time
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from autonomous_explorer.consciousness import JeevesConsciousness


# ===================================================================
# Initialization and lifetime stats
# ===================================================================

class TestConsciousnessInit:
    """Test consciousness layer initialization."""

    def test_creates_directories(self, tmp_dir):
        jc = JeevesConsciousness(stats_dir=tmp_dir)
        assert os.path.isdir(tmp_dir)
        assert os.path.isdir(os.path.join(tmp_dir, 'journals'))
        assert os.path.isdir(os.path.join(tmp_dir, 'logs'))

    def test_first_outing(self, tmp_dir):
        jc = JeevesConsciousness(stats_dir=tmp_dir)
        assert jc.outing_number == 1

    def test_increments_outing_on_new_session(self, tmp_dir):
        jc1 = JeevesConsciousness(stats_dir=tmp_dir)
        assert jc1.outing_number == 1
        jc1.save()
        jc2 = JeevesConsciousness(stats_dir=tmp_dir)
        assert jc2.outing_number == 2

    def test_default_stats_structure(self, tmp_dir):
        jc = JeevesConsciousness(stats_dir=tmp_dir)
        stats = jc.stats
        assert 'total_outings' in stats
        assert 'total_cycles' in stats
        assert 'total_distance_m' in stats
        assert 'total_cost_usd' in stats
        assert 'rooms_discovered' in stats

    def test_corrupted_stats_file_starts_fresh(self, tmp_dir):
        stats_path = os.path.join(tmp_dir, 'jeeves_lifetime_stats.json')
        with open(stats_path, 'w') as f:
            f.write('{invalid json')
        jc = JeevesConsciousness(stats_dir=tmp_dir)
        assert jc.outing_number == 1

    def test_forward_compat_adds_missing_keys(self, tmp_dir):
        stats_path = os.path.join(tmp_dir, 'jeeves_lifetime_stats.json')
        minimal = {'total_outings': 5, 'last_outing_date': '', 'outings_today': 0}
        with open(stats_path, 'w') as f:
            json.dump(minimal, f)
        jc = JeevesConsciousness(stats_dir=tmp_dir)
        stats = jc.stats
        assert 'total_cycles' in stats
        assert 'rooms_discovered' in stats


# ===================================================================
# Age calculation
# ===================================================================

class TestAge:
    """Test age calculation from birthday."""

    def test_age_positive(self, tmp_dir):
        jc = JeevesConsciousness(
            stats_dir=tmp_dir,
            birthday='2020-01-01',
        )
        assert jc.age_days > 0

    def test_invalid_birthday_returns_zero(self, tmp_dir):
        jc = JeevesConsciousness(
            stats_dir=tmp_dir,
            birthday='not-a-date',
        )
        assert jc.age_days == 0

    def test_future_birthday_returns_negative(self, tmp_dir):
        jc = JeevesConsciousness(
            stats_dir=tmp_dir,
            birthday='2099-01-01',
        )
        assert jc.age_days < 0


# ===================================================================
# Identity context
# ===================================================================

class TestIdentityContext:
    """Test the compact identity string for LLM prompts."""

    def test_contains_name(self, tmp_dir):
        jc = JeevesConsciousness(stats_dir=tmp_dir, master='Vivek')
        ctx = jc.get_identity_context()
        assert 'Jeeves' in ctx
        assert 'Vivek' in ctx

    def test_contains_outing_number(self, tmp_dir):
        jc = JeevesConsciousness(stats_dir=tmp_dir)
        ctx = jc.get_identity_context()
        assert '#1' in ctx or 'Outing' in ctx

    def test_includes_lifetime_stats(self, tmp_dir):
        jc = JeevesConsciousness(stats_dir=tmp_dir)
        ctx = jc.get_identity_context()
        assert 'LIFETIME' in ctx

    def test_includes_known_rooms_if_any(self, tmp_dir):
        jc = JeevesConsciousness(stats_dir=tmp_dir)
        jc.record_room('kitchen')
        jc.record_room('hallway')
        ctx = jc.get_identity_context()
        assert 'kitchen' in ctx
        assert 'hallway' in ctx


# ===================================================================
# Session intro
# ===================================================================

class TestSessionIntro:
    """Test the TTS greeting."""

    def test_first_outing_greeting(self, tmp_dir):
        jc = JeevesConsciousness(stats_dir=tmp_dir)
        intro = jc.get_session_intro()
        assert 'first outing' in intro.lower() or 'outing' in intro.lower()
        assert 'Sir' in intro or 'Jeeves' in intro

    def test_subsequent_outing_greeting(self, tmp_dir):
        # Create initial outing
        jc1 = JeevesConsciousness(stats_dir=tmp_dir)
        jc1.save()
        jc2 = JeevesConsciousness(stats_dir=tmp_dir)
        intro = jc2.get_session_intro()
        assert 'outing' in intro.lower()


# ===================================================================
# record_cycle
# ===================================================================

class TestRecordCycle:
    """Test per-cycle recording."""

    def test_increments_session_cycles(self, tmp_dir):
        jc = JeevesConsciousness(stats_dir=tmp_dir)
        jc.record_cycle(
            result={'action': 'forward', 'speech': 'Moving.'},
            safety_info={'triggered': False},
            cost=0.01,
        )
        assert jc._session_cycles == 1

    def test_tracks_cost(self, tmp_dir):
        jc = JeevesConsciousness(stats_dir=tmp_dir)
        jc.record_cycle(
            result={'speech': ''},
            safety_info={},
            cost=0.05,
        )
        assert abs(jc._session_cost - 0.05) < 1e-10

    def test_counts_safety_overrides(self, tmp_dir):
        jc = JeevesConsciousness(stats_dir=tmp_dir)
        jc.record_cycle(
            result={'speech': ''},
            safety_info={'triggered': True},
            cost=0.0,
        )
        assert jc._session_safety_overrides == 1

    def test_counts_discoveries(self, tmp_dir):
        jc = JeevesConsciousness(stats_dir=tmp_dir)
        jc.record_cycle(
            result={'speech': 'I see a cat!'},
            safety_info={},
            cost=0.0,
        )
        assert jc._session_discoveries == 1

    def test_no_discovery_without_keywords(self, tmp_dir):
        jc = JeevesConsciousness(stats_dir=tmp_dir)
        jc.record_cycle(
            result={'speech': 'Moving forward.'},
            safety_info={},
            cost=0.0,
        )
        assert jc._session_discoveries == 0

    def test_tracks_distance(self, tmp_dir):
        jc = JeevesConsciousness(stats_dir=tmp_dir)
        jc.record_cycle(
            result={'speech': ''},
            safety_info={},
            cost=0.0,
            distance_delta=1.5,
        )
        assert abs(jc._session_distance - 1.5) < 1e-10


# ===================================================================
# Reflections
# ===================================================================

class TestReflections:
    """Test reflection recording."""

    def test_records_reflection_to_file(self, tmp_dir):
        jc = JeevesConsciousness(stats_dir=tmp_dir)
        jc.record_reflection("The hallway was longer than I expected.")
        assert os.path.exists(jc.reflections_path)
        with open(jc.reflections_path) as f:
            content = f.read()
        assert 'hallway' in content

    def test_empty_reflection_ignored(self, tmp_dir):
        jc = JeevesConsciousness(stats_dir=tmp_dir)
        jc.record_reflection('')
        jc.record_reflection('   ')
        assert not os.path.exists(jc.reflections_path)


# ===================================================================
# Room tracking
# ===================================================================

class TestRoomTracking:
    """Test room discovery tracking."""

    def test_add_room(self, tmp_dir):
        jc = JeevesConsciousness(stats_dir=tmp_dir)
        jc.record_room('kitchen')
        assert 'kitchen' in jc.stats['rooms_discovered']

    def test_no_duplicate_rooms(self, tmp_dir):
        jc = JeevesConsciousness(stats_dir=tmp_dir)
        jc.record_room('kitchen')
        jc.record_room('Kitchen')  # case insensitive
        jc.record_room('kitchen')
        assert jc.stats['rooms_discovered'].count('kitchen') == 1

    def test_empty_room_ignored(self, tmp_dir):
        jc = JeevesConsciousness(stats_dir=tmp_dir)
        jc.record_room('')
        jc.record_room('  ')
        assert len(jc.stats['rooms_discovered']) == 0


# ===================================================================
# Save (idempotent)
# ===================================================================

class TestSave:
    """Test save persistence and idempotency."""

    def test_save_persists_stats(self, tmp_dir):
        jc = JeevesConsciousness(stats_dir=tmp_dir)
        jc.record_cycle(
            result={'speech': ''},
            safety_info={},
            cost=0.05,
            distance_delta=2.0,
        )
        jc.save()
        stats_path = os.path.join(tmp_dir, 'jeeves_lifetime_stats.json')
        with open(stats_path) as f:
            data = json.load(f)
        assert data['total_cost_usd'] >= 0.05
        assert data['total_distance_m'] >= 2.0

    def test_save_is_idempotent(self, tmp_dir):
        jc = JeevesConsciousness(stats_dir=tmp_dir)
        jc.record_cycle(
            result={'speech': ''},
            safety_info={},
            cost=0.10,
        )
        jc.save()
        jc.save()  # second save should not double-count
        stats_path = os.path.join(tmp_dir, 'jeeves_lifetime_stats.json')
        with open(stats_path) as f:
            data = json.load(f)
        assert abs(data['total_cost_usd'] - 0.10) < 0.02


# ===================================================================
# Journal writing
# ===================================================================

class TestJournal:
    """Test end-of-session journal generation."""

    def test_dryrun_journal(self, tmp_dir):
        from autonomous_explorer.llm_provider import DryRunProvider
        from autonomous_explorer.exploration_memory import ExplorationMemory

        jc = JeevesConsciousness(stats_dir=tmp_dir)
        mem = ExplorationMemory(os.path.join(tmp_dir, 'mem.json'))
        mem.total_actions = 15
        provider = DryRunProvider()

        jc.write_journal(mem, provider)

        journals_dir = os.path.join(tmp_dir, 'journals')
        files = os.listdir(journals_dir)
        assert len(files) >= 1
        journal_content = open(os.path.join(journals_dir, files[0])).read()
        assert 'Jeeves' in journal_content
        assert '15' in journal_content
