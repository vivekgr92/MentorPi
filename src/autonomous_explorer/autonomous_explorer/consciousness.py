#!/usr/bin/env python3
# encoding: utf-8
"""
Jeeves Consciousness Layer — persistent identity, lifetime stats, reflections, and journals.

Transforms Jeeves from a stateless navigation agent into a persistent embodied AI
that accumulates knowledge, tracks its lifetime, reflects on experiences, and writes journals.

Lifetime stats persist at ~/mentorpi_explorer/jeeves_lifetime_stats.json.
Reflections are appended per-session to ~/mentorpi_explorer/logs/reflections_*.txt.
Journals are written at end-of-session to ~/mentorpi_explorer/journals/.
"""
import json
import os
import time
from datetime import datetime, date


class JeevesConsciousness:
    """Manages Jeeves' persistent identity, lifetime stats, and reflections."""

    def __init__(self, stats_dir: str, birthday: str = '2026-03-01',
                 master: str = 'Vivek', logger=None):
        self.stats_dir = os.path.expanduser(stats_dir)
        self.birthday = birthday
        self.master = master
        self.logger = logger

        self._stats_path = os.path.join(self.stats_dir, 'jeeves_lifetime_stats.json')
        self._journals_dir = os.path.join(self.stats_dir, 'journals')

        os.makedirs(self.stats_dir, exist_ok=True)
        os.makedirs(self._journals_dir, exist_ok=True)

        # Session reflections file
        session_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        logs_dir = os.path.join(self.stats_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        self._reflections_path = os.path.join(
            logs_dir, f'reflections_{session_ts}.txt'
        )

        # Load or create lifetime stats
        self._stats = self._load_stats()

        # Increment outing counter
        today = date.today().isoformat()
        if self._stats['last_outing_date'] == today:
            self._stats['outings_today'] += 1
        else:
            self._stats['last_outing_date'] = today
            self._stats['outings_today'] = 1
        self._stats['total_outings'] += 1

        # Session-level accumulators (flushed to stats on save)
        self._session_cycles = 0
        self._session_distance = 0.0
        self._session_cost = 0.0
        self._session_discoveries = 0
        self._session_safety_overrides = 0
        self._session_start = time.time()

        self.save()
        self._log(f'Consciousness loaded: outing #{self._stats["total_outings"]}, '
                  f'age {self.age_days} days')

    def _load_stats(self) -> dict:
        """Load lifetime stats from disk, or create defaults."""
        if os.path.exists(self._stats_path):
            try:
                with open(self._stats_path, 'r') as f:
                    stats = json.load(f)
                # Ensure all keys exist (forward compat)
                return self._merge_defaults(stats)
            except (json.JSONDecodeError, OSError):
                pass
        return self._default_stats()

    def _default_stats(self) -> dict:
        return {
            'birthday': self.birthday,
            'master': self.master,
            'total_outings': 0,
            'total_cycles': 0,
            'total_distance_m': 0.0,
            'total_runtime_s': 0,
            'total_cost_usd': 0.0,
            'total_discoveries': 0,
            'total_safety_overrides': 0,
            'rooms_discovered': [],
            'last_outing_date': '',
            'outings_today': 0,
        }

    def _merge_defaults(self, stats: dict) -> dict:
        defaults = self._default_stats()
        for key, val in defaults.items():
            if key not in stats:
                stats[key] = val
        return stats

    @property
    def age_days(self) -> int:
        """Days since Jeeves' birthday."""
        try:
            born = date.fromisoformat(self.birthday)
            return (date.today() - born).days
        except ValueError:
            return 0

    @property
    def outing_number(self) -> int:
        return self._stats['total_outings']

    def get_identity_context(self) -> str:
        """Compact identity string for the user prompt (~50 tokens).

        Injected into the user prompt each cycle so the LLM knows who it is.
        """
        s = self._stats
        age = self.age_days
        lines = [
            f'IDENTITY: You are Jeeves, age {age} days (born {self.birthday}). '
            f'Outing #{s["total_outings"]}. Master: {self.master} ("Sir").',
            f'LIFETIME: {s["total_distance_m"]:.1f}m traveled, '
            f'{s["total_discoveries"]} discoveries, '
            f'{s["total_outings"]} outings.',
        ]
        if s['rooms_discovered']:
            lines.append(f'KNOWN ROOMS: {", ".join(s["rooms_discovered"][:8])}.')
        return '\n'.join(lines)

    def get_session_intro(self) -> str:
        """Greeting for session startup, spoken via TTS."""
        s = self._stats
        age = self.age_days
        today_str = date.today().strftime('%B %d, %Y')
        outing = s['total_outings']

        if outing == 1:
            return (f'Good day, Sir. I am Jeeves. Day {age} of service. '
                    f'{today_str}. Reporting for my very first outing.')
        return (f'Good day, Sir. Outing number {outing}. '
                f'I am {age} days old. Ready to explore.')

    def record_cycle(self, result: dict, safety_info: dict, cost: float,
                     distance_delta: float = 0.0):
        """Update in-memory counters for this cycle."""
        self._session_cycles += 1
        self._session_cost += cost
        self._session_distance += distance_delta
        if safety_info.get('triggered', False):
            self._session_safety_overrides += 1

        # Count discoveries from speech keywords
        speech = result.get('speech', '').lower()
        discovery_keywords = ['see', 'found', 'notice', 'discover', 'spot',
                              'interesting', 'detect', 'observe']
        if any(kw in speech for kw in discovery_keywords):
            self._session_discoveries += 1

    def record_reflection(self, text: str):
        """Append an embodied reflection to the session reflections file."""
        if not text or not text.strip():
            return
        try:
            timestamp = datetime.now().strftime('%H:%M:%S')
            with open(self._reflections_path, 'a') as f:
                f.write(f'[{timestamp}] {text.strip()}\n')
        except OSError:
            pass

    def record_room(self, room_name: str):
        """Add a room to the discovered rooms list if not already there."""
        name = room_name.strip().lower()
        if name and name not in self._stats['rooms_discovered']:
            self._stats['rooms_discovered'].append(name)

    def save(self):
        """Persist lifetime stats to disk, merging session accumulators."""
        s = self._stats
        s['total_cycles'] += self._session_cycles
        s['total_distance_m'] += self._session_distance
        s['total_runtime_s'] += int(time.time() - self._session_start)
        s['total_cost_usd'] += self._session_cost
        s['total_discoveries'] += self._session_discoveries
        s['total_safety_overrides'] += self._session_safety_overrides

        # Reset session accumulators so save() is idempotent
        self._session_cycles = 0
        self._session_distance = 0.0
        self._session_cost = 0.0
        self._session_discoveries = 0
        self._session_safety_overrides = 0
        self._session_start = time.time()

        try:
            with open(self._stats_path, 'w') as f:
                json.dump(s, f, indent=2)
        except OSError as e:
            self._log(f'Failed to save lifetime stats: {e}', warn=True)

    def write_journal(self, memory, llm_provider):
        """Write an end-of-session journal entry using one LLM call.

        Args:
            memory: ExplorationMemory instance with session data.
            llm_provider: LLMProvider instance for the journal generation call.
        """
        if llm_provider.provider_name == 'dryrun':
            self._write_dryrun_journal(memory)
            return

        s = self._stats
        today_str = date.today().strftime('%B %d, %Y')
        outing = s['total_outings']

        # Read reflections if available
        reflections = ''
        if os.path.exists(self._reflections_path):
            try:
                with open(self._reflections_path, 'r') as f:
                    reflections = f.read()
            except OSError:
                pass

        # Recent discoveries
        discoveries_text = ''
        if memory.discoveries:
            recent = memory.discoveries[-10:]
            discoveries_text = '\n'.join(
                f'- {d["description"][:80]}' for d in recent
            )

        prompt = (
            f'You are Jeeves, a butler-robot. Write a brief first-person journal entry '
            f'for Outing #{outing} on {today_str}.\n\n'
            f'Session stats: {memory.total_actions} actions, '
            f'{len(memory.visited_cells)} areas explored.\n\n'
        )
        if discoveries_text:
            prompt += f'Discoveries:\n{discoveries_text}\n\n'
        if reflections:
            prompt += f'My reflections during this outing:\n{reflections[:1500]}\n\n'
        prompt += (
            'Write 3-5 sentences in Jeeves\' voice (polite, witty butler). '
            'End with "Respectfully submitted, Jeeves". '
            'Output ONLY the journal text, no JSON.'
        )

        try:
            # Use the LLM but we just want raw text, not JSON actions
            result = llm_provider.analyze_scene(
                '',  # no image needed
                'You are Jeeves the butler-robot. Write a journal entry. Output plain text only.',
                prompt,
            )
            # The response might be in speech or raw_response
            meta = result.get('_meta', {})
            journal_text = meta.get('raw_response', '')
            if not journal_text:
                journal_text = result.get('speech', str(result))
        except Exception as e:
            journal_text = (
                f'Outing #{outing} — {today_str}\n\n'
                f'I regret that I was unable to compose a proper journal entry. '
                f'Error: {e}\n\n'
                f'Session included {memory.total_actions} actions.\n\n'
                f'Respectfully submitted,\nJeeves'
            )

        self._save_journal(journal_text, outing, today_str)

    def _write_dryrun_journal(self, memory):
        """Write a simple journal in dry-run mode (no LLM call)."""
        s = self._stats
        today_str = date.today().strftime('%B %d, %Y')
        outing = s['total_outings']
        text = (
            f'Outing #{outing} — {today_str}\n\n'
            f'A routine survey in dry-run mode. '
            f'{memory.total_actions} actions were executed, '
            f'{len(memory.visited_cells)} areas explored. '
            f'All systems nominal.\n\n'
            f'Respectfully submitted,\nJeeves'
        )
        self._save_journal(text, outing, today_str)

    def _save_journal(self, text: str, outing: int, date_str: str):
        """Save journal entry to disk."""
        today_file = os.path.join(
            self._journals_dir,
            f'journal_{date.today().isoformat()}.md',
        )
        try:
            header = f'\n## Outing #{outing} — {date_str}\n\n'
            with open(today_file, 'a') as f:
                f.write(header)
                f.write(text.strip())
                f.write('\n\n')
            self._log(f'Journal saved: {today_file}')
        except OSError as e:
            self._log(f'Failed to save journal: {e}', warn=True)

    @property
    def reflections_path(self) -> str:
        return self._reflections_path

    @property
    def stats(self) -> dict:
        return self._stats.copy()

    def _log(self, msg: str, warn: bool = False):
        if self.logger:
            if warn:
                self.logger.warning(msg)
            else:
                self.logger.info(msg)
