#!/usr/bin/env python3
# encoding: utf-8
"""
Jeeves Knowledge Browser — standalone CLI tool (no ROS2 dependency).

Usage:
    python3 jeeves_knowledge.py              # summary dashboard
    python3 jeeves_knowledge.py rooms        # spatial knowledge
    python3 jeeves_knowledge.py objects      # known objects
    python3 jeeves_knowledge.py stats        # lifetime stats
    python3 jeeves_knowledge.py journal      # today's journal
    python3 jeeves_knowledge.py reflections  # today's reflections
    python3 jeeves_knowledge.py lessons      # learned behaviors
"""
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

DATA_DIR = os.path.expanduser(
    os.environ.get('JEEVES_DATA_DIR', '~/mentorpi_explorer')
)
STATS_FILE = os.path.join(DATA_DIR, 'jeeves_lifetime_stats.json')
KNOWLEDGE_DIR = os.path.join(DATA_DIR, 'knowledge')
JOURNALS_DIR = os.path.join(DATA_DIR, 'journals')
LOGS_DIR = os.path.join(DATA_DIR, 'logs')


def load_json(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def cmd_stats():
    stats = load_json(STATS_FILE)
    if not stats:
        print('No lifetime stats found.')
        return

    birthday = stats.get('birthday', '2026-03-01')
    try:
        age = (date.today() - date.fromisoformat(birthday)).days
    except ValueError:
        age = 0

    print(f'=== Jeeves Lifetime Stats ===')
    print(f'  Birthday:       {birthday} ({age} days old)')
    print(f'  Master:         {stats.get("master", "Unknown")}')
    print(f'  Total outings:  {stats.get("total_outings", 0)}')
    print(f'  Total cycles:   {stats.get("total_cycles", 0)}')
    print(f'  Distance:       {stats.get("total_distance_m", 0):.1f} m')
    print(f'  Runtime:        {stats.get("total_runtime_s", 0) / 60:.1f} min')
    print(f'  API cost:       ${stats.get("total_cost_usd", 0):.4f}')
    print(f'  Discoveries:    {stats.get("total_discoveries", 0)}')
    print(f'  Safety stops:   {stats.get("total_safety_overrides", 0)}')
    rooms = stats.get('rooms_discovered', [])
    if rooms:
        print(f'  Rooms:          {", ".join(rooms)}')


def cmd_rooms():
    data = load_json(os.path.join(KNOWLEDGE_DIR, 'world_map.json'))
    rooms = data.get('rooms', {})
    if not rooms:
        print('No rooms discovered yet.')
        return
    print(f'=== Spatial Knowledge ({len(rooms)} rooms) ===')
    for name, info in sorted(rooms.items()):
        visits = info.get('times_visited', 0)
        desc = info.get('description', '')[:60]
        conns = info.get('connections', [])
        conf = info.get('confidence', 0)
        print(f'\n  {name} (visited {visits}x, confidence {conf:.0%})')
        if desc:
            print(f'    {desc}')
        if conns:
            print(f'    Connects to: {", ".join(conns)}')
        landmarks = info.get('landmarks', [])
        if landmarks:
            print(f'    Landmarks: {", ".join(landmarks)}')


def cmd_objects():
    data = load_json(os.path.join(KNOWLEDGE_DIR, 'known_objects.json'))
    objects = data.get('objects', {})
    if not objects:
        print('No objects catalogued yet.')
        return
    print(f'=== Known Objects ({len(objects)}) ===')
    for name, info in sorted(objects.items()):
        cat = info.get('category', 'unknown')
        seen = info.get('times_seen', 0)
        loc = info.get('usual_location', '?')
        dynamic = ' [DYNAMIC]' if info.get('is_dynamic') else ''
        print(f'  {name} [{cat}]{dynamic}: seen {seen}x, at {loc}')


def cmd_lessons():
    data = load_json(os.path.join(KNOWLEDGE_DIR, 'learned_behaviors.json'))
    lessons = data.get('navigation_lessons', [])
    surfaces = data.get('surface_types', {})
    timing = data.get('timing_patterns', [])

    if not lessons and not surfaces and not timing:
        print('No learned behaviors yet.')
        return

    if lessons:
        print(f'=== Navigation Lessons ({len(lessons)}) ===')
        for l in lessons:
            conf = l.get('confidence', 0)
            learned = l.get('learned_on', '?')
            print(f'  [{conf:.0%}] {l.get("lesson", "")} (learned {learned})')

    if surfaces:
        print(f'\n=== Surface Types ({len(surfaces)}) ===')
        for name, info in surfaces.items():
            traction = info.get('traction', '?')
            speed = info.get('speed_safe', '?')
            print(f'  {name}: traction={traction}, safe_speed={speed}')

    if timing:
        print(f'\n=== Timing Patterns ({len(timing)}) ===')
        for t in timing:
            print(f'  [{t.get("confidence", 0):.0%}] {t.get("observation", "")}')


def cmd_journal():
    today_file = os.path.join(JOURNALS_DIR, f'journal_{date.today().isoformat()}.md')
    if os.path.exists(today_file):
        print(f'=== Journal for {date.today()} ===')
        with open(today_file, 'r') as f:
            print(f.read())
    else:
        # Show most recent journal
        if not os.path.exists(JOURNALS_DIR):
            print('No journals yet.')
            return
        journals = sorted(Path(JOURNALS_DIR).glob('journal_*.md'), reverse=True)
        if journals:
            print(f'=== Most recent journal: {journals[0].name} ===')
            print(journals[0].read_text())
        else:
            print('No journals yet.')


def cmd_reflections():
    if not os.path.exists(LOGS_DIR):
        print('No reflections yet.')
        return
    today_prefix = f'reflections_{date.today().strftime("%Y%m%d")}'
    files = sorted(
        [f for f in Path(LOGS_DIR).glob('reflections_*.txt')
         if f.name.startswith(today_prefix)],
        reverse=True,
    )
    if not files:
        # Show most recent
        files = sorted(Path(LOGS_DIR).glob('reflections_*.txt'), reverse=True)
    if files:
        print(f'=== Reflections: {files[0].name} ===')
        print(files[0].read_text())
    else:
        print('No reflections yet.')


def cmd_summary():
    cmd_stats()
    print()

    data = load_json(os.path.join(KNOWLEDGE_DIR, 'world_map.json'))
    rooms = data.get('rooms', {})
    objects = load_json(os.path.join(KNOWLEDGE_DIR, 'known_objects.json')).get('objects', {})
    lessons = load_json(os.path.join(KNOWLEDGE_DIR, 'learned_behaviors.json')).get('navigation_lessons', [])

    print(f'Knowledge: {len(rooms)} rooms, {len(objects)} objects, {len(lessons)} lessons')

    if os.path.exists(JOURNALS_DIR):
        journal_count = len(list(Path(JOURNALS_DIR).glob('journal_*.md')))
        print(f'Journals: {journal_count} entries')


def main():
    commands = {
        'stats': cmd_stats,
        'rooms': cmd_rooms,
        'objects': cmd_objects,
        'lessons': cmd_lessons,
        'journal': cmd_journal,
        'reflections': cmd_reflections,
    }

    if len(sys.argv) < 2:
        cmd_summary()
        return

    cmd = sys.argv[1].lower()
    if cmd in commands:
        commands[cmd]()
    elif cmd in ('-h', '--help', 'help'):
        print(__doc__)
    else:
        print(f'Unknown command: {cmd}')
        print(f'Available: {", ".join(commands.keys())}')


if __name__ == '__main__':
    main()
