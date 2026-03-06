#!/usr/bin/env python3
# encoding: utf-8
"""
Analytics dashboard for autonomous explorer logs.

Reads JSONL log files and generates summary statistics, charts, and
a map of the robot's traveled path.

Usage:
    # Analyze a single session
    python3 analytics_dashboard.py ~/mentorpi_explorer/logs/exploration_20260301_143000.jsonl

    # Analyze all sessions in a directory
    python3 analytics_dashboard.py ~/mentorpi_explorer/logs/

    # Save charts to PNG files instead of showing
    python3 analytics_dashboard.py ~/mentorpi_explorer/logs/ --save-dir ./charts/

    # Text-only mode (no matplotlib required)
    python3 analytics_dashboard.py ~/mentorpi_explorer/logs/ --text-only
"""
import argparse
import gzip
import json
import os
import sys
from collections import Counter
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    """Load records from a JSONL file (supports .jsonl and .jsonl.gz)."""
    records = []
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_all_sessions(path: str) -> dict[str, list[dict]]:
    """Load one or many sessions from a file or directory path."""
    sessions = {}
    p = Path(path)
    if p.is_file():
        sessions[p.stem] = load_jsonl(str(p))
    elif p.is_dir():
        for f in sorted(p.iterdir()):
            if f.name.endswith('.jsonl') or f.name.endswith('.jsonl.gz'):
                records = load_jsonl(str(f))
                if records:
                    sessions[f.stem] = records
    return sessions


def print_summary(name: str, records: list[dict]):
    """Print text summary statistics for a session."""
    if not records:
        print(f'\n  {name}: (empty)')
        return

    n = len(records)
    # Check if minimal format
    is_minimal = 'llm_output' not in records[0]

    first_ts = records[0].get('timestamp', '')
    last_ts = records[-1].get('timestamp', '')

    # Action distribution
    if is_minimal:
        actions = [r.get('parsed_action', 'unknown') for r in records]
        response_times = [r.get('response_time_ms', 0) for r in records]
        safety_count = sum(1 for r in records if r.get('safety_triggered'))
    else:
        actions = [r.get('llm_output', {}).get('parsed_action', 'unknown')
                   for r in records]
        response_times = [r.get('llm_output', {}).get('response_time_ms', 0)
                          for r in records]
        safety_count = sum(
            1 for r in records
            if r.get('safety_override', {}).get('triggered')
        )

    action_counts = Counter(actions)
    avg_response = sum(response_times) / len(response_times) if response_times else 0

    # Cost tracking
    total_cost = 0.0
    total_input_tok = 0
    total_output_tok = 0
    if not is_minimal:
        for r in records:
            llm_out = r.get('llm_output', {})
            total_cost += llm_out.get('cost_usd', 0.0)
            tok = llm_out.get('tokens_used', {})
            total_input_tok += tok.get('input', 0)
            total_output_tok += tok.get('output', 0)

    # Providers used
    providers = set()
    if not is_minimal:
        for r in records:
            p = r.get('llm_input', {}).get('provider', '')
            if p:
                providers.add(p)

    # Discoveries
    discoveries = []
    if not is_minimal:
        for r in records:
            mem = r.get('exploration_memory', {})
            objs = mem.get('objects_discovered', [])
            for obj in objs:
                if obj not in discoveries:
                    discoveries.append(obj)

    # Path distance from odometry
    total_distance = 0.0
    if not is_minimal:
        prev_x, prev_y = None, None
        for r in records:
            odom = r.get('sensor_data', {}).get('odometry')
            if odom and 'x' in odom and 'y' in odom:
                x, y = odom['x'], odom['y']
                if prev_x is not None:
                    dx = x - prev_x
                    dy = y - prev_y
                    total_distance += (dx * dx + dy * dy) ** 0.5
                prev_x, prev_y = x, y

    print(f'\n{"=" * 60}')
    print(f'  Session: {name}')
    print(f'{"=" * 60}')
    print(f'  Time range      : {first_ts} → {last_ts}')
    print(f'  Total cycles     : {n}')
    print(f'  Distance traveled: {total_distance:.2f} m')
    print(f'  Avg response time: {avg_response:.0f} ms')
    print(f'  Safety overrides : {safety_count} ({100*safety_count/n:.1f}%)')
    if providers:
        print(f'  LLM providers    : {", ".join(sorted(providers))}')
    print(f'  Total tokens     : {total_input_tok:,} in / {total_output_tok:,} out')
    print(f'  Total cost       : ${total_cost:.4f}')
    print()
    print('  Action distribution:')
    for action, count in action_counts.most_common():
        bar = '#' * int(40 * count / n)
        print(f'    {action:15s} {count:4d} ({100*count/n:5.1f}%) {bar}')
    if discoveries:
        print(f'\n  Discoveries ({len(discoveries)}):')
        for d in discoveries[-10:]:
            print(f'    - {d}')
    print()


def plot_charts(
    sessions: dict[str, list[dict]],
    save_dir: str | None = None,
):
    """Generate matplotlib charts for the sessions."""
    try:
        import matplotlib
        if save_dir:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed. Use --text-only or: pip install matplotlib')
        return

    # Merge all records for aggregate analysis
    all_records = []
    for records in sessions.values():
        all_records.extend(records)
    if not all_records:
        return

    is_minimal = 'llm_output' not in all_records[0]
    if is_minimal:
        actions = [r.get('parsed_action', '?') for r in all_records]
        response_times = [r.get('response_time_ms', 0) for r in all_records]
    else:
        actions = [r.get('llm_output', {}).get('parsed_action', '?')
                   for r in all_records]
        response_times = [r.get('llm_output', {}).get('response_time_ms', 0)
                          for r in all_records]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Explorer Analytics — {len(all_records)} cycles across '
                 f'{len(sessions)} session(s)', fontsize=14)

    # 1. Action distribution pie chart
    ax = axes[0, 0]
    action_counts = Counter(actions)
    labels = list(action_counts.keys())
    sizes = list(action_counts.values())
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title('Action Distribution')

    # 2. Response time histogram
    ax = axes[0, 1]
    valid_times = [t for t in response_times if t > 0]
    if valid_times:
        ax.hist(valid_times, bins=30, color='steelblue', edgecolor='white')
        ax.axvline(sum(valid_times) / len(valid_times), color='red',
                   linestyle='--', label=f'Mean: {sum(valid_times)/len(valid_times):.0f}ms')
        ax.legend()
    ax.set_xlabel('Response Time (ms)')
    ax.set_ylabel('Count')
    ax.set_title('LLM Response Time')

    # 3. Cost over time / safety overrides
    ax = axes[1, 0]
    if not is_minimal:
        cumulative_cost = []
        safety_points_x = []
        safety_points_y = []
        running_cost = 0.0
        for i, r in enumerate(all_records):
            c = r.get('llm_output', {}).get('cost_usd', 0.0)
            running_cost += c
            cumulative_cost.append(running_cost)
            if r.get('safety_override', {}).get('triggered'):
                safety_points_x.append(i)
                safety_points_y.append(running_cost)
        ax.plot(cumulative_cost, color='green', label='Cumulative cost')
        if safety_points_x:
            ax.scatter(safety_points_x, safety_points_y, color='red',
                       marker='x', s=40, label='Safety override', zorder=5)
        ax.set_xlabel('Cycle')
        ax.set_ylabel('Cost (USD)')
        ax.legend()
    ax.set_title('Cost & Safety Overrides')

    # 4. Path map from odometry
    ax = axes[1, 1]
    if not is_minimal:
        xs, ys = [], []
        for r in all_records:
            odom = r.get('sensor_data', {}).get('odometry')
            if odom and 'x' in odom and 'y' in odom:
                xs.append(odom['x'])
                ys.append(odom['y'])
        if xs:
            ax.plot(xs, ys, 'b-', alpha=0.6, linewidth=1)
            ax.plot(xs[0], ys[0], 'go', markersize=10, label='Start')
            ax.plot(xs[-1], ys[-1], 'rs', markersize=10, label='End')
            ax.set_aspect('equal')
            ax.legend()
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Path Traveled')

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, 'analytics_dashboard.png')
        plt.savefig(path, dpi=150)
        print(f'Charts saved to {path}')
    else:
        plt.show()

    # Provider comparison if multiple providers
    if not is_minimal:
        provider_data = {}
        for r in all_records:
            prov = r.get('llm_input', {}).get('provider', 'unknown')
            if prov not in provider_data:
                provider_data[prov] = {
                    'times': [], 'costs': [], 'actions': [], 'count': 0,
                }
            pd = provider_data[prov]
            pd['count'] += 1
            pd['times'].append(r.get('llm_output', {}).get('response_time_ms', 0))
            pd['costs'].append(r.get('llm_output', {}).get('cost_usd', 0.0))
            pd['actions'].append(r.get('llm_output', {}).get('parsed_action', '?'))

        if len(provider_data) > 1:
            fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
            fig2.suptitle('Provider Comparison', fontsize=14)

            providers_list = list(provider_data.keys())

            # Response time comparison
            ax = axes2[0]
            data_times = [provider_data[p]['times'] for p in providers_list]
            ax.boxplot(data_times, labels=providers_list)
            ax.set_ylabel('Response Time (ms)')
            ax.set_title('Response Time by Provider')

            # Action distribution comparison
            ax = axes2[1]
            width = 0.35
            all_actions = sorted(set(actions))
            x_pos = list(range(len(all_actions)))
            for i, prov in enumerate(providers_list):
                counts = Counter(provider_data[prov]['actions'])
                vals = [counts.get(a, 0) for a in all_actions]
                offset = width * (i - len(providers_list) / 2 + 0.5)
                ax.bar([x + offset for x in x_pos], vals, width,
                       label=prov)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(all_actions, rotation=45, ha='right')
            ax.set_ylabel('Count')
            ax.set_title('Actions by Provider')
            ax.legend()

            plt.tight_layout()
            if save_dir:
                path = os.path.join(save_dir, 'provider_comparison.png')
                plt.savefig(path, dpi=150)
                print(f'Provider comparison saved to {path}')
            else:
                plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Analytics dashboard for autonomous explorer logs',
    )
    parser.add_argument(
        'path',
        help='Path to a .jsonl file or directory of log files',
    )
    parser.add_argument(
        '--save-dir',
        help='Save charts as PNG to this directory instead of showing',
    )
    parser.add_argument(
        '--text-only',
        action='store_true',
        help='Print text summary only, skip charts',
    )
    args = parser.parse_args()

    sessions = load_all_sessions(args.path)
    if not sessions:
        print(f'No log files found at: {args.path}')
        sys.exit(1)

    # Print text summaries
    for name, records in sessions.items():
        print_summary(name, records)

    # Aggregate summary
    total_records = sum(len(r) for r in sessions.values())
    total_cost = 0.0
    for records in sessions.values():
        for r in records:
            total_cost += r.get('llm_output', {}).get('cost_usd', 0.0)
    print(f'{"=" * 60}')
    print(f'  TOTAL: {total_records} cycles across {len(sessions)} session(s)')
    print(f'  Total API cost: ${total_cost:.4f}')
    print(f'{"=" * 60}')

    if not args.text_only:
        plot_charts(sessions, save_dir=args.save_dir)


if __name__ == '__main__':
    main()
