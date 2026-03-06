#!/usr/bin/env python3
# encoding: utf-8
"""
Dataset export tool for autonomous explorer logs.

Converts JSONL cycle logs into formats useful for:
  - Imitation learning (observation-action pairs)
  - HuggingFace datasets
  - CSV for pandas/Excel analysis

Usage:
    # Export to CSV
    python3 dataset_export.py ~/mentorpi_explorer/logs/exploration_*.jsonl --format csv

    # Export for imitation learning (observation-action JSONL pairs)
    python3 dataset_export.py ~/mentorpi_explorer/logs/ --format imitation

    # Export as HuggingFace dataset
    python3 dataset_export.py ~/mentorpi_explorer/logs/ --format huggingface

    # All formats at once
    python3 dataset_export.py ~/mentorpi_explorer/logs/ --format all --output-dir ./dataset/
"""
import argparse
import csv
import gzip
import json
import os
import sys
from pathlib import Path


def load_records(paths: list[str]) -> list[dict]:
    """Load all JSONL records from one or more paths (files or dirs)."""
    records = []
    for path in paths:
        p = Path(path)
        files = []
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.glob('*.jsonl')))
            files.extend(sorted(p.glob('*.jsonl.gz')))
        for f in files:
            opener = gzip.open if str(f).endswith('.gz') else open
            with opener(str(f), 'rt') as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
    return records


def export_csv(records: list[dict], output_path: str):
    """Export flattened records to CSV for pandas/Excel."""
    if not records:
        print('No records to export.')
        return

    rows = []
    for r in records:
        is_minimal = 'llm_output' not in r
        if is_minimal:
            row = {
                'timestamp': r.get('timestamp', ''),
                'cycle_id': r.get('cycle_id', ''),
                'action': r.get('parsed_action', ''),
                'speed': r.get('speed', 0),
                'duration': r.get('duration', 0),
                'safety_triggered': r.get('safety_triggered', False),
                'safety_reason': r.get('safety_reason', ''),
                'response_time_ms': r.get('response_time_ms', 0),
            }
        else:
            sensor = r.get('sensor_data', {})
            llm_in = r.get('llm_input', {})
            llm_out = r.get('llm_output', {})
            safety = r.get('safety_override', {})
            execution = r.get('execution', {})
            voice = r.get('voice', {})
            memory = r.get('exploration_memory', {})
            tokens = llm_out.get('tokens_used', {})
            odom = sensor.get('odometry') or {}
            imu = sensor.get('imu') or {}
            orientation = imu.get('orientation') or {}
            lidar_sectors = sensor.get('lidar_sectors') or {}
            motors = execution.get('motor_commands', {})
            servos = execution.get('servo_commands', {})

            row = {
                'timestamp': r.get('timestamp', ''),
                'cycle_id': r.get('cycle_id', ''),
                # Sensor summary
                'lidar_front': lidar_sectors.get('front'),
                'lidar_left': lidar_sectors.get('left'),
                'lidar_right': lidar_sectors.get('right'),
                'lidar_back': lidar_sectors.get('back'),
                'lidar_min': sensor.get('lidar_min_distance'),
                'imu_roll': orientation.get('roll'),
                'imu_pitch': orientation.get('pitch'),
                'imu_yaw': orientation.get('yaw'),
                'odom_x': odom.get('x'),
                'odom_y': odom.get('y'),
                'odom_theta': odom.get('theta'),
                'battery_voltage': sensor.get('battery_voltage'),
                'rgb_frame_path': sensor.get('rgb_frame_path', ''),
                'depth_frame_path': sensor.get('depth_frame_path', ''),
                # LLM
                'provider': llm_in.get('provider', ''),
                'model': llm_in.get('model', ''),
                'image_resolution': llm_in.get('image_resolution', ''),
                # Action
                'parsed_action': llm_out.get('parsed_action', ''),
                'speed': llm_out.get('speed', 0),
                'duration': llm_out.get('duration', 0),
                'speech': llm_out.get('speech', ''),
                'reasoning': llm_out.get('reasoning', ''),
                'response_time_ms': llm_out.get('response_time_ms', 0),
                'tokens_input': tokens.get('input', 0),
                'tokens_output': tokens.get('output', 0),
                'cost_usd': llm_out.get('cost_usd', 0),
                # Safety
                'safety_triggered': safety.get('triggered', False),
                'safety_reason': safety.get('reason', ''),
                'safety_original_action': safety.get('original_action', ''),
                'safety_override_action': safety.get('override_action', ''),
                # Execution
                'actual_action': execution.get('actual_action', ''),
                'motor_linear_mps': motors.get('linear_speed_mps', motors.get('left_speed', 0)),
                'motor_angular_radps': motors.get('angular_speed_radps', motors.get('right_speed', 0)),
                'servo_pan': servos.get('pan', 0),
                'servo_tilt': servos.get('tilt', 0),
                'execution_duration_ms': execution.get('execution_duration_ms', 0),
                # Voice
                'voice_command': voice.get('voice_command_received', ''),
                'speech_output': voice.get('speech_output', ''),
                # Memory
                'total_distance': memory.get('total_distance_traveled', 0),
                'areas_visited': memory.get('areas_visited', 0),
            }
        rows.append(row)

    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'CSV exported: {output_path} ({len(rows)} rows, {len(fieldnames)} columns)')


def export_imitation(records: list[dict], output_path: str):
    """Export observation-action pairs for imitation learning.

    Each record is a (observation, action) pair:
      observation: lidar sectors, depth summary, imu, odom, image path
      action: action name, speed, duration
    """
    pairs = []
    for r in records:
        if 'llm_output' not in r:
            continue  # Skip minimal records

        sensor = r.get('sensor_data', {})
        llm_out = r.get('llm_output', {})
        safety = r.get('safety_override', {})
        execution = r.get('execution', {})

        # Use the actual executed action (after safety overrides)
        actual = execution.get('actual_action', llm_out.get('parsed_action', 'stop'))

        pair = {
            'observation': {
                'rgb_frame_path': sensor.get('rgb_frame_path', ''),
                'depth_frame_path': sensor.get('depth_frame_path', ''),
                'lidar_sectors': sensor.get('lidar_sectors'),
                'lidar_min_distance': sensor.get('lidar_min_distance'),
                'imu': sensor.get('imu'),
                'odometry': sensor.get('odometry'),
            },
            'action': {
                'name': actual,
                'speed': llm_out.get('speed', 0.0),
                'duration': llm_out.get('duration', 0.0),
            },
            'metadata': {
                'timestamp': r.get('timestamp', ''),
                'cycle_id': r.get('cycle_id', 0),
                'safety_override': safety.get('triggered', False),
                'provider': r.get('llm_input', {}).get('provider', ''),
                'reasoning': llm_out.get('reasoning', ''),
            },
        }
        pairs.append(pair)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair, separators=(',', ':')) + '\n')
    print(f'Imitation dataset exported: {output_path} ({len(pairs)} pairs)')


def export_huggingface(records: list[dict], output_dir: str):
    """Export in a HuggingFace-compatible dataset structure.

    Creates:
      output_dir/
        dataset_info.json
        data/
          train.jsonl
    """
    data_dir = os.path.join(output_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Map actions to integer labels
    all_actions = sorted(set(
        r.get('llm_output', {}).get('parsed_action', 'stop')
        for r in records if 'llm_output' in r
    ))
    action_to_id = {a: i for i, a in enumerate(all_actions)}

    hf_records = []
    for r in records:
        if 'llm_output' not in r:
            continue

        sensor = r.get('sensor_data', {})
        llm_out = r.get('llm_output', {})
        execution = r.get('execution', {})
        actual = execution.get('actual_action', llm_out.get('parsed_action', 'stop'))

        lidar_sectors = sensor.get('lidar_sectors') or {}
        odom = sensor.get('odometry') or {}
        imu_data = sensor.get('imu') or {}
        orientation = imu_data.get('orientation') or {}

        hf_record = {
            'cycle_id': r.get('cycle_id', 0),
            'timestamp': r.get('timestamp', ''),
            # Image paths (relative, for loading with datasets.Image)
            'rgb_image': sensor.get('rgb_frame_path', ''),
            'depth_image': sensor.get('depth_frame_path', ''),
            # Numeric features
            'lidar_front': lidar_sectors.get('front', -1.0),
            'lidar_left': lidar_sectors.get('left', -1.0),
            'lidar_right': lidar_sectors.get('right', -1.0),
            'lidar_back': lidar_sectors.get('back', -1.0),
            'lidar_min': sensor.get('lidar_min_distance', -1.0),
            'imu_roll': orientation.get('roll', 0.0),
            'imu_pitch': orientation.get('pitch', 0.0),
            'imu_yaw': orientation.get('yaw', 0.0),
            'odom_x': odom.get('x', 0.0),
            'odom_y': odom.get('y', 0.0),
            'odom_theta': odom.get('theta', 0.0),
            # Labels
            'action': actual,
            'action_id': action_to_id.get(actual, -1),
            'speed': llm_out.get('speed', 0.0),
            'duration': llm_out.get('duration', 0.0),
            # Text
            'reasoning': llm_out.get('reasoning', ''),
            'speech': llm_out.get('speech', ''),
            'provider': r.get('llm_input', {}).get('provider', ''),
        }
        hf_records.append(hf_record)

    # Write train split
    train_path = os.path.join(data_dir, 'train.jsonl')
    with open(train_path, 'w') as f:
        for rec in hf_records:
            f.write(json.dumps(rec, separators=(',', ':')) + '\n')

    # Write dataset info
    info = {
        'description': 'MentorPi autonomous explorer dataset — LLM-guided robot exploration cycles',
        'citation': '',
        'license': 'MIT',
        'features': {
            'cycle_id': {'dtype': 'int32'},
            'timestamp': {'dtype': 'string'},
            'rgb_image': {'dtype': 'string'},
            'depth_image': {'dtype': 'string'},
            'lidar_front': {'dtype': 'float32'},
            'lidar_left': {'dtype': 'float32'},
            'lidar_right': {'dtype': 'float32'},
            'lidar_back': {'dtype': 'float32'},
            'lidar_min': {'dtype': 'float32'},
            'imu_roll': {'dtype': 'float32'},
            'imu_pitch': {'dtype': 'float32'},
            'imu_yaw': {'dtype': 'float32'},
            'odom_x': {'dtype': 'float32'},
            'odom_y': {'dtype': 'float32'},
            'odom_theta': {'dtype': 'float32'},
            'action': {'dtype': 'string', 'class_label': {
                'names': all_actions,
            }},
            'action_id': {'dtype': 'int32'},
            'speed': {'dtype': 'float32'},
            'duration': {'dtype': 'float32'},
            'reasoning': {'dtype': 'string'},
            'speech': {'dtype': 'string'},
            'provider': {'dtype': 'string'},
        },
        'splits': {
            'train': {
                'num_examples': len(hf_records),
                'file': 'data/train.jsonl',
            },
        },
        'action_labels': action_to_id,
        'num_records': len(hf_records),
    }
    info_path = os.path.join(output_dir, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    print(f'HuggingFace dataset exported: {output_dir}/')
    print(f'  {len(hf_records)} records, {len(all_actions)} action classes')
    print(f'  Action labels: {action_to_id}')
    print(f'  Load with: datasets.load_dataset("json", data_files="{train_path}")')


def main():
    parser = argparse.ArgumentParser(
        description='Export autonomous explorer logs to ML-ready formats',
    )
    parser.add_argument(
        'paths', nargs='+',
        help='JSONL file(s) or directory of log files',
    )
    parser.add_argument(
        '--format', '-f',
        choices=['csv', 'imitation', 'huggingface', 'all'],
        default='all',
        help='Export format (default: all)',
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='./explorer_dataset',
        help='Output directory (default: ./explorer_dataset)',
    )
    args = parser.parse_args()

    records = load_records(args.paths)
    if not records:
        print(f'No records found in: {args.paths}')
        sys.exit(1)
    print(f'Loaded {len(records)} records')

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    fmt = args.format

    if fmt in ('csv', 'all'):
        export_csv(records, os.path.join(output_dir, 'explorer_data.csv'))

    if fmt in ('imitation', 'all'):
        export_imitation(records, os.path.join(output_dir, 'imitation_pairs.jsonl'))

    if fmt in ('huggingface', 'all'):
        export_huggingface(records, os.path.join(output_dir, 'huggingface'))

    print(f'\nDone. All exports in: {output_dir}/')


if __name__ == '__main__':
    main()
