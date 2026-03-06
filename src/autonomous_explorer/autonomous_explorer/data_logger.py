#!/usr/bin/env python3
# encoding: utf-8
"""
Asynchronous data logger for the autonomous explorer.

Captures every sense -> think -> act cycle as a JSONL record for later
analysis, fine-tuning, or imitation learning.  Runs entirely on a
background thread so the exploration loop is never blocked by I/O.

Log levels:
  "full"    — JSONL records + RGB/depth frames saved to disk
  "compact" — JSONL records only, no frame files
  "minimal" — Only action and safety-override fields are logged

Storage layout:
  ~/mentorpi_explorer/logs/
    exploration_YYYYMMDD_HHMMSS.jsonl   <- one file per session
    exploration_YYYYMMDD_HHMMSS/
      frames/rgb/cycle_000001.jpg
      frames/depth/cycle_000001.png
    exploration_YYYYMMDD_older.jsonl.gz  <- auto-compressed
"""
import gzip
import json
import os
import queue
import shutil
import threading
import time
from datetime import datetime, timezone

import cv2
import numpy as np


class DataLogger:
    """Background-threaded JSONL cycle logger with optional frame saving."""

    def __init__(
        self,
        log_dir: str,
        log_level: str = 'full',
        flush_interval: int = 5,
        compress_after_hours: float = 24,
        rgb_subdir: str = 'frames/rgb',
        depth_subdir: str = 'frames/depth',
        logger=None,
    ):
        self.log_level = log_level.lower().strip()
        self.flush_interval = flush_interval
        self.compress_after_hours = compress_after_hours
        self.logger = logger

        # Create session directory structure
        self._session_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = os.path.expanduser(log_dir)
        self._session_dir = os.path.join(self.log_dir, f'exploration_{self._session_ts}')
        self._jsonl_path = os.path.join(
            self.log_dir, f'exploration_{self._session_ts}.jsonl',
        )

        self._rgb_dir = ''
        self._depth_dir = ''
        if self.log_level == 'full':
            self._rgb_dir = os.path.join(self._session_dir, rgb_subdir)
            self._depth_dir = os.path.join(self._session_dir, depth_subdir)

        os.makedirs(self.log_dir, exist_ok=True)
        if self._rgb_dir:
            os.makedirs(self._rgb_dir, exist_ok=True)
        if self._depth_dir:
            os.makedirs(self._depth_dir, exist_ok=True)

        # Background writer state
        self._queue: queue.Queue = queue.Queue(maxsize=500)
        self._cycle_counter = 0
        self._file_handle = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._unflushed = 0

        # Session-level accumulators
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0

        self._log_info(f'DataLogger: level={self.log_level}, dir={self.log_dir}')

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Open the JSONL file and start the background writer thread."""
        self._file_handle = open(self._jsonl_path, 'a')
        self._running = True
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
        # Compress old sessions in background
        threading.Thread(target=self._compress_old_sessions, daemon=True).start()
        self._log_info(f'DataLogger started: {self._jsonl_path}')

    def stop(self):
        """Flush remaining records and close the file."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        self._flush()
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
        self._log_info(
            f'DataLogger stopped. {self._cycle_counter} cycles logged. '
            f'Total cost: ${self.total_cost_usd:.4f}'
        )

    # ------------------------------------------------------------------
    # Public API — called from the exploration loop
    # ------------------------------------------------------------------

    def log_cycle(
        self,
        *,
        # Sensor data
        rgb_image: np.ndarray | None = None,
        depth_image: np.ndarray | None = None,
        lidar_ranges: list | None = None,
        lidar_sectors: dict | None = None,
        imu_data: dict | None = None,
        odom_data: dict | None = None,
        battery_voltage: float | None = None,
        # LLM I/O
        provider: str = '',
        model: str = '',
        system_prompt: str = '',
        user_prompt: str = '',
        image_resolution: str = '',
        image_size_bytes: int = 0,
        raw_response: str = '',
        parsed_action: str = '',
        speed: float = 0.0,
        duration: float = 0.0,
        speech: str = '',
        reasoning: str = '',
        response_time_ms: int = 0,
        tokens_input: int = 0,
        tokens_output: int = 0,
        cost_usd: float = 0.0,
        # Safety
        safety_triggered: bool = False,
        safety_reason: str = '',
        safety_original_action: str = '',
        safety_override_action: str = '',
        # Execution
        actual_action: str = '',
        motor_left_speed: float = 0.0,
        motor_right_speed: float = 0.0,
        servo_pan: int = 0,
        servo_tilt: int = 0,
        execution_duration_ms: int = 0,
        # Voice
        voice_command: str | None = None,
        speech_output: str = '',
        # Exploration memory
        total_distance: float = 0.0,
        areas_visited: int = 0,
        objects_discovered: list | None = None,
        map_coverage_pct: float = 0.0,
    ):
        """Enqueue a complete cycle record for background writing."""
        self._cycle_counter += 1
        cycle_id = self._cycle_counter

        # Accumulate token stats
        self.total_input_tokens += tokens_input
        self.total_output_tokens += tokens_output
        self.total_cost_usd += cost_usd

        timestamp = datetime.now(timezone.utc).isoformat()

        # -- Save frames (full mode only) --
        rgb_frame_path = ''
        depth_frame_path = ''
        if self.log_level == 'full':
            frame_name = f'cycle_{cycle_id:06d}'
            if rgb_image is not None:
                rgb_frame_path = os.path.join(self._rgb_dir, f'{frame_name}.jpg')
                self._enqueue_frame_save(rgb_image, rgb_frame_path, 'rgb')
            if depth_image is not None:
                depth_frame_path = os.path.join(self._depth_dir, f'{frame_name}.png')
                self._enqueue_frame_save(depth_image, depth_frame_path, 'depth')

        # -- Build the record --
        if self.log_level == 'minimal':
            record = {
                'timestamp': timestamp,
                'cycle_id': cycle_id,
                'parsed_action': parsed_action or actual_action,
                'speed': speed,
                'duration': duration,
                'safety_triggered': safety_triggered,
                'safety_reason': safety_reason,
                'response_time_ms': response_time_ms,
            }
        else:
            # compact and full share the same JSON structure;
            # full additionally saves frame files (handled above)
            lidar_min_distance = None
            lidar_min_angle = None
            if lidar_ranges:
                valid = [(d, i) for i, d in enumerate(lidar_ranges)
                         if d > 0 and d < float('inf') and d == d]
                if valid:
                    lidar_min_distance, min_idx = min(valid)
                    lidar_min_angle = min_idx  # index, not radians

            record = {
                'timestamp': timestamp,
                'cycle_id': cycle_id,

                'sensor_data': {
                    'rgb_frame_path': rgb_frame_path,
                    'depth_frame_path': depth_frame_path,
                    'lidar_scan': lidar_ranges if self.log_level == 'full' else None,
                    'lidar_sectors': lidar_sectors,
                    'lidar_min_distance': lidar_min_distance,
                    'lidar_min_angle': lidar_min_angle,
                    'imu': imu_data,
                    'odometry': odom_data,
                    'battery_voltage': battery_voltage,
                },

                'llm_input': {
                    'provider': provider,
                    'model': model,
                    'system_prompt': system_prompt if self.log_level == 'full' else '(omitted)',
                    'user_prompt': user_prompt,
                    'image_resolution': image_resolution,
                    'image_size_bytes': image_size_bytes,
                },

                'llm_output': {
                    'raw_response': raw_response,
                    'parsed_action': parsed_action,
                    'speed': speed,
                    'duration': duration,
                    'speech': speech,
                    'reasoning': reasoning,
                    'response_time_ms': response_time_ms,
                    'tokens_used': {
                        'input': tokens_input,
                        'output': tokens_output,
                    },
                    'cost_usd': cost_usd,
                },

                'safety_override': {
                    'triggered': safety_triggered,
                    'reason': safety_reason,
                    'original_action': safety_original_action,
                    'override_action': safety_override_action,
                },

                'execution': {
                    'actual_action': actual_action,
                    'motor_commands': {
                        'left_speed': motor_left_speed,
                        'right_speed': motor_right_speed,
                    },
                    'servo_commands': {
                        'pan': servo_pan,
                        'tilt': servo_tilt,
                    },
                    'execution_duration_ms': execution_duration_ms,
                },

                'voice': {
                    'voice_command_received': voice_command,
                    'speech_output': speech_output,
                },

                'exploration_memory': {
                    'total_distance_traveled': total_distance,
                    'areas_visited': areas_visited,
                    'objects_discovered': objects_discovered or [],
                    'current_map_coverage_pct': map_coverage_pct,
                },
            }

        # Enqueue for background writing (never block the caller)
        try:
            self._queue.put_nowait(('jsonl', record))
        except queue.Full:
            self._log_warn('DataLogger queue full — dropping record')

    # ------------------------------------------------------------------
    # Background writer
    # ------------------------------------------------------------------

    def _writer_loop(self):
        """Drain the queue and write records to disk."""
        while self._running or not self._queue.empty():
            try:
                item_type, payload = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item_type == 'jsonl':
                self._write_jsonl(payload)
            elif item_type == 'frame':
                self._write_frame(payload)

        # Final flush
        self._flush()

    def _write_jsonl(self, record: dict):
        """Append a JSON record to the JSONL file."""
        if not self._file_handle:
            return
        try:
            line = json.dumps(record, default=_json_default, separators=(',', ':'))
            self._file_handle.write(line + '\n')
            self._unflushed += 1
            if self._unflushed >= self.flush_interval:
                self._flush()
        except Exception as e:
            self._log_warn(f'JSONL write error: {e}')

    def _write_frame(self, payload: dict):
        """Save a single image frame to disk."""
        try:
            image = payload['image']
            path = payload['path']
            mode = payload['mode']
            if mode == 'rgb':
                cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            else:
                # depth: save as 16-bit PNG to preserve full precision
                cv2.imwrite(path, image)
        except Exception as e:
            self._log_warn(f'Frame write error: {e}')

    def _enqueue_frame_save(self, image: np.ndarray, path: str, mode: str):
        """Queue a frame for async saving."""
        try:
            self._queue.put_nowait(('frame', {
                'image': image.copy(),
                'path': path,
                'mode': mode,
            }))
        except queue.Full:
            pass  # Drop frame silently rather than block

    def _flush(self):
        """Flush buffered writes to disk."""
        if self._file_handle:
            try:
                self._file_handle.flush()
                os.fsync(self._file_handle.fileno())
            except Exception:
                pass
        self._unflushed = 0

    # ------------------------------------------------------------------
    # Old session compression
    # ------------------------------------------------------------------

    def _compress_old_sessions(self):
        """gzip JSONL files older than compress_after_hours."""
        cutoff = time.time() - self.compress_after_hours * 3600
        try:
            for entry in os.scandir(self.log_dir):
                if not entry.name.endswith('.jsonl'):
                    continue
                if entry.name == os.path.basename(self._jsonl_path):
                    continue  # Skip current session
                if entry.stat().st_mtime < cutoff:
                    gz_path = entry.path + '.gz'
                    if os.path.exists(gz_path):
                        continue
                    self._log_info(f'Compressing old log: {entry.name}')
                    with open(entry.path, 'rb') as f_in:
                        with gzip.open(gz_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.unlink(entry.path)
        except Exception as e:
            self._log_warn(f'Compression error: {e}')

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_ts

    @property
    def jsonl_path(self) -> str:
        return self._jsonl_path

    @property
    def cycle_count(self) -> int:
        return self._cycle_counter

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_info(self, msg: str):
        if self.logger:
            self.logger.info(msg)

    def _log_warn(self, msg: str):
        if self.logger:
            self.logger.warning(msg)


def _json_default(obj):
    """JSON serializer for types not handled by default."""
    if isinstance(obj, float):
        if obj != obj:  # NaN
            return None
        if obj == float('inf') or obj == float('-inf'):
            return None
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
