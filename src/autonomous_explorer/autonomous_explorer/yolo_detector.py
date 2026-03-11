#!/usr/bin/env python3
# encoding: utf-8
"""
YOLO11n local object detector for Jeeves.

Runs YOLO11n (ONNX) on Pi 5 CPU to detect objects in camera frames.
Returns structured detection results with depth-fused distance estimates.
Eliminates the need to send base64 images to the LLM for object detection.

Usage:
    detector = YoloDetector()
    detections = detector.detect(rgb_frame, depth_frame)
    text = detector.detections_to_text(detections)
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default model path (relative to package)
_DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'models'
)
DEFAULT_MODEL_PATH = os.path.join(_DEFAULT_MODEL_DIR, 'yolo11n.onnx')


@dataclass
class Detection:
    """A single detected object with spatial context."""
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    position: str = ''      # 'left', 'center', 'right'
    distance_m: float = -1  # meters from depth camera, -1 if unknown
    size: str = ''           # 'small', 'medium', 'large' based on bbox area


class YoloDetector:
    """Lazy-loaded YOLO11n detector using ultralytics + ONNX Runtime.

    Model is loaded on first detect() call (~2s startup, ~200MB RAM).
    Subsequent calls run in ~200ms on Pi 5.
    """

    def __init__(
        self,
        model_path: str = '',
        confidence_threshold: float = 0.4,
        max_detections: int = 15,
        input_size: int = 640,
    ):
        self._model_path = model_path or DEFAULT_MODEL_PATH
        self._conf_threshold = confidence_threshold
        self._max_detections = max_detections
        self._input_size = input_size
        self._model = None
        self._last_inference_ms: float = 0.0

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def last_inference_ms(self) -> float:
        return self._last_inference_ms

    def _load_model(self):
        """Lazy-load the YOLO model."""
        if self._model is not None:
            return

        if not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f'YOLO model not found: {self._model_path}. '
                f'Run: cd /tmp && python3 -c "from ultralytics import YOLO; '
                f"YOLO('yolo11n.pt').export(format='onnx')\" "
                f'then copy yolo11n.onnx to {os.path.dirname(self._model_path)}/'
            )

        from ultralytics import YOLO
        t0 = time.time()
        self._model = YOLO(self._model_path, task='detect')
        # Warmup with a dummy frame
        dummy = np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8)
        self._model(dummy, verbose=False)
        logger.info(
            f'YOLO model loaded in {time.time() - t0:.1f}s: {self._model_path}'
        )

    def detect(
        self,
        rgb_frame: np.ndarray,
        depth_frame: np.ndarray | None = None,
    ) -> list[Detection]:
        """Run detection on an RGB frame.

        Args:
            rgb_frame: BGR or RGB uint8 image from camera.
            depth_frame: Optional uint16 depth image (mm). Same resolution
                         as rgb_frame or will be scaled. Used for distance
                         estimation at each detection's center.

        Returns:
            List of Detection objects, sorted by confidence (highest first).
        """
        self._load_model()

        t0 = time.time()
        results = self._model(
            rgb_frame,
            conf=self._conf_threshold,
            verbose=False,
            imgsz=self._input_size,
        )
        self._last_inference_ms = (time.time() - t0) * 1000

        if not results or len(results[0].boxes) == 0:
            return []

        h, w = rgb_frame.shape[:2]
        boxes = results[0].boxes
        detections: list[Detection] = []

        for i in range(min(len(boxes), self._max_detections)):
            box = boxes[i]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = self._model.names.get(cls_id, f'class_{cls_id}')

            # Position: left/center/right based on bbox center x
            cx = (x1 + x2) / 2
            if cx < w / 3:
                position = 'left'
            elif cx > 2 * w / 3:
                position = 'right'
            else:
                position = 'center'

            # Size estimate based on bbox area relative to frame
            bbox_area = (x2 - x1) * (y2 - y1)
            frame_area = h * w
            ratio = bbox_area / frame_area if frame_area > 0 else 0
            if ratio > 0.15:
                size = 'large'
            elif ratio > 0.03:
                size = 'medium'
            else:
                size = 'small'

            # Distance from depth camera
            distance_m = -1.0
            if depth_frame is not None:
                distance_m = self._estimate_distance(
                    depth_frame, x1, y1, x2, y2, h, w)

            detections.append(Detection(
                label=label,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                position=position,
                distance_m=round(distance_m, 2) if distance_m > 0 else -1,
                size=size,
            ))

        # Sort by confidence
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    @staticmethod
    def _estimate_distance(
        depth_frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        rgb_h: int, rgb_w: int,
    ) -> float:
        """Estimate distance to an object using depth image.

        Samples a small ROI at the bbox center in the depth image.
        Returns distance in meters, or -1 if no valid depth.
        """
        dh, dw = depth_frame.shape[:2]

        # Scale bbox coords if depth and RGB have different resolutions
        sx = dw / rgb_w if rgb_w > 0 else 1
        sy = dh / rgb_h if rgb_h > 0 else 1

        cx = int((x1 + x2) / 2 * sx)
        cy = int((y1 + y2) / 2 * sy)

        # Sample a small ROI around center
        r = max(5, min(int((x2 - x1) * sx * 0.15), 20))
        y_lo = max(0, cy - r)
        y_hi = min(dh, cy + r)
        x_lo = max(0, cx - r)
        x_hi = min(dw, cx + r)

        roi = depth_frame[y_lo:y_hi, x_lo:x_hi]
        if roi.size == 0:
            return -1.0

        # Filter valid depth values (uint16 mm, 0 = invalid, max ~40m)
        valid = roi[(roi > 0) & (roi < 40000)]
        if len(valid) == 0:
            return -1.0

        # Use median for robustness
        return float(np.median(valid)) / 1000.0

    @staticmethod
    def detections_to_text(detections: list[Detection]) -> str:
        """Format detections as concise text for LLM context.

        Example output:
            DETECTED OBJECTS (5):
            - person (92%) center, 1.8m away, large
            - chair (78%) left, 2.3m away, medium
            - bottle (65%) right, 0.9m away, small
        """
        if not detections:
            return 'DETECTED OBJECTS: none visible'

        lines = [f'DETECTED OBJECTS ({len(detections)}):']
        for d in detections:
            parts = [f'{d.label} ({d.confidence:.0%})']
            parts.append(d.position)
            if d.distance_m > 0:
                parts.append(f'{d.distance_m:.1f}m away')
            parts.append(d.size)
            lines.append(f'  - {", ".join(parts)}')

        return '\n'.join(lines)

    @staticmethod
    def detections_to_dict(detections: list[Detection]) -> list[dict]:
        """Convert detections to JSON-serializable dicts for tool results."""
        return [
            {
                'name': d.label,
                'confidence': round(d.confidence, 2),
                'position': d.position,
                'distance': f'{d.distance_m:.2f}m' if d.distance_m > 0 else 'unknown',
                'size': d.size,
            }
            for d in detections
        ]
