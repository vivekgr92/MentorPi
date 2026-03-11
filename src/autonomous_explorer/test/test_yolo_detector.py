#!/usr/bin/env python3
"""Tests for YoloDetector."""
import numpy as np
import pytest

from autonomous_explorer.yolo_detector import Detection, YoloDetector


class TestDetection:
    def test_detection_dataclass(self):
        d = Detection(
            label='cup', confidence=0.85,
            bbox=(10, 20, 100, 200),
            position='center', distance_m=1.5, size='medium',
        )
        assert d.label == 'cup'
        assert d.confidence == 0.85
        assert d.position == 'center'
        assert d.distance_m == 1.5

    def test_detection_defaults(self):
        d = Detection(label='cat', confidence=0.5, bbox=(0, 0, 50, 50))
        assert d.position == ''
        assert d.distance_m == -1
        assert d.size == ''


class TestDetectionsToText:
    def test_empty(self):
        assert YoloDetector.detections_to_text([]) == 'DETECTED OBJECTS: none visible'

    def test_single_detection(self):
        dets = [Detection('bottle', 0.9, (0, 0, 50, 50), 'center', 1.2, 'small')]
        text = YoloDetector.detections_to_text(dets)
        assert 'DETECTED OBJECTS (1)' in text
        assert 'bottle' in text
        assert '90%' in text
        assert '1.2m' in text

    def test_multiple_detections(self):
        dets = [
            Detection('person', 0.95, (0, 0, 200, 400), 'center', 2.0, 'large'),
            Detection('chair', 0.7, (300, 100, 400, 300), 'right', -1, 'medium'),
        ]
        text = YoloDetector.detections_to_text(dets)
        assert 'DETECTED OBJECTS (2)' in text
        assert 'person' in text
        assert 'chair' in text
        # No distance for chair
        assert 'right' in text

    def test_no_distance_omitted(self):
        dets = [Detection('dog', 0.8, (0, 0, 50, 50), 'left', -1, 'small')]
        text = YoloDetector.detections_to_text(dets)
        assert 'away' not in text


class TestDetectionsToDict:
    def test_conversion(self):
        dets = [Detection('cup', 0.85, (10, 20, 100, 200), 'center', 1.5, 'medium')]
        result = YoloDetector.detections_to_dict(dets)
        assert len(result) == 1
        assert result[0]['name'] == 'cup'
        assert result[0]['confidence'] == 0.85
        assert result[0]['distance'] == '1.50m'
        assert result[0]['position'] == 'center'

    def test_unknown_distance(self):
        dets = [Detection('cat', 0.5, (0, 0, 50, 50), 'left', -1, 'small')]
        result = YoloDetector.detections_to_dict(dets)
        assert result[0]['distance'] == 'unknown'


class TestEstimateDistance:
    def test_valid_depth(self):
        # 100x100 depth image, all 2000mm = 2.0m
        depth = np.full((100, 100), 2000, dtype=np.uint16)
        dist = YoloDetector._estimate_distance(depth, 20, 20, 80, 80, 100, 100)
        assert abs(dist - 2.0) < 0.1

    def test_zero_depth_returns_negative(self):
        depth = np.zeros((100, 100), dtype=np.uint16)
        dist = YoloDetector._estimate_distance(depth, 20, 20, 80, 80, 100, 100)
        assert dist == -1.0

    def test_scaled_coordinates(self):
        # RGB is 640x480, depth is 320x240
        depth = np.full((240, 320), 3000, dtype=np.uint16)
        dist = YoloDetector._estimate_distance(depth, 100, 100, 300, 300, 480, 640)
        assert abs(dist - 3.0) < 0.1


class TestYoloDetectorInit:
    def test_not_loaded_initially(self):
        det = YoloDetector(model_path='/nonexistent/model.onnx')
        assert not det.is_loaded
        assert det.last_inference_ms == 0.0

    def test_missing_model_raises(self):
        det = YoloDetector(model_path='/nonexistent/model.onnx')
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        with pytest.raises(FileNotFoundError):
            det.detect(dummy)


class TestGuessRoom:
    """Test the _guess_room helper in ToolHandlers."""

    def test_kitchen(self):
        from autonomous_explorer.tool_handlers import ToolHandlers
        assert ToolHandlers._guess_room(['oven', 'chair']) == 'kitchen'
        assert ToolHandlers._guess_room(['refrigerator', 'cup']) == 'kitchen'

    def test_bedroom(self):
        from autonomous_explorer.tool_handlers import ToolHandlers
        assert ToolHandlers._guess_room(['bed', 'lamp']) == 'bedroom'

    def test_office(self):
        from autonomous_explorer.tool_handlers import ToolHandlers
        assert ToolHandlers._guess_room(['laptop', 'chair']) == 'office'

    def test_unknown(self):
        from autonomous_explorer.tool_handlers import ToolHandlers
        assert ToolHandlers._guess_room(['bottle', 'backpack']) == 'unknown'
