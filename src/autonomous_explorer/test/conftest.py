"""Shared fixtures for autonomous_explorer tests."""
import json
import os
import tempfile

import numpy as np
import pytest


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_llm_response():
    """Standard LLM JSON response for testing."""
    return {
        'action': 'forward',
        'speed': 0.5,
        'duration': 2.0,
        'speech': 'I see a hallway ahead, moving forward.',
        'reasoning': 'Front LiDAR shows 2.0m clearance.',
        'embodied_reflection': 'The path looks inviting.',
    }


@pytest.fixture
def sample_odom():
    """Sample odometry data."""
    return {'x': 1.0, 'y': 2.0, 'theta': 0.5}


@pytest.fixture
def sample_lidar_sectors():
    """Sample LiDAR sector distances."""
    return {
        'front': 1.5,
        'left': 0.8,
        'right': 2.0,
        'back': 3.0,
    }


@pytest.fixture
def sample_rgb_image():
    """640x480 RGB image for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_depth_image():
    """640x480 uint16 depth image (mm)."""
    return np.random.randint(200, 4000, (480, 640), dtype=np.uint16)
