# tests/test_display.py
"""Tests for src/eyesee/vision/display."""
# pylint: disable=redefined-outer-name

import math
import pytest

from eyesee.vision.display import Display, cm_to_deg, deg_to_cm


## Standalone function tests ––––––––––––––––––––––––––––––
def test_cm_to_deg_known_value():
    """1 cm at 57.3 cm distance ≈ 1 degree (the classic approximation)."""
    result = cm_to_deg(1.0, 57.29577951)
    assert math.isclose(result, 1.0, abs_tol=1e-4)

def test_deg_to_cm_inverts_cm_to_deg():
    """Round-tripping through both functions recovers the original value."""
    original_cm = 3.5
    distance = 60.0
    degrees = cm_to_deg(original_cm, distance)
    recovered = deg_to_cm(degrees, distance)
    assert math.isclose(recovered, original_cm, abs_tol=1e-9)

def test_cm_to_deg_zero_size_is_zero():
    """Zero physical size maps to zero visual angle."""
    assert cm_to_deg(0.0, 57.0) == 0.0


## Display property tests –––––––––––––––––––––––––––––––––
@pytest.fixture
def standard_display() -> Display:
    """A 1920x1080 display, 53x30 cm, viewed from 57 cm."""
    return Display(
        resolution=(1920, 1080),
        size_cm=(53.0, 30.0),
        viewing_distance_cm=57.0,
    )

def test_pix_per_cm(standard_display):
    """Pixel density matches resolution / physical size."""
    h, v = standard_display.pix_per_cm
    assert math.isclose(h, 1920 / 53.0)
    assert math.isclose(v, 1080 / 30.0)

def test_pix_to_deg_round_trips(standard_display):
    """Converting pixels -> deg -> pixels recovers the original."""
    original_px = 100.0
    deg = standard_display.pix_to_deg(original_px)
    recovered = standard_display.deg_to_pix(deg)
    assert math.isclose(recovered, original_px, abs_tol=1e-6)
