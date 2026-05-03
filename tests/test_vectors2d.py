"""Tests for src/linalg/vectors/vectors2d."""
import pytest

from linalg.vectors.vectors2d import Vector2D
from linalg.utils.guards import NumericTypeError


def test_bool_is_only_false_for_exact_zero():
    assert not Vector2D(0, 0)
    assert Vector2D(1e-7, 0)

def test_is_near_zero_uses_epsilon_threshold():
    assert Vector2D(1e-7, 0).is_near_zero()
    assert not Vector2D(1e-3, 0).is_near_zero()

def test_from_polar_uses_same_numeric_guards_as_constructor():
    with pytest.raises(NumericTypeError):
        Vector2D.from_polar(True, 0)

    with pytest.raises(ValueError):
        Vector2D.from_polar(float("inf"), 0)
