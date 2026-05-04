"""Tests for src/linalg/vectors/vectors2d."""
import pytest

from linalg.geometry.vectors2d import Vector2D
from linalg.utils.guards import NumericTypeError


def test_bool_is_only_false_for_exact_zero():
    """__bool__ returns False only for the exact zero vector."""
    assert not Vector2D(0, 0)
    assert Vector2D(1e-7, 0)

def test_is_near_zero_uses_epsilon_threshold():
    """is_near_zero returns True when magnitude is below epsilon."""
    assert Vector2D(1e-7, 0).is_near_zero()
    assert not Vector2D(1e-3, 0).is_near_zero()

def test_from_polar_uses_same_numeric_guards_as_constructor():
    """from_polar rejects bools and non-finite values like the constructor."""
    with pytest.raises(NumericTypeError):
        Vector2D.from_polar(True, 0)

    with pytest.raises(ValueError):
        Vector2D.from_polar(float("inf"), 0)

def test_from_polar_accepts_numeric_strings():
    """from_polar coerces numeric strings to float via to_float."""
    assert Vector2D.from_polar("2", "0") == Vector2D(2, 0)

def test_vector_coercion_rejects_plain_strings():
    """Arithmetic with a bare string raises TypeError, not coercion."""
    with pytest.raises(TypeError):
        _ = Vector2D(1, 2) + "12"

def test_equality_matches_general_two_item_sequences():
    """__eq__ accepts any 2-item numeric sequence, not just tuples."""
    assert Vector2D(1, 2) == (1, 2)
    assert Vector2D(1, 2) == [1, 2]
