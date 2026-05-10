"""Tests for src/linalg/geometry/matrix2d."""
import math

import pytest

from linalg.geometry.matrix2d import Mat2
from linalg.geometry.vectors2d import Vector2D
from linalg.utils.guards import NumericTypeError


## Construction & input guard tests
def test_constructor_coerces_scalar_like_inputs():
    """Mat2 accepts int, float, and numeric strings via to_float."""
    m = Mat2("1", 2, 3.0, 4)
    assert m.a == 1.0
    assert m.d == 4.0

def test_constructor_rejects_bools():
    """Booleans are refused by the same guard used in Vector2D."""
    with pytest.raises(NumericTypeError):
        Mat2(True, 0, 0, 1)

def test_frozen_dataclass_prevents_mutation():
    """Slots + frozen means attribute assignment is forbidden."""
    i = Mat2.identity()
    with pytest.raises(AttributeError):
        i.a = 99


## Factory class methods tests
def test_scale_uniform_when_sy_omitted():
    """scale(k) puts k on both diagonal entries."""
    m = Mat2.scale(3)
    assert (m.a, m.b, m.c, m.d) == (3, 0, 0, 3)

def test_rotation_deg_matches_known_angles():
    """A 90-degree CCW rotation sends (1,0)->(0,1) and (0,1)->(-1,0)."""
    r90 = Mat2.rotation_deg(90)
    assert r90.apply(Vector2D(1, 0)).is_close(Vector2D(0, 1))
    assert r90.apply(Vector2D(0, 1)).is_close(Vector2D(-1, 0))


## Mathematical invariant tests
def test_rotation_has_unit_determinant():
    """Every rotation matrix should have determinant = 1."""
    for deg in (0, 30, 45, 90, 137, 180, 270):
        r = Mat2.rotation_deg(deg)
        assert math.isclose(r.determinant, 1.0, abs_tol=1e-12)
