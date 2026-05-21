"""Tests for src/linalg/geometry/matrix2d."""
import math

import pytest

from linalg.geometry.matrix2d import Mat2
from linalg.geometry.vectors2d import Vector2D
from linalg.utils.guards import NumericTypeError


## Construction & input guard tests –––––––––––––––––––––––
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


## Factory class methods tests ––––––––––––––––––––––––––––
def test_identity_returns_unit_matrix():
    """identity() produces [[1,0],[0,1]]."""
    eye = Mat2.identity()
    assert (eye.a, eye.b, eye.c, eye.d) == (1, 0, 0, 1)

def test_scale_uniform_when_sy_omitted():
    """scale(k) puts k on both diagonal entries."""
    m = Mat2.scale(3)
    assert (m.a, m.b, m.c, m.d) == (3, 0, 0, 3)

def test_scale_non_uniform():
    """scale(sx, sy) puts sx and sy on the diagonal independently."""
    m = Mat2.scale(2, 5)
    assert (m.a, m.d) == (2, 5)
    assert (m.b, m.c) == (0, 0)

def test_rotation_deg_matches_known_angles():
    """A 90-degree CCW rotation sends (1,0)->(0,1) and (0,1)->(-1,0)."""
    r90 = Mat2.rotation_deg(90)
    assert r90.apply(Vector2D(1, 0)).is_close(Vector2D(0, 1))
    assert r90.apply(Vector2D(0, 1)).is_close(Vector2D(-1, 0))


## Property tests –––––––––––––––––––––––––––––––––––––––––
def test_determinant_of_identity_is_one():
    """The identity matrix has determinant 1."""
    assert Mat2.identity().determinant == 1.0

def test_determinant_of_singular_matrix_is_zero():
    """A matrix with linearly dependent rows has determinant 0."""
    assert Mat2(1, 2, 2, 4).determinant == 0.0

def test_transpose_swaps_off_diagonal():
    """Transposing exchanges b and c while keeping a and d."""
    m = Mat2(1, 2, 3, 4)
    t = m.transpose
    assert (t.a, t.b, t.c, t.d) == (1, 3, 2, 4)


## Instance method tests ––––––––––––––––––––––––––––––––––
def test_inverse_of_identity_is_identity():
    """Inverting the identity gives back the identity."""
    assert Mat2.identity().inverse().is_close(Mat2.identity())

def test_inverse_roundtrip_restores_identity():
    """M @ M^-1 should approximate the identity for a non-trivial M."""
    m = Mat2(2, 1, 1, 3)
    product = m.compose(m.inverse())
    assert product.is_close(Mat2.identity())

def test_inverse_raises_for_singular_matrix():
    """A singular matrix (det ~ 0) cannot be inverted."""
    with pytest.raises(ValueError):
        Mat2(1, 2, 2, 4).inverse()

def test_apply_transforms_vector():
    """Scaling by 2 doubles both components of the input vector."""
    scaled = Mat2.scale(2).apply(Vector2D(3, 4))
    assert scaled == Vector2D(6, 8)

def test_compose_matches_sequential_application():
    """(A @ B) @ v should equal A @ (B @ v) for any v."""
    a = Mat2.rotation_deg(30)
    b = Mat2.scale(2, 3)
    v = Vector2D(1, 1)

    sequential = a.apply(b.apply(v))
    composed = a.compose(b).apply(v)
    assert sequential.is_close(composed)


## Dunder (__matmul__) tests ––––––––––––––––––––––––––––––
def test_matmul_dispatches_to_compose_for_mat2():
    """The @ operator between two Mat2s returns a Mat2."""
    result = Mat2.identity() @ Mat2.scale(5)
    assert isinstance(result, Mat2)
    assert result.is_close(Mat2.scale(5))

def test_matmul_dispatches_to_apply_for_vector():
    """The @ operator with a Vector2D returns a Vector2D."""
    result = Mat2.scale(2) @ Vector2D(1, 1)
    assert isinstance(result, Vector2D)
    assert result == Vector2D(2, 2)

def test_matmul_returns_not_implemented_for_other_types():
    """@ with an unsupported operand raises TypeError."""
    with pytest.raises(TypeError):
        _ = Mat2.identity() @ 42


## Approximate equality tests –––––––––––––––––––––––––––––
def test_is_close_within_tolerance():
    """Matrices differing by less than EPSILON are considered close."""
    m1 = Mat2(1, 0, 0, 1)
    m2 = Mat2(1 + 1e-9, 0, 0, 1)
    assert m1.is_close(m2)

def test_is_close_rejects_outside_tolerance():
    """Matrices differing by more than the tolerance are not close."""
    m1 = Mat2(1, 0, 0, 1)
    m2 = Mat2(1.1, 0, 0, 1)
    assert not m1.is_close(m2)


## Mathematical invariant tests –––––––––––––––––––––––––––
def test_rotation_has_unit_determinant():
    """Every rotation matrix should have determinant = 1."""
    for deg in (0, 30, 45, 90, 137, 180, 270):
        r = Mat2.rotation_deg(deg)
        assert math.isclose(r.determinant, 1.0, abs_tol=1e-12)
