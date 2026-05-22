"""Tests for src/eyesee/vectors/vectors2d."""
import math
import pytest

from eyesee.geometry.vectors2d import Vector2D
from eyesee.utils.guards import NumericTypeError


## Constructor & input guard tests ––––––––––––––––––––––––
def test_constructor_coerces_int_and_string_to_float():
    """Components are stored as float regardless of input type."""
    v = Vector2D(3, "4")
    assert v.x == 3.0 and isinstance(v.x, float)
    assert v.y == 4.0 and isinstance(v.y, float)

def test_constructor_rejects_bools():
    """Booleans are not numeric; to_float raises NumericTypeError."""
    with pytest.raises(NumericTypeError):
        Vector2D(True, 0)

def test_constructor_rejects_non_finite():
    """Infinity and NaN are caught by to_float's finiteness guard."""
    with pytest.raises(ValueError):
        Vector2D(float("inf"), 0)
    with pytest.raises(ValueError):
        Vector2D(0, float("nan"))


## String representation tests ––––––––––––––––––––––––––––
def test_repr_round_trips():
    """__repr__ produces the 'Vector2D(x, y)' form."""
    v = Vector2D(1, 2)
    assert repr(v) == "Vector2D(1.0, 2.0)"

def test_str_uses_parenthesized_form():
    """__str__ gives the short '(x, y)' form."""
    assert str(Vector2D(3, 4)) == "(3.0, 4.0)"

def test_format_applies_spec_to_each_component():
    """__format__ passes the format spec through to both x and y."""
    assert f"{Vector2D(1.456, 2.789):.1f}" == "(1.5, 2.8)"


## Sequence-protocol dunder tests –––––––––––––––––––––––––
def test_len_is_always_two():
    """__len__ returns 2 for any Vector2D."""
    assert len(Vector2D(0, 0)) == 2

def test_getitem_accesses_components_by_index():
    """__getitem__ supports 0, 1, and negative indices."""
    v = Vector2D(3, 7)
    assert v[0] == 3.0
    assert v[1] == 7.0
    assert v[-1] == 7.0

def test_getitem_raises_for_out_of_bounds():
    """Indexing beyond 0-1 raises IndexError."""
    with pytest.raises(IndexError):
        _ = Vector2D(1, 2)[2]

def test_iter_yields_components():
    """__iter__ unpacks into (x, y)."""
    x, y = Vector2D(5, 6)
    assert (x, y) == (5.0, 6.0)


## Unary operation tests ––––––––––––––––––––––––––––––––––
def test_neg_flips_both_components():
    """Negation reverses the sign of x and y."""
    assert -Vector2D(3, -4) == Vector2D(-3, 4)

def test_abs_returns_magnitude():
    """__abs__ delegates to the magnitude property."""
    assert abs(Vector2D(3, 4)) == 5.0

def test_round_truncates_to_n_digits():
    """__round__ rounds each component independently."""
    assert round(Vector2D(1.456, 2.789), 1) == Vector2D(1.5, 2.8)

def test_bool_is_only_false_for_exact_zero():
    """__bool__ returns False only for the exact zero vector."""
    assert not Vector2D(0, 0)
    assert Vector2D(1e-7, 0)

def test_equality_matches_general_two_item_sequences():
    """__eq__ accepts any 2-item numeric sequence, not just tuples."""
    assert Vector2D(1, 2) == (1, 2)
    assert Vector2D(1, 2) == [1, 2]


## Arithmetic tests –––––––––––––––––––––––––––––––––––––––
def test_add_two_vectors():
    """Component-wise addition of two Vector2D objects."""
    assert Vector2D(1, 2) + Vector2D(3, 4) == Vector2D(4, 6)

def test_add_vector_and_tuple():
    """__add__ coerces a 2-tuple on the right-hand side."""
    assert Vector2D(1, 2) + (3, 4) == Vector2D(4, 6)

def test_radd_tuple_plus_vector():
    """__radd__ lets a tuple appear on the left-hand side."""
    assert (3, 4) + Vector2D(1, 2) == Vector2D(4, 6)

def test_vector_coercion_rejects_plain_strings():
    """Arithmetic with a bare string raises TypeError, not coercion."""
    with pytest.raises(TypeError):
        _ = Vector2D(1, 2) + "12"

def test_sub_two_vectors():
    """Component-wise subtraction."""
    assert Vector2D(5, 7) - Vector2D(2, 3) == Vector2D(3, 4)

def test_rsub_tuple_minus_vector():
    """__rsub__ computes other - self (not self - other)."""
    result = (10, 10) - Vector2D(3, 4)
    assert result == Vector2D(7, 6)

def test_mul_scales_both_components():
    """Scalar multiplication scales x and y uniformly."""
    assert Vector2D(2, 3) * 4 == Vector2D(8, 12)

def test_rmul_scalar_on_left():
    """__rmul__ lets the scalar appear on the left side."""
    assert 2 * Vector2D(3, 8) == Vector2D(6, 16)

def test_truediv_divides_components():
    """Scalar division divides each component."""
    assert Vector2D(8, 4) / 4 == Vector2D(2, 1)

def test_truediv_raises_for_near_zero_divisor():
    """Division by a near-zero scalar raises ValueError."""
    with pytest.raises(ValueError):
        _ = Vector2D(1, 2) / 0


## Property tests –––––––––––––––––––––––––––––––––––––––––
def test_magnitude_of_3_4_is_5():
    """Classic 3-4-5 Pythagorean triple."""
    assert Vector2D(3, 4).magnitude == 5.0

def test_magnitude_of_zero_vector_is_zero():
    """The zero vector has magnitude 0."""
    assert Vector2D(0, 0).magnitude == 0.0

def test_theta_along_positive_x_is_zero():
    """A vector along +x has angle 0."""
    assert Vector2D(1, 0).theta == 0.0

def test_theta_along_positive_y_is_half_pi():
    """A vector along +y has angle pi/2."""
    assert math.isclose(Vector2D(0, 1).theta, math.pi / 2)

def test_is_near_zero_uses_epsilon_threshold():
    """is_near_zero returns True when magnitude is below epsilon."""
    assert Vector2D(1e-7, 0).is_near_zero()
    assert not Vector2D(1e-3, 0).is_near_zero()

def test_polar_round_trips_with_from_polar():
    """polar property decomposes; from_polar recomposes."""
    v = Vector2D(3, 4)
    r, angle = v.polar
    reconstructed = Vector2D.from_polar(r, angle)
    assert v.is_close(reconstructed)

def test_from_polar_uses_same_numeric_guards_as_constructor():
    """from_polar rejects bools and non-finite values like the constructor."""
    with pytest.raises(NumericTypeError):
        Vector2D.from_polar(True, 0)

    with pytest.raises(ValueError):
        Vector2D.from_polar(float("inf"), 0)

def test_from_polar_accepts_numeric_strings():
    """from_polar coerces numeric strings to float via to_float."""
    assert Vector2D.from_polar("2", "0") == Vector2D(2, 0)


## Dot & cross product tests ––––––––––––––––––––––––––––––
def test_dot_product_of_perpendicular_vectors_is_zero():
    """Orthogonal vectors have zero dot product."""
    assert Vector2D(1, 0).dot(Vector2D(0, 1)) == 0.0

def test_dot_product_of_parallel_vectors():
    """Parallel vectors: dot(v, v) == magnitude^2."""
    v = Vector2D(4, 6)
    assert math.isclose(v.dot(v), v.magnitude**2)

def test_cross_product_of_parallel_vectors_is_zero():
    """Parallel (or anti-parallel) vectors have zero cross product."""
    assert Vector2D(2, 4).cross(Vector2D(1, 2)) == 0.0

def test_cross_product_of_basis_vectors():
    """x-hat cross y-hat == 1 (positive orientation)."""
    assert Vector2D(1, 0).cross(Vector2D(0, 1)) == 1.0

def test_dot_accepts_tuple():
    """dot coerces a 2-tuple via _coerce."""
    assert Vector2D(2, 3).dot((4, 5)) == 23.0


## Normalization tests ––––––––––––––––––––––––––––––––––––
def test_normalize_returns_unit_vector():
    """The normalized vector has magnitude 1."""
    unit = Vector2D(3, 4).normalize()
    assert math.isclose(unit.magnitude, 1.0)

def test_normalize_preserves_direction():
    """Normalizing doesn't change the angle."""
    v = Vector2D(3, 4)
    assert math.isclose(v.normalize().theta, v.theta)

def test_normalize_raises_for_zero_vector():
    """Cannot normalize a zero-length vector."""
    with pytest.raises(ValueError):
        Vector2D(0, 0).normalize()


## Projection, rejection & component tests ––––––––––––––––
def test_projection_onto_axis():
    """Projecting (3,4) onto the x-axis yields (3,0)."""
    proj = Vector2D(3, 4).projection_onto(Vector2D(1, 0))
    assert proj == Vector2D(3, 0)

def test_rejection_is_orthogonal_complement():
    """rejection = original - projection, orthogonal to the target."""
    v = Vector2D(3, 4)
    axis = Vector2D(1, 0)
    rej = v.rejection_from(axis)
    assert rej == Vector2D(0, 4)

def test_projection_plus_rejection_equals_original():
    """The projection and rejection sum back to the original vector."""
    v = Vector2D(3, 4)
    onto = Vector2D(1, 1)
    assert v.is_close(v.projection_onto(onto) + v.rejection_from(onto))

def test_projection_onto_zero_vector_raises():
    """Cannot project onto a near-zero vector."""
    with pytest.raises(ValueError):
        Vector2D(1, 2).projection_onto(Vector2D(0, 0))

def test_component_along_axis():
    """Scalar component of (3,4) along x-axis is 3."""
    assert Vector2D(3, 4).component_along(Vector2D(1, 0)) == 3.0

def test_component_along_zero_vector_raises():
    """Cannot compute scalar component along a near-zero vector."""
    with pytest.raises(ValueError):
        Vector2D(1, 2).component_along(Vector2D(0, 0))


## Angle tests ––––––––––––––––––––––––––––––––––––––––––––
def test_angle_to_perpendicular_vectors_is_90_degrees():
    """Signed angle from +x to +y is 90 degrees."""
    angle = Vector2D(1, 0).angle_to(Vector2D(0, 1))
    assert math.isclose(angle, 90.0)

def test_angle_to_signed_can_be_negative():
    """Signed angle from +y to +x is -90 degrees (clockwise)."""
    angle = Vector2D(0, 1).angle_to(Vector2D(1, 0))
    assert math.isclose(angle, -90.0)

def test_angle_to_unsigned():
    """Unsigned angle between +y and +x is 90 (never negative)."""
    angle = Vector2D(0, 1).angle_to(Vector2D(1, 0), signed=False)
    assert math.isclose(angle, 90.0)

def test_angle_to_in_radians():
    """Passing degrees=False returns radians."""
    angle = Vector2D(1, 0).angle_to(Vector2D(0, 1), degrees=False)
    assert math.isclose(angle, math.pi / 2)

def test_angle_to_raises_for_zero_vector():
    """Angle is undefined for a zero-magnitude operand."""
    with pytest.raises(ValueError):
        Vector2D(0, 0).angle_to(Vector2D(1, 0))
    with pytest.raises(ValueError):
        Vector2D(1, 0).angle_to(Vector2D(0, 0))

def test_angle_between_delegates_unsigned():
    """angle_between is a convenience for angle_to(signed=False)."""
    a, b = Vector2D(1, 0), Vector2D(-1, 1)
    assert math.isclose(
        a.angle_between(b),
        a.angle_to(b, signed=False),
    )


## Rotation tests –––––––––––––––––––––––––––––––––––––––––
def test_rotate_90_degrees():
    """Rotating (1,0) by 90 degrees CCW gives (0,1)."""
    rotated = Vector2D(1, 0).rotate(90)
    assert rotated.is_close(Vector2D(0, 1))

def test_rotate_in_radians():
    """degrees=False uses the angle directly as radians."""
    rotated = Vector2D(1, 0).rotate(math.pi / 2, degrees=False)
    assert rotated.is_close(Vector2D(0, 1))

def test_rotate_360_returns_to_original():
    """A full revolution is the identity transformation."""
    v = Vector2D(3, 4)
    assert v.is_close(v.rotate(360))


## Some edge cases ––––––––––––––––––––––––––––––––––––––––
def test_is_close_within_epsilon():
    """Vectors differing by less than EPSILON are considered close."""
    v1 = Vector2D(1, 2)
    v2 = Vector2D(1 + 1e-9, 2)
    assert v1.is_close(v2)

def test_is_close_outside_epsilon():
    """Vectors differing by more than EPSILON are not close."""
    assert not Vector2D(1, 2).is_close(Vector2D(1.1, 2))

def test_is_close_returns_false_for_non_coercible():
    """is_close returns False (not an error) for incompatible types."""
    assert not Vector2D(1, 2).is_close("not a vector")

def test_eq_returns_not_implemented_for_incompatible_type():
    """Comparing to a non-coercible type doesn't raise — returns False."""
    assert Vector2D(1, 2) != "nope"
    assert Vector2D(1, 2) != 42
