"""Tests for src/linalg/vectors/vectors2d."""
import pytest

from linalg.geometry.vectors2d import Vector2D
from linalg.utils.guards import NumericTypeError


## Constructor & input guard tests
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


## String representation tests
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


## Sequence-protocol dunder tests
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
