from __future__ import annotations
from typing import SupportsFloat, TYPE_CHECKING
from math import isfinite

if TYPE_CHECKING:
    try:
        from .vectors2d import Vector2D
    except ImportError:
        from vectors2d import Vector2D

type ScalarLike = SupportsFloat | str
type Vector2DLike = Vector2D | tuple[ScalarLike, ScalarLike]

class NumericTypeError(TypeError):
    """Raised when a numeric parameter receives a non-numeric argument."""


def to_float(usr_input: SupportsFloat | str, *, name: str) -> float:
    try:
        if isinstance(usr_input, bool):
            raise TypeError("Booleans are not accepted!")

        floating = float(usr_input)
    except (TypeError, ValueError) as e:
        raise NumericTypeError(f"{name.capitalize()} must be numeric; got {type(usr_input).__name__}!") from e

    if not isfinite(floating):
        raise ValueError(f"{name} must be finite; got {floating}!")

    return floating


def parse_vec2d(s: str, name: str) -> Vector2D:
    try:
        from .vectors2d import Vector2D
    except ImportError:
        from vectors2d import Vector2D

    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]

    dims = [d.strip() for d in s.split(",")]
    if len(dims) != 2:
        raise ValueError(f"{name} must be two comma-separated numbers!")
    x = to_float(dims[0], name=f"{name} x")
    y = to_float(dims[1], name=f"{name} y")

    return Vector2D(x, y)
