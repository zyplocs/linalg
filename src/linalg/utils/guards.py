# linalg/utils/guards.py
"""Numeric coercion and input-guard utilities for linalg."""

from __future__ import annotations
from typing import SupportsFloat, TYPE_CHECKING
from math import isfinite

type ScalarLike = SupportsFloat | str

if TYPE_CHECKING:
    try:
        from ..geometry.vectors2d import Vector2D
    except ImportError:
        from vectors2d import Vector2D

    type Vector2DLike = Vector2D | tuple[ScalarLike, ScalarLike]  # pylint: disable=invalid-name


class NumericTypeError(TypeError):
    """Raised when a numeric parameter receives a non-numeric argument."""


def to_float(usr_input: SupportsFloat | str, /, *, name: str) -> float:
    """Coerce `usr_input` to float, rejecting bools, non-numerics, etc."""
    try:
        if isinstance(usr_input, bool):
            raise TypeError("Booleans are not accepted!")

        floating = float(usr_input)
    except (TypeError, ValueError) as e:
        raise NumericTypeError(
            f"{name.capitalize()} must be numeric; "
            f"got {type(usr_input).__name__}!"
        ) from e

    if not isfinite(floating):
        raise ValueError(f"{name} must be finite; got {floating}!")

    return floating


def parse_vec2d(raw: str, label: str) -> Vector2D:
    """Parse a comma-separated string into a `Vector2D`."""
    try:
        from ..vectors.vectors2d import Vector2D  # pylint: disable=import-outside-toplevel
    except ImportError:
        from vectors2d import Vector2D  # pylint: disable=import-outside-toplevel

    cleaned = raw.strip()
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1]

    dims = [d.strip() for d in cleaned.split(",")]
    if len(dims) != 2:
        raise ValueError(f"{label} must be two comma-separated numbers!")
    x = to_float(dims[0], name=f"{label} x")
    y = to_float(dims[1], name=f"{label} y")

    return Vector2D(x, y)
