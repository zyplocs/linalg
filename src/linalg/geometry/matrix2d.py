# src/linalg/geometry/matrix2d.py
"""2x2 matrices for planar linear transformations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from dataclasses import dataclass

from ..utils import guards as gd
#from ..geometry.vectors2d import EPSILON, Vector2D

if TYPE_CHECKING:
    from ..utils.guards import ScalarLike


@dataclass(frozen=True, slots=True)
class Mat2:
    """
    A 2x2 matrix stored in row-major order.

    [a b]
    [c d]
    """
    a: float
    b: float
    c: float
    d: float

    def __init__(
        self,
        a: ScalarLike,
        b: ScalarLike,
        c: ScalarLike,
        d: ScalarLike,
    ) -> None:
        object.__setattr__(self, "a", gd.to_float(a, name="a"))
        object.__setattr__(self, "b", gd.to_float(b, name="b"))
        object.__setattr__(self, "c", gd.to_float(c, name="c"))
        object.__setattr__(self, "d", gd.to_float(d, name="d"))

    @classmethod
    def identity(cls) -> Mat2:
        """Return the 2x2 identity matrix."""
        return cls(1, 0, 0, 1)

    @classmethod
    def scale(cls, sx: ScalarLike, sy: ScalarLike | None = None) -> Mat2:
        """Return a scaling matrix; uniform if `sy` is omitted."""
        x = gd.to_float(sx, name="sx")
        y = x if sy is None else gd.to_float(sy, name="sy")
        return cls(x, 0, 0, y)

    @classmethod
    def rotation(cls, theta: ScalarLike) -> Mat2:
        """Construct a CCW rotation matrix for angle `theta` in radians."""
        angle = gd.to_float(theta, name="theta")
        cos_t = math.cos(angle)
        sin_t = math.sin(angle)
        return cls(cos_t, -sin_t, sin_t, cos_t)

    @classmethod
    def rotation_deg(cls, degrees: ScalarLike) -> Mat2:
        """Create a 2x2 CCW rotation matrix from an angle given in degrees."""
        return cls.rotation(math.radians(gd.to_float(degrees, name="degrees")))

    @property
    def determinant(self) -> float:
        """
        Compute the determinant of this 2x2 matrix.

        The [2x2] determinant is calculated as `(a*d - b*c)` and represents
        the signed area scaling factor of the linear transformation.
        A zero determinant indicates a singular (non-invertible) matrix.
        """
        return self.a * self.d - self.b * self.c

    @property
    def transpose(self) -> Mat2:
        """
        Return the transpose of this matrix.

        To transpose means to swap rows and columns, reflecting the 
        matrix across its main diagonal. For a matrix `[[a, b], [c, d]]`,
        the transpose is `[[a, c], [b, d]]`.
        """
        return Mat2(self.a, self.c, self.b, self.d)
