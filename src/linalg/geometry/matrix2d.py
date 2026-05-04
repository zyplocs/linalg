# src/linalg/geometry/matrix2d.py
"""2x2 matrices for planar linear transformations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from dataclasses import dataclass

from ..utils import guards as gd
from ..geometry.vectors2d import EPSILON, Vector2D

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

    def __matmul__(self, other: Mat2 | Vector2D) -> Mat2 | Vector2D:
        if isinstance(other, Mat2):
            return self.compose(other)
        if isinstance(other, Vector2D):
            return self.apply(other)
        return NotImplemented

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

    def inverse(self) -> Mat2:
        """
        Return the inverse of this matrix.

        The inverse of a 2x2 matrix `[[a, b], [c, d]]` is computed as
        `(1/det) * [[d, -b], [-c, a]]`. Raises `ValueError` if the
        determinant is near zero (singular matrix).
        """
        det = self.determinant
        if abs(det) <= EPSILON:
            raise ValueError("Cannot invert a near-singular matrix!")
        return Mat2(self.d / det, -self.b / det, -self.c / det, self.a / det)

    def apply(self, vec: Vector2D) -> Vector2D:
        """
        Apply this matrix transformation to a vector.

        Computes the matrix-vector product, transforming the input
        vector by this linear transformation.
        """
        return Vector2D(
            self.a * vec.x + self.b * vec.y,
            self.c * vec.x + self.d * vec.y,
        )

    def compose(self, other: Mat2) -> Mat2:
        """
        Return the composition of this matrix with another.

        Computes `self @ other`, representing the transformation
        `self(other(x))` — applying `other` first, then `self`.
        """
        return Mat2(
            self.a * other.a + self.b * other.c,
            self.a * other.b + self.b * other.d,
            self.c * other.a + self.d * other.c,
            self.c * other.b + self.d * other.d,
        )

    def is_close(self, other: Mat2, *, abs_tol: float = EPSILON) -> bool:
        """
        Approximate equality comparison of two `Mat2` objects.

        Returns `True` if all corresponding elements are within
        `abs_tol` of each other.
        """
        return (
            math.isclose(self.a, other.a, rel_tol=0.0, abs_tol=abs_tol)
            and math.isclose(self.b, other.b, rel_tol=0.0, abs_tol=abs_tol)
            and math.isclose(self.c, other.c, rel_tol=0.0, abs_tol=abs_tol)
            and math.isclose(self.d, other.d, rel_tol=0.0, abs_tol=abs_tol)
        )
