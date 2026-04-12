from __future__ import annotations
import math
from ..utils import guards as gd
from ..utils.guards import NumericTypeError, ScalarLike, Vector2DLike

EPSILON = 1e-6  # close to zero threshold

class Vector2D:
    __slots__ = ('x', 'y')
    def __init__(self, x: float, y: float):
        self.x = gd.to_float(x, name="x")
        self.y = gd.to_float(y, name="y")

    @classmethod
    def _coerce(cls, other: Vector2DLike, name: str) -> Vector2D:
        if isinstance(other, cls):
            return other
        try:
            if len(other) != 2:
                raise TypeError
            x = gd.to_float(other[0], name=f"{name} x")
            y = gd.to_float(other[1], name=f"{name} y")
            return cls(x, y)
        except (TypeError, IndexError):
            raise TypeError(f"{name} must be a Vector2D or a tuple of two numbers, not {type(other).__name__}")

    def __repr__(self) -> str:
        return f"Vector2D({self.x}, {self.y})"

    def __add__(self, other: Vector2DLike) -> Vector2D:
        other_vec = self._coerce(other, name="other")
        return Vector2D(self.x + other_vec.x, self.y + other_vec.y)

    def __sub__(self, other: Vector2DLike) -> Vector2D:
        other_vec = self._coerce(other, name="other")
        return Vector2D(self.x - other_vec.x, self.y - other_vec.y)

    def __mul__(self, scalar: ScalarLike) -> Vector2D:
        scal = gd.to_float(scalar, name="scalar")
        return Vector2D(self.x * scal, self.y * scal)

    def __rmul__(self, scalar: ScalarLike) -> Vector2D:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: ScalarLike) -> Vector2D:
        scal = gd.to_float(scalar, name="scalar")
        if abs(scal) < EPSILON:
            raise ValueError("Cannot divide vector by a near-zero scalar!")
        return Vector2D(self.x / scal, self.y / scal)

    @property
    def magnitude(self) -> float:
        return math.hypot(self.x, self.y)

    def normalize(self) -> Vector2D:
        mag = self.magnitude
        if mag <= EPSILON:
            raise ValueError(f"Vector magnitude is too small to normalize (<= {EPSILON})!")
        return self / mag

    def dot(self, other: Vector2DLike) -> float:
        other_vec = self._coerce(other, name="other")
        return self.x * other_vec.x + self.y * other_vec.y

    def cross(self, other: Vector2DLike) -> float:
        other_vec = self._coerce(other, name="other")
        return self.x * other_vec.y - self.y * other_vec.x
    
    def projection_onto(self, other: Vector2DLike) -> Vector2D:
        other_vec = self._coerce(other, name="other")
        denom = other_vec.x**2 + other_vec.y**2
        if denom <= EPSILON**2:
            raise ValueError("Cannot project onto a near-zero vector!")
        scale = self.dot(other_vec) / denom
        return Vector2D(other_vec.x * scale, other_vec.y * scale)

    def rejection_from(self, other: Vector2DLike) -> Vector2D:
        return self - self.projection_onto(other)
    
    def component_along(self, other: Vector2DLike) -> float:
        """Return the scalar component of `self` along `other`."""

        other_vec = self._coerce(other, name="other")
        denom = other_vec.x**2 + other_vec.y**2
        if denom <= EPSILON**2:
            raise ValueError("Cannot compute component along a near-zero vector!")
        return self.dot(other_vec) / math.sqrt(denom)

    def angle_to(self, other: Vector2DLike, *, signed: bool = True, degrees: bool = True) -> float:
        other_vec = self._coerce(other, name="other")
        if self.magnitude <= EPSILON or other_vec.magnitude <= EPSILON:
            raise ValueError("Cannot compute angle for a zero-magnitude vector!")

        dot_prod = self.dot(other_vec)
        cross_prod = self.cross(other_vec)

        if signed:
            angle_rad = math.atan2(cross_prod, dot_prod)
        else:
            angle_rad = math.atan2(abs(cross_prod), dot_prod)
        
        if not degrees:
            return angle_rad
            
        return math.degrees(angle_rad)

    def angle_between(self, other: Vector2DLike, *, degrees: bool = True) -> float:
        return self.angle_to(other, signed=False, degrees=degrees)


def main():
    print("Calculate an output vector from two inputs. Separate dimensions with a comma")

    while True:
        vec1_raw = input("> Vector 1: ").strip()
        if vec1_raw.lower() in {"q", "quit", ""}:
            break
        vec2_raw = input("> Vector 2: ").strip()
        if vec2_raw.lower() in {"q", "quit", ""}:
            break

        try:
            vec1 = gd.parse_vec2d(vec1_raw, "Vector 1")
            vec2 = gd.parse_vec2d(vec2_raw, "Vector 2")

            new_vec = vec1 + vec2
            new_mag = new_vec.magnitude
            new_unit_vec = new_vec.normalize()

            print(f"Sum vector: ({new_vec.x:.2f}, {new_vec.y:.2f})")
            print(f"Magnitude: {new_mag:.3f}")
            print(f"Unit vector: ({new_unit_vec.x:.2f}, {new_unit_vec.y:.2f})")
            break
        except NumericTypeError as e:
            print(f"Input error: {e}")
            continue
        except ValueError as e:
            print(f"Invalid value: {e}")
            continue


if __name__ == "__main__":
    main()
