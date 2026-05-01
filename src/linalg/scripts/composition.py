from __future__ import annotations

from typing import NamedTuple

from ..vectors.vectors2d import EPSILON, Vector2D
from ..utils import guards as gd
from ..utils.banners import banner
from ..utils.guards import NumericTypeError


class Mat2(NamedTuple):
    a: float
    b: float
    c: float
    d: float

    def __str__(self) -> str:
        return f"[{self.a:.4f}  {self.b:.4f} | {self.c:.4f}  {self.d:.4f}]"

def mat_mul_vec(mat: Mat2, vec: Vector2D) -> Vector2D:
    return Vector2D(
        mat.a * vec.x + mat.b * vec.y,
        mat.c * vec.x + mat.d * vec.y,
    )

def mat_mul_mat(left: Mat2, right: Mat2) -> Mat2:
    return Mat2(
        a=left.a * right.a + left.b * right.c,
        b=left.a * right.b + left.b * right.d,
        c=left.c * right.a + left.d * right.c,
        d=left.c * right.b + left.d * right.d,
    )

def parse_mat2(label: str) -> Mat2:
    row1_raw = input(f"> {label}, row 1: ").strip()
    row1 = gd.parse_vec2d(row1_raw, f"{label} row 1")

    row2_raw = input(f"> {label}, row 2: ").strip()
    row2 = gd.parse_vec2d(row1_raw, f"{label} row 2")

    return Mat2(a=row1.x, b=row1.y, c=row2.x, d=row2.y)


TEST_VECTORS: list[Vector2D] = [
    Vector2D(1, 0),
    Vector2D(0, 1),
    Vector2D(2, 3),
    Vector2D(-1, 3),
    Vector2D(0.5, 0.5)
]


def print_report(mat_a: Mat2, mat_b: Mat2, product_ab: Mat2) -> None:
    """Compare A(Bx) vs (AB)x for each test vector."""
    print(f"\n  A  = {mat_a}")
    print(f"  B  = {mat_b}")
    print(f"  AB = {product_ab}")
    print()

    all_match = True

    for vec in TEST_VECTORS:
        bx = mat_mul_vec(mat_b, vec)
        sequential = mat_mul_vec(mat_a, bx)
        composed = mat_mul_vec(product_ab, vec)

        matching = sequential.is_close(composed)
        tag = "match" if matching else "MISMATCH"
        if not matching:
            all_match = False

        print(
            f"  x = {vec:.2f}   "
            f"A(Bx) = {sequential:.4f}   "
            f"(AB)x = {composed:.4f}   "
            f"{tag}"
        )

    print()
    if all_match:
        print("  All vectors matched.")
    else:
        print("  ** MISMATCH detected - check your inputs!  **")


def main():
    banner("Composition of Transformations")
    print(
        "Enter each matrix as two rows of x,y (e.g. 1,0 then 0,1 = identity).\n"
        "Exit with q / quit at any prompt."
    )
 
    while True:
        print()
        try:
            mat_a = parse_mat2("Matrix A")
            mat_b = parse_mat2("Matrix B")
        except NumericTypeError as exc:
            print(f"  Input error: {exc}")
            continue
        except ValueError as exc:
            print(f"  Invalid value: {exc}")
            continue
 
        product_ab = mat_mul_mat(mat_a, mat_b)
        print_report(mat_a, mat_b, product_ab)
 
        again = input("\nAnother pair? (y/n): ").strip().lower()
        if again not in {"y", "yes"}:
            break


if __name__ == "__main__":
    main()