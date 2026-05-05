# linalg/scripts/composition.py
"""Composition of linear transformations via 2x2 matrices."""

from __future__ import annotations

from ..geometry.vectors2d import Vector2D
from ..geometry.matrix2d import Mat2
from ..utils import guards as gd
from ..utils.banners import banner
from ..utils.guards import NumericTypeError


class UserQuit(Exception):
    """Raised when the user chooses to exit an interactive prompt."""

def _read_or_quit(prompt: str) -> str:
    value = input(prompt).strip()
    if value.lower() in {"q", "quit", ""}:
        raise UserQuit
    return value

def parse_mat2(label: str) -> Mat2:
    """Read two rows of interactive input and return a Mat2."""
    row1_raw = _read_or_quit(f"> {label}, row 1: ")
    row1 = gd.parse_vec2d(row1_raw, f"{label} row 1")

    row2_raw = _read_or_quit(f"> {label}, row 2: ")
    row2 = gd.parse_vec2d(row2_raw, f"{label} row 2")

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
        bx = mat_b @ vec
        sequential = mat_a @ bx
        composed = product_ab @ vec

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
    """
    REPL loop: 
    
    Read two matrices, compose, and verify against test vectors.
    """
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
        except UserQuit:
            break
        except NumericTypeError as exc:
            print(f"  Input error: {exc}")
            continue
        except ValueError as exc:
            print(f"  Invalid value: {exc}")
            continue

        product_ab = mat_a @ mat_b
        print_report(mat_a, mat_b, product_ab)

        again = input("\nAnother pair? (y/n): ").strip().lower()
        if again not in {"y", "yes"}:
            break


if __name__ == "__main__":
    main()
