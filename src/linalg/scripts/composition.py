from __future__ import annotations

from ..vectors.vectors2d import Vector2D
from ..utils import guards as gd
from ..utils.banners import banner
from ..utils.guards import NumericTypeError

def main():
    banner("Vector Composition")

    while True:
        vec_raw = input("> Vector: ").strip()
        if vec_raw.lower() in {"q", "quit", ""}:
            break

        try:
            vec = gd.parse_vec2d(vec_raw, "Input Vector")
        except NumericTypeError as exc:
            print(f"Input error: {exc}")
            continue
        except ValueError as exc:
            print(f"Invalid value: {exc}")
            continue


if __name__ == "__main__":
    main()