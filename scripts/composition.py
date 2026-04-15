from __future__ import annotations

from utils.banners import banner

from ..vectors.vectors2d import Vector2D
from ..utils import guards as gd
from ..utils.banners import banner
from ..utils.guards import NumericTypeError

def main():
    banner("Composition Validator")

    while True:
        vec_raw = input("> Vector: ").strip()
        if vec_raw.lower() in {"q", "quit", ""}:
            break