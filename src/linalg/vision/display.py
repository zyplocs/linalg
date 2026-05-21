# src/linalg/vision/display.py
"""Display specification and visual-angle conversions."""

from __future__ import annotations

import math
#from dataclasses import dataclass


def cm_to_deg(size_cm: float, distance_cm: float) -> float:
    """
    Convert a physical size in cm to degrees of visual angle.

    Uses the formula:  θ = 2 * arctan(s / 2d)
    """
    return 2.0 * math.degrees(math.atan2(size_cm, 2.0 * distance_cm))

def deg_to_cm(angle_deg: float, distance_cm: float) -> float:
    """
    Convert degrees of visual angle to physical size in cm.

    Inverse of `cm_to_deg`:  s = 2d * tan(θ / 2)
    """
    return 2.0 * distance_cm * math.tan(math.radians(angle_deg / 2.0))
