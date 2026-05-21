# src/linalg/vision/display.py
"""Display specification and visual-angle conversions."""

from __future__ import annotations

import math
from dataclasses import dataclass


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


@dataclass(frozen=True, slots=True)
class Display:
    """Physical display specification for visual-angle calculations."""
    resolution: tuple[int, int]
    size_cm: tuple[float, float]
    viewing_distance_cm: float

    @property
    def pix_per_cm(self) -> tuple[float, float]:
        """Pixels per centimeter (horizontal, vertical)."""
        return (
            self.resolution[0] / self.size_cm[0],
            self.resolution[1] / self.size_cm[1],
        )

    @property
    def cm_per_pix(self) -> tuple[float, float]:
        """Centimeters per pixel (horizontal, vertical)."""
        return (
            self.size_cm[0] / self.resolution[0],
            self.size_cm[1] / self.resolution[1],
        )

    @property
    def pix_per_deg(self) -> tuple[float, float]:
        """
        Pixels per degree of visual angle (horizontal, vertical).

        Computed at screen center (0° eccentricity).
        """
        h_cm_per_deg = deg_to_cm(1.0, self.viewing_distance_cm)
        return (
            self.pix_per_cm[0] * h_cm_per_deg,
            self.pix_per_cm[1] * h_cm_per_deg,
        )

    @property
    def deg_per_pix(self) -> tuple[float, float]:
        """Degrees of visual angle per pixel (horizontal, vertical)."""
        ppd = self.pix_per_deg
        return (1.0 / ppd[0], 1.0 / ppd[1])
