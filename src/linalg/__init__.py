# linalg/__init__.py
"""Linear Algebra for Visual Applications"""

from .geometry.vectors2d import Vector2D
from .geometry.matrix2d import Mat2
from .utils.guards import NumericTypeError

__all__ = ['Vector2D', 'Mat2', 'NumericTypeError']

# Metadata
#__version__ = '0.1.0'
#__author__ = 'Eli M J'
