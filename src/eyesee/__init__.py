# eyesee/__init__.py
"""Linear Algebra for Visual Applications"""

from .geometry.vector2d import Vector2D
from .geometry.mat2 import Mat2
from .utils.guards import NumericTypeError
from . import vision

__all__ = ['Vector2D', 'Mat2', 'NumericTypeError', 'vision']

# Metadata
#__version__ = '0.0.0-1'
#__author__ = 'Eli M J'
