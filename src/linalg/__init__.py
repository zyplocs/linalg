# linalg/__init__.py
"""Linear Algebra for Visual Applications"""
from .vectors.vectors2d import Vector2D
from .utils.guards import NumericTypeError

__all__ = ['Vector2D', 'NumericTypeError']

# Metadata
#__version__ = '0.1.0'
#__author__ = 'Eli M J'