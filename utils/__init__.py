# linalg/utils/__init__.py
"""Utility functions and validation helpers"""

from .guards import NumericTypeError, to_float, parse_vec2d

__all__ = ['NumericTypeError', 'to_float', 'parse_vec2d']