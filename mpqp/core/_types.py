"""Shared type aliases for MPQP core objects."""

from numbers import Complex
from typing import Mapping, Union

from sympy import Expr


BindingParameters = Union[
    Mapping[str, Complex | float],
    Mapping[Expr, Complex | float],
    Mapping[Expr | str, Complex | float],
]
"""Parameter values keyed by symbolic expressions or parameter names."""
