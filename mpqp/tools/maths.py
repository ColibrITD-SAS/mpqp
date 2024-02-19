from __future__ import annotations

from numbers import Complex, Real

import numpy as np
import numpy.typing as npt
import sympy as sp
from sympy import Expr, I, pi
from typeguard import typechecked

from mpqp.tools.generics import Matrix

rtol = 1e-05
"""The relative tolerance parameter."""
atol = 1e-08
"""The absolute tolerance parameter."""


@typechecked
def normalize(v: npt.NDArray[np.complex64]) -> npt.NDArray[np.complex64]:
    """Normalizes an array representing the amplitudes of the state.

    Examples:
        >>> vector = np.array([1,0,0,1])
        >>> normalize(vector)
        array([0.70710678, 0., 0., 0.70710678])
        >>> vector = np.array([0,0,0,0])
        >>> normalize(vector)
        array([0, 0, 0, 0])

    Args:
        v: vector to be normalized

    Returns:
        The normalized vector.
    """
    norm = np.linalg.norm(v, ord=2)
    return v if norm == 0 else v / norm


@typechecked
def matrix_eq(lhs: Matrix, rhs: Matrix) -> bool:
    r"""Checks whether two matrix are element-wise equal, within a tolerance.

    For respectively each elements `a` and `b` of both inputs, we check this
    specific condition: `|a - b| \leq (atol + rtol * |b|)`.

    Args:
        lhs: Left-hand side matrix of the equality.
        rhs: Right-hand side matrix of the equality.

    Returns:
        ``True`` if the two matrix are equal (according to the definition above).
    """
    for elt in zip(np.ndarray.flatten(lhs), np.ndarray.flatten(rhs)):
        if isinstance(elt[0], Expr) or isinstance(elt[1], Expr):
            if elt[0] != elt[1]:
                return False
        else:
            if abs(elt[0] - elt[1]) > (atol + rtol * abs(elt[1])):
                return False
    return True


@typechecked
def is_hermitian(matrix: Matrix) -> bool:
    """Checks whether the matrix in parameter is hermitian.

    Args:
        matrix: matrix for which we want to know if it is hermitian

    Examples:
        >>> m1 = np.array([[1,2j,3j],[-2j,4,5j],[-3j,-5j,6]])
        >>> is_hermitian(m1)
        True
        >>> m2 = np.diag([1,2,3,4])
        >>> is_hermitian(m2)
        True
        >>> m3 = np.array([[1,2,3],[2,4,5],[3,5,6]])
        >>> is_hermitian(m3)
        True
        >>> m4 = np.array([[1,2,3],[4,5,6],[7,8,9]])
        >>> is_hermitian(m4)
        False
        >>> x = symbols("x", real=True)
        >>> m5 = np.diag([1,x])
        >>> is_hermitian(m5)
        True
        >>> m6 = np.array([[1,x],[-x,2]])
        >>> is_hermitian(m6)
        False

    Returns:
        ``True`` if the matrix in parameter is Hermitian.
    """
    return matrix_eq(np.array(matrix).transpose().conjugate(), matrix)  # type: ignore


@typechecked
def is_unitary(matrix: Matrix) -> bool:
    """Checks whether the matrix in parameter is unitary.

    Example:
        >>> a = np.array([[1,1],[1,-1]])
        >>> is_unitary(a)
        False
        >>> is_unitary(a/np.sqrt(2))
        True

    Args:
        matrix: Matrix for which we want to know if it is unitary.

    Returns:
        ``True`` if the matrix in parameter is Unitary.
    """
    return matrix_eq(np.eye(len(matrix)), matrix.transpose().conjugate().dot(matrix))


@typechecked
def cos(angle: sp.Expr | float) -> sp.Expr | float:
    """Generalization of the cosine function, to take as input either
    ``sympy``'s expressions or floating numbers.

    Args:
        angle: The angle considered.

    Returns:
        Cosine of the given ``angle``.
    """
    return np.cos(angle) if isinstance(angle, Real) else sp.cos(angle)


@typechecked
def sin(angle: sp.Expr | float) -> sp.Expr | float:
    """Generalization of the sine function, to take as input either
    ``sympy``'s expressions or floating numbers.

    Args:
        angle: The angle considered.

    Returns:
        Sine of the given ``angle``.
    """
    return np.sin(angle) if isinstance(angle, Real) else sp.sin(angle)


@typechecked
def exp(angle: sp.Expr | complex) -> sp.Expr | complex:
    """Generalization of the exponential function, to take as input either
    ``sympy``'s expressions or floating numbers.

    Args:
        angle: The angle considered.

    Returns:
        Exponential of the given ``angle``.
    """
    return np.exp(angle) if isinstance(angle, Complex) else sp.exp(angle)
