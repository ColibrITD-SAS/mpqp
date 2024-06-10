"""Mathematical tools for linear algebra, functions generalized to more data 
types, etc…"""

from __future__ import annotations

import math
from functools import reduce
from numbers import Complex, Real
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sympy import Expr
    import sympy as sp

import numpy as np
import numpy.typing as npt
from scipy.linalg import inv, sqrtm
from typeguard import typechecked

from mpqp.tools.generics import Matrix

rtol = 1e-05
"""The relative tolerance parameter."""
atol = 1e-08
"""The absolute tolerance parameter."""


@typechecked
def normalize(v: npt.NDArray[np.complex64]) -> npt.NDArray[np.complex64]:
    """Normalizes an array representing the amplitudes of the state.

    Args:
        v: The vector to be normalized.

    Returns:
        The normalized vector.

    Examples:
        >>> vector = np.array([1,0,0,1])
        >>> normalize(vector)
        array([0.70710678, 0.        , 0.        , 0.70710678])
        >>> vector = np.array([0,0,0,0])
        >>> normalize(vector)
        array([0, 0, 0, 0])

    """
    norm = np.linalg.norm(v, ord=2)
    return v if norm == 0 else v / norm


@typechecked
def matrix_eq(lhs: Matrix, rhs: Matrix, atol: float = atol, rtol: float = rtol) -> bool:
    r"""Checks whether two matrix (including vectors) are element-wise equal, within a tolerance.

    For respectively each elements `a` and `b` of both inputs, we check this
    specific condition: `|a - b| \leq (atol + rtol * |b|)`.

    Args:
        lhs: Left-hand side matrix of the equality.
        rhs: Right-hand side matrix of the equality.

    Returns:
        ``True`` if the two matrix are equal (according to the definition above).
    """
    from sympy import Expr

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
        matrix: matrix for which we want to know if it is hermitian.

    Returns:
        ``True`` if the matrix in parameter is Hermitian.

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

    """
    return matrix_eq(np.array(matrix).transpose().conjugate(), matrix)  # type: ignore


@typechecked
def is_unitary(matrix: Matrix) -> bool:
    """Checks whether the matrix in parameter is unitary.

    Args:
        matrix: Matrix for which we want to know if it is unitary.

    Returns:
        ``True`` if the matrix in parameter is Unitary.

    Example:
        >>> a = np.array([[1,1],[1,-1]])
        >>> is_unitary(a)
        False
        >>> is_unitary(a/np.sqrt(2))
        True

    """
    return matrix_eq(
        np.eye(len(matrix), dtype=np.complex64),
        matrix.transpose().conjugate().dot(matrix),
    )


@typechecked
def cos(angle: Expr | Real) -> sp.Expr | float:
    """Generalization of the cosine function, to take as input either
    ``sympy``'s expressions or floating numbers.

    Args:
        angle: The angle considered.

    Returns:
        Cosine of the given ``angle``.
    """
    if isinstance(angle, Real):
        if TYPE_CHECKING:
            assert isinstance(angle, float)
        return np.cos(angle)
    else:
        import sympy as sp
        from sympy import Expr

        res = sp.cos(angle)
        assert isinstance(res, Expr)
        return res


@typechecked
def sin(angle: Expr | Real) -> sp.Expr | float:
    """Generalization of the sine function, to take as input either
    ``sympy``'s expressions or floating numbers.

    Args:
        angle: The angle considered.

    Returns:
        Sine of the given ``angle``.
    """
    if isinstance(angle, Real):
        if TYPE_CHECKING:
            assert isinstance(angle, float)
        return np.sin(angle)
    else:
        import sympy as sp
        from sympy import Expr

        res = sp.sin(angle)
        assert isinstance(res, Expr)
        return res


@typechecked
def exp(angle: Expr | Complex) -> sp.Expr | complex:
    """Generalization of the exponential function, to take as input either
    ``sympy``'s expressions or floating numbers.

    Args:
        angle: The angle considered.

    Returns:
        Exponential of the given ``angle``.
    """
    if isinstance(angle, Complex):
        if TYPE_CHECKING:
            assert isinstance(angle, complex)
        return np.exp(angle)
    else:
        import sympy as sp
        from sympy import Expr

        res = sp.exp(angle)
        assert isinstance(res, Expr)
        return res


def rand_orthogonal_matrix(
    size: int, seed: Optional[int] = None
) -> npt.NDArray[np.complex64]:
    """Generate a random orthogonal matrix optionally with a given seed.

    Args:
        size: Size (number of columns) of the square matrix to generate.
        seed: Seed used to control the random generation of the matrix.

    Returns:
        A random orthogonal matrix.

    Examples:
        >>> rand_orthogonal_matrix(3) # doctest: +SKIP
        array([[ 0.94569439,  0.2903415 ,  0.14616405],
               [-0.32503798,  0.83976928,  0.43489984],
               [ 0.0035254 , -0.45879121,  0.88853711]])

        >>> rand_orthogonal_matrix(3, seed=42)
        array([[ 0.21667149,  0.1867762 ,  0.95821089],
               [ 0.9608116 ,  0.13303749, -0.24319148],
               [-0.17290035,  0.9733528 , -0.15063131]])

    """
    np.random.seed(seed)
    m = np.random.rand(size, size)
    return m.dot(inv(sqrtm(m.T.dot(m))))


def rand_clifford_matrix(nb_qubits: int) -> npt.NDArray[np.complex64]:
    """Generate a random Clifford matrix.

    Args:
        size: Size (number of columns) of the square matrix to generate.

    Returns:
        A random Clifford matrix.

    Examples:
        >>> rand_clifford_matrix(2) # doctest: +SKIP
        array([[ 0.5+0.j, -0.5+0.j,  0.5+0.j, -0.5+0.j],
               [-0.5+0.j,  0.5+0.j,  0.5+0.j, -0.5+0.j],
               [ 0.5+0.j,  0.5+0.j,  0.5+0.j,  0.5+0.j],
               [-0.5+0.j, -0.5+0.j,  0.5+0.j,  0.5+0.j]])

    """
    from qiskit import quantum_info

    return quantum_info.random_clifford(
        nb_qubits
    ).to_matrix()  # pyright: ignore[reportReturnType]


def rand_unitary_2x2_matrix() -> npt.NDArray[np.complex64]:
    """Generate a random one-qubit unitary matrix.

    Args:
        size: Size (number of columns) of the square matrix to generate.

    Returns:
        A random Clifford matrix.

    Examples:
        >>> rand_unitary_2x2_matrix() # doctest: +SKIP
        array([[ 0.86889957+0.j        ,  0.44138577+0.22403602j],
               [-0.44138577-0.22403602j, -0.72981565-0.47154594j]])

    """
    theta, phi, gamma = np.random.rand(3) * 2 * math.pi
    c, s, eg, ep = (
        np.cos(theta / 2),
        np.sin(theta / 2),
        np.exp(gamma * 1j),
        np.exp(phi * 1j),
    )
    return np.array([[c, -eg * s], [eg * s, eg * ep * c]])


def rand_product_local_unitaries(nb_qubits: int) -> npt.NDArray[np.complex64]:
    """Generate a pseudo random matrix, resulting from a tensor product of
    random unitary matrices.

    Args:
        nb_qubits: Number of qubits on which the product of unitaries will act.

    Returns:
        A tensor product of random unitary matrices.

    Example:
        >>> rand_product_local_unitaries(2) # doctest: +SKIP
        array([[-0.39648015+0.j        ,  0.49842218-0.16609181j,
                   0.39826454-0.21692223j, -0.40979321+0.43953607j],
               [-0.49842218+0.16609181j,  0.14052896-0.37073997j,
                   0.40979321-0.43953607j,  0.06167784+0.44929471j],
               [-0.39826454+0.21692223j,  0.40979321-0.43953607j,
                   0.16112375-0.36226461j, -0.05079312+0.52290651j],
               [-0.40979321+0.43953607j, -0.06167784-0.44929471j,
                   0.05079312-0.52290651j,  0.28163685+0.27906487j]])

    """
    return reduce(np.kron, [rand_unitary_2x2_matrix() for _ in range(nb_qubits - 1)])


def rand_hermitian_matrix(size: int) -> npt.NDArray[np.complex64]:
    """Generate a random Hermitian matrix.

    Args:
        size: Size (number of columns) of the square matrix to generate.

    Returns:
        A random Hermitian Matrix.

    Example:
        >>> rand_hermitian_matrix(2) # doctest: +SKIP
        array([[1.4488624 +0.j, 0.20804943+0.j],
               [0.20804943+0.j, 0.7826408 +0.j]], dtype=complex64)

    """
    m = np.random.rand(size, size).astype(np.complex64)
    return m + m.conjugate().transpose()
