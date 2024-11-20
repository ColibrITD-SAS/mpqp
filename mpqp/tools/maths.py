"""Mathematical tools for linear algebra, functions generalized to more data 
types, etc…"""

from __future__ import annotations

import math
from functools import reduce
from numbers import Complex, Real
from typing import TYPE_CHECKING, Optional, Union

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

    for elt in zip(np.ndarray.flatten(lhs), np.ndarray.flatten(rhs)):
        try:
            if abs(elt[0] - elt[1]) > (atol + rtol * abs(elt[1])):
                return False
        except TypeError:
            if elt[0] != elt[1]:
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
        >>> is_hermitian(np.array([[1,2j,3j],[-2j,4,5j],[-3j,-5j,6]]))
        True
        >>> is_hermitian(np.diag([1,2,3,4]))
        True
        >>> m3 = np.array([[1,2,3],[2,4,5],[3,5,6]])
        >>> is_hermitian(np.array([[1,2,3],[2,4,5],[3,5,6]]))
        True
        >>> m4 = np.array([[1,2,3],[4,5,6],[7,8,9]])
        >>> is_hermitian(np.array([[1,2,3],[4,5,6],[7,8,9]]))
        False
        >>> x = symbols("x", real=True)
        >>> is_hermitian(np.diag([1,x]))
        True
        >>> is_hermitian(np.array([[1,x],[-x,2]]))
        False

    """
    return matrix_eq(
        np.array(matrix).transpose().conjugate(),  # pyright: ignore[reportArgumentType]
        matrix,
    )


@typechecked
def is_unitary(matrix: Matrix) -> bool:
    """Checks whether the matrix in parameter is unitary.

    Args:
        matrix: Matrix for which we want to know if it is unitary.

    Returns:
        ``True`` if the matrix in parameter is Unitary.

    Example:
        >>> is_unitary(np.array([[1,1],[1,-1]]))
        False
        >>> is_unitary(np.array([[1,1],[1,-1]])/np.sqrt(2))
        True

    """
    return matrix_eq(
        np.eye(len(matrix), dtype=np.complex64),
        matrix.transpose().conjugate().dot(matrix),
    )


@typechecked
def closest_unitary(matrix: Matrix):
    """Calculate the unitary matrix that is closest with respect to the operator
    norm distance to the general matrix in parameter.

    Args:
        matrix: Matrix for which we want to determine the closest unitary matrix.

        Return U as a numpy matrix.

    Example:
        >>> is_unitary(np.array([[1, 2], [3, 4]]))
        False
        >>> u = closest_unitary(np.array([[1, 2], [3, 4]]))
        >>> u
        array([[-0.51449576,  0.85749293],
               [ 0.85749293,  0.51449576]])
        >>> is_unitary(u)
        True

    """
    from scipy.linalg import svd

    V, _, Wh = svd(matrix)
    return np.array(V.dot(Wh))


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
        if TYPE_CHECKING:
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
        if TYPE_CHECKING:
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
        if TYPE_CHECKING:
            assert isinstance(res, Expr)
        return res


@typechecked
def rand_orthogonal_matrix(
    size: int, seed: Optional[int] = None
) -> npt.NDArray[np.complex64]:
    """Generate a random orthogonal matrix optionally with a given seed.

    Args:
        size: Size (number of columns) of the square matrix to generate.
        seed: Seed used to initialize the random number generation.

    Returns:
        A random orthogonal matrix.

    Examples:
        >>> rand_orthogonal_matrix(3)
        array([[ 0.70957328,  0.1395875 ,  0.69066713],
               [ 0.61432236,  0.35754246, -0.7033999 ],
               [-0.34512866,  0.92340604,  0.16795085]])

        >>> rand_orthogonal_matrix(3, seed=123)
        array([[ 0.75285974, -0.65782143,  0.02175293],
               [-0.22777817, -0.22939368,  0.94630632],
               [ 0.61751058,  0.71739077,  0.32253863]])

    """
    rng = np.random.default_rng(seed)

    m = rng.random((size, size))
    return m.dot(inv(sqrtm(m.T.dot(m))))


@typechecked
def rand_clifford_matrix(
    nb_qubits: int, seed: Optional[int] = None
) -> npt.NDArray[np.complex64]:
    """Generate a random Clifford matrix.

    Args:
        size: Size (number of columns) of the square matrix to generate.
        seed: Seed used to initialize the random number generation.

    Returns:
        A random Clifford matrix.

    Examples:
        >>> rand_clifford_matrix(2)
        array([[-0.5+0.j ,  0.5+0.j ,  0. +0.5j,  0. -0.5j],
               [ 0. -0.5j,  0. -0.5j,  0.5+0.j ,  0.5+0.j ],
               [ 0. -0.5j,  0. -0.5j, -0.5+0.j , -0.5+0.j ],
               [-0.5+0.j ,  0.5+0.j ,  0. -0.5j,  0. +0.5j]])

        >>> rand_clifford_matrix(2, seed=123)
        array([[0.+0.70710678j, 0.+0.j        , 0.-0.70710678j, 0.+0.j        ],
               [0.+0.j        , 0.-0.70710678j, 0.+0.j        , 0.-0.70710678j],
               [0.+0.j        , 0.+0.70710678j, 0.+0.j        , 0.-0.70710678j],
               [0.+0.70710678j, 0.+0.j        , 0.+0.70710678j, 0.+0.j        ]])

    """
    from qiskit import quantum_info

    rng = np.random.default_rng(seed)

    res = quantum_info.random_clifford(nb_qubits, seed=rng).to_matrix()
    if TYPE_CHECKING:
        assert isinstance(res, np.ndarray)
    return res


@typechecked
def rand_unitary_2x2_matrix(
    seed: Optional[Union[int, np.random.Generator]] = None
) -> npt.NDArray[np.complex64]:
    """Generate a random one-qubit unitary matrix.

    Args:
        size: Size (number of columns) of the square matrix to generate.
        seed: Used for the random number generation. If unspecified, a new
            generator will be used. If a ``Generator`` is provided, it will be
            used to generate any random number needed. Finally if an ``int`` is
            provided, it will be used to initialize a new generator.

    Returns:
        A random Clifford matrix.

    Examples:
        >>> rand_unitary_2x2_matrix()
        array([[-0.44233606+0.j        , -0.57071368-0.69182707j],
               [ 0.57071368+0.69182707j,  0.27325473+0.34784055j]])

        >>> rand_unitary_2x2_matrix(seed=123)
        array([[-0.54205051+0.j        , -0.15559823-0.82581501j],
               [ 0.15559823+0.82581501j,  0.08203889-0.53580629j]])

    """
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    theta, phi, gamma = rng.random(3) * 2 * math.pi
    c, s, eg, ep = (
        np.cos(theta / 2),
        np.sin(theta / 2),
        np.exp(gamma * 1j),
        np.exp(phi * 1j),
    )
    return np.array([[c, -eg * s], [eg * s, eg * ep * c]])


@typechecked
def rand_product_local_unitaries(
    nb_qubits: int, seed: Optional[int] = None
) -> npt.NDArray[np.complex64]:
    """Generate a pseudo random matrix, resulting from a tensor product of
    random unitary matrices.

    Args:
        nb_qubits: Number of qubits on which the product of unitaries will act.
        seed: Seed used to initialize the random number generation.

    Returns:
        A tensor product of random unitary matrices.

    Example:
        >>> rand_product_local_unitaries(2)
        array([[ 0.07058754-0.j        ,  0.43591425+0.02563897j,
                 0.09107391+0.11040107j,  0.52232796+0.71486324j],
               [-0.43591425-0.02563897j, -0.05999574-0.03719022j,
                -0.52232796-0.71486324j, -0.01924144-0.14181897j],
               [-0.09107391-0.11040107j, -0.52232796-0.71486324j,
                -0.04360571-0.05550804j, -0.24912587-0.35862918j],
               [ 0.52232796+0.71486324j,  0.01924144+0.14181897j,
                 0.24912587+0.35862918j,  0.00781725+0.07015334j]])

        >>> rand_product_local_unitaries(2, seed=123)
        array([[-0.45363624+0.j        ,  0.11284472-0.27440661j,
                -0.13021848-0.6911157j ,  0.45045163+0.0931494j ],
               [-0.11284472+0.27440661j, -0.4523475 +0.03416981j,
                -0.45045163-0.0931494j , -0.18190632-0.67934369j],
               [ 0.13021848+0.6911157j , -0.45045163-0.0931494j ,
                 0.06865747-0.44841051j,  0.25416659+0.153076j  ],
               [ 0.45045163+0.0931494j ,  0.18190632+0.67934369j,
                -0.25416659-0.153076j  ,  0.03468623-0.45230819j]])

    """
    rng = np.random.default_rng(seed)

    return reduce(np.kron, [rand_unitary_2x2_matrix(rng) for _ in range(nb_qubits)])


@typechecked
def rand_hermitian_matrix(
    size: int, seed: Optional[int] = None
) -> npt.NDArray[np.complex64]:
    """Generate a random Hermitian matrix.

    Args:
        size: Size (number of columns) of the square matrix to generate.
        seed: Seed used to initialize the random number generation.

    Returns:
        A random Hermitian Matrix.

    Example:
        >>> rand_hermitian_matrix(2)
        array([[1.2917002 +0.j, 0.64402145+0.j],
               [0.64402145+0.j, 1.1020273 +0.j]], dtype=complex64)

        >>> rand_hermitian_matrix(2, seed=123)
        array([[1.3647038 +0.j, 0.2741809 +0.j],
               [0.2741809 +0.j, 0.36874363+0.j]], dtype=complex64)

    """
    rng = np.random.default_rng(seed)

    m = rng.random((size, size)).astype(np.complex64)
    return m + m.conjugate().transpose()


@typechecked
def is_power_of_two(n: int):
    """Checks if the integer in parameter is a (positive) power of two.

    Args:
        n: Integer for which we want to check if it is a power of two.

    Returns:
        True if the integer in parameter is a power of two.

    Example:
        >>> is_power_of_two(4)
        True
        >>> is_power_of_two(6)
        False

    """
    return n >= 1 and (n & (n - 1)) == 0
