import numpy as np
import pytest
from sympy import symbols

from mpqp.tools.generics import Matrix
from mpqp.tools.maths import is_hermitian, rand_hermitian_matrix, rand_unitary_matrix

x = symbols("x", real=True)


@pytest.mark.parametrize(
    "matrix, isHermitian",
    [
        (np.array([[1, 2j, 3j], [-2j, 4, 5j], [-3j, -5j, 6]]), True),
        (np.diag([1, 2, 3, 4]), True),
        (np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]]), True),
        (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), False),
        (np.diag([1, x]), True),
        (np.array([[1, x], [-x, 2]]), False),
    ],
)
def test_is_hermitian(matrix: Matrix, isHermitian: bool):
    assert is_hermitian(matrix) == isHermitian


def test_rand_hermitian():
    assert is_hermitian(rand_hermitian_matrix(3))


@pytest.mark.parametrize(
    ("matrix", "targets"),
    [
        (rand_unitary_matrix(4), [1, 0]),
        (rand_unitary_matrix(8), [1, 0, 2]),
        (rand_unitary_matrix(8), [2, 0, 1]),
    ],
)
def test_rearrange_matrix(matrix: Matrix, targets: list[int]):
    from mpqp.gates import CustomGate
    from mpqp.tools.maths import rearrange_matrix, matrix_eq
    from mpqp import QCircuit

    g = CustomGate(matrix, targets)
    m = rearrange_matrix(matrix, targets)
    g2 = CustomGate(m, sorted(targets))
    assert matrix_eq(QCircuit([g]).to_matrix(), g2.to_matrix())
