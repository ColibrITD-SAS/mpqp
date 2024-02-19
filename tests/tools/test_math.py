import pytest
import numpy as np

from mpqp.tools.generics import Matrix
from mpqp.gates import symbols
from mpqp.tools.maths import is_hermitian

x = symbols("x", real=True)


@pytest.mark.parametrize(
    "matrix, isHermitian", [
        (np.array([[1, 2j, 3j], [-2j, 4, 5j], [-3j, -5j, 6]]), True),
        (np.diag([1, 2, 3, 4]), True),
        (np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]]), True),
        (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), False),
        (np.diag([1, x]), True),
        (np.array([[1, x], [-x, 2]]), False)
    ]
)
def test_is_hermitian(matrix: Matrix, isHermitian: bool):
    assert is_hermitian(matrix) == isHermitian
