from numbers import Real

import numpy as np
import pytest
from mpqp.core.instruction import PauliString
from mpqp.measures import I, X, Y, Z

from mpqp.tools.generics import Matrix


@pytest.mark.parametrize(
    "matrix, pauliString",
    [
        (np.array([[2, 3], [3, 1]]), 3 / 2 * I + 3 * X + 1 / 2 * Z),
        (np.array([[-1, 1 - 1j], [1 + 1j, 0]]), -1 / 2 * I + X - Y - 1 / 2 * Z),
        (np.diag([-2, 4, 5, 3]), 5 / 2 * I @ I - I @ Z - 3 / 2 * Z @ I + 2 * Z @ Z),
        (np.diag([-2, -3, 2, 1]), -1 / 2 * I @ I + 1 / 2 * I @ Z - 2 * Z @ I),
    ],
)
def test_decompose_general_observable(matrix: Matrix, pauliString: PauliString): ...


@pytest.mark.parametrize(
    "diag, pauliString",
    [
        ([-2, 4, 5, 3], 5 / 2 * I @ I - I @ Z - 3 / 2 * Z @ I + 2 * Z @ Z),
    ],
)
def test_decompose_diagonal_observable(diag: list[Real], pauliString: PauliString): ...
