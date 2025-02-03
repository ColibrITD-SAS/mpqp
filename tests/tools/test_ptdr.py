from numbers import Real

import numpy as np
import pytest
from mpqp.core.instruction import PauliString
from mpqp.measures import I, X, Y, Z

from mpqp.tools.generics import Matrix


@pytest.mark.parametrize(
    "matrix, pauliString",
    [
        (np.diag([-2, 4, 5, 3]), 5 / 2 * I @ I - I @ Z - 3 / 2 * Z @ I + 2 * Z @ Z),
    ],
)
def test_decompose_general_observable(matrix: Matrix, pauliString: PauliString): ...


@pytest.mark.parametrize(
    "diag, pauliString",
    [
        ([-2, 4, 5, 3], 5 / 2 * I @ I - I @ Z - 3 / 2 * Z @ I + 2 * Z @ Z),
    ],
)
def test_decompose_general_observable(diag: list[Real], pauliString: PauliString): ...
