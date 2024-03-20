import numpy as np
import numpy.typing as npt
import pytest

from mpqp.core.instruction.measurement.pauli_string import I, PauliString
from mpqp.tools.maths import matrix_eq


@pytest.mark.parametrize(
    "ps, matrix",
    [
        (I @ I, np.eye(4)),
        (I @ (I @ I), np.eye(8)),
        ((I @ I) @ I, np.eye(8)),
        (I @ (I + I), 2 * np.eye(4)),
        ((I + I) @ I, 2 * np.eye(4)),
        (I / 2, np.eye(2) / 2),
        ((1 * I) / 2, np.eye(2) / 2),
        ((I + I) / 2, np.eye(2)),
        (2 * I, np.eye(2) * 2),
        (2 * (2 * I), np.eye(2) * 4),
        (2 * (I + I), np.eye(2) * 4),
        (I * 2, np.eye(2) * 2),
        ((2 * I) * 2, np.eye(2) * 4),
        ((I + I) * 2, np.eye(2) * 4),
    ],
)
def test_operations(ps: PauliString, matrix: npt.NDArray[np.complex64]):
    assert matrix_eq(ps.to_matrix(), matrix)
