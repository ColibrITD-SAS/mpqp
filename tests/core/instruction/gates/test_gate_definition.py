import numpy as np
import numpy.typing as npt
import pytest

from mpqp.core.instruction.gates.custom_gate import CustomGate
from mpqp.core.instruction.gates.gate_definition import UnitaryMatrix
from mpqp.tools.maths import is_unitary
from mpqp.gates import symbols


def test_custom_gate_created():
    definition = UnitaryMatrix(np.array([[1, 0], [0, 1j]]))
    assert is_unitary(CustomGate(definition, [0, 1]).to_matrix())


@pytest.mark.parametrize(
    "matrix_1, matrix_2",
    [
        (np.array([[1, 0], [0, -1]]), np.array([[3, 0], [0, -3.0]]) / 3),
    ],
)
def test_unitary_matrix_is_equivalent(
    matrix_1: npt.NDArray[np.complex64], matrix_2: npt.NDArray[np.complex64]
):
    unitary_matrix_1 = UnitaryMatrix(matrix_1)
    unitary_matrix_2 = UnitaryMatrix(matrix_2)
    assert unitary_matrix_1.is_equivalent(unitary_matrix_2)


@pytest.mark.parametrize(
    "matrix",
    [np.array([[symbols("theta"), 0], [0, 1]])],
)
def test_unitary_matrix_inverse_failing(matrix: npt.NDArray[np.complex64]):
    with pytest.raises(
        ValueError,
        match="Cannot invert arbitrary gates using symbolic variables",
    ):
        with pytest.warns(
            UserWarning,
            match="Cannot ensure that a operator defined with variables is unitary.",
        ):
            UnitaryMatrix(matrix).inverse()


@pytest.mark.parametrize(
    "matrix",
    [np.array([[1, 0], [0, 1]])],
)
def test_unitary_matrix_inverse_(matrix: npt.NDArray[np.complex64]):
    unit = UnitaryMatrix(matrix)
    assert unit.is_equivalent(unit.inverse())
