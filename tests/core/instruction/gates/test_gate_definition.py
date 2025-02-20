import numpy as np
import numpy.typing as npt
import pytest
from sympy import symbols

from mpqp.core.instruction.gates.gate_definition import UnitaryMatrix
from mpqp.core.instruction.gates.native_gates import *


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
    with pytest.warns(
        UserWarning,
        match="Cannot ensure that a operator defined with symbolic variables is unitary.",
    ):
        unitary_matrix = UnitaryMatrix(matrix)
    with pytest.raises(
        ValueError,
        match="Cannot invert arbitrary gates using symbolic variables",
    ):
        unitary_matrix.inverse()


@pytest.mark.parametrize(
    "matrix",
    [np.array([[1, 0], [0, 1]])],
)
def test_unitary_matrix_inverse_(matrix: npt.NDArray[np.complex64]):
    unit = UnitaryMatrix(matrix)
    assert unit.is_equivalent(unit.inverse())


@pytest.mark.parametrize(
    "gate, matrix",
    [
        (
            SWAP(0, 1),
            np.array(
                [
                    [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
                ]
            ),
        ),
        (
            SWAP(0, 2),
            np.array(
                [
                    [
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                    ],
                ]
            ),
        ),
        (
            CNOT(0, 1),
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ),
        ),
        (
            CNOT(1, 0),
            np.array(
                [
                    [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                ]
            ),
        ),
        (
            CNOT(0, 2),
            np.array(
                [
                    [
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                ]
            ),
        ),
        (
            CNOT(1, 2),
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ),
        ),
        (
            TOF([0, 1], 2),
            np.array(
                [
                    [
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                ]
            ),
        ),
        (
            TOF([0, 2], 1),
            np.array(
                [
                    [
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                ]
            ),
        ),
        (
            CZ(0, 1),
            np.array(
                [
                    [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -1.0 + 0.0j],
                ]
            ),
        ),
        (
            CRk(2, 0, 1),
            np.array(
                [
                    [
                        1.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                    ],
                    [
                        0.000000e00 + 0.0j,
                        1.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                    ],
                    [
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        1.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                    ],
                    [
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        6.123234e-17 + 1.0j,
                    ],
                ]
            ),
        ),
        (
            CRk(2, 1, 0),
            np.array(
                [
                    [
                        1.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                    ],
                    [
                        0.000000e00 + 0.0j,
                        1.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                    ],
                    [
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        1.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                    ],
                    [
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        6.123234e-17 + 1.0j,
                    ],
                ]
            ),
        ),
        (
            CRk(2, 2, 0),
            np.array(
                [
                    [
                        1.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                    ],
                    [
                        0.000000e00 + 0.0j,
                        1.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                    ],
                    [
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        1.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                    ],
                    [
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        1.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                    ],
                    [
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        1.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                    ],
                    [
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        6.123234e-17 + 1.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                    ],
                    [
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        1.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                    ],
                    [
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        0.000000e00 + 0.0j,
                        6.123234e-17 + 1.0j,
                    ],
                ]
            ),
        ),
        (
            CRk(3, 0, 2),
            np.array(
                [
                    [
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.70710678 + 0.70710678j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        1.0 + 0.0j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.70710678 + 0.70710678j,
                    ],
                ]
            ),
        ),
    ],
)
def test_gate_to_matrix(gate: Gate, matrix: npt.NDArray[np.complex64]):
    gate_matrix = UnitaryMatrix(gate.to_matrix())
    assert gate_matrix.is_equivalent(UnitaryMatrix(matrix))
