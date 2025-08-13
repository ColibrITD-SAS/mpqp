from typing import Union

import numpy as np
import numpy.typing as npt
import pytest
from mpqp.core.instruction import Observable
from mpqp.core.instruction.measurement.pauli_string import PI, PauliString, PX, PY, PZ
from mpqp.tools.generics import Matrix
from mpqp.tools.maths import matrix_eq


@pytest.fixture
def list_matrix_pauli_string() -> list[tuple[Matrix, PauliString]]:
    return [
        (
            np.array(
                [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]],
                dtype=np.complex128,
            ),
            PI @ PZ + PI @ PI,
        ),
        (
            np.zeros((4, 4), dtype=np.complex128),
            PI @ PI - PI @ PI,
        ),
        (
            np.kron(PI.matrix, PY.matrix) - np.kron(PI.matrix, PZ.matrix),
            PI @ PY - PI @ PZ,
        ),
        (
            np.kron(PI.matrix, PY.matrix)
            + 4 * np.kron(PI.matrix, PZ.matrix)
            + 5 * np.kron(PY.matrix, PZ.matrix),
            PI @ PY + 4 * PI @ PZ + 5 * PY @ PZ,
        ),
        (
            PI.matrix - PI.matrix,
            PI - PI,
        ),
        (
            PI.matrix + PI.matrix,
            PI + PI,
        ),
        (
            np.kron(PX.matrix, PY.matrix)
            + np.kron(PY.matrix, PZ.matrix)
            - np.kron(PZ.matrix, PX.matrix),
            PX @ PY + PY @ PZ - PZ @ PX,
        ),
        (
            np.kron(PZ.matrix, PX.matrix)
            + np.kron(PY.matrix, PY.matrix)
            - np.kron(PX.matrix, PZ.matrix),
            PZ @ PX + PY @ PY - PX @ PZ,
        ),
        (
            np.kron(PY.matrix, PX.matrix) - np.kron(PX.matrix, PY.matrix),
            PY @ PX - PX @ PY,
        ),
        (
            np.array([[2, 3], [3, 1]], dtype=np.complex128),
            3 / 2 * PI + 3 * PX + 1 / 2 * PZ,
        ),
        (
            np.array([[-1, 1 + 1j], [1 - 1j, 0]], dtype=np.complex128),
            -1 / 2 * PI + PX - PY - 1 / 2 * PZ,
        ),
        (
            np.diag([-2, 4, 5, 3]),
            5 / 2 * PI @ PI - PI @ PZ - 3 / 2 * PZ @ PI - 2 * PZ @ PZ,
        ),
        (
            np.diag([2, 0, 1, 7]),
            5 / 2 * PI @ PI - PI @ PZ - 3 / 2 * PZ @ PI + 2 * PZ @ PZ,
        ),
        (np.diag([-2, -3, 2, 1]), -1 / 2 * PI @ PI + 1 / 2 * PI @ PZ - 2 * PZ @ PI),
        (np.zeros((4, 4), dtype=np.complex128), 1 * PI @ PI - 1 * PI @ PI),
    ]


@pytest.fixture
def list_diagonal_elements_pauli_string() -> list[tuple[list[float], PauliString]]:
    return [
        (
            [-2, 4, 5, 3],
            5 / 2 * PI @ PI - PI @ PZ - 3 / 2 * PZ @ PI - 2 * PZ @ PZ,
        ),
        (
            [2, 0, 1, 7],
            5 / 2 * PI @ PI - PI @ PZ - 3 / 2 * PZ @ PI + 2 * PZ @ PZ,
        ),
        (
            [-2, -3, 2, 1],
            -1 / 2 * PI @ PI + 1 / 2 * PI @ PZ - 2 * PZ @ PI,
        ),
        ([0, 0, 0, 0], 1 * PI @ PI - 1 * PI @ PI),
    ]


@pytest.fixture
def list_diagonal_observable_inputs() -> list[Union[Matrix, PauliString, list[float]]]:
    return [
        [-2, 4, 5, 3],
        [-2, -3, 2, 1],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        5 * PI @ PI - PI @ PZ - 3 * PZ @ PI - 2 * PZ @ PZ,
        -1 * PI @ PI + 3 / 2 * PI @ PZ - 7 * PZ @ PI,
        PZ @ PZ + PZ @ PI - PI @ PZ,
        np.array([[1, 0], [0, -6]]),
        np.array([[1, 0, 0, 0], [0, 3, 0, 0], [0, 0, 6, 0], [0, 0, 0, -6]]),
        np.diag([3, 2, 5, 4, 2, 5, 4, 3]),
    ]


def test_matrix_to_pauli(list_matrix_pauli_string: list[tuple[Matrix, PauliString]]):
    for matrix, ps in list_matrix_pauli_string:
        assert PauliString.from_matrix(matrix, method="ptdr") == ps
        assert PauliString.from_matrix(matrix, method="trace") == ps


def test_diagonal_elements_to_pauli(
    list_diagonal_elements_pauli_string: list[tuple[list[float], PauliString]],
):
    for diag, ps in list_diagonal_elements_pauli_string:
        assert PauliString.from_diagonal_elements(diag, method="ptdr") == ps
        assert PauliString.from_diagonal_elements(diag, method="walsh") == ps
        assert Observable(diag).pauli_string == ps


def test_pauli_to_matrix(
    list_matrix_pauli_string: list[tuple[npt.NDArray[np.complex128], PauliString]],
):
    for matrix, ps in list_matrix_pauli_string:
        assert matrix_eq(ps.to_matrix(), matrix)
        assert matrix_eq(Observable(ps).matrix, matrix)


def test_matrix_to_pauli_to_matrix(
    list_matrix_pauli_string: list[tuple[Matrix, PauliString]],
):
    for matrix, _ in list_matrix_pauli_string:
        assert matrix_eq(
            PauliString.from_matrix(matrix, method="ptdr").to_matrix(), matrix
        )
        assert matrix_eq(
            PauliString.from_matrix(matrix, method="trace").to_matrix(), matrix
        )


def test_pauli_to_matrix_to_pauli(
    list_matrix_pauli_string: list[tuple[Matrix, PauliString]],
):
    for _, ps in list_matrix_pauli_string:
        assert PauliString.from_matrix(ps.to_matrix(), method="ptdr") == ps
        assert PauliString.from_matrix(ps.to_matrix(), method="trace") == ps
        assert Observable(ps.to_matrix()).pauli_string == ps


def test_diagonal_observable_attributes(
    list_diagonal_observable_inputs: list[Union[Matrix, PauliString, list[float]]],
):
    for ii in list_diagonal_observable_inputs:
        o = Observable(ii)
        assert o.is_diagonal is True
        assert o.pauli_string.is_diagonal() is True
        assert matrix_eq(
            np.diag(o.diagonal_elements) - o.matrix,
            np.zeros((2**o.nb_qubits, 2**o.nb_qubits)),
        )


def test_repr_observable_from_diag_elements():
    o = Observable([1, 2, 3, 4])
    repr_o = o.__repr__()
    oo = eval(repr_o)
    assert (
        oo._matrix is None and o._matrix is None  # pyright: ignore[reportPrivateUsage]
    )
    assert oo._is_diagonal == o._is_diagonal  # pyright: ignore[reportPrivateUsage]
    assert (
        oo._pauli_string is None
        and o._pauli_string is None  # pyright: ignore[reportPrivateUsage]
    )
    assert matrix_eq(
        oo._diag_elements,
        np.array(o._diag_elements),  # pyright: ignore[reportPrivateUsage]
    )
