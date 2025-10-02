from typing import Union

import numpy as np
import numpy.typing as npt
import pytest
from mpqp.core.instruction import Observable
from mpqp.core.instruction.measurement.pauli_string import pI, PauliString, pX, pY, pZ
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
            pI @ pZ + pI @ pI,
        ),
        (
            np.zeros((4, 4), dtype=np.complex128),
            pI @ pI - pI @ pI,
        ),
        (
            np.kron(pI.matrix, pY.matrix) - np.kron(pI.matrix, pZ.matrix),
            pI @ pY - pI @ pZ,
        ),
        (
            np.kron(pI.matrix, pY.matrix)
            + 4 * np.kron(pI.matrix, pZ.matrix)
            + 5 * np.kron(pY.matrix, pZ.matrix),
            pI @ pY + 4 * pI @ pZ + 5 * pY @ pZ,
        ),
        (
            pI.matrix - pI.matrix,
            pI - pI,
        ),
        (
            pI.matrix + pI.matrix,
            pI + pI,
        ),
        (
            np.kron(pX.matrix, pY.matrix)
            + np.kron(pY.matrix, pZ.matrix)
            - np.kron(pZ.matrix, pX.matrix),
            pX @ pY + pY @ pZ - pZ @ pX,
        ),
        (
            np.kron(pZ.matrix, pX.matrix)
            + np.kron(pY.matrix, pY.matrix)
            - np.kron(pX.matrix, pZ.matrix),
            pZ @ pX + pY @ pY - pX @ pZ,
        ),
        (
            np.kron(pY.matrix, pX.matrix) - np.kron(pX.matrix, pY.matrix),
            pY @ pX - pX @ pY,
        ),
        (
            np.array([[2, 3], [3, 1]], dtype=np.complex128),
            3 / 2 * pI + 3 * pX + 1 / 2 * pZ,
        ),
        (
            np.array([[-1, 1 + 1j], [1 - 1j, 0]], dtype=np.complex128),
            -1 / 2 * pI + pX - pY - 1 / 2 * pZ,
        ),
        (
            np.diag([-2, 4, 5, 3]),
            5 / 2 * pI @ pI - pI @ pZ - 3 / 2 * pZ @ pI - 2 * pZ @ pZ,
        ),
        (
            np.diag([2, 0, 1, 7]),
            5 / 2 * pI @ pI - pI @ pZ - 3 / 2 * pZ @ pI + 2 * pZ @ pZ,
        ),
        (np.diag([-2, -3, 2, 1]), -1 / 2 * pI @ pI + 1 / 2 * pI @ pZ - 2 * pZ @ pI),
        (np.zeros((4, 4), dtype=np.complex128), 1 * pI @ pI - 1 * pI @ pI),
    ]


@pytest.fixture
def list_diagonal_elements_pauli_string() -> list[tuple[list[float], PauliString]]:
    return [
        (
            [-2, 4, 5, 3],
            5 / 2 * pI @ pI - pI @ pZ - 3 / 2 * pZ @ pI - 2 * pZ @ pZ,
        ),
        (
            [2, 0, 1, 7],
            5 / 2 * pI @ pI - pI @ pZ - 3 / 2 * pZ @ pI + 2 * pZ @ pZ,
        ),
        (
            [-2, -3, 2, 1],
            -1 / 2 * pI @ pI + 1 / 2 * pI @ pZ - 2 * pZ @ pI,
        ),
        ([0, 0, 0, 0], 1 * pI @ pI - 1 * pI @ pI),
    ]


@pytest.fixture
def list_diagonal_observable_inputs() -> list[Union[Matrix, PauliString, list[float]]]:
    return [
        [-2, 4, 5, 3],
        [-2, -3, 2, 1],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        5 * pI @ pI - pI @ pZ - 3 * pZ @ pI - 2 * pZ @ pZ,
        -1 * pI @ pI + 3 / 2 * pI @ pZ - 7 * pZ @ pI,
        pZ @ pZ + pZ @ pI - pI @ pZ,
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
        assert o.pauli_string.is_diagonal()
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
