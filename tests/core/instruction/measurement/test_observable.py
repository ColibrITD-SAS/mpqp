from typing import Union

import numpy as np
import numpy.typing as npt
import pytest
from mpqp.core.instruction import Observable
from mpqp.core.instruction.measurement.pauli_string import I, PauliString, X, Y, Z
from mpqp.tools.generics import Matrix
from mpqp.tools.maths import matrix_eq


@pytest.fixture
def list_matrix_pauli_string() -> list[tuple[Matrix, PauliString]]:
    return [
        (
            np.array(
                [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]],
                dtype=np.complex64,
            ),
            I @ Z + I @ I,
        ),
        (
            np.zeros((4, 4), dtype=np.complex64),
            I @ I - I @ I,
        ),
        (
            np.kron(I.matrix, Y.matrix) - np.kron(I.matrix, Z.matrix),
            I @ Y - I @ Z,
        ),
        (
            np.kron(I.matrix, Y.matrix)
            + 4 * np.kron(I.matrix, Z.matrix)
            + 5 * np.kron(Y.matrix, Z.matrix),
            I @ Y + 4 * I @ Z + 5 * Y @ Z,
        ),
        (
            I.matrix - I.matrix,
            I - I,
        ),
        (
            I.matrix + I.matrix,
            I + I,
        ),
        (
            np.kron(X.matrix, Y.matrix)
            + np.kron(Y.matrix, Z.matrix)
            - np.kron(Z.matrix, X.matrix),
            X @ Y + Y @ Z - Z @ X,
        ),
        (
            np.kron(Z.matrix, X.matrix)
            + np.kron(Y.matrix, Y.matrix)
            - np.kron(X.matrix, Z.matrix),
            Z @ X + Y @ Y - X @ Z,
        ),
        (
            np.kron(Y.matrix, X.matrix) - np.kron(X.matrix, Y.matrix),
            Y @ X - X @ Y,
        ),
        (np.array([[2, 3], [3, 1]], dtype=np.complex64), 3 / 2 * I + 3 * X + 1 / 2 * Z),
        (
            np.array([[-1, 1 - 1j], [1 + 1j, 0]], dtype=np.complex64),
            -1 / 2 * I + X - Y - 1 / 2 * Z,
        ),
        (np.diag([-2, 4, 5, 3]), 5 / 2 * I @ I - I @ Z - 3 / 2 * Z @ I - 2 * Z @ Z),
        (np.diag([2, 0, 1, 7]), 5 / 2 * I @ I - I @ Z - 3 / 2 * Z @ I + 2 * Z @ Z),
        (np.diag([-2, -3, 2, 1]), -1 / 2 * I @ I + 1 / 2 * I @ Z - 2 * Z @ I),
        (np.zeros((4, 4), dtype=np.complex64), 1 * I @ I - 1 * I @ I),
    ]


@pytest.fixture
def list_diagonal_elements_pauli_string() -> list[tuple[list[float], PauliString]]:
    return [
        (
            [-2, 4, 5, 3],
            5 / 2 * I @ I - I @ Z - 3 / 2 * Z @ I - 2 * Z @ Z,
        ),
        (
            [2, 0, 1, 7],
            5 / 2 * I @ I - I @ Z - 3 / 2 * Z @ I + 2 * Z @ Z,
        ),
        (
            [-2, -3, 2, 1],
            -1 / 2 * I @ I + 1 / 2 * I @ Z - 2 * Z @ I,
        ),
        ([0, 0, 0, 0], 1 * I @ I - 1 * I @ I),
    ]


@pytest.fixture
def list_diagonal_observable_inputs() -> list[Union[Matrix, PauliString, list[float]]]:
    return [
        [-2, 4, 5, 3],
        [-2, -3, 2, 1],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        5 * I @ I - I @ Z - 3 * Z @ I - 2 * Z @ Z,
        -1 * I @ I + 3 / 2 * I @ Z - 7 * Z @ I,
        Z @ Z + Z @ I - I @ Z,
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
    list_matrix_pauli_string: list[tuple[npt.NDArray[np.complex64], PauliString]],
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
            np.zeros((2**o.nb_qubits, 2**o.nb_qubits)).astype(np.complex64),
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
