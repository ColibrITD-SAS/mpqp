from numbers import Real

import numpy as np
import pytest

from mpqp.core.instruction.measurement.pauli_string import I, PauliString, X, Y, Z
from mpqp.tools.generics import Matrix
from mpqp.tools.maths import matrix_eq
from mpqp.tools.obs_decomposition import decompose_hermitian_matrix_ptdr


def list_matrix_pauli_string():
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
        (
            np.array([[2, 3], [3, 1]]),
            3 / 2 * I + 3 * X + 1 / 2 * Z
        ),
        (
            np.array([[-1, 1 - 1j], [1 + 1j, 0]]),
            -1 / 2 * I + X - Y - 1 / 2 * Z
        ),
        (
            np.diag([-2, 4, 5, 3]),
            5 / 2 * I @ I - I @ Z - 3 / 2 * Z @ I - 2 * Z @ Z
        ),
        (
            np.diag([2, 0, 1, 7]),
            5 / 2 * I @ I - I @ Z - 3 / 2 * Z @ I + 2 * Z @ Z
        ),
        (
            np.diag([-2, -3, 2, 1]),
            -1 / 2 * I @ I + 1 / 2 * I @ Z - 2 * Z @ I
        ),
    ]


def list_diagonal_elements_pauli_string():
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
    ]


@pytest.mark.parametrize("matrix, ps", list_matrix_pauli_string())
def test_matrix_to_pauli(matrix: Matrix, ps: PauliString):
    assert PauliString.from_matrix(matrix, method="ptdr") == ps
    assert PauliString.from_matrix(matrix, method="trace") == ps


@pytest.mark.parametrize("diag, ps", list_diagonal_elements_pauli_string())
def test_diagonal_elements_to_pauli(diag: list[Real], ps: PauliString):
    assert PauliString.from_diagonal_elements(diag, method="ptdr") == ps
    assert PauliString.from_diagonal_elements(diag, method="walsh") == ps


@pytest.mark.parametrize("matrix, ps", list_matrix_pauli_string())
def test_pauli_to_matrix(matrix: Matrix, ps: PauliString):
    assert matrix_eq(ps.to_matrix(), matrix)


@pytest.mark.parametrize("matrix, ps", list_matrix_pauli_string())
def test_matrix_to_pauli_to_matrix(matrix: Matrix, ps: PauliString):
    assert matrix_eq(PauliString.from_matrix(matrix, method="ptdr").to_matrix(), matrix)
    assert matrix_eq(
        PauliString.from_matrix(matrix, method="trace").to_matrix(), matrix
    )


@pytest.mark.parametrize("matrix, ps", list_matrix_pauli_string())
def test_pauli_to_matrix_to_pauli(matrix: Matrix, ps: PauliString):
    assert PauliString.from_matrix(ps.to_matrix(), method="ptdr") == ps
    assert PauliString.from_matrix(ps.to_matrix(), method="trace") == ps


# @pytest.mark.parametrize(
#     "matrix, pauliString",
#     [
#         (np.array([[2, 3], [3, 1]]), 3 / 2 * I + 3 * X + 1 / 2 * Z),
#         (np.array([[-1, 1 - 1j], [1 + 1j, 0]]), -1 / 2 * I + X - Y - 1 / 2 * Z),
#         (np.diag([-2, 4, 5, 3]), 5 / 2 * I @ I - I @ Z - 3 / 2 * Z @ I + 2 * Z @ Z),
#         (np.diag([-2, -3, 2, 1]), -1 / 2 * I @ I + 1 / 2 * I @ Z - 2 * Z @ I),
#     ],
# )
# def test_decompose_general_observable(matrix: Matrix, pauliString: PauliString):
#     decomposed_pauli_string = decompose_hermitian_matrix_ptdr(matrix)
#
#     assert decomposed_pauli_string == pauliString, (
#         f"Decomposition failed for matrix:\n{matrix}\n"
#         f"Expected: {pauliString}\n"
#         f"Got: {decomposed_pauli_string}"
#     )
