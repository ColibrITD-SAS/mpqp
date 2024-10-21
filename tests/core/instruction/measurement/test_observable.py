import numpy as np
import pytest

from mpqp.core.instruction.measurement.pauli_string import I, PauliString, X, Y, Z
from mpqp.tools.generics import Matrix
from mpqp.tools.maths import matrix_eq


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
    ]


@pytest.mark.parametrize("matrix, ps", list_matrix_pauli_string())
def test_matrix_to_pauli(matrix: Matrix, ps: PauliString):
    assert PauliString().from_matrix(matrix) == ps


@pytest.mark.parametrize("matrix, ps", list_matrix_pauli_string())
def test_pauli_to_matrix(matrix: Matrix, ps: PauliString):
    assert matrix_eq(ps.to_matrix(), matrix)


@pytest.mark.parametrize("matrix, ps", list_matrix_pauli_string())
def test_matrix_to_pauli_to_matrix(matrix: Matrix, ps: PauliString):
    assert matrix_eq(PauliString().from_matrix(matrix).to_matrix(), matrix)


@pytest.mark.parametrize("matrix, ps", list_matrix_pauli_string())
def test_pauli_to_matrix_to_pauli(matrix: Matrix, ps: PauliString):
    print(PauliString().from_matrix(ps.to_matrix()))
    assert PauliString().from_matrix(ps.to_matrix()) == ps
