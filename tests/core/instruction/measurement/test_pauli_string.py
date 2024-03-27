from itertools import product
from operator import add, matmul, mul, neg, pos, sub, truediv
from random import randint

import numpy as np
import numpy.typing as npt
import pytest

from mpqp.core.instruction.measurement.pauli_string import I, X, Y, Z, PauliString
from mpqp.tools.maths import matrix_eq


def pauli_string_combinations():
    scalar_bin_operation = [mul, truediv]
    homogeneous_bin_operation = [add, sub]
    bin_operation = [matmul]
    un_operation = [pos, neg]
    pauli = [
        (I, np.eye(2)),
        ((I @ I), np.eye(4)),
        ((I + I), (2 * np.eye(2))),
        ((I + I) @ I, (2 * np.eye(4))),
    ]
    result = []

    for ps in pauli:
        for op in scalar_bin_operation:
            a = randint(1, 9)
            result.append((op(ps[0], a), op(ps[1], a)))
        for op in un_operation:
            result.append((op(ps[0]), op(ps[1])))
    for ps_1, ps_2 in product(pauli, repeat=2):
        for op in bin_operation:
            converted_op = op if op != matmul else np.kron
            result.append((op(ps_1[0], ps_2[0]), converted_op(ps_1[1], ps_2[1])))
        if ps_1[0].nb_qubits == ps_2[0].nb_qubits:
            for op in homogeneous_bin_operation:
                result.append((op(ps_1[0], ps_2[0]), op(ps_1[1], ps_2[1])))

    return result


@pytest.mark.parametrize("ps, matrix", pauli_string_combinations())
def test_operations(ps: PauliString, matrix: npt.NDArray[np.complex64]):
    assert matrix_eq(ps.to_matrix(), matrix)


@pytest.mark.parametrize(
    "init_ps, simplified_ps",
    [
        # Test cases with single terms
        (I @ I, I @ I),
        (2 * I @ I, 2 * I @ I),
        (-I @ I, -I @ I),
        (0 * I @ I, 0 * I @ I),
        # Test cases with multiple terms
        (I @ I + I @ I + I @ I, 3 * I @ I),
        (2 * I @ I + 3 * I @ I - 2 * I @ I, 3 * I @ I),
        (2 * I @ I - 3 * I @ I + I @ I, 0),
        (-I @ I + I @ I - I @ I, 0),
        (I @ I - 2 * I @ I + I @ I, I @ I),
        (2 * I @ I + I @ I - I @ I, 2 * I @ I),
        (I @ I + I @ I + I @ I, 3 * I @ I),
        (2 * I @ I + 3 * I @ I, 5 * I @ I),
        (I @ I - I @ I + I @ I, I @ I),
        # Test cases with cancellation
        (I @ I - I @ I, 0 * I @ I),
        (2 * I @ I - 2 * I @ I, 0 * I @ I),
        (-2 * I @ I + 2 * I @ I, 0 * I @ I),
        (I @ I + I @ I - 2 * I @ I, 0 * I @ I),
        # Test cases with mixed terms
        (I @ I - 2 * I @ I + 3 * I @ I, 2 * I @ I),
        (2 * I @ I + I @ I - I @ I, 2 * I @ I),
        (I @ I + I @ I + I @ I - 3 * I @ I, I @ I),
        # Test cases with combinations of different gates
        (I @ X + X @ X - X @ I, 2 * X @ X),
        (Y @ Z + Z @ Y - Z @ Z, Y @ Z + Z @ Y),
        (I @ X + X @ Y - Y @ X - X @ I, 0 * I @ I),
        (I @ X + X @ X - X @ Y - Y @ X + Y @ Y, I @ X - X @ Y - Y @ X + Y @ Y),
        (2 * X @ X - X @ Y + Y @ X - X @ X, X @ X - X @ Y + Y @ X),
        (X @ X + X @ Y + Y @ X - X @ X - X @ Y - Y @ X, 0 * I @ I),
        (2 * X @ X - 3 * X @ Y + 2 * Y @ X - X @ X, X @ X - 3 * X @ Y + 2 * Y @ X),
    ],
)
def test_simplify(init_ps: PauliString, simplified_ps: PauliString):
    simplified_ps = init_ps.simplify()
    assert simplified_ps == simplified_ps
