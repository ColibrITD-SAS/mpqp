from itertools import product
from operator import add, matmul, mul, neg, pos, sub, truediv
from random import randint

import numpy as np
import numpy.typing as npt
import pytest

from mpqp.core.instruction.measurement.pauli_string import I, X, Y, Z, PauliString
from mpqp.tools.maths import matrix_eq
from cirq.ops.identity import I as Cirq_I
from cirq.ops.pauli_gates import X as Cirq_X
from cirq.ops.pauli_gates import Y as Cirq_Y
from cirq.ops.pauli_gates import Z as Cirq_Z
from cirq.devices.line_qubit import LineQubit


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


a, b, c = LineQubit.range(3)


@pytest.mark.parametrize(
    "other_ps, mpqp_ps",
    [
        (
            Cirq_X(a) + Cirq_Y(b) + Cirq_Z(c),
            X @ I @ I + I @ Y @ I + I @ I @ Z,
        ),
        (
            Cirq_X(a) * Cirq_Y(b) * Cirq_Z(c),
            X @ Y @ Z,
        ),
        (
            Cirq_I(a) + Cirq_Z(b) + Cirq_X(c),
            I @ I @ I + I @ Z @ I + I @ I @ X,
        ),
        (
            Cirq_Y(a) * Cirq_Z(b) * Cirq_X(c),
            Y @ Z @ X,
        ),
        (
            Cirq_Z(a) * Cirq_Y(b) + Cirq_X(c),
            Z @ Y @ I + I @ I @ X,
        ),
        (
            Cirq_X(a) + Cirq_I(b) * Cirq_Y(c),
            X @ I @ I + I @ I @ Y,
        ),
        (
            Cirq_I(a) * Cirq_X(b) + Cirq_Y(c),
            I @ X @ I + I @ I @ Y,
        ),
        (
            2 * Cirq_X(a) + 3 * Cirq_Y(b) + 4 * Cirq_Z(c),
            2 * X @ I @ I + 3 * I @ Y @ I + 4 * I @ I @ Z,
        ),
        (
            -Cirq_X(a) * 1.5 * Cirq_Y(b) * 0.5 * Cirq_Z(c),
            -1.5 * 0.5 * X @ Y @ Z,
        ),
        (
            0.5 * Cirq_Z(a) * 0.5 * Cirq_Y(b) + 2 * Cirq_X(c),
            0.25 * Z @ Y @ I + 2 * I @ I @ X,
        ),
        (
            1.5 * Cirq_X(a) + Cirq_I(b) * -2.5 * Cirq_Y(c),
            1.5 * X @ I @ I + -2.5 * I @ I @ Y,
        ),
        (
            0.25 * Cirq_I(a) * 4 * Cirq_X(b) + 3 * Cirq_Y(c),
            1.0 * I @ X @ I + 3 * I @ I @ Y,
        ),
    ],
)
def test_from_other_languages(other_ps: any, mpqp_ps: PauliString):
    assert PauliString.from_other_languages(other_ps) == mpqp_ps
