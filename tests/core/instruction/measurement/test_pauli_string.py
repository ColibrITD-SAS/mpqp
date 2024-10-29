from copy import deepcopy
from itertools import product
from operator import (
    add,
    iadd,
    imatmul,
    imul,
    itruediv,
    matmul,
    mul,
    neg,
    pos,
    sub,
    truediv,
)
from random import randint
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit.quantum_info import SparsePauliOp

import numpy as np
import numpy.typing as npt
import pytest
from braket.circuits.observables import I as Braket_I
from braket.circuits.observables import Sum as BraketSum
from braket.circuits.observables import X as Braket_X
from braket.circuits.observables import Y as Braket_Y
from braket.circuits.observables import Z as Braket_Z
from cirq.devices.line_qubit import LineQubit
from cirq.ops.identity import I as Cirq_I
from cirq.ops.linear_combinations import PauliSum
from cirq.ops.pauli_gates import X as Cirq_X
from cirq.ops.pauli_gates import Y as Cirq_Y
from cirq.ops.pauli_gates import Z as Cirq_Z
from qat.core.wrappers.observable import Term

from mpqp.core.instruction.measurement.pauli_string import I, PauliString, X, Y, Z
from mpqp.core.languages import Language
from mpqp.tools.maths import matrix_eq


def pauli_string_combinations():
    scalar_bin_operation = [mul, truediv, imul, itruediv]
    homogeneous_bin_operation = [add, sub, iadd]
    bin_operation = [matmul, imatmul]
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
            ps_ = deepcopy(ps[0])
            ps_matrix = deepcopy(ps[1])
            result.append((op(ps_, a), op(ps_matrix, a)))
        for op in un_operation:
            result.append((op(ps[0]), op(ps[1])))
    for ps_1, ps_2 in product(pauli, repeat=2):
        for op in bin_operation:
            converted_op = op if (op != matmul and op != imatmul) else np.kron
            ps1 = deepcopy(ps_1[0])
            result.append((op(ps1, ps_2[0]), converted_op(ps_1[1], ps_2[1])))
        if ps_1[0].nb_qubits == ps_2[0].nb_qubits:
            for op in homogeneous_bin_operation:
                ps1 = deepcopy(ps_1[0])
                ps1_matrix = deepcopy(ps_1[1])
                result.append((op(ps1, ps_2[0]), op(ps1_matrix, ps_2[1])))

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


def pauli_strings_in_all_languages():
    from qiskit.quantum_info import SparsePauliOp

    return [
        (
            Cirq_X(a) + Cirq_Y(b) + Cirq_Z(c),  # pyright: ignore[reportOperatorIssue]
            Braket_X() @ Braket_I() @ Braket_I()
            + Braket_I() @ Braket_Y() @ Braket_I()
            + Braket_I() @ Braket_I() @ Braket_Z(),
            SparsePauliOp(["IIX", "IYI", "ZII"]),
            [Term(1, "X", [0]), Term(1, "Y", [1]), Term(1, "Z", [2])],
            X @ I @ I + I @ Y @ I + I @ I @ Z,
        ),
        (
            Cirq_X(a) * Cirq_Y(b) * Cirq_Z(c),  # pyright: ignore[reportOperatorIssue]
            Braket_X() @ Braket_Y() @ Braket_Z(),
            SparsePauliOp(["ZYX"]),
            Term(1, "XYZ", [0, 1, 2]),
            X @ Y @ Z,
        ),
        (
            Cirq_I(a) + Cirq_Z(b) + Cirq_X(c),
            Braket_I() @ Braket_I() @ Braket_I()
            + Braket_I() @ Braket_Z() @ Braket_I()
            + Braket_I() @ Braket_I() @ Braket_X(),
            SparsePauliOp(["III", "IZI", "XII"]),
            [Term(1, "I", [0]), Term(1, "Z", [1]), Term(1, "X", [2])],
            I @ I @ I + I @ Z @ I + I @ I @ X,
        ),
        (
            Cirq_Y(a) * Cirq_Z(b) * Cirq_X(c),  # pyright: ignore[reportOperatorIssue]
            Braket_Y() @ Braket_Z() @ Braket_X(),
            SparsePauliOp(["XZY"]),
            Term(1, "YZX", [0, 1, 2]),
            Y @ Z @ X,
        ),
        (
            Cirq_Z(a) * Cirq_Y(b) + Cirq_X(c),  # pyright: ignore[reportOperatorIssue]
            Braket_Z() @ Braket_Y() @ Braket_I() + Braket_I() @ Braket_I() @ Braket_X(),
            SparsePauliOp(["IYZ", "XII"]),
            [Term(1, "ZY", [0, 1]), Term(1, "X", [2])],
            Z @ Y @ I + I @ I @ X,
        ),
        (
            Cirq_X(a) + Cirq_I(b) * Cirq_Y(c),
            Braket_X() @ Braket_I() @ Braket_I() + Braket_I() @ Braket_I() @ Braket_Y(),
            SparsePauliOp(["IIX", "YII"]),
            [Term(1, "X", [0]), Term(1, "Y", [2])],
            X @ I @ I + I @ I @ Y,
        ),
        (
            Cirq_I(a) * Cirq_X(b) + Cirq_Y(c),
            Braket_I() @ Braket_X() @ Braket_I() + Braket_I() @ Braket_I() @ Braket_Y(),
            SparsePauliOp(["IXI", "YII"]),
            [Term(1, "X", [1]), Term(1, "Y", [2])],
            I @ X @ I + I @ I @ Y,
        ),
        (
            2 * Cirq_X(a)  # pyright: ignore[reportOperatorIssue]
            + 3 * Cirq_Y(b)  # pyright: ignore[reportOperatorIssue]
            + 4 * Cirq_Z(c),  # pyright: ignore[reportOperatorIssue]
            2
            * Braket_X()  # pyright: ignore[reportOperatorIssue]
            @ Braket_I()
            @ Braket_I()
            + 3
            * Braket_I()  # pyright: ignore[reportOperatorIssue]
            @ Braket_Y()
            @ Braket_I()
            + 4
            * Braket_I()  # pyright: ignore[reportOperatorIssue]
            @ Braket_I()
            @ Braket_Z(),
            SparsePauliOp(["IIX", "IYI", "ZII"], coeffs=np.array([2, 3, 4])),
            [Term(2, "X", [0]), Term(3, "Y", [1]), Term(4, "Z", [2])],
            2 * X @ I @ I + 3 * I @ Y @ I + 4 * I @ I @ Z,
        ),
        (
            -Cirq_X(a)  # pyright: ignore[reportOperatorIssue]
            * (1.5 * Cirq_Y(b))  # pyright: ignore[reportOperatorIssue]
            * (0.5 * Cirq_Z(c)),  # pyright: ignore[reportOperatorIssue]
            (-1 * Braket_X())  # pyright: ignore[reportOperatorIssue]
            @ (1.5 * Braket_Y())  # pyright: ignore[reportOperatorIssue]
            @ (0.5 * Braket_Z()),  # pyright: ignore[reportOperatorIssue]
            SparsePauliOp(["ZYX"], coeffs=np.array([-1 * 1.5 * 0.5])),
            Term(-0.75, "XYZ", [0, 1, 2]),
            -X @ (1.5 * Y) @ (0.5 * Z),
        ),
        (
            0.5 * Cirq_Z(a) * 0.5 * Cirq_Y(b)  # pyright: ignore[reportOperatorIssue]
            + 2 * Cirq_X(c),  # pyright: ignore[reportOperatorIssue]
            (0.5 * Braket_Z())  # pyright: ignore[reportOperatorIssue]
            @ (0.5 * Braket_Y())  # pyright: ignore[reportOperatorIssue]
            @ Braket_I()
            + Braket_I()
            @ Braket_I()
            @ (2 * Braket_X()),  # pyright: ignore[reportOperatorIssue]
            SparsePauliOp(["IYZ", "XII"], coeffs=np.array([0.5 * 0.5, 2])),
            [Term(0.25, "ZY", [0, 1]), Term(2, "X", [2])],
            ((0.5 * Z) @ (0.5 * Y) @ I) + (2 * I @ I @ X),
        ),
        (
            1.5 * Cirq_X(a)  # pyright: ignore[reportOperatorIssue]
            + Cirq_I(b) * -2.5 * Cirq_Y(c),
            1.5
            * Braket_X()  # pyright: ignore[reportOperatorIssue]
            @ Braket_I()
            @ Braket_I()
            + Braket_I()
            @ Braket_I()
            @ (-2.5 * Braket_Y()),  # pyright: ignore[reportOperatorIssue]
            SparsePauliOp(["IIX", "YII"], coeffs=np.array([1.5, -2.5])),
            [Term(1.5, "X", [0]), Term(-2.5, "Y", [2])],
            (1.5 * X @ I @ I) + (I @ I @ (-2.5 * Y)),
        ),
        (
            0.25 * Cirq_I(a) * 4 * Cirq_X(b)
            + 3 * Cirq_Y(c),  # pyright: ignore[reportOperatorIssue]
            (0.25 * Braket_I())  # pyright: ignore[reportOperatorIssue]
            @ (4 * Braket_X())  # pyright: ignore[reportOperatorIssue]
            @ Braket_I()
            + Braket_I()
            @ Braket_I()
            @ (3 * Braket_Y()),  # pyright: ignore[reportOperatorIssue]
            SparsePauliOp(["IXI", "YII"], coeffs=np.array([0.25 * 4, 3])),
            [Term(4 * 0.25, "X", [1]), Term(3, "Y", [2])],
            ((0.25 * I) @ (4 * X) @ I) + (I @ I @ (3 * Y)),
        ),
        (
            Cirq_I(a),
            Braket_I(),
            SparsePauliOp(["I"]),
            Term(1, "I", [0]),
            I,
        ),
        (
            Cirq_X(a),
            Braket_X(),
            SparsePauliOp(["X"]),
            Term(1, "X", [0]),
            X,
        ),
        (
            Cirq_Z(a),
            Braket_Z(),
            SparsePauliOp(["Z"]),
            Term(1, "Z", [0]),
            Z,
        ),
        (
            Cirq_Y(a),
            Braket_Y(),
            SparsePauliOp(["Y"]),
            Term(1, "Y", [0]),
            Y,
        ),
        (
            1 * Cirq_I(b),
            Braket_I() @ Braket_I(),
            SparsePauliOp(["II"]),
            Term(1, "II", [0, 1]),
            I @ I,
        ),
        (
            1 * Cirq_X(b),  # pyright: ignore[reportOperatorIssue]
            Braket_I() @ Braket_X(),
            SparsePauliOp(["XI"]),
            Term(1, "X", [1]),
            I @ X,
        ),
        (
            1 * Cirq_Z(b),  # pyright: ignore[reportOperatorIssue]
            Braket_I() @ Braket_Z(),
            SparsePauliOp(["ZI"]),
            Term(1, "Z", [1]),
            I @ Z,
        ),
        (
            1 * Cirq_Y(b),  # pyright: ignore[reportOperatorIssue]
            Braket_I() @ Braket_Y(),
            SparsePauliOp(["YI"]),
            Term(1, "Y", [1]),
            I @ Y,
        ),
        (
            1 * Cirq_I(a) + 1 * Cirq_I(a),
            Braket_I() + Braket_I(),
            SparsePauliOp(["I", "I"]),
            [Term(1, "I", [0]), Term(1, "I", [0])],
            I + I,
        ),
        (
            1 * Cirq_I(a) + 1 * Cirq_X(a),  # pyright: ignore[reportOperatorIssue]
            Braket_I() + Braket_X(),
            SparsePauliOp(["I", "X"]),
            [Term(1, "I", [0]), Term(1, "X", [0])],
            I + X,
        ),
        (
            1 * Cirq_X(a) + 1 * Cirq_Z(a),  # pyright: ignore[reportOperatorIssue]
            Braket_Z() + Braket_X(),
            SparsePauliOp(["Z", "X"]),
            [Term(1, "Z", [0]), Term(1, "X", [0])],
            Z + X,
        ),
        (
            1 * Cirq_Y(a) + 1 * Cirq_Z(a),  # pyright: ignore[reportOperatorIssue]
            Braket_Y() + Braket_Z(),
            SparsePauliOp(["Y", "Z"]),
            [Term(1, "Y", [0]), Term(1, "Z", [0])],
            Y + Z,
        ),
        (
            1 * Cirq_X(a) + 1 * Cirq_Y(a),  # pyright: ignore[reportOperatorIssue]
            Braket_X() + Braket_Y(),
            SparsePauliOp(["X", "Y"]),
            [Term(1, "X", [0]), Term(1, "Y", [0])],
            X + Y,
        ),
    ]


@pytest.mark.parametrize(
    "cirq_ps, braket_ps, qiskit_ps, my_qml_ps, mpqp_ps",
    pauli_strings_in_all_languages(),
)
def test_from_other_language(
    cirq_ps: PauliSum,
    braket_ps: BraketSum,
    qiskit_ps: "SparsePauliOp",
    my_qml_ps: Term,
    mpqp_ps: PauliString,
):
    assert PauliString.from_other_language(cirq_ps, mpqp_ps.nb_qubits) == mpqp_ps
    assert PauliString.from_other_language(braket_ps) == mpqp_ps
    assert PauliString.from_other_language(qiskit_ps) == mpqp_ps
    assert PauliString.from_other_language(my_qml_ps) == mpqp_ps


@pytest.mark.parametrize(
    "cirq_ps, braket_ps, qiskit_ps, my_qml_ps, mpqp_ps",
    pauli_strings_in_all_languages(),
)
def test_to_other_language(
    cirq_ps: PauliSum,
    braket_ps: BraketSum,
    qiskit_ps: "SparsePauliOp",
    my_qml_ps: Term,
    mpqp_ps: PauliString,
):
    assert mpqp_ps.to_other_language(Language.CIRQ) == cirq_ps
    assert repr(mpqp_ps.to_other_language(Language.BRAKET)) == repr(braket_ps)
    assert mpqp_ps.to_other_language(Language.QISKIT) == qiskit_ps
    assert mpqp_ps.to_other_language(Language.MY_QLM) == my_qml_ps


@pytest.mark.parametrize(
    "mpqp_ps", [all_ps[-1] for all_ps in pauli_strings_in_all_languages()]
)
def test_to_from_other_language(
    mpqp_ps: PauliString,
):
    assert (
        PauliString.from_other_language(
            mpqp_ps.to_other_language(Language.CIRQ), mpqp_ps.nb_qubits
        )
        == mpqp_ps
    )
    assert (
        PauliString.from_other_language(mpqp_ps.to_other_language(Language.BRAKET))
        == mpqp_ps
    )
    assert (
        PauliString.from_other_language(mpqp_ps.to_other_language(Language.QISKIT))
        == mpqp_ps
    )
    assert (
        PauliString.from_other_language(mpqp_ps.to_other_language(Language.MY_QLM))
        == mpqp_ps
    )
