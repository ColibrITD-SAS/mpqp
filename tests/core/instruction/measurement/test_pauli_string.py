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
from typing import TYPE_CHECKING, Optional, Union
from sympy import symbols

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

from mpqp.core.instruction.measurement.pauli_string import (
    PI,
    Coef,
    PauliString,
    PX,
    PY,
    PZ,
    PauliStringAtom,
    pauli_string_from_str,
    pauli_string_with_atom,
)
from mpqp.core.languages import Language
from mpqp.tools.maths import matrix_eq


def pauli_string_combinations():
    scalar_bin_operation = [mul, truediv, imul, itruediv]
    homogeneous_bin_operation = [add, sub, iadd]
    bin_operation = [matmul, imatmul]
    un_operation = [pos, neg]
    pauli = [
        (PI, np.eye(2)),
        ((PI @ PI), np.eye(4)),
        ((PI + PI), (2 * np.eye(2))),
        ((PI + PI) @ PI, (2 * np.eye(4))),
        ((PX + PX), np.array([[0, 2.0], [2.0, 0]])),
        ((PX + PZ), np.array([[1.0, 1.0], [1.0, -1.0]])),
        ((2 * PI), (2 * np.eye(2))),
        ((symbols("a") * PI), (symbols("a") * np.eye(2))),
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
            ps1_matrix = deepcopy(ps_1[1])
            result.append((op(ps1, ps_2[0]), converted_op(ps1_matrix, ps_2[1])))
        if ps_1[0].nb_qubits == ps_2[0].nb_qubits:
            for op in homogeneous_bin_operation:
                ps1 = deepcopy(ps_1[0])
                ps1_matrix = deepcopy(ps_1[1])
                if ps_2[1].dtype == object:
                    ps1_matrix = np.array(ps1_matrix, dtype=object)
                result.append((op(ps1, ps_2[0]), op(ps1_matrix, ps_2[1])))

    return result


@pytest.mark.parametrize("ps, matrix", pauli_string_combinations())
def test_operations(ps: PauliString, matrix: npt.NDArray[np.complex128]):
    assert matrix_eq(ps.to_matrix(), matrix)


@pytest.mark.parametrize(
    "init_ps, simplified_ps",
    [
        # Test cases with single terms
        (PI @ PI, PI @ PI),
        (2 * PI @ PI, 2 * PI @ PI),
        (-PI @ PI, -PI @ PI),
        (0 * PI @ PI, 0 * PI @ PI),
        # Test cases with multiple terms
        (PI @ PI + PI @ PI + PI @ PI, 3 * PI @ PI),
        (2 * PI @ PI + 3 * PI @ PI - 2 * PI @ PI, 3 * PI @ PI),
        (2 * PI @ PI - 3 * PI @ PI + PI @ PI, 0),
        (-PI @ PI + PI @ PI - PI @ PI, 0),
        (PI @ PI - 2 * PI @ PI + PI @ PI, PI @ PI),
        (2 * PI @ PI + PI @ PI - PI @ PI, 2 * PI @ PI),
        (PI @ PI + PI @ PI + PI @ PI, 3 * PI @ PI),
        (2 * PI @ PI + 3 * PI @ PI, 5 * PI @ PI),
        (PI @ PI - PI @ PI + PI @ PI, PI @ PI),
        # Test cases with cancellation
        (PI @ PI - PI @ PI, 0 * PI @ PI),
        (2 * PI @ PI - 2 * PI @ PI, 0 * PI @ PI),
        (-2 * PI @ PI + 2 * PI @ PI, 0 * PI @ PI),
        (PI @ PI + PI @ PI - 2 * PI @ PI, 0 * PI @ PI),
        # Test cases with mixed terms
        (PI @ PI - 2 * PI @ PI + 3 * PI @ PI, 2 * PI @ PI),
        (2 * PI @ PI + PI @ PI - PI @ PI, 2 * PI @ PI),
        (PI @ PI + PI @ PI + PI @ PI - 3 * PI @ PI, PI @ PI),
        # Test cases with combinations of different gates
        (PI @ PX + PX @ PX - PX @ PI, 2 * PX @ PX),
        (PY @ PZ + PZ @ PY - PZ @ PZ, PY @ PZ + PZ @ PY),
        (PI @ PX + PX @ PY - PY @ PX - PX @ PI, 0 * PI @ PI),
        (
            PI @ PX + PX @ PX - PX @ PY - PY @ PX + PY @ PY,
            PI @ PX - PX @ PY - PY @ PX + PY @ PY,
        ),
        (2 * PX @ PX - PX @ PY + PY @ PX - PX @ PX, PX @ PX - PX @ PY + PY @ PX),
        (PX @ PX + PX @ PY + PY @ PX - PX @ PX - PX @ PY - PY @ PX, 0 * PI @ PI),
        (
            2 * PX @ PX - 3 * PX @ PY + 2 * PY @ PX - PX @ PX,
            PX @ PX - 3 * PX @ PY + 2 * PY @ PX,
        ),
    ],
)
def test_simplify(init_ps: PauliString, simplified_ps: PauliString):
    simplified_ps = init_ps.simplify()
    assert simplified_ps == simplified_ps


@pytest.mark.parametrize(
    "init_ps, subs_dict, expected_ps",
    [
        # Basic substitution with numeric values
        (symbols("theta") * PI @ PX, {"theta": np.pi}, np.pi * PI @ PX),
        (symbols("k") * PX @ PY, {"k": 2}, 2 * PX @ PY),
        (symbols("a") * PY @ PZ, {"a": -1}, -PY @ PZ),
        # Multiple variable substitutions
        (
            symbols("theta") * PI @ PX + symbols("k") * PZ @ PY,
            {"theta": np.pi, "k": 1},
            np.pi * PI @ PX + PZ @ PY,
        ),
        (
            symbols("a") * PX @ PX + symbols("b") * PY @ PY,
            {"a": 0, "b": 3},
            3 * PY @ PY,
        ),
        # Removing symbolic values
        (symbols("theta") * PI @ PX, {"theta": np.pi}, np.pi * PI @ PX),
        (
            symbols("theta") * PX @ PY + symbols("phi") * PY @ PZ,
            {"theta": 1, "phi": 2},
            PX @ PY + 2 * PY @ PZ,
        ),
        # No substitutions (should remain the same)
        (symbols("theta") * PI @ PX, {}, symbols("theta") * PI @ PX),
    ],
)
def test_subs(
    init_ps: PauliString,
    subs_dict: dict[str, Union[float, int]],
    expected_ps: PauliString,
):
    result_ps = init_ps.subs(subs_dict)  # pyright: ignore
    assert result_ps == expected_ps, f"Expected {expected_ps}, but got {result_ps}"


a, b, c = LineQubit.range(3)


def pauli_strings_in_all_languages():
    from qiskit.quantum_info import SparsePauliOp

    return [
        (
            Cirq_X(a) + Cirq_Y(b) + Cirq_Z(c),  # pyright: ignore[reportOperatorIssue]
            Braket_X() @ Braket_I() @ Braket_I()
            + Braket_I() @ Braket_Y() @ Braket_I()
            + Braket_I() @ Braket_I() @ Braket_Z(),
            SparsePauliOp(["XII", "IYI", "IIZ"]),
            [Term(1, "X", [0]), Term(1, "Y", [1]), Term(1, "Z", [2])],
            PX @ PI @ PI + PI @ PY @ PI + PI @ PI @ PZ,
        ),
        (
            Cirq_X(a) * Cirq_Y(b) * Cirq_Z(c),  # pyright: ignore[reportOperatorIssue]
            Braket_X() @ Braket_Y() @ Braket_Z(),
            SparsePauliOp(["XYZ"]),
            Term(1, "XYZ", [0, 1, 2]),
            PX @ PY @ PZ,
        ),
        (
            Cirq_I(a) + Cirq_Z(b) + Cirq_X(c),
            Braket_I() @ Braket_I() @ Braket_I()
            + Braket_I() @ Braket_Z() @ Braket_I()
            + Braket_I() @ Braket_I() @ Braket_X(),
            SparsePauliOp(["III", "IZI", "IIX"]),
            [Term(1, "I", [0]), Term(1, "Z", [1]), Term(1, "X", [2])],
            PI @ PI @ PI + PI @ PZ @ PI + PI @ PI @ PX,
        ),
        (
            Cirq_Y(a) * Cirq_Z(b) * Cirq_X(c),  # pyright: ignore[reportOperatorIssue]
            Braket_Y() @ Braket_Z() @ Braket_X(),
            SparsePauliOp(["YZX"]),
            Term(1, "YZX", [0, 1, 2]),
            PY @ PZ @ PX,
        ),
        (
            Cirq_Z(a) * Cirq_Y(b) + Cirq_X(c),  # pyright: ignore[reportOperatorIssue]
            Braket_Z() @ Braket_Y() @ Braket_I() + Braket_I() @ Braket_I() @ Braket_X(),
            SparsePauliOp(["ZYI", "IIX"]),
            [Term(1, "ZY", [0, 1]), Term(1, "X", [2])],
            PZ @ PY @ PI + PI @ PI @ PX,
        ),
        (
            Cirq_X(a) + Cirq_I(b) * Cirq_Y(c),
            Braket_X() @ Braket_I() @ Braket_I() + Braket_I() @ Braket_I() @ Braket_Y(),
            SparsePauliOp(["XII", "IIY"]),
            [Term(1, "X", [0]), Term(1, "Y", [2])],
            PX @ PI @ PI + PI @ PI @ PY,
        ),
        (
            Cirq_I(a) * Cirq_X(b) + Cirq_Y(c),
            Braket_I() @ Braket_X() @ Braket_I() + Braket_I() @ Braket_I() @ Braket_Y(),
            SparsePauliOp(["IXI", "IIY"]),
            [Term(1, "X", [1]), Term(1, "Y", [2])],
            PI @ PX @ PI + PI @ PI @ PY,
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
            SparsePauliOp(["XII", "IYI", "IIZ"], coeffs=np.array([2, 3, 4])),
            [Term(2, "X", [0]), Term(3, "Y", [1]), Term(4, "Z", [2])],
            2 * PX @ PI @ PI + 3 * PI @ PY @ PI + 4 * PI @ PI @ PZ,
        ),
        (
            -Cirq_X(a)  # pyright: ignore[reportOperatorIssue]
            * (1.5 * Cirq_Y(b))  # pyright: ignore[reportOperatorIssue]
            * (0.5 * Cirq_Z(c)),  # pyright: ignore[reportOperatorIssue]
            (-1 * Braket_X())  # pyright: ignore[reportOperatorIssue]
            @ (1.5 * Braket_Y())  # pyright: ignore[reportOperatorIssue]
            @ (0.5 * Braket_Z()),  # pyright: ignore[reportOperatorIssue]
            SparsePauliOp(["XYZ"], coeffs=np.array([-1 * 1.5 * 0.5])),
            Term(-0.75, "XYZ", [0, 1, 2]),
            -PX @ (1.5 * PY) @ (0.5 * PZ),
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
            SparsePauliOp(["ZYI", "IIX"], coeffs=np.array([0.5 * 0.5, 2])),
            [Term(0.25, "ZY", [0, 1]), Term(2, "X", [2])],
            ((0.5 * PZ) @ (0.5 * PY) @ PI) + (2 * PI @ PI @ PX),
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
            SparsePauliOp(["XII", "IIY"], coeffs=np.array([1.5, -2.5])),
            [Term(1.5, "X", [0]), Term(-2.5, "Y", [2])],
            (1.5 * PX @ PI @ PI) + (PI @ PI @ (-2.5 * PY)),
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
            SparsePauliOp(["IXI", "IIY"], coeffs=np.array([0.25 * 4, 3])),
            [Term(4 * 0.25, "X", [1]), Term(3, "Y", [2])],
            ((0.25 * PI) @ (4 * PX) @ PI) + (PI @ PI @ (3 * PY)),
        ),
        (
            Cirq_I(a),
            Braket_I(),
            SparsePauliOp(["I"]),
            Term(1, "I", [0]),
            PI,
        ),
        (
            Cirq_X(a),
            Braket_X(),
            SparsePauliOp(["X"]),
            Term(1, "X", [0]),
            PX,
        ),
        (
            Cirq_Z(a),
            Braket_Z(),
            SparsePauliOp(["Z"]),
            Term(1, "Z", [0]),
            PZ,
        ),
        (
            Cirq_Y(a),
            Braket_Y(),
            SparsePauliOp(["Y"]),
            Term(1, "Y", [0]),
            PY,
        ),
        (
            1 * Cirq_I(b),
            Braket_I() @ Braket_I(),
            SparsePauliOp(["II"]),
            Term(1, "II", [0, 1]),
            PI @ PI,
        ),
        (
            1 * Cirq_X(b),  # pyright: ignore[reportOperatorIssue]
            Braket_I() @ Braket_X(),
            SparsePauliOp(["IX"]),
            Term(1, "X", [1]),
            PI @ PX,
        ),
        (
            1 * Cirq_Z(b),  # pyright: ignore[reportOperatorIssue]
            Braket_I() @ Braket_Z(),
            SparsePauliOp(["IZ"]),
            Term(1, "Z", [1]),
            PI @ PZ,
        ),
        (
            1 * Cirq_Y(b),  # pyright: ignore[reportOperatorIssue]
            Braket_I() @ Braket_Y(),
            SparsePauliOp(["IY"]),
            Term(1, "Y", [1]),
            PI @ PY,
        ),
        (
            1 * Cirq_I(a) + 1 * Cirq_I(a),
            Braket_I() + Braket_I(),
            SparsePauliOp(["I", "I"]),
            [Term(1, "I", [0]), Term(1, "I", [0])],
            PI + PI,
        ),
        (
            1 * Cirq_I(a) + 1 * Cirq_X(a),  # pyright: ignore[reportOperatorIssue]
            Braket_I() + Braket_X(),
            SparsePauliOp(["I", "X"]),
            [Term(1, "I", [0]), Term(1, "X", [0])],
            PI + PX,
        ),
        (
            1 * Cirq_X(a) + 1 * Cirq_Z(a),  # pyright: ignore[reportOperatorIssue]
            Braket_Z() + Braket_X(),
            SparsePauliOp(["Z", "X"]),
            [Term(1, "Z", [0]), Term(1, "X", [0])],
            PZ + PX,
        ),
        (
            1 * Cirq_Y(a) + 1 * Cirq_Z(a),  # pyright: ignore[reportOperatorIssue]
            Braket_Y() + Braket_Z(),
            SparsePauliOp(["Y", "Z"]),
            [Term(1, "Y", [0]), Term(1, "Z", [0])],
            PY + PZ,
        ),
        (
            1 * Cirq_X(a) + 1 * Cirq_Y(a),  # pyright: ignore[reportOperatorIssue]
            Braket_X() + Braket_Y(),
            SparsePauliOp(["X", "Y"]),
            [Term(1, "X", [0]), Term(1, "Y", [0])],
            PX + PY,
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


@pytest.mark.parametrize(
    "input_str, subs_dict, expected_str",
    [
        ("2*XZ", None, 2 * PX @ PZ),
        ("theta*IX", {}, symbols("theta") * PI @ PX),
        ("theta*IX", {"theta": 2}, 2 * PI @ PX),
        ("k*XY", {"k": 2}, 2 * PX @ PY),
        ("theta*IX + k*ZY", {"theta": 7, "k": 1}, 7 * PI @ PX + PZ @ PY),
        ("-a*YZ", {"a": -1}, PY @ PZ),
        ("o2*XZ + YI - 3*ZZ", {"o": 3}, 6 * PX @ PZ + PY @ PI - 3 * PZ @ PZ),
        (
            "o*2*XZ + YI - 3o*ZZ",
            None,
            symbols("o") * 2 * PX @ PZ + PY @ PI - 3 * symbols("o") * PZ @ PZ,
        ),
    ],
)
def test_pauli_string_from_str(
    input_str: str, subs_dict: Optional[dict[str, Coef]], expected_str: PauliString
):
    ps = pauli_string_from_str(input_str, subs_dict)
    assert ps == expected_str


@pytest.mark.parametrize(
    "n, atom, qubit_index, expected_ps",
    [
        (3, PX, None, PI @ PI @ PX),
        (3, PY, 0, PY @ PI @ PI),
        (4, PZ, 2, PI @ PI @ PZ @ PI),
        (2, PI, 1, PI @ PI),
        (1, PX, 0, PX),
    ],
)
def test_pauli_string_with_atom(
    n: int, atom: PauliStringAtom, qubit_index: Optional[int], expected_ps: PauliString
):
    result = pauli_string_with_atom(n, atom, qubit_index)
    assert result == expected_ps
