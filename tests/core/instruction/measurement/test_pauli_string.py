from copy import deepcopy
from itertools import product
from numbers import Real
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
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

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
from sympy import Basic, Expr

from mpqp import Language, pI, pX, pY, pZ
from mpqp.core.instruction.measurement.pauli_string import PauliString, PauliStringAtom
from mpqp.tools.maths import matrix_eq

Coef = Union[Real, float, Expr, Basic]


def pauli_matrix_mapping():
    return [
        (pI, np.eye(2)),
        ((pI @ pI), np.eye(4)),
        ((pI + pI), (2 * np.eye(2))),
        ((pI + pI) @ pI, (2 * np.eye(4))),
        ((pX + pX), np.array([[0, 2.0], [2.0, 0]])),
        ((pX + pZ), np.array([[1.0, 1.0], [1.0, -1.0]])),
        ((2 * pI), (2 * np.eye(2))),
        ((symbols("a") * pI), (symbols("a") * np.eye(2))),
    ]


@pytest.mark.parametrize(
    "ps, matrix, op",
    (
        (ps, matrix, op)
        for (ps, matrix), op in product(
            pauli_matrix_mapping(), [mul, truediv, imul, itruediv]
        )
    ),
)
def test_scalar_bin_operation(
    ps: PauliString, matrix: npt.NDArray[np.complex128], op: Callable[[Any, Any], Any]
):
    a = randint(1, 9)
    ps_clean = deepcopy(ps)
    matrix_clean = deepcopy(matrix)
    assert matrix_eq(op(ps_clean, a).to_matrix(), op(matrix_clean, a))


@pytest.mark.parametrize(
    "ps, matrix, op",
    (
        (ps, matrix, op)
        for (ps, matrix), op in product(pauli_matrix_mapping(), [pos, neg])
    ),
)
def test_un_operation(
    ps: PauliString, matrix: npt.NDArray[np.complex128], op: Callable[[Any], Any]
):
    assert matrix_eq(op(ps).to_matrix(), op(matrix))


@pytest.mark.parametrize(
    "ps1, ps2, matrix1, matrix2, op",
    (
        (ps1, ps2, matrix1, matrix2, op)
        for (ps1, matrix1), (ps2, matrix2), op in product(
            pauli_matrix_mapping(), pauli_matrix_mapping(), [matmul, imatmul]
        )
    ),
)
def test_bin_operation(
    ps1: PauliString,
    ps2: PauliString,
    matrix1: npt.NDArray[np.complex128],
    matrix2: npt.NDArray[np.complex128],
    op: Callable[[Any, Any], Any],
):
    clean_ps1 = deepcopy(ps1)
    clean_matrix1 = deepcopy(matrix1)
    assert matrix_eq(op(clean_ps1, ps2).to_matrix(), np.kron(clean_matrix1, matrix2))


@pytest.mark.parametrize(
    "ps1, ps2, matrix1, matrix2, op",
    (
        (ps1, ps2, matrix1, matrix2, op)
        for (ps1, matrix1), (ps2, matrix2), op in product(
            pauli_matrix_mapping(), pauli_matrix_mapping(), [add, sub, iadd]
        )
    ),
)
def test_homogeneous_bin_operation(
    ps1: PauliString,
    ps2: PauliString,
    matrix1: npt.NDArray[np.complex128],
    matrix2: npt.NDArray[np.complex128],
    op: Callable[[Any, Any], Any],
):
    clean_ps1 = deepcopy(ps1)
    clean_matrix1 = deepcopy(matrix1)
    if clean_ps1.nb_qubits != ps2.nb_qubits:
        return
    if matrix2.dtype == object:
        clean_matrix1 = np.array(clean_matrix1, dtype=object)
    assert matrix_eq(op(clean_ps1, ps2).to_matrix(), op(clean_matrix1, matrix2))


@pytest.mark.parametrize(
    "init_ps, simplified_ps",
    [
        # Test cases with single terms
        (pI @ pI, pI @ pI),
        (2 * pI @ pI, 2 * pI @ pI),
        (-pI @ pI, -pI @ pI),
        (0 * pI @ pI, 0 * pI @ pI),
        # Test cases with multiple terms
        (pI @ pI + pI @ pI + pI @ pI, 3 * pI @ pI),
        (2 * pI @ pI + 3 * pI @ pI - 2 * pI @ pI, 3 * pI @ pI),
        (2 * pI @ pI - 3 * pI @ pI + pI @ pI, 0),
        (-pI @ pI + pI @ pI - pI @ pI, 0),
        (pI @ pI - 2 * pI @ pI + pI @ pI, pI @ pI),
        (2 * pI @ pI + pI @ pI - pI @ pI, 2 * pI @ pI),
        (pI @ pI + pI @ pI + pI @ pI, 3 * pI @ pI),
        (2 * pI @ pI + 3 * pI @ pI, 5 * pI @ pI),
        (pI @ pI - pI @ pI + pI @ pI, pI @ pI),
        # Test cases with cancellation
        (pI @ pI - pI @ pI, 0 * pI @ pI),
        (2 * pI @ pI - 2 * pI @ pI, 0 * pI @ pI),
        (-2 * pI @ pI + 2 * pI @ pI, 0 * pI @ pI),
        (pI @ pI + pI @ pI - 2 * pI @ pI, 0 * pI @ pI),
        # Test cases with mixed terms
        (pI @ pI - 2 * pI @ pI + 3 * pI @ pI, 2 * pI @ pI),
        (2 * pI @ pI + pI @ pI - pI @ pI, 2 * pI @ pI),
        (pI @ pI + pI @ pI + pI @ pI - 3 * pI @ pI, pI @ pI),
        # Test cases with combinations of different gates
        (pI @ pX + pX @ pX - pX @ pI, 2 * pX @ pX),
        (pY @ pZ + pZ @ pY - pZ @ pZ, pY @ pZ + pZ @ pY),
        (pI @ pX + pX @ pY - pY @ pX - pX @ pI, 0 * pI @ pI),
        (
            pI @ pX + pX @ pX - pX @ pY - pY @ pX + pY @ pY,
            pI @ pX - pX @ pY - pY @ pX + pY @ pY,
        ),
        (2 * pX @ pX - pX @ pY + pY @ pX - pX @ pX, pX @ pX - pX @ pY + pY @ pX),
        (pX @ pX + pX @ pY + pY @ pX - pX @ pX - pX @ pY - pY @ pX, 0 * pI @ pI),
        (
            2 * pX @ pX - 3 * pX @ pY + 2 * pY @ pX - pX @ pX,
            pX @ pX - 3 * pX @ pY + 2 * pY @ pX,
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
        (symbols("theta") * pI @ pX, {"theta": np.pi}, np.pi * pI @ pX),
        (symbols("k") * pX @ pY, {"k": 2}, 2 * pX @ pY),
        (symbols("a") * pY @ pZ, {"a": -1}, -pY @ pZ),
        # Multiple variable substitutions
        (
            symbols("theta") * pI @ pX + symbols("k") * pZ @ pY,
            {"theta": np.pi, "k": 1},
            np.pi * pI @ pX + pZ @ pY,
        ),
        (
            symbols("a") * pX @ pX + symbols("b") * pY @ pY,
            {"a": 0, "b": 3},
            3 * pY @ pY,
        ),
        # Removing symbolic values
        (symbols("theta") * pI @ pX, {"theta": np.pi}, np.pi * pI @ pX),
        (
            symbols("theta") * pX @ pY + symbols("phi") * pY @ pZ,
            {"theta": 1, "phi": 2},
            pX @ pY + 2 * pY @ pZ,
        ),
        # No substitutions (should remain the same)
        (symbols("theta") * pI @ pX, {}, symbols("theta") * pI @ pX),
    ],
)
def test_subs(
    init_ps: PauliString,
    subs_dict: dict[str, Union[float, int]],
    expected_ps: PauliString,
):
    result_ps = init_ps.subs(subs_dict)  # pyright: ignore
    assert result_ps == expected_ps


a, b, c = LineQubit.range(3)


def pauli_strings_in_all_languages() -> list[
    dict[
        Optional[Language],
        Union[PauliSum, BraketSum, "SparsePauliOp", Term, PauliString],
    ]
]:
    from qiskit.quantum_info import SparsePauliOp

    return [
        {
            Language.CIRQ: Cirq_X(a)
            + Cirq_Y(b)  # pyright: ignore[reportOperatorIssue]
            + Cirq_Z(c),
            Language.BRAKET: Braket_X(0) @ Braket_I(1) @ Braket_I(2)
            + Braket_I(0) @ Braket_Y(1) @ Braket_I(2)
            + Braket_I(0) @ Braket_I(1) @ Braket_Z(2),
            Language.QISKIT: SparsePauliOp(["XII", "IYI", "IIZ"]),
            Language.MY_QLM: [
                Term(1, "X", [0]),
                Term(1, "Y", [1]),
                Term(1, "Z", [2]),
            ],
            None: pX @ pI @ pI + pI @ pY @ pI + pI @ pI @ pZ,
        },
        {
            Language.CIRQ: Cirq_X(a)
            * Cirq_Y(b)  # pyright: ignore[reportOperatorIssue]
            * Cirq_Z(c),
            Language.BRAKET: Braket_X(0) @ Braket_Y(1) @ Braket_Z(2),
            Language.QISKIT: SparsePauliOp(["XYZ"]),
            Language.MY_QLM: Term(1, "XYZ", [0, 1, 2]),
            None: pX @ pY @ pZ,
        },
        {
            Language.CIRQ: Cirq_I(a) + Cirq_Z(b) + Cirq_X(c),
            Language.BRAKET: Braket_I(0) @ Braket_I(1) @ Braket_I(2)
            + Braket_I(0) @ Braket_Z(1) @ Braket_I(2)
            + Braket_I(0) @ Braket_I(1) @ Braket_X(2),
            Language.QISKIT: SparsePauliOp(["III", "IZI", "IIX"]),
            Language.MY_QLM: [
                Term(1, "I", [0]),
                Term(1, "Z", [1]),
                Term(1, "X", [2]),
            ],
            None: pI @ pI @ pI + pI @ pZ @ pI + pI @ pI @ pX,
        },
        {
            Language.CIRQ: Cirq_Y(a)
            * Cirq_Z(b)  # pyright: ignore[reportOperatorIssue]
            * Cirq_X(c),
            Language.BRAKET: Braket_Y(0) @ Braket_Z(1) @ Braket_X(2),
            Language.QISKIT: SparsePauliOp(["YZX"]),
            Language.MY_QLM: Term(1, "YZX", [0, 1, 2]),
            None: pY @ pZ @ pX,
        },
        {
            Language.CIRQ: Cirq_Z(a) * Cirq_Y(b)  # pyright: ignore[reportOperatorIssue]
            + Cirq_X(c),
            Language.BRAKET: Braket_Z(0) @ Braket_Y(1) @ Braket_I(2)
            + Braket_I(0) @ Braket_I(1) @ Braket_X(2),
            Language.QISKIT: SparsePauliOp(["ZYI", "IIX"]),
            Language.MY_QLM: [Term(1, "ZY", [0, 1]), Term(1, "X", [2])],
            None: pZ @ pY @ pI + pI @ pI @ pX,
        },
        {
            Language.CIRQ: Cirq_X(a) + Cirq_I(b) * Cirq_Y(c),
            Language.BRAKET: Braket_X(0) @ Braket_I(1) @ Braket_I(2)
            + Braket_I(0) @ Braket_I(1) @ Braket_Y(2),
            Language.QISKIT: SparsePauliOp(["XII", "IIY"]),
            Language.MY_QLM: [Term(1, "X", [0]), Term(1, "Y", [2])],
            None: pX @ pI @ pI + pI @ pI @ pY,
        },
        {
            Language.CIRQ: Cirq_I(a) * Cirq_X(b) + Cirq_Y(c),
            Language.BRAKET: Braket_I(0) @ Braket_X(1) @ Braket_I(2)
            + Braket_I(0) @ Braket_I(1) @ Braket_Y(2),
            Language.QISKIT: SparsePauliOp(["IXI", "IIY"]),
            Language.MY_QLM: [Term(1, "X", [1]), Term(1, "Y", [2])],
            None: pI @ pX @ pI + pI @ pI @ pY,
        },
        {
            Language.CIRQ: 2 * Cirq_X(a)  # pyright: ignore[reportOperatorIssue]
            + 3 * Cirq_Y(b)  # pyright: ignore[reportOperatorIssue]
            + 4 * Cirq_Z(c),  # pyright: ignore[reportOperatorIssue]
            Language.BRAKET: 2
            * Braket_X(0)  # pyright: ignore[reportOperatorIssue]
            @ Braket_I(1)
            @ Braket_I(2)
            + 3
            * Braket_I(0)  # pyright: ignore[reportOperatorIssue]
            @ Braket_Y(1)
            @ Braket_I(2)
            + 4
            * Braket_I(0)  # pyright: ignore[reportOperatorIssue]
            @ Braket_I(1)
            @ Braket_Z(2),
            Language.QISKIT: SparsePauliOp(
                ["XII", "IYI", "IIZ"], coeffs=np.array([2, 3, 4])
            ),
            Language.MY_QLM: [
                Term(2, "X", [0]),
                Term(3, "Y", [1]),
                Term(4, "Z", [2]),
            ],
            None: 2 * pX @ pI @ pI + 3 * pI @ pY @ pI + 4 * pI @ pI @ pZ,
        },
        {
            Language.CIRQ: -Cirq_X(a)  # pyright: ignore[reportOperatorIssue]
            * (1.5 * Cirq_Y(b))  # pyright: ignore[reportOperatorIssue]
            * (0.5 * Cirq_Z(c)),  # pyright: ignore[reportOperatorIssue]
            Language.BRAKET: (-1 * Braket_X(0))  # pyright: ignore[reportOperatorIssue]
            @ (1.5 * Braket_Y(1))  # pyright: ignore[reportOperatorIssue]
            @ (0.5 * Braket_Z(2)),  # pyright: ignore[reportOperatorIssue]
            Language.QISKIT: SparsePauliOp(["XYZ"], coeffs=np.array([-1 * 1.5 * 0.5])),
            Language.MY_QLM: Term(-0.75, "XYZ", [0, 1, 2]),
            None: -pX @ (1.5 * pY) @ (0.5 * pZ),
        },
        {
            Language.CIRQ: 0.5
            * Cirq_Z(a)  # pyright: ignore[reportOperatorIssue]
            * 0.5
            * Cirq_Y(b)
            + 2 * Cirq_X(c),  # pyright: ignore[reportOperatorIssue]
            Language.BRAKET: (0.5 * Braket_Z(0))  # pyright: ignore[reportOperatorIssue]
            @ (0.5 * Braket_Y(1))  # pyright: ignore[reportOperatorIssue]
            @ Braket_I(2)
            + Braket_I(0)
            @ Braket_I(1)
            @ (2 * Braket_X(2)),  # pyright: ignore[reportOperatorIssue]
            Language.QISKIT: SparsePauliOp(
                ["ZYI", "IIX"], coeffs=np.array([0.5 * 0.5, 2])
            ),
            Language.MY_QLM: [Term(0.25, "ZY", [0, 1]), Term(2, "X", [2])],
            None: ((0.5 * pZ) @ (0.5 * pY) @ pI) + (2 * pI @ pI @ pX),
        },
        {
            Language.CIRQ: 1.5 * Cirq_X(a)  # pyright: ignore[reportOperatorIssue]
            + Cirq_I(b) * -2.5 * Cirq_Y(c),
            Language.BRAKET: 1.5
            * Braket_X(0)  # pyright: ignore[reportOperatorIssue]
            @ Braket_I(1)
            @ Braket_I(2)
            + Braket_I(0)
            @ Braket_I(1)
            @ (-2.5 * Braket_Y(2)),  # pyright: ignore[reportOperatorIssue]
            Language.QISKIT: SparsePauliOp(
                ["XII", "IIY"], coeffs=np.array([1.5, -2.5])
            ),
            Language.MY_QLM: [Term(1.5, "X", [0]), Term(-2.5, "Y", [2])],
            None: (1.5 * pX @ pI @ pI) + (pI @ pI @ (-2.5 * pY)),
        },
        {
            Language.CIRQ: 0.25 * Cirq_I(a) * 4 * Cirq_X(b)
            + 3 * Cirq_Y(c),  # pyright: ignore[reportOperatorIssue]
            Language.BRAKET: (
                0.25 * Braket_I(0)
            )  # pyright: ignore[reportOperatorIssue]
            @ (4 * Braket_X(1))  # pyright: ignore[reportOperatorIssue]
            @ Braket_I(2)
            + Braket_I(0)
            @ Braket_I(1)
            @ (3 * Braket_Y(2)),  # pyright: ignore[reportOperatorIssue]
            Language.QISKIT: SparsePauliOp(
                ["IXI", "IIY"], coeffs=np.array([0.25 * 4, 3])
            ),
            Language.MY_QLM: [Term(4 * 0.25, "X", [1]), Term(3, "Y", [2])],
            None: ((0.25 * pI) @ (4 * pX) @ pI) + (pI @ pI @ (3 * pY)),
        },
        {
            Language.CIRQ: Cirq_I(a),
            Language.BRAKET: Braket_I(0),
            Language.QISKIT: SparsePauliOp(["I"]),
            Language.MY_QLM: Term(1, "I", [0]),
            None: pI,
        },
        {
            Language.CIRQ: Cirq_X(a),
            Language.BRAKET: Braket_X(0),
            Language.QISKIT: SparsePauliOp(["X"]),
            Language.MY_QLM: Term(1, "X", [0]),
            None: pX,
        },
        {
            Language.CIRQ: Cirq_Z(a),
            Language.BRAKET: Braket_Z(0),
            Language.QISKIT: SparsePauliOp(["Z"]),
            Language.MY_QLM: Term(1, "Z", [0]),
            None: pZ,
        },
        {
            Language.CIRQ: Cirq_Y(a),
            Language.BRAKET: Braket_Y(0),
            Language.QISKIT: SparsePauliOp(["Y"]),
            Language.MY_QLM: Term(1, "Y", [0]),
            None: pY,
        },
        {
            Language.CIRQ: 1 * Cirq_I(b),
            Language.BRAKET: Braket_I(0) @ Braket_I(1),
            Language.QISKIT: SparsePauliOp(["II"]),
            Language.MY_QLM: Term(1, "II", [0, 1]),
            None: pI @ pI,
        },
        {
            Language.CIRQ: 1 * Cirq_X(b),  # pyright: ignore[reportOperatorIssue]
            Language.BRAKET: Braket_I(0) @ Braket_X(1),
            Language.QISKIT: SparsePauliOp(["IX"]),
            Language.MY_QLM: Term(1, "X", [1]),
            None: pI @ pX,
        },
        {
            Language.CIRQ: 1 * Cirq_Z(b),  # pyright: ignore[reportOperatorIssue]
            Language.BRAKET: Braket_I(0) @ Braket_Z(1),
            Language.QISKIT: SparsePauliOp(["IZ"]),
            Language.MY_QLM: Term(1, "Z", [1]),
            None: pI @ pZ,
        },
        {
            Language.CIRQ: 1 * Cirq_Y(b),  # pyright: ignore[reportOperatorIssue]
            Language.BRAKET: Braket_I(0) @ Braket_Y(1),
            Language.QISKIT: SparsePauliOp(["IY"]),
            Language.MY_QLM: Term(1, "Y", [1]),
            None: pI @ pY,
        },
        {
            Language.CIRQ: 1 * Cirq_I(a) + 1 * Cirq_I(a),
            Language.BRAKET: Braket_I(0) + Braket_I(0),
            Language.QISKIT: SparsePauliOp(["I", "I"]),
            Language.MY_QLM: [Term(1, "I", [0]), Term(1, "I", [0])],
            None: pI + pI,
        },
        {
            Language.CIRQ: 1 * Cirq_I(a)
            + 1 * Cirq_X(a),  # pyright: ignore[reportOperatorIssue]
            Language.BRAKET: Braket_I(0) + Braket_X(0),
            Language.QISKIT: SparsePauliOp(["I", "X"]),
            Language.MY_QLM: [Term(1, "I", [0]), Term(1, "X", [0])],
            None: pI + pX,
        },
        {
            Language.CIRQ: 1 * Cirq_X(a)  # pyright: ignore[reportOperatorIssue]
            + 1 * Cirq_Z(a),  # pyright: ignore[reportOperatorIssue]
            Language.BRAKET: Braket_Z(0) + Braket_X(0),
            Language.QISKIT: SparsePauliOp(["Z", "X"]),
            Language.MY_QLM: [Term(1, "Z", [0]), Term(1, "X", [0])],
            None: pZ + pX,
        },
        {
            Language.CIRQ: 1 * Cirq_Y(a)  # pyright: ignore[reportOperatorIssue]
            + 1 * Cirq_Z(a),  # pyright: ignore[reportOperatorIssue]
            Language.BRAKET: Braket_Y(0) + Braket_Z(0),
            Language.QISKIT: SparsePauliOp(["Y", "Z"]),
            Language.MY_QLM: [Term(1, "Y", [0]), Term(1, "Z", [0])],
            None: pY + pZ,
        },
        {
            Language.CIRQ: 1 * Cirq_X(a)  # pyright: ignore[reportOperatorIssue]
            + 1 * Cirq_Y(a),  # pyright: ignore[reportOperatorIssue]
            Language.BRAKET: Braket_X(0) + Braket_Y(0),
            Language.QISKIT: SparsePauliOp(["X", "Y"]),
            Language.MY_QLM: [Term(1, "X", [0]), Term(1, "Y", [0])],
            None: pX + pY,
        },
    ]


@pytest.mark.parametrize(
    "pauli_strings",
    pauli_strings_in_all_languages(),
)
def test_from_other_language(
    pauli_strings: dict[
        Optional[Language],
        Union[PauliSum, BraketSum, "SparsePauliOp", Term, PauliString],
    ],
):
    mpqp_ps = pauli_strings[None]
    assert isinstance(mpqp_ps, PauliString)
    for language, ps in pauli_strings.items():
        if language is not None:
            assert (
                PauliString.from_other_language(
                    ps, mpqp_ps.nb_qubits if language == Language.CIRQ else 1
                )
                == mpqp_ps
            )


@pytest.mark.parametrize(
    "pauli_strings",
    pauli_strings_in_all_languages(),
)
def test_to_other_language(
    pauli_strings: dict[
        Optional[Language],
        Union[PauliSum, BraketSum, "SparsePauliOp", Term, PauliString],
    ],
):
    mpqp_ps = pauli_strings[None]
    assert isinstance(mpqp_ps, PauliString)
    for language, ps in pauli_strings.items():
        if language is not None:
            if language == Language.BRAKET:
                assert repr(mpqp_ps.to_other_language(language)) == repr(ps)
            else:
                assert mpqp_ps.to_other_language(language) == ps


@pytest.mark.parametrize(
    "mpqp_ps, language",
    product(
        [all_ps[None] for all_ps in pauli_strings_in_all_languages()],
        [Language.BRAKET, Language.CIRQ, Language.MY_QLM, Language.QISKIT],
    ),
)
def test_to_from_other_language(mpqp_ps: PauliString, language: Language):
    print(
        PauliString.from_other_language(
            mpqp_ps.to_other_language(language),
            mpqp_ps.nb_qubits if language == Language.CIRQ else 1,
        ).to_dict()  # type: ignore
    )
    print(
        PauliString.from_other_language(
            mpqp_ps.to_other_language(language),
            mpqp_ps.nb_qubits if language == Language.CIRQ else 1,
        )
        .monomials[0]  # type: ignore
        .coef
    )
    print(mpqp_ps.to_dict())
    assert (
        PauliString.from_other_language(
            mpqp_ps.to_other_language(language),
            mpqp_ps.nb_qubits if language == Language.CIRQ else 1,
        )
        == mpqp_ps
    )


@pytest.mark.parametrize(
    "input_str, subs_dict, expected_str",
    [
        ("2*XZ", None, 2 * pX @ pZ),
        ("theta*IX", {}, symbols("theta") * pI @ pX),
        ("theta*IX", {"theta": 2}, 2 * pI @ pX),
        ("k*XY", {"k": 2}, 2 * pX @ pY),
        ("theta*IX + k*ZY", {"theta": 7, "k": 1}, 7 * pI @ pX + pZ @ pY),
        ("-a*YZ", {"a": -1}, pY @ pZ),
        ("o2*XZ + YI - 3*ZZ", {"o": 3}, 6 * pX @ pZ + pY @ pI - 3 * pZ @ pZ),
        (
            "o*2*XZ + YI - 3o*ZZ",
            None,
            symbols("o") * 2 * pX @ pZ + pY @ pI - 3 * symbols("o") * pZ @ pZ,
        ),
    ],
)
def test_pauli_string_from_str(
    input_str: str, subs_dict: Optional[dict[str, Coef]], expected_str: PauliString
):
    assert PauliString.from_str(input_str, subs_dict) == expected_str


@pytest.mark.parametrize(
    "prefix, atom, postfix, expected_ps",
    [
        (3, pX, None, pI @ pI @ pI @ pX),
        (0, pY, 2, pY @ pI @ pI),
        (2, pZ, 1, pI @ pI @ pZ @ pI),
        (0, pI, 1, pI @ pI),
        (0, pX, 0, pX),
    ],
)
def test_pauli_monomial_from_atom(
    prefix: int, atom: PauliStringAtom, postfix: Optional[int], expected_ps: PauliString
):
    result = atom(prefix) if postfix is None else atom(prefix, postfix)
    assert result == expected_ps
