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

from mpqp import Language
from mpqp.measures import I, PauliString, X, Y, Z
from mpqp.tools.maths import matrix_eq


def pauli_matrix_mapping():
    return [
        (I, np.eye(2)),
        ((I @ I), np.eye(4)),
        ((I + I), (2 * np.eye(2))),
        ((I + I) @ I, (2 * np.eye(4))),
        ((X + X), np.array([[0, 2.0], [2.0, 0]])),
        ((X + Z), np.array([[1.0, 1.0], [1.0, -1.0]])),
        ((2 * I), (2 * np.eye(2))),
        ((symbols("a") * I), (symbols("a") * np.eye(2))),
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
    print(ps1)
    print(ps2)
    ps1 @= ps1
    print(ps1)
    assert False
    # assert matrix_eq(op(clean_ps1, ps2).to_matrix(), np.kron(clean_matrix1, matrix2))


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
    assert matrix_eq(op(clean_ps1, ps2).to_matrix(), np.kron(clean_matrix1, matrix2))


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


@pytest.mark.parametrize(
    "init_ps, subs_dict, expected_ps",
    [
        # Basic substitution with numeric values
        (symbols("theta") * I @ X, {"theta": np.pi}, np.pi * I @ X),
        (symbols("k") * X @ Y, {"k": 2}, 2 * X @ Y),
        (symbols("a") * Y @ Z, {"a": -1}, -Y @ Z),
        # Multiple variable substitutions
        (
            symbols("theta") * I @ X + symbols("k") * Z @ Y,
            {"theta": np.pi, "k": 1},
            np.pi * I @ X + Z @ Y,
        ),
        (symbols("a") * X @ X + symbols("b") * Y @ Y, {"a": 0, "b": 3}, 3 * Y @ Y),
        # Removing symbolic values
        (symbols("theta") * I @ X, {"theta": np.pi}, np.pi * I @ X),
        (
            symbols("theta") * X @ Y + symbols("phi") * Y @ Z,
            {"theta": 1, "phi": 2},
            X @ Y + 2 * Y @ Z,
        ),
        # No substitutions (should remain the same)
        (symbols("theta") * I @ X, {}, symbols("theta") * I @ X),
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
            Language.BRAKET: Braket_X() @ Braket_I() @ Braket_I()
            + Braket_I() @ Braket_Y() @ Braket_I()
            + Braket_I() @ Braket_I() @ Braket_Z(),
            Language.QISKIT: SparsePauliOp(["IIX", "IYI", "ZII"]),
            Language.MY_QLM: [Term(1, "X", [0]), Term(1, "Y", [1]), Term(1, "Z", [2])],
            None: X @ I @ I + I @ Y @ I + I @ I @ Z,
        },
        {
            Language.CIRQ: Cirq_X(a)
            * Cirq_Y(b)  # pyright: ignore[reportOperatorIssue]
            * Cirq_Z(c),
            Language.BRAKET: Braket_X() @ Braket_Y() @ Braket_Z(),
            Language.QISKIT: SparsePauliOp(["ZYX"]),
            Language.MY_QLM: Term(1, "XYZ", [0, 1, 2]),
            None: X @ Y @ Z,
        },
        {
            Language.CIRQ: Cirq_I(a) + Cirq_Z(b) + Cirq_X(c),
            Language.BRAKET: Braket_I() @ Braket_I() @ Braket_I()
            + Braket_I() @ Braket_Z() @ Braket_I()
            + Braket_I() @ Braket_I() @ Braket_X(),
            Language.QISKIT: SparsePauliOp(["III", "IZI", "XII"]),
            Language.MY_QLM: [Term(1, "I", [0]), Term(1, "Z", [1]), Term(1, "X", [2])],
            None: I @ I @ I + I @ Z @ I + I @ I @ X,
        },
        {
            Language.CIRQ: Cirq_Y(a)
            * Cirq_Z(b)  # pyright: ignore[reportOperatorIssue]
            * Cirq_X(c),
            Language.BRAKET: Braket_Y() @ Braket_Z() @ Braket_X(),
            Language.QISKIT: SparsePauliOp(["XZY"]),
            Language.MY_QLM: Term(1, "YZX", [0, 1, 2]),
            None: Y @ Z @ X,
        },
        {
            Language.CIRQ: Cirq_Z(a) * Cirq_Y(b)  # pyright: ignore[reportOperatorIssue]
            + Cirq_X(c),
            Language.BRAKET: Braket_Z() @ Braket_Y() @ Braket_I()
            + Braket_I() @ Braket_I() @ Braket_X(),
            Language.QISKIT: SparsePauliOp(["IYZ", "XII"]),
            Language.MY_QLM: [Term(1, "ZY", [0, 1]), Term(1, "X", [2])],
            None: Z @ Y @ I + I @ I @ X,
        },
        {
            Language.CIRQ: Cirq_X(a) + Cirq_I(b) * Cirq_Y(c),
            Language.BRAKET: Braket_X() @ Braket_I() @ Braket_I()
            + Braket_I() @ Braket_I() @ Braket_Y(),
            Language.QISKIT: SparsePauliOp(["IIX", "YII"]),
            Language.MY_QLM: [Term(1, "X", [0]), Term(1, "Y", [2])],
            None: X @ I @ I + I @ I @ Y,
        },
        {
            Language.CIRQ: Cirq_I(a) * Cirq_X(b) + Cirq_Y(c),
            Language.BRAKET: Braket_I() @ Braket_X() @ Braket_I()
            + Braket_I() @ Braket_I() @ Braket_Y(),
            Language.QISKIT: SparsePauliOp(["IXI", "YII"]),
            Language.MY_QLM: [Term(1, "X", [1]), Term(1, "Y", [2])],
            None: I @ X @ I + I @ I @ Y,
        },
        {
            Language.CIRQ: 2 * Cirq_X(a)  # pyright: ignore[reportOperatorIssue]
            + 3 * Cirq_Y(b)  # pyright: ignore[reportOperatorIssue]
            + 4 * Cirq_Z(c),  # pyright: ignore[reportOperatorIssue]
            Language.BRAKET: 2
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
            Language.QISKIT: SparsePauliOp(
                ["IIX", "IYI", "ZII"], coeffs=np.array([2, 3, 4])
            ),
            Language.MY_QLM: [Term(2, "X", [0]), Term(3, "Y", [1]), Term(4, "Z", [2])],
            None: 2 * X @ I @ I + 3 * I @ Y @ I + 4 * I @ I @ Z,
        },
        {
            Language.CIRQ: -Cirq_X(a)  # pyright: ignore[reportOperatorIssue]
            * (1.5 * Cirq_Y(b))  # pyright: ignore[reportOperatorIssue]
            * (0.5 * Cirq_Z(c)),  # pyright: ignore[reportOperatorIssue]
            Language.BRAKET: (-1 * Braket_X())  # pyright: ignore[reportOperatorIssue]
            @ (1.5 * Braket_Y())  # pyright: ignore[reportOperatorIssue]
            @ (0.5 * Braket_Z()),  # pyright: ignore[reportOperatorIssue]
            Language.QISKIT: SparsePauliOp(["ZYX"], coeffs=np.array([-1 * 1.5 * 0.5])),
            Language.MY_QLM: Term(-0.75, "XYZ", [0, 1, 2]),
            None: -X @ (1.5 * Y) @ (0.5 * Z),
        },
        {
            Language.CIRQ: 0.5
            * Cirq_Z(a)  # pyright: ignore[reportOperatorIssue]
            * 0.5
            * Cirq_Y(b)
            + 2 * Cirq_X(c),  # pyright: ignore[reportOperatorIssue]
            Language.BRAKET: (0.5 * Braket_Z())  # pyright: ignore[reportOperatorIssue]
            @ (0.5 * Braket_Y())  # pyright: ignore[reportOperatorIssue]
            @ Braket_I()
            + Braket_I()
            @ Braket_I()
            @ (2 * Braket_X()),  # pyright: ignore[reportOperatorIssue]
            Language.QISKIT: SparsePauliOp(
                ["IYZ", "XII"], coeffs=np.array([0.5 * 0.5, 2])
            ),
            Language.MY_QLM: [Term(0.25, "ZY", [0, 1]), Term(2, "X", [2])],
            None: ((0.5 * Z) @ (0.5 * Y) @ I) + (2 * I @ I @ X),
        },
        {
            Language.CIRQ: 1.5 * Cirq_X(a)  # pyright: ignore[reportOperatorIssue]
            + Cirq_I(b) * -2.5 * Cirq_Y(c),
            Language.BRAKET: 1.5
            * Braket_X()  # pyright: ignore[reportOperatorIssue]
            @ Braket_I()
            @ Braket_I()
            + Braket_I()
            @ Braket_I()
            @ (-2.5 * Braket_Y()),  # pyright: ignore[reportOperatorIssue]
            Language.QISKIT: SparsePauliOp(
                ["IIX", "YII"], coeffs=np.array([1.5, -2.5])
            ),
            Language.MY_QLM: [Term(1.5, "X", [0]), Term(-2.5, "Y", [2])],
            None: (1.5 * X @ I @ I) + (I @ I @ (-2.5 * Y)),
        },
        {
            Language.CIRQ: 0.25 * Cirq_I(a) * 4 * Cirq_X(b)
            + 3 * Cirq_Y(c),  # pyright: ignore[reportOperatorIssue]
            Language.BRAKET: (0.25 * Braket_I())  # pyright: ignore[reportOperatorIssue]
            @ (4 * Braket_X())  # pyright: ignore[reportOperatorIssue]
            @ Braket_I()
            + Braket_I()
            @ Braket_I()
            @ (3 * Braket_Y()),  # pyright: ignore[reportOperatorIssue]
            Language.QISKIT: SparsePauliOp(
                ["IXI", "YII"], coeffs=np.array([0.25 * 4, 3])
            ),
            Language.MY_QLM: [Term(4 * 0.25, "X", [1]), Term(3, "Y", [2])],
            None: ((0.25 * I) @ (4 * X) @ I) + (I @ I @ (3 * Y)),
        },
        {
            Language.CIRQ: Cirq_I(a),
            Language.BRAKET: Braket_I(),
            Language.QISKIT: SparsePauliOp(["I"]),
            Language.MY_QLM: Term(1, "I", [0]),
            None: I,
        },
        {
            Language.CIRQ: Cirq_X(a),
            Language.BRAKET: Braket_X(),
            Language.QISKIT: SparsePauliOp(["X"]),
            Language.MY_QLM: Term(1, "X", [0]),
            None: X,
        },
        {
            Language.CIRQ: Cirq_Z(a),
            Language.BRAKET: Braket_Z(),
            Language.QISKIT: SparsePauliOp(["Z"]),
            Language.MY_QLM: Term(1, "Z", [0]),
            None: Z,
        },
        {
            Language.CIRQ: Cirq_Y(a),
            Language.BRAKET: Braket_Y(),
            Language.QISKIT: SparsePauliOp(["Y"]),
            Language.MY_QLM: Term(1, "Y", [0]),
            None: Y,
        },
        {
            Language.CIRQ: 1 * Cirq_I(b),
            Language.BRAKET: Braket_I() @ Braket_I(),
            Language.QISKIT: SparsePauliOp(["II"]),
            Language.MY_QLM: Term(1, "II", [0, 1]),
            None: I @ I,
        },
        {
            Language.CIRQ: 1 * Cirq_X(b),  # pyright: ignore[reportOperatorIssue]
            Language.BRAKET: Braket_I() @ Braket_X(),
            Language.QISKIT: SparsePauliOp(["XI"]),
            Language.MY_QLM: Term(1, "X", [1]),
            None: I @ X,
        },
        {
            Language.CIRQ: 1 * Cirq_Z(b),  # pyright: ignore[reportOperatorIssue]
            Language.BRAKET: Braket_I() @ Braket_Z(),
            Language.QISKIT: SparsePauliOp(["ZI"]),
            Language.MY_QLM: Term(1, "Z", [1]),
            None: I @ Z,
        },
        {
            Language.CIRQ: 1 * Cirq_Y(b),  # pyright: ignore[reportOperatorIssue]
            Language.BRAKET: Braket_I() @ Braket_Y(),
            Language.QISKIT: SparsePauliOp(["YI"]),
            Language.MY_QLM: Term(1, "Y", [1]),
            None: I @ Y,
        },
        {
            Language.CIRQ: 1 * Cirq_I(a) + 1 * Cirq_I(a),
            Language.BRAKET: Braket_I() + Braket_I(),
            Language.QISKIT: SparsePauliOp(["I", "I"]),
            Language.MY_QLM: [Term(1, "I", [0]), Term(1, "I", [0])],
            None: I + I,
        },
        {
            Language.CIRQ: 1 * Cirq_I(a)
            + 1 * Cirq_X(a),  # pyright: ignore[reportOperatorIssue]
            Language.BRAKET: Braket_I() + Braket_X(),
            Language.QISKIT: SparsePauliOp(["I", "X"]),
            Language.MY_QLM: [Term(1, "I", [0]), Term(1, "X", [0])],
            None: I + X,
        },
        {
            Language.CIRQ: 1 * Cirq_X(a)  # pyright: ignore[reportOperatorIssue]
            + 1 * Cirq_Z(a),  # pyright: ignore[reportOperatorIssue]
            Language.BRAKET: Braket_Z() + Braket_X(),
            Language.QISKIT: SparsePauliOp(["Z", "X"]),
            Language.MY_QLM: [Term(1, "Z", [0]), Term(1, "X", [0])],
            None: Z + X,
        },
        {
            Language.CIRQ: 1 * Cirq_Y(a)  # pyright: ignore[reportOperatorIssue]
            + 1 * Cirq_Z(a),  # pyright: ignore[reportOperatorIssue]
            Language.BRAKET: Braket_Y() + Braket_Z(),
            Language.QISKIT: SparsePauliOp(["Y", "Z"]),
            Language.MY_QLM: [Term(1, "Y", [0]), Term(1, "Z", [0])],
            None: Y + Z,
        },
        {
            Language.CIRQ: 1 * Cirq_X(a)  # pyright: ignore[reportOperatorIssue]
            + 1 * Cirq_Y(a),  # pyright: ignore[reportOperatorIssue]
            Language.BRAKET: Braket_X() + Braket_Y(),
            Language.QISKIT: SparsePauliOp(["X", "Y"]),
            Language.MY_QLM: [Term(1, "X", [0]), Term(1, "Y", [0])],
            None: X + Y,
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
