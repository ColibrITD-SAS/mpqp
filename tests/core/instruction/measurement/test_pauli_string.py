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
    from cirq.ops.linear_combinations import PauliSum
    from braket.circuits.observables import Sum as BraketSum
    from braket.circuits.observables import I as Braket_I
    from braket.circuits.observables import X as Braket_X
    from braket.circuits.observables import Y as Braket_Y
    from braket.circuits.observables import Z as Braket_Z
    from qat.core.wrappers.observable import Term

import numpy as np
import numpy.typing as npt
import pytest
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


@pytest.fixture
def list_pauli_strings() -> list[PauliString]:
    return [
        pX @ pI @ pI + pI @ pY @ pI + pI @ pI @ pZ,
        pX @ pY @ pZ,
        pI @ pI @ pI + pI @ pZ @ pI + pI @ pI @ pX,
        pY @ pZ @ pX,
        pZ @ pY @ pI + pI @ pI @ pX,
        pX @ pI @ pI + pI @ pI @ pY,
        pI @ pX @ pI + pI @ pI @ pY,
        2 * pX @ pI @ pI + 3 * pI @ pY @ pI + 4 * pI @ pI @ pZ,
        -pX @ (1.5 * pY) @ (0.5 * pZ),
        ((0.5 * pZ) @ (0.5 * pY) @ pI) + (2 * pI @ pI @ pX),
        (1.5 * pX @ pI @ pI) + (pI @ pI @ (-2.5 * pY)),
        ((0.25 * pI) @ (4 * pX) @ pI) + (pI @ pI @ (3 * pY)),
        pI,
        pX,
        pZ,
        pY,
        pI @ pI,
        pI @ pX,
        pI @ pZ,
        pI @ pY,
        pI + pI,
        pI + pX,
        pZ + pX,
        pY + pZ,
        pX + pY,
    ]


@pytest.fixture
def list_pauli_strings_cirq() -> list["PauliSum"]:
    from cirq.ops.identity import I as Cirq_I
    from cirq.ops.pauli_gates import X as Cirq_X
    from cirq.ops.pauli_gates import Y as Cirq_Y
    from cirq.ops.pauli_gates import Z as Cirq_Z
    from cirq.devices.line_qubit import LineQubit

    a, b, c = LineQubit.range(3)

    return [
        Cirq_X(a) + Cirq_Y(b) + Cirq_Z(c),  # pyright: ignore[reportOperatorIssue]
        Cirq_X(a) * Cirq_Y(b) * Cirq_Z(c),  # pyright: ignore[reportOperatorIssue]
        Cirq_I(a) + Cirq_Z(b) + Cirq_X(c),
        Cirq_Y(a) * Cirq_Z(b) * Cirq_X(c),  # pyright: ignore[reportOperatorIssue]
        Cirq_Z(a) * Cirq_Y(b) + Cirq_X(c),  # pyright: ignore[reportOperatorIssue]
        Cirq_X(a) + Cirq_I(b) * Cirq_Y(c),
        Cirq_I(a) * Cirq_X(b) + Cirq_Y(c),
        2 * Cirq_X(a)  # pyright: ignore[reportOperatorIssue]
        + 3 * Cirq_Y(b)  # pyright: ignore[reportOperatorIssue]
        + 4 * Cirq_Z(c),  # pyright: ignore[reportOperatorIssue]
        -Cirq_X(a)  # pyright: ignore[reportOperatorIssue]
        * (1.5 * Cirq_Y(b))  # pyright: ignore[reportOperatorIssue]
        * (0.5 * Cirq_Z(c)),  # pyright: ignore[reportOperatorIssue]
        0.5 * Cirq_Z(a) * 0.5 * Cirq_Y(b)  # pyright: ignore[reportOperatorIssue]
        + 2 * Cirq_X(c),  # pyright: ignore[reportOperatorIssue]
        1.5 * Cirq_X(a)  # pyright: ignore[reportOperatorIssue]
        + Cirq_I(b) * -2.5 * Cirq_Y(c),
        0.25 * Cirq_I(a) * 4 * Cirq_X(b)
        + 3 * Cirq_Y(c),  # pyright: ignore[reportOperatorIssue]
        Cirq_I(a),
        Cirq_X(a),
        Cirq_Z(a),
        Cirq_Y(a),
        1 * Cirq_I(b),
        1 * Cirq_X(b),  # pyright: ignore[reportOperatorIssue]
        1 * Cirq_Z(b),  # pyright: ignore[reportOperatorIssue]
        1 * Cirq_Y(b),  # pyright: ignore[reportOperatorIssue]
        1 * Cirq_I(a) + 1 * Cirq_I(a),
        1 * Cirq_I(a) + 1 * Cirq_X(a),  # pyright: ignore[reportOperatorIssue]
        1 * Cirq_X(a)  # pyright: ignore[reportOperatorIssue]
        + 1 * Cirq_Z(a),  # pyright: ignore[reportOperatorIssue]
        1 * Cirq_Y(a)  # pyright: ignore[reportOperatorIssue]
        + 1 * Cirq_Z(a),  # pyright: ignore[reportOperatorIssue]
        1 * Cirq_X(a)  # pyright: ignore[reportOperatorIssue]
        + 1 * Cirq_Y(a),  # pyright: ignore[reportOperatorIssue]
    ]


@pytest.fixture
def list_pauli_strings_braket() -> (
    list[Union["BraketSum", "Braket_I", "Braket_X", "Braket_Y", "Braket_Z"]]
):
    from braket.circuits.observables import I as Braket_I
    from braket.circuits.observables import X as Braket_X
    from braket.circuits.observables import Y as Braket_Y
    from braket.circuits.observables import Z as Braket_Z

    return [
        Braket_X() @ Braket_I() @ Braket_I()
        + Braket_I() @ Braket_Y() @ Braket_I()
        + Braket_I() @ Braket_I() @ Braket_Z(),
        Braket_X() @ Braket_Y() @ Braket_Z(),
        Braket_I() @ Braket_I() @ Braket_I()
        + Braket_I() @ Braket_Z() @ Braket_I()
        + Braket_I() @ Braket_I() @ Braket_X(),
        Braket_Y() @ Braket_Z() @ Braket_X(),
        Braket_Z() @ Braket_Y() @ Braket_I() + Braket_I() @ Braket_I() @ Braket_X(),
        Braket_X() @ Braket_I() @ Braket_I() + Braket_I() @ Braket_I() @ Braket_Y(),
        Braket_I() @ Braket_X() @ Braket_I() + Braket_I() @ Braket_I() @ Braket_Y(),
        2 * Braket_X() @ Braket_I() @ Braket_I()  # pyright: ignore[reportOperatorIssue]
        + 3
        * Braket_I()  # pyright: ignore[reportOperatorIssue]
        @ Braket_Y()
        @ Braket_I()
        + 4
        * Braket_I()  # pyright: ignore[reportOperatorIssue]
        @ Braket_I()
        @ Braket_Z(),
        (-1 * Braket_X())  # pyright: ignore[reportOperatorIssue]
        @ (1.5 * Braket_Y())  # pyright: ignore[reportOperatorIssue]
        @ (0.5 * Braket_Z()),  # pyright: ignore[reportOperatorIssue]
        (0.5 * Braket_Z())  # pyright: ignore[reportOperatorIssue]
        @ (0.5 * Braket_Y())  # pyright: ignore[reportOperatorIssue]
        @ Braket_I()
        + Braket_I()
        @ Braket_I()
        @ (2 * Braket_X()),  # pyright: ignore[reportOperatorIssue]
        1.5
        * Braket_X()  # pyright: ignore[reportOperatorIssue]
        @ Braket_I()
        @ Braket_I()
        + Braket_I()
        @ Braket_I()
        @ (-2.5 * Braket_Y()),  # pyright: ignore[reportOperatorIssue]
        (0.25 * Braket_I())  # pyright: ignore[reportOperatorIssue]
        @ (4 * Braket_X())  # pyright: ignore[reportOperatorIssue]
        @ Braket_I()
        + Braket_I()
        @ Braket_I()
        @ (3 * Braket_Y()),  # pyright: ignore[reportOperatorIssue]
        Braket_I(),
        Braket_X(),
        Braket_Z(),
        Braket_Y(),
        Braket_I() @ Braket_I(),
        Braket_I() @ Braket_X(),
        Braket_I() @ Braket_Z(),
        Braket_I() @ Braket_Y(),
        Braket_I() + Braket_I(),
        Braket_I() + Braket_X(),
        Braket_Z() + Braket_X(),
        Braket_Y() + Braket_Z(),
        Braket_X() + Braket_Y(),
    ]


@pytest.fixture
def list_pauli_strings_qiskit() -> list["SparsePauliOp"]:
    from qiskit.quantum_info import SparsePauliOp

    return [
        SparsePauliOp(["XII", "IYI", "IIZ"]),
        SparsePauliOp(["XYZ"]),
        SparsePauliOp(["III", "IZI", "IIX"]),
        SparsePauliOp(["YZX"]),
        SparsePauliOp(["ZYI", "IIX"]),
        SparsePauliOp(["XII", "IIY"]),
        SparsePauliOp(["IXI", "IIY"]),
        SparsePauliOp(
            ["XII", "IYI", "IIZ"],
            coeffs=[2.0, 3.0, 4.0],  # pyright: ignore[reportArgumentType]
        ),
        SparsePauliOp(["XYZ"], coeffs=[-0.75]),  # pyright: ignore[reportArgumentType]
        SparsePauliOp(
            ["ZYI", "IIX"], coeffs=[0.25, 2.0]  # pyright: ignore[reportArgumentType]
        ),
        SparsePauliOp(
            ["XII", "IIY"], coeffs=[1.5, -2.5]  # pyright: ignore[reportArgumentType]
        ),
        SparsePauliOp(
            ["IXI", "IIY"], coeffs=[1.0, 3.0]  # pyright: ignore[reportArgumentType]
        ),
        SparsePauliOp(["I"]),
        SparsePauliOp(["X"]),
        SparsePauliOp(["Z"]),
        SparsePauliOp(["Y"]),
        SparsePauliOp(["II"]),
        SparsePauliOp(["IX"]),
        SparsePauliOp(["IZ"]),
        SparsePauliOp(["IY"]),
        SparsePauliOp(["I", "I"]),
        SparsePauliOp(["I", "X"]),
        SparsePauliOp(["Z", "X"]),
        SparsePauliOp(["Y", "Z"]),
        SparsePauliOp(["X", "Y"]),
    ]


@pytest.fixture
def list_pauli_strings_my_qlm() -> list[list["Term"]]:
    from qat.core.wrappers.observable import Term

    return [
        [
            Term(1, "X", [0]),
            Term(1, "Y", [1]),
            Term(1, "Z", [2]),
        ],
        Term(1, "XYZ", [0, 1, 2]),
        [
            Term(1, "I", [0]),
            Term(1, "Z", [1]),
            Term(1, "X", [2]),
        ],
        Term(1, "YZX", [0, 1, 2]),
        [Term(1, "ZY", [0, 1]), Term(1, "X", [2])],
        [Term(1, "X", [0]), Term(1, "Y", [2])],
        [Term(1, "X", [1]), Term(1, "Y", [2])],
        [Term(2, "X", [0]), Term(3, "Y", [1]), Term(4, "Z", [2])],
        Term(-0.75, "XYZ", [0, 1, 2]),
        [Term(0.25, "ZY", [0, 1]), Term(2, "X", [2])],
        [Term(1.5, "X", [0]), Term(-2.5, "Y", [2])],
        [Term(4 * 0.25, "X", [1]), Term(3, "Y", [2])],
        Term(1, "I", [0]),
        Term(1, "X", [0]),
        Term(1, "Z", [0]),
        Term(1, "Y", [0]),
        Term(1, "II", [0, 1]),
        Term(1, "X", [1]),
        Term(1, "Z", [1]),
        Term(1, "Y", [1]),
        [Term(1, "I", [0]), Term(1, "I", [0])],
        [Term(1, "I", [0]), Term(1, "X", [0])],
        [Term(1, "Z", [0]), Term(1, "X", [0])],
        [Term(1, "Y", [0]), Term(1, "Z", [0])],
        [Term(1, "X", [0]), Term(1, "Y", [0])],
    ]


@pytest.mark.provider("cirq")
def test_from_other_language_cirq(
    list_pauli_strings: list[PauliString],
    list_pauli_strings_cirq: list["PauliSum"],
):
    for mpqp_ps, cirq_ps in zip(list_pauli_strings, list_pauli_strings_cirq):
        assert PauliString.from_other_language(cirq_ps, mpqp_ps.nb_qubits) == mpqp_ps


@pytest.mark.provider("braket")
def test_from_other_language_braket(
    list_pauli_strings: list[PauliString],
    list_pauli_strings_braket: list["BraketSum"],
):
    for mpqp_ps, braket_ps in zip(list_pauli_strings, list_pauli_strings_braket):
        assert PauliString.from_other_language(braket_ps) == mpqp_ps


@pytest.mark.provider("qiskit")
def test_from_other_language_qiskit(
    list_pauli_strings: list[PauliString],
    list_pauli_strings_qiskit: list["SparsePauliOp"],
):
    for mpqp_ps, qiskit_ps in zip(list_pauli_strings, list_pauli_strings_qiskit):
        assert PauliString.from_other_language(qiskit_ps) == mpqp_ps


@pytest.mark.provider("myqlm")
def test_from_other_language_my_qlm(
    list_pauli_strings: list[PauliString],
    list_pauli_strings_my_qlm: list["Term"],
):
    for mpqp_ps, my_qlm_ps in zip(list_pauli_strings, list_pauli_strings_my_qlm):
        assert PauliString.from_other_language(my_qlm_ps) == mpqp_ps


@pytest.mark.provider("cirq")
def test_to_other_language_cirq(
    list_pauli_strings: list[PauliString],
    list_pauli_strings_cirq: list["PauliSum"],
):
    for mpqp_ps, cirq_ps in zip(list_pauli_strings, list_pauli_strings_cirq):
        assert mpqp_ps.to_other_language(Language.CIRQ) == cirq_ps


@pytest.mark.provider("braket")
def test_to_other_language_braket(
    list_pauli_strings: list[PauliString],
    list_pauli_strings_braket: list["BraketSum"],
):
    for mpqp_ps, braket_ps in zip(list_pauli_strings, list_pauli_strings_braket):
        assert repr(mpqp_ps.to_other_language(Language.BRAKET)) == repr(braket_ps)


@pytest.mark.provider("qiskit")
def test_to_other_language_qiskit(
    list_pauli_strings: list[PauliString],
    list_pauli_strings_qiskit: list["SparsePauliOp"],
):
    for mpqp_ps, qiskit_ps in zip(list_pauli_strings, list_pauli_strings_qiskit):
        assert mpqp_ps.to_other_language(Language.QISKIT) == qiskit_ps


@pytest.mark.provider("myqlm")
def test_to_other_language_my_qlm(
    list_pauli_strings: list[PauliString],
    list_pauli_strings_my_qlm: list[list["Term"]],
):
    for mpqp_ps, my_qlm_ps in zip(list_pauli_strings, list_pauli_strings_my_qlm):
        assert mpqp_ps.to_other_language(Language.MY_QLM) == my_qlm_ps


@pytest.mark.provider("cirq")
def test_to_from_other_language_cirq(
    list_pauli_strings: list[PauliString],
):
    for mpqp_ps in list_pauli_strings:
        assert (
            PauliString.from_other_language(
                mpqp_ps.to_other_language(Language.CIRQ),
                mpqp_ps.nb_qubits,
            )
            == mpqp_ps
        )


@pytest.mark.provider("braket")
def test_to_from_other_language_braket(
    list_pauli_strings: list[PauliString],
):
    for mpqp_ps in list_pauli_strings:
        assert (
            PauliString.from_other_language(
                mpqp_ps.to_other_language(Language.BRAKET),
            )
            == mpqp_ps
        )


@pytest.mark.provider("qiskit")
def test_to_from_other_language_qiskit(
    list_pauli_strings: list[PauliString],
):
    for mpqp_ps in list_pauli_strings:
        assert (
            PauliString.from_other_language(
                mpqp_ps.to_other_language(Language.QISKIT),
            )
            == mpqp_ps
        )


@pytest.mark.provider("myqlm")
def test_to_from_other_language_my_qlm(
    list_pauli_strings: list[PauliString],
):
    for mpqp_ps in list_pauli_strings:
        assert (
            PauliString.from_other_language(
                mpqp_ps.to_other_language(Language.MY_QLM),
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
