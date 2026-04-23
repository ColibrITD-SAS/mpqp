from typing import Any, Type

import numpy as np
import pytest
from numpy import array  # pyright: ignore[reportUnusedImport]
from mpqp.core.circuit import QCircuit
from mpqp.core.languages import Language
from mpqp.tools.maths import matrix_eq, rand_unitary_matrix
from mpqp.gates import *


def all_cases_controlled_gates():
    return [
        CustomControlledGate([0, 2], X(1)),
        CustomControlledGate([0, 1], X(2)),
        CustomControlledGate([1, 2], X(0)),
        CustomControlledGate([1], CustomGate(rand_unitary_matrix(4), [0, 2])),
        CustomControlledGate([2], CustomGate(rand_unitary_matrix(4), [0, 1])),
        CustomControlledGate([0], CustomGate(rand_unitary_matrix(4), [1, 2])),
        CustomControlledGate([1], CustomGate(rand_unitary_matrix(8), [0, 2, 3])),
    ]


@pytest.mark.parametrize(
    "gate",
    [
        (CustomControlledGate([0, 1], CustomGate(np.array([[1, 0], [0, -1]]), 2))),
        (CustomControlledGate([0], CustomGate(np.array([[1, 0], [0, -1]]), 2))),
        (CustomControlledGate(0, Z(2))),
        (CustomControlledGate(1, Rz(np.pi, 2))),
        (CustomControlledGate(1, SWAP(0, 2))),
    ],
)
def test_gate_repr(gate: CustomControlledGate) -> None:
    assert gate == eval(gate.__repr__())


@pytest.mark.parametrize(
    "gate",
    [CustomControlledGate(3, TOF([0, 1], 2)), CustomControlledGate(3, CNOT(0, 1))],
)
def test_controlled_gate_repr(gate: CustomControlledGate) -> None:
    assert isinstance(gate.non_controlled_gate, X)


@pytest.mark.parametrize(
    "gate, args",
    [
        (CNOT, (-1, 0)),
        (TOF, ([1, -1], 0)),
        (CustomControlledGate, (-10, Z(0))),
        (CustomControlledGate, ([0, 1, -3], S(2))),
    ],
)
def test_negative_indices(gate: Type[Gate], args: tuple[Any]):
    with pytest.raises(ValueError):
        gate(*args)


@pytest.mark.parametrize(
    "gate, expected",
    [
        (CustomControlledGate(0, Y(1)), CustomControlledGate(0, Y(1))),
        (CustomControlledGate(1, Ry(np.pi, 0)), CustomControlledGate(1, Ry(-np.pi, 0))),
        (
            CustomControlledGate(2, CustomGate(S(0).to_matrix(), [0])),
            CustomControlledGate(2, CustomGate(S_dagger(0).to_matrix(), [0])),
        ),
    ],
)
def test_inverse(gate: CustomControlledGate, expected: CustomControlledGate):
    assert gate.inverse() == expected


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize(
    "gate",
    all_cases_controlled_gates(),
)
def test_translation_customcontrolledgate_qiskit(gate: CustomControlledGate):
    c = QCircuit([gate])
    c_qiskit = c.to_other_language(Language.QISKIT)
    c_translated = QCircuit().from_other_language(c_qiskit)
    assert matrix_eq(c.to_matrix(), c_translated.to_matrix())


@pytest.mark.provider("cirq")
@pytest.mark.parametrize(
    "gate",
    all_cases_controlled_gates(),
)
def test_translation_customcontrolledgate_cirq(gate: CustomControlledGate):
    c = QCircuit([gate])
    c_qiskit = c.to_other_language(Language.CIRQ)
    c_translated = QCircuit().from_other_language(c_qiskit)
    assert matrix_eq(c.to_matrix(), c_translated.to_matrix())


@pytest.mark.provider("braket")
@pytest.mark.parametrize(
    "gate",
    all_cases_controlled_gates(),
)
def test_translation_customcontrolledgate_braket(gate: CustomControlledGate):
    c = QCircuit([gate])
    c_qiskit = c.to_other_language(Language.BRAKET)
    c_translated = QCircuit().from_other_language(c_qiskit)
    assert matrix_eq(c.to_matrix(), c_translated.to_matrix())


@pytest.mark.provider("qasm3")
@pytest.mark.parametrize(
    "gate",
    all_cases_controlled_gates(),
)
def test_translation_customcontrolledgate_qasm3(gate: CustomControlledGate):
    c = QCircuit([gate])
    c_qiskit = c.to_other_language(Language.QASM3)
    c_translated = QCircuit().from_other_language(c_qiskit)
    assert matrix_eq(c.to_matrix(), c_translated.to_matrix())


@pytest.mark.provider("qasm2")
@pytest.mark.parametrize(
    "gate",
    all_cases_controlled_gates(),
)
def test_translation_customcontrolledgate_qasm2(gate: CustomControlledGate):
    c = QCircuit([gate])
    c_qiskit = c.to_other_language(Language.QASM2)
    c_translated = QCircuit().from_other_language(c_qiskit)
    assert matrix_eq(c.to_matrix(), c_translated.to_matrix())
