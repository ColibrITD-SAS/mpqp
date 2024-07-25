from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest
from qiskit.circuit import Parameter

from mpqp.core.instruction.gates.controlled_gate import ControlledGate
from mpqp.core.instruction.gates.gate import Gate
from mpqp.core.languages import Language
from mpqp.gates import *


@pytest.mark.parametrize(
    "gate, expected_repr",
    [
        (Id(0), "Id(0)"),
        (X(0), "X(0)"),
        (Y(0), "Y(0)"),
        (Z(0), "Z(0)"),
        (H(0), "H(0)"),
        (P(np.pi / 3, 1), f"P({np.pi / 3}, 1)"),
        (S(0), "S(0)"),
        (T(0), "T(0)"),
        (SWAP(0, 1), "SWAP(0, 1)"),
        (U(np.pi / 3, 0, np.pi / 4, 0), f"U({np.pi / 3}, 0, {np.pi / 4}, 0)"),
        (Rx(np.pi / 5, 1), f"Rx({np.pi / 5}, 1)"),
        (Ry(np.pi / 5, 1), f"Ry({np.pi / 5}, 1)"),
        (Rz(np.pi / 5, 1), f"Rz({np.pi / 5}, 1)"),
        (Rk(5, 0), "Rk(5, 0)"),
        (CNOT(0, 1), "CNOT(0, 1)"),
        (CZ(0, 1), "CZ(0, 1)"),
        (CRk(4, 0, 1), "CRk(4, 0, 1)"),
        (TOF([0, 1], 2), "TOF([0, 1], 2)"),
    ],
)
def test_gate_repr(gate: Gate, expected_repr: str) -> None:
    assert repr(gate) == expected_repr


class CustomControlledGate(ControlledGate):
    def to_other_language(
        self,
        language: Language = Language.QISKIT,
        qiskit_parameters: Optional[set[Parameter]] = None,
    ):
        assert self.non_controlled_gate is not None
        return self.non_controlled_gate.to_other_language(language, qiskit_parameters)

    def to_matrix(self) -> npt.NDArray[np.complex64]:
        return np.array([[1, 0], [0, 1]], dtype=np.complex64)


@pytest.fixture
def gate_mock() -> MagicMock:
    return MagicMock(spec=Gate)


@pytest.fixture
def controlled_gate(gate_mock: MagicMock):
    controls = [0]
    targets = [1]
    return CustomControlledGate(controls, targets, non_controlled_gate=gate_mock)


def test_init(controlled_gate: ControlledGate, gate_mock: MagicMock):
    assert controlled_gate.controls == [0]
    assert controlled_gate.targets == [1]

    gate_mock.to_matrix.return_value = [[1, 0], [0, 1]]
    gate_mock.to_other_language.return_value = "Mock representation"

    assert np.array_equal(controlled_gate.to_matrix(), np.array([[1, 0], [0, 1]]))
    assert np.array_equal(controlled_gate.to_matrix(), np.array([[1, 0], [0, 1]]))
    assert controlled_gate.to_other_language() == "Mock representation"


@pytest.mark.parametrize(
    "controls, targets, label",
    [([0], [1], "CX"), ([1], [0], "X"), ([0, 1], [2, 3], "H")],
)
def test_controlled_gate_label(
    gate_mock: MagicMock, controls: list[int], targets: list[int], label: str
):
    controlled_gate = CustomControlledGate(
        controls, targets, non_controlled_gate=gate_mock, label=label
    )
    assert controlled_gate.label == label
