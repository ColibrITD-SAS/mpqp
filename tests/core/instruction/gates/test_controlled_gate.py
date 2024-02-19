from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest
from qiskit.circuit import Parameter

from mpqp.core.instruction.gates.controlled_gate import ControlledGate
from mpqp.core.instruction.gates.gate import Gate
from mpqp.core.languages import Language


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
