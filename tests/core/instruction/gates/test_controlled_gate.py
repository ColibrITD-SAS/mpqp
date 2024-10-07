from typing import Optional, TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from qiskit.circuit import Parameter

from mpqp.core.instruction.gates.controlled_gate import ControlledGate
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
        qiskit_parameters: Optional[set["Parameter"]] = None,
    ):
        assert self.non_controlled_gate is not None
        return self.non_controlled_gate.to_other_language(language, qiskit_parameters)

    def to_matrix(self, desired_gate_size: int = 0):
        return np.array([[1, 0], [0, 1]], dtype=np.complex64)
