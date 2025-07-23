import numpy as np
import pytest

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
