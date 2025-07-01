import numpy as np
import pytest

from mpqp.core.instruction.gates.custom_controlled_gate import CustomControlledGate
from mpqp.gates import *


@pytest.mark.parametrize(
    "gate",
    [
        (CustomControlledGate([0, 1], CustomGate(np.array([[1, 0], [0, -1]]), 2))),
        (CustomControlledGate([0], CustomGate(np.array([[1, 0], [0, -1]]), 2))),
        (CustomControlledGate(0, Z(2))),
        (CustomControlledGate(1, Rz(np.pi, 2))),
        (CustomControlledGate(1, SWAP(0, 2))),
        (CustomControlledGate(3, TOF([0, 1], 2))),
    ],
)
def test_gate_repr(gate: CustomControlledGate) -> None:
    assert gate == eval(gate.__repr__())


@pytest.mark.parametrize(
    "gate, args",
    [
        (CNOT, (-1, 0)),
        (TOF, ([1, -1], 0)),
        (CustomControlledGate, (-10, Z(0))),
        (CustomControlledGate, ([0, 1, -3], S(2))),
    ],
)
def test_negative_indices(gate: Gate, args: tuple[Any]):
    with pytest.raises(ValueError):
        gate(*args)
