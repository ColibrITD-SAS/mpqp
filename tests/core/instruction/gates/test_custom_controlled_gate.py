from typing import Any, Type
import numpy as np
import pytest
from numpy import array  # pyright: ignore[reportUnusedImport]
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
