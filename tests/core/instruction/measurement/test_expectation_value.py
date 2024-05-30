from __future__ import annotations

import warnings

import numpy as np
import pytest
from braket.circuits.observables import Hermitian
from cirq.circuits.circuit import Circuit
from cirq.devices.line_qubit import LineQubit
from cirq.ops.identity import I as Cirq_I
from cirq.ops.linear_combinations import PauliSum as CirqPauliSum
from cirq.ops.pauli_gates import X as Cirq_X
from cirq.ops.pauli_string import PauliString as CirqPauliString
from qat.core.wrappers.observable import Observable as QLMObservable
from qiskit.quantum_info import Operator

from mpqp.core.instruction.measurement.pauli_string import I, X
from mpqp.core.languages import Language
from mpqp.measures import ExpectationMeasure, Observable

q = LineQubit.range(3)
c = Circuit()
for q_ in q:
    c.append(Cirq_X(q_))


@pytest.mark.parametrize(
    "targets",
    [[0, 1, 2], [1, 2, 3], list(range(4, 10))],
)
def test_expectation_measure_right_targets(targets: list[int]):
    obs = Observable(np.diag([1] * 2 ** len(targets)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ExpectationMeasure(targets, obs)


@pytest.mark.parametrize(
    "targets, expected_swaps",
    [
        ([1, 3, 4], [{2, 3}, {3, 4}]),
        ([1, 0, 2], [{1, 0}]),
        ([2, 0, 3], [{0, 2}, {1, 2}, {2, 3}]),
    ],
)
def test_expectation_measure_wrong_targets(
    targets: list[int], expected_swaps: list[tuple[int, int]]
):
    obs = Observable(np.diag([1] * 2 ** len(targets)))
    with pytest.warns(UserWarning):
        measure = ExpectationMeasure(targets, obs)
    assert [
        set(swap.targets) for swap in measure.pre_measure.instructions
    ] == expected_swaps


# TODO: complete this
@pytest.mark.parametrize(
    "obs, translation",
    [
        (
            Observable(I @ I + I @ X),
            sum(1.0 * Cirq_I(q[0]) * Cirq_I(q[1]) + Cirq_X(q[1])),
        ),
    ],
)
def test_to_other_language(
    obs: Observable,
    translation: Operator | QLMObservable | Hermitian | CirqPauliSum | CirqPauliString,
):
    assert obs.to_other_language(Language.CIRQ, c) == translation
