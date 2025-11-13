from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from cirq.ops.linear_combinations import PauliSum as CirqPauliSum
    from cirq.ops.pauli_string import PauliString as CirqPauliString

import numpy as np
import pytest

from mpqp import ExpectationMeasure, Language, Observable, pI, pX


@pytest.mark.parametrize(
    "targets",
    [[0, 1, 2], [1, 2, 3], list(range(4, 10))],
)
def test_expectation_measure_right_targets(targets: list[int]):
    obs = Observable(np.diag([1] * 2 ** len(targets)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ExpectationMeasure(obs, targets)


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
        measure = ExpectationMeasure(obs, targets)
    assert [set(swap.targets) for swap in measure.pre_measure] == expected_swaps


# TODO: complete this
@pytest.fixture
def list_to_cirq_pauli() -> (
    list[tuple[Observable, Union[CirqPauliSum, CirqPauliString]]]
):
    from cirq.devices.line_qubit import LineQubit
    from cirq.ops.identity import I as Cirq_I
    from cirq.ops.pauli_gates import X as Cirq_X

    a, b = LineQubit.range(2)

    return [
        (
            [
                (
                    Observable(pI @ pI + pI @ pX),
                    sum(1.0 * Cirq_I(a) * Cirq_I(b) + Cirq_X(b)),
                ),
            ],
        )
    ]


@pytest.mark.provider("cirq")
def test_to_other_language_cirq(
    list_to_cirq_pauli: list[tuple[Observable, Union[CirqPauliSum, CirqPauliString]]],
):
    for obs, translation in list_to_cirq_pauli:
        assert obs.to_other_language(Language.CIRQ) == translation
