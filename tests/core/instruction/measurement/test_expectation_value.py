import warnings
import numpy as np
import pytest

from mpqp.measures import Observable, ExpectationMeasure


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
