import numpy as np
import pytest

from mpqp import QCircuit
from mpqp.gates import H, Rx
from mpqp.measures import ExpectationMeasure, Observable
from mpqp.execution import adjust_measure
from mpqp.tools.maths import matrix_eq


@pytest.mark.parametrize(
    "measure_targets, circuit, nb_ids_before, nb_ids_after",
    [
        ([0, 1], QCircuit([H(0), Rx(1.76, 2)]), 0, 1),
        ([1, 2], QCircuit([H(0), Rx(1.76, 2)]), 1, 0),
        ([1, 2], QCircuit([H(0), Rx(1.76, 3)]), 1, 1),
    ],
)
def test_adjust_measure(
    measure_targets: list[int],
    circuit: QCircuit,
    nb_ids_before: int,
    nb_ids_after: int,
):
    obs_matrix = np.array(
        [
            [0.63, 0.5, 1, 1],
            [0.5, 0.82, 1, 1],
            [1, 1, 1, 0.33],
            [1, 1, 0.33, 0.3],
        ],
    )
    measure = ExpectationMeasure(Observable(obs_matrix), measure_targets)
    adjusted_observable_matrix = np.kron(
        np.kron(
            np.eye(2**nb_ids_before, dtype=np.complex64), measure.observable.matrix
        ),
        np.eye(2**nb_ids_after),
    )
    assert matrix_eq(
        adjust_measure(measure, circuit).observable.matrix, adjusted_observable_matrix
    )
