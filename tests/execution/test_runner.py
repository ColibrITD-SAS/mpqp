import numpy as np
import pytest

from mpqp import ExpectationMeasure, H, Observable, QCircuit, Rx, pI, pX, pY, pZ
from mpqp.core.instruction.measurement import PauliString
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
            np.eye(2**nb_ids_before, dtype=np.complex128), measure.observables[0].matrix
        ),
        np.eye(2**nb_ids_after),
    )
    assert matrix_eq(
        adjust_measure(measure, circuit).observables[0].matrix,
        adjusted_observable_matrix,
    )


@pytest.mark.parametrize(
    ("observable", "measure_targets", "circuit_size", "expected_observable"),
    [
        (pX @ pZ, [0, 2], 3, pX @ pI @ pZ),  # ordered, non-contiguous
        (pX @ pZ, [2, 0], 3, pZ @ pI @ pX),  # unordered, non-contiguous
        (
            pX @ pY @ pZ,
            [1, 2, 0],
            4,
            pZ @ pX @ pY @ pI,
        ),  # unordered targets, multiple positions reordered
    ],
)
def test_adjust_measure_target_order(
    observable: PauliString,
    measure_targets: list[int],
    circuit_size: int,
    expected_observable: PauliString,
):
    measure = ExpectationMeasure(Observable(observable), measure_targets)

    adjusted_measure = adjust_measure(measure, QCircuit(circuit_size))

    assert adjusted_measure.targets == list(range(circuit_size))
    assert matrix_eq(
        adjusted_measure.observables[0].matrix,
        expected_observable.to_matrix(),
    )


def test_adjust_measure_matrix_reordering():
    observable = Observable((pX @ pY @ pZ).to_matrix())
    measure = ExpectationMeasure(
        observable,
        targets=[1, 2, 0],
        optimize_measurement=False,
    )
    original_matrix = observable.matrix

    adjusted_measure = adjust_measure(measure, QCircuit(3))

    assert matrix_eq(
        adjusted_measure.observables[0].matrix,
        (pZ @ pX @ pY).to_matrix(),
    )
    assert matrix_eq(measure.observables[0].matrix, original_matrix)


def test_adjust_measure_targets_mismatch():
    measure = ExpectationMeasure(Observable(pX), targets=[0, 1])

    with pytest.raises(ValueError, match="Each observable must act on 2 qubits"):
        adjust_measure(measure, QCircuit(2))
