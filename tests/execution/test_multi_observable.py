import numpy as np
import pytest
from mpqp import QCircuit
from mpqp.core.instruction import ExpectationMeasure, Observable
from mpqp.execution import AvailableDevice, IBMDevice
from mpqp.execution.runner import _run_single  # pyright: ignore[reportPrivateUsage]
from mpqp.gates import *


def list_circuits():
    return [
        QCircuit([H(0), CNOT(0, 1)]),
        QCircuit([H(0), X(1)]),
        # TODO add random circuit
    ]


def list_observables():
    return [
        [
            Observable(np.ones((4, 4)).astype(np.complex64)),
            Observable(np.diag([1, 2, -3, 4])),
            # TODO add random observable ?
        ]
    ]


def list_devices():
    return [IBMDevice.AER_SIMULATOR]


@pytest.mark.parametrize(
    "circuit, observables, device",
    [
        (i, j, k)
        for i in list_circuits()
        for j in list_observables()
        for k in list_devices()
    ],
)
def test_sequential_versus_multi(
    circuit: QCircuit, observables: list[Observable], device: AvailableDevice
):
    seq_results = [
        _run_single(
            circuit
            + QCircuit([ExpectationMeasure(obs, shots=0)], nb_qubits=circuit.nb_qubits),
            device,
            {},
        )
        for obs in observables
    ]

    multi_result = _run_single(
        circuit
        + QCircuit(
            [ExpectationMeasure(observables, shots=0)], nb_qubits=circuit.nb_qubits
        ),
        device,
        {},
    )
    assert isinstance(multi_result.expectation_values, dict)
    assert len(seq_results) == len(multi_result.expectation_values)

    # TODO modify here to match the logic of dict and observable.label etc
    for r1, e2 in zip(seq_results, multi_result.expectation_values.values()):
        assert r1.expectation_values == e2
