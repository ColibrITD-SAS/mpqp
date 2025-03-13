import numpy as np
import pytest
from mpqp import QCircuit
from mpqp.core.instruction import ExpectationMeasure, Observable
from mpqp.execution import AvailableDevice, run, IBMDevice, Result, BatchResult
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
        run(
            circuit
            + QCircuit([ExpectationMeasure(obs, shots=0)], nb_qubits=circuit.nb_qubits),
            device,
        )
        for obs in observables
    ]

    multi_result = run(
        circuit
        + QCircuit(
            [ExpectationMeasure(observables, shots=0)], nb_qubits=circuit.nb_qubits
        ),
        device,
    )

    assert isinstance(multi_result, BatchResult)
    assert len(seq_results) == len(multi_result.results)

    for r1, r2 in zip(seq_results, multi_result.results):
        assert isinstance(r1, Result)
        assert isinstance(r2, Result)
        assert r1.expectation_value == r2.expectation_value
