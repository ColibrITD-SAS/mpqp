"""Add ``--long`` to the cli args to run free test but with account (disabled by default)
Add ``--long-cost`` to the cli args to run test with credits (disabled by default)"""

import sys
from unittest.mock import Mock

import numpy as np
import pytest
from qiskit.quantum_info import Statevector
from sympy import Symbol

from mpqp import (
    CNOT,
    SWAP,
    AZUREDevice,
    BasisMeasure,
    BatchResult,
    Gate,
    H,
    QCircuit,
    Ry,
    S,
    T,
    X,
    Y,
    run,
    submit,
)
from mpqp.execution.providers import azure as azure_provider


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("cached", [False, True])
def test_azure_submission_translates_and_binds_locally(
    monkeypatch: pytest.MonkeyPatch, cached: bool
):
    theta = Symbol("theta")
    device = AZUREDevice.IONQ_SIMULATOR
    circuit = QCircuit([Ry(theta, 0), BasisMeasure(shots=10)])
    if cached:
        circuit.transpiled_for_device(device)
    backend = Mock()
    backend.run.return_value.id.return_value = "mock-azure-job"
    provider = Mock()
    provider.get_backend.return_value = backend
    monkeypatch.setattr(azure_provider, "get_azure_provider", lambda: provider)

    job_id, _ = submit(circuit, device, values={theta: np.pi})

    assert job_id == "mock-azure-job"
    provider.get_backend.assert_called_once_with(device.value)
    backend.run.assert_called_once()
    translated = backend.run.call_args.args[0]
    assert translated.count_ops()["measure"] == 1
    assert backend.run.call_args.kwargs == {"shots": 10}
    assert Statevector(
        translated.remove_final_measurements(inplace=False)
    ).probabilities()[1] == pytest.approx(1)
    assert circuit.variables() == {theta}


sampling_devices_cost = [
    AZUREDevice.IONQ_SIMULATOR,
    AZUREDevice.IONQ_QPU_ARIA_1,
    AZUREDevice.IONQ_QPU_ARIA_2,
    AZUREDevice.IONQ_QPU,
    AZUREDevice.QUANTINUUM_SIM_H1_1,
    AZUREDevice.QUANTINUUM_SIM_H1_1E,
    AZUREDevice.QUANTINUUM_SIM_H1_1SC,
    AZUREDevice.RIGETTI_SIM_QPU_ANKAA_3,
    AZUREDevice.RIGETTI_SIM_QVM,
]

sampling_devices = [
    AZUREDevice.IONQ_SIMULATOR,
    # AZUREDevice.RIGETTI_SIM_QVM,
]


@pytest.mark.parametrize(
    "gates, basis_states",
    [
        (
            [
                H(0),
                CNOT(0, 1),
                CNOT(1, 2),
            ],
            ["000", "111"],
        ),
        (
            [H(0), H(2), CNOT(0, 1), Ry(1.87, 1), H(0), CNOT(2, 3), H(4)],
            [
                "00000",
                "00011",
                "00110",
                "00111",
                "01000",
                "01001",
                "01110",
                "01111",
                "10000",
                "10001",
                "10110",
                "10111",
                "11000",
                "11001",
                "11110",
                "11111",
            ],
        ),
        ([X(0), SWAP(0, 1), X(2), Y(0), CNOT(1, 2), S(0), T(1), H(2)], ["110", "111"]),
    ],
)
def azure_sample_cost(gates: list[Gate], basis_states: list[str]):
    # 3M-TODO check result with noise
    c = QCircuit(gates)
    c.add(BasisMeasure(list(range(c.nb_qubits)), shots=10))
    batch = run(c, sampling_devices_cost)
    assert isinstance(batch, BatchResult)


@pytest.mark.parametrize(
    "gates, basis_states",
    [
        (
            [
                H(0),
                CNOT(0, 1),
                CNOT(1, 2),
            ],
            ["000", "111"],
        ),
        (
            [H(0), H(2), CNOT(0, 1), Ry(1.87, 1), H(0), CNOT(2, 3), H(4)],
            [
                "00000",
                "00011",
                "00110",
                "00111",
                "01000",
                "01001",
                "01110",
                "01111",
                "10000",
                "10001",
                "10110",
                "10111",
                "11000",
                "11001",
                "11110",
                "11111",
            ],
        ),
        ([X(0), SWAP(0, 1), X(2), Y(0), CNOT(1, 2), S(0), T(1), H(2)], ["110", "111"]),
    ],
)
def azure_sample(gates: list[Gate], basis_states: list[str]):
    # 3M-TODO check result with noise
    c = QCircuit(gates)
    c.add(BasisMeasure(list(range(c.nb_qubits)), shots=10))
    batch = run(c, sampling_devices)
    assert isinstance(batch, BatchResult)


if "--long" in sys.argv:
    test_running_remote_azure_sample = azure_sample

if "--long-cost" in sys.argv:
    test_running_remote_azure_sample_cost = azure_sample_cost
