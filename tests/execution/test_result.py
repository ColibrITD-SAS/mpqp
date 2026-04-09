from __future__ import annotations

import numpy as np
import pytest
from numpy import pi

from mpqp import (
    CNOT,
    ATOSDevice,
    AWSDevice,
    BasisMeasure,
    ExpectationMeasure,
    GOOGLEDevice,
    H,
    IBMDevice,
    Job,
    JobType,
    Observable,
    QCircuit,
    Result,
    Rx,
    Sample,
    StateVector,
    run,
)
from mpqp.execution.devices import AvailableDevice


@pytest.mark.parametrize(
    "job_type, data",
    [
        (JobType.STATE_VECTOR, 0.4),
        (JobType.STATE_VECTOR, []),
        (JobType.OBSERVABLE, []),
        (JobType.OBSERVABLE, StateVector([1])),  # pyright: ignore[reportArgumentType]
        (JobType.SAMPLE, 0.4),
        (JobType.SAMPLE, StateVector([1])),  # pyright: ignore[reportArgumentType]
    ],
)
def test_result_wrong_type(job_type: JobType, data: float | StateVector | list[Sample]):
    c = QCircuit(3)
    c.add(Rx(pi / 2, 1))
    j = Job(job_type, c, IBMDevice.AER_SIMULATOR)
    with pytest.raises(TypeError):
        Result(j, data)


@pytest.mark.parametrize(
    "job_type, data",
    [
        (JobType.STATE_VECTOR, StateVector(np.ones(4, dtype=np.complex128) / 2)),
        (JobType.OBSERVABLE, 0.4),
        (
            JobType.SAMPLE,
            [Sample(3, index=3, count=250), Sample(3, index=6, count=250)],
        ),
    ],
)
def test_result_right_type(job_type: JobType, data: float | StateVector | list[Sample]):
    size = 3
    c = QCircuit(size)
    c.add(Rx(pi / 2, 1))
    c.add(BasisMeasure(list(range(size)))) if job_type == JobType.SAMPLE else None
    j = Job(job_type, c, IBMDevice.AER_SIMULATOR)
    if job_type == JobType.SAMPLE:
        assert isinstance(data, list)
        assert all(sample.count is not None for sample in data)
        shots = sum(sample.count for sample in data if sample.count is not None)
    else:
        shots = 0
    r = Result(j, data, shots=shots)
    assert r.device == r.job.device


@pytest.mark.parametrize(
    "result, expected_string",
    [
        (
            Result(
                Job(
                    JobType.STATE_VECTOR,
                    QCircuit(2),
                    IBMDevice.AER_SIMULATOR_STATEVECTOR,
                ),
                StateVector(np.ones(4, dtype=np.complex128) / 2),
            ),
            """Result: IBMDevice, AER_SIMULATOR_STATEVECTOR
  State vector: [0.5, 0.5, 0.5, 0.5]
  Probabilities: [0.25, 0.25, 0.25, 0.25]
  Number of qubits: 2""",
        ),
        (
            Result(
                Job(
                    JobType.SAMPLE,
                    QCircuit([BasisMeasure([0, 1])]),
                    IBMDevice.AER_SIMULATOR,
                ),
                [
                    Sample(2, index=0, count=135),
                    Sample(2, index=1, count=226),
                    Sample(2, index=2, count=8),
                    Sample(2, index=3, count=231),
                ],
                shots=600,
            ),
            """Result: IBMDevice, AER_SIMULATOR
  Counts: [135, 226, 8, 231]
  Probabilities: [0.225, 0.37667, 0.01333, 0.385]
  Samples:
    State: 00, Index: 0, Count: 135, Probability: 0.225
    State: 01, Index: 1, Count: 226, Probability: 0.3766667
    State: 10, Index: 2, Count: 8, Probability: 0.0133333
    State: 11, Index: 3, Count: 231, Probability: 0.385
  Error: None""",
        ),
        (
            Result(Job(JobType.OBSERVABLE, QCircuit(2), IBMDevice.AER_SIMULATOR), 0.65),
            """Result: IBMDevice, AER_SIMULATOR
  Expectation value: 0.65
  Error/Variance: None""",
        ),
    ],
)
def test_result_str(result: Result, expected_string: str):
    assert str(result) == expected_string


state_vector_devices_qiskit: list[AvailableDevice] = [
    device
    for device in IBMDevice
    if not device.is_remote() and device.supports_state_vector()
]

state_vector_devices_cirq: list[AvailableDevice] = [
    device
    for device in GOOGLEDevice
    if not device.is_remote() and device.supports_state_vector()
]

state_vector_devices_braket: list[AvailableDevice] = [
    device
    for device in AWSDevice
    if not device.is_remote() and device.supports_state_vector()
]

state_vector_devices_myqlm: list[AvailableDevice] = [
    device
    for device in ATOSDevice
    if not device.is_remote() and device.supports_state_vector()
]

sampling_devices_qiskit: list[AvailableDevice] = [
    device
    for device in IBMDevice
    if not device.is_remote() and device.supports_samples()
]

sampling_devices_cirq: list[AvailableDevice] = [
    device
    for device in GOOGLEDevice
    if not device.is_remote() and device.supports_samples()
]

sampling_devices_braket: list[AvailableDevice] = [
    device
    for device in AWSDevice
    if not device.is_remote() and device.supports_samples()
]

sampling_devices_myqlm: list[AvailableDevice] = [
    device
    for device in ATOSDevice
    if not device.is_remote() and device.supports_samples()
]


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("device", sampling_devices_qiskit)
def test_sample_nb_shot_handle_qiskit(device: AvailableDevice):
    exec_sample_nb_shot_handle(device)


@pytest.mark.provider("braket")
@pytest.mark.parametrize("device", sampling_devices_braket)
def test_sample_nb_shot_handle_braket(device: AvailableDevice):
    exec_sample_nb_shot_handle(device)


@pytest.mark.provider("cirq")
@pytest.mark.parametrize("device", sampling_devices_cirq)
def test_sample_nb_shot_handle_cirq(device: AvailableDevice):
    exec_sample_nb_shot_handle(device)


@pytest.mark.provider("myqlm")
@pytest.mark.parametrize("device", sampling_devices_myqlm)
def test_sample_nb_shot_handle_myqlm(device: AvailableDevice):
    exec_sample_nb_shot_handle(device)


def exec_sample_nb_shot_handle(device: AvailableDevice):
    circuit = QCircuit([H(0), CNOT(0, 1), BasisMeasure(shots=1024)])
    result = run(circuit, device)
    assert result.error != 0.0
    assert result.shots == 1024


@pytest.mark.provider("qiskit")
@pytest.mark.parametrize("device", state_vector_devices_qiskit)
def test_state_vector_nb_shot_handle_qiskit(device: AvailableDevice):
    exec_state_vector_nb_shot_handle(device)


@pytest.mark.provider("braket")
@pytest.mark.parametrize("device", state_vector_devices_braket)
def test_state_vector_nb_shot_handle_braket(device: AvailableDevice):
    exec_state_vector_nb_shot_handle(device)


@pytest.mark.provider("cirq")
@pytest.mark.parametrize("device", state_vector_devices_cirq)
def test_state_vector_nb_shot_handle_cirq(device: AvailableDevice):
    exec_state_vector_nb_shot_handle(device)


@pytest.mark.provider("myqlm")
@pytest.mark.parametrize("device", state_vector_devices_myqlm)
def test_state_vector_nb_shot_handle_myqlm(device: AvailableDevice):
    exec_state_vector_nb_shot_handle(device)


def exec_state_vector_nb_shot_handle(device: AvailableDevice):
    circuit = QCircuit(
        [
            H(0),
            CNOT(0, 1),
            ExpectationMeasure(
                Observable(
                    np.array(
                        [
                            [4, 2, 3, 8],
                            [2, -3, 1, 0],
                            [3, 1, -1, 5],
                            [8, 0, 5, 2],
                        ]
                    )
                ),
                shots=1024,
            ),
        ]
    )
    result = run(circuit, device)
    assert result.error != 0.0
    assert result.shots == 1024
