"""Add ``--long`` to the CLI arguments to run the remote tests (disabled by
default because they require a configured Quantinuum Nexus account and project).
"""

import sys

import numpy as np
import pytest

from mpqp import (
    CNOT,
    BasisMeasure,
    ExpectationMeasure,
    H,
    IBMDevice,
    Observable,
    QCircuit,
    QUANTINUUMDevice,
    X,
    get_remote_result,
    pX,
    pZ,
    run,
    submit,
)
from mpqp.core.instruction.measurement.pauli_string import CommutingTypes
from mpqp.execution.job import Job, JobType
from mpqp.execution.providers.quantinuum import check_job_compatibility
from mpqp.tools.errors import DeviceJobIncompatibleError
from mpqp.tools.maths import matrix_eq

pytestmark = pytest.mark.provider("quantinuum")


@pytest.mark.parametrize(
    "device",
    [
        QUANTINUUMDevice.TKET_AER_SIMULATOR,
        QUANTINUUMDevice.TKET_QULACS_SIMULATOR,
    ],
)
def test_local_tket_sample(device: QUANTINUUMDevice):
    result = run(
        QCircuit([X(0), BasisMeasure([0], shots=100)]),
        device,
    )

    assert result.counts == [0, 100]
    assert result.probabilities.tolist() == [0, 1]
    assert result.shots == 100


@pytest.mark.parametrize(
    "device",
    [
        QUANTINUUMDevice.TKET_AER_STATEVECTOR_SIMULATOR,
        QUANTINUUMDevice.TKET_QULACS_SIMULATOR,
    ],
)
def test_local_tket_state_vector(device: QUANTINUUMDevice):
    result = run(
        QCircuit([H(0), CNOT(0, 1)]),
        device,
    )

    expected = np.array([2**-0.5, 0, 0, 2**-0.5])
    assert matrix_eq(result.amplitudes, expected)
    assert matrix_eq(result.probabilities, np.abs(expected) ** 2)
    assert result.shots == 0


@pytest.mark.parametrize(
    ("job", "message"),
    [
        (
            Job(JobType.STATE_VECTOR, QCircuit(1), IBMDevice.AER_SIMULATOR),
            "must correspond to a `QUANTINUUMDevice`",
        ),
        (
            Job(
                JobType.SAMPLE,
                QCircuit([ExpectationMeasure(Observable(pZ), shots=100)]),
                QUANTINUUMDevice.TKET_AER_SIMULATOR,
            ),
            "SAMPLE job requires",
        ),
        (
            Job(
                JobType.SAMPLE,
                QCircuit([BasisMeasure([0], shots=100)]),
                QUANTINUUMDevice.TKET_AER_STATEVECTOR_SIMULATOR,
            ),
            "does not support `SAMPLE` jobs",
        ),
        (
            Job(
                JobType.STATE_VECTOR,
                QCircuit(1),
                QUANTINUUMDevice.TKET_AER_SIMULATOR,
            ),
            "does not support `STATE_VECTOR` jobs",
        ),
        (
            Job(
                JobType.OBSERVABLE,
                QCircuit([ExpectationMeasure(Observable(pX), shots=0)]),
                QUANTINUUMDevice.H1_EMULATOR,
            ),
            "positive number of shots",
        ),
        (
            Job(
                JobType.OBSERVABLE,
                QCircuit([ExpectationMeasure(Observable(pX), shots=100)]),
                QUANTINUUMDevice.TKET_AER_STATEVECTOR_SIMULATOR,
            ),
            "does not support sampled observable jobs",
        ),
        (
            Job(
                JobType.OBSERVABLE,
                QCircuit(
                    [
                        ExpectationMeasure(
                            Observable(pX),
                            shots=100,
                            commuting_type=CommutingTypes.FULL,
                        )
                    ]
                ),
                QUANTINUUMDevice.H1_EMULATOR,
            ),
            "qubit-wise commuting",
        ),
    ],
)
def test_job_compatibility(job: Job, message: str):
    with pytest.raises(DeviceJobIncompatibleError, match=message):
        check_job_compatibility(job)


@pytest.mark.parametrize(
    "device",
    [
        QUANTINUUMDevice.TKET_AER_SIMULATOR,
        QUANTINUUMDevice.TKET_AER_STATEVECTOR_SIMULATOR,
        QUANTINUUMDevice.TKET_QULACS_SIMULATOR,
    ],
)
def test_local_tket_exact_observables(device: QUANTINUUMDevice):
    circuit = QCircuit(
        [
            H(0),
            ExpectationMeasure(
                [
                    Observable(pX, label="X"),
                    Observable(np.diag([1, -1]), label="Z"),
                ],
                shots=0,
            ),
        ]
    )

    result = run(circuit, device)

    assert isinstance(result.expectation_values, dict)
    assert np.isclose(result.expectation_values["X"], 1)
    assert np.isclose(result.expectation_values["Z"], 0)
    assert result.error == {"X": 0, "Z": 0}
    assert result.shots == 0


@pytest.mark.parametrize(
    "device",
    [
        QUANTINUUMDevice.TKET_AER_SIMULATOR,
        QUANTINUUMDevice.TKET_QULACS_SIMULATOR,
    ],
)
def test_local_tket_sampled_observables(device: QUANTINUUMDevice):
    shots = 1000
    circuit = QCircuit(
        [
            H(0),
            ExpectationMeasure(
                Observable(pX + pZ),
                shots=shots,
            ),
        ]
    )

    result = run(circuit, device)

    assert isinstance(result.expectation_values, float)
    assert isinstance(result.error, float)
    assert np.isclose(result.expectation_values, 1, atol=0.15)

    sampled_z_expectation = result.expectation_values - 1
    expected_variance = (1 - sampled_z_expectation**2) / shots

    assert np.isclose(result.error, expected_variance)
    assert result.shots == shots


@pytest.mark.skipif(
    "--long" not in sys.argv,
    reason="requires a configured Quantinuum Nexus account and project",
)
def test_remote_quantinuum_jobs():
    sample_result = run(
        QCircuit([X(0), BasisMeasure([0], shots=100)]),
        QUANTINUUMDevice.NEXUS_AER_SIMULATOR,
    )
    assert sample_result.counts == [0, 100]

    state_vector_result = run(
        QCircuit([H(0), CNOT(0, 1)]),
        QUANTINUUMDevice.NEXUS_AER_STATEVECTOR_SIMULATOR,
    )
    assert matrix_eq(
        state_vector_result.amplitudes,
        np.array([2**-0.5, 0, 0, 2**-0.5]),
    )

    exact_circuit = QCircuit(
        [
            H(0),
            ExpectationMeasure(Observable(pX, label="X"), shots=0),
        ]
    )
    _, exact_job = submit(
        exact_circuit,
        QUANTINUUMDevice.NEXUS_AER_STATEVECTOR_SIMULATOR,
    )
    exact_result = get_remote_result(exact_job)
    assert isinstance(exact_result.expectation_values, float)
    assert np.isclose(exact_result.expectation_values, 1)

    sampled_circuit = QCircuit(
        [
            H(0),
            ExpectationMeasure(Observable(pX + pZ), shots=100),
        ]
    )
    _, sampled_job = submit(
        sampled_circuit,
        QUANTINUUMDevice.NEXUS_AER_SIMULATOR,
    )
    sampled_result = get_remote_result(sampled_job)
    assert isinstance(sampled_result.expectation_values, float)
    assert np.isclose(sampled_result.expectation_values, 1, atol=0.3)
