import sys

import numpy as np
import pytest

from mpqp import (
    CNOT,
    AWSDevice,
    BasisMeasure,
    ExpectationMeasure,
    H,
    Observable,
    QCircuit,
    run,
)
from mpqp.execution.providers.aws import run_braket, submit_job_braket
from mpqp.execution.runner import generate_job
from mpqp.measures import pX, pZ


@pytest.mark.provider("braket")
@pytest.mark.parametrize(
    "circuit",
    [
        QCircuit([H(2)]),
        QCircuit([H(0), H(2)]),
        QCircuit([H(0), CNOT(1, 3), H(2)]),
        QCircuit([H(2)], nb_qubits=5),
    ],
)
def test_braket_non_contiguous_qubits(circuit: QCircuit):
    run(circuit, AWSDevice.BRAKET_LOCAL_SIMULATOR)


@pytest.mark.provider("braket")
def test_submit_job_braket_binds_task_id():
    job = generate_job(QCircuit([H(0)]), AWSDevice.BRAKET_LOCAL_SIMULATOR)

    job_id, task = submit_job_braket(job)

    assert job_id == task.id
    assert job.id == task.id


@pytest.mark.provider("braket")
def test_run_braket_preserves_task_id_in_result():
    job = generate_job(QCircuit([H(0)]), AWSDevice.BRAKET_LOCAL_SIMULATOR)

    result = run_braket(job)

    assert result.job is job
    assert result.job.id == job.id
    assert isinstance(result.job.id, str)


@pytest.mark.provider("braket")
@pytest.mark.parametrize(
    "observables, optimize_measurement",
    [
        (
            [
                Observable(np.array([[1, 0.2], [0.2, -1]], dtype=np.complex128)),
                Observable(np.array([[0.5, 0.3j], [-0.3j, -0.5]], dtype=np.complex128)),
            ],
            False,
        ),
        ([Observable(pX), Observable(pZ)], True),
    ],
    ids=["hermitian", "pauli-grouping"],
)
def test_run_braket_observables_bind_multiple_task_ids(
    observables: list[Observable], optimize_measurement: bool
):
    circuit = QCircuit(
        [
            H(0),
            ExpectationMeasure(
                observables,
                shots=0,
                optimize_measurement=optimize_measurement,
            ),
        ]
    )

    result = run(circuit, AWSDevice.BRAKET_LOCAL_SIMULATOR)

    assert isinstance(result.job.id, list)
    assert len(result.job.id) == len(observables)


@pytest.mark.provider("braket")
def remote_braket_execution_binds_task_arn():
    circuit = QCircuit([H(0), BasisMeasure([0], shots=1)])

    result = run(circuit, AWSDevice.BRAKET_SV1_SIMULATOR)

    assert isinstance(result.job.id, str)
    assert result.job.id.startswith("arn:aws:braket:")


if "--long-costly" in sys.argv:
    test_remote_braket_execution_binds_task_arn = remote_braket_execution_binds_task_arn
