import sys

import numpy as np
import pytest
from mpqp import BatchResult, H, IBMDevice, Job, JobType, QCircuit, Result
from mpqp.execution.providers.ibm import extract_result, submit_remote_ibm
from mpqp.execution.runner import generate_job
from mpqp.gates import *
from mpqp.measures import *

from qiskit import QuantumCircuit, transpile
from qiskit.primitives import PubResult
from qiskit.primitives.containers import DataBin
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2, SamplerV2


@pytest.mark.provider("qiskit")
def remote_ibm_execution_binds_job_id():
    circuit = QCircuit([H(0), BasisMeasure([0], shots=1)])
    job = generate_job(circuit, IBMDevice.IBM_LEAST_BUSY)

    job_id, ibm_job = submit_remote_ibm(job)

    assert job_id == ibm_job.job_id()
    assert job.id == ibm_job.job_id()

    assert isinstance(job.device, IBMDevice)
    result = extract_result(ibm_job.result(), job, job.device)

    assert result.job.id == job_id


if "--long-costly" in sys.argv:
    test_remote_ibm_execution_binds_job_id = remote_ibm_execution_binds_job_id


@pytest.mark.provider("qiskit")
def test_extract_result_accepts_estimator_primitive_result():
    """Remote jobs (``RuntimeJobV2.result()``) return a ``PrimitiveResult``,
    with or without an MPQP job attached.
    This test produces locally a PrimitiveResult and pass to extract result
    to test that the extraction is happening correctly."""

    qc = QuantumCircuit(1)
    qc.x(0)
    backend = AerSimulator()
    primitive_result = (
        EstimatorV2(mode=backend)
        .run([(transpile(qc, backend), [SparsePauliOp("Z")])])
        .result()
    )
    job = Job(
        JobType.OBSERVABLE,
        QCircuit([X(0), ExpectationMeasure(Observable(pZ), shots=1000)]),
        IBMDevice.AER_SIMULATOR,
    )

    for attached_job in (job, None):  # None: path of get_result_from_ibm_job_id
        result = extract_result(primitive_result, attached_job, IBMDevice.AER_SIMULATOR)
        assert isinstance(result, Result)
        assert result.expectation_values == pytest.approx(-1.0)


@pytest.mark.provider("qiskit")
def test_extract_result_accepts_sampler_primitive_result():

    qc = QuantumCircuit(1, 1)  # classical register named "c", as produced by MPQP
    qc.x(0)
    qc.measure(0, 0)
    backend = AerSimulator()
    primitive_result = (
        SamplerV2(mode=backend).run([transpile(qc, backend)], shots=100).result()
    )
    job = Job(
        JobType.SAMPLE,
        QCircuit([X(0), BasisMeasure([0], shots=100)]),
        IBMDevice.AER_SIMULATOR,
    )
    result = extract_result(primitive_result, job, IBMDevice.AER_SIMULATOR)
    assert isinstance(result, Result)
    assert result.counts == [0, 100]


@pytest.mark.provider("qiskit")
def test_extract_result_accepts_1d_pub_result_with_multiple_measurements():
    job = Job(
        JobType.OBSERVABLE,
        QCircuit(
            [
                X(0),
                ExpectationMeasure(Observable(pZ), shots=1000),
                ExpectationMeasure(Observable(pZ), shots=1000),
            ]
        ),
        IBMDevice.AER_SIMULATOR,
    )
    pub_result = PubResult(
        DataBin(
            evs=np.array([-1.0, -1.0]),
            stds=np.array([0.1, 0.2]),
            shape=(2,),
        ),
        metadata={"shots": 1000},
    )

    result = extract_result(pub_result, job, IBMDevice.AER_SIMULATOR)

    assert isinstance(result, BatchResult)
    assert [item.error for item in result.results] == [0.1, 0.2]
