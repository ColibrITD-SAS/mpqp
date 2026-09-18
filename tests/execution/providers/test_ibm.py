import sys

import pytest

from mpqp import BasisMeasure, H, IBMDevice, QCircuit
from mpqp.execution.providers.ibm import extract_result, submit_remote_ibm
from mpqp.execution.runner import generate_job


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
