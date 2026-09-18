import pytest

from mpqp import AWSDevice, IBMDevice, QCircuit
from mpqp.execution import get_remote_result
from mpqp.execution.job import Job, JobType


def test_get_remote_result_rejects_job_without_id():
    job = Job(JobType.SAMPLE, QCircuit(1), IBMDevice.IBM_LEAST_BUSY)

    with pytest.raises(ValueError, match="id is None"):
        get_remote_result(job)


def test_get_remote_result_rejects_job_with_multiple_ids():
    job = Job(JobType.OBSERVABLE, QCircuit(1), AWSDevice.BRAKET_SV1_SIMULATOR)
    job.id = ["task_id_1", "task_id_2"]

    with pytest.raises(NotImplementedError, match="multiple ids"):
        get_remote_result(job)


def test_get_remote_result_rejects_local_job():
    job = Job(JobType.STATE_VECTOR, QCircuit(1), IBMDevice.AER_SIMULATOR)
    job.id = "local_task_id"

    with pytest.raises(ValueError, match="device of the job was local"):
        get_remote_result(job)
