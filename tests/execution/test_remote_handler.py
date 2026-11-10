from unittest.mock import Mock

import pytest

from mpqp import (
    ATOSDevice,
    AWSDevice,
    AZUREDevice,
    GOOGLEDevice,
    IBMDevice,
    QCircuit,
    QUANTINUUMDevice,
)
from mpqp.execution import get_remote_result, remote_handler
from mpqp.execution.devices import AvailableDevice
from mpqp.execution.job import Job, JobType


def test_job_without_id():
    job = Job(JobType.SAMPLE, QCircuit(1), IBMDevice.IBM_LEAST_BUSY)

    with pytest.raises(ValueError, match="id is None"):
        get_remote_result(job)


def test_job_with_multiple_ids():
    job = Job(JobType.OBSERVABLE, QCircuit(1), AWSDevice.BRAKET_SV1_SIMULATOR)
    job.id = ["task_id_1", "task_id_2"]

    with pytest.raises(NotImplementedError, match="multiple ids"):
        get_remote_result(job)


def test_local_job():
    job = Job(JobType.STATE_VECTOR, QCircuit(1), IBMDevice.AER_SIMULATOR)
    job.id = "local_task_id"

    with pytest.raises(ValueError, match="device of the job was local"):
        get_remote_result(job)


def test_id_without_device():
    with pytest.raises(ValueError, match="also provide the device"):
        get_remote_result("remote_job_id")


def test_unsupported_device():
    with pytest.raises(NotImplementedError, match="not supported for remote features"):
        get_remote_result("remote_job_id", GOOGLEDevice.IONQ_SIMULATOR)


@pytest.mark.parametrize(
    ("device", "result_handler", "passes_job"),
    [
        (IBMDevice.IBM_LEAST_BUSY, "get_result_from_ibm_job_id", False),
        (ATOSDevice.QLM_LINALG, "get_result_from_qlm_job_id", False),
        (AWSDevice.BRAKET_SV1_SIMULATOR, "get_result_from_aws_task_arn", True),
        (AZUREDevice.IONQ_SIMULATOR, "get_result_from_azure_job_id", False),
        (
            QUANTINUUMDevice.NEXUS_AER_SIMULATOR,
            "get_result_from_quantinuum_job_id",
            True,
        ),
    ],
    ids=["ibm", "aws", "atos", "azure", "quantinuum"],
)
@pytest.mark.parametrize("from_job", [False, True], ids=["job-id", "job"])
def test_supported_provider(
    monkeypatch: pytest.MonkeyPatch,
    device: AvailableDevice,
    result_handler: str,
    passes_job: bool,
    from_job: bool,
):
    expected_result = Mock()
    get_provider_result = Mock(return_value=expected_result)
    monkeypatch.setattr(remote_handler, result_handler, get_provider_result)

    job_id = "remote_job_id"
    if from_job:
        job = Job(JobType.SAMPLE, QCircuit(1), device)
        job.id = job_id
        result = get_remote_result(job)
    else:
        job = None
        result = get_remote_result(job_id, device)

    assert result is expected_result
    if passes_job:
        get_provider_result.assert_called_once_with(job_id, job=job)
    else:
        get_provider_result.assert_called_once_with(job_id)
