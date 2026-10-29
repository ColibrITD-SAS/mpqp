import pytest

from mpqp import AWSDevice, Job, JobStatus, JobType, QCircuit


@pytest.mark.parametrize(
    ("current_status", "expected_status"),
    [
        (JobStatus.DONE, JobStatus.DONE),
        (JobStatus.ERROR, JobStatus.ERROR),
        (JobStatus.CANCELLED, JobStatus.CANCELLED),
        (JobStatus.RUNNING, JobStatus.RUNNING),
        (JobStatus.QUEUED, JobStatus.QUEUED),
    ],
)
def test_status_with_multiple_remote_job_ids(
    monkeypatch: pytest.MonkeyPatch,
    current_status: JobStatus,
    expected_status: JobStatus,
):
    status_by_id = {
        "task_id_1": JobStatus.DONE,
        "task_id_2": current_status,
    }
    monkeypatch.setattr(
        "mpqp.execution.job.get_aws_job_status",
        status_by_id.__getitem__,
    )
    job = Job(
        JobType.STATE_VECTOR,
        QCircuit(1),
        AWSDevice.BRAKET_SV1_SIMULATOR,
    )
    job.id = list(status_by_id)

    assert job.status == expected_status
