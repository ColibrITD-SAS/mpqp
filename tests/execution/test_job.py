import pytest

from mpqp import AWSDevice, Job, JobStatus, JobType, QCircuit

JOB_STATUSES = (
    JobStatus.INIT,
    JobStatus.QUEUED,
    JobStatus.RUNNING,
    JobStatus.CANCELLED,
    JobStatus.ERROR,
    JobStatus.DONE,
)
STATUS_PRIORITY = {
    JobStatus.ERROR: 0,
    JobStatus.CANCELLED: 1,
    JobStatus.RUNNING: 2,
    JobStatus.QUEUED: 3,
    JobStatus.INIT: 4,
    JobStatus.DONE: 5,
}


@pytest.mark.parametrize("job_ids", [None, []], ids=["none", "empty-list"])
def test_status_without_remote_job_ids(job_ids: None | list[str]):
    job = Job(
        JobType.STATE_VECTOR,
        QCircuit(1),
        AWSDevice.BRAKET_SV1_SIMULATOR,
    )
    job.id = job_ids

    assert job.status == JobStatus.INIT


@pytest.mark.parametrize(
    ("first_status", "second_status", "expected_status"),
    [
        pytest.param(
            first_status,
            second_status,
            min(
                (first_status, second_status),
                key=lambda status: STATUS_PRIORITY[status],
            ),
            id=f"{first_status.name.lower()}-{second_status.name.lower()}",
        )
        for first_status in JOB_STATUSES
        for second_status in JOB_STATUSES
    ],
)
def test_status_with_multiple_remote_job_ids(
    monkeypatch: pytest.MonkeyPatch,
    first_status: JobStatus,
    second_status: JobStatus,
    expected_status: JobStatus,
):
    status_by_id = {
        "task_id_1": first_status,
        "task_id_2": second_status,
    }
    monkeypatch.setattr(
        "mpqp.execution.job.get_aws_job_status",
        lambda job_id: status_by_id[job_id],
    )
    job = Job(
        JobType.STATE_VECTOR,
        QCircuit(1),
        AWSDevice.BRAKET_SV1_SIMULATOR,
    )
    job.id = list(status_by_id)

    assert job.status == expected_status
