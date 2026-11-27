from pathlib import Path

import pytest

from mpqp import AWSDevice, BasisMeasure, Job, JobStatus, JobType, QCircuit
from mpqp.environment import env_manager

JOB_STATUSES = tuple(JobStatus)
STATUS_PRIORITY = (
    JobStatus.ERROR,
    JobStatus.CANCELLED,
    JobStatus.RUNNING,
    JobStatus.QUEUED,
    JobStatus.INIT,
    JobStatus.DONE,
)


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
                key=STATUS_PRIORITY.index,
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


@pytest.mark.provider("braket")
def test_aws_status_uses_selected_profile_and_task_region(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from braket.aws import AwsSession

    credentials_file = tmp_path / "credentials"
    credentials_file.write_text(
        "[default]\n"
        "aws_access_key_id = default-key\n"
        "aws_secret_access_key = default-secret\n"
        "[team-profile]\n"
        "aws_access_key_id = team-key\n"
        "aws_secret_access_key = team-secret\n",
        encoding="utf-8",
    )
    config_file = tmp_path / "config"
    config_file.write_text(
        "[default]\nregion = us-east-1\n"
        "[profile team-profile]\nregion = us-east-1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(env_manager, "MPQP_ENV", tmp_path / ".env")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials_file))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config_file))
    monkeypatch.setenv("AWS_PROFILE", "default")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("BRAKET_PROFILE", "team-profile")
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.delenv(key, raising=False)

    task_arn = "arn:aws:braket:us-west-1:123456789012:quantum-task/test-task"

    def get_quantum_task(session: AwsSession, arn: str) -> dict[str, str]:
        assert arn == task_arn
        assert session.boto_session.profile_name == "team-profile"
        assert session.region == "us-west-1"
        credentials = session.boto_session.get_credentials()
        assert credentials is not None
        assert credentials.access_key == "team-key"
        return {"quantumTaskArn": arn, "status": "QUEUED"}

    monkeypatch.setattr(AwsSession, "get_quantum_task", get_quantum_task)
    job = Job(
        JobType.SAMPLE,
        QCircuit([BasisMeasure(shots=10)]),
        AWSDevice.BRAKET_SV1_SIMULATOR,
    )
    job.id = task_arn

    assert job.status == JobStatus.QUEUED
