"""Deterministic storage queries for distinct executions with equal samples."""

from pathlib import Path

import pytest

from mpqp import AWSDevice, BasisMeasure, IBMDevice, QCircuit
from mpqp.environment import env_manager
from mpqp.execution import Job, JobType, Result, Sample
from mpqp.local_storage.load import (
    get_jobs_with_result,
    get_results_with_result,
    get_results_with_result_and_job,
)
from mpqp.local_storage.queries import (
    fetch_jobs_with_result,
    fetch_results_with_id,
    fetch_results_with_result,
    fetch_results_with_result_and_job,
)
from mpqp.local_storage.save import insert_jobs, insert_results
from mpqp.local_storage.setup import setup_local_storage


@pytest.mark.parametrize(
    "order",
    [(0, 1, 2, 3), (2, 1, 3, 0), (1, 3, 0, 2)],
    ids=["repeat-last", "other-device-first", "different-counts-first"],
)
def test_queries_preserve_equal_sample_executions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, order: tuple[int, ...]
):
    # Isolate both the settings file and the environment variable it updates.
    monkeypatch.setattr(env_manager, "MPQP_ENV", tmp_path / "mpqp.env")
    monkeypatch.setenv("DB_PATH", "")
    setup_local_storage(str(tmp_path / "results.db"))

    circuit = QCircuit([BasisMeasure(shots=1024)], nb_qubits=2)
    job = Job(JobType.SAMPLE, circuit, IBMDevice.AER_SIMULATOR)
    other_job = Job(JobType.SAMPLE, circuit, AWSDevice.BRAKET_LOCAL_SIMULATOR)
    job_id = insert_jobs(job)[0]
    other_job_id = insert_jobs(other_job)[0]

    # Construct samples directly: no SDK, simulator or random seed is needed.
    results = [
        Result(
            execution_job,
            [
                Sample(2, index=0, count=zeros),
                Sample(2, index=3, count=1024 - zeros),
            ],
            shots=1024,
        )
        for execution_job, zeros in [
            (job, 512),
            (job, 511),
            (other_job, 512),
            (job, 512),
        ]
    ]
    result_ids: dict[int, int] = {}
    for index in order:
        result_ids[index] = insert_results(results[index])[0]

    # Repeated equal results share a job, but retain separate execution records.
    assert len(set(result_ids.values())) == 4
    assert [
        row["job_id"] for row in fetch_results_with_id([result_ids[0], result_ids[3]])
    ] == [job_id, job_id]
    assert job_id != other_job_id

    query = results[0]
    matching_ids = sorted([result_ids[0], result_ids[2], result_ids[3]])
    same_job_ids = sorted([result_ids[0], result_ids[3]])
    assert sorted(row["id"] for row in fetch_results_with_result(query)) == matching_ids
    assert (
        sorted(row["id"] for row in fetch_results_with_result_and_job(query))
        == same_job_ids
    )
    assert sorted(row["id"] for row in fetch_jobs_with_result(query)) == sorted(
        [job_id, other_job_id, job_id]
    )

    assert sorted(get_results_with_result(query), key=repr) == sorted(
        [results[0], results[2], results[3]], key=repr
    )
    assert sorted(get_results_with_result_and_job(query), key=repr) == sorted(
        [results[0], results[3]], key=repr
    )
    assert sorted(get_jobs_with_result(query), key=repr) == sorted(
        [job, other_job, job], key=repr
    )
