"""Provides functions to insert :class:`~mpqp.execution.job.Job` and
:class:`~mpqp.execution.result.Result` into the local database.
"""

# TODO: put DB specific errors here ?
# TODO: document the raised errors

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable

from mpqp.core.circuit import QCircuit
from mpqp.environment.env_manager import get_env_variable
from mpqp.execution import BatchResult, Job, Result
from mpqp.local_storage.queries import fetch_jobs_with_job
from mpqp.local_storage.setup import ensure_local_storage


class _InputCircuitSerializer:
    """Serialize shared inputs once per save, releasing them after their last use."""

    def __init__(self, remaining: dict[int, int]):
        self._remaining = remaining
        self._memo: dict[int, tuple[QCircuit, str]] = {}

    @classmethod
    def for_results(cls, results: Iterable[Result]) -> _InputCircuitSerializer | None:
        counts = Counter(
            id(result.input_circuit)
            for result in results
            if result.input_circuit is not None
        )
        remaining = {key: count for key, count in counts.items() if count > 1}
        return cls(remaining) if remaining else None

    def __call__(self, source: QCircuit) -> str:
        key = id(source)
        count = self._remaining.get(key)
        if count is None:
            return json.dumps(repr(source))
        remaining = count - 1
        cached = self._memo.get(key)
        encoded = json.dumps(repr(source)) if cached is None else cached[1]
        if remaining > 0:
            self._remaining[key] = remaining
            # Keep the source alive against id reuse while its text is cached.
            self._memo[key] = (source, encoded)
        else:
            self._memo.pop(key, None)
            self._remaining.pop(key, None)
        return encoded


@ensure_local_storage
def insert_jobs(jobs: Job | list[Job]) -> list[int]:
    """Insert a job in the database.

    Method corresponding: :meth:`~mpqp.execution.job.Job.save`.

    Args:
        jobs: The job(s) to be inserted.

    Returns:
        The ID of the newly inserted job.

    Example:
        >>> job = Job(JobType.STATE_VECTOR, QCircuit(2), IBMDevice.AER_SIMULATOR)
        >>> insert_jobs(job)
        [17]

    """
    from sqlite3 import connect

    connection = connect(get_env_variable("DB_PATH"))
    try:
        cursor = connection.cursor()
        try:
            job_ids: list[int] = []
            if isinstance(jobs, list):
                for job in jobs:
                    circuit_json = json.dumps(repr(job.circuit))
                    measure_json = (
                        json.dumps(repr(job.measure)) if job.measure else None
                    )

                    cursor.execute(
                        '''
                        INSERT INTO jobs (type, circuit, device, measure, remote_id, status)
                        VALUES (?, ?, ?, ?)
                    ''',
                        (
                            job.job_type.name,
                            circuit_json,
                            str(job.device),
                            measure_json,
                            str(job.id),
                            str(job.status),
                        ),
                    )
                    id = cursor.lastrowid
                    if id is None:
                        raise ValueError("Job saving failed")
                    job_ids.append(id)

                    connection.commit()
            else:
                # TODO: I think we could factorize this part
                circuit_json = json.dumps(repr(jobs.circuit))
                measure_json = json.dumps(repr(jobs.measure)) if jobs.measure else None

                cursor.execute(
                    '''
                    INSERT INTO jobs (type, circuit, device, measure)
                    VALUES (?, ?, ?, ?)
                ''',
                    (
                        jobs.job_type.name,
                        circuit_json,
                        str(jobs.device),
                        measure_json,
                    ),
                )
                id = cursor.lastrowid
                if id is None:
                    raise ValueError("Job saving failed")
                job_ids.append(id)

                connection.commit()
        finally:
            cursor.close()
    finally:
        connection.close()
    return job_ids


@ensure_local_storage
def insert_results(
    result: Result | BatchResult | list[Result], reuse_similar_job: bool = True
) -> list[int]:
    """Insert a result or batch result into the database.

    Methods corresponding: :meth:`Result.save
    <mpqp.execution.result.Result.save>` and :meth:`BatchResult.save
    <mpqp.execution.result.BatchResult.save>`.

    Args:
        result: The result(s) to be inserted.
        reuse_similar_job: If ``True``, checks for an existing job in the
            database and reuses its ID to avoid duplicates.

    Returns:
        List of IDs of the inserted result(s).

    Example:
        >>> result = Result(Job(JobType.STATE_VECTOR, QCircuit(2), IBMDevice.AER_SIMULATOR), StateVector([1, 0, 0, 0]))
        >>> insert_results(result)
        [21]

    """
    results = [result] if isinstance(result, Result) else result
    input_circuits = _InputCircuitSerializer.for_results(results)
    return [_insert_result(item, reuse_similar_job, input_circuits) for item in results]


def _insert_result(
    result: Result,
    reuse_similar_job: bool = True,
    input_circuits: _InputCircuitSerializer | None = None,
):
    from sqlite3 import connect

    if reuse_similar_job:
        job_ids = fetch_jobs_with_job(result.job)
        if len(job_ids) == 0:
            job_id = insert_jobs(result.job)[0]
        else:
            job_id = job_ids[0]['id']
    else:
        job_id = insert_jobs(result.job)[0]

    connection = connect(get_env_variable("DB_PATH"))
    try:
        cursor = connection.cursor()
        try:
            data_json = json.dumps(
                repr(result._data)  # pyright: ignore[reportPrivateUsage]
            )
            error_json = (
                json.dumps(repr(result.error)) if result.error is not None else None
            )
            input_circuit_json = None
            if result.input_circuit is not None:
                input_circuit_json = (
                    json.dumps(repr(result.input_circuit))
                    if input_circuits is None
                    else input_circuits(result.input_circuit)
                )

            cursor.execute(
                '''
                INSERT INTO results (job_id, data, error, shots, label, input_circuit)
                VALUES (?, ?, ?, ?, ?, ?)
            ''',
                (
                    job_id,
                    data_json,
                    error_json,
                    result.shots,
                    result.label,
                    input_circuit_json,
                ),
            )

            result_id = cursor.lastrowid
            if result_id is None:
                raise ValueError("Result saving failed")

            connection.commit()
        finally:
            cursor.close()
    finally:
        connection.close()
    return result_id
