"""
This module provides functions to insert `Job` and `Result` objects into the database, 
ensuring proper linkage between jobs and results.

Functions:
- `insert_job`: Inserts a single `Job` into the database and returns its ID.
- `insert_result`: Inserts a `Result` or `BatchResult` into the database. Handles compilation of jobs to avoid duplicates.

"""

from __future__ import annotations

from mpqp.db.db_query import fetch_jobs_with_job
from mpqp.execution.connection.env_manager import get_env_variable
from mpqp.execution.job import Job
from mpqp.execution.result import BatchResult, Result


def insert_jobs(jobs: Job | list[Job]) -> list[int | None]:
    """
    Insert a `Job` into the database.

    Args:
        job: The `Job`(s) object to be inserted.

    Returns:
        The ID of the newly inserted job.

    Example:
        >>> job = Job(JobType.STATE_VECTOR, QCircuit(2), IBMDevice.AER_SIMULATOR)
        >>> insert_jobs(job)
        [7]

    """
    import json
    from sqlite3 import connect

    connection = connect(get_env_variable("DATA_BASE"))
    cursor = connection.cursor()

    job_ids = []
    if isinstance(jobs, list):
        for job in jobs:
            circuit_json = json.dumps(repr(job.circuit))
            measure_json = json.dumps(repr(job.measure)) if job.measure else None

            cursor.execute(
                '''
                INSERT INTO jobs (type, circuit, device, measure)
                VALUES (?, ?, ?, ?)
            ''',
                (
                    job.job_type.name,
                    circuit_json,
                    str(job.device),
                    measure_json,
                ),
            )
            job_ids.append(cursor.lastrowid)

            connection.commit()
    else:
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
        job_ids.append(cursor.lastrowid)

        connection.commit()

    connection.close()

    return job_ids


def insert_results(
    result: Result | BatchResult | list[Result], compile_same_job: bool = True
) -> list[int | None]:
    """
    Insert a `Result` or `BatchResult` into the database.

    Args:
        result: The `Result`(s) to be inserted.
        compile_same_job: If `True`, checks for an existing job in the database
                            and reuses its ID to avoid duplicates.

    Returns:
        List of IDs of the inserted result(s). Returns `None` for failed insertions.

    Example:
        >>> result = Result(Job(JobType.STATE_VECTOR, QCircuit(2), IBMDevice.AER_SIMULATOR), StateVector([1, 0, 0, 0]))
        >>> insert_results(result)
        [8]

    """
    if isinstance(result, Result):
        return [_insert_result(result, compile_same_job)]
    else:
        return [_insert_result(item, compile_same_job) for item in result]


def _insert_result(result: Result, compile_same_job: bool = True):
    import json
    from sqlite3 import connect

    if compile_same_job:
        job_ids = fetch_jobs_with_job(result.job)
        if len(job_ids) == 0:
            job_id = insert_jobs(result.job)[0]
        else:
            job_id = job_ids[0]['id']
    else:
        job_id = insert_jobs(result.job)[0]

    connection = connect(get_env_variable("DATA_BASE"))
    cursor = connection.cursor()

    data_json = json.dumps(repr(result._data))  # pyright: ignore[reportPrivateUsage]
    error_json = json.dumps(repr(result.error)) if result.error is not None else None

    cursor.execute(
        '''
        INSERT INTO results (job_id, data, error, shots)
        VALUES (?, ?, ?, ?)
    ''',
        (job_id, data_json, error_json, result.shots),
    )

    result_id = cursor.lastrowid

    connection.commit()
    connection.close()
    return result_id
