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


def insert_job(job: Job):
    """
    Insert a `Job` into the database.

    Args:
        job: The `Job` object to be inserted.

    Returns:
        The ID of the newly inserted job.

    Example:
        >>> job = Job(JobType.STATE_VECTOR, QCircuit(2), IBMDevice.AER_SIMULATOR)
        >>> insert_job(job)
        7

    """
    import json
    from sqlite3 import connect

    connection = connect(get_env_variable("DATA_BASE"))
    cursor = connection.cursor()

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
    job_id = cursor.lastrowid

    connection.commit()
    connection.close()
    return job_id


def insert_result(
    result: Result | BatchResult, compile_same_job: bool = True
) -> list[int | None]:
    """
    Insert a `Result` or `BatchResult` into the database.

    Args:
        result: The result(s) to be inserted.
        compile_same_job: If `True`, checks for an existing job in the database
                            and reuses its ID to avoid duplicates.

    Returns:
        List of IDs of the inserted result(s). Returns `None` for failed insertions.

    Example:
        >>> result = Result(Job(JobType.STATE_VECTOR, QCircuit(2), IBMDevice.AER_SIMULATOR), StateVector([1, 0, 0, 0]))
        >>> insert_result(result)
        [8]

    """
    if isinstance(result, BatchResult):
        return [_insert_result(item, compile_same_job) for item in result.results]
    else:
        return [_insert_result(result, compile_same_job)]


def _insert_result(result: Result, compile_same_job: bool = True):
    import json
    from sqlite3 import connect

    if compile_same_job:
        job_ids = fetch_jobs_with_job(result.job)
        if len(job_ids) == 0:
            job_id = insert_job(result.job)
        else:
            job_id = job_ids[0]['id']
    else:
        job_id = insert_job(result.job)

    connection = connect(get_env_variable("DATA_BASE"))
    cursor = connection.cursor()

    data_json = json.dumps(repr(result._data))  # pyright: ignore[reportPrivateUsage]
    error_json = json.dumps(repr(result.error)) if result.error else None

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
