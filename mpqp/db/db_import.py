from __future__ import annotations

from mpqp.db.db_query import fetch_jobs_with_job
from mpqp.execution.connection.env_manager import get_env_variable
from mpqp.execution.job import Job
from mpqp.execution.result import BatchResult, Result


def insert_job(job: Job):
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
    if isinstance(result, BatchResult):
        return [insert_result_(item, compile_same_job) for item in result.results]
    else:
        return [insert_result_(result, compile_same_job)]


def insert_result_(result: Result, compile_same_job: bool = True):
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
