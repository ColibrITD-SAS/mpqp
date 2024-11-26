from __future__ import annotations

from mpqp.execution.connection.env_manager import get_env_variable
from mpqp.execution.job import Job
from mpqp.execution.result import BatchResult, Result, Sample, StateVector


def insert_job(job: Job):
    import json
    from sqlite3 import connect

    connection = connect(get_env_variable("DATA_BASE"))
    cursor = connection.cursor()

    circuit_json = json.dumps(job.circuit.to_dict())
    measure_json = json.dumps(job.measure.to_dict()) if job.measure else None

    cursor.execute(
        '''
        INSERT INTO jobs (type, circuit, device, measure, status)
        VALUES (?, ?, ?, ?, ?)
    ''',
        (
            str(job.job_type),
            circuit_json,
            str(job.device),
            measure_json,
            str(job.status),
        ),
    )
    connection.commit()
    connection.close()
    return cursor.lastrowid


def insert_result(result: Result | BatchResult) -> list[int | None]:
    if isinstance(result, BatchResult):
        return [insert_result_(item) for item in result.results]
    else:
        return [insert_result_(result)]


def insert_result_(result: Result):
    import json
    from sqlite3 import connect

    job_id = insert_job(result.job)
    connection = connect(get_env_variable("DATA_BASE"))
    cursor = connection.cursor()
    data = result._data  # pyright: ignore[reportPrivateUsage]
    if isinstance(data, StateVector):
        data_json = json.dumps(data.to_dict())
    elif isinstance(data, list) and (
        isinstance(sample, Sample)  # pyright: ignore[reportUnnecessaryIsInstance]
        for sample in data
    ):
        data_json = json.dumps([sample.to_dict() for sample in data])
    else:
        data_json = json.dumps(data)
    error_json = json.dumps(result.error) if result.error else None

    cursor.execute(
        '''
        INSERT INTO results (job_id, data, error, shots)
        VALUES (?, ?, ?, ?)
    ''',
        (job_id, data_json, error_json, result.shots),
    )
    connection.commit()
    connection.close()
    return cursor.lastrowid
