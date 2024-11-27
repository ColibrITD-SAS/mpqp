import json

from mpqp.execution.connection.env_manager import get_env_variable
from mpqp.execution.job import Job
from mpqp.execution.result import Result


def fetch_all_jobs():
    from sqlite3 import connect, Row

    with connect(get_env_variable("DATA_BASE")) as connection:
        cursor = connection.cursor()
        cursor.row_factory = Row  # pyright: ignore[reportAttributeAccessIssue]

        cursor.execute('SELECT * FROM jobs')
        jobs = cursor.fetchall()
        if jobs:
            return [dict(job) for job in jobs]
        return None


def fetch_all_results():
    from sqlite3 import connect, Row

    with connect(get_env_variable("DATA_BASE")) as connection:
        cursor = connection.cursor()
        cursor.row_factory = Row  # pyright: ignore[reportAttributeAccessIssue]

        cursor.execute('SELECT * FROM results')
        results = cursor.fetchall()
        if results:
            return [dict(result) for result in results]
        return None


def fetch_jobs_with_job(job: Job):
    from sqlite3 import connect, Row

    with connect(get_env_variable("DATA_BASE")) as connection:
        cursor = connection.cursor()
        cursor.row_factory = Row  # pyright: ignore[reportAttributeAccessIssue]

        circuit_json = json.dumps(repr(job.circuit))
        measure_json = json.dumps(repr(job.measure)) if job.measure else None

        cursor.execute(
            '''
            SELECT * FROM jobs
            WHERE type is ? AND circuit is ? AND device is ? AND measure is ?
            ''',
            (job.job_type.name, circuit_json, str(job.device), measure_json),
        )

        jobs = cursor.fetchall()
        if jobs:
            return [dict(job) for job in jobs]
        return None


def fetch_jobs_with_results(result: Result):
    from sqlite3 import connect, Row

    with connect(get_env_variable("DATA_BASE")) as connection:
        cursor = connection.cursor()
        cursor.row_factory = Row  # pyright: ignore[reportAttributeAccessIssue]

        data_json = json.dumps(
            repr(result._data)  # pyright: ignore[reportPrivateUsage]
        )
        error_json = json.dumps(repr(result.error)) if result.error else None

        circuit_json = json.dumps(repr(result.job.circuit))
        measure_json = (
            json.dumps(repr(result.job.measure)) if result.job.measure else None
        )

        cursor.execute(
            '''
            SELECT jobs.* FROM jobs
            INNER JOIN results ON jobs.id is results.job_id
            WHERE results.data is ? AND results.error is ? AND results.shots is ?
                AND jobs.type is ? AND jobs.circuit is ? AND jobs.device is ? AND jobs.measure is ?
            ''',
            (
                data_json,
                error_json,
                result.shots,
                result.job.job_type.name,
                circuit_json,
                str(result.job.device),
                measure_json,
            ),
        )

        jobs = cursor.fetchall()
        if jobs:
            return [dict(job) for job in jobs]
        return None


def fetch_results_with_results_and_job(result: Result):
    from sqlite3 import connect, Row

    with connect(get_env_variable("DATA_BASE")) as connection:
        cursor = connection.cursor()
        cursor.row_factory = Row  # pyright: ignore[reportAttributeAccessIssue]

        data_json = json.dumps(
            repr(result._data)  # pyright: ignore[reportPrivateUsage]
        )
        error_json = json.dumps(repr(result.error)) if result.error else None

        circuit_json = json.dumps(repr(result.job.circuit))
        measure_json = (
            json.dumps(repr(result.job.measure)) if result.job.measure else None
        )

        cursor.execute(
            '''
            SELECT * FROM results
            WHERE job_id is (
            SELECT id FROM jobs
            WHERE id is (
                SELECT job_id FROM results
                WHERE data is ? AND error is ? AND shots is ?
            )
            AND type is ? AND circuit is ? AND device is ? AND measure is ?
            )
            ''',
            (
                data_json,
                error_json,
                result.shots,
                result.job.job_type.name,
                circuit_json,
                str(result.job.device),
                measure_json,
            ),
        )

        results = cursor.fetchall()
        if results:
            return [dict(result) for result in results]
        return None


def fetch_results_with_results(result: Result):
    from sqlite3 import connect, Row

    with connect(get_env_variable("DATA_BASE")) as connection:
        cursor = connection.cursor()
        cursor.row_factory = Row  # pyright: ignore[reportAttributeAccessIssue]

        data_json = json.dumps(
            repr(result._data)  # pyright: ignore[reportPrivateUsage]
        )
        error_json = json.dumps(repr(result.error)) if result.error else None

        cursor.execute(
            '''
            SELECT * FROM results
            WHERE data is ? AND error is ? AND shots is ?
            ''',
            (
                data_json,
                error_json,
                result.shots,
            ),
        )

        results = cursor.fetchall()
        if results:
            return [dict(result) for result in results]
        return None


def fetch_result_with_id(result_id: int):
    from sqlite3 import connect, Row

    with connect(get_env_variable("DATA_BASE")) as connection:
        cursor = connection.cursor()
        cursor.row_factory = Row  # pyright: ignore[reportAttributeAccessIssue]

        cursor.execute(
            '''
            SELECT * FROM results
            WHERE id is ?
            ''',
            (result_id,),
        )

        result = cursor.fetchone()
        if result:
            return dict(result)
        return None


def fetch_job_with_id(job_id: int):
    from sqlite3 import connect, Row

    with connect(get_env_variable("DATA_BASE")) as connection:
        cursor = connection.cursor()
        cursor.row_factory = Row  # pyright: ignore[reportAttributeAccessIssue]

        cursor.execute(
            '''
            SELECT * FROM jobs
            WHERE id is ?
            ''',
            (job_id,),
        )

        job = cursor.fetchone()
        if job:
            return dict(job)
        return None
