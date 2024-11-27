from __future__ import annotations
import json

from mpqp.execution.connection.env_manager import get_env_variable
from mpqp.execution.job import Job
from mpqp.execution.result import BatchResult, Result


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
        return [dict(result) for result in results] if results else []


def fetch_jobs_with_job(job: Job | list[Job]):
    from sqlite3 import connect, Row

    with connect(get_env_variable("DATA_BASE")) as connection:
        cursor = connection.cursor()
        cursor.row_factory = Row  # pyright: ignore[reportAttributeAccessIssue]

        job_filters = []
        params = []

        if isinstance(job, Job):
            job = [job]

        for j in job:
            circuit_json = json.dumps(repr(j.circuit))
            measure_json = json.dumps(repr(j.measure)) if j.measure else None
            job_filters.append(
                "(type is ? AND circuit is ? AND device is ? AND measure IS ?)"
            )
            params.extend([j.job_type.name, circuit_json, str(j.device), measure_json])

        query = f"SELECT * FROM jobs WHERE {' OR '.join(job_filters)}"
        cursor.execute(query, params)

        jobs = cursor.fetchall()
        return [dict(job) for job in jobs] if jobs else []


def fetch_jobs_with_result(result: Result | BatchResult | list[Result]):
    from sqlite3 import connect, Row

    with connect(get_env_variable("DATA_BASE")) as connection:
        cursor = connection.cursor()
        cursor.row_factory = Row  # pyright: ignore[reportAttributeAccessIssue]

        job_filters = []
        params = []

        if isinstance(result, Result):
            result = [result]

        for res in result:
            data_json = json.dumps(
                repr(res._data)  # pyright: ignore[reportPrivateUsage]
            )
            error_json = json.dumps(repr(res.error)) if res.error else None
            circuit_json = json.dumps(repr(res.job.circuit))
            measure_json = (
                json.dumps(repr(res.job.measure)) if res.job.measure else None
            )

            job_filters.append(
                """
                (results.data is ? AND results.error is ? AND results.shots is ? 
                AND jobs.type is ? AND jobs.circuit is ? AND jobs.device is ? AND jobs.measure is ?)
            """
            )
            params.extend(
                [
                    data_json,
                    error_json,
                    res.shots,
                    res.job.job_type.name,
                    circuit_json,
                    str(res.job.device),
                    measure_json,
                ]
            )

        query = f"""
            SELECT * FROM jobs
            INNER JOIN results ON jobs.id is results.job_id
            WHERE {' OR '.join(job_filters)}
        """

        cursor.execute(query, params)
        jobs = cursor.fetchall()

        return [dict(job) for job in jobs] if jobs else []


def fetch_results_with_result_and_job(result: Result | BatchResult | list[Result]):
    from sqlite3 import connect, Row

    with connect(get_env_variable("DATA_BASE")) as connection:
        cursor = connection.cursor()
        cursor.row_factory = Row  # pyright: ignore[reportAttributeAccessIssue]

        result_filters = []
        params = []

        if isinstance(result, Result):
            result = [result]

        for res in result:
            data_json = json.dumps(
                repr(res._data)  # pyright: ignore[reportPrivateUsage]
            )
            error_json = json.dumps(repr(res.error)) if res.error else None
            circuit_json = json.dumps(repr(res.job.circuit))
            measure_json = (
                json.dumps(repr(res.job.measure)) if res.job.measure else None
            )

            # Build the WHERE clause for each result/job pair
            result_filters.append(
                """
                (results.data is ? AND results.error is ? AND results.shots is ? 
                AND jobs.type is ? AND jobs.circuit is ? AND jobs.device is ? AND jobs.measure is ?)
            """
            )
            params.extend(
                [
                    data_json,
                    error_json,
                    res.shots,
                    res.job.job_type.name,
                    circuit_json,
                    str(res.job.device),
                    measure_json,
                ]
            )

        # Construct the final query with OR conditions for multiple results
        query = f"""
            SELECT * FROM results
            INNER JOIN jobs ON jobs.id is results.job_id
            WHERE {' OR '.join(result_filters)}
        """

        # Execute the query
        cursor.execute(query, params)
        results = cursor.fetchall()

        # Return the results as a list of dictionaries, or None if no results were found
        return [dict(result) for result in results] if results else []


def fetch_results_with_result(result: Result | BatchResult | list[Result]):
    from sqlite3 import connect, Row

    with connect(get_env_variable("DATA_BASE")) as connection:
        cursor = connection.cursor()
        cursor.row_factory = Row  # pyright: ignore[reportAttributeAccessIssue]

        result_filters = []
        params = []

        if isinstance(result, Result):
            result = [result]

        for res in result:
            data_json = json.dumps(
                repr(res._data)  # pyright: ignore[reportPrivateUsage]
            )
            error_json = json.dumps(repr(res.error)) if res.error else None

            result_filters.append(
                """
                (results.data is ? AND results.error is ? AND results.shots is ?)
            """
            )
            params.extend([data_json, error_json, res.shots])

        query = f"""
            SELECT * FROM results
            WHERE {' OR '.join(result_filters)}
        """

        cursor.execute(query, params)
        results = cursor.fetchall()

        return [dict(result) for result in results] if results else []


def fetch_results_with_id(result_id: int | list[int]):
    from sqlite3 import connect, Row

    with connect(get_env_variable("DATA_BASE")) as connection:
        cursor = connection.cursor()
        cursor.row_factory = Row  # pyright: ignore[reportAttributeAccessIssue]

        if isinstance(result_id, int):
            result_id = [result_id]

        placeholders = ",".join("?" for _ in result_id)

        cursor.execute(
            f"SELECT * FROM results WHERE id IN ({placeholders})", tuple(result_id)
        )

        results = cursor.fetchall()
        return [dict(result) for result in results] if results else []


def fetch_jobs_with_id(job_id: int | list[int]):
    from sqlite3 import connect, Row

    with connect(get_env_variable("DATA_BASE")) as connection:
        cursor = connection.cursor()
        cursor.row_factory = Row  # pyright: ignore[reportAttributeAccessIssue]

        if isinstance(job_id, int):
            job_id = [job_id]

        placeholders = ",".join("?" for _ in job_id)

        cursor.execute(
            f"SELECT * FROM jobs WHERE id IN ({placeholders})", tuple(job_id)
        )

        jobs = cursor.fetchall()
        return [dict(job) for job in jobs] if jobs else []


def fetch_results_with_job_id(job_id: int | list[int]):
    from sqlite3 import connect, Row

    with connect(get_env_variable("DATA_BASE")) as connection:
        cursor = connection.cursor()
        cursor.row_factory = Row  # pyright: ignore[reportAttributeAccessIssue]

        if isinstance(job_id, int):
            job_id = [job_id]

        placeholders = ",".join("?" for _ in job_id)

        cursor.execute(
            f"SELECT * FROM results WHERE job_id IN ({placeholders})", tuple(job_id)
        )

        results = cursor.fetchall()
        return [dict(result) for result in results] if results else []
