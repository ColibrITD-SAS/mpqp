"""
This module provides utility functions to query and fetch data from the quantum job and result database.

It includes methods to retrieve all records, specific records by ID, and filtered records based on `Job` or `Result` objects.

Functions:
- `fetch_all_jobs`: Fetch all job records from the database.
- `fetch_all_results`: Fetch all result records from the database.
- `fetch_jobs_with_job`: Fetch job records that match specific `Job` attributes.
- `fetch_jobs_with_result`: Fetch jobs associated with specific `Result` or `BatchResult` objects.
- `fetch_results_with_result_and_job`: Fetch results and their associated jobs based on specific `Result` attributes.
- `fetch_results_with_result`: Fetch results matching specific `Result` attributes.
- `fetch_results_with_id`: Fetch results by their ID(s).
- `fetch_jobs_with_id`: Fetch jobs by their ID(s).
- `fetch_results_with_job_id`: Fetch results associated with specific job ID(s).

"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from mpqp.execution.connection.env_manager import get_env_variable
from mpqp.execution.job import Job
from mpqp.execution.result import BatchResult, Result
from mpqp.local_storage.setup import ensure_db


@dataclass
class QueryJob:
    id: Optional[str] = None


@ensure_db
def fetch_all_jobs():
    """
    Fetch all job records from the database.

    Returns:
        List of job records as dictionaries, or an empty list if no jobs exist.

    Examples:
        >>> jobs = fetch_all_jobs()
        >>> for job in jobs:
        ...    print("job_id:", job['id'])
        job_id: 1
        job_id: 2
        job_id: 3
        job_id: 4
        job_id: 5
        job_id: 6

    """
    from sqlite3 import Row, connect

    with connect(get_env_variable("DATA_BASE")) as connection:
        cursor = connection.cursor()
        cursor.row_factory = Row  # pyright: ignore[reportAttributeAccessIssue]

        cursor.execute('SELECT * FROM jobs')
        jobs = cursor.fetchall()
        return [dict(job) for job in jobs]


@ensure_db
def fetch_all_results():
    """
    Fetch all result records from the database.

    Returns:
        List of result records as dictionaries, or an empty list if no results exist.

    Examples:
        >>> results = fetch_all_results()
        >>> for result in results:
        ...    print("result_id:", result['id'])
        result_id: 1
        result_id: 2
        result_id: 3
        result_id: 4
        result_id: 5
        result_id: 6
        result_id: 7

    """
    from sqlite3 import Row, connect

    with connect(get_env_variable("DATA_BASE")) as connection:
        cursor = connection.cursor()
        cursor.row_factory = Row  # pyright: ignore[reportAttributeAccessIssue]

        cursor.execute('SELECT * FROM results')
        results = cursor.fetchall()
        return [dict(result) for result in results]


@ensure_db
def fetch_results_with_id(result_id: int | list[int]):
    """
    Fetch results by their ID(s).

    Args:
        result_id: A result ID or list of result IDs.

    Returns:
        List of result records as dictionaries, or an empty list if no matches exist.

    Examples:
        >>> results = fetch_results_with_id([1, 2, 3])
        >>> for result in results:
        ...    print("result_id:", result['id'])
        result_id: 1
        result_id: 2
        result_id: 3

    """
    from sqlite3 import Row, connect

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
        return [dict(result) for result in results]


@ensure_db
def fetch_jobs_with_id(job_id: int | list[int]):
    """
    Fetch jobs by their ID(s).

    Args:
        job_id: A job ID or list of job IDs.

    Returns:
        List of job records as dictionaries, or an empty list if no matches exist.

    Examples:
        >>> jobs = fetch_jobs_with_id(1)
        >>> for job in jobs:
        ...    print("job_id:", job['id'])
        job_id: 1
        >>> jobs = fetch_jobs_with_id([2, 3])
        >>> for job in jobs:
        ...    print("job_id:", job['id'])
        job_id: 2
        job_id: 3

    """
    from sqlite3 import Row, connect

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
        return [dict(job) for job in jobs]


@ensure_db
def fetch_results_with_job_id(job_id: int | list[int]):
    """
    Fetch results associated with specific job ID(s).

    Args:
        job_id: A job ID or list of job IDs.

    Returns:
        List of result records as dictionaries, or an empty list if no matches exist.

    Examples:
        >>> results = fetch_results_with_job_id(1)
        >>> for result in results:
        ...    print("result_id:", result['id'], ", job_id:", result['job_id'])
        result_id: 1 , job_id: 1
        result_id: 2 , job_id: 1

    """
    from sqlite3 import Row, connect

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
        return [dict(result) for result in results]


@ensure_db
def fetch_jobs_with_job(job: Job | list[Job]):
    """
    Fetch job records matching specific `Job` attributes.

    Args:
        job: A `Job` or list of `Job` objects to match.

    Returns:
        List of matching job records as dictionaries, or an empty list if no matches exist.

    Examples:
        >>> job = Job(JobType.STATE_VECTOR, QCircuit(2), IBMDevice.AER_SIMULATOR)
        >>> matching_jobs = fetch_jobs_with_job(job)

    """
    from sqlite3 import Row, connect

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
        return [dict(job) for job in jobs]


@ensure_db
def fetch_jobs_with_result(result: Result | BatchResult | list[Result]):
    """
    Fetch jobs associated with specific `Result` or `BatchResult` objects.

    Args:
        result: A `Result`, `BatchResult`, or list of results to match.

    Returns:
        List of matching job records as dictionaries, or an empty list if no matches exist.

    Examples:
        >>> result = Result(Job(JobType.STATE_VECTOR,QCircuit([], nb_qubits=2, label="circuit 1"),IBMDevice.AER_SIMULATOR,),StateVector([1, 0, 0, 0]),0,0)
        >>> jobs = fetch_jobs_with_result(result)
        >>> for job in jobs:
        ...    print("job_id:", job['id'])
        job_id: 5

    """
    from sqlite3 import Row, connect

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
            error_json = json.dumps(repr(res.error)) if res.error is not None else None
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
            SELECT jobs.* FROM jobs
            INNER JOIN results ON jobs.id is results.job_id
            WHERE {' OR '.join(job_filters)}
        """

        cursor.execute(query, params)
        jobs = cursor.fetchall()

        return [dict(job) for job in jobs]


@ensure_db
def fetch_results_with_result_and_job(result: Result | BatchResult | list[Result]):
    """
    Fetch results and their associated jobs based on specific `Result` attributes.

    Args:
        result: A `Result`, `BatchResult`, or list of results to match.

    Returns:
        List of matching result records and their associated jobs as dictionaries.

    Examples:
        >>> result = Result(Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, label="circuit 1"), IBMDevice.AER_SIMULATOR), StateVector([1, 0, 0, 0]), 0, 0)
        >>> results = fetch_results_with_result_and_job(result)
        >>> for result in results:
        ...    print("result_id:", result['id'], ", job_id:", result['job_id'])
        result_id: 6 , job_id: 5

    """
    from sqlite3 import Row, connect

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
            error_json = json.dumps(repr(res.error)) if res.error is not None else None
            circuit_json = json.dumps(repr(res.job.circuit))
            measure_json = (
                json.dumps(repr(res.job.measure)) if res.job.measure else None
            )

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

        query = f"""
            SELECT results.* FROM results
            INNER JOIN jobs ON jobs.id is results.job_id
            WHERE {' OR '.join(result_filters)}
        """

        cursor.execute(query, params)
        results = cursor.fetchall()

        return [dict(result) for result in results]


@ensure_db
def fetch_results_with_job(jobs: Job | list[Job]):
    """
    Fetch results and their associated jobs based on specific attributes.

    Args:
        jobs: A `Job` or list of `Job` objects to match.

    Returns:
        List of matching result records and their associated jobs as dictionaries.

    Examples:
        >>> job = Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, label="circuit 1"), IBMDevice.AER_SIMULATOR)
        >>> results = fetch_results_with_job(job)
        >>> for result in results:
        ...    print("result_id:", result['id'], ", job_id:", result['job_id'])
        result_id: 6 , job_id: 5

    """
    from sqlite3 import Row, connect

    with connect(get_env_variable("DATA_BASE")) as connection:
        cursor = connection.cursor()
        cursor.row_factory = Row  # pyright: ignore[reportAttributeAccessIssue]

        result_filters = []
        params = []

        if isinstance(jobs, Job):
            jobs = [jobs]

        for job in jobs:
            circuit_json = json.dumps(repr(job.circuit))
            measure_json = json.dumps(repr(job.measure)) if job.measure else None

            result_filters.append(
                """
                (jobs.type is ? AND jobs.circuit is ? AND jobs.device is ? AND jobs.measure is ?)
            """
            )
            params.extend(
                [
                    job.job_type.name,
                    circuit_json,
                    str(job.device),
                    measure_json,
                ]
            )

        query = f"""
            SELECT results.* FROM results
            INNER JOIN jobs ON jobs.id is results.job_id
            WHERE {' OR '.join(result_filters)}
        """

        cursor.execute(query, params)
        jobs_db = cursor.fetchall()

        return [dict(job) for job in jobs_db]


@ensure_db
def fetch_results_with_result(result: Result | BatchResult | list[Result]):
    """
    Fetch results matching specific `Result` attributes.

    Args:
        result: A `Result`, `BatchResult`, or list of results to match.

    Returns:
        List of matching result records as dictionaries, or an empty list if no matches exist.

    Examples:
        >>> result = Result(Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, label="circuit 1"), IBMDevice.AER_SIMULATOR), StateVector([1, 0, 0, 0]), 0, 0)
        >>> results = fetch_results_with_result(result)
        >>> for result in results:
        ...    print("result_id:", result['id'], ", job_id:", result['job_id'])
        result_id: 6 , job_id: 5
        result_id: 7 , job_id: 6

    """
    from sqlite3 import Row, connect

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
            error_json = json.dumps(repr(res.error)) if res.error is not None else None

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

        return [dict(result) for result in results]
