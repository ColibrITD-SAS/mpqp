"""provides utility functions to query and fetch data from the quantum job and
result database. It includes methods to retrieve all records, specific records
by ID, and filtered records based on :class:`~mpqp.execution.job.Job` or
:class:`~mpqp.execution.result.Result` objects."""

from __future__ import annotations

import json

from mpqp.execution.connection.env_manager import get_env_variable
from mpqp.execution.job import Job
from mpqp.execution.result import BatchResult, Result
from mpqp.local_storage.setup import DictDB, ensure_local_storage


@ensure_local_storage
def fetch_all_jobs() -> list[DictDB]:
    """Fetch all job records from the database.

    Returns:
        All jobs as dictionaries.

    Examples:
        >>> jobs = fetch_all_jobs()
        >>> for job in jobs:
        ...    print("job:", job) # doctest: +ELLIPSIS
        job: {'id': 1, 'type': 'SAMPLE', 'circuit': '"QCircuit(...)"', 'device': 'IBMDevice.AER_SIMULATOR', 'measure': '"BasisMeasure([0, 1], c_targets=[0, 1])"', 'remote_id': None, 'status': None, 'created_at': '...'}
        job: {'id': 2, 'type': 'SAMPLE', 'circuit': '"QCircuit(...)"', 'device': 'GOOGLEDevice.CIRQ_LOCAL_SIMULATOR', 'measure': '"BasisMeasure([0, 1], c_targets=[0, 1])"', 'remote_id': None, 'status': None, 'created_at': '...'}
        job: {'id': 3, 'type': 'SAMPLE', 'circuit': '"QCircuit(...)"', 'device': 'IBMDevice.AER_SIMULATOR', 'measure': '"BasisMeasure([0], c_targets=[0])"', 'remote_id': None, 'status': None, 'created_at': '...'}
        job: {'id': 4, 'type': 'SAMPLE', 'circuit': '"QCircuit(...)"', 'device': 'GOOGLEDevice.CIRQ_LOCAL_SIMULATOR', 'measure': '"BasisMeasure([0], c_targets=[0])"', 'remote_id': None, 'status': None, 'created_at': '...'}
        job: {'id': 5, 'type': 'STATE_VECTOR', 'circuit': '"QCircuit(...)"', 'device': 'IBMDevice.AER_SIMULATOR', 'measure': None, 'remote_id': None, 'status': None, 'created_at': '...'}
        job: {'id': 6, 'type': 'STATE_VECTOR', 'circuit': '"QCircuit(...)"', 'device': 'IBMDevice.AER_SIMULATOR', 'measure': None, 'remote_id': None, 'status': None, 'created_at': '...'}

    """
    from sqlite3 import Row, connect

    connection = connect(get_env_variable("DB_PATH"))
    try:
        connection.row_factory = Row
        cursor = connection.cursor()
        try:
            cursor.execute('SELECT * FROM jobs')
            jobs = cursor.fetchall()

            return [dict(job) for job in jobs]
        finally:
            cursor.close()
    finally:
        connection.close()


@ensure_local_storage
def fetch_all_results() -> list[DictDB]:
    """Fetch all result records from the database.

    Returns:
        All results as dictionaries.

    Examples:
        >>> results = fetch_all_results()
        >>> for result in results:
        ...    print("result:", result) # doctest: +ELLIPSIS
        result: {'id': 1, 'job_id': 1, 'data': '"[Sample(...), Sample(...)]"', 'error': None, 'shots': 1024, 'created_at': '...'}
        result: {'id': 2, 'job_id': 1, 'data': '"[Sample(...), Sample(...)]"', 'error': None, 'shots': 1024, 'created_at': '...'}
        result: {'id': 3, 'job_id': 2, 'data': '"[Sample(...), Sample(...)]"', 'error': None, 'shots': 1024, 'created_at': '...'}
        result: {'id': 4, 'job_id': 3, 'data': '"[Sample(...), Sample(...)]"', 'error': None, 'shots': 1024, 'created_at': '...'}
        result: {'id': 5, 'job_id': 4, 'data': '"[Sample(...), Sample(...)]"', 'error': None, 'shots': 1024, 'created_at': '...'}
        result: {'id': 6, 'job_id': 5, 'data': '"StateVector([1, 0, 0, 0])"', 'error': '"0"', 'shots': 0, 'created_at': '...'}
        result: {'id': 7, 'job_id': 6, 'data': '"StateVector([1, 0, 0, 0])"', 'error': '"0"', 'shots': 0, 'created_at': '...'}

    """
    from sqlite3 import Row, connect

    connection = connect(get_env_variable("DB_PATH"))
    try:
        connection.row_factory = Row
        cursor = connection.cursor()
        try:

            cursor.execute('SELECT * FROM results')
            results = cursor.fetchall()
        finally:
            cursor.close()
    finally:
        connection.close()
    return [dict(result) for result in results]


@ensure_local_storage
def fetch_results_with_id(result_id: int | list[int]) -> list[DictDB]:
    """Fetch result(s) by their ID(s).

    Args:
        result_id: The ID(s) of the result(s) to fetch.

    Returns:
        Matching result(s) as dictionaries corresponding to the id(s).

    Examples:
        >>> results = fetch_results_with_id([1, 2, 3])
        >>> for result in results:
        ...    print("result:", result) # doctest: +ELLIPSIS
        result: {'id': 1, 'job_id': 1, 'data': '"[Sample(...), Sample(...)]"', 'error': None, 'shots': 1024, 'created_at': '...'}
        result: {'id': 2, 'job_id': 1, 'data': '"[Sample(...), Sample(...)]"', 'error': None, 'shots': 1024, 'created_at': '...'}
        result: {'id': 3, 'job_id': 2, 'data': '"[Sample(...), Sample(...)]"', 'error': None, 'shots': 1024, 'created_at': '...'}

    """
    from sqlite3 import Row, connect

    connection = connect(get_env_variable("DB_PATH"))
    try:
        connection.row_factory = Row
        cursor = connection.cursor()
        try:

            if isinstance(result_id, int):
                result_id = [result_id]

            placeholders = ",".join("?" for _ in result_id)

            cursor.execute(
                f"SELECT * FROM results WHERE id IN ({placeholders})", tuple(result_id)
            )

            results = cursor.fetchall()
        finally:
            cursor.close()
    finally:
        connection.close()
    return [dict(result) for result in results]


@ensure_local_storage
def fetch_jobs_with_id(job_id: int | list[int]) -> list[DictDB]:
    """Fetch job(s) by their ID(s).

    Args:
        job_id: The ID(s) of the job(s) to fetch.

    Returns:
        Matching job(s) as dictionaries corresponding to the id(s).

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

    connection = connect(get_env_variable("DB_PATH"))
    try:
        connection.row_factory = Row
        cursor = connection.cursor()
        try:

            if isinstance(job_id, int):
                job_id = [job_id]

            placeholders = ",".join("?" for _ in job_id)

            cursor.execute(
                f"SELECT * FROM jobs WHERE id IN ({placeholders})", tuple(job_id)
            )

            jobs = cursor.fetchall()
        finally:
            cursor.close()
    finally:
        connection.close()
    return [dict(job) for job in jobs]


@ensure_local_storage
def fetch_results_with_job_id(job_id: int | list[int]) -> list[DictDB]:
    """Fetch result(s) associated with specific job ID(s).

    Args:
        job_id: The ID(s) of job(s) to which the desired result(s) are attached.

    Returns:
        Matching result(s) as dictionaries corresponding to the job id(s).

    Examples:
        >>> results = fetch_results_with_job_id(1)
        >>> for result in results:
        ...    print("result_id:", result['id'], ", job_id:", result['job_id'])
        result_id: 1 , job_id: 1
        result_id: 2 , job_id: 1

    """
    from sqlite3 import Row, connect

    connection = connect(get_env_variable("DB_PATH"))
    try:
        connection.row_factory = Row
        cursor = connection.cursor()
        try:

            if isinstance(job_id, int):
                job_id = [job_id]

            placeholders = ",".join("?" for _ in job_id)

            cursor.execute(
                f"SELECT * FROM results WHERE job_id IN ({placeholders})", tuple(job_id)
            )

            results = cursor.fetchall()
        finally:
            cursor.close()
    finally:
        connection.close()

    return [dict(result) for result in results]


@ensure_local_storage
def fetch_jobs_with_job(job: Job | list[Job]) -> list[DictDB]:
    """Fetch job(s) records matching specific job(s) attributes:

    - job type,
    - circuit,
    - device,
    - measure.

    Args:
        job: Job(s) to match.

    Returns:
        Matching job(s) as dictionaries corresponding to the job(s) attributes

    Examples:
        >>> job = Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, label="circuit 1"), IBMDevice.AER_SIMULATOR)
        >>> jobs = fetch_jobs_with_job(job)
        >>> for job in jobs:
        ...    print("job:", job) # doctest: +ELLIPSIS
        job: {'id': 5, 'type': 'STATE_VECTOR', 'circuit': '"QCircuit(...)"', 'device': 'IBMDevice.AER_SIMULATOR', 'measure': None, 'remote_id': None, 'status': None, 'created_at': '...'}

    """
    from sqlite3 import Row, connect

    connection = connect(get_env_variable("DB_PATH"))
    try:
        connection.row_factory = Row
        cursor = connection.cursor()
        try:

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
                params.extend(
                    [j.job_type.name, circuit_json, str(j.device), measure_json]
                )

            query = f"SELECT * FROM jobs WHERE {' OR '.join(job_filters)}"
            cursor.execute(query, params)

            jobs = cursor.fetchall()
            return [dict(job) for job in jobs]
        finally:
            cursor.close()
    finally:
        connection.close()


@ensure_local_storage
def fetch_jobs_with_result_and_job(
    result: Result | BatchResult | list[Result],
) -> list[DictDB]:
    """Fetch job(s) associated with specific results(s) attributes:

    - data,
    - error,
    - shots.

    And also with the job attribute of the results(s):

    - job type,
    - circuit,
    - device,
    - measure.

    Args:
        result: Result(s) to match.

    Returns:
        Matching job(s) as dictionaries corresponding to the result(s) attribute and job attribute of the result(s).

    Examples:
        >>> result = Result(Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, label="circuit 1"), IBMDevice.AER_SIMULATOR,), StateVector([1, 0, 0, 0]),0,0)
        >>> jobs = fetch_jobs_with_result_and_job(result)
        >>> for job in jobs:
        ...    print("job:", job) # doctest: +ELLIPSIS
        job: {'id': 5, 'type': 'STATE_VECTOR', 'circuit': '"QCircuit(...)"', 'device': 'IBMDevice.AER_SIMULATOR', 'measure': None, 'remote_id': None, 'status': None, 'created_at': '...'}

    """
    from sqlite3 import Row, connect

    connection = connect(get_env_variable("DB_PATH"))
    try:
        connection.row_factory = Row
        cursor = connection.cursor()
        try:

            job_filters = []
            params = []

            if isinstance(result, Result):
                result = [result]

            for res in result:
                data_json = json.dumps(
                    repr(res._data)  # pyright: ignore[reportPrivateUsage]
                )
                error_json = (
                    json.dumps(repr(res.error)) if res.error is not None else None
                )
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
        finally:
            cursor.close()
    finally:
        connection.close()
    return [dict(job) for job in jobs]


@ensure_local_storage
def fetch_jobs_with_result(result: Result | BatchResult | list[Result]) -> list[DictDB]:
    """Fetch job(s) associated with specific results(s) attributes:

        - data,
        - error,
        - shots.

    Args:
        result: Result(s) to match.

    Returns:
        Matching job(s) as dictionaries corresponding to the result(s) attribute.

    Examples:
        >>> result = Result(Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, label="circuit 1"), IBMDevice.AER_SIMULATOR,), StateVector([1, 0, 0, 0]),0,0)
        >>> jobs = fetch_jobs_with_result(result)
        >>> for job in jobs:
        ...    print("job:", job) # doctest: +ELLIPSIS
        job: {'id': 5, 'type': 'STATE_VECTOR', 'circuit': '"QCircuit(...)"', 'device': 'IBMDevice.AER_SIMULATOR', 'measure': None, 'remote_id': None, 'status': None, 'created_at': '...'}
        job: {'id': 6, 'type': 'STATE_VECTOR', 'circuit': '"QCircuit(...)"', 'device': 'IBMDevice.AER_SIMULATOR', 'measure': None, 'remote_id': None, 'status': None, 'created_at': '...'}


    """
    from sqlite3 import Row, connect

    connection = connect(get_env_variable("DB_PATH"))
    try:
        connection.row_factory = Row
        cursor = connection.cursor()
        try:
            job_filters = []
            params = []

            if isinstance(result, Result):
                result = [result]

            for res in result:
                data_json = json.dumps(
                    repr(res._data)  # pyright: ignore[reportPrivateUsage]
                )
                error_json = (
                    json.dumps(repr(res.error)) if res.error is not None else None
                )

                job_filters.append(
                    """
                    (results.data is ? AND results.error is ? AND results.shots is ?)
                """
                )
                params.extend(
                    [
                        data_json,
                        error_json,
                        res.shots,
                    ]
                )

            query = f"""
                SELECT jobs.* FROM jobs
                INNER JOIN results ON jobs.id is results.job_id
                WHERE {' OR '.join(job_filters)}
            """

            cursor.execute(query, params)
            jobs = cursor.fetchall()
        finally:
            cursor.close()
    finally:
        connection.close()
    return [dict(job) for job in jobs]


@ensure_local_storage
def fetch_results_with_result_and_job(
    result: Result | BatchResult | list[Result],
) -> list[DictDB]:
    """Fetch result(s) associated with specific results(s) attributes:

    - data,
    - error,
    - shots.

    And also with the job attribute of the results(s):

    - job type,
    - circuit,
    - device,
    - measure.

    Args:
        result: The Result(s) to match.

    Returns:
        Matching result(s) as dictionaries corresponding to the result(s) attribute and job attribute of the result(s).

    Examples:
        >>> result = Result(Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, label="circuit 1"), IBMDevice.AER_SIMULATOR), StateVector([1, 0, 0, 0]), 0, 0)
        >>> results = fetch_results_with_result_and_job(result)
        >>> for result in results:
        ...    print("result:", result) # doctest: +ELLIPSIS
        result: {'id': 6, 'job_id': 5, 'data': '"StateVector([1, 0, 0, 0])"', 'error': '"0"', 'shots': 0, 'created_at': '...'}

    """
    from sqlite3 import Row, connect

    connection = connect(get_env_variable("DB_PATH"))
    try:
        connection.row_factory = Row
        cursor = connection.cursor()
        try:
            result_filters = []
            params = []

            if isinstance(result, Result):
                result = [result]

            for res in result:
                data_json = json.dumps(
                    repr(res._data)  # pyright: ignore[reportPrivateUsage]
                )
                error_json = (
                    json.dumps(repr(res.error)) if res.error is not None else None
                )
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
        finally:
            cursor.close()
    finally:
        connection.close()
    return [dict(result) for result in results]


@ensure_local_storage
def fetch_results_with_job(jobs: Job | list[Job]) -> list[DictDB]:
    """Fetch result(s) associated with the job attribute of the results(s)

    - job type,
    - circuit,
    - device,
    - measure.

    Args:
        jobs: The job(s) to match.

    Returns:
        Matching result(s) as dictionaries corresponding to job attribute of the result(s).

    Examples:
        >>> job = Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, label="circuit 1"), IBMDevice.AER_SIMULATOR)
        >>> results = fetch_results_with_job(job)
        >>> for result in results:
        ...    print("result:", result) # doctest: +ELLIPSIS
        result: {'id': 6, 'job_id': 5, 'data': '"StateVector([1, 0, 0, 0])"', 'error': '"0"', 'shots': 0, 'created_at': '...'}

    """
    from sqlite3 import Row, connect

    connection = connect(get_env_variable("DB_PATH"))
    try:
        connection.row_factory = Row
        cursor = connection.cursor()
        try:

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
            jobs_local_storage = cursor.fetchall()
        finally:
            cursor.close()
    finally:
        connection.close()
    return [dict(job) for job in jobs_local_storage]


@ensure_local_storage
def fetch_results_with_result(
    result: Result | BatchResult | list[Result],
) -> list[DictDB]:
    """Fetch result(s) matching specific results(s) attributes:

    - data,
    - error,
    - shots.

    Args:
        result: The result(s) to match.

    Returns:
        Matching result(s) as dictionaries corresponding to the result(s) attribute.

    Examples:
        >>> result = Result(Job(JobType.STATE_VECTOR, QCircuit([], nb_qubits=2, label="circuit 1"), IBMDevice.AER_SIMULATOR), StateVector([1, 0, 0, 0]), 0, 0)
        >>> results = fetch_results_with_result(result)
        >>> for result in results:
        ...    print("result:", result) # doctest: +ELLIPSIS
        result: {'id': 6, 'job_id': 5, 'data': '"StateVector([1, 0, 0, 0])"', 'error': '"0"', 'shots': 0, 'created_at': '...'}
        result: {'id': 7, 'job_id': 6, 'data': '"StateVector([1, 0, 0, 0])"', 'error': '"0"', 'shots': 0, 'created_at': '...'}

    """
    from sqlite3 import Row, connect

    connection = connect(get_env_variable("DB_PATH"))
    try:
        connection.row_factory = Row
        cursor = connection.cursor()
        try:

            result_filters = []
            params = []

            if isinstance(result, Result):
                result = [result]

            for res in result:
                data_json = json.dumps(
                    repr(res._data)  # pyright: ignore[reportPrivateUsage]
                )
                error_json = (
                    json.dumps(repr(res.error)) if res.error is not None else None
                )

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
        finally:
            cursor.close()
    finally:
        connection.close()

    return [dict(result) for result in results]
