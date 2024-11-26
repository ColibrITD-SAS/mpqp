import json

from mpqp.execution.connection.env_manager import get_env_variable
from mpqp.execution.job import Job
from mpqp.execution.result import Result, Sample, StateVector


def fetch_all_jobs():
    from sqlite3 import connect

    with connect(get_env_variable("DATA_BASE")) as connection:
        cursor = connection.cursor()
        cursor.execute('SELECT * FROM jobs')
        return cursor.fetchall()


def fetch_all_results():
    from sqlite3 import connect

    with connect(get_env_variable("DATA_BASE")) as connection:
        cursor = connection.cursor()

        cursor.execute('SELECT * FROM results')
        return cursor.fetchall()


def fetch_job_with_job(job: Job):
    from sqlite3 import connect

    with connect(get_env_variable("DATA_BASE")) as connection:
        cursor = connection.cursor()

        circuit_json = json.dumps(job.circuit.to_dict())
        measure_json = json.dumps(job.measure.to_dict()) if job.measure else None
        cursor.execute(
            '''
            SELECT * FROM jobs
            WHERE type is ? AND circuit is ? AND device is ? AND measure is ?
            ''',
            (str(job.job_type), circuit_json, str(job.device), measure_json),
        )

        return cursor.fetchall()


def fetch_job_with_results(result: Result):
    from sqlite3 import connect

    with connect(get_env_variable("DATA_BASE")) as connection:
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

        circuit_json = json.dumps(result.job.circuit.to_dict())
        measure_json = (
            json.dumps(result.job.measure.to_dict()) if result.job.measure else None
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
                str(result.job.job_type),
                circuit_json,
                str(result.job.device),
                measure_json,
            ),
        )

        return cursor.fetchall()


def fetch_results_with_results_and_job(result: Result):
    from sqlite3 import connect

    with connect(get_env_variable("DATA_BASE")) as connection:
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

        circuit_json = json.dumps(result.job.circuit.to_dict())
        measure_json = (
            json.dumps(result.job.measure.to_dict()) if result.job.measure else None
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
                str(result.job.job_type),
                circuit_json,
                str(result.job.device),
                measure_json,
            ),
        )

        return cursor.fetchall()


def fetch_results_with_results(result: Result):
    from sqlite3 import connect

    with connect(get_env_variable("DATA_BASE")) as connection:
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
            SELECT * FROM results
            WHERE data is ? AND error is ? AND shots is ?
            ''',
            (
                data_json,
                error_json,
                result.shots,
            ),
        )

        return cursor.fetchall()


def fetch_results_with_results_id(result_id: int):
    from sqlite3 import connect

    with connect(get_env_variable("DATA_BASE")) as connection:
        cursor = connection.cursor()

        cursor.execute(
            '''
            SELECT * FROM results
            WHERE result_id is ?
            ''',
            (result_id,),
        )

        return cursor.fetchall()


def fetch_job_with_job_id(job_id: int):
    from sqlite3 import connect

    with connect(get_env_variable("DATA_BASE")) as connection:
        cursor = connection.cursor()

        cursor.execute(
            '''
            SELECT * FROM jobs
            WHERE job_id is ?
            ''',
            (job_id,),
        )

        return cursor.fetchall()
