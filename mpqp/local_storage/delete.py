from __future__ import annotations

from typing import Optional

from mpqp.execution.connection.env_manager import get_env_variable
from mpqp.execution.job import Job
from mpqp.execution.result import BatchResult, Result
from mpqp.local_storage.setup import DictDB

delete = 0


def clear_local_storage():
    """Clears all records from the database, including jobs and results.

    This function resets the tables and their auto-increment counters.

    Example:
        >>> clear_local_storage()
        >>> fetch_all_results()
        []
        >>> fetch_all_jobs()
        []

    """
    from sqlite3 import connect

    connection = connect(get_env_variable("DB_PATH"))
    try:
        cursor = connection.cursor()
        try:
            cursor.execute('DELETE FROM results')
            cursor.execute('DELETE FROM jobs')
            cursor.execute(
                'DELETE FROM sqlite_sequence WHERE name IN ("results", "jobs")'
            )

            connection.commit()
        finally:
            cursor.close()
    finally:
        connection.close()


def remove_all_with_job_id(job_id: int | list[int]):
    """Removes jobs and their associated results for the specified job IDs.

    Args:
        job_id: Job ID(s) to remove.

    Example:
        >>> remove_all_with_job_id(1)
        >>> fetch_jobs_with_id(1)
        []
        >>> fetch_results_with_job_id(1)
        []
        >>> remove_all_with_job_id([2, 3])
        >>> fetch_jobs_with_id([2, 3])
        []
        >>> fetch_results_with_job_id([2, 3])
        []

    """
    remove_results_with_job_id(job_id)
    remove_jobs_with_id(job_id)


def remove_jobs_with_id(job_id: int | list[int]):
    """Removes jobs with the specified job IDs.

    Method of the class corresponding: :meth:`~mpqp.execution.job.Job.delete_by_local_id`.

    Args:
        job_id: Job ID(s) to remove.

    Example:
        >>> remove_jobs_with_id(1)
        >>> fetch_jobs_with_id(1)
        []
        >>> remove_jobs_with_id([2, 3])
        >>> fetch_jobs_with_id([2,3])
        []

    """
    from sqlite3 import connect

    connection = connect(get_env_variable("DB_PATH"))
    try:
        cursor = connection.cursor()
        try:
            if isinstance(job_id, int):
                cursor.execute('DELETE FROM jobs WHERE id is ?', (job_id,))
            else:
                cursor.executemany(
                    'DELETE FROM jobs WHERE id is ?', [(id,) for id in job_id]
                )
            connection.commit()
        finally:
            cursor.close()
    finally:
        connection.close()


def remove_results_with_id(result_id: int | list[int]):
    """Removes results with the specified result IDs.

    Method of the class corresponding: :meth:`~mpqp.execution.result.Result.delete_by_local_id`.

    Args:
        result_id: Result ID(s) to remove.

    Example:
        >>> remove_results_with_id(1)
        >>> fetch_results_with_id(1)
        []
        >>> remove_results_with_id([2, 3])
        >>> fetch_results_with_id([2, 3])
        []

    """
    from sqlite3 import connect

    connection = connect(get_env_variable("DB_PATH"))
    try:
        cursor = connection.cursor()
        try:
            if isinstance(result_id, int):
                cursor.execute('DELETE FROM results WHERE id is ?', (result_id,))
                connection.commit()
            else:
                cursor.executemany(
                    'DELETE FROM results WHERE id is ?', [(id,) for id in result_id]
                )
                connection.commit()
        finally:
            cursor.close()
    finally:
        connection.close()


def remove_results_with_results_local_storage(results: Optional[list[DictDB] | DictDB]):
    """Removes the matching results.

    Args:
        results: Result dictionary(ies) for which the matching database row
            should be deleted.

    Example:
        >>> results = fetch_results_with_id(1)
        >>> remove_results_with_results_local_storage(results)
        >>> fetch_results_with_id(1)
        []

    """
    if results is None:
        return
    if isinstance(results, dict):
        remove_results_with_id(results['id'])
    else:
        results_id = [result['id'] for result in results]
        remove_results_with_id(results_id)


def remove_jobs_with_jobs_local_storage(jobs: Optional[list[DictDB] | DictDB]):
    """Removes the matching jobs.

    Args:
        jobs: Job dictionary(ies) for which the matching database row should be
            deleted.

    Example:
        >>> jobs = fetch_jobs_with_id(1)
        >>> remove_jobs_with_jobs_local_storage(jobs)
        >>> fetch_jobs_with_id(1)
        []

    """
    if jobs is None:
        return
    if isinstance(jobs, dict):
        remove_jobs_with_id(jobs['id'])
    else:
        jobs_id = [job['id'] for job in jobs]
        remove_jobs_with_id(jobs_id)


def remove_results_with_result(result: Result | BatchResult | list[Result]):
    """Removes results matching the given result(s).

    Args:
        result: Result(s) to remove.

    Example:
        >>> result = Result(Job(JobType.STATE_VECTOR, QCircuit(2), IBMDevice.AER_SIMULATOR), StateVector([1, 0, 0, 0]))
        >>> remove_results_with_result(result)
        >>> fetch_results_with_result(result)
        []

    """
    from mpqp.local_storage.queries import fetch_results_with_result

    results_local_storage = fetch_results_with_result(result)
    remove_results_with_results_local_storage(results_local_storage)


def remove_results_with_job_id(job_id: int | list[int]):
    """Removes results related to the job(s) who's ID is given as input.

    Args:
        job_id: Result Job_ID(s) to remove.

    Example:
        >>> remove_results_with_job_id(1)
        >>> fetch_results_with_job_id(1)
        []
        >>> remove_results_with_job_id([2, 3])
        >>> fetch_results_with_job_id([2, 3])
        []

    """
    from sqlite3 import connect

    connection = connect(get_env_variable("DB_PATH"))
    try:
        cursor = connection.cursor()
        try:
            if isinstance(job_id, int):
                cursor.execute('DELETE FROM results WHERE job_id is ?', (job_id,))
                connection.commit()
            else:
                cursor.executemany(
                    'DELETE FROM results WHERE job_id is ?', [(id,) for id in job_id]
                )
                connection.commit()
        finally:
            cursor.close()
    finally:
        connection.close()


def remove_results_with_job(jobs: Job | list[Job]):
    """Removes results associated with the specified job(s).

    Args:
        jobs: Job(s) to remove results for.

    Example:
        >>> job = Job(JobType.STATE_VECTOR, QCircuit(2), IBMDevice.AER_SIMULATOR)
        >>> remove_results_with_job(job)
        >>> fetch_results_with_job(job)
        []

    """
    from mpqp.local_storage.queries import fetch_jobs_with_job

    jobs_local_storage = fetch_jobs_with_job(jobs)
    remove_results_with_job_id([job['id'] for job in jobs_local_storage])
