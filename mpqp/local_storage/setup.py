"""This module provides utilities for managing a SQLite database for quantum job 
and result records, as well as functions for removing entries based on various criteria.

It allows storing and managing job and result metadata related to quantum circuit executions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from mpqp.execution.connection.env_manager import get_env_variable, save_env_variable
from mpqp.execution.job import Job
from mpqp.execution.result import BatchResult, Result
from mpqp.tools.generics import T

DictDB = dict[str, Any]


def ensure_db(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator for functions needing the db to be present"""

    def wrapper(*args: Any, **kwargs: dict[str, Any]) -> T:
        if get_env_variable("DATA_BASE") == "":
            setup_db()

        return func(*args, **kwargs)

    return wrapper


def setup_db(path: Optional[str] = None):
    """Sets up a SQLite database for storing quantum job and result records.

    Two tables will be created:

        - `jobs`
        - `results`

    Args:
        path: Directory to save the database file. Defaults to the current working directory.

    Example:
        >>> setup_db("~/documents/my_database.db")
        >>> os.remove(Path("~/documents/my_database.db").expanduser())
        >>> setup_db("my_database.db")
        >>> os.remove("my_database.db")
        >>> setup_db()

    """
    import sqlite3

    if path is not None:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(exist_ok=True)
        path = str(p)
    else:
        path = str(Path("~/.mpqp/result_storage.db").expanduser().resolve())
    save_env_variable("DATA_BASE", path)
    connection = sqlite3.connect(path)
    cursor = connection.cursor()

    # Create the jobs table
    cursor.execute(
        '''
    CREATE TABLE IF NOT EXISTS jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        type TEXT NOT NULL,
        circuit TEXT NOT NULL,         -- Store JSON as text
        device TEXT NOT NULL,
        measure TEXT,                  -- Optional, stored as JSON text
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    '''
    )

    # Create the results table
    cursor.execute(
        '''
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER NOT NULL,
        data TEXT NOT NULL,            -- Store JSON as text
        error TEXT,                    -- Optional, stored as JSON text
        shots INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    '''
    )

    connection.commit()
    connection.close()


def clear_db():
    """Clears all records from the database, including jobs and results.

    This function resets the tables and their auto-increment counters.

    Example:
        >>> clear_db()
        >>> fetch_all_results()
        []
        >>> fetch_all_jobs()
        []

    """
    from sqlite3 import connect

    with connect(get_env_variable("DATA_BASE")) as connection:
        cursor = connection.cursor()

        cursor.execute('DELETE FROM results')
        cursor.execute('DELETE FROM jobs')
        cursor.execute('DELETE FROM sqlite_sequence WHERE name IN ("results", "jobs")')

        connection.commit()


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

    with connect(get_env_variable("DATA_BASE")) as connection:
        cursor = connection.cursor()
        if isinstance(job_id, int):
            cursor.execute('DELETE FROM jobs WHERE id is ?', (job_id,))
        else:
            cursor.executemany(
                'DELETE FROM jobs WHERE id is ?', [(id,) for id in job_id]
            )
        connection.commit()


def remove_results_with_id(result_id: int | list[int]):
    """Removes results with the specified result IDs.

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

    with connect(get_env_variable("DATA_BASE")) as connection:
        cursor = connection.cursor()
        if isinstance(result_id, int):
            cursor.execute('DELETE FROM results WHERE id is ?', (result_id,))
            connection.commit()
        else:
            cursor.executemany(
                'DELETE FROM results WHERE id is ?', [(id,) for id in result_id]
            )
            connection.commit()


def remove_results_with_results_db(results: Optional[list[DictDB] | DictDB]):
    """Removes the matching results.

    Args:
        results: Result dictionary(ies) for which the matching database row
            should be deleted.

    Example:
        >>> results = fetch_results_with_id(1)
        >>> remove_results_with_results_db(results)
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


def remove_jobs_with_jobs_db(jobs: Optional[list[DictDB] | DictDB]):
    """Removes the matching jobs.

    Args:
        jobs: Job dictionary(ies) for which the matching database row should be
            deleted.

    Example:
        >>> jobs = fetch_jobs_with_id(1)
        >>> remove_jobs_with_jobs_db(jobs)
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

    results_db = fetch_results_with_result(result)
    remove_results_with_results_db(results_db)


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

    with connect(get_env_variable("DATA_BASE")) as connection:
        cursor = connection.cursor()
        if isinstance(job_id, int):
            cursor.execute('DELETE FROM results WHERE job_id is ?', (job_id,))
            connection.commit()
        else:
            cursor.executemany(
                'DELETE FROM results WHERE job_id is ?', [(id,) for id in job_id]
            )
            connection.commit()


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

    jobs_db = fetch_jobs_with_job(jobs)
    remove_results_with_job_id([job['id'] for job in jobs_db])
