"""
This module provides utilities for managing a SQLite database for quantum job 
and result records, as well as functions for removing entries based on various criteria.

It allows storing and managing job and result metadata related to quantum circuit executions.

Classes and Functions:
- `setup_db`: Initializes the database with the necessary tables.
- `clear_db`: Clears all records from the database.
- `remove_all_with_job_id`: Removes all jobs and associated results for specific job IDs.
- `remove_jobs_with_id`: Removes jobs with specific IDs.
- `remove_results_with_id`: Removes results with specific IDs.
- `remove_results_with_results_db`: Removes results using a dictionary or list of dictionaries from the database.
- `remove_jobs_with_jobs_db`: Removes jobs using a dictionary or list of dictionaries from the database.
- `remove_results_with_result`: Removes results associated with a specific `Result` or `BatchResult`.
- `remove_results_with_job`: Removes results associated with specific `Job` objects.

"""

from __future__ import annotations

from typing import Any, Optional

from mpqp.db.db_query import fetch_jobs_with_job, fetch_results_with_result
from mpqp.execution.connection.env_manager import get_env_variable, save_env_variable
from mpqp.execution.job import Job
from mpqp.execution.result import BatchResult, Result


DictDB = dict[str, Any]


def setup_db(data_base_name: Optional[str] = None, path: Optional[str] = None):
    """
    Sets up a SQLite database for storing quantum job and result records.
    Creates, two tables:
        - `jobs`: Stores metadata about quantum jobs (e.g., type, circuit, device).
        - `results`: Stores metadata about results of quantum jobs (e.g., data, errors, shots).

    Args:
        data_base_name: Name of the database file. Defaults to 'quantum_jobs.db'.
        path: Directory to save the database file. Defaults to the current working directory.

    Example:
        >>> setup_db("my_database.db")

    """
    import sqlite3
    import os

    if data_base_name is None:
        data_base_name = 'quantum_jobs.db'
    else:
        if not data_base_name.endswith(".db"):
            data_base_name += ".db"

    if path is not None:
        os.makedirs(path, exist_ok=True)
        path_name = os.path.join(path, data_base_name)
    else:
        path_name = os.path.join(os.getcwd(), data_base_name)

    save_env_variable("DATA_BASE", path_name)
    connection = sqlite3.connect(path_name)
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
    """
    Clears all records from the database, including jobs and results.

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
    """
    Removes jobs and their associated results for the specified job IDs.

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
    """
    Removes jobs with the specified job IDs.

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
    """
    Removes results with the specified result IDs.

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
    """
    Removes results using result(s) from the database.

    Args:
        results: Result dictionary or list of dictionaries from the database.

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
    """
    Removes jobs using a dictionary or list of dictionaries from the database.

    Args:
        jobs: Job dictionary or list of dictionaries from the database.

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
    """
    Removes results associated with specific `Result` or `BatchResult` objects.

    Args:
        result: Result(s) to remove.

    Example:
        >>> result = Result(Job(JobType.STATE_VECTOR, QCircuit(2), IBMDevice.AER_SIMULATOR), StateVector([1, 0, 0, 0]))
        >>> remove_results_with_result(result)
        >>> fetch_results_with_result(result)
        []

    """
    results_db = fetch_results_with_result(result)
    remove_results_with_results_db(results_db)


def remove_results_with_job_id(job_id: int | list[int]):
    """
    Removes results with the specified result IDs.

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
    """
    Removes results associated with specific `Job` objects.

    Args:
        jobs: Job(s) to remove results for.

    Example:
        >>> job = Job(JobType.STATE_VECTOR, QCircuit(2), IBMDevice.AER_SIMULATOR)
        >>> remove_results_with_job(job)
        >>> fetch_results_with_job(job)
        []

    """
    jobs_db = fetch_jobs_with_job(jobs)
    remove_results_with_job_id([job['id'] for job in jobs_db])
