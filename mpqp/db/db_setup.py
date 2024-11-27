from __future__ import annotations

from typing import Any, Optional

from mpqp.db.db_query import fetch_jobs_with_job, fetch_results_with_result
from mpqp.execution.connection.env_manager import get_env_variable, save_env_variable
from mpqp.execution.job import Job
from mpqp.execution.result import BatchResult, Result


def setup_db(data_base_name: Optional[str] = None, path: Optional[str] = None):
    import sqlite3
    import os

    if data_base_name is None:
        data_base_name = 'quantum_jobs.db'

    if path:
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
    import sqlite3

    database_name = get_env_variable("DATA_BASE")
    connection = sqlite3.connect(database_name)
    cursor = connection.cursor()

    cursor.execute('DELETE FROM results')
    cursor.execute('DELETE FROM jobs')
    cursor.execute('DELETE FROM sqlite_sequence WHERE name IN ("results", "jobs")')

    connection.commit()
    connection.close()


def remove_all_with_job_id(job_id: int | list[int]):
    import sqlite3

    database_name = get_env_variable("DATA_BASE")
    connection = sqlite3.connect(database_name)
    cursor = connection.cursor()
    if isinstance(job_id, int):
        cursor.execute('DELETE FROM results WHERE job_id is ?', (job_id,))
        cursor.execute('DELETE FROM jobs WHERE id is ?', (job_id,))
        connection.commit()
        connection.close()
    else:
        cursor.executemany(
            'DELETE FROM results WHERE job_id is ?', [(id,) for id in job_id]
        )
        cursor.executemany('DELETE FROM jobs WHERE id is ?', [(id,) for id in job_id])
        connection.commit()
        connection.close()


def remove_jobs_with_id(job_id: int | list[int]):
    import sqlite3

    database_name = get_env_variable("DATA_BASE")
    connection = sqlite3.connect(database_name)
    cursor = connection.cursor()
    if isinstance(job_id, int):
        cursor.execute('DELETE FROM jobs WHERE id is ?', (job_id,))
        connection.commit()
        connection.close()
    else:
        cursor.executemany('DELETE FROM jobs WHERE id is ?', [(id,) for id in job_id])
        connection.commit()
        connection.close()


def remove_results_with_id(result_id: int | list[int]):
    import sqlite3

    database_name = get_env_variable("DATA_BASE")
    connection = sqlite3.connect(database_name)
    cursor = connection.cursor()
    if isinstance(result_id, int):
        cursor.execute('DELETE FROM results WHERE id is ?', (result_id,))
        connection.commit()
        connection.close()
    else:
        cursor.executemany(
            'DELETE FROM results WHERE id is ?', [(id,) for id in result_id]
        )
        connection.commit()
        connection.close()


def remove_results_with_results_db(
    results: Optional[list[dict[Any, Any]] | dict[Any, Any]]
):
    if results is None:
        return
    if isinstance(results, dict):
        remove_results_with_id(results['id'])
    else:
        results_id = [result['id'] for result in results]
        remove_results_with_id(results_id)


def remove_jobs_with_jobs_db(jobs: Optional[list[dict[Any, Any]] | dict[Any, Any]]):
    if jobs is None:
        return
    if isinstance(jobs, dict):
        remove_jobs_with_id(jobs['id'])
    else:
        jobs_id = [job['id'] for job in jobs]
        remove_jobs_with_id(jobs_id)


def remove_results_with_result(result: Result | BatchResult | list[Result]):
    results_db = fetch_results_with_result(result)
    remove_results_with_results_db(results_db)


def remove_results_with_job(jobs: Job | list[Job]):
    jobs_db = fetch_jobs_with_job(jobs)
    remove_jobs_with_jobs_db(jobs_db)
