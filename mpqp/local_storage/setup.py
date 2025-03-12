"""This module provides utilities for managing a SQLite database for quantum job
and result records, as well as functions for removing entries based on various criteria.

It allows storing and managing job and result metadata related to quantum circuit executions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional
from functools import wraps

from mpqp.execution.connection.env_manager import get_env_variable, save_env_variable
from mpqp.tools.generics import T

DictDB = dict[str, Any]
DATABASE_VERSION = 1.0


def get_database_version() -> float:
    """Retrieves the current database version from the version table."""
    from sqlite3 import connect

    with connect(get_env_variable("DATA_BASE")) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT version FROM version WHERE id = 1")
        result = cursor.fetchone()
        return float(result[0]) if result else 0.0


def ensure_local_storage(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator for functions needing the setup_local to be present"""

    @wraps(func)  # This keeps the original function's docstring for test
    def wrapper(*args: Any, **kwargs: dict[str, Any]) -> T:
        if get_env_variable("DATA_BASE") == "":
            setup_local_storage()

        db_version = get_database_version()
        if db_version != DATABASE_VERSION:
            raise RuntimeError(
                f"Database version {db_version} is outdated. Expected {DATABASE_VERSION}. "
                "Please upgrade the database schema."
            )

        return func(*args, **kwargs)

    return wrapper


def setup_local_storage(path: Optional[str] = None):
    """Sets up a SQLite database for storing quantum job and result records.

    Two tables will be created:

        - :class:`~mpqp.execution.job.Job`
        - :class:`~mpqp.execution.result.Result`

    Args:
        path: Directory to save the database file. Defaults to the current working directory.

    Example:
        >>> setup_local_storage("~/documents/my_database.db")
        >>> os.remove(Path("~/documents/my_database.db").expanduser())
        >>> setup_local_storage("my_database.db")
        >>> os.remove("my_database.db")
        >>> setup_local_storage()

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
        remote_id TEXT, 
        status TEXT,
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

    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS version (
            id INTEGER PRIMARY KEY CHECK (id = 1),  -- Ensures only one row exists
            version FLOAT NOT NULL
        )
        '''
    )

    cursor.execute(
        "INSERT OR IGNORE INTO version (id, version) VALUES (1, ?)", (DATABASE_VERSION,)
    )

    connection.commit()
    connection.close()
