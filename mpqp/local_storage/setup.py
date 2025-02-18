"""This module provides utilities for managing a SQLite database for quantum job
and result records, as well as functions for removing entries based on various criteria.

It allows storing and managing job and result metadata related to quantum circuit executions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from mpqp.execution.connection.env_manager import get_env_variable, save_env_variable
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

    connection.commit()
    connection.close()
