"""This module provides utilities for managing a SQLite database for quantum job
and result records, as well as functions for removing entries based on various
criteria.

It allows storing and managing job and result metadata related to quantum
circuit executions.
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

from mpqp.environment.env_manager import get_env_variable, save_env_variable
from mpqp.tools.generics import T

DictDB = dict[str, Any]
DATABASE_VERSION = "1.1"


def get_database_version() -> str:
    """Retrieves the current database version from the version table."""
    from sqlite3 import connect

    connection = connect(get_env_variable("DB_PATH"))
    try:
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT version FROM version WHERE id = 1")
            result = cursor.fetchone()
            return str(result[0]) if result else "0.0"
        finally:
            cursor.close()
    finally:
        connection.close()


def _ensure_database_version() -> None:
    """Upgrade supported schemas and reject incompatible database versions."""
    from sqlite3 import connect

    db_version = get_database_version()
    if db_version == "1.0":
        connection = connect(get_env_variable("DB_PATH"))
        try:
            with connection:
                # Serialize upgrades and recheck after acquiring the write lock.
                connection.execute("BEGIN IMMEDIATE")
                version = connection.execute(
                    "SELECT version FROM version WHERE id = 1"
                ).fetchone()
                if version is not None and version[0] == "1.0":
                    connection.execute("ALTER TABLE results ADD COLUMN label TEXT")
                    connection.execute(
                        "UPDATE version SET version = ? WHERE id = 1",
                        (DATABASE_VERSION,),
                    )
        finally:
            connection.close()
        db_version = get_database_version()
    if db_version != DATABASE_VERSION:
        raise RuntimeError(
            f"Database version {db_version} is not supported. "
            f"Current supported version: {DATABASE_VERSION}. "
            "No automatic migration is available for this version; "
            "please contact library authors for help."
        )


def ensure_local_storage(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator for functions needing the database to be present.

    Args:
        func: The function to be decorated."""

    @wraps(func)  # This keeps the original function's docstring for test
    def wrapper(*args: Any, **kwargs: dict[str, Any]) -> T:
        if get_env_variable("DB_PATH") == "":
            setup_local_storage()

        _ensure_database_version()

        return func(*args, **kwargs)

    return wrapper


def setup_local_storage(path: Optional[str] = None):
    """Sets up a SQLite database for storing quantum job and result records.

    Two tables will be created, one for jobs and the other for results.

    Args:
        path: Directory to save the database file. Defaults to the current working directory.

    Raises:
        RuntimeError: If the database version has no supported upgrade path.

    Example:
        >>> setup_local_storage("~/documents/my_database.db")
        >>> os.remove(Path("~/documents/my_database.db").expanduser())
        >>> setup_local_storage("my_database.db")
        >>> os.remove("my_database.db")
        >>> setup_local_storage()

    Note:
        Existing version 1.0 databases are upgraded automatically, preserving
        stored jobs and results. Older results retain their circuit's display
        label until a separate result label is assigned.

    """
    from sqlite3 import connect

    if path is not None:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(exist_ok=True)
        path = str(p)
    else:
        path = str(Path("~/.mpqp/result_storage.db").expanduser().resolve())
    save_env_variable("DB_PATH", path)
    connection = connect(path)
    cursor = connection.cursor()

    # Create the jobs table
    cursor.execute('''
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
    ''')

    # Create the results table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id INTEGER NOT NULL,
        data TEXT NOT NULL,            -- Store JSON as text
        error TEXT,                    -- Optional, stored as JSON text
        shots INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        label TEXT
    )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS version (
            id INTEGER PRIMARY KEY CHECK (id = 1),  -- Ensures only one row exists
            version VARCHAR
        )
        ''')

    cursor.execute(
        "INSERT OR IGNORE INTO version (id, version) VALUES (1, ?)", (DATABASE_VERSION,)
    )

    cursor.close()
    connection.commit()
    connection.close()
    _ensure_database_version()
