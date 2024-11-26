from typing import Optional

from mpqp.execution.connection.env_manager import get_env_variable, save_env_variable


def setup_db(data_base_name: Optional[str] = 'quantum_jobs.db'):
    import sqlite3

    if data_base_name is None:
        save_env_variable("DATA_BASE", 'quantum_jobs.db')
    else:
        save_env_variable("DATA_BASE", data_base_name)

    database_name = get_env_variable("DATA_BASE")
    connection = sqlite3.connect(database_name)
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
        status TEXT DEFAULT 'INIT',
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
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
    )
    '''
    )

    connection.commit()
    connection.close()
