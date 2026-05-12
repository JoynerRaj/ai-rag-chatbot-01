"""
db.py

SQLite helpers for the audio event store.
The database lives in the data/ folder so it isn't mixed in with source code.
"""

import sqlite3
import os

# Resolve the path relative to this file so it works regardless of the working directory.
# audio/db.py → ../data/audio_events.db → fastapi_service/data/audio_events.db
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "audio_events.db")

def init_db():
    """Initializes the SQLite database and creates the events table if it doesn't exist."""
    # Ensure the parent directory exists — required on Render and fresh environments
    # where the data/ folder may not be present yet.
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event TEXT,
            start_time TEXT,
            end_time TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()


def insert_events(events: list[dict]):
    """Inserts a list of event dictionaries into the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for e in events:
        cursor.execute(
            "INSERT INTO events (event, start_time, end_time, confidence) VALUES (?, ?, ?, ?)",
            (e["event"], e["start_time"], e["end_time"], e["confidence"])
        )
    conn.commit()
    conn.close()

def execute_query(sql_query: str) -> list[tuple]:
    """Executes a read-only SQL query and returns the results."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query)
        results = cursor.fetchall()
        return results
    except Exception as e:
        print(f"Error executing SQL: {e}")
        return []
    finally:
        conn.close()
