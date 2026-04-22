import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "audio_events.db")

def init_db():
    """Initializes the SQLite database and creates the events table if it doesn't exist."""
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
