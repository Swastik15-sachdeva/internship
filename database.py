"""
database.py — SQLite database setup and operations
"""

import sqlite3
import pickle
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "recruiter.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            name    TEXT    NOT NULL UNIQUE,
            encoding BLOB   NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate   TEXT    NOT NULL,
            event       TEXT    NOT NULL,
            timestamp   TEXT    NOT NULL,
            image_path  TEXT,
            risk_points INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def save_candidate(name: str, encoding) -> bool:
    """Store a candidate's face encoding. Returns True on success."""
    try:
        conn = get_connection()
        blob = pickle.dumps(encoding)
        conn.execute(
            "INSERT OR REPLACE INTO candidates (name, encoding) VALUES (?, ?)",
            (name, blob)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[DB] Error saving candidate: {e}")
        return False


def load_candidate(name: str):
    """Load a candidate's face encoding. Returns numpy array or None."""
    try:
        conn = get_connection()
        row = conn.execute(
            "SELECT encoding FROM candidates WHERE name = ?", (name,)
        ).fetchone()
        conn.close()
        if row:
            return pickle.loads(row["encoding"])
        return None
    except Exception as e:
        print(f"[DB] Error loading candidate: {e}")
        return None


def list_candidates():
    """Return list of all candidate names."""
    conn = get_connection()
    rows = conn.execute("SELECT name, created_at FROM candidates ORDER BY name").fetchall()
    conn.close()
    return [(r["name"], r["created_at"]) for r in rows]


def log_violation(candidate: str, event: str, image_path: str = None, risk_points: int = 0):
    """Log a single violation event."""
    try:
        conn = get_connection()
        ts = datetime.now().strftime("%H:%M:%S")
        conn.execute(
            "INSERT INTO violations (candidate, event, timestamp, image_path, risk_points) VALUES (?,?,?,?,?)",
            (candidate, event, ts, image_path, risk_points)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Error logging violation: {e}")


def get_violations(candidate: str):
    """Return all violations for a candidate as list of dicts."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM violations WHERE candidate = ? ORDER BY timestamp",
        (candidate,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_risk_score(candidate: str) -> int:
    """Sum all risk_points for a candidate."""
    conn = get_connection()
    row = conn.execute(
        "SELECT COALESCE(SUM(risk_points), 0) AS total FROM violations WHERE candidate = ?",
        (candidate,)
    ).fetchone()
    conn.close()
    return int(row["total"])


def clear_violations(candidate: str):
    """Clear violations for a fresh session."""
    conn = get_connection()
    conn.execute("DELETE FROM violations WHERE candidate = ?", (candidate,))
    conn.commit()
    conn.close()
