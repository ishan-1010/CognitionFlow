"""
Lightweight SQLite-based run history for CognitionFlow.
Designed for minimal memory footprint on Render free tier.
"""
import os
import json
import sqlite3
from datetime import datetime
from typing import Optional
from contextlib import contextmanager


DB_PATH = os.environ.get("COGNITIONFLOW_DB", "data/runs.db")


def _ensure_db_dir():
    """Ensure the database directory exists."""
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)


@contextmanager
def get_db():
    """Context manager for database connection."""
    _ensure_db_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Initialize the runs table."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                config TEXT,
                started_at TEXT,
                completed_at TEXT,
                duration_ms INTEGER,
                artifact_report TEXT,
                artifact_plot TEXT,
                error TEXT
            )
        """)
        conn.commit()


def save_run(
    run_id: str,
    status: str,
    config: dict | None = None,
    started_at: str | None = None,
    completed_at: str | None = None,
    duration_ms: int | None = None,
    artifact_report: str | None = None,
    artifact_plot: str | None = None,
    error: str | None = None,
):
    """Save or update a run record."""
    with get_db() as conn:
        conn.execute("""
            INSERT INTO runs (id, status, config, started_at, completed_at, duration_ms, artifact_report, artifact_plot, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status = excluded.status,
                completed_at = excluded.completed_at,
                duration_ms = excluded.duration_ms,
                artifact_report = excluded.artifact_report,
                artifact_plot = excluded.artifact_plot,
                error = excluded.error
        """, (
            run_id,
            status,
            json.dumps(config) if config else None,
            started_at,
            completed_at,
            duration_ms,
            artifact_report,
            artifact_plot,
            error,
        ))
        conn.commit()



def get_run_history(limit: int = 20, offset: int = 0) -> list[dict]:
    """Get recent runs with pagination."""
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_run_by_id(run_id: str) -> Optional[dict]:
    """Get a specific run by ID."""
    with get_db() as conn:
        cursor = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
    return None



def get_metrics() -> dict:
    """Get aggregate metrics for all runs."""
    with get_db() as conn:
        cursor = conn.execute("""
            SELECT 
                COUNT(*) as total_runs,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                AVG(duration_ms) as avg_duration_ms
            FROM runs
        """)
        row = cursor.fetchone()
        total = row["total_runs"] or 0
        successful = row["successful"] or 0
        
        return {
            "total_runs": total,
            "successful": successful,
            "failed": row["failed"] or 0,
            "success_rate": round(successful / total * 100, 1) if total > 0 else 0,
            "avg_duration_ms": round(row["avg_duration_ms"] or 0),
        }


# Initialize DB on import
init_db()
