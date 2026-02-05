"""Database tests - verify SQLite operations work correctly."""
import os
import tempfile
import pytest


@pytest.fixture
def temp_db(monkeypatch):
    """Use a temporary database for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_runs.db")
        monkeypatch.setenv("COGNITIONFLOW_DB", db_path)
        
        # Force reimport to use new DB path
        import importlib
        from api import db
        importlib.reload(db)
        
        yield db_path


def test_db_init_creates_table(temp_db):
    """Database initializes and creates runs table."""
    from api.db import get_db
    
    with get_db() as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
        )
        assert cursor.fetchone() is not None


def test_save_and_retrieve_run(temp_db):
    """Can save a run and retrieve it by ID."""
    from api.db import save_run, get_run_by_id
    
    run_id = "test-run-123"
    save_run(
        run_id=run_id,
        status="completed",
        config={"model": "test-model"},
        started_at="2024-01-01T00:00:00Z",
        completed_at="2024-01-01T00:01:00Z",
        duration_ms=60000,
    )
    
    result = get_run_by_id(run_id)
    
    assert result is not None
    assert result["id"] == run_id
    assert result["status"] == "completed"
    assert result["duration_ms"] == 60000


def test_run_history_pagination(temp_db):
    """Run history supports pagination."""
    from api.db import save_run, get_run_history
    
    # Create multiple runs
    for i in range(5):
        save_run(
            run_id=f"run-{i}",
            status="completed",
            started_at=f"2024-01-0{i+1}T00:00:00Z",
        )
    
    # Get first page
    history = get_run_history(limit=2, offset=0)
    assert len(history) == 2
    
    # Get second page
    history = get_run_history(limit=2, offset=2)
    assert len(history) == 2


def test_metrics_calculation(temp_db):
    """Metrics are calculated correctly."""
    from api.db import save_run, get_metrics
    
    # Create some runs
    save_run(run_id="success-1", status="completed", duration_ms=1000)
    save_run(run_id="success-2", status="completed", duration_ms=2000)
    save_run(run_id="fail-1", status="failed")
    
    metrics = get_metrics()
    
    assert metrics["total_runs"] == 3
    assert metrics["successful"] == 2
    assert metrics["failed"] == 1
    assert metrics["success_rate"] == pytest.approx(66.7, rel=0.1)


def test_nonexistent_run_returns_none(temp_db):
    """Querying nonexistent run returns None, not error."""
    from api.db import get_run_by_id
    
    result = get_run_by_id("does-not-exist")
    assert result is None
