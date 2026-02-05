"""API smoke tests - verify endpoints work without requiring LLM keys."""
import os
import sys
import tempfile
import pytest

# Ensure src is importable
_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.join(os.path.dirname(_here), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


def test_imports_succeed():
    """Critical: verify all modules can be imported (catches missing deps like autogen)."""
    # These imports would fail if dependencies are missing
    from cognitionflow import config
    from cognitionflow import orchestration
    from cognitionflow import agents  # This imports autogen
    from api import main
    from api import db


def test_health_endpoint():
    """Health endpoint returns 200 with expected structure."""
    from fastapi.testclient import TestClient
    from api.main import app
    
    client = TestClient(app)
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "concurrent_limit" in data


def test_config_endpoint():
    """Config endpoint returns valid structure with all required fields."""
    from fastapi.testclient import TestClient
    from api.main import app
    
    client = TestClient(app)
    response = client.get("/config")
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify structure
    assert "models" in data
    assert "agent_modes" in data
    assert "task_templates" in data
    assert "output_formats" in data
    assert "defaults" in data
    
    # Verify we have options
    assert len(data["models"]) > 0
    assert len(data["task_templates"]) > 0


def test_index_serves_html():
    """Index route serves the frontend HTML."""
    from fastapi.testclient import TestClient
    from api.main import app
    
    client = TestClient(app)
    response = client.get("/")
    
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


def test_nonexistent_run_returns_404():
    """Requesting a non-existent run returns 404."""
    from fastapi.testclient import TestClient
    from api.main import app
    
    client = TestClient(app)
    response = client.get("/runs/nonexistent-run-id")
    
    assert response.status_code == 404
