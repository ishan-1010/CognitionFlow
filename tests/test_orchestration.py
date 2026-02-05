"""Orchestration tests - verify template lookup and artifact discovery."""
import os
import tempfile
import pytest


def test_get_template_prompt_valid():
    """get_template_prompt returns prompt for valid template ID."""
    from cognitionflow.orchestration import get_template_prompt
    
    prompt = get_template_prompt("data_analysis")
    
    assert prompt is not None
    assert len(prompt) > 0
    assert "data" in prompt.lower() or "analysis" in prompt.lower()


def test_get_template_prompt_invalid_returns_default():
    """get_template_prompt returns default for invalid template ID."""
    from cognitionflow.orchestration import get_template_prompt
    
    prompt = get_template_prompt("nonexistent_template")
    
    # Should return first template as fallback
    assert prompt is not None
    assert len(prompt) > 0


def test_discover_artifacts_finds_files():
    """discover_artifacts finds generated files in work directory."""
    from cognitionflow.orchestration import discover_artifacts
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some test files
        with open(os.path.join(tmpdir, "report.md"), "w") as f:
            f.write("# Report")
        with open(os.path.join(tmpdir, "data.json"), "w") as f:
            f.write("{}")
        with open(os.path.join(tmpdir, "chart.png"), "wb") as f:
            f.write(b"fake png")
        
        artifacts = discover_artifacts(tmpdir)
        
        assert len(artifacts) == 3
        
        names = [a["name"] for a in artifacts]
        assert "report.md" in names
        assert "data.json" in names
        assert "chart.png" in names
        
        # Check types are assigned correctly
        types = {a["name"]: a["type"] for a in artifacts}
        assert types["report.md"] == "markdown"
        assert types["data.json"] == "json"
        assert types["chart.png"] == "image"


def test_discover_artifacts_empty_dir():
    """discover_artifacts returns empty list for empty directory."""
    from cognitionflow.orchestration import discover_artifacts
    
    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts = discover_artifacts(tmpdir)
        assert artifacts == []
