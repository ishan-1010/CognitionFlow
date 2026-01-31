"""Unit tests for config (env loading and get_config)."""
import os
import pytest

from cognitionflow.config import get_config


def test_get_config_requires_key(monkeypatch):
    """Without GROQ_API_KEY or OPENAI_API_KEY, get_config raises."""
    monkeypatch.setattr("cognitionflow.config.load_env", lambda: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY or OPENAI_API_KEY"):
        get_config()


def test_get_config_groq(monkeypatch):
    """With GROQ_API_KEY, get_config returns Groq config."""
    monkeypatch.setattr("cognitionflow.config.load_env", lambda: None)
    monkeypatch.setenv("GROQ_API_KEY", "test_groq_key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = get_config()
    assert cfg["config_list"][0]["model"] == "openai/gpt-oss-120b"
    assert cfg["config_list"][0]["base_url"] == "https://api.groq.com/openai/v1"
    assert cfg["config_list"][0]["api_key"] == "test_groq_key"
    assert cfg["temperature"] == 0.1


def test_get_config_openai(monkeypatch):
    """With only OPENAI_API_KEY, get_config returns OpenAI config."""
    monkeypatch.setattr("cognitionflow.config.load_env", lambda: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    cfg = get_config()
    assert cfg["config_list"][0]["model"] == "gpt-4o"
    assert cfg["config_list"][0]["api_key"] == "test_openai_key"
    assert "base_url" not in cfg["config_list"][0] or cfg["config_list"][0].get("base_url")


def test_get_workspace_dir_default(monkeypatch):
    """Default workspace is project_workspace."""
    monkeypatch.delenv("COGNITIONFLOW_WORKSPACE", raising=False)
    from cognitionflow.config import get_workspace_dir
    assert get_workspace_dir() == "project_workspace"


def test_get_workspace_dir_custom(monkeypatch):
    """COGNITIONFLOW_WORKSPACE overrides default."""
    monkeypatch.setenv("COGNITIONFLOW_WORKSPACE", "/custom/workspace")
    from cognitionflow.config import get_workspace_dir
    assert get_workspace_dir() == "/custom/workspace"
