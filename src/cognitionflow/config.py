"""
Configuration from environment. Supports Groq and OpenAI.
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def load_env() -> None:
    """Load .env if python-dotenv is available. Idempotent."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def get_config() -> dict:
    """
    Build LLM config from environment.
    Prefers GROQ_API_KEY (Groq). Falls back to OPENAI_API_KEY (OpenAI).
    """
    load_env()
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Set GROQ_API_KEY or OPENAI_API_KEY in environment or .env"
        )

    if os.environ.get("GROQ_API_KEY"):
        return {
            "config_list": [{
                "model": os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
                "api_key": api_key,
                "base_url": "https://api.groq.com/openai/v1",
            }],
            "temperature": float(os.environ.get("GROQ_TEMPERATURE", "0.1")),
            "timeout": int(os.environ.get("GROQ_TIMEOUT", "600")),
        }
    # OpenAI
    entry = {"model": os.environ.get("OPENAI_MODEL", "gpt-4o"), "api_key": api_key}
    if os.environ.get("OPENAI_BASE_URL"):
        entry["base_url"] = os.environ["OPENAI_BASE_URL"]
    return {
        "config_list": [entry],
        "temperature": float(os.environ.get("OPENAI_TEMPERATURE", "0.1")),
        "timeout": int(os.environ.get("OPENAI_TIMEOUT", "600")),
    }


def get_workspace_dir() -> str:
    """Directory for agent code execution and artifacts."""
    load_env()
    return os.environ.get("COGNITIONFLOW_WORKSPACE", "project_workspace")
