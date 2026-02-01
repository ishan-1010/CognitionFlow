"""
CognitionFlow: Multi-agent RCA pipeline.
Lightweight version - no heavy vector memory dependencies (torch/chromadb removed).
"""
from cognitionflow.config import get_config, load_env

__all__ = [
    "get_config",
    "load_env",
    "build_agents",
    "run_workflow",
    "DEFAULT_TASK_PROMPT",
]


def __getattr__(name):
    if name == "build_agents":
        from cognitionflow.agents import build_agents
        return build_agents
    if name in ("run_workflow", "DEFAULT_TASK_PROMPT"):
        from cognitionflow.orchestration import run_workflow, DEFAULT_TASK_PROMPT
        return run_workflow if name == "run_workflow" else DEFAULT_TASK_PROMPT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
