"""
CognitionFlow: Multi-agent pipeline with review loop.
Three-agent architecture: Executor, Engineer, Reviewer.
"""
from cognitionflow.config import get_config, load_env

__all__ = [
    "get_config",
    "load_env",
    "build_agents",
    "run_workflow",
    "get_template_prompt",
]


def __getattr__(name):
    if name == "build_agents":
        from cognitionflow.agents import build_agents
        return build_agents
    if name == "run_workflow":
        from cognitionflow.orchestration import run_workflow
        return run_workflow
    if name == "get_template_prompt":
        from cognitionflow.orchestration import get_template_prompt
        return get_template_prompt
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
