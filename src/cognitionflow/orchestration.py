"""
Run the multi-agent workflow with review loop.
Three agents (Executor, Engineer, Reviewer) collaborate via GroupChat.
The Reviewer validates the Engineer's output before approving completion.
"""
import gc
import os
import re
import glob
import logging
from datetime import datetime
from typing import Callable, Optional

import autogen

logger = logging.getLogger(__name__)

from cognitionflow.config import get_config, get_workspace_dir, get_config_with_overrides, TASK_TEMPLATES
from cognitionflow.agents import build_agents, is_pipeline_complete


def get_template_prompt(template_id: str) -> str:
    """Get the prompt for a task template by ID."""
    for template in TASK_TEMPLATES:
        if template["id"] == template_id:
            return template["prompt"]
    return TASK_TEMPLATES[0]["prompt"] if TASK_TEMPLATES else ""


def discover_artifacts(work_dir: str) -> list[dict]:
    """
    Discover all generated artifacts in the work directory.
    Returns a list of dicts with path, type, and name.
    """
    artifacts = []
    extensions = {
        ".png": "image",
        ".jpg": "image",
        ".jpeg": "image",
        ".md": "markdown",
        ".json": "json",
        ".py": "code",
        ".txt": "text",
        ".csv": "data",
        ".html": "html",
    }

    for filepath in glob.glob(os.path.join(work_dir, "*")):
        if os.path.isfile(filepath):
            ext = os.path.splitext(filepath)[1].lower()
            file_type = extensions.get(ext, "file")
            artifacts.append({
                "path": filepath,
                "name": os.path.basename(filepath),
                "type": file_type,
            })

    return artifacts


# ============================================================================
# Message Utilities
# ============================================================================

def _extract_code_blocks(content: str) -> list[str]:
    """Extract Python code blocks from markdown."""
    pattern = r"```python\n(.*?)\n```"
    return re.findall(pattern, content, re.DOTALL)


def _make_message_dict(sender: str, receiver: str, content: str) -> dict:
    """Create a structured message dict from agent communication."""
    code_blocks = _extract_code_blocks(content)
    has_code = len(code_blocks) > 0

    msg_type = "agent_message"
    if has_code:
        msg_type = "code_generation"
    if "PIPELINE_COMPLETE" in content:
        msg_type = "review_approved"
    if "exitcode:" in content.lower():
        msg_type = "code_execution"

    return {
        "type": msg_type,
        "name": sender,
        "sender": sender,
        "receiver": receiver,
        "content": content,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "has_code": has_code,
        "code_blocks": code_blocks,
    }


# ============================================================================
# Workflow
# ============================================================================

def run_workflow(
    task_prompt: str | None = None,
    work_dir: str | None = None,
    on_message: Optional[Callable] = None,
    model: str | None = None,
    temperature: float | None = None,
    agent_mode: str = "standard",
):
    """
    Execute the three-agent workflow with review loop.

    Flow: Executor -> Engineer -> Executor (execute) -> Reviewer -> (approve or iterate)

    Args:
        task_prompt: Task description for agents (uses default template if None)
        work_dir: Directory for code execution and artifacts
        on_message: Optional callback for real-time message streaming
        model: LLM model override
        temperature: LLM temperature override (0.0-1.0)
        agent_mode: Agent verbosity: 'standard', 'detailed', or 'concise'

    Returns:
        dict with result, work_dir, artifacts, and messages
    """
    if not task_prompt:
        task_prompt = get_template_prompt("data_analysis")

    work_dir = work_dir or get_workspace_dir()
    os.makedirs(work_dir, exist_ok=True)

    llm_config = get_config_with_overrides(model=model, temperature=temperature)

    executor, engineer, reviewer = build_agents(
        work_dir=work_dir,
        llm_config=llm_config,
        agent_mode=agent_mode,
    )

    executor.clear_history()
    engineer.clear_history()
    reviewer.clear_history()

    streamed_messages: list[dict] = []

    # Send initialization phase change
    if on_message:
        try:
            on_message({
                "type": "phase_change",
                "phase": "initializing",
                "message": f"Initializing fresh agent instance (Session: {os.path.basename(work_dir)[:8]})...",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })
        except Exception:
            pass

    # ----------------------------------------------------------------
    # Custom speaker selection: deterministic flow with review loop
    # ----------------------------------------------------------------
    def select_next_speaker(last_speaker, groupchat):
        messages = groupchat.messages
        if not messages:
            return engineer

        last_content = (messages[-1].get("content", "") or "").lower()

        if last_speaker == executor:
            # First message is the task prompt → send to Engineer
            if len(messages) <= 1:
                return engineer
            # Code was executed (output contains "exitcode:") → Reviewer evaluates
            if "exitcode:" in last_content:
                return reviewer
            # No code executed (e.g. Engineer sent text only) → Reviewer evaluates
            return reviewer

        if last_speaker == engineer:
            # Engineer wrote code (or text) → Executor handles it
            return executor

        if last_speaker == reviewer:
            # Reviewer gave feedback → Engineer fixes issues
            # (PIPELINE_COMPLETE termination is caught by the manager before this)
            return engineer

        return engineer  # Default

    # ----------------------------------------------------------------
    # Build GroupChat
    # ----------------------------------------------------------------
    groupchat = autogen.GroupChat(
        agents=[executor, engineer, reviewer],
        messages=[],
        max_round=12,  # Safety limit: ~3 review cycles (Engineer+Executor+Reviewer = 3 msgs/cycle)
        speaker_selection_method=select_next_speaker,
    )

    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config,
        is_termination_msg=is_pipeline_complete,
    )

    # ----------------------------------------------------------------
    # Real-time streaming via monkey-patching manager._process_received_message
    # ----------------------------------------------------------------
    # Why this approach:
    #   register_reply() uses copy.copy(config) to store the GroupChat, so
    #   any instance-level patches on the *original* GroupChat (append,
    #   messages list, etc.) are invisible to run_chat() which operates
    #   on the *copy*.  The GroupChatManager is NOT copied, so patching
    #   a method on the manager instance works reliably.
    #
    # Every agent message passes through manager._process_received_message:
    #   - Initial task prompt (executor → manager)
    #   - Each speaker reply  (speaker.send(reply, manager))
    # This is the single choke-point for all inbound messages.
    streamed_keys: set = set()

    if on_message:
        _original_process = manager._process_received_message

        def _streaming_process(message, sender, silent):
            """Intercept every message the manager receives and push to SSE."""
            _original_process(message, sender, silent)
            # Extract content
            if isinstance(message, dict):
                content = message.get("content", "")
            elif isinstance(message, str):
                content = message
            else:
                content = ""
            name = getattr(sender, "name", "Unknown")
            if not content:
                return
            key = f"{name}:{content[:80]}"
            if key in streamed_keys:
                return
            streamed_keys.add(key)
            msg_dict = _make_message_dict(name, "GroupChat", content)
            try:
                on_message(msg_dict)
            except Exception:
                pass

        manager._process_received_message = _streaming_process

    # ----------------------------------------------------------------
    # Run the GroupChat (with graceful error handling)
    # ----------------------------------------------------------------
    result = None
    try:
        result = executor.initiate_chat(manager, message=task_prompt)
    except Exception as chat_err:
        # Log but don't crash — we may still have useful artifacts
        logger.warning("GroupChat ended with error: %s", chat_err)
        if on_message:
            try:
                on_message({
                    "type": "phase_change",
                    "phase": "warning",
                    "message": f"Agent conversation ended early: {type(chat_err).__name__}",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                })
            except Exception:
                pass

    # Collect all messages for the return value
    # Access the *copy* that run_chat used via the manager's reply func list
    gc_messages = groupchat.messages
    for entry in getattr(manager, "_reply_func_list", []):
        cfg = entry.get("config")
        if cfg is not None and hasattr(cfg, "messages") and cfg.messages:
            gc_messages = cfg.messages
            break

    final_messages = []
    for item in gc_messages:
        if isinstance(item, dict) and item.get("content"):
            final_messages.append(
                _make_message_dict(
                    sender=item.get("name", item.get("role", "System")),
                    receiver="GroupChat",
                    content=item["content"],
                )
            )

    artifacts = discover_artifacts(work_dir)

    # Force garbage collection to free LLM response caches
    gc.collect()

    return {
        "result": result,
        "work_dir": work_dir,
        "artifacts": artifacts,
        "messages": final_messages,
    }
