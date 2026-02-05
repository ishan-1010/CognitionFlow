"""
Run the multi-agent workflow: PM -> Engineer -> artifacts.
Lightweight version without vector memory (ChromaDB/sentence-transformers removed).
Real-time message streaming via callbacks.
Generalized to support any task, not just server spike analysis.
"""
import os
import re
import glob
from datetime import datetime
from typing import Callable, Optional

from cognitionflow.config import get_config, get_workspace_dir, get_config_with_overrides, TASK_TEMPLATES
from cognitionflow.agents import build_agents


def get_template_prompt(template_id: str) -> str:
    """Get the prompt for a task template by ID."""
    for template in TASK_TEMPLATES:
        if template["id"] == template_id:
            return template["prompt"]
    # Default to first template if not found
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


def _extract_code_blocks(content: str) -> list[str]:
    """Extract Python code blocks from markdown."""
    pattern = r"```python\n(.*?)\n```"
    matches = re.findall(pattern, content, re.DOTALL)
    return matches


def _make_message_dict(sender: str, receiver: str, content: str) -> dict:
    """Create a structured message dict from agent communication."""
    code_blocks = _extract_code_blocks(content)
    has_code = len(code_blocks) > 0
    
    # Determine message type
    msg_type = "agent_message"
    if has_code:
        msg_type = "code_generation"
    if "TERMINATE" in content:
        msg_type = "termination"
    # Check for code execution output
    if "exitcode:" in content.lower() or "code output:" in content.lower():
        msg_type = "code_execution"
    
    return {
        "type": msg_type,
        "name": sender, # Changed from sender to name for UI
        "sender": sender,
        "receiver": receiver,
        "content": content,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "has_code": has_code,
        "code_blocks": code_blocks,
    }



def _create_message_hook(agent_name: str, other_agent_name: str, on_message: Callable, streamed_messages: list):
    """
    Create a reply hook that streams messages in real-time.
    This hook fires when an agent is about to generate a reply.
    """
    def hook(recipient, messages, sender, config):
        """
        AutoGen reply hook - fires when this agent receives messages.
        We use it to stream the last message from the conversation.
        """
        if messages and on_message:
            # Get the last message (the one just received)
            last_msg = messages[-1] if isinstance(messages, list) else messages
            if isinstance(last_msg, dict):
                content = last_msg.get("content", "")
                if content:
                    # The sender of this message is the OTHER agent
                    # (the agent_name is the recipient in this context)
                    msg_dict = _make_message_dict(
                        sender=other_agent_name,
                        receiver=agent_name,
                        content=content
                    )
                    # Avoid duplicates (AutoGen may call hooks multiple times)
                    msg_key = f"{msg_dict['sender']}:{msg_dict['content'][:50]}"
                    if msg_key not in [m.get('_key') for m in streamed_messages]:
                        msg_dict['_key'] = msg_key
                        streamed_messages.append(msg_dict)
                        try:
                            on_message(msg_dict)
                        except Exception:
                            pass
        # Return False, None to let normal processing continue
        return False, None
    return hook


def run_workflow(
    task_prompt: str | None = None,
    work_dir: str | None = None,
    on_message: Optional[Callable] = None,
    model: str | None = None,
    temperature: float | None = None,
    agent_mode: str = "standard",
    # Deprecated but kept for backward compatibility if needed
    anomaly_count: int = 5,
):
    """
    Execute the agent workflow. Creates workspace and agents; runs chat; returns result.
    Artifacts are discovered dynamically from work_dir.
    
    Args:
        task_prompt: Task description for agents (uses default template if None)
        work_dir: Directory for code execution and artifacts
        on_message: Optional callback for real-time message streaming
        model: LLM model override (e.g., 'llama3-8b-8192')
        temperature: LLM temperature override (0.0-1.0)
        agent_mode: Agent verbosity: 'standard', 'detailed', or 'concise'
    
    Returns:
        dict with result, work_dir, artifacts, and messages
    """
    # Build task prompt if not provided (default to first template)
    if not task_prompt:
        task_prompt = get_template_prompt("data_analysis")
    
    work_dir = work_dir or get_workspace_dir()
    os.makedirs(work_dir, exist_ok=True)

    # Get LLM config with overrides
    llm_config = get_config_with_overrides(model=model, temperature=temperature)
    
    pm_agent, engineer_agent = build_agents(
        work_dir=work_dir,
        llm_config=llm_config,
        agent_mode=agent_mode,
    )

    engineer_agent.clear_history()
    pm_agent.clear_history()

    # Track messages for deduplication and final return
    streamed_messages: list[dict] = []

    # Send initial phase change callback
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

    # Register real-time message hooks on both agents
    # These fire DURING the chat, enabling live streaming
    if on_message:
        # Hook on PM: fires when PM receives a message from Engineer
        pm_hook = _create_message_hook(
            agent_name="Product_Manager",
            other_agent_name="Senior_Engineer", 
            on_message=on_message,
            streamed_messages=streamed_messages
        )
        pm_agent.register_reply(
            trigger=engineer_agent,
            reply_func=pm_hook,
            position=0  # Run first, before normal reply generation
        )
        
        # Hook on Engineer: fires when Engineer receives a message from PM
        eng_hook = _create_message_hook(
            agent_name="Senior_Engineer",
            other_agent_name="Product_Manager",
            on_message=on_message,
            streamed_messages=streamed_messages
        )
        engineer_agent.register_reply(
            trigger=pm_agent,
            reply_func=eng_hook,
            position=0
        )
        
        # Send the initial task prompt as first message
        try:
            init_msg = _make_message_dict(
                sender="Product_Manager",
                receiver="Senior_Engineer",
                content=task_prompt
            )
            init_msg['_key'] = f"Product_Manager:{task_prompt[:50]}"
            streamed_messages.append(init_msg)
            on_message(init_msg)
        except Exception:
            pass

    result = pm_agent.initiate_chat(engineer_agent, message=task_prompt)

    # Clean up message keys before returning
    final_messages = []
    for msg in streamed_messages:
        clean_msg = {k: v for k, v in msg.items() if not k.startswith('_')}
        final_messages.append(clean_msg)

    # dynamically discover all created artifacts
    artifacts = discover_artifacts(work_dir)
    
    return {
        "result": result,
        "work_dir": work_dir,
        "artifacts": artifacts, # List of {path, name, type}
        "messages": final_messages,
    }
