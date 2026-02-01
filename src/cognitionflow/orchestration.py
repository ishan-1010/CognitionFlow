"""
Run the multi-agent RCA workflow: PM -> Engineer -> artifacts.
Lightweight version without vector memory (ChromaDB/sentence-transformers removed).
"""
import os
import re
from datetime import datetime
from typing import Callable, Optional

from cognitionflow.config import get_config, get_workspace_dir
from cognitionflow.agents import build_agents


DEFAULT_TASK_PROMPT = """
**Mission:** Perform a Root Cause Analysis on server instability.

**Tasks:**
1. **Data Simulation:** Generate 1,000 server logs (Timestamp, Latency, Status). Inject distinct 'Spike' anomalies.
2. **Visualization:** - Create a dark-themed plot ('server_health.png').
    - Plot Latency vs. Moving Average.
    - Highlight anomalies in Red.
3. **Reporting:** - Analyze the data you just generated.
    - Write a short executive summary to a file named 'incident_report.md'.
    - Explain WHEN the spikes happened and give a hypothetical reason (e.g., "DB Backup").

**Constraints:**
- Use the libraries I taught you (Polars/Seaborn).
- Save both files locally.
"""


def _extract_code_blocks(content: str) -> list[str]:
    """Extract Python code blocks from markdown."""
    pattern = r"```python\n(.*?)\n```"
    matches = re.findall(pattern, content, re.DOTALL)
    return matches


def _parse_chat_messages(chat_history: list, on_message: Optional[Callable] = None) -> list[dict]:
    """
    Parse AutoGen chat history into structured messages.
    Each message dict: {sender, receiver, content, timestamp, has_code, code_blocks, type}
    """
    messages = []
    prev_sender = None
    
    for msg in chat_history:
        if not isinstance(msg, dict):
            continue
            
        content = msg.get("content", "")
        role = msg.get("role", "")
        
        # Check if we already marked the sender
        if "_sender" in msg:
            sender = msg["_sender"]
        else:
            # AutoGen stores messages with role='assistant' for both agents
            # We need to infer sender from context (alternating pattern)
            # Or use the name from the message if available
            sender_name = msg.get("name", "")
            
            # Infer sender from name or role pattern
            if "Product_Manager" in sender_name or (role == "user" and not prev_sender):
                sender = "Product_Manager"
            elif "Senior_Engineer" in sender_name or (role == "assistant" and prev_sender != "Senior_Engineer"):
                sender = "Senior_Engineer"
            else:
                # Fallback: alternate based on previous sender
                if prev_sender == "Product_Manager":
                    sender = "Senior_Engineer"
                else:
                    sender = "Product_Manager"
        
        # Determine receiver
        receiver = "Senior_Engineer" if sender == "Product_Manager" else "Product_Manager"
        prev_sender = sender
        
        # Extract code blocks
        code_blocks = _extract_code_blocks(content)
        has_code = len(code_blocks) > 0
        
        # Determine message type
        msg_type = "agent_message"
        if has_code:
            msg_type = "code_generation"
        if "TERMINATE" in content and not has_code:
            msg_type = "termination"
        
        message_dict = {
            "type": msg_type,
            "sender": sender,
            "receiver": receiver,
            "content": content,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "has_code": has_code,
            "code_blocks": code_blocks,
        }
        
        messages.append(message_dict)
        
        # Call callback if provided
        if on_message:
            try:
                on_message(message_dict)
            except Exception:
                pass  # Don't fail workflow if callback errors
    
    return messages


def run_workflow(
    task_prompt: str | None = None,
    work_dir: str | None = None,
    on_message: Optional[Callable] = None,
):
    """
    Execute the agent workflow. Creates workspace and agents; runs chat; returns result.
    Artifacts (server_health.png, incident_report.md) are written under work_dir.
    
    Args:
        task_prompt: Task description for agents
        work_dir: Directory for code execution and artifacts
        on_message: Optional callback function called for each agent message.
                   Receives dict: {type, sender, receiver, content, timestamp, has_code, code_blocks}
    
    Returns:
        dict with result, work_dir, artifacts, and messages
    """
    task_prompt = task_prompt or DEFAULT_TASK_PROMPT
    work_dir = work_dir or get_workspace_dir()

    os.makedirs(work_dir, exist_ok=True)

    pm_agent, engineer_agent = build_agents(
        work_dir=work_dir,
        llm_config=get_config(),
    )

    engineer_agent.clear_history()
    pm_agent.clear_history()

    # Send initial message callback
    if on_message:
        try:
            on_message({
                "type": "phase_change",
                "phase": "initializing",
                "message": "Initializing agents...",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })
        except Exception:
            pass

    result = pm_agent.initiate_chat(engineer_agent, message=task_prompt)

    # Extract conversation history
    # AutoGen stores messages in agent.chat_messages dict
    # Format: {recipient_agent: [list of messages]}
    all_messages = []
    
    # Try to get from ChatResult first
    if hasattr(result, 'chat_history') and result.chat_history:
        all_messages = result.chat_history
    else:
        # Get messages from PM's perspective (messages sent to Engineer)
        pm_to_eng = pm_agent.chat_messages.get(engineer_agent, [])
        # Get messages from Engineer's perspective (messages sent to PM)
        eng_to_pm = engineer_agent.chat_messages.get(pm_agent, [])
        
        # Combine: PM sends first (task prompt), then alternate
        # We need to interleave them properly
        combined = []
        max_len = max(len(pm_to_eng), len(eng_to_pm))
        for i in range(max_len):
            if i < len(pm_to_eng):
                msg = pm_to_eng[i]
                if isinstance(msg, dict):
                    msg['_sender'] = 'Product_Manager'
                combined.append(msg)
            if i < len(eng_to_pm):
                msg = eng_to_pm[i]
                if isinstance(msg, dict):
                    msg['_sender'] = 'Senior_Engineer'
                combined.append(msg)
        
        all_messages = combined
    
    # Parse messages
    parsed_messages = _parse_chat_messages(all_messages, on_message)

    return {
        "result": result,
        "work_dir": work_dir,
        "artifact_report": os.path.join(work_dir, "incident_report.md"),
        "artifact_plot": os.path.join(work_dir, "server_health.png"),
        "messages": parsed_messages,
    }
