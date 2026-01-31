"""
Run the multi-agent RCA workflow: PM -> Engineer -> artifacts.
"""
import os

from cognitionflow.config import get_config, get_workspace_dir
from cognitionflow.memory import create_memory
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


def run_workflow(
    task_prompt: str | None = None,
    work_dir: str | None = None,
    memory_dir: str | None = None,
    reset_memory: bool = False,
):
    """
    Execute the agent workflow. Creates workspace, memory, and agents; runs chat; returns result.
    Artifacts (server_health.png, incident_report.md) are written under work_dir.
    """
    task_prompt = task_prompt or DEFAULT_TASK_PROMPT
    work_dir = work_dir or get_workspace_dir()
    memory_dir = memory_dir or "./tmp/agent_memory_db"

    os.makedirs(work_dir, exist_ok=True)

    memory = create_memory(
        path_to_db_dir=memory_dir,
        reset_db=reset_memory,
        recall_threshold=1.5,
    )
    pm_agent, engineer_agent = build_agents(
        work_dir=work_dir,
        memory=memory,
        llm_config=get_config(),
    )

    engineer_agent.clear_history()
    pm_agent.clear_history()

    result = pm_agent.initiate_chat(engineer_agent, message=task_prompt)

    return {
        "result": result,
        "work_dir": work_dir,
        "artifact_report": os.path.join(work_dir, "incident_report.md"),
        "artifact_plot": os.path.join(work_dir, "server_health.png"),
    }
