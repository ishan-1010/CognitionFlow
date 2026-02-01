"""
AutoGen agent team: Product Manager (proxy) and Senior Engineer (assistant).
Lightweight version - no vector memory dependencies.
"""
import autogen

from cognitionflow.config import get_config, get_workspace_dir


def build_agents(
    work_dir: str | None = None,
    llm_config: dict | None = None,
):
    """
    Build PM (UserProxy) and Engineer (Assistant) agents.
    """
    work_dir = work_dir or get_workspace_dir()
    llm_config = llm_config or get_config()

    pm_agent = autogen.UserProxyAgent(
        name="Product_Manager",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: (
            (x.get("content", "") is None or x.get("content", "").strip() == "")
            or (
                "TERMINATE" in x.get("content", "")
                and "```" not in x.get("content", "")
            )
        ),
        code_execution_config={
            "work_dir": work_dir,
            "use_docker": False,
        },
    )

    engineer_agent = autogen.AssistantAgent(
        name="Senior_Engineer",
        system_message="""You are an Elite Python Developer.
    1. Write the code to solve the task using Polars and Seaborn.
    2. Put the code in a markdown block.
    3. AFTER the code block, on a new line, write 'TERMINATE'.
    4. Do not apologize. Do not explain. Just Code + Terminate.
    """,
        llm_config=llm_config,
    )

    return pm_agent, engineer_agent
