"""
AutoGen agent team: Product Manager (proxy) and Senior Engineer (assistant).
Lightweight version - no vector memory dependencies.
"""
import autogen

from cognitionflow.config import get_config, get_workspace_dir


SYSTEM_PROMPTS = {
    "standard": """You are an Elite AI Agent capable of solving any coding or analysis task.

**Instructions:**
1. Read the user's task carefully and understand the objective.
2. Write clean, production-quality Python code to solve it.
3. Put ALL code in a single markdown code block (```python ... ```).
4. Save any output files to the current working directory.
5. After the code block, on a new line, write 'TERMINATE'.

**Rules:**
- Do not apologize or explain excessively.
- Code must be complete and runnable.
- Handle errors gracefully.
""",
    "detailed": """You are an Elite AI Agent with deep expertise in Python development and data analysis.

**Instructions:**
1. Carefully analyze the user's task and identify key requirements.
2. Write production-quality Python code with:
   - Clear comments explaining your approach
   - Proper error handling
   - Efficient algorithms
3. Put ALL code in a properly formatted markdown code block.
4. Save any generated files (plots, reports, data) to the current directory.
5. After the code, provide a brief summary of what was accomplished.
6. End with 'TERMINATE' on its own line.

**Best Practices:**
- Use type hints where appropriate
- Follow PEP 8 style guidelines
- Validate inputs when necessary
""",
    "concise": """Elite AI Agent. Solve the task with Python code.
- Single code block only
- Save outputs to current directory
- End with TERMINATE
- No explanations
""",
}


def build_agents(
    work_dir: str | None = None,
    llm_config: dict | None = None,
    agent_mode: str = "standard",
):
    """
    Build PM (UserProxy) and Engineer (Assistant) agents.
    
    Args:
        work_dir: Directory for code execution
        llm_config: LLM configuration dict
        agent_mode: One of 'standard', 'detailed', 'concise'
    """
    work_dir = work_dir or get_workspace_dir()
    llm_config = llm_config or get_config()
    
    system_message = SYSTEM_PROMPTS.get(agent_mode, SYSTEM_PROMPTS["standard"])

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
        system_message=system_message,
        llm_config=llm_config,
    )

    return pm_agent, engineer_agent
