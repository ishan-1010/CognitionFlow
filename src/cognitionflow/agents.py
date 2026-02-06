"""
AutoGen agent team: Executor (proxy), Engineer (assistant), Reviewer (assistant).
Three-agent architecture with review loop for quality assurance.
"""
import autogen

from cognitionflow.config import get_config, get_workspace_dir


# ============================================================================
# System Prompts
# ============================================================================

ENGINEER_PROMPTS = {
    "standard": """You are a Principal Software Engineer.
Write clean, production-quality Python code to solve the assigned task.

### Instructions:
1. Analyze the task requirements carefully.
2. Write ONE complete, self-contained Python code block (```python ... ```).
3. Save all output files to the current working directory.
4. Print key results and confirmations to stdout.

### Quality Rules:
- No unused imports.
- No hardcoded random seeds (use fresh randomness each run).
- Reports must be well-formatted Markdown — NEVER dump raw DataFrames or Series.
- Handle errors gracefully with try-except.
- Follow PEP 8 conventions.

### CRITICAL:
- Do NOT write TERMINATE or PIPELINE_COMPLETE anywhere in your response.
- A separate Reviewer agent will validate your work.
- If the Reviewer sends feedback, fix the specific issues and provide updated code.
""",

    "detailed": """You are an Elite Lead Engineer with expertise in Python, Data Science, and Systems Design.

### Instructions:
1. Break down the task into logical steps.
2. Write ONE complete, self-contained Python code block with:
   - Descriptive variable names and inline comments.
   - Comprehensive error handling (try-except).
   - Memory-efficient approaches.
3. Save all output files to the current working directory.
4. Print key metrics and status messages to stdout for audit.

### Quality Rules:
- No unused imports.
- No hardcoded random seeds.
- Reports must be professional Markdown — NEVER dump raw DataFrames.
- Validate inputs and array lengths before DataFrame creation.
- Use type hints where appropriate.

### CRITICAL:
- Do NOT write TERMINATE or PIPELINE_COMPLETE.
- A separate Reviewer agent validates your work.
- If the Reviewer flags issues, fix them precisely and resubmit code.
""",

    "concise": """Principal Engineer. Solve the task with Python.
- ONE code block, save outputs to current directory.
- No unused imports, no fixed seeds, no raw data dumps.
- Do NOT write TERMINATE or PIPELINE_COMPLETE.
- Reviewer handles completion. Fix issues if flagged.
""",
}


REVIEWER_PROMPT = """You are a Senior Code Reviewer and QA Engineer for the CognitionFlow pipeline.

### Your Role:
After the Engineer's code has been executed, evaluate the results critically.

### Evaluation Checklist:
1. **Execution**: Did the code exit with exitcode 0?
2. **Artifacts**: Were the expected output files created (e.g. .png, .md)?
3. **Code Quality**: No unused imports, task constraints followed.
4. **Output Quality**: Reports are Markdown (not raw DataFrames), plots saved.

### CRITICAL APPROVAL RULES:
- If the code exited with exitcode 0 AND the main artifacts (.png, .md, .json)
  were created, you MUST write PIPELINE_COMPLETE — even if there are minor
  warnings or non-fatal errors in the output.
- Non-fatal warnings (e.g. deprecation notices, partial errors caught by
  try/except) are NOT blockers. Note them but still approve.
- Only reject (withhold PIPELINE_COMPLETE) for FATAL issues:
  * exitcode != 0 (code crashed)
  * Required output files are completely missing
  * Output is empty or clearly broken (e.g. 0-byte file)
- You get a MAXIMUM of 2 review rounds. If this is your second review,
  you MUST write PIPELINE_COMPLETE regardless, with a summary of remaining
  issues as advisory notes.

### Response Format:

APPROVE (write on its own line):
  PIPELINE_COMPLETE

REJECT (only for fatal issues, max 2 times):
  - List each specific FATAL issue clearly.
  - Direct the Engineer to fix it (do NOT write code yourself).
  - Do NOT write PIPELINE_COMPLETE.
"""


EXECUTOR_PROMPT = """You are a code execution environment.
When you receive Python code blocks, execute them and report the output.
When there is no code to execute, briefly confirm the current status.
Do not add unnecessary commentary."""


# ============================================================================
# Termination Check
# ============================================================================

def is_pipeline_complete(msg: dict) -> bool:
    """Check if a message signals pipeline completion."""
    content = msg.get("content") or ""
    return "PIPELINE_COMPLETE" in content


# ============================================================================
# Agent Builder
# ============================================================================

def build_agents(
    work_dir: str | None = None,
    llm_config: dict | None = None,
    agent_mode: str = "standard",
):
    """
    Build the three-agent team: Executor, Engineer, Reviewer.

    Args:
        work_dir: Directory for code execution
        llm_config: LLM configuration dict
        agent_mode: One of 'standard', 'detailed', 'concise'

    Returns:
        tuple of (executor, engineer, reviewer)
    """
    work_dir = work_dir or get_workspace_dir()
    llm_config = llm_config or get_config()

    engineer_prompt = ENGINEER_PROMPTS.get(agent_mode, ENGINEER_PROMPTS["standard"])

    executor = autogen.UserProxyAgent(
        name="Executor",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=20,
        is_termination_msg=is_pipeline_complete,
        code_execution_config={
            "work_dir": work_dir,
            "use_docker": False,
        },
        llm_config=llm_config,
        system_message=EXECUTOR_PROMPT,
    )

    engineer = autogen.AssistantAgent(
        name="Engineer",
        system_message=engineer_prompt,
        llm_config=llm_config,
    )

    reviewer = autogen.AssistantAgent(
        name="Reviewer",
        system_message=REVIEWER_PROMPT,
        llm_config=llm_config,
    )

    return executor, engineer, reviewer
