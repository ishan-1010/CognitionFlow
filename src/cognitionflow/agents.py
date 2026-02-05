"""
AutoGen agent team: Product Manager (proxy) and Senior Engineer (assistant).
Lightweight version - no vector memory dependencies.
"""
import autogen

from cognitionflow.config import get_config, get_workspace_dir


SYSTEM_PROMPTS = {
    "standard": """You are a Principal Software Engineer and Data Scientist. 
Your goal is to provide logically sound, bug-free, and highly efficient Python solutions.

### 🧠 Logic & Reasoning:
1. **Analyze Requirements**: Break down the task into logical steps.
2. **Defensive Programming**: 
   - **Variable Scope**: ALWAYS initialize dictionaries and variables before referencing them.
   - **Array Alignment**: In Pandas/Numpy, ENSURE all arrays have identical lengths before creating a DataFrame.
   - **Syntax Safety**: Use raw strings (r"...") if including symbols like '%' in formatted strings to avoid SyntaxWarnings.
3. **Library Awareness**: assume 'mdformat' or 'polars' might be missing. Standardize on Pandas/Matplotlib/Seaborn.

### 🛠 Execution Pattern:
- First, briefly explain your technical approach.
- Provide the complete solution in ONE markdown block (```python ... ```).
- Ensure all file operations (`open`, `savefig`, `to_csv`) use the current directory.
- Print clear status messages (e.g., "Calculating metrics...", "Generating report...") to the console.

### 🛑 Python Guardrails (CRITICAL):
- **NEVER** include the word `TERMINATE` inside a Python code block (```python ... ```). 
- If you put `TERMINATE` inside a code block, it will cause a `NameError` and fail the assignment.
- The word `TERMINATE` must ONLY appear on the very last line of your message, outside of any markdown code blocks.

### ⚠️ Banned Patterns:
- **NEVER** use `data['Column']` before the `data` dictionary is fully defined.

### 🛑 Termination:
Write 'TERMINATE' on a new line OUTSIDE and AFTER your code block once the goal is reached and output is verified.
""",
    "detailed": """You are an Elite Lead Engineer with deep expertise in Python, Data Engineering, and Statistical Analysis.

### 📋 Technical Objectives:
1. **Requirement Analysis**: Identify edge cases and data validation needs.
2. **High-Quality Code**:
   - Use clear, descriptive variable names.
   - Implement comprehensive error handling (try-except blocks).
   - Optimize for performance and memory efficiency.
3. **Data Integrity**: 
   - Verify array lengths before DataFrame instantiation.
   - Ensure all generated files (plots/reports) are correctly named and saved.

### 📝 Reporting & Docs:
- Include inline comments explaining complex logic.
- Ensure Markdown reports are well-formatted and professional.
- Print key metrics and execution confirmations to the terminal for audit.

### 🏁 Completion Trace:
Once the code is robust and the mission is fulfilled, conclude with 'TERMINATE'.
""",
    "concise": """Expert AI Developer.
- Provide a brief technical plan.
- Single clean code block (current directory outputs).
- Defensive coding: Verify variable scope and array lengths.
- End with TERMINATE.
""",
}
def is_termination(msg: dict) -> bool:
    """Check if a message should terminate the conversation."""
    content = msg.get("content", "")
    if content is None:
        return True
    content = content.strip()
    # Stricter check: only terminate if the message ends with TERMINATE
    # or if the last non-empty line is exactly TERMINATE.
    if "TERMINATE" in content:
        # If there's a code block, it's NOT a termination yet 
        # (the user proxy needs to execute the code first).
        if "```" in content:
            return False
            
        # Strip trailing whitespace and check if it ends with TERMINATE
        if content.rstrip().endswith("TERMINATE"):
            return True
            
        # Also check if it's the only thing in the last line (ignoring trailing whitespace)
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if lines and lines[-1] == "TERMINATE":
            return True

    return False


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
        max_consecutive_auto_reply=10, # Increased from 3 to allow more complex debugging turns
        is_termination_msg=is_termination,
        llm_config=llm_config, # ENABLE LLM for the PM to act as a Technical Reviewer
        code_execution_config={
            "work_dir": work_dir,
            "use_docker": False,
        },
        system_message="""You are a Technical Product Manager. 
Your goal is to coordinate with the Senior Engineer to ensure the mission is completed perfectly.

**Feedback Mode:**
- Explicitly point out the error (e.g., NameError, ValueError) and ask the Engineer to fix that specific issue.
- **Critical Fix**: If you see `NameError: name 'TERMINATE' is not defined`, you MUST say: "ERROR: You put the word 'TERMINATE' inside the Python code block. This is NOT valid Python. You must move the word 'TERMINATE' to the very end of your message, OUTSIDE the code block."
- Do not be vague. Provide technical feedback.

**Success Mode:**
- Only write 'TERMINATE' once the code has run successfully and all requested files (plots, reports, etc.) have been confirmed as created.
""",
    )

    engineer_agent = autogen.AssistantAgent(
        name="Senior_Engineer",
        system_message=system_message,
        llm_config=llm_config,
        is_termination_msg=is_termination,
    )


    return pm_agent, engineer_agent
