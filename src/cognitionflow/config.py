"""
Configuration from environment. Supports Groq and OpenAI.
"""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def load_env() -> None:
    """Load .env if python-dotenv is available. Idempotent."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def get_config() -> dict:
    """
    Build LLM config from environment.
    Prefers GROQ_API_KEY (Groq). Falls back to OPENAI_API_KEY (OpenAI).
    """
    load_env()
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Set GROQ_API_KEY or OPENAI_API_KEY in environment or .env"
        )

    if os.environ.get("GROQ_API_KEY"):
        return {
            "config_list": [{
                "model": os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
                "api_key": api_key,
                "base_url": "https://api.groq.com/openai/v1",
            }],
            "temperature": float(os.environ.get("GROQ_TEMPERATURE", "0.1")),
            "timeout": int(os.environ.get("GROQ_TIMEOUT", "600")),
        }
    # OpenAI
    entry = {"model": os.environ.get("OPENAI_MODEL", "gpt-4o"), "api_key": api_key}
    if os.environ.get("OPENAI_BASE_URL"):
        entry["base_url"] = os.environ["OPENAI_BASE_URL"]
    return {
        "config_list": [entry],
        "temperature": float(os.environ.get("OPENAI_TEMPERATURE", "0.1")),
        "timeout": int(os.environ.get("OPENAI_TIMEOUT", "600")),
    }


def get_workspace_dir() -> str:
    """Directory for agent code execution and artifacts."""
    load_env()
    return os.environ.get("COGNITIONFLOW_WORKSPACE", "project_workspace")


# Available models for user selection
AVAILABLE_MODELS = [
    {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B", "description": "Most capable & versatile"},
    {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B", "description": "Fastest response"},
    {"id": "openai/gpt-oss-120b", "name": "GPT-OSS 120B", "description": "High reasoning capability"},
    {"id": "qwen/qwen3-32b", "name": "Qwen 3 32B", "description": "Strong open model"},
]

AGENT_MODES = [
    {"id": "standard", "name": "Standard", "description": "Balanced output with code + brief context"},
    {"id": "detailed", "name": "Detailed", "description": "Verbose with explanations and comments"},
    {"id": "concise", "name": "Concise", "description": "Minimal output, code only"},
]

# Pre-built task templates for generalized agent
TASK_TEMPLATES = [
    {
        "id": "data_analysis",
        "name": "📊 Data Analysis",
        "description": "Analyze a dataset, find patterns, create visualizations",
        "prompt": """**Mission:** Perform data analysis on a synthetic dataset.

**Tasks:**
1. **Data Generation:** Create a sample dataset with 1,000 rows containing realistic business metrics (dates, values, categories).
2. **Analysis:** Calculate key statistics, identify trends, and detect any anomalies.
3. **Visualization:** Create an informative plot saved as 'analysis_chart.png'.
4. **Report:** Write findings to 'analysis_report.md' with insights and recommendations.

**Libraries:** Use Polars for data manipulation, Seaborn/Matplotlib for visualization.""",
        "output_files": ["analysis_chart.png", "analysis_report.md"],
    },
    {
        "id": "code_generator",
        "name": "💻 Code Generator",
        "description": "Generate Python code based on a specification",
        "prompt": """**Mission:** Generate a well-structured Python utility module.

**Tasks:**
1. **Create Module:** Build a Python file 'utils.py' with useful utility functions.
2. **Include:**
   - A function for data validation
   - A function for file handling
   - A function for logging
3. **Documentation:** Add docstrings and type hints to all functions.
4. **Report:** Write 'code_summary.md' explaining the module's capabilities.

**Requirements:** Clean, PEP 8 compliant code with proper error handling.""",
        "output_files": ["utils.py", "code_summary.md"],
    },
    {
        "id": "report_generator",
        "name": "📝 Report Generator",
        "description": "Generate a structured Markdown report",
        "prompt": """**Mission:** Create a comprehensive business analysis report.

**Tasks:**
1. **Generate Data:** Create sample quarterly business data.
2. **Analysis:** Analyze revenue trends, growth rates, and key metrics.
3. **Report:** Write a detailed Markdown report 'business_report.md' including:
   - Executive Summary
   - Key Metrics
   - Trend Analysis
   - Recommendations
4. **Visualization:** Create a supporting chart 'metrics_chart.png'.

**Format:** Professional business report style with tables and bullet points.""",
        "output_files": ["business_report.md", "metrics_chart.png"],
    },
    {
        "id": "web_scraper",
        "name": "🌐 Web Scraper",
        "description": "Scrape and analyze web content (simulated)",
        "prompt": """**Mission:** Simulate web scraping and data extraction.

**Tasks:**
1. **Simulate Scraping:** Create a realistic scraped dataset (news headlines, prices, or product info).
2. **Data Cleaning:** Clean and structure the extracted data.
3. **Analysis:** Analyze patterns in the scraped data.
4. **Output:** Save structured data to 'scraped_data.json' and summary to 'scraping_report.md'.

**Note:** Since we cannot access live URLs, simulate realistic scraped content.""",
        "output_files": ["scraped_data.json", "scraping_report.md"],
    },
    {
        "id": "api_builder",
        "name": "🔧 API Builder",
        "description": "Generate a FastAPI endpoint skeleton",
        "prompt": """**Mission:** Create a FastAPI application skeleton.

**Tasks:**
1. **Create API:** Build 'api_app.py' with a FastAPI application including:
   - Health check endpoint
   - CRUD endpoints for a sample resource
   - Pydantic models for validation
   - A function for logging
2. **Documentation:** Write 'api_docs.md' explaining the endpoints.
3. **Requirements:** Generate 'requirements.txt' for the API.

**Requirements:** Production-ready code with proper error handling and validation.""",
        "output_files": ["api_app.py", "api_docs.md", "requirements.txt"],
    },
]

# Output format options
OUTPUT_FORMATS = [
    {"id": "markdown", "name": "Markdown", "description": "Structured text report (.md)"},
    {"id": "json", "name": "JSON", "description": "Structured data output (.json)"},
    {"id": "code", "name": "Code", "description": "Python source file (.py)"},
    {"id": "plot", "name": "Visualization", "description": "Chart/plot image (.png)"},
    {"id": "auto", "name": "Auto", "description": "Let agent decide based on task"},
]


def get_config_with_overrides(
    model: str | None = None,
    temperature: float | None = None,
) -> dict:
    """
    Build LLM config with optional runtime overrides.
    Used for user-customizable runs.
    """
    base_config = get_config()
    
    if model:
        base_config["config_list"][0]["model"] = model
    if temperature is not None:
        base_config["temperature"] = temperature
    
    return base_config
