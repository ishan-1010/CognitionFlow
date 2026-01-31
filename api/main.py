"""
FastAPI service for CognitionFlow: trigger RCA workflow and return artifact paths.
"""
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Add src to path so cognitionflow is importable when running from repo root
import sys
_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.join(os.path.dirname(_here), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from cognitionflow.config import load_env, get_workspace_dir
from cognitionflow.orchestration import run_workflow, DEFAULT_TASK_PROMPT


# Optional: persist run ids and paths for GET /runs/{id}
RUNS: dict[str, dict] = {}


def _run_sync(work_dir: str, run_id: str) -> None:
    """Run workflow synchronously and record result."""
    try:
        load_env()
        result = run_workflow(
            task_prompt=DEFAULT_TASK_PROMPT,
            work_dir=work_dir,
        )
        RUNS[run_id] = {
            "status": "completed",
            "work_dir": result["work_dir"],
            "artifact_report": result["artifact_report"],
            "artifact_plot": result["artifact_plot"],
        }
    except Exception as e:
        RUNS[run_id] = {"status": "failed", "error": str(e)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_env()
    yield
    # cleanup if any


app = FastAPI(
    title="CognitionFlow API",
    description="Multi-agent RCA pipeline: trigger analysis and retrieve artifacts.",
    lifespan=lifespan,
)


class RunResponse(BaseModel):
    run_id: str
    status: str
    message: str


@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse)
def run_analysis(background_tasks: BackgroundTasks):
    """
    Start the RCA workflow. Runs in background; use GET /runs/{run_id} for status and artifacts.
    """
    run_id = str(uuid.uuid4())
    base_dir = get_workspace_dir()
    work_dir = os.path.join(base_dir, run_id)
    os.makedirs(work_dir, exist_ok=True)

    RUNS[run_id] = {"status": "running", "work_dir": work_dir}
    background_tasks.add_task(_run_sync, work_dir, run_id)

    return RunResponse(
        run_id=run_id,
        status="started",
        message="Workflow started. Poll GET /runs/{run_id} for status and artifact paths.",
    )


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    """Get status and artifact paths for a run."""
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="Run not found")
    return RUNS[run_id]


@app.get("/runs/{run_id}/incident_report")
def get_incident_report(run_id: str):
    """Serve incident_report.md for a run."""
    if run_id not in RUNS or RUNS[run_id].get("status") != "completed":
        raise HTTPException(status_code=404, detail="Run not found or not completed")
    path = RUNS[run_id].get("artifact_report")
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path, media_type="text/markdown")


@app.get("/runs/{run_id}/server_health.png")
def get_server_health_plot(run_id: str):
    """Serve server_health.png for a run."""
    if run_id not in RUNS or RUNS[run_id].get("status") != "completed":
        raise HTTPException(status_code=404, detail="Run not found or not completed")
    path = RUNS[run_id].get("artifact_plot")
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path, media_type="image/png")
