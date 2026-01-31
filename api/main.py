"""
FastAPI service for CognitionFlow: trigger RCA workflow and return artifact paths.
"""
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
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


@app.get("/", response_class=HTMLResponse)
def index():
    """Minimal web UI: run analysis and view report/plot."""
    return _INDEX_HTML


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


_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CognitionFlow</title>
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: system-ui, -apple-system, sans-serif;
      max-width: 560px;
      margin: 2rem auto;
      padding: 0 1rem;
      color: #1a1a1a;
      background: #f5f5f5;
    }
    h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
    p { color: #555; margin-bottom: 1.5rem; }
    button {
      background: #0d6efd;
      color: white;
      border: none;
      padding: 0.6rem 1.2rem;
      font-size: 1rem;
      border-radius: 6px;
      cursor: pointer;
    }
    button:hover { background: #0b5ed7; }
    button:disabled { background: #6c757d; cursor: not-allowed; }
    #status { margin-top: 1rem; padding: 0.75rem; background: #e9ecef; border-radius: 6px; min-height: 2rem; }
    #links { margin-top: 1rem; }
    #links a {
      display: inline-block;
      margin-right: 1rem;
      margin-bottom: 0.5rem;
      color: #0d6efd;
      text-decoration: none;
    }
    #links a:hover { text-decoration: underline; }
    .error { color: #dc3545; }
  </style>
</head>
<body>
  <h1>CognitionFlow</h1>
  <p>Multi-agent RCA: run server health analysis and view the report and plot.</p>
  <button id="runBtn" type="button">Run analysis</button>
  <div id="status"></div>
  <div id="links"></div>
  <script>
    const runBtn = document.getElementById('runBtn');
    const status = document.getElementById('status');
    const links = document.getElementById('links');

    function setStatus(msg, isError) {
      status.textContent = msg;
      status.className = isError ? 'error' : '';
    }

    async function pollRun(runId) {
      const base = window.location.origin;
      for (let i = 0; i < 600; i++) {
        const r = await fetch(base + '/runs/' + runId);
        if (!r.ok) { setStatus('Run not found.', true); return; }
        const data = await r.json();
        if (data.status === 'completed') {
          setStatus('Done.');
          links.innerHTML =
            '<a href="' + base + '/runs/' + runId + '/incident_report" target="_blank">View report</a>' +
            '<a href="' + base + '/runs/' + runId + '/server_health.png" target="_blank">View plot</a>';
          runBtn.disabled = false;
          return;
        }
        if (data.status === 'failed') {
          setStatus('Run failed: ' + (data.error || 'unknown'), true);
          runBtn.disabled = false;
          return;
        }
        setStatus('Running... (this may take 1–2 minutes)');
        await new Promise(function(r) { setTimeout(r, 2000); });
      }
      setStatus('Timed out.', true);
      runBtn.disabled = false;
    }

    runBtn.addEventListener('click', async function() {
      runBtn.disabled = true;
      links.innerHTML = '';
      setStatus('Starting...');
      try {
        const r = await fetch(window.location.origin + '/run', { method: 'POST' });
        const data = await r.json();
        if (!r.ok) { setStatus('Error: ' + (data.detail || r.status), true); runBtn.disabled = false; return; }
        setStatus('Running... (this may take 1–2 minutes)');
        pollRun(data.run_id);
      } catch (e) {
        setStatus('Error: ' + e.message, true);
        runBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""
