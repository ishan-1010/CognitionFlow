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
  <title>CognitionFlow - Multi-Agent RCA Pipeline</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
      min-height: 100vh;
      color: #e2e8f0;
    }
    .container {
      max-width: 720px;
      margin: 0 auto;
      padding: 3rem 1.5rem;
    }
    header {
      text-align: center;
      margin-bottom: 2.5rem;
    }
    .logo {
      font-size: 2.5rem;
      font-weight: 700;
      background: linear-gradient(135deg, #60a5fa, #a78bfa);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      margin-bottom: 0.5rem;
    }
    .tagline {
      color: #94a3b8;
      font-size: 1.1rem;
    }
    .card {
      background: rgba(30, 41, 59, 0.8);
      border: 1px solid rgba(148, 163, 184, 0.1);
      border-radius: 16px;
      padding: 2rem;
      margin-bottom: 1.5rem;
      backdrop-filter: blur(10px);
    }
    .card h2 {
      font-size: 1.25rem;
      margin-bottom: 1rem;
      color: #f1f5f9;
    }
    .card p {
      color: #94a3b8;
      line-height: 1.6;
      margin-bottom: 1rem;
    }
    .tech-stack {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-top: 1rem;
    }
    .tech-badge {
      background: rgba(96, 165, 250, 0.15);
      color: #60a5fa;
      padding: 0.35rem 0.75rem;
      border-radius: 20px;
      font-size: 0.8rem;
      font-weight: 500;
    }
    .run-section {
      text-align: center;
    }
    .run-btn {
      background: linear-gradient(135deg, #3b82f6, #8b5cf6);
      color: white;
      border: none;
      padding: 1rem 2.5rem;
      font-size: 1.1rem;
      font-weight: 600;
      border-radius: 12px;
      cursor: pointer;
      transition: all 0.3s ease;
      box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    .run-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4); }
    .run-btn:disabled { background: #475569; cursor: not-allowed; transform: none; box-shadow: none; }
    .status-box {
      margin-top: 1.5rem;
      padding: 1rem;
      background: rgba(15, 23, 42, 0.6);
      border-radius: 8px;
      min-height: 3rem;
      color: #cbd5e1;
    }
    .status-box.error { color: #f87171; background: rgba(248, 113, 113, 0.1); }
    .status-box.success { color: #4ade80; background: rgba(74, 222, 128, 0.1); }
    .results {
      display: flex;
      gap: 1rem;
      margin-top: 1rem;
      justify-content: center;
      flex-wrap: wrap;
    }
    .result-link {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      background: rgba(59, 130, 246, 0.2);
      color: #60a5fa;
      padding: 0.75rem 1.25rem;
      border-radius: 8px;
      text-decoration: none;
      font-weight: 500;
      transition: background 0.2s;
    }
    .result-link:hover { background: rgba(59, 130, 246, 0.3); }
    .features {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 1rem;
      margin-top: 1rem;
    }
    .feature {
      background: rgba(15, 23, 42, 0.5);
      padding: 1rem;
      border-radius: 8px;
    }
    .feature h3 {
      font-size: 0.95rem;
      color: #f1f5f9;
      margin-bottom: 0.5rem;
    }
    .feature p {
      font-size: 0.85rem;
      color: #64748b;
      margin: 0;
    }
    footer {
      text-align: center;
      margin-top: 2rem;
      color: #475569;
      font-size: 0.85rem;
    }
    footer a { color: #60a5fa; text-decoration: none; }
    footer a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="logo">CognitionFlow</div>
      <p class="tagline">Multi-Agent Root Cause Analysis Pipeline</p>
    </header>

    <div class="card">
      <h2>About This Project</h2>
      <p>
        CognitionFlow demonstrates a <strong>multi-agent AI system</strong> for automated server health analysis.
        Two AI agents collaborate: a <em>Product Manager</em> orchestrates the workflow while a
        <em>Senior Engineer</em> generates analysis code, visualizations, and incident reports.
      </p>
      <div class="features">
        <div class="feature">
          <h3>Multi-Agent Orchestration</h3>
          <p>Microsoft AutoGen powers agent collaboration and code execution</p>
        </div>
        <div class="feature">
          <h3>Automated Analysis</h3>
          <p>Generates server logs, detects anomalies, creates visualizations</p>
        </div>
        <div class="feature">
          <h3>Production-Ready</h3>
          <p>FastAPI backend, async processing, containerized deployment</p>
        </div>
      </div>
      <div class="tech-stack">
        <span class="tech-badge">Python</span>
        <span class="tech-badge">AutoGen</span>
        <span class="tech-badge">FastAPI</span>
        <span class="tech-badge">Polars</span>
        <span class="tech-badge">Seaborn</span>
        <span class="tech-badge">Groq LPU</span>
        <span class="tech-badge">Docker</span>
      </div>
    </div>

    <div class="card run-section">
      <h2>Run Analysis</h2>
      <p>Click below to trigger the multi-agent workflow. The agents will simulate server logs, detect anomalies, generate a visualization, and write an incident report.</p>
      <button id="runBtn" class="run-btn" type="button">Start Analysis</button>
      <div id="status" class="status-box"></div>
      <div id="links" class="results"></div>
    </div>

    <footer>
      Built by <a href="https://github.com/ishan-1010" target="_blank">Ishan Katoch</a> |
      <a href="https://github.com/ishan-1010/CognitionFlow" target="_blank">View on GitHub</a>
    </footer>
  </div>

  <script>
    const runBtn = document.getElementById('runBtn');
    const status = document.getElementById('status');
    const links = document.getElementById('links');

    function setStatus(msg, type) {
      status.textContent = msg;
      status.className = 'status-box' + (type ? ' ' + type : '');
    }

    async function pollRun(runId) {
      const base = window.location.origin;
      for (let i = 0; i < 300; i++) {
        try {
          const r = await fetch(base + '/runs/' + runId);
          if (!r.ok) { setStatus('Run not found. The server may have restarted.', 'error'); runBtn.disabled = false; return; }
          const data = await r.json();
          if (data.status === 'completed') {
            setStatus('Analysis complete!', 'success');
            links.innerHTML =
              '<a class="result-link" href="' + base + '/runs/' + runId + '/incident_report" target="_blank">View Incident Report</a>' +
              '<a class="result-link" href="' + base + '/runs/' + runId + '/server_health.png" target="_blank">View Health Plot</a>';
            runBtn.disabled = false;
            return;
          }
          if (data.status === 'failed') {
            setStatus('Analysis failed: ' + (data.error || 'Unknown error'), 'error');
            runBtn.disabled = false;
            return;
          }
          setStatus('Agents working... (typically 1-2 minutes)');
        } catch (e) {
          setStatus('Connection error. Retrying...', 'error');
        }
        await new Promise(function(r) { setTimeout(r, 2000); });
      }
      setStatus('Timed out waiting for results.', 'error');
      runBtn.disabled = false;
    }

    runBtn.addEventListener('click', async function() {
      runBtn.disabled = true;
      links.innerHTML = '';
      setStatus('Initializing agents...');
      try {
        const r = await fetch(window.location.origin + '/run', { method: 'POST' });
        const data = await r.json();
        if (!r.ok) { setStatus('Error: ' + (data.detail || r.status), 'error'); runBtn.disabled = false; return; }
        setStatus('Agents working... (typically 1-2 minutes)');
        pollRun(data.run_id);
      } catch (e) {
        setStatus('Error: ' + e.message, 'error');
        runBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""
