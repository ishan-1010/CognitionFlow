"""
FastAPI service for CognitionFlow: trigger RCA workflow and return artifact paths.
Enhanced with user customization, memory optimization, and production features.
"""
# Force Agg backend before any matplotlib imports (memory optimization)
import matplotlib
matplotlib.use('Agg')

import os
import uuid
import json
import queue
import asyncio
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

# Add src to path so cognitionflow is importable when running from repo root
import sys
_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.join(os.path.dirname(_here), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from cognitionflow.config import (
    load_env, get_workspace_dir, 
    AVAILABLE_MODELS, AGENT_MODES,
    TASK_TEMPLATES, OUTPUT_FORMATS, get_config_with_overrides
)
from cognitionflow.orchestration import run_workflow, DEFAULT_TASK_PROMPT

# Import database functions
from api.db import save_run, get_run_history, get_metrics as db_get_metrics


# ============================================================================
# Memory Optimization: Concurrent Run Limiter
# ============================================================================
MAX_CONCURRENT_RUNS = 2
run_semaphore = asyncio.Semaphore(MAX_CONCURRENT_RUNS)

# Store run state and message queues for SSE streaming
RUNS: dict[str, dict] = {}
RUN_QUEUES: dict[str, queue.Queue] = {}

# Rate limiting: simple in-memory counter (resets on restart)
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 10  # requests per window
rate_limit_store: dict[str, list] = defaultdict(list)


# ============================================================================
# Workspace Cleanup (Memory Optimization)
# ============================================================================
CLEANUP_AGE_HOURS = 1  # Delete workspaces older than this


def cleanup_old_workspaces():
    """Delete workspace folders older than CLEANUP_AGE_HOURS."""
    base_dir = get_workspace_dir()
    if not os.path.exists(base_dir):
        return
    
    cutoff = datetime.utcnow() - timedelta(hours=CLEANUP_AGE_HOURS)
    cleaned = 0
    
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        
        try:
            # Check folder modification time
            mtime = datetime.utcfromtimestamp(os.path.getmtime(folder_path))
            if mtime < cutoff:
                shutil.rmtree(folder_path)
                cleaned += 1
        except Exception:
            pass
    
    return cleaned


# ============================================================================
# Rate Limiting
# ============================================================================
def check_rate_limit(client_ip: str) -> bool:
    """Check if client has exceeded rate limit. Returns True if allowed."""
    now = datetime.utcnow()
    window_start = now - timedelta(seconds=RATE_LIMIT_WINDOW)
    
    # Clean old entries
    rate_limit_store[client_ip] = [
        ts for ts in rate_limit_store[client_ip] if ts > window_start
    ]
    
    if len(rate_limit_store[client_ip]) >= RATE_LIMIT_MAX:
        return False
    
    rate_limit_store[client_ip].append(now)
    return True


# ============================================================================
# Pydantic Models
# ============================================================================
class RunConfig(BaseModel):
    """Configuration for a custom analysis run."""
    task_prompt: Optional[str] = Field(None, description="Custom task prompt or overridden template prompt")
    template_id: str = Field("data_analysis", description="ID of the task template to use")
    output_format: str = Field("markdown", description="Desired output format (markdown, json, code, plot, auto)")
    
    model: str = Field(
        "llama-3.1-8b-instant",
        description="LLM model to use"
    )
    temperature: float = Field(
        0.7, 
        ge=0.0, 
        le=1.0, 
        description="LLM temperature (0.0 = deterministic, 1.0 = creative)"
    )
    # Keeping anomaly_count for backward compatibility with old 'data_analysis' template
    anomaly_count: int = Field(5, ge=1, le=10, description="Number of items/anomalies (if applicable)")
    agent_mode: str = Field("standard", description="Agent verbosity: standard, detailed, or concise")


class RunResponse(BaseModel):
    run_id: str
    status: str
    message: str


class ConfigResponse(BaseModel):
    models: List[dict]
    agent_modes: List[dict]
    task_templates: List[dict]
    output_formats: List[dict]
    defaults: dict


# ============================================================================
# Workflow Runner
# ============================================================================
def _run_sync(work_dir: str, run_id: str, config: RunConfig) -> None:
    """Run workflow synchronously and record result. Pushes messages to queue for SSE."""
    q = RUN_QUEUES.get(run_id)
    start_time = datetime.utcnow()
    
    def on_message(msg: dict) -> None:
        """Callback to push messages to queue for SSE streaming."""
        if q:
            try:
                q.put(msg)
            except Exception:
                pass  # Don't fail if queue is closed
    
    try:
        load_env()
        
        # Send phase change
        if q:
            q.put({
                "type": "phase_change",
                "phase": "initializing",
                "message": f"Initializing agents (model: {config.model})...",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })
        
        args = {
            "task_prompt": config.task_prompt,
            "work_dir": work_dir,
            "on_message": on_message,
            "model": config.model,
            "temperature": config.temperature,
            "anomaly_count": config.anomaly_count, # Passed but may be ignored by generic templates
            "agent_mode": config.agent_mode,
            "template_id": config.template_id,
            "output_format": config.output_format,
        }
        
        # If using a specific template (and not overridden prompt), we might fetch prompts here
        # But run_workflow handles template lookups if task_prompt is None
        if config.template_id and not config.task_prompt:
            # If the user selected a template but didn't customize the prompt,
            # we let run_workflow handle fetching the template prompt. 
            # But we need to pass the template ID if we wanted orchestration to handle it.
            # However, our updated orchestration.py just takes task_prompt.
            # So let's fetch it here if needed, or update orchestration to take template_id.
            # Strategy: Fetch generic prompt here to pass as task_prompt if null.
            args["task_prompt"] = get_template_prompt(config.template_id)
            
        result = run_workflow(**args)
        
        end_time = datetime.utcnow()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Send completion message
        if q:
            q.put({
                "type": "done",
                "status": "completed",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })
        
        RUNS[run_id] = {
            "status": "completed",
            "work_dir": result["work_dir"],
            "artifact_report": result["artifact_report"],
            "artifact_plot": result["artifact_plot"],
            "started_at": start_time.isoformat() + "Z",
            "completed_at": end_time.isoformat() + "Z",
            "duration_ms": duration_ms,
            "messages": result.get("messages", []),
            "config": config.model_dump(),
        }
        
        # Save to database
        save_run(
            run_id=run_id,
            status="completed",
            config=config.model_dump(),
            started_at=start_time.isoformat() + "Z",
            completed_at=end_time.isoformat() + "Z",
            duration_ms=duration_ms,
            artifact_report=result["artifact_report"],
            artifact_plot=result["artifact_plot"],
        )
        
    except Exception as e:
        if q:
            q.put({
                "type": "done",
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })
        
        RUNS[run_id] = {
            "status": "failed",
            "error": str(e),
            "started_at": start_time.isoformat() + "Z",
            "failed_at": datetime.utcnow().isoformat() + "Z",
            "config": config.model_dump(),
        }
        
        # Save failure to database
        save_run(
            run_id=run_id,
            status="failed",
            config=config.model_dump(),
            started_at=start_time.isoformat() + "Z",
            error=str(e),
        )


# ============================================================================
# FastAPI App
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_env()
    # Cleanup old workspaces on startup
    cleanup_old_workspaces()
    yield
    # cleanup if any


app = FastAPI(
    title="CognitionFlow API",
    description="Multi-agent RCA pipeline with user customization. Trigger analysis and retrieve artifacts.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================
@app.get("/", response_class=HTMLResponse)
def index():
    """Minimal web UI: run analysis and view report/plot."""
    return _INDEX_HTML


@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok", "concurrent_limit": MAX_CONCURRENT_RUNS}


@app.get("/config", response_model=ConfigResponse)
def get_config_options():
    """Get available configuration options for runs."""
    return ConfigResponse(
        models=AVAILABLE_MODELS,
        agent_modes=AGENT_MODES,
        defaults={
            "model": "llama-3.1-8b-instant",
            "temperature": 0.7,
            "anomaly_count": 5,
            "agent_mode": "standard",
        }
    )


@app.post("/run", response_model=RunResponse)
async def run_analysis(
    request: Request,
    background_tasks: BackgroundTasks,
    config: RunConfig = None,
):
    """
    Start the RCA workflow with optional custom configuration.
    Runs in background; use GET /runs/{run_id}/stream for SSE or GET /runs/{run_id} for status.
    """
    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429, 
            detail=f"Rate limit exceeded. Max {RATE_LIMIT_MAX} requests per {RATE_LIMIT_WINDOW}s."
        )
    
    # Check concurrent run limit
    if run_semaphore.locked():
        active_count = MAX_CONCURRENT_RUNS - run_semaphore._value
        if active_count >= MAX_CONCURRENT_RUNS:
            raise HTTPException(
                status_code=503,
                detail=f"Server busy. Max {MAX_CONCURRENT_RUNS} concurrent runs. Please try again."
            )
    
    # Cleanup old workspaces on each run request
    cleanup_old_workspaces()
    
    config = config or RunConfig()
    run_id = str(uuid.uuid4())
    base_dir = get_workspace_dir()
    work_dir = os.path.join(base_dir, run_id)
    os.makedirs(work_dir, exist_ok=True)

    # Create message queue for SSE streaming
    RUN_QUEUES[run_id] = queue.Queue()
    
    RUNS[run_id] = {
        "status": "running",
        "work_dir": work_dir,
        "started_at": datetime.utcnow().isoformat() + "Z",
        "messages": [],
        "current_phase": "initializing",
        "config": config.model_dump(),
    }
    
    # Save initial run state
    save_run(
        run_id=run_id,
        status="running",
        config=config.model_dump(),
        started_at=datetime.utcnow().isoformat() + "Z",
    )
    
    async def run_with_semaphore():
        async with run_semaphore:
            # Run in thread pool to avoid blocking
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                await asyncio.get_event_loop().run_in_executor(
                    executor, _run_sync, work_dir, run_id, config
                )
    
    background_tasks.add_task(lambda: asyncio.run(run_with_semaphore()))

    return RunResponse(
        run_id=run_id,
        status="started",
        message="Workflow started. Connect to GET /runs/{run_id}/stream for real-time updates.",
    )


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    """Get status and artifact paths for a run."""
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="Run not found")
    return RUNS[run_id]


@app.get("/runs/{run_id}/stream")
async def stream_run(run_id: str):
    """SSE endpoint: stream agent messages in real-time."""
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="Run not found")
    
    q = RUN_QUEUES.get(run_id)
    if not q:
        # If run is already done, return final status immediately
        run_state = RUNS.get(run_id, {})
        if run_state.get("status") in ("completed", "failed"):
            async def finished_generator():
                yield {
                    "event": "done",
                    "data": json.dumps({
                        "type": "done",
                        "status": run_state.get("status"),
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                    })
                }
            return EventSourceResponse(finished_generator())

        raise HTTPException(status_code=404, detail="Run queue not found")
    
    async def event_generator():
        try:
            while True:
                try:
                    # Get message with timeout to allow checking if run is done
                    try:
                        msg = q.get(timeout=1.0)
                    except queue.Empty:
                        # Check if run is complete
                        run_state = RUNS.get(run_id, {})
                        if run_state.get("status") in ("completed", "failed"):
                            yield {
                                "event": "done",
                                "data": json.dumps({
                                    "type": "done",
                                    "status": run_state.get("status"),
                                    "timestamp": datetime.utcnow().isoformat() + "Z",
                                })
                            }
                            break
                        continue
                    
                    if msg.get("type") == "done":
                        yield {
                            "event": "done",
                            "data": json.dumps(msg)
                        }
                        break
                    
                    yield {
                        "event": "message",
                        "data": json.dumps(msg)
                    }
                except Exception as e:
                    yield {
                        "event": "error",
                        "data": json.dumps({"error": str(e)})
                    }
                    break
        finally:
            # Cleanup queue when done
            if run_id in RUN_QUEUES:
                del RUN_QUEUES[run_id]
    
    return EventSourceResponse(event_generator())


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


@app.get("/history")
def get_history(limit: int = 20, offset: int = 0):
    """Get run history with pagination."""
    return {
        "runs": get_run_history(limit=limit, offset=offset),
        "limit": limit,
        "offset": offset,
    }


@app.get("/metrics")
def get_metrics():
    """Get aggregate metrics."""
    return db_get_metrics()


# ============================================================================
# Frontend HTML with Customization Panel
# ============================================================================
_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CognitionFlow - Multi-Agent RCA Pipeline</title>
  <meta name="description" content="CognitionFlow: A multi-agent AI system for automated server health analysis using Microsoft AutoGen and Groq LPU.">
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
      min-height: 100vh;
      color: #e2e8f0;
    }
    .container {
      max-width: 900px;
      margin: 0 auto;
      padding: 2rem 1.5rem;
    }
    header {
      text-align: center;
      margin-bottom: 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 1rem;
    }
    .logo {
      font-size: 2.5rem;
      font-weight: 700;
      background: linear-gradient(135deg, #60a5fa, #a78bfa);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .tagline {
      color: #94a3b8;
      font-size: 1.1rem;
    }
    .header-actions {
      display: flex;
      gap: 0.75rem;
    }
    .btn-secondary {
      background: rgba(59, 130, 246, 0.2);
      color: #60a5fa;
      border: 1px solid rgba(59, 130, 246, 0.3);
      padding: 0.6rem 1.2rem;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 500;
      transition: all 0.2s;
    }
    .btn-secondary:hover {
      background: rgba(59, 130, 246, 0.3);
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
    
    /* Customization Panel */
    .config-panel {
      margin: 1.5rem 0;
      padding: 1.5rem;
      background: rgba(15, 23, 42, 0.6);
      border-radius: 12px;
      text-align: left;
    }
    .config-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      cursor: pointer;
      padding-bottom: 0.5rem;
    }
    .config-header h3 {
      color: #f1f5f9;
      font-size: 1rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .config-toggle {
      color: #60a5fa;
      font-size: 1.2rem;
      transition: transform 0.2s;
    }
    .config-toggle.open {
      transform: rotate(180deg);
    }
    .config-body {
      display: none;
      padding-top: 1rem;
    }
    .config-body.active {
      display: block;
    }
    .form-group {
      margin-bottom: 1.25rem;
    }
    .form-group label {
      display: block;
      color: #cbd5e1;
      font-size: 0.9rem;
      margin-bottom: 0.5rem;
    }
    .form-group input, .form-group select, .form-group textarea {
      width: 100%;
      background: rgba(30, 41, 59, 0.8);
      border: 1px solid rgba(148, 163, 184, 0.2);
      border-radius: 8px;
      padding: 0.75rem;
      color: #e2e8f0;
      font-size: 0.95rem;
    }
    .form-group textarea {
      min-height: 100px;
      resize: vertical;
      font-family: 'Monaco', 'Menlo', monospace;
      font-size: 0.85rem;
    }
    .form-group input:focus, .form-group select:focus, .form-group textarea:focus {
      outline: none;
      border-color: #60a5fa;
    }
    .form-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
    }
    .slider-container {
      display: flex;
      align-items: center;
      gap: 1rem;
    }
    .slider-container input[type="range"] {
      flex: 1;
      -webkit-appearance: none;
      height: 6px;
      background: rgba(148, 163, 184, 0.3);
      border-radius: 3px;
    }
    .slider-container input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 18px;
      height: 18px;
      background: linear-gradient(135deg, #3b82f6, #8b5cf6);
      border-radius: 50%;
      cursor: pointer;
    }
    .slider-value {
      color: #60a5fa;
      font-weight: 600;
      min-width: 40px;
      text-align: right;
    }
    .radio-group {
      display: flex;
      gap: 0.75rem;
      flex-wrap: wrap;
    }
    .radio-option {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.5rem 1rem;
      background: rgba(30, 41, 59, 0.8);
      border: 1px solid rgba(148, 163, 184, 0.2);
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .radio-option:hover {
      border-color: rgba(59, 130, 246, 0.5);
    }
    .radio-option.selected {
      border-color: #60a5fa;
      background: rgba(59, 130, 246, 0.2);
    }
    .radio-option input {
      display: none;
    }
    
    /* Tooltips */
    .help-icon {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 16px;
      height: 16px;
      background: rgba(148, 163, 184, 0.2);
      color: #94a3b8;
      border-radius: 50%;
      font-size: 0.7rem;
      font-weight: bold;
      margin-left: 0.5rem;
      cursor: help;
      position: relative;
    }
    .help-icon:hover {
      background: #60a5fa;
      color: white;
    }
    .help-icon:hover::after {
      content: attr(data-tooltip);
      position: absolute;
      bottom: 150%;
      left: 50%;
      transform: translateX(-50%);
      background: #0f172a;
      color: #e2e8f0;
      padding: 0.5rem 0.75rem;
      border-radius: 6px;
      font-size: 0.75rem;
      white-space: nowrap;
      border: 1px solid rgba(148, 163, 184, 0.2);
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
      z-index: 100;
      width: max-content;
      max-width: 250px;
      white-space: normal;
      text-align: center;
      line-height: 1.4;
    }
    .help-icon:hover::before {
      content: '';
      position: absolute;
      bottom: 100%;
      left: 50%;
      transform: translateX(-50%);
      border-width: 6px;
      border-style: solid;
      border-color: #0f172a transparent transparent transparent;
      z-index: 100;
    }
    
    /* Conversation Panel */
    .conversation-panel {
      max-height: 500px;
      overflow-y: auto;
      background: rgba(15, 23, 42, 0.6);
      border-radius: 12px;
      padding: 1.5rem;
      margin-top: 1rem;
      display: none;
      text-align: left;
    }
    .conversation-panel.active {
      display: block;
    }
    .message {
      margin-bottom: 1.5rem;
      animation: fadeIn 0.3s ease;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .message-header {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      margin-bottom: 0.5rem;
    }
    .agent-badge {
      padding: 0.35rem 0.75rem;
      border-radius: 6px;
      font-size: 0.85rem;
      font-weight: 600;
    }
    .badge-pm {
      background: rgba(59, 130, 246, 0.2);
      color: #60a5fa;
    }
    .badge-eng {
      background: rgba(167, 139, 250, 0.2);
      color: #a78bfa;
    }
    .message-time {
      color: #64748b;
      font-size: 0.75rem;
    }
    .message-content {
      color: #cbd5e1;
      line-height: 1.6;
      word-wrap: break-word;
    }
    .message-content p { margin-bottom: 0.75rem; }
    .message-content ul, .message-content ol { margin-left: 1.5rem; margin-bottom: 0.75rem; }
    .message-content code {
      background: rgba(15, 23, 42, 0.5);
      padding: 0.2rem 0.4rem;
      border-radius: 4px;
      font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
      font-size: 0.85rem;
      color: #e2e8f0;
    }
    .message-content pre {
      background: #0f172a;
      border: 1px solid rgba(148, 163, 184, 0.2);
      border-radius: 8px;
      padding: 1rem;
      margin: 0.75rem 0;
      overflow-x: auto;
    }
    .message-content pre code {
      background: transparent;
      padding: 0;
      color: inherit;
    }
    .keyword { color: #c678dd; }
    .number { color: #d19a66; }
    .string { color: #98c379; }
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
    .modal {
      display: none;
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.8);
      z-index: 1000;
      overflow-y: auto;
    }
    .modal.active {
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 2rem;
    }
    .modal-content {
      background: #1e293b;
      border-radius: 16px;
      padding: 2rem;
      max-width: 800px;
      width: 100%;
      max-height: 90vh;
      overflow-y: auto;
      border: 1px solid rgba(148, 163, 184, 0.2);
    }
    .modal-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.5rem;
    }
    .modal-header h2 {
      color: #f1f5f9;
      font-size: 1.5rem;
    }
    .close-btn {
      background: transparent;
      border: none;
      color: #94a3b8;
      font-size: 1.5rem;
      cursor: pointer;
      padding: 0.5rem;
      line-height: 1;
    }
    .close-btn:hover {
      color: #e2e8f0;
    }
    .mermaid {
      background: rgba(15, 23, 42, 0.5);
      padding: 1.5rem;
      border-radius: 8px;
      margin: 1rem 0;
    }
    footer {
      text-align: center;
      margin-top: 2rem;
      color: #475569;
      font-size: 0.85rem;
    }
    footer a { color: #60a5fa; text-decoration: none; }
    footer a:hover { text-decoration: underline; }
    .typing-indicator {
      display: inline-flex;
      gap: 0.25rem;
      padding: 0.5rem 1rem;
      background: rgba(59, 130, 246, 0.1);
      border-radius: 8px;
      margin-top: 0.5rem;
    }
    .typing-dot {
      width: 8px;
      height: 8px;
      background: #60a5fa;
      border-radius: 50%;
      animation: typing 1.4s infinite;
    }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes typing {
      0%, 60%, 100% { transform: translateY(0); opacity: 0.7; }
      30% { transform: translateY(-10px); opacity: 1; }
    }
    @media (max-width: 640px) {
      .form-row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <div class="logo">CognitionFlow</div>
        <p class="tagline">Multi-Agent Root Cause Analysis Pipeline</p>
      </div>
      <div class="header-actions">
        <button class="btn-secondary" id="howItWorksBtn">How It Works</button>
      </div>
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
          <h3>Customizable Analysis</h3>
          <p>Configure prompts, models, temperature, and anomaly patterns</p>
        </div>
        <div class="feature">
          <h3>Production-Ready</h3>
          <p>Rate limiting, SQLite history, memory optimization for cloud</p>
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
        <span class="tech-badge">SQLite</span>
      </div>
    </div>

    <div class="card run-section">
      <h2>Run Analysis</h2>
      <p>Configure your analysis below, then click to start. Watch the agents collaborate in real-time.</p>
      
      <!-- Customization Panel -->
      <div class="config-panel">
        <div class="config-header" id="configToggle">
          <h3>⚙️ Advanced Options</h3>
          <span class="config-toggle" id="configArrow">▼</span>
        </div>
        <div class="config-body" id="configBody">
          <div class="form-group">
            <label>
              Custom Task Prompt
              <span class="help-icon" data-tooltip="Define specific scenarios or problems for the agents to solve. Use markdown for structure.">?</span>
            </label>
            <textarea id="taskPrompt" placeholder="**Mission:** Your custom analysis task...

**Tasks:**
1. First task...
2. Second task...

**Constraints:**
- Use Polars and Seaborn"></textarea>
          </div>
          
          <div class="form-row">
            <div class="form-group">
              <label>
                LLM Model
                <span class="help-icon" data-tooltip="Select the AI model. 70B models are smarter but slower. 8B models are faster.">?</span>
              </label>
              <select id="modelSelect">
                <option value="llama-3.1-8b-instant" selected>Llama 3.1 8B (Fast)</option>
                <option value="llama-3.3-70b-versatile">Llama 3.3 70B (Capable)</option>
                <option value="openai/gpt-oss-120b">GPT-OSS 120B (Reasoning)</option>
                <option value="qwen/qwen3-32b">Qwen 3 32B (Open)</option>
              </select>
            </div>
            <div class="form-group">
              <label>
                Anomaly Count
                <span class="help-icon" data-tooltip="Number of simulated error spikes to inject into the generated server logs.">?</span>
              </label>
              <select id="anomalySelect">
                <option value="3">3 anomalies</option>
                <option value="5" selected>5 anomalies</option>
                <option value="7">7 anomalies</option>
                <option value="10">10 anomalies</option>
              </select>
            </div>
          </div>
          
          <div class="form-group">
            <label>
              Temperature: <span id="tempValue">0.7</span>
              <span class="help-icon" data-tooltip="Controls randomness. 0.0 is deterministic (fact-based), 1.0 is creative (varied).">?</span>
            </label>
            <div class="slider-container">
              <span style="color: #64748b; font-size: 0.8rem;">Deterministic</span>
              <input type="range" id="tempSlider" min="0" max="1" step="0.1" value="0.7">
              <span style="color: #64748b; font-size: 0.8rem;">Creative</span>
            </div>
          </div>
          
          <div class="form-group">
            <label>
              Agent Mode
              <span class="help-icon" data-tooltip="Controls the verbosity and style of the agents. Standard is balanced, Detailed adds explanations.">?</span>
            </label>
            <div class="radio-group" id="agentModeGroup">
              <label class="radio-option selected">
                <input type="radio" name="agentMode" value="standard" checked>
                <span>Standard</span>
              </label>
              <label class="radio-option">
                <input type="radio" name="agentMode" value="detailed">
                <span>Detailed</span>
              </label>
              <label class="radio-option">
                <input type="radio" name="agentMode" value="concise">
                <span>Concise</span>
              </label>
            </div>
          </div>
        </div>
      </div>
      
      <button id="runBtn" class="run-btn" type="button">Start Analysis</button>
      <div id="status" class="status-box"></div>
      
      <div id="conversation" class="conversation-panel">
        <div id="messages"></div>
        <div id="typingIndicator" class="typing-indicator" style="display: none;">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>
      </div>
      
      <div id="links" class="results"></div>
    </div>

    <footer>
      Built by <a href="https://github.com/ishan-1010" target="_blank">Ishan Katoch</a> |
      <a href="https://github.com/ishan-1010/CognitionFlow" target="_blank">View on GitHub</a>
    </footer>
  </div>

  <!-- Architecture Modal -->
  <div id="modal" class="modal">
    <div class="modal-content">
      <div class="modal-header">
        <h2>How CognitionFlow Works</h2>
        <button class="close-btn" id="closeModal">&times;</button>
      </div>
      <div>
        <h3 style="color: #f1f5f9; margin-bottom: 1rem;">System Architecture</h3>
        <div class="mermaid">
sequenceDiagram
    participant Browser
    participant FastAPI
    participant PM as Product_Manager
    participant Eng as Senior_Engineer
    participant LLM as Groq LLM
    
    Browser->>FastAPI: POST /run {config}
    FastAPI-->>Browser: run_id
    Browser->>FastAPI: GET /runs/{id}/stream (SSE)
    
    FastAPI->>PM: initiate_chat()
    PM->>Eng: Task prompt
    Note over PM,Eng: AutoGen callback fires
    FastAPI-->>Browser: SSE: PM message
    
    Eng->>LLM: Generate code
    LLM-->>Eng: Python code
    Note over PM,Eng: AutoGen callback fires
    FastAPI-->>Browser: SSE: Engineer message
    
    PM->>PM: Execute code
    Note over PM,Eng: Execution callback
    FastAPI-->>Browser: SSE: Code output
    
    Eng->>PM: TERMINATE
    FastAPI-->>Browser: SSE: Complete
        </div>
        
        <h3 style="color: #f1f5f9; margin-top: 2rem; margin-bottom: 1rem;">v2.0 Features</h3>
        <ul style="color: #94a3b8; line-height: 2;">
          <li><strong style="color: #e2e8f0;">User Customization</strong>: Configure task prompts, models, temperature, and anomaly patterns</li>
          <li><strong style="color: #e2e8f0;">Memory Optimization</strong>: Concurrent run limiter, workspace cleanup, Agg backend</li>
          <li><strong style="color: #e2e8f0;">Rate Limiting</strong>: 10 requests per minute per IP</li>
          <li><strong style="color: #e2e8f0;">Run History</strong>: SQLite-backed persistence with /history endpoint</li>
          <li><strong style="color: #e2e8f0;">Metrics</strong>: /metrics endpoint for success rates and performance stats</li>
        </ul>
      </div>
    </div>
  </div>

  <script>
    mermaid.initialize({ startOnLoad: true, theme: 'dark' });
    
    const runBtn = document.getElementById('runBtn');
    const status = document.getElementById('status');
    const links = document.getElementById('links');
    const conversation = document.getElementById('conversation');
    const messages = document.getElementById('messages');
    const typingIndicator = document.getElementById('typingIndicator');
    const howItWorksBtn = document.getElementById('howItWorksBtn');
    const modal = document.getElementById('modal');
    const closeModal = document.getElementById('closeModal');
    
    // Config panel elements
    const configToggle = document.getElementById('configToggle');
    const configBody = document.getElementById('configBody');
    const configArrow = document.getElementById('configArrow');
    const taskPrompt = document.getElementById('taskPrompt');
    const modelSelect = document.getElementById('modelSelect');
    const anomalySelect = document.getElementById('anomalySelect');
    const tempSlider = document.getElementById('tempSlider');
    const tempValue = document.getElementById('tempValue');
    const agentModeGroup = document.getElementById('agentModeGroup');
    
    let eventSource = null;

    // Config panel toggle
    configToggle.addEventListener('click', function() {
      configBody.classList.toggle('active');
      configArrow.classList.toggle('open');
    });
    
    // Temperature slider
    tempSlider.addEventListener('input', function() {
      tempValue.textContent = this.value;
    });
    
    // Agent mode radio buttons
    agentModeGroup.querySelectorAll('.radio-option').forEach(option => {
      option.addEventListener('click', function() {
        agentModeGroup.querySelectorAll('.radio-option').forEach(o => o.classList.remove('selected'));
        this.classList.add('selected');
        this.querySelector('input').checked = true;
      });
    });

    function setStatus(msg, type) {
      status.textContent = msg;
      status.className = 'status-box' + (type ? ' ' + type : '');
    }

    function formatTime(timestamp) {
      if (!timestamp) return '';
      const date = new Date(timestamp);
      return date.toLocaleTimeString();
    }

    function escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }

    marked.use({
      renderer: {
        code(code, language) {
          const highlighted = highlightCode(code);
          return `<pre><code class="language-${language}">${highlighted}</code></pre>`;
        }
      }
    });

    function highlightCode(code) {
      return escapeHtml(code)
        .replace(/(import|from|def|class|if|elif|else|for|while|return|try|except|with|as|in|and|or|not|True|False|None)\\b/g, '<span class="keyword">$1</span>')
        .replace(/(\\b\\d+\\.?\\d*\\b)/g, '<span class="number">$1</span>')
        .replace(/(".*?"|'.*?')/g, '<span class="string">$1</span>');
    }

    function addMessage(msgData) {
      const msgDiv = document.createElement('div');
      msgDiv.className = 'message';
      
      const sender = msgData.sender || 'Unknown';
      const isPM = sender.includes('Product_Manager') || sender.includes('PM');
      const badgeClass = isPM ? 'badge-pm' : 'badge-eng';
      const badgeText = isPM ? 'Product Manager' : 'Senior Engineer';
      
      let contentHtml = marked.parse(msgData.content || '');
      
      msgDiv.innerHTML = `
        <div class="message-header">
          <span class="agent-badge ${badgeClass}">${badgeText}</span>
          <span class="message-time">${formatTime(msgData.timestamp)}</span>
        </div>
        <div class="message-content">${contentHtml}</div>
      `;
      
      messages.appendChild(msgDiv);
      conversation.scrollTop = conversation.scrollHeight;
      conversation.classList.add('active');
    }

    function streamRun(runId) {
      const base = window.location.origin;
      eventSource = new EventSource(base + '/runs/' + runId + '/stream');
      
      typingIndicator.style.display = 'flex';
      
      eventSource.addEventListener('message', function(e) {
        try {
          const msg = JSON.parse(e.data);
          
          if (msg.type === 'phase_change') {
            setStatus(msg.message || 'Phase: ' + msg.phase);
          } else if (msg.type === 'agent_message' || msg.type === 'code_generation' || msg.type === 'termination') {
            addMessage(msg);
            typingIndicator.style.display = 'none';
          }
        } catch (err) {
          console.error('Error parsing SSE message:', err);
        }
      });
      
      eventSource.addEventListener('done', function(e) {
        try {
          const data = JSON.parse(e.data);
          typingIndicator.style.display = 'none';
          
          if (data.status === 'completed') {
            setStatus('Analysis complete!', 'success');
            links.innerHTML =
              '<a class="result-link" href="' + base + '/runs/' + runId + '/incident_report" target="_blank">View Incident Report</a>' +
              '<a class="result-link" href="' + base + '/runs/' + runId + '/server_health.png" target="_blank">View Health Plot</a>';
            runBtn.disabled = false;
          } else if (data.status === 'failed') {
            setStatus('Analysis failed: ' + (data.error || 'Unknown error'), 'error');
            runBtn.disabled = false;
          }
          
          if (eventSource) {
            eventSource.close();
            eventSource = null;
          }
        } catch (err) {
          console.error('Error parsing done event:', err);
        }
      });
      
      eventSource.addEventListener('error', function(e) {
        console.error('SSE error:', e);
        typingIndicator.style.display = 'none';
        if (eventSource) {
          eventSource.close();
          eventSource = null;
        }
      });
    }

    function getConfig() {
      const selectedMode = agentModeGroup.querySelector('input:checked');
      return {
        task_prompt: taskPrompt.value.trim() || null,
        model: modelSelect.value,
        temperature: parseFloat(tempSlider.value),
        anomaly_count: parseInt(anomalySelect.value),
        agent_mode: selectedMode ? selectedMode.value : 'standard'
      };
    }

    runBtn.addEventListener('click', async function() {
      runBtn.disabled = true;
      links.innerHTML = '';
      messages.innerHTML = '';
      conversation.classList.remove('active');
      
      const config = getConfig();
      setStatus(`Initializing agents (${config.model})...`);
      
      try {
        const r = await fetch(window.location.origin + '/run', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(config)
        });
        const data = await r.json();
        if (!r.ok) {
          setStatus('Error: ' + (data.detail || r.status), 'error');
          runBtn.disabled = false;
          return;
        }
        setStatus('Agents working... (watch conversation below)');
        streamRun(data.run_id);
      } catch (e) {
        setStatus('Error: ' + e.message, 'error');
        runBtn.disabled = false;
      }
    });

    howItWorksBtn.addEventListener('click', function() {
      modal.classList.add('active');
    });

    closeModal.addEventListener('click', function() {
      modal.classList.remove('active');
    });

    modal.addEventListener('click', function(e) {
      if (e.target === modal) {
        modal.classList.remove('active');
      }
    });
  </script>
</body>
</html>
"""
