"""
FastAPI service for CognitionFlow: trigger RCA workflow and return artifact paths.
Enhanced with user customization, memory optimization, and production features.
"""
# Force Agg backend before any matplotlib imports (memory optimization)
import matplotlib
matplotlib.use('Agg')

import gc
import os
import uuid
import json
import queue
import asyncio
import shutil
import resource
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional, List

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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
from cognitionflow.orchestration import run_workflow, get_template_prompt

# Import database functions
from api.db import save_run, get_run_history, get_metrics as db_get_metrics, get_run_by_id

logger = logging.getLogger(__name__)

# ============================================================================
# Memory Cap: 500 MB (app-level enforcement)
# ============================================================================
# NOTE: We do NOT use resource.setrlimit(RLIMIT_AS) because RLIMIT_AS caps
# *virtual* address space, not physical RAM.  Python + numpy + pydantic alone
# map ~1-2 GB of virtual memory at import time while only using ~150 MB RSS.
# Setting RLIMIT_AS to 500 MB causes MemoryError during pydantic import.
# Instead we enforce the cap at the application level: single concurrent run,
# gc.collect() after every run, and a pre-run memory check.
MEMORY_CAP_MB = int(os.environ.get("COGNITIONFLOW_MEM_CAP_MB", "500"))

def _get_process_memory_mb() -> float:
    """Return current peak RSS of the process in MB."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in KB on Linux, bytes on macOS
        if sys.platform == "darwin":
            return usage.ru_maxrss / (1024 * 1024)
        return usage.ru_maxrss / 1024
    except Exception:
        return 0.0

# ============================================================================
# Memory Optimization: Concurrent Run Limiter
# ============================================================================
MAX_CONCURRENT_RUNS = 1  # Strict: one run at a time to stay under 500 MB
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
        
        task_prompt = config.task_prompt
        if config.template_id and not task_prompt:
            task_prompt = get_template_prompt(config.template_id)
        
        result = run_workflow(
            task_prompt=task_prompt,
            work_dir=work_dir,
            on_message=on_message,
            model=config.model,
            temperature=config.temperature,
            agent_mode=config.agent_mode,
        )
        
        end_time = datetime.utcnow()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Derive primary report/plot from dynamic artifacts (backward compat)
        artifacts = result.get("artifacts", [])
        artifact_report = None
        artifact_plot = None
        for a in artifacts:
            p = a.get("path", "")
            if p.endswith(".md") and artifact_report is None:
                artifact_report = p
            if p.endswith(".png") and artifact_plot is None:
                artifact_plot = p
        
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
            "artifact_report": artifact_report,
            "artifact_plot": artifact_plot,
            "artifacts": artifacts,
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
            artifact_report=artifact_report,
            artifact_plot=artifact_plot,
        )
        
    except Exception as e:
        logger.warning("Run %s error: %s", run_id, e)
        end_time = datetime.utcnow()
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        # Graceful degradation: check if artifacts were generated despite error
        from cognitionflow.orchestration import discover_artifacts
        artifacts = discover_artifacts(work_dir) if os.path.isdir(work_dir) else []

        if artifacts:
            # Artifacts exist → mark as completed with warning
            artifact_report = next((a["path"] for a in artifacts if a["path"].endswith(".md")), None)
            artifact_plot = next((a["path"] for a in artifacts if a["path"].endswith(".png")), None)

            if q:
                q.put({
                    "type": "done",
                    "status": "completed",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                })

            RUNS[run_id] = {
                "status": "completed",
                "work_dir": work_dir,
                "artifact_report": artifact_report,
                "artifact_plot": artifact_plot,
                "artifacts": artifacts,
                "started_at": start_time.isoformat() + "Z",
                "completed_at": end_time.isoformat() + "Z",
                "duration_ms": duration_ms,
                "warning": str(e),
                "config": config.model_dump(),
            }
            save_run(
                run_id=run_id, status="completed", config=config.model_dump(),
                started_at=start_time.isoformat() + "Z",
                completed_at=end_time.isoformat() + "Z",
                duration_ms=duration_ms,
                artifact_report=artifact_report, artifact_plot=artifact_plot,
            )
        else:
            # No artifacts → true failure
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
                "failed_at": end_time.isoformat() + "Z",
                "config": config.model_dump(),
            }
            save_run(
                run_id=run_id, status="failed", config=config.model_dump(),
                started_at=start_time.isoformat() + "Z", error=str(e),
            )
    finally:
        # Aggressive memory cleanup after every run
        gc.collect()
        mem_mb = _get_process_memory_mb()
        logger.info("Run %s finished — memory: %.0f MB / %d MB cap", run_id, mem_mb, MEMORY_CAP_MB)


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
    description="Multi-agent pipeline with review loop. Three agents collaborate to produce verified artifacts.",
    version="3.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=os.path.join(_here, "static")), name="static")

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
@app.get("/", response_class=FileResponse)
def index():
    """Minimal web UI: run analysis and view report/plot."""
    return FileResponse(os.path.join(_here, "static", "index.html"))


@app.get("/health")
def health():
    """Liveness check with memory stats."""
    mem_mb = _get_process_memory_mb()
    return {
        "status": "ok",
        "concurrent_limit": MAX_CONCURRENT_RUNS,
        "memory_mb": round(mem_mb, 1),
        "memory_cap_mb": MEMORY_CAP_MB,
    }


@app.get("/config", response_model=ConfigResponse)
def get_config_options():
    """Get available configuration options for runs."""
    return ConfigResponse(
        models=AVAILABLE_MODELS,
        agent_modes=AGENT_MODES,
        task_templates=TASK_TEMPLATES,
        output_formats=OUTPUT_FORMATS,
        defaults={
            "model": "llama-3.1-8b-instant",
            "temperature": 0.7,
            "agent_mode": "standard",
            "template_id": "data_analysis",
            "output_format": "markdown",
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
    
    # Memory guard: reject if approaching the cap
    current_mem = _get_process_memory_mb()
    if current_mem > MEMORY_CAP_MB * 0.85:
        gc.collect()  # try to reclaim first
        current_mem = _get_process_memory_mb()
        if current_mem > MEMORY_CAP_MB * 0.85:
            raise HTTPException(
                status_code=503,
                detail=f"Memory pressure: {current_mem:.0f} MB / {MEMORY_CAP_MB} MB cap. Try again shortly."
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
    # Check in-memory first
    if run_id in RUNS:
        return RUNS[run_id]
        
    # Fallback to DB
    run_data = get_run_by_id(run_id)
    if run_data:
        # Re-hydrate needed fields if necessary, or just return as is
        # Note: DB stores config as JSON string, need to ensure compatibility
        if run_data.get("config") and isinstance(run_data["config"], str):
             import json
             try:
                 run_data["config"] = json.loads(run_data["config"])
             except:
                 pass
        return run_data

    raise HTTPException(status_code=404, detail="Run not found")



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
    
    def _blocking_get(q, timeout):
        """Wrapper so we can call q.get in a thread executor."""
        try:
            return q.get(timeout=timeout)
        except queue.Empty:
            return None

    async def event_generator():
        loop = asyncio.get_event_loop()
        try:
            while True:
                try:
                    # Non-blocking queue read: run the blocking q.get in a
                    # thread so the event-loop stays free to flush SSE chunks.
                    msg = await loop.run_in_executor(
                        None, _blocking_get, q, 1.0
                    )

                    if msg is None:
                        # Timeout — check if run finished while we waited
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
    """Serve server_health.png for a run (backward compat)."""
    if run_id not in RUNS or RUNS[run_id].get("status") != "completed":
        raise HTTPException(status_code=404, detail="Run not found or not completed")
    path = RUNS[run_id].get("artifact_plot")
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path, media_type="image/png")


@app.get("/runs/{run_id}/artifacts/{filename:path}")
def get_run_artifact(run_id: str, filename: str):
    """Serve any discovered artifact by name (dynamic artifact discovery)."""
    if run_id not in RUNS or RUNS[run_id].get("status") != "completed":
        raise HTTPException(status_code=404, detail="Run not found or not completed")
    artifacts = RUNS[run_id].get("artifacts", [])
    for a in artifacts:
        if os.path.basename(a.get("path", "")) == filename or a.get("name") == filename:
            path = a["path"]
            if os.path.isfile(path):
                ext = os.path.splitext(filename)[1].lower()
                media_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                               ".md": "text/markdown", ".json": "application/json", ".txt": "text/plain",
                               ".html": "text/html", ".csv": "text/csv", ".py": "text/x-python"}
                return FileResponse(path, media_type=media_types.get(ext, "application/octet-stream"))
    raise HTTPException(status_code=404, detail="Artifact not found")


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
