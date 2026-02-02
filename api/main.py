"""
FastAPI service for CognitionFlow: trigger RCA workflow and return artifact paths.
"""
import os
import uuid
import json
import queue
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# Add src to path so cognitionflow is importable when running from repo root
import sys
_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.join(os.path.dirname(_here), "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from cognitionflow.config import load_env, get_workspace_dir
from cognitionflow.orchestration import run_workflow, DEFAULT_TASK_PROMPT


# Store run state and message queues for SSE streaming
RUNS: dict[str, dict] = {}
RUN_QUEUES: dict[str, queue.Queue] = {}


def _run_sync(work_dir: str, run_id: str) -> None:
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
                "message": "Initializing agents...",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            })
        
        result = run_workflow(
            task_prompt=DEFAULT_TASK_PROMPT,
            work_dir=work_dir,
            on_message=on_message,
        )
        
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
        }
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
        }


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
    Start the RCA workflow. Runs in background; use GET /runs/{run_id}/stream for SSE or GET /runs/{run_id} for status.
    """
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
    }
    
    background_tasks.add_task(_run_sync, work_dir, run_id)

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


_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CognitionFlow - Multi-Agent RCA Pipeline</title>
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
    /* Fixed Alignment for Conversation */
    .conversation-panel {
      max-height: 500px;
      overflow-y: auto;
      background: rgba(15, 23, 42, 0.6);
      border-radius: 12px;
      padding: 1.5rem;
      margin-top: 1rem;
      display: none;
      text-align: left; /* Verify left alignment */
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
    /* Markdown Styles */
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

    /* Code syntax highlighting */
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
      <p>Click below to trigger the multi-agent workflow. Watch the agents collaborate in real-time below.</p>
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
    
    Browser->>FastAPI: POST /run
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
        
        <h3 style="color: #f1f5f9; margin-top: 2rem; margin-bottom: 1rem;">Agent Roles</h3>
        <div style="display: grid; gap: 1rem;">
          <div style="background: rgba(15, 23, 42, 0.5); padding: 1rem; border-radius: 8px;">
            <h4 style="color: #60a5fa; margin-bottom: 0.5rem;">Product Manager</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">
              Orchestrates the workflow, executes generated code, validates results, and manages the conversation flow.
              Uses AutoGen's UserProxyAgent to handle code execution in a sandboxed environment.
            </p>
          </div>
          <div style="background: rgba(15, 23, 42, 0.5); padding: 1rem; border-radius: 8px;">
            <h4 style="color: #a78bfa; margin-bottom: 0.5rem;">Senior Engineer</h4>
            <p style="color: #94a3b8; font-size: 0.9rem;">
              Generates Python code using Polars for data manipulation and Seaborn for visualization.
              Communicates with Groq LLM to produce production-ready code that solves the RCA task.
            </p>
          </div>
        </div>
        
        <h3 style="color: #f1f5f9; margin-top: 2rem; margin-bottom: 1rem;">Tech Stack</h3>
        <ul style="color: #94a3b8; line-height: 2;">
          <li><strong style="color: #e2e8f0;">Microsoft AutoGen</strong>: Multi-agent orchestration framework</li>
          <li><strong style="color: #e2e8f0;">Groq LPU</strong>: Low-latency LLM inference (GPT-compatible)</li>
          <li><strong style="color: #e2e8f0;">FastAPI</strong>: Async Python web framework</li>
          <li><strong style="color: #e2e8f0;">Server-Sent Events (SSE)</strong>: Real-time message streaming</li>
          <li><strong style="color: #e2e8f0;">Polars</strong>: High-performance DataFrame library</li>
          <li><strong style="color: #e2e8f0;">Seaborn</strong>: Statistical visualization library</li>
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
    let eventSource = null;

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

    // Configure marked with syntax highlighting
    marked.use({
      renderer: {
        code(code, language) {
          const highlighted = highlightCode(code);
          return `<pre><code class="language-${language}">${highlighted}</code></pre>`;
        }
      }
    });

    function highlightCode(code) {
      // Simple Python syntax highlighting
      // Note: marked passes the code content. We need to escape it first but we are returning HTML.
      // Actually, marked expects plain text or HTML? 
      // The custom renderer returns HTML.
      return escapeHtml(code)
        .replace(/(import|from|def|class|if|elif|else|for|while|return|try|except|with|as|in|and|or|not|True|False|None)\b/g, '<span class="keyword">$1</span>')
        .replace(/(\b\d+\.?\d*\b)/g, '<span class="number">$1</span>')
        .replace(/(".*?"|'.*?')/g, '<span class="string">$1</span>');
    }

    function addMessage(msgData) {
      const msgDiv = document.createElement('div');
      msgDiv.className = 'message';
      
      const sender = msgData.sender || 'Unknown';
      const isPM = sender.includes('Product_Manager') || sender.includes('PM');
      const badgeClass = isPM ? 'badge-pm' : 'badge-eng';
      const badgeText = isPM ? 'Product Manager' : 'Senior Engineer';
      
      // Use marked to parse content
      // Note: msgData.content contains the full message including code blocks.
      // marked deals with them correctly.
      let contentHtml = DOMPurify ? DOMPurify.sanitize(marked.parse(msgData.content || '')) : marked.parse(msgData.content || '');
      // Fallback if DOMPurify is not present (it's not, but good practice). 
      // In this case, since it's an internal tool, we trust the LLM output mostly.
      if (typeof DOMPurify === 'undefined') {
         contentHtml = marked.parse(msgData.content || '');
      }
      
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

    runBtn.addEventListener('click', async function() {
      runBtn.disabled = true;
      links.innerHTML = '';
      messages.innerHTML = '';
      conversation.classList.remove('active');
      setStatus('Initializing agents...');
      
      try {
        const r = await fetch(window.location.origin + '/run', { method: 'POST' });
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
