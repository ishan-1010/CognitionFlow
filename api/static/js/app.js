
document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const form = document.getElementById('runForm');
    const runBtn = document.getElementById('runBtn');
    const msgContainer = document.getElementById('msgContainer');
    const scrollAnchor = document.getElementById('scrollAnchor');
    const connectionStatus = document.getElementById('connectionStatus');
    const tempInput = document.getElementById('temperature');
    const tempValue = document.getElementById('tempValue');
    const agentPlan = document.getElementById('agentPlan');
    const agentPlanText = document.getElementById('agentPlanText');

    // Config Data State
    let configData = null;

    // Initialize Markdown and Mermaid
    marked.setOptions({
        highlight: function (code, lang) {
            return code; // Syntax highlighting handled by CSS classes for now
        }
    });
    mermaid.initialize({
        startOnLoad: false,
        theme: 'dark',
        fontFamily: 'Inter',
        securityLevel: 'loose'
    });

    // 1. Fetch Configuration & Init
    async function init() {
        try {
            const res = await fetch('/config');
            configData = await res.json();
            populateOptions(configData);
            updatePlanPreview(); // Initial preview
        } catch (e) {
            console.error('Failed to load config:', e);
            logSystemMessage('Connection to backend failed. Verification required.', 'error');
        }
    }

    // 2. Populate Dropdowns
    function populateOptions(data) {
        // Models
        const modelSelect = document.getElementById('model');
        data.models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.id;
            opt.textContent = m.name || m.id;
            if (m.id === data.defaults.model) opt.selected = true;
            modelSelect.appendChild(opt);
        });

        // Templates
        const templateSelect = document.getElementById('template_id');
        data.task_templates.forEach(t => {
            const opt = document.createElement('option');
            opt.value = t.id;
            opt.textContent = t.name.replace(/^[^\w]+/, '').trim(); // Remove emojis from name if present
            opt.dataset.description = t.description;
            if (t.id === data.defaults.template_id) opt.selected = true;
            templateSelect.appendChild(opt);
        });

        // Agent Modes
        const modeSelect = document.getElementById('agent_mode');
        data.agent_modes.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.id;
            opt.textContent = m.name;
            if (m.id === data.defaults.agent_mode) opt.selected = true;
            modeSelect.appendChild(opt);
        });

        // Output Formats
        const outputSelect = document.getElementById('output_format');
        data.output_formats.forEach(f => {
            const opt = document.createElement('option');
            opt.value = f.id;
            opt.textContent = f.name;
            if (f.id === data.defaults.output_format) opt.selected = true;
            outputSelect.appendChild(opt);
        });
    }

    // 3. UX Interactions
    // Update Temperature Display
    tempInput.addEventListener('input', (e) => {
        tempValue.textContent = e.target.value;
    });

    // Update Plan Preview on Selection Change
    document.getElementById('template_id').addEventListener('change', updatePlanPreview);

    function updatePlanPreview() {
        if (!configData) return;

        const templateId = document.getElementById('template_id').value;
        const template = configData.task_templates.find(t => t.id === templateId);

        if (template) {
            agentPlan.classList.remove('hidden');
            let plan = `<strong>Objective:</strong> ${template.description}<br>`;
            plan += `<span class="text-subtle">Agents will analyze inputs and generate compliant artifacts according to parameters.</span>`;
            agentPlanText.innerHTML = plan;
        }
    }

    // 4. Handle Form Submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Clear previous runs and welcome state
        msgContainer.innerHTML = '';
        document.getElementById('artifactsContainer').classList.add('hidden');
        document.getElementById('artifactsList').innerHTML = '';



        // UI State: Busy
        runBtn.disabled = true;
        runBtn.innerHTML = '<span>Initializing Workflow...</span>';
        connectionStatus.textContent = 'Pipeline Active';

        // Build Payload
        const formData = new FormData(form);
        const payload = {
            template_id: formData.get('template_id'),
            task_prompt: formData.get('task_prompt') || null,
            model: formData.get('model'),
            temperature: parseFloat(formData.get('temperature')),
            agent_mode: formData.get('agent_mode'),
            output_format: formData.get('output_format'),
        };

        try {
            // Start Run
            const res = await fetch('/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!res.ok) throw new Error((await res.json()).detail || 'Run failed');

            const data = await res.json();
            const runId = data.run_id;

            logSystemMessage(`Pipeline Initialized (ID: ${runId})`);

            // Connect to Stream
            connectStream(runId);

        } catch (err) {
            console.error(err);
            logSystemMessage(`Initialization failed: ${err.message}`, 'error');
            resetUIState();
        }
    });

    // 5. SSE Streaming
    function connectStream(runId) {
        const eventSource = new EventSource(`/runs/${runId}/stream`);

        eventSource.addEventListener('message', (e) => {
            const msg = JSON.parse(e.data);
            renderMessage(msg);
        });

        eventSource.addEventListener('done', (e) => {
            const data = JSON.parse(e.data);
            logSystemMessage(`Pipeline Execution Completed (Status: ${data.status})`);
            eventSource.close();
            resetUIState();

            // Fetch final artifacts if successful
            if (data.status === 'completed') {
                fetchArtifacts(runId);
            }
        });

        eventSource.addEventListener('error', (e) => {
            console.error('Stream error', e);
            eventSource.close();
            resetUIState();
        });
    }

    // 6. Rendering
    function renderMessage(msg) {
        const div = document.createElement('div');
        div.className = 'message';

        let contentHtml = '';
        let role = 'System';
        let roleClass = '';

        if (msg.type === 'phase_change') {
            role = 'Orchestrator';
            roleClass = 'system';
            contentHtml = `<p><strong>PHASE: ${msg.phase.toUpperCase()}</strong> &mdash; ${msg.message}</p>`;
        } else if (msg.name) {
            role = msg.name.replace('_', ' ');
            contentHtml = marked.parse(msg.content || '');
        } else {
            contentHtml = `<pre>${JSON.stringify(msg, null, 2)}</pre>`;
        }

        // Role-based badge styling
        let badgeClass = 'role-badge';
        if (role === 'Engineer') badgeClass += ' role-engineer';
        else if (role === 'Reviewer') badgeClass += ' role-reviewer';
        else if (role === 'Executor') badgeClass += ' role-executor';

        // Special card styling for approved reviews
        let cardClass = `message-card ${roleClass} prose`;
        if (msg.type === 'review_approved') cardClass += ' approved';

        div.innerHTML = `
            <div class="message-header">
                <span class="${badgeClass}">${role}</span>
                <span class="timestamp">${new Date().toLocaleTimeString()}</span>
            </div>
            <div class="${cardClass}">
                ${contentHtml}
            </div>
        `;

        msgContainer.appendChild(div);

        // Render mermaid if present
        if (contentHtml.includes('mermaid')) {
            setTimeout(() => {
                mermaid.init(undefined, div.querySelectorAll('.mermaid'));
            }, 100);
        }

        scrollToBottom();
    }

    function logSystemMessage(text, type = 'info') {
        const div = document.createElement('div');
        div.className = `message ${type}`;

        div.innerHTML = `
            <div class="message-header">
                <span class="role-badge">System</span>
                <span class="timestamp">${new Date().toLocaleTimeString()}</span>
            </div>
            <div class="message-card">
                <p>${text}</p>
            </div>
        `;
        msgContainer.appendChild(div);
        scrollToBottom();
    }

    async function fetchArtifacts(run_id) {
        try {
            const res = await fetch(`/runs/${run_id}`);
            if (!res.ok) return;
            const data = await res.json();

            const artifactsContainer = document.getElementById('artifactsContainer');
            const artifactsList = document.getElementById('artifactsList');

            if (data.artifacts && data.artifacts.length > 0) {
                artifactsContainer.classList.remove('hidden');
                artifactsList.innerHTML = ''; // Clear old ones

                data.artifacts.forEach(a => {
                    const filename = a.path.split('/').pop();
                    const url = `/runs/${run_id}/artifacts/${filename}`;

                    const item = document.createElement('a');
                    item.href = url;
                    item.target = '_blank';
                    item.className = 'status-item';
                    item.style.textDecoration = 'none';
                    item.style.padding = '0.5rem';
                    item.style.borderRadius = '6px';
                    item.style.background = 'var(--bg-secondary)';
                    item.style.fontSize = '0.85rem';

                    // Simple Icon based on type
                    let icon = '<svg class="icon" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>';
                    if (a.type === 'image') {
                        icon = '<svg class="icon" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>';
                    }

                    item.innerHTML = `${icon} <span style="margin-left: 0.5rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${filename}</span>`;
                    artifactsList.appendChild(item);
                });
            }
        } catch (e) {
            console.error('Artifact fetch failed', e);
        }
    }


    function scrollToBottom() {
        if (scrollAnchor) {
            scrollAnchor.scrollIntoView({ behavior: 'smooth' });
        }
    }

    function resetUIState() {
        runBtn.disabled = false;
        runBtn.innerHTML = `
            <svg class="icon" style="width: 14px; height: 14px; margin-right: 0.5rem;" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
            <span>Execute Pipeline</span>
        `;
        connectionStatus.textContent = 'System Ready';
    }

    // Cleanup Clear Button
    const clearBtn = document.getElementById('clearBtn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            msgContainer.innerHTML = '';
            // Restore empty state if cleared
            const emptyState = document.createElement('div');
            emptyState.className = 'empty-state';
            emptyState.innerHTML = `
                <svg class="icon" style="width: 48px; height: 48px; color: var(--border-active);" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
                <p>Output console cleared. Ready for new tasks.</p>
            `;
            msgContainer.appendChild(emptyState);
        });
    }

    // Init App
    init();
});
