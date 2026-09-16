const api = {
  status: "/rag/status",
  knowledgeBases: "/rag/knowledge-bases",
  index: "/rag/index",
  ask: "/rag/ask",
  retrieve: "/rag/retrieve",
  reset: "/rag/reset",
  evaluationSummary: "/evaluation/summary",
};

const state = {
  files: [],
  knowledgeBases: [],
  activeKnowledgeBaseId: "default",
};

const sampleQuestions = [
  "What is the minimum attendance requirement in a course?",
  "What happens if attendance is below required minimum?",
  "Who can relax attendance and by how much?",
  "Summarize attendance policy and penalties in one answer.",
  "What confidentiality obligations are mentioned before patent filing?",
];

const $ = (id) => document.getElementById(id);

function bind(id, eventName, handler) {
  const element = $(id);
  if (!element) {
    console.warn(`Missing UI element: ${id}`);
    return;
  }
  element.addEventListener(eventName, handler);
}

function showToast(message) {
  const toast = $("toast");
  if (!toast) {
    console.warn(message);
    return;
  }
  toast.textContent = message;
  toast.classList.remove("hidden");
  window.setTimeout(() => toast.classList.add("hidden"), 3600);
}

function showKnowledgeBaseStatus(message) {
  const status = $("kb-operation-status");
  if (!status) {
    showToast(message);
    return;
  }
  status.textContent = message;
  status.classList.remove("hidden");
}

function hideKnowledgeBaseStatus() {
  window.setTimeout(() => $("kb-operation-status")?.classList.add("hidden"), 1800);
}

function setLoading(isLoading) {
  [
    "index-docs",
    "ask-question",
    "preview-retrieval",
    "compare-question",
    "reset-index",
    "refresh-status",
    "refresh-kbs",
    "create-kb",
    "rename-kb",
    "delete-kb",
  ].forEach((id) => {
    const element = $(id);
    if (element) {
      element.disabled = isLoading;
    }
  });
}

function setOperationStatus(kind, message, percent, isVisible = true) {
  const status = $(`${kind}-status`);
  const text = $(`${kind}-status-text`);
  const percentText = $(`${kind}-status-percent`);
  const progress = $(`${kind}-progress`);

  if (!status || !text || !percentText || !progress) {
    return;
  }

  status.classList.toggle("hidden", !isVisible);
  text.textContent = message;
  percentText.textContent = `${percent}%`;
  progress.style.width = `${percent}%`;
}

function hideOperationStatus(kind) {
  window.setTimeout(() => setOperationStatus(kind, "", 0, false), 900);
}

async function requestJson(path, options = {}) {
  const timeoutMs = options.timeoutMs ?? 180000;
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(path, { ...options, signal: controller.signal });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`${response.status} ${response.statusText}: ${text}`);
    }
    return response.json();
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error("Request timed out. Check backend logs or use offline hash embeddings.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

function formatNumber(value, digits = 3) {
  const number = Number(value || 0);
  return number.toFixed(digits);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function currentPayload() {
  return {
    question: $("question").value.trim(),
    mode: $("mode").value,
    top_k: Number($("top-k").value || 5),
    knowledge_base_id: state.activeKnowledgeBaseId,
  };
}

function syncModeLabels() {
  const mode = $("mode").value;
  $("active-mode").textContent = mode;
  $("hero-mode").textContent = mode === "baseline" ? "Baseline" : "Proposed";
}

function updateFileList() {
  const list = $("file-list");
  $("file-count").textContent = `${state.files.length} files`;
  if (!state.files.length) {
    list.textContent = "No files selected.";
    return;
  }
  list.textContent = state.files.map((file) => file.name).join(" | ");
}

function syncIndexModeLabel() {
  const appendMode = $("append-index")?.checked;
  $("index-docs").textContent = appendMode ? "Add Documents" : "Rebuild Index";
}

function activeKnowledgeBase() {
  return state.knowledgeBases.find((kb) => kb.knowledge_base_id === state.activeKnowledgeBaseId);
}

function knowledgeBaseQuery() {
  return `knowledge_base_id=${encodeURIComponent(state.activeKnowledgeBaseId)}`;
}

function renderKnowledgeBases() {
  const select = $("knowledge-base-select");
  if (!state.knowledgeBases.length) {
    select.innerHTML = '<option value="default">Default Knowledge Base</option>';
    state.activeKnowledgeBaseId = "default";
  } else {
    select.innerHTML = state.knowledgeBases
      .map(
        (kb) => `
          <option value="${escapeHtml(kb.knowledge_base_id)}">
            ${escapeHtml(kb.name)} (${kb.document_count || 0} docs, ${kb.total_chunks || 0} chunks)
          </option>
        `,
      )
      .join("");
  }

  select.value = state.activeKnowledgeBaseId;
  const active = activeKnowledgeBase();
  $("kb-documents").textContent = `${active?.document_count || 0} documents`;
  $("kb-chunks").textContent = `${active?.total_chunks || 0} chunks`;
  $("new-kb-name").placeholder = active?.name || "Example: History textbook";
}

async function loadKnowledgeBases() {
  try {
    const result = await requestJson(api.knowledgeBases);
    state.knowledgeBases = result.knowledge_bases || [];
    state.activeKnowledgeBaseId = result.active_knowledge_base_id || "default";
    renderKnowledgeBases();
  } catch (error) {
    showToast(`Knowledge-base list failed: ${error.message}`);
  }
}

async function selectKnowledgeBase(knowledgeBaseId) {
  if (!knowledgeBaseId) {
    return;
  }

  setLoading(true);
  try {
    await requestJson(`${api.knowledgeBases}/${encodeURIComponent(knowledgeBaseId)}/select`, {
      method: "POST",
    });
    state.activeKnowledgeBaseId = knowledgeBaseId;
    renderAnswer(null);
    renderContexts([]);
    renderTrace([]);
    renderTimings({});
    renderClaims([], []);
    await refreshStatus();
    showToast(`Selected ${activeKnowledgeBase()?.name || knowledgeBaseId}.`);
  } catch (error) {
    showToast(`Knowledge-base selection failed: ${error.message}`);
  } finally {
    setLoading(false);
  }
}

async function createKnowledgeBase() {
  const name = $("new-kb-name").value.trim();
  if (!name) {
    showToast("Enter a knowledge-base name first.");
    showKnowledgeBaseStatus("Enter a name before creating a new KB.");
    return;
  }

  setLoading(true);
  showKnowledgeBaseStatus(`Creating "${name}"...`);
  try {
    const created = await requestJson(api.knowledgeBases, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    await requestJson(`${api.knowledgeBases}/${encodeURIComponent(created.knowledge_base_id)}/select`, {
      method: "POST",
    });
    $("new-kb-name").value = "";
    await loadKnowledgeBases();
    state.activeKnowledgeBaseId = created.knowledge_base_id;
    renderKnowledgeBases();
    await refreshStatus();
    showKnowledgeBaseStatus(`Created and selected "${created.name}".`);
    showToast(`Created and selected ${created.name}.`);
    hideKnowledgeBaseStatus();
  } catch (error) {
    showKnowledgeBaseStatus(`Create failed: ${error.message}`);
    showToast(`Knowledge-base creation failed: ${error.message}`);
  } finally {
    setLoading(false);
  }
}

async function renameKnowledgeBase() {
  const name = $("new-kb-name").value.trim();
  if (!name) {
    showToast("Enter the new name first.");
    showKnowledgeBaseStatus("Enter a name before saving.");
    return;
  }

  setLoading(true);
  showKnowledgeBaseStatus(`Saving name "${name}"...`);
  try {
    const renamed = await requestJson(`${api.knowledgeBases}/${encodeURIComponent(state.activeKnowledgeBaseId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    $("new-kb-name").value = "";
    await loadKnowledgeBases();
    state.activeKnowledgeBaseId = renamed.knowledge_base_id;
    renderKnowledgeBases();
    await refreshStatus();
    showKnowledgeBaseStatus(`Saved current KB as "${renamed.name}".`);
    showToast(`Saved current KB as ${renamed.name}.`);
    hideKnowledgeBaseStatus();
  } catch (error) {
    showKnowledgeBaseStatus(`Save failed: ${error.message}`);
    showToast(`Rename failed: ${error.message}`);
  } finally {
    setLoading(false);
  }
}

async function deleteKnowledgeBase() {
  const active = activeKnowledgeBase();
  const label = active?.name || state.activeKnowledgeBaseId;
  const confirmed = window.confirm(`Delete or clear "${label}"? Default is cleared, custom KBs are removed.`);
  if (!confirmed) {
    return;
  }

  setLoading(true);
  try {
    const result = await requestJson(
      `${api.knowledgeBases}/${encodeURIComponent(state.activeKnowledgeBaseId)}`,
      { method: "DELETE" },
    );
    state.knowledgeBases = result.knowledge_bases || [];
    state.activeKnowledgeBaseId = result.active_knowledge_base_id || "default";
    renderKnowledgeBases();
    renderAnswer(null);
    renderContexts([]);
    renderTrace([]);
    renderTimings({});
    renderClaims([], []);
    await refreshStatus();
    showToast("Knowledge base updated.");
  } catch (error) {
    showToast(`Delete failed: ${error.message}`);
  } finally {
    setLoading(false);
  }
}

async function refreshStatus() {
  try {
    const status = await requestJson(`${api.status}?${knowledgeBaseQuery()}`);
    state.activeKnowledgeBaseId = status.active_knowledge_base_id || state.activeKnowledgeBaseId;
    state.knowledgeBases = status.knowledge_bases || state.knowledgeBases;
    renderKnowledgeBases();
    $("status-chunks").textContent = status.total_chunks ?? 0;
    $("status-provider").textContent = status.llm_provider ?? "-";
    $("status-embedding").textContent = status.embedding_backend ?? "-";
    $("status-reranking").textContent = status.reranking_enabled ? "enabled" : "disabled";
  } catch (error) {
    showToast(`Status check failed: ${error.message}`);
  }
}

async function indexDocuments() {
  if (!state.files.length) {
    showToast("Choose at least one PDF before indexing.");
    return;
  }

  const formData = new FormData();
  state.files.forEach((file) => formData.append("files", file));

  setLoading(true);
  setOperationStatus("index", "Preparing PDF files for upload...", 12);
  try {
    const shouldReset = !$("append-index")?.checked;
    setOperationStatus("index", "Uploading PDFs to backend...", 36);
    const result = await requestJson(`${api.index}?reset=${shouldReset}&${knowledgeBaseQuery()}`, {
      method: "POST",
      body: formData,
    });
    setOperationStatus("index", "Building embeddings and FAISS index...", 78);
    showToast(
      `${shouldReset ? "Rebuilt" : "Added"} ${result.indexed_chunks} chunks from ${result.indexed_files} file(s).`,
    );
    setOperationStatus("index", "Knowledge base ready.", 100);
    await loadKnowledgeBases();
    await refreshStatus();
    hideOperationStatus("index");
  } catch (error) {
    setOperationStatus("index", "Indexing failed. Check backend logs.", 100);
    showToast(`Indexing failed: ${error.message}`);
  } finally {
    setLoading(false);
  }
}

async function resetIndex() {
  setLoading(true);
  try {
    await requestJson(`${api.reset}?${knowledgeBaseQuery()}`, { method: "POST" });
    renderAnswer(null);
    renderContexts([]);
    renderTrace([]);
    renderTimings({});
    renderClaims([], []);
    showToast("Knowledge base reset.");
    await loadKnowledgeBases();
    await refreshStatus();
  } catch (error) {
    showToast(`Reset failed: ${error.message}`);
  } finally {
    setLoading(false);
  }
}

async function askQuestion() {
  const payload = currentPayload();
  if (!payload.question) {
    showToast("Enter a question first.");
    return;
  }

  setLoading(true);
  setOperationStatus("answer", "Retrieving candidate evidence chunks...", 30);
  try {
    const result = await requestJson(api.ask, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    setOperationStatus("answer", "Reranking, selecting, generating, and verifying...", 75);
    renderAnswer(result);
    renderContexts(result.contexts || []);
    renderTrace(result.agent_trace || []);
    renderTimings(result.timings || {});
    renderClaims(result.supported_claims || [], result.unsupported_claims || []);
    setOperationStatus("answer", "Answer ready.", 100);
    hideOperationStatus("answer");
  } catch (error) {
    setOperationStatus("answer", "Answer generation failed.", 100);
    showToast(`Question failed: ${error.message}`);
  } finally {
    setLoading(false);
  }
}

async function comparePipelines() {
  const payload = currentPayload();
  if (!payload.question) {
    showToast("Enter a question first.");
    return;
  }

  setLoading(true);
  setOperationStatus("answer", "Running baseline and proposed pipelines...", 35);
  try {
    const baseline = await requestJson(api.ask, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: payload.question,
        mode: "baseline",
        top_k: payload.top_k,
        knowledge_base_id: payload.knowledge_base_id,
      }),
    });
    setOperationStatus("answer", "Baseline complete. Running proposed pipeline...", 68);
    const proposed = await requestJson(api.ask, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: payload.question,
        mode: "multi_agent",
        top_k: payload.top_k,
        knowledge_base_id: payload.knowledge_base_id,
      }),
    });
    renderComparison(baseline, proposed);
    renderAnswer(proposed);
    renderContexts(proposed.contexts || []);
    renderTrace(proposed.agent_trace || []);
    renderTimings(proposed.timings || {});
    renderClaims(proposed.supported_claims || [], proposed.unsupported_claims || []);
    setOperationStatus("answer", "Comparison ready.", 100);
    hideOperationStatus("answer");
  } catch (error) {
    setOperationStatus("answer", "Comparison failed.", 100);
    showToast(`Comparison failed: ${error.message}`);
  } finally {
    setLoading(false);
  }
}

async function previewRetrieval() {
  const payload = currentPayload();
  if (!payload.question) {
    showToast("Enter a question first.");
    return;
  }

  setLoading(true);
  setOperationStatus("answer", "Retrieving evidence preview...", 45);
  try {
    const result = await requestJson(api.retrieve, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: payload.question,
        top_k: payload.top_k,
        knowledge_base_id: payload.knowledge_base_id,
      }),
    });
    $("answer-text").textContent = "Retrieval preview generated. Generate an answer to run QA.";
    renderContexts(result.contexts || []);
    $("context-count").textContent = `${result.retrieved_context_count || 0} chunks`;
    setOperationStatus("answer", "Retrieval preview ready.", 100);
    hideOperationStatus("answer");
  } catch (error) {
    setOperationStatus("answer", "Retrieval preview failed.", 100);
    showToast(`Retrieval failed: ${error.message}`);
  } finally {
    setLoading(false);
  }
}

function renderAnswer(result) {
  if (!result) {
    $("answer-text").textContent = "No answer generated yet.";
    $("answer-provider").textContent = "-";
    $("verified-badge").textContent = "Waiting";
    $("verified-badge").className = "badge";
    $("confidence").textContent = "0.000";
    $("hero-confidence").textContent = "0.000";
    $("faithfulness").textContent = "0.000";
    $("coverage").textContent = "0.000";
    $("rerank-gain").textContent = "0.000";
    $("regenerated").textContent = "No";
    $("retrieved-count").textContent = "0";
    $("selected-count").textContent = "0";
    $("redundant-count").textContent = "0";
    $("before-rerank-count").textContent = "0";
    $("after-rerank-count").textContent = "0";
    $("query-complexity").textContent = "-";
    $("total-time").textContent = "0 ms";
    $("strip-coverage").textContent = "0.000";
    $("strip-reduction").textContent = "0.0%";
    $("strip-rerank").textContent = "0.000";
    $("strip-verified").textContent = "Waiting";
    return;
  }

  const reduction = contextReduction(
    result.retrieved_context_count || 0,
    result.selected_context_count || 0,
  );
  $("answer-text").textContent = result.answer || "No answer returned.";
  $("answer-provider").textContent = result.llm_provider || "-";
  $("verified-badge").textContent = result.verified ? "Verified" : "Needs review";
  $("verified-badge").className = result.verified ? "badge success" : "badge warning";
  $("confidence").textContent = formatNumber(result.confidence);
  $("hero-confidence").textContent = formatNumber(result.confidence);
  $("faithfulness").textContent = formatNumber(result.semantic_similarity);
  $("coverage").textContent = formatNumber(result.evidence_coverage_score);
  $("rerank-gain").textContent = formatNumber(result.reranking_gain);
  $("regenerated").textContent = result.regenerated ? "Yes" : "No";
  $("retrieved-count").textContent = result.retrieved_context_count ?? 0;
  $("selected-count").textContent = result.selected_context_count ?? 0;
  $("redundant-count").textContent = result.removed_redundant_chunks ?? 0;
  $("before-rerank-count").textContent = result.retrieved_before_reranking_count ?? 0;
  $("after-rerank-count").textContent = result.retrieved_after_reranking_count ?? 0;
  $("query-complexity").textContent = result.query_complexity || result.mode || "-";
  $("total-time").textContent = `${formatNumber(result.timings?.total_ms, 1)} ms`;
  $("strip-coverage").textContent = formatNumber(result.evidence_coverage_score);
  $("strip-reduction").textContent = `${formatNumber(reduction, 1)}%`;
  $("strip-rerank").textContent = formatNumber(result.reranking_gain);
  $("strip-verified").textContent = result.verified ? "Verified" : "Review";
}

function contextReduction(retrieved, selected) {
  if (!retrieved) {
    return 0;
  }
  return Math.max(0, ((retrieved - selected) / retrieved) * 100);
}

function renderComparison(baseline, proposed) {
  $("comparison-status").textContent = "Complete";
  $("comparison-status").className = "badge success";
  const rows = [
    ["Confidence", baseline.confidence, proposed.confidence, 3],
    ["Faithfulness", baseline.semantic_similarity, proposed.semantic_similarity, 3],
    ["Evidence Coverage", baseline.evidence_coverage_score, proposed.evidence_coverage_score, 3],
    ["Selected Chunks", baseline.selected_context_count, proposed.selected_context_count, 0],
    ["Context Reduction %", contextReduction(baseline.retrieved_context_count, baseline.selected_context_count), contextReduction(proposed.retrieved_context_count, proposed.selected_context_count), 1],
    ["Latency (ms)", baseline.timings?.total_ms || 0, proposed.timings?.total_ms || 0, 1],
  ];

  $("comparison-grid").innerHTML = rows
    .map(([label, b, p, digits]) => {
      const delta = Number(p) - Number(b);
      const formattedDelta = `${delta >= 0 ? "+" : ""}${delta.toFixed(digits)}`;
      return `
        <article class="comparison-card">
          <span class="metric-label">${escapeHtml(label)}</span>
          <div class="comparison-values">
            <div><small>Baseline</small><strong>${Number(b).toFixed(digits)}</strong></div>
            <div><small>Proposed</small><strong>${Number(p).toFixed(digits)}</strong></div>
            <div><small>Delta</small><strong>${formattedDelta}</strong></div>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderContexts(contexts) {
  const container = $("contexts");
  $("context-count").textContent = `${contexts.length} chunks`;
  if (!contexts.length) {
    container.innerHTML = '<div class="context-card">No evidence chunks to show.</div>';
    return;
  }

  container.innerHTML = contexts
    .map(
      (chunk, index) => `
        <article class="context-card">
          <div class="context-meta">
            <span>Chunk ${index + 1}: ${escapeHtml(chunk.source || "unknown")} | page ${
              chunk.page ?? "-"
            }</span>
            <span>score=${formatNumber(chunk.score)}</span>
          </div>
          <p>${escapeHtml(chunk.text || "")}</p>
          <div class="context-meta"><span>${escapeHtml(chunk.chunk_id || "")}</span></div>
        </article>
      `,
    )
    .join("");
}

function renderClaims(supported, unsupported) {
  $("supported-count").textContent = supported.length;
  $("unsupported-count").textContent = unsupported.length;
  $("supported-claims").innerHTML = supported.length
    ? supported.map((claim) => `<article class="claim-card"><p>${escapeHtml(claim)}</p></article>`).join("")
    : '<article class="claim-card"><p>No supported claims returned yet.</p></article>';
  $("unsupported-claims").innerHTML = unsupported.length
    ? unsupported.map((claim) => `<article class="claim-card"><p>${escapeHtml(claim)}</p></article>`).join("")
    : '<article class="claim-card"><p>No unsupported claims returned.</p></article>';
}

function renderTrace(trace) {
  const container = $("agent-trace");
  $("trace-count").textContent = `${trace.length} steps`;
  if (!trace.length) {
    container.innerHTML = '<div class="trace-card">No trace available yet.</div>';
    return;
  }

  container.innerHTML = trace
    .map(
      (step) => `
        <article class="trace-card">
          <div class="trace-meta">
            <strong>${escapeHtml(step.agent || "Agent")}</strong>
            <span>${escapeHtml(step.status || "-")} | ${formatNumber(step.duration_ms, 1)} ms</span>
          </div>
          <p>${escapeHtml(step.detail || "")}</p>
        </article>
      `,
    )
    .join("");
}

function renderTimings(timings) {
  const rows = Object.entries(timings).filter(([key]) => key !== "total_ms");
  const container = $("timings-list");
  if (!rows.length) {
    container.innerHTML = '<div class="timing-row"><span>No timings yet</span><span>0 ms</span></div>';
    return;
  }
  container.innerHTML = rows
    .map(
      ([key, value]) => `
        <div class="timing-row">
          <span>${escapeHtml(key.replaceAll("_", " "))}</span>
          <span>${formatNumber(value, 1)} ms</span>
        </div>
      `,
    )
    .join("");
}

async function loadEvaluation() {
  try {
    const result = await requestJson(api.evaluationSummary);
    const rows = result.rows || [];
    if (!rows.length) {
      $("evaluation-table").innerHTML = "<p>No evaluation summary found.</p>";
      return;
    }
    const headers = Object.keys(rows[0]);
    $("evaluation-table").innerHTML = `
      <table class="data-table">
        <thead><tr>${headers.map((h) => `<th>${escapeHtml(h)}</th>`).join("")}</tr></thead>
        <tbody>
          ${rows
            .map(
              (row) =>
                `<tr>${headers.map((h) => `<td>${escapeHtml(row[h] ?? "")}</td>`).join("")}</tr>`,
            )
            .join("")}
        </tbody>
      </table>
    `;
  } catch (error) {
    showToast(`Evaluation results unavailable: ${error.message}`);
  }
}

function loadExampleQuestion() {
  const current = $("question").value.trim();
  const index = sampleQuestions.indexOf(current);
  const next = sampleQuestions[(index + 1) % sampleQuestions.length];
  $("question").value = next;
}

function boot() {
  bind("pdf-files", "change", (event) => {
    state.files = Array.from(event.target.files || []);
    updateFileList();
  });
  bind("mode", "change", syncModeLabels);
  bind("knowledge-base-select", "change", (event) =>
    selectKnowledgeBase(event.target.value),
  );
  bind("index-docs", "click", indexDocuments);
  bind("ask-question", "click", askQuestion);
  bind("preview-retrieval", "click", previewRetrieval);
  bind("compare-question", "click", comparePipelines);
  bind("reset-index", "click", resetIndex);
  bind("append-index", "change", syncIndexModeLabel);
  bind("refresh-status", "click", refreshStatus);
  bind("refresh-kbs", "click", loadKnowledgeBases);
  bind("create-kb", "click", createKnowledgeBase);
  bind("rename-kb", "click", renameKnowledgeBase);
  bind("delete-kb", "click", deleteKnowledgeBase);
  bind("new-kb-name", "keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      renameKnowledgeBase();
    }
  });
  bind("load-examples", "click", loadExampleQuestion);
  bind("load-evaluation", "click", loadEvaluation);
  updateFileList();
  syncIndexModeLabel();
  syncModeLabels();
  renderAnswer(null);
  renderContexts([]);
  renderTrace([]);
  renderTimings({});
  renderClaims([], []);
  loadKnowledgeBases().then(refreshStatus);
}

try {
  boot();
} catch (error) {
  console.error("UI initialization failed", error);
  showKnowledgeBaseStatus(`UI initialization failed: ${error.message}`);
}
