const api = {
  health: "/health",
  status: "/rag/status",
  index: "/rag/index",
  ask: "/rag/ask",
  retrieve: "/rag/retrieve",
  reset: "/rag/reset",
};

const state = {
  files: [],
};

const $ = (id) => document.getElementById(id);

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.remove("hidden");
  window.setTimeout(() => toast.classList.add("hidden"), 3600);
}

function setLoading(isLoading) {
  ["index-docs", "ask-question", "preview-retrieval", "reset-index", "refresh-status"].forEach((id) => {
    $(id).disabled = isLoading;
  });
}

async function requestJson(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${text}`);
  }
  return response.json();
}

function formatNumber(value, digits = 3) {
  const number = Number(value || 0);
  return number.toFixed(digits);
}

function updateFileList() {
  const list = $("file-list");
  if (!state.files.length) {
    list.textContent = "No files selected.";
    return;
  }
  list.textContent = state.files.map((file) => file.name).join(" | ");
}

async function refreshStatus() {
  try {
    const status = await requestJson(api.status);
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
  try {
    const result = await requestJson(`${api.index}?reset=true`, {
      method: "POST",
      body: formData,
    });
    showToast(`Indexed ${result.indexed_chunks} chunks from ${result.indexed_files} file(s).`);
    await refreshStatus();
  } catch (error) {
    showToast(`Indexing failed: ${error.message}`);
  } finally {
    setLoading(false);
  }
}

async function resetIndex() {
  setLoading(true);
  try {
    await requestJson(api.reset, { method: "POST" });
    renderAnswer(null);
    renderContexts([]);
    renderTrace([]);
    renderTimings({});
    showToast("Knowledge base reset.");
    await refreshStatus();
  } catch (error) {
    showToast(`Reset failed: ${error.message}`);
  } finally {
    setLoading(false);
  }
}

function currentPayload() {
  return {
    question: $("question").value.trim(),
    mode: $("mode").value,
    top_k: Number($("top-k").value || 5),
  };
}

async function askQuestion() {
  const payload = currentPayload();
  if (!payload.question) {
    showToast("Enter a question first.");
    return;
  }

  setLoading(true);
  try {
    const result = await requestJson(api.ask, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderAnswer(result);
    renderContexts(result.contexts || []);
    renderTrace(result.agent_trace || []);
    renderTimings(result.timings || {});
  } catch (error) {
    showToast(`Question failed: ${error.message}`);
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
  try {
    const result = await requestJson(api.retrieve, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: payload.question, top_k: payload.top_k }),
    });
    $("answer-text").textContent = "Retrieval preview generated. Select Generate Answer for QA.";
    renderContexts(result.contexts || []);
    $("context-count").textContent = `${result.retrieved_context_count || 0} chunks`;
  } catch (error) {
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
    $("faithfulness").textContent = "0.000";
    $("coverage").textContent = "0.000";
    $("retrieved-count").textContent = "0";
    $("selected-count").textContent = "0";
    $("rerank-gain").textContent = "0.000";
    $("query-complexity").textContent = "-";
    $("total-time").textContent = "0 ms";
    return;
  }

  $("answer-text").textContent = result.answer || "No answer returned.";
  $("answer-provider").textContent = result.llm_provider || "-";
  $("verified-badge").textContent = result.verified ? "Verified" : "Needs review";
  $("verified-badge").className = result.verified ? "badge success" : "badge warning";
  $("confidence").textContent = formatNumber(result.confidence);
  $("faithfulness").textContent = formatNumber(result.semantic_similarity);
  $("coverage").textContent = formatNumber(result.evidence_coverage_score);
  $("retrieved-count").textContent = result.retrieved_context_count ?? 0;
  $("selected-count").textContent = result.selected_context_count ?? 0;
  $("rerank-gain").textContent = formatNumber(result.reranking_gain);
  $("query-complexity").textContent = result.query_complexity || result.mode || "-";
  $("total-time").textContent = `${formatNumber(result.timings?.total_ms, 1)} ms`;
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
            <span>Chunk ${index + 1}: ${escapeHtml(chunk.source || "unknown")} | page ${chunk.page ?? "-"}</span>
            <span>score=${formatNumber(chunk.score)}</span>
          </div>
          <p>${escapeHtml(chunk.text || "")}</p>
          <div class="context-meta"><span>${escapeHtml(chunk.chunk_id || "")}</span></div>
        </article>
      `,
    )
    .join("");
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

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function boot() {
  $("pdf-files").addEventListener("change", (event) => {
    state.files = Array.from(event.target.files || []);
    updateFileList();
  });
  $("index-docs").addEventListener("click", indexDocuments);
  $("ask-question").addEventListener("click", askQuestion);
  $("preview-retrieval").addEventListener("click", previewRetrieval);
  $("reset-index").addEventListener("click", resetIndex);
  $("refresh-status").addEventListener("click", refreshStatus);
  updateFileList();
  renderContexts([]);
  renderTrace([]);
  renderTimings({});
  refreshStatus();
}

boot();
