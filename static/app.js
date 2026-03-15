const state = {
  sessionId: null,
  busy: false,
};

const els = {
  sessionId: document.querySelector("#session-id"),
  docList: document.querySelector("#document-list"),
  pathInput: document.querySelector("#path-input"),
  fileInput: document.querySelector("#file-input"),
  chatLog: document.querySelector("#chat-log"),
  chatInput: document.querySelector("#chat-input"),
  chatForm: document.querySelector("#chat-form"),
  status: document.querySelector("#status"),
  loadSamples: document.querySelector("#load-samples"),
  loadPaths: document.querySelector("#load-paths"),
  uploadDocs: document.querySelector("#upload-docs"),
  resetHistory: document.querySelector("#reset-history"),
  clearSession: document.querySelector("#clear-session"),
};

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "请求失败");
  }
  return data;
}

function setStatus(message) {
  els.status.textContent = message || "";
}

function setBusy(nextBusy) {
  state.busy = nextBusy;
  const buttons = document.querySelectorAll("button");
  buttons.forEach((button) => {
    button.disabled = nextBusy;
  });
}

function renderDocuments(documents) {
  if (!documents.length) {
    els.docList.innerHTML = '<div class="doc-card"><span>当前还没有知识库文档。</span></div>';
    return;
  }

  els.docList.innerHTML = documents
    .map(
      (doc) => `
        <div class="doc-card">
          <strong>${escapeHtml(doc.source_name)}</strong>
          <span>类型: ${escapeHtml(doc.source_type)} | chunks: ${doc.chunk_count}</span>
          <span>${escapeHtml(doc.source_path)}</span>
        </div>
      `
    )
    .join("");
}

function renderMessage(role, content, extra = {}) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const citations = (extra.citations || [])
    .map(
      (item) =>
        `<span class="chip">${escapeHtml(item.source_name)} / ${escapeHtml(item.segment_label)}</span>`
    )
    .join("");

  const sourceDocuments = (extra.source_documents || [])
    .map(
      (doc) => `
        <div class="source-item">
          <strong>${escapeHtml(doc.source_name)} / ${escapeHtml(doc.segment_label)}</strong>
          <span>score: ${doc.score.toFixed(4)}</span>
          <span>${escapeHtml(doc.content)}</span>
        </div>
      `
    )
    .join("");

  const rewritten = extra.rewritten_question
    ? `<div class="meta-row"><span class="chip">检索改写: ${escapeHtml(extra.rewritten_question)}</span></div>`
    : "";

  wrapper.innerHTML = `
    <div class="role">${role === "user" ? "User" : "Assistant"}</div>
    <div class="content">${escapeHtml(content)}</div>
    ${rewritten}
    ${citations ? `<div class="citation-row">${citations}</div>` : ""}
    ${sourceDocuments ? `<div class="source-row">${sourceDocuments}</div>` : ""}
  `;
  els.chatLog.appendChild(wrapper);
  els.chatLog.scrollTop = els.chatLog.scrollHeight;
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function ensureSession() {
  const cached = window.localStorage.getItem("rag-session-id");
  if (cached) {
    state.sessionId = cached;
    els.sessionId.textContent = cached;
    await refreshDocuments();
    return;
  }

  const data = await request("/api/session", { method: "POST" });
  state.sessionId = data.session_id;
  window.localStorage.setItem("rag-session-id", data.session_id);
  els.sessionId.textContent = data.session_id;
}

async function refreshDocuments() {
  if (!state.sessionId) return;
  const data = await request(`/api/sessions/${state.sessionId}/documents`);
  if (data.session_id !== state.sessionId) {
    state.sessionId = data.session_id;
    window.localStorage.setItem("rag-session-id", data.session_id);
    els.sessionId.textContent = data.session_id;
  }
  renderDocuments(data.documents);
}

async function ingestSamples() {
  setBusy(true);
  setStatus("正在加载内置业务样例知识库...");
  try {
    await request("/api/documents/sample", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId }),
    });
    await refreshDocuments();
    setStatus("样例知识库已加载。");
  } catch (error) {
    setStatus(error.message);
  } finally {
    setBusy(false);
  }
}

async function ingestPaths() {
  const paths = els.pathInput.value
    .split("\n")
    .map((item) => item.trim())
    .filter(Boolean);
  if (!paths.length) {
    setStatus("请先输入至少一个文档路径。");
    return;
  }

  setBusy(true);
  setStatus("正在按路径建立知识库...");
  try {
    const data = await request("/api/documents/path", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId, paths }),
    });
    await refreshDocuments();
    setStatus(`已导入 ${data.documents.length} 份文档。`);
  } catch (error) {
    setStatus(error.message);
  } finally {
    setBusy(false);
  }
}

async function uploadDocuments() {
  const files = Array.from(els.fileInput.files || []);
  if (!files.length) {
    setStatus("请选择要上传的文档。");
    return;
  }

  setBusy(true);
  setStatus("正在上传并切块向量化...");
  try {
    const form = new FormData();
    form.append("session_id", state.sessionId);
    files.forEach((file) => form.append("files", file));
    const data = await request("/api/documents/upload", {
      method: "POST",
      body: form,
    });
    await refreshDocuments();
    els.fileInput.value = "";
    setStatus(`已上传 ${data.documents.length} 份文档。`);
  } catch (error) {
    setStatus(error.message);
  } finally {
    setBusy(false);
  }
}

async function sendQuestion(event) {
  event.preventDefault();
  const question = els.chatInput.value.trim();
  if (!question) {
    setStatus("请输入问题。");
    return;
  }

  renderMessage("user", question);
  els.chatInput.value = "";
  setBusy(true);
  setStatus("正在进行检索和回答...");

  try {
    const data = await request("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId, question }),
    });
    renderMessage("assistant", data.answer, data);
    setStatus("回答完成。");
  } catch (error) {
    renderMessage("assistant", `失败: ${error.message}`);
    setStatus(error.message);
  } finally {
    setBusy(false);
  }
}

async function resetSession(clearDocuments) {
  setBusy(true);
  setStatus(clearDocuments ? "正在清空会话和知识库..." : "正在清空会话历史...");
  try {
    const data = await request("/api/session/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId, clear_documents: clearDocuments }),
    });
    els.chatLog.innerHTML = "";
    renderDocuments(data.documents);
    setStatus(clearDocuments ? "会话和知识库已清空。" : "会话历史已清空。");
  } catch (error) {
    setStatus(error.message);
  } finally {
    setBusy(false);
  }
}

async function bootstrap() {
  try {
    await ensureSession();
    setStatus("会话已创建，可以先加载样例知识库再开始提问。");
  } catch (error) {
    setStatus(error.message);
  }
}

els.loadSamples.addEventListener("click", ingestSamples);
els.loadPaths.addEventListener("click", ingestPaths);
els.uploadDocs.addEventListener("click", uploadDocuments);
els.chatForm.addEventListener("submit", sendQuestion);
els.resetHistory.addEventListener("click", () => resetSession(false));
els.clearSession.addEventListener("click", () => resetSession(true));

bootstrap();
