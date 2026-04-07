/* ============================================================
   智能文档问答助手 — 前端逻辑 v2.0
   完全自主实现 RAG 管道版
   ============================================================ */

const API_BASE = '';  // 同源，无需前缀

// ── 工具函数 ──────────────────────────────

/** HTML 转义 */
function escapeHtml(str) {
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
}

/** 发起 API 请求 */
async function api(path, options = {}) {
    const resp = await fetch(`${API_BASE}${path}`, {
        headers: { 'Content-Type': 'application/json', ...options.headers },
        ...options,
    });
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || '请求失败');
    }
    return resp.json();
}

/** 上传文件 */
async function apiUpload(path, file) {
    const fd = new FormData();
    fd.append('file', file);
    const resp = await fetch(`${API_BASE}${path}`, { method: 'POST', body: fd });
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || '上传失败');
    }
    return resp.json();
}

/** 显示 loading */
function showLoading(text = '处理中...') {
    document.getElementById('loading-text').textContent = text;
    document.getElementById('loading-overlay').classList.add('show');
}

/** 隐藏 loading */
function hideLoading() {
    document.getElementById('loading-overlay').classList.remove('show');
}

/** 弹出 toast */
function toast(msg, type = 'info') {
    const container = document.getElementById('toast-container');
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = msg;
    container.appendChild(el);
    setTimeout(() => el.remove(), 3200);
}

/** 设置结果框 */
function setResult(id, text, type = 'success') {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = text;
    el.className = `result-box show ${type}`;
}

// ── 页面切换 ──────────────────────────────

const navItems = document.querySelectorAll('.nav-item');
const pages    = document.querySelectorAll('.page');

navItems.forEach(item => {
    item.addEventListener('click', () => {
        const target = item.dataset.page;
        navItems.forEach(n => n.classList.remove('active'));
        pages.forEach(p => p.classList.remove('active'));
        item.classList.add('active');
        document.getElementById(`page-${target}`).classList.add('active');
    });
});

// ── 状态管理 ──────────────────────────────

let isInitialized = false;

function updateStatus(online) {
    isInitialized = online;
    const badge = document.getElementById('status-badge');
    const dot   = badge.querySelector('.status-dot');
    const text  = badge.querySelector('.status-text');
    dot.className = `status-dot ${online ? 'online' : 'offline'}`;
    text.textContent = online ? '已就绪' : '未初始化';
}

// ── 初始化助手 ────────────────────────────

document.getElementById('init-btn').addEventListener('click', async () => {
    showLoading('初始化 RAG 管道...');
    try {
        const res = await api('/api/init', { method: 'POST', body: '{}' });
        setResult('init-result', `✅ ${res.message}`, 'success');
        updateStatus(true);
        toast('RAG 管道初始化成功', 'success');
        loadDocuments();
    } catch (e) {
        setResult('init-result', `❌ ${e.message}`, 'error');
        toast(e.message, 'error');
    } finally {
        hideLoading();
    }
});

// ── 文件上传 ──────────────────────────────

const uploadZone    = document.getElementById('upload-zone');
const pdfInput      = document.getElementById('pdf-input');
const uploadBtn     = document.getElementById('upload-btn');
const placeholder   = document.getElementById('upload-placeholder');
const fileInfoEl    = document.getElementById('upload-file-info');
const fileNameEl    = document.getElementById('file-name');
const removeFileBtn = document.getElementById('remove-file-btn');

let selectedFile = null;

uploadZone.addEventListener('click', (e) => {
    if (e.target === removeFileBtn || removeFileBtn.contains(e.target)) return;
    pdfInput.click();
});

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) {
        selectFile(file);
    }
});

pdfInput.addEventListener('change', () => {
    if (pdfInput.files[0]) selectFile(pdfInput.files[0]);
});

removeFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFile();
});

function selectFile(file) {
    selectedFile = file;
    fileNameEl.textContent = file.name;
    placeholder.style.display = 'none';
    fileInfoEl.style.display = 'flex';
    uploadBtn.disabled = false;
}

function clearFile() {
    selectedFile = null;
    pdfInput.value = '';
    placeholder.style.display = 'flex';
    fileInfoEl.style.display = 'none';
    uploadBtn.disabled = true;
}

uploadBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    if (!isInitialized) { toast('请先初始化助手', 'error'); return; }

    showLoading('加载文档中，请稍候...');
    try {
        const res = await apiUpload('/api/upload', selectedFile);
        if (res.success) {
            setResult('upload-result', `✅ ${res.message}`, 'success');
            toast('文档加载成功', 'success');
            clearFile();
            loadDocuments();
        } else {
            setResult('upload-result', `❌ ${res.message}`, 'error');
            toast(res.message, 'error');
        }
    } catch (e) {
        setResult('upload-result', `❌ ${e.message}`, 'error');
        toast(e.message, 'error');
    } finally {
        hideLoading();
    }
});

// ── 聊天 ──────────────────────────────────

const chatMessages = document.getElementById('chat-messages');
const chatInput    = document.getElementById('chat-input');
const sendBtn      = document.getElementById('send-btn');

/** 添加消息到聊天窗 */
function appendMessage(role, text, label) {
    const welcome = chatMessages.querySelector('.chat-welcome');
    if (welcome) welcome.remove();

    const div = document.createElement('div');
    div.className = `message ${role}`;
    if (role === 'assistant' && label) {
        div.innerHTML = `<span class="msg-label">${label}</span>${escapeHtml(text)}`;
    } else {
        div.textContent = text;
    }
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendMessage() {
    const msg = chatInput.value.trim();
    if (!msg) return;
    if (!isInitialized) { toast('请先初始化助手', 'error'); return; }

    appendMessage('user', msg);
    chatInput.value = '';
    chatInput.style.height = 'auto';
    sendBtn.disabled = true;

    // 创建思考气泡
    const welcome = chatMessages.querySelector('.chat-welcome');
    if (welcome) welcome.remove();

    const thinkingDiv = document.createElement('div');
    thinkingDiv.className = 'message assistant';
    const stepsContainer = document.createElement('div');
    stepsContainer.className = 'pipeline-steps';
    stepsContainer.innerHTML = '<div class="step-item active"><span class="step-dot"></span>正在分析问题...</div>';
    thinkingDiv.innerHTML = '<span class="msg-label">🧠 思考中</span>';
    thinkingDiv.appendChild(stepsContainer);
    chatMessages.appendChild(thinkingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // 记录已完成的步骤名 → 用于更新而非追加
    const stepElements = {};
    let answerText = '';
    let totalTime = '';

    try {
        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg }),
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // 保留不完整的行

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const event = JSON.parse(line.slice(6));

                    if (event.type === 'step') {
                        const key = event.name;
                        if (event.time) {
                            // 步骤完成 → 更新为完成态
                            if (stepElements[key]) {
                                stepElements[key].className = 'step-item done';
                                stepElements[key].innerHTML = `<span class="step-icon">${event.icon}</span><span class="step-name">${escapeHtml(event.name)}</span><span class="step-detail">${escapeHtml(event.detail)}</span><span class="step-time">${event.time}</span>`;
                            }
                        } else {
                            // 步骤开始 → 新增进行中行
                            const el = document.createElement('div');
                            el.className = 'step-item active';
                            el.innerHTML = `<span class="step-icon"><span class="step-dot"></span></span><span class="step-name">${escapeHtml(event.name)}</span><span class="step-detail">${escapeHtml(event.detail)}</span>`;
                            stepsContainer.appendChild(el);
                            stepElements[key] = el;
                        }
                    } else if (event.type === 'answer') {
                        answerText = event.content;
                    } else if (event.type === 'done') {
                        totalTime = event.total_time;
                    }

                    chatMessages.scrollTop = chatMessages.scrollHeight;
                } catch (_) { /* skip malformed */ }
            }
        }

        // 最终渲染：步骤 + 总耗时 + 回答
        if (totalTime) {
            const totalEl = document.createElement('div');
            totalEl.className = 'step-total';
            totalEl.textContent = `⏱️ 总耗时 ${totalTime}`;
            stepsContainer.appendChild(totalEl);
        }

        // 移除初始的「正在分析问题」
        const initStep = stepsContainer.querySelector('.step-item.active');
        if (initStep && !stepElements[initStep.querySelector('.step-name')?.textContent]) {
            initStep.remove();
        }

        thinkingDiv.querySelector('.msg-label').textContent = '💡 回答';
        const divider = document.createElement('div');
        divider.className = 'step-divider';
        thinkingDiv.appendChild(divider);
        const answerDiv = document.createElement('div');
        answerDiv.className = 'answer-content';
        answerDiv.textContent = answerText;
        thinkingDiv.appendChild(answerDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

    } catch (e) {
        thinkingDiv.innerHTML = `<span class="msg-label">❌ 错误</span><div class="answer-content">⚠️ ${escapeHtml(e.message)}</div>`;
    } finally {
        sendBtn.disabled = false;
        chatInput.focus();
    }
}

sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
});

function sendQuickQuestion(q) {
    chatInput.value = q;
    sendMessage();
}

// ── 统计 ──────────────────────────────────

document.getElementById('refresh-stats-btn').addEventListener('click', async () => {
    if (!isInitialized) { toast('请先初始化助手', 'error'); return; }

    try {
        const res = await api('/api/stats');
        const d = res.data;
        document.getElementById('stat-val-duration').textContent  = d['会话时长'] || '--';
        document.getElementById('stat-val-docs').textContent      = d['加载文档'] ?? 0;
        document.getElementById('stat-val-questions').textContent  = d['提问次数'] ?? 0;
        document.getElementById('stat-val-vectors').textContent   = d['知识库向量数'] ?? 0;
        toast('统计已刷新', 'info');
    } catch (e) {
        toast(e.message, 'error');
    }
});

// ── 文档管理 ─────────────────────────────

const docListEl = document.getElementById('doc-list');

async function loadDocuments() {
    if (!isInitialized) return;
    try {
        const res = await api('/api/documents');
        const docs = res.data?.documents || [];

        if (docs.length === 0) {
            docListEl.innerHTML = '<div class="doc-list-empty">📭 知识库中暂无文档，请上传文件</div>';
            return;
        }

        docListEl.innerHTML = docs.map((doc, idx) => {
            const addedDate = doc.added_at ? new Date(doc.added_at * 1000).toLocaleDateString('zh-CN') : '未知';
            return `
                <div class="doc-item" data-idx="${idx}">
                    <span class="doc-item-icon">📄</span>
                    <div class="doc-item-info">
                        <div class="doc-item-name">${escapeHtml(doc.source)}</div>
                        <div class="doc-item-meta">${doc.chunks} 个分块 · 添加于 ${addedDate}</div>
                    </div>
                    <button class="doc-item-delete" data-idx="${idx}">删除</button>
                </div>
            `;
        }).join('');

        docListEl._docs = docs;
    } catch(e) {
        docListEl.innerHTML = `<div class="doc-list-empty">⚠️ 加载失败: ${e.message}</div>`;
    }
}

async function deleteDocument(source) {
    showLoading('删除文档...');
    try {
        const res = await api('/api/documents/delete', {
            method: 'POST',
            body: JSON.stringify({ source }),
        });
        if (res.success) {
            toast(res.message, 'success');
            loadDocuments();
        } else {
            toast(res.message, 'error');
        }
    } catch(e) {
        toast(e.message, 'error');
    } finally {
        hideLoading();
    }
}

// 事件委托：点击删除按钮
docListEl.addEventListener('click', (e) => {
    const btn = e.target.closest('.doc-item-delete');
    if (!btn) return;
    const idx = parseInt(btn.dataset.idx, 10);
    const docs = docListEl._docs;
    if (!docs || idx < 0 || idx >= docs.length) return;
    deleteDocument(docs[idx].source);
});

document.getElementById('refresh-docs-btn').addEventListener('click', () => {
    if (!isInitialized) { toast('请先初始化助手', 'error'); return; }
    loadDocuments();
});
