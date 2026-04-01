/* ============================================================
   智能文档问答助手 — 前端逻辑
   ============================================================ */

const API_BASE = '';  // 同源，无需前缀

// ── 工具函数 ──────────────────────────────

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
    const userId = document.getElementById('user-id-input').value.trim() || 'web_user';
    showLoading('初始化助手...');
    try {
        const res = await api('/api/init', {
            method: 'POST',
            body: JSON.stringify({ user_id: userId }),
        });
        setResult('init-result', `✅ ${res.message}`, 'success');
        updateStatus(true);
        toast('助手初始化成功', 'success');
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
    if (file && file.name.toLowerCase().endsWith('.pdf')) {
        selectFile(file);
    } else {
        toast('仅支持 PDF 文件', 'error');
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
            setResult('upload-result', `✅ ${res.message}\n📄 文档: ${res.data?.document || ''}`, 'success');
            toast('文档加载成功', 'success');
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
    // 首次发消息时清除欢迎页
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

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

async function sendMessage() {
    const msg = chatInput.value.trim();
    if (!msg) return;
    if (!isInitialized) { toast('请先初始化助手', 'error'); return; }

    appendMessage('user', msg);
    chatInput.value = '';
    chatInput.style.height = 'auto';
    sendBtn.disabled = true;

    try {
        const res = await api('/api/chat', {
            method: 'POST',
            body: JSON.stringify({ message: msg }),
        });
        const label = res.data?.type === 'recall' ? '🧠 学习回顾' : '💡 回答';
        appendMessage('assistant', res.message, label);
    } catch (e) {
        appendMessage('assistant', `⚠️ 错误: ${e.message}`, '❌ 错误');
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

// 自动增高
chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
});

// 快捷问题
function sendQuickQuestion(q) {
    chatInput.value = q;
    sendMessage();
}

// ── 学习笔记 ─────────────────────────────

document.getElementById('save-note-btn').addEventListener('click', async () => {
    const content = document.getElementById('note-content').value.trim();
    const concept = document.getElementById('note-concept').value.trim();

    if (!content) { toast('笔记内容不能为空', 'error'); return; }
    if (!isInitialized) { toast('请先初始化助手', 'error'); return; }

    showLoading('保存笔记...');
    try {
        const res = await api('/api/note', {
            method: 'POST',
            body: JSON.stringify({ content, concept: concept || null }),
        });
        setResult('note-result', `✅ ${res.message}`, 'success');
        toast('笔记已保存', 'success');
        document.getElementById('note-content').value = '';
        document.getElementById('note-concept').value = '';
    } catch (e) {
        setResult('note-result', `❌ ${e.message}`, 'error');
        toast(e.message, 'error');
    } finally {
        hideLoading();
    }
});

// ── 学习回顾 ─────────────────────────────

document.getElementById('recall-btn').addEventListener('click', async () => {
    const query = document.getElementById('recall-query').value.trim();
    if (!query) { toast('请输入查询关键词', 'error'); return; }
    if (!isInitialized) { toast('请先初始化助手', 'error'); return; }

    showLoading('检索记忆...');
    try {
        const res = await api('/api/recall', {
            method: 'POST',
            body: JSON.stringify({ query }),
        });
        const el = document.getElementById('recall-result');
        el.textContent = res.message;
        el.classList.add('show');
    } catch (e) {
        toast(e.message, 'error');
    } finally {
        hideLoading();
    }
});

// ── 统计 & 报告 ──────────────────────────

document.getElementById('refresh-stats-btn').addEventListener('click', async () => {
    if (!isInitialized) { toast('请先初始化助手', 'error'); return; }

    try {
        const res = await api('/api/stats');
        const d = res.data;
        document.getElementById('stat-val-duration').textContent  = d['会话时长'] || '--';
        document.getElementById('stat-val-docs').textContent      = d['加载文档'] ?? 0;
        document.getElementById('stat-val-questions').textContent  = d['提问次数'] ?? 0;
        document.getElementById('stat-val-notes').textContent     = d['学习笔记'] ?? 0;
        toast('统计已刷新', 'info');
    } catch (e) {
        toast(e.message, 'error');
    }
});

document.getElementById('gen-report-btn').addEventListener('click', async () => {
    if (!isInitialized) { toast('请先初始化助手', 'error'); return; }

    showLoading('生成报告...');
    try {
        const res = await api('/api/report');
        const card = document.getElementById('report-card');
        card.style.display = 'block';
        document.getElementById('report-content').textContent = JSON.stringify(res.data, null, 2);
        toast('报告已生成', 'success');
    } catch (e) {
        toast(e.message, 'error');
    } finally {
        hideLoading();
    }
});
