# ============================================================================
# 智能文档问答助手 — Docker 镜像
# 
# 构建：docker build -t doc-assistant .
# 运行：docker run --env-file .env -p 8000:8000 doc-assistant
# ============================================================================

FROM python:3.11-slim

# 系统依赖（jieba 等 C 扩展编译需要 gcc）
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── 第 1 层：Python 依赖（利用 Docker 缓存，改代码不重装依赖）──
COPY backend/requirements.txt .

# 先装 PyTorch CPU 版（约 200MB vs 完整版 800MB）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 再装项目依赖
RUN pip install --no-cache-dir -r requirements.txt

# ── 第 2 层：预下载 Reranker 模型（约 1.1GB，构建时一次性下载）──
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"

# ── 第 3 层：复制项目代码 ──
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# 工作目录设为 backend（main.py 用 __file__ 定位 frontend 目录）
WORKDIR /app/backend

EXPOSE 8000

# 健康检查：每 30 秒探测一次 /docs 端点
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/docs')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
