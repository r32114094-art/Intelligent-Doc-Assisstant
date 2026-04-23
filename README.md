# 智能文档助手 (Intelligent Document Assistant)

基于 RAG (Retrieval-Augmented Generation) 的文档问答系统。上传 PDF 文档后，通过语义检索 + 大语言模型生成准确的上下文答案。

## ✨ 核心特性

- **自主实现的 RAG 管道** — 9 个独立模块从零搭建，不依赖 LangChain 等框架
- **6 轮迭代优化** — 通过消融实验逐步引入高级技术，Faithfulness 评分从 4.2 提升至 5.0
- **多策略检索** — MQE (多查询扩展) + HyDE (假设文档嵌入) + Small-to-Big 分层检索
- **Cross-Encoder 重排序** — Bi-Encoder 粗召回 + Cross-Encoder 精排，兼顾效率与质量
- **流式问答** — SSE 实时推送管道每一步的处理进度
- **Docker 一键部署** — 含 Dockerfile + docker-compose，内置健康检查

## 🏗️ 技术架构

```
Frontend (Vanilla JS SPA)
    ↕ HTTP / SSE
FastAPI (backend/main.py) — REST API + 静态文件服务
    ↕
DocumentAssistant (backend/assistant.py) — 管道编排层
    ↕
backend/rag/               — 9 个可插拔模块
    ↕
外部服务: DeepSeek (LLM) · DashScope (Embeddings) · Qdrant Cloud (Vectors)
```

### RAG 管道模块

| 模块 | 说明 |
|------|------|
| `config.py` | 统一配置管理 (dataclass + .env) |
| `parser.py` | 文档解析 (MarkItDown, PDF 后处理) |
| `chunker.py` | 文本分块 (固定窗口 / 标题感知 / Small-to-Big) |
| `embedder.py` | 向量化 (DashScope API + 本地 SentenceTransformer 备选) |
| `vector_store.py` | Qdrant 向量数据库 (upsert / search / 文档管理) |
| `retriever.py` | 多策略检索 (Basic / MQE / HyDE + RRF 融合) |
| `bm25_index.py` | BM25 稀疏检索 (jieba 分词) |
| `reranker.py` | Cross-Encoder 重排序 (BAAI/bge-reranker-base) |
| `llm_client.py` | LLM 调用封装 (OpenAI 兼容 API) |

## 📊 迭代实验记录

| 版本 | 技术 | 关键结果 |
|------|------|----------|
| v1 | Baseline 纯向量检索 | 基准参考 |
| v2 | Heading-aware 标题感知分块 | 结构保持更好 |
| v3 | Hybrid Search (Dense + BM25 + RRF) | 关键词精确匹配 + 语义理解 |
| v4 | MQE + HyDE (移除 BM25) | 查询扩展 > 稀疏检索 |
| v5 | Small-to-Big (400→1200 tokens) | 检索精准 + 上下文完整 |
| v6 | Cross-Encoder Rerank | **Faithfulness 4.2→5.0** |

评估维度：Context Relevance · Faithfulness · Answer Relevance · 拒答准确率（LLM-as-Judge）

## 🚀 快速开始

### 环境要求

- Python 3.10+
- 需要的 API Key: DeepSeek (LLM)、阿里云 DashScope (Embedding)、Qdrant Cloud (向量库)

### 本地运行

```bash
# 1. 安装依赖
pip install -r backend/requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 API Key

# 3. 启动服务
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 访问 http://localhost:8000
```

### Docker 部署

```bash
# 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 API Key

# 一键部署
docker compose up -d

# 查看日志
docker compose logs -f
```

## 🔗 API 接口

| Endpoint | Method | 说明 |
|----------|--------|------|
| `/api/init` | POST | 初始化 RAG 管道 |
| `/api/upload` | POST | 上传文档 (PDF/MD/TXT/DOCX/CSV/JSON) |
| `/api/chat` | POST | 同步问答 |
| `/api/chat/stream` | POST | SSE 流式问答 |
| `/api/documents` | GET | 列出已索引文档 |
| `/api/documents/delete` | POST | 删除文档 |
| `/api/stats` | GET | 系统统计信息 |

## 📁 项目结构

```
.
├── backend/
│   ├── main.py              # FastAPI 应用入口
│   ├── assistant.py          # RAG 管道编排器
│   ├── eval_rag.py           # 消融实验评估脚本
│   ├── requirements.txt
│   └── rag/                  # RAG 管道模块
│       ├── config.py         # 配置管理
│       ├── parser.py         # 文档解析
│       ├── chunker.py        # 文本分块
│       ├── embedder.py       # 向量化
│       ├── vector_store.py   # Qdrant 向量库
│       ├── retriever.py      # 多策略检索
│       ├── bm25_index.py     # BM25 索引
│       ├── reranker.py       # 重排序
│       └── llm_client.py     # LLM 客户端
├── frontend/
│   ├── index.html            # SPA 页面
│   ├── app.js                # 前端逻辑
│   └── style.css             # 样式
├── docx/
│   └── test.md               # 评测数据集 (25 题)
├── Dockerfile
├── docker-compose.yml
└── .env.example              # 环境变量模板
```

## 📜 License

[MIT](./LICENSE)
