# V3 Hybrid Search — 配置快照

> 日期: 2026-04-07
> 版本: V3 (BM25 + Dense Hybrid Search + Eval Pipeline Fix)

---

## 核心变更 (相对于 V1 Baseline)

### 1. 检索架构升级
- **新增 BM25 稀疏检索**: 基于 `rank-bm25` + `jieba` 分词
- **RRF 融合排序**: Dense(向量) + Sparse(BM25) 双路检索，`k=60`
- **所有检索路径统一**: Basic / MQE / HyDE 均走混合检索

### 2. 评估管道修复 (3 个关键 Bug)
- **Bug 1 — Judge 截断**: CR/FF/AR 评分时将 chunk 截断到 300 字符 → **去掉截断，传完整内容**
- **Bug 2 — 二次检索**: eval 脚本先调 ask() 拿 answer，再独立检索拿 context → **统一从 ask() 返回值取 context**
- **Bug 3 — MQE/HyDE 双重调用**: ask() 手动调一次 MQE/HyDE，retriever.retrieve() 内部又调一次 → **删除 ask() 中的手动调用**

### 3. ask_streaming 同步修复
- 从手动 store.search → 统一走 retriever.retrieve()
- 生产前端和评估后端使用同一检索路径

---

## RAG 管道配置

| 参数 | 值 |
|---|---|
| Embedding 模型 | BAAI/bge-base-zh-v1.5 |
| Reranker 模型 | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | DeepSeek (deepseek-chat) |
| 向量数据库 | Qdrant Cloud |
| chunk_size | 800 tokens |
| chunk_overlap | 200 tokens |
| top_k | 5 |
| rerank_candidates | 20 |
| rerank_top_k | 5 |
| BM25 分词器 | jieba |
| RRF k 参数 | 60 |

## 消融实验配置 (6 组)

| 配置 | MQE | HyDE | Rerank |
|---|---|---|---|
| Baseline | ❌ | ❌ | ❌ |
| +MQE | ✅ | ❌ | ❌ |
| +HyDE | ❌ | ✅ | ❌ |
| +Rerank | ❌ | ❌ | ✅ |
| +MQE+HyDE | ✅ | ✅ | ❌ |
| Full (MQE+HyDE+RR) | ✅ | ✅ | ✅ |

## 新增依赖

```
rank-bm25>=0.2.2
jieba>=0.42.1
```

## 变更文件

| 文件 | 操作 |
|---|---|
| `rag/bm25_index.py` | 新增 — BM25 索引构建与检索 |
| `rag/retriever.py` | 重写 — 集成混合检索 + RRF |
| `assistant.py` | 修改 — BM25 初始化 + ask()/ask_streaming() 统一检索 |
| `eval_rag.py` | 修复 — 去截断 + 去二次检索 |
| `requirements.txt` | 修改 — 添加依赖 |
