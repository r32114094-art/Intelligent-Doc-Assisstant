# V4 BM25 Ablation — 配置快照

> 日期: 2026-04-08
> 版本: V4 (BM25 增量贡献隔离实验)
> 基线: V3 (BM25 + Dense Hybrid Search + 评估管道修复)

---

## 实验目的

**隔离 BM25 混合检索的增量贡献**。

V3 同时做了两件事：修复评估 Bug + 引入 BM25。V3 vs V1 的提升无法区分两者的贡献。
本实验通过新增 `enable_bm25` 开关，在**同一修复后的评估管道**下对比纯 Dense 与 Hybrid 检索，精确量化 BM25 的增量。

## 核心改动 (相对于 V3)

### 1. 新增 BM25 开关
- `RAGConfig` 新增 `enable_bm25: bool = True`
- `Retriever.retrieve()` / `_hybrid_search()` 支持 `enable_bm25` 参数
- `assistant.ask()` / `ask_streaming()` 传递开关
- `eval_rag.py` 的 `PipelineConfig` 新增 `enable_bm25` 字段

### 2. 禁用 Rerank
- V3 已证实 Rerank（英文 Cross-Encoder）在中文场景下有害
- V4 全部配置 `enable_rerank=False`

### 3. 代码清理
- 删除死代码 `_basic_search` 方法
- 清理未使用的 `preprocess_for_embedding` 导入
- 修复 `print_report` 中硬编码 `"Full"` 配置名的潜在 Bug

---

## 消融实验配置 (4 组)

| 配置 | MQE | HyDE | BM25 | Rerank |
|---|---|---|---|---|
| Dense Only | ❌ | ❌ | ❌ | ❌ |
| Dense+MQE+HyDE | ✅ | ✅ | ❌ | ❌ |
| Hybrid (Dense+BM25) | ❌ | ❌ | ✅ | ❌ |
| Hybrid+MQE+HyDE | ✅ | ✅ | ✅ | ❌ |

### 实验设计逻辑

```
BM25 增量  = Hybrid Baseline - Dense Only
MQE+HyDE 增量 = Dense+MQE+HyDE - Dense Only
叠加效果  = Hybrid+MQE+HyDE - Dense Only
是否超线性 = 叠加效果 > BM25增量 + MQE+HyDE增量 ?
```

## RAG 管道配置 (与 V3 相同)

| 参数 | 值 |
|---|---|
| Embedding 模型 | BAAI/bge-base-zh-v1.5 |
| LLM | DeepSeek (deepseek-chat) |
| 向量数据库 | Qdrant Cloud |
| chunk_size | 800 tokens |
| chunk_overlap | 200 tokens |
| top_k | 5 |
| BM25 分词器 | jieba |
| RRF k 参数 | 60 |

## 变更文件

| 文件 | 操作 |
|---|---|
| `rag/config.py` | 新增 `enable_bm25` 字段 |
| `rag/retriever.py` | `retrieve()` + `_hybrid_search()` 加 BM25 开关；删除死代码 |
| `assistant.py` | `ask()` / `ask_streaming()` 传递 `enable_bm25` |
| `eval_rag.py` | V4 消融配置 + BM25 保存/恢复 + 修复报告打印 |
