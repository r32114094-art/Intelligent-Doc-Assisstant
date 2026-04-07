# V1 Baseline 配置快照

> 记录本轮实验时的 RAG 管道参数，用于跨版本对比

## 分块参数
- 策略: semantic (语义分块)
- chunk_size: 512
- chunk_overlap: 50
- 章节前缀: ❌ 无

## 嵌入模型
- 模型: text-embedding-v4 (DashScope API)
- 维度: 1024

## 向量存储
- 引擎: Qdrant Cloud
- 集合: rag_knowledge_base
- 距离度量: Cosine

## 检索参数
- top_k: 5
- rerank_candidates: 20
- rerank_top_k: 5
- mqe_expansions: 2

## 重排序
- 模型: cross-encoder/ms-marco-MiniLM-L-6-v2

## LLM
- 模型: DeepSeek Chat API
- temperature (问答): 0.3
- temperature (MQE): 0.8
- temperature (HyDE): 0.5

## 测试集
- 文件: docx/test.md
- 格式: 结构化 Markdown (### Qn + **Q/A/S**)
- 题数: 25 (5 easy + 10 medium + 5 hard + 5 irrelevant)
