"""
多策略检索器 — 实现 Basic / MQE / HyDE + Hybrid Search (BM25 + Dense)

检索策略说明：
- Basic：直接用查询向量做相似度搜索（速度快，基线方案）
- MQE (Multi-Query Expansion)：用 LLM 生成多个语义等价查询，合并结果
  解决"用户表述与文档用词不一致"的问题
- HyDE (Hypothetical Document Embedding)：用 LLM 生成假设性答案文档
  原理是"答案的向量比问题的向量更接近正确文档"

V3 新增：
- Hybrid Search：Dense (向量) + Sparse (BM25) 双路检索
- RRF (Reciprocal Rank Fusion)：工业界标准融合排序算法
"""

import time
from typing import List, Dict, Optional

from .embedder import TextEmbedder, preprocess_for_embedding
from .vector_store import VectorStore
from .llm_client import LLMClient
from .bm25_index import BM25Index


class Retriever:
    """多策略检索器"""

    def __init__(self, vector_store: VectorStore, embedder: TextEmbedder, llm_client: LLMClient,
                 bm25_index: BM25Index = None):
        self.store = vector_store
        self.embedder = embedder
        self.llm = llm_client
        self.bm25 = bm25_index  # V3: BM25 稀疏检索索引

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        enable_mqe: bool = True,
        mqe_expansions: int = 2,
        enable_hyde: bool = True,
        candidate_pool_multiplier: int = 4,
        namespace: Optional[str] = None,
    ) -> List[Dict]:
        """
        统一检索入口

        Args:
            query: 用户查询
            top_k: 最终返回数量
            enable_mqe: 是否启用 Multi-Query Expansion
            mqe_expansions: MQE 扩展查询数量
            enable_hyde: 是否启用 Hypothetical Document Embedding
            candidate_pool_multiplier: 候选池大小倍数
            namespace: 命名空间过滤

        Returns:
            检索结果列表（按相似度降序）
        """
        start_time = time.time()

        # 构建过滤条件
        filters = {"is_rag_data": True}

        # 如果既不启用 MQE 也不启用 HyDE，走混合检索
        if not enable_mqe and not enable_hyde:
            results = self._hybrid_search(query, top_k, filters)
            elapsed = time.time() - start_time
            print(f"🔍 混合检索完成: {len(results)} 条结果 (耗时: {elapsed:.1f}s)")
            return results

        # 高级检索：多查询合并
        all_queries = [query]

        # MQE: 生成扩展查询
        if enable_mqe:
            expanded = self.llm.expand_query(query, n=mqe_expansions)
            all_queries.extend(expanded)
            if expanded:
                print(f"📝 MQE 扩展查询: {expanded}")

        # HyDE: 生成假设文档
        if enable_hyde:
            hyde_doc = self.llm.generate_hypothetical_doc(query)
            if hyde_doc:
                all_queries.append(hyde_doc)
                print(f"📄 HyDE 假设文档: {hyde_doc[:80]}...")

        # 去重
        unique_queries = list(dict.fromkeys(all_queries))

        # 计算每个查询的候选池大小
        pool_size = max(top_k * candidate_pool_multiplier, 20)
        per_query = max(1, pool_size // len(unique_queries))

        # 多查询并行检索 → 合并去重（每个子查询都用混合检索）
        aggregated: Dict[str, Dict] = {}
        for q in unique_queries:
            hits = self._hybrid_search(q, per_query, filters)
            for hit in hits:
                hit_id = hit["id"]
                if hit_id not in aggregated or hit["score"] > aggregated[hit_id]["score"]:
                    aggregated[hit_id] = hit

        # 按分数降序排列
        results = sorted(aggregated.values(), key=lambda x: x["score"], reverse=True)[:top_k * candidate_pool_multiplier]

        elapsed = time.time() - start_time
        print(f"🔍 高级混合检索完成: {len(unique_queries)} 个查询, 召回 {len(results)} 条 (耗时: {elapsed:.1f}s)")

        return results

    # ── V3: 混合检索 (Dense + BM25 + RRF) ────────

    def _hybrid_search(self, query: str, limit: int, filters: Dict) -> List[Dict]:
        """
        混合检索：Dense 向量 + BM25 关键词 → RRF 融合排序

        如果 BM25 索引不可用，降级为纯向量检索
        """
        # Dense 向量检索
        dense_results = self._dense_search(query, limit * 4, filters)

        # BM25 稀疏检索（如果可用）
        if self.bm25 and self.bm25.size > 0:
            sparse_results = self.bm25.search(query, top_k=limit * 4)
            # RRF 融合
            fused = self._rrf_fusion(dense_results, sparse_results, k=60)
            return fused[:limit]
        else:
            # 降级为纯向量检索
            return dense_results[:limit]

    def _dense_search(self, query: str, limit: int, filters: Dict) -> List[Dict]:
        """基础向量检索"""
        qv = self.embedder.embed_query(query)
        return self.store.search(query_vector=qv, limit=limit, filters=filters)

    def _rrf_fusion(self, *result_lists, k: int = 60) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF) — 工业界标准融合排序

        公式: RRF_score(doc) = Σ 1/(k + rank_i)
        k=60 是论文推荐的标准常数

        优势：不需要归一化分数，天然适配不同评分体系 (cosine vs BM25)
        """
        # 收集所有文档的 RRF 分数
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict] = {}  # id -> 完整文档数据

        for results in result_lists:
            for rank, result in enumerate(results):
                doc_id = result["id"]
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
                # 保留最完整的文档数据
                if doc_id not in doc_map:
                    doc_map[doc_id] = result

        # 按 RRF 分数降序排列
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

        results = []
        for doc_id in sorted_ids:
            doc = doc_map[doc_id].copy()
            doc["score"] = rrf_scores[doc_id]  # 用 RRF 分数替代原始分数
            results.append(doc)

        return results

    # 兼容旧接口
    def _basic_search(self, query: str, limit: int, filters: Dict) -> List[Dict]:
        """兼容旧接口，内部调用混合检索"""
        return self._hybrid_search(query, limit, filters)
