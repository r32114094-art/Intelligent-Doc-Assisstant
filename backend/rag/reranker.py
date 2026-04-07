"""
Cross-Encoder 重排序器 — 对粗召回结果做精排

原理：
- Bi-Encoder（双塔模型）：query 和 doc 分别编码为向量，用余弦相似度打分
  → 速度快，适合从百万文档中粗召回 Top-20
- Cross-Encoder（交互模型）：将 query+doc 拼接后一起过 Transformer
  → 精度高但慢，适合对 Top-20 做精排取 Top-5

两者配合：Bi-Encoder 粗召回 → Cross-Encoder 精排，兼顾效率与精度
"""

import time
from typing import List, Dict, Optional


class Reranker:
    """Cross-Encoder 重排序器"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        self._load_model()

    def _load_model(self):
        """加载 Cross-Encoder 模型"""
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            print(f"✅ Reranker 模型就绪: {self.model_name}")
        except ImportError:
            print("[WARNING] sentence-transformers 未安装，重排序将跳过")
        except Exception as e:
            print(f"[WARNING] Reranker 模型加载失败: {e}")

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        对检索候选结果做精排

        Args:
            query: 用户查询
            candidates: 粗召回的候选结果列表
            top_k: 精排后保留的数量

        Returns:
            按 rerank_score 降序排列的结果列表
        """
        if not self._model or not candidates:
            return candidates[:top_k]

        start_time = time.time()

        # 构建 [query, document] 对
        pairs = [[query, c.get("content", "")] for c in candidates]

        try:
            scores = self._model.predict(pairs)

            for candidate, score in zip(candidates, scores):
                candidate["rerank_score"] = float(score)

            candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

            elapsed = time.time() - start_time
            print(f"✅ Rerank 完成: 从 {len(candidates)} 条精选出 {top_k} 条 (耗时: {elapsed:.1f}s)")

            return candidates[:top_k]

        except Exception as e:
            print(f"[WARNING] Rerank 失败: {e}，返回原始排序")
            return candidates[:top_k]
