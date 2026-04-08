"""
BM25 稀疏检索索引 — 关键词精确匹配补充向量语义检索

工业界 Hybrid Search 标准组件：
- Dense (向量)：擅长语义理解，但术语精确匹配弱
- Sparse (BM25)：擅长精确关键词匹配（RLHF、BPE 等专有名词）
- 两者互补，通过 RRF 融合排序

使用 jieba 中文分词 + rank_bm25 实现
"""

import jieba
from rank_bm25 import BM25Okapi
from typing import List, Dict, Optional


class BM25Index:
    """基于 BM25 的稀疏检索索引"""

    def __init__(self):
        self.corpus: List[List[str]] = []   # 分词后的文档列表
        self.doc_ids: List[str] = []        # 文档 ID
        self.doc_contents: List[str] = []   # 原始文本
        self.doc_metadata: List[Dict] = []  # 元数据
        self.bm25: Optional[BM25Okapi] = None
        self._built = False

    def add_documents(self, doc_ids: List[str], contents: List[str], metadata: List[Dict] = None):
        """
        添加文档到索引

        Args:
            doc_ids: 文档 ID 列表
            contents: 文档原始文本列表
            metadata: 文档元数据列表
        """
        if metadata is None:
            metadata = [{}] * len(doc_ids)

        for did, content, meta in zip(doc_ids, contents, metadata):
            tokens = list(jieba.cut(content))
            self.corpus.append(tokens)
            self.doc_ids.append(did)
            self.doc_contents.append(content)
            self.doc_metadata.append(meta)

        # 重新构建 BM25 索引
        self._build()

    def _build(self):
        """构建 BM25 索引"""
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
            self._built = True

    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        BM25 关键词检索

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            检索结果列表，格式与 VectorStore.search 一致
        """
        if not self._built or self.bm25 is None:
            return []

        tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokens)

        # 取 top_k 个非零分数的结果
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for i in top_indices:
            if scores[i] <= 0:
                break
            results.append({
                "id": self.doc_ids[i],
                "score": float(scores[i]),
                "content": self.doc_contents[i],
                "metadata": self.doc_metadata[i],
            })

        return results

    def clear(self):
        """清空索引"""
        self.corpus = []
        self.doc_ids = []
        self.doc_contents = []
        self.doc_metadata = []
        self.bm25 = None
        self._built = False

    @property
    def size(self) -> int:
        return len(self.doc_ids)
