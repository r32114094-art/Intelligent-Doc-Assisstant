"""
Qdrant 向量存储 — 直接使用 qdrant-client SDK 操作向量数据库

不再通过 hello-agents 的封装层，直接控制：
- 集合创建与管理
- 向量写入（upsert）
- 相似度搜索（带过滤）
- 文档删除（按 source_path 过滤）
- 文档列表（聚合 source_path）
"""

import os
import time
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
    PointIdsList,
)

from .config import QdrantConfig


class VectorStore:
    """Qdrant 向量存储管理"""

    def __init__(self, config: QdrantConfig = None, dimension: int = 1024):
        cfg = config or QdrantConfig()
        self.collection_name = cfg.collection_name
        self.client = QdrantClient(url=cfg.url, api_key=cfg.api_key, timeout=30)
        self._ensure_collection(dimension, cfg.distance)
        print(f"✅ Qdrant 连接成功: {cfg.url}, 集合={self.collection_name}")

    # ── 集合管理 ──────────────────────────────

    def _ensure_collection(self, dimension: int, distance: str):
        """确保集合存在，不存在则创建"""
        dist_map = {"cosine": Distance.COSINE, "euclid": Distance.EUCLID, "dot": Distance.DOT}
        dist = dist_map.get(distance.lower(), Distance.COSINE)

        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in collections:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=dimension, distance=dist),
        )
        print(f"✅ 创建 Qdrant 集合: {self.collection_name} (维度={dimension})")

    # ── 写入 ──────────────────────────────────

    def upsert(self, ids: List[str], vectors: List[List[float]], payloads: List[Dict], batch_size: int = 64):
        """
        批量写入向量

        Args:
            ids: 点 ID 列表
            vectors: 向量列表
            payloads: 元数据列表
            batch_size: 每批写入数量
        """
        total = len(ids)
        for i in range(0, total, batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_vecs = vectors[i : i + batch_size]
            batch_payloads = payloads[i : i + batch_size]

            points = [
                PointStruct(id=pid, vector=vec, payload=payload)
                for pid, vec, payload in zip(batch_ids, batch_vecs, batch_payloads)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

        print(f"✅ 写入 {total} 个向量到 Qdrant")

    # ── 搜索 ──────────────────────────────────

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        向量相似度搜索（适配 qdrant-client 1.17.x query_points API）

        Args:
            query_vector: 查询向量
            limit: 返回数量
            filters: 过滤条件 {"key": value}
            score_threshold: 最低相似度阈值

        Returns:
            搜索结果列表，每项包含 id, score, content, metadata
        """
        qdrant_filter = None
        if filters:
            must_conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=must_conditions)

        # qdrant-client >= 1.12 使用 query_points 替代 search
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            query_filter=qdrant_filter,
            score_threshold=score_threshold,
            with_payload=True,
        )

        return [
            {
                "id": str(point.id),
                "score": float(point.score),
                "content": point.payload.get("content", ""),
                "metadata": point.payload,
            }
            for point in response.points
        ]

    # ── 文档管理 ──────────────────────────────

    def list_documents(self, namespace: str = None) -> List[Dict]:
        """
        列出所有已索引文档（从向量元数据中聚合）

        Returns:
            文档列表，每项包含 source, chunks, added_at
        """
        # 构建过滤条件
        must = [FieldCondition(key="is_rag_data", match=MatchValue(value=True))]

        rag_filter = Filter(must=must)

        doc_map = {}
        offset = None
        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=rag_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in results:
                payload = point.payload or {}
                src = payload.get("source_path") or payload.get("data_source") or "unknown"
                src_name = os.path.basename(src)
                if src_name not in doc_map:
                    doc_map[src_name] = {
                        "source": src_name,
                        "full_path": src,
                        "chunks": 0,
                        "added_at": payload.get("added_at", 0),
                    }
                doc_map[src_name]["chunks"] += 1

            if next_offset is None:
                break
            offset = next_offset

        return sorted(doc_map.values(), key=lambda d: d["added_at"], reverse=True)

    def delete_by_source(self, source_name: str) -> int:
        """
        删除指定文档的所有向量分块

        Args:
            source_name: 文件名

        Returns:
            删除的向量数量
        """
        # 先 scroll 收集匹配的点 ID
        rag_filter = Filter(must=[
            FieldCondition(key="is_rag_data", match=MatchValue(value=True)),
        ])
        ids_to_delete = []
        offset = None
        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=rag_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in results:
                payload = point.payload or {}
                src = payload.get("source_path", "")
                if os.path.basename(src) == source_name or src == source_name:
                    ids_to_delete.append(point.id)
            if next_offset is None:
                break
            offset = next_offset

        if ids_to_delete:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=ids_to_delete),
                wait=True,
            )

        return len(ids_to_delete)

    def get_collection_info(self) -> Dict:
        """获取集合统计信息"""
        info = self.client.get_collection(self.collection_name)
        return {
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status.value if info.status else "unknown",
        }
