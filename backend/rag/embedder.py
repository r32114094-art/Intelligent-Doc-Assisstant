"""
嵌入向量化 — 将文本转为高维向量用于语义检索

选型说明：
- 主方案：阿里云 DashScope text-embedding-v4（1024维，中英文，API 调用）
- 备用方案：本地 sentence-transformers（如 BAAI/bge-m3）
- 选择 DashScope 的理由：无需 GPU、延迟稳定、中文效果优秀
"""

import os
import re
from typing import List, Union, Optional

import numpy as np
import requests

from .config import EmbeddingConfig


class TextEmbedder:
    """文本嵌入器"""

    def __init__(self, config: EmbeddingConfig = None):
        cfg = config or EmbeddingConfig()
        self.model_name = cfg.model_name
        self.api_key = cfg.api_key
        self.base_url = cfg.base_url
        self.model_type = cfg.model_type
        self._dimension: Optional[int] = None
        self._local_model = None

        # 初始化并探测维度
        if self.model_type == "local":
            self._init_local()
        else:
            self._init_api()

    @property
    def dimension(self) -> int:
        return self._dimension or 1024

    # ── 初始化 ────────────────────────────────

    def _init_api(self):
        """初始化 API 模式并探测向量维度"""
        test_vec = self._call_api(["dimension_probe"])
        self._dimension = len(test_vec[0])
        print(f"✅ 嵌入模型就绪 (API): {self.model_name}, 维度={self._dimension}")

    def _init_local(self):
        """初始化本地 SentenceTransformer 模型"""
        try:
            from sentence_transformers import SentenceTransformer
            self._local_model = SentenceTransformer(self.model_name)
            self._dimension = self._local_model.get_sentence_embedding_dimension()
            print(f"✅ 嵌入模型就绪 (本地): {self.model_name}, 维度={self._dimension}")
        except ImportError:
            raise ImportError("本地模式需要安装 sentence-transformers: pip install sentence-transformers")

    # ── 编码接口 ──────────────────────────────

    def embed_texts(self, texts: List[str], batch_size: int = 8) -> List[List[float]]:
        """
        批量嵌入文本

        Args:
            texts: 文本列表
            batch_size: 每批大小（API 模式用）

        Returns:
            向量列表，每个向量是 float 列表
        """
        if self.model_type == "local":
            return self._encode_local(texts)
        return self._encode_api(texts, batch_size)

    def embed_query(self, query: str) -> List[float]:
        """嵌入单个查询文本"""
        vecs = self.embed_texts([query])
        return vecs[0]

    # ── API 模式编码 ──────────────────────────

    def _encode_api(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """通过 DashScope / OpenAI 兼容 API 批量编码"""
        all_vecs = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_num = i // batch_size + 1
            print(f"  🧮 嵌入批次 {batch_num}/{total_batches} ({len(batch)} 条)...", end=" ", flush=True)
            try:
                vecs = self._call_api(batch)
                all_vecs.extend(vecs)
                print("✅")
            except Exception as e:
                print(f"❌ {e}")
                all_vecs.extend([[0.0] * self.dimension] * len(batch))
        return all_vecs

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """调用 Embedding API（OpenAI 兼容格式）"""
        url = self.base_url.rstrip("/") + "/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model_name, "input": texts}
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code >= 400:
            raise RuntimeError(f"Embedding API 调用失败: {resp.status_code} {resp.text[:200]}")
        data = resp.json()
        items = data.get("data", [])
        return [[float(x) for x in item["embedding"]] for item in items]

    # ── 本地模型编码 ──────────────────────────

    def _encode_local(self, texts: List[str]) -> List[List[float]]:
        """使用本地 SentenceTransformer 编码"""
        vecs = self._local_model.encode(texts, show_progress_bar=False)
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]


def preprocess_for_embedding(text: str) -> str:
    """
    预处理 Markdown 文本以提升嵌入质量

    去除 Markdown 标记但保留语义内容：
    - 去掉 # 标题符号但保留标题文字
    - 去掉链接语法但保留链接文字
    - 去掉强调标记（**粗体**、*斜体*）
    - 去掉代码块标记但保留代码内容
    """
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"```[^\n]*\n([\s\S]*?)```", r"\1", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()
