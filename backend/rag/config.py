"""
统一配置管理 — 从 .env 读取所有外部服务配置
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# 加载 .env：尝试多个路径
_this_dir = os.path.dirname(os.path.abspath(__file__))
for _candidate in [
    os.path.join(_this_dir, "..", "..", ".env"),    # backend/rag/ → 智能文档助手/.env
    os.path.join(_this_dir, "..", ".env"),           # backend/rag/ → backend/.env
    os.path.join(os.getcwd(), "..", ".env"),         # CWD 上一级
    os.path.join(os.getcwd(), ".env"),               # CWD 当前目录
]:
    _env_path = os.path.normpath(_candidate)
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
        break


@dataclass
class LLMConfig:
    api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL_ID", "deepseek-chat"))
    timeout: int = field(default_factory=lambda: int(os.getenv("LLM_TIMEOUT", "60")))


@dataclass
class EmbeddingConfig:
    model_type: str = field(default_factory=lambda: os.getenv("EMBED_MODEL_TYPE", "dashscope"))
    model_name: str = field(default_factory=lambda: os.getenv("EMBED_MODEL_NAME", "text-embedding-v4"))
    api_key: str = field(default_factory=lambda: os.getenv("EMBED_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("EMBED_BASE_URL", ""))


@dataclass
class QdrantConfig:
    url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    api_key: str = field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))
    collection_name: str = "rag_knowledge_base"
    distance: str = "cosine"


@dataclass
class RAGConfig:
    """RAG 管道总配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)

    # 分块参数 — V5 最佳: Small-to-Big (小检索，大上下文)
    chunk_size: int = 400                         # V5: 小 chunk 检索更精准
    chunk_overlap: int = 50                       # V5: 小 chunk 不需要大重叠
    chunk_strategy: str = "heading_aware"         # 标题感知分块
    enable_small_to_big: bool = True              # V5: 开启 Small-to-Big
    parent_chunk_size: int = 1200                 # V5: 父 chunk 保证上下文完整

    # 检索参数 — V4 最佳: Dense + MQE + HyDE
    top_k: int = 5
    enable_mqe: bool = True                       # V4: 多查询扩展
    mqe_expansions: int = 2
    enable_hyde: bool = True                      # V4: 假设文档嵌入
    enable_bm25: bool = False                     # V4: 与 MQE+HyDE 冗余，禁用

    # 重排序 — V6: 中文 Rerank 消灭幻觉 (FF: 4.2→5.0)
    enable_rerank: bool = True                    # V6: bge-reranker-base 显著提升忠实度
    rerank_model: str = "BAAI/bge-reranker-base"  # 备用: 中文友好模型
    rerank_top_k: int = 5
    rerank_candidates: int = 20
