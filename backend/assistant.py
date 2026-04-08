"""
智能文档问答助手 — 核心业务逻辑层

完全使用自建 RAG 管道，不依赖 hello-agents 库：
- 文档解析：rag/parser.py
- 文本分块：rag/chunker.py
- 嵌入向量化：rag/embedder.py（DashScope API）
- 向量存储：rag/vector_store.py（Qdrant）
- 检索策略：rag/retriever.py（Basic + MQE + HyDE）
- 重排序：rag/reranker.py（Cross-Encoder）
- LLM 问答：rag/llm_client.py（DeepSeek）
"""

import os
import time
from datetime import datetime
from typing import Dict, Any, Optional

from rag.config import RAGConfig
from rag.parser import DocumentParser
from rag.chunker import TextChunker
from rag.embedder import TextEmbedder, preprocess_for_embedding
from rag.vector_store import VectorStore
from rag.llm_client import LLMClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from rag.retriever import Retriever
from rag.reranker import Reranker
from rag.bm25_index import BM25Index


class DocumentAssistant:
    """智能文档问答助手"""

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.initialized = False

        # 各组件延迟初始化
        self.parser: Optional[DocumentParser] = None
        self.chunker: Optional[TextChunker] = None
        self.embedder: Optional[TextEmbedder] = None
        self.store: Optional[VectorStore] = None
        self.llm: Optional[LLMClient] = None
        self.retriever: Optional[Retriever] = None
        self.reranker: Optional[Reranker] = None
        self.bm25_index: Optional[BM25Index] = None

        # 会话统计
        self.stats = {
            "session_start": datetime.now(),
            "documents_loaded": 0,
            "questions_asked": 0,
        }

    def initialize(self):
        """初始化所有组件"""
        print("🚀 开始初始化 RAG 管道...")
        start = time.time()

        # 1. 文档解析器
        self.parser = DocumentParser()

        # 2. 文本分块器
        self.chunker = TextChunker()

        # 3. 嵌入模型
        self.embedder = TextEmbedder(self.config.embedding)

        # 4. 向量存储
        self.store = VectorStore(self.config.qdrant, dimension=self.embedder.dimension)

        # 5. LLM 客户端
        self.llm = LLMClient(self.config.llm)

        # 6. BM25 索引 (V3: Hybrid Search)
        self.bm25_index = BM25Index()
        self._rebuild_bm25_from_store()

        # 7. 检索器 (注入 BM25 索引)
        self.retriever = Retriever(self.store, self.embedder, self.llm, self.bm25_index)

        # 8. 重排序器
        self.reranker = Reranker(self.config.rerank_model)

        self.initialized = True
        elapsed = time.time() - start
        print(f"✅ RAG 管道初始化完成 (耗时: {elapsed:.1f}s)")
    # ── BM25 索引重建 ─────────────────────────

    def _rebuild_bm25_from_store(self):
        """从 Qdrant 现有数据重建 BM25 索引（启动时调用）"""
        try:
            rag_filter = Filter(must=[
                FieldCondition(key="is_rag_data", match=MatchValue(value=True))
            ])
            offset = None
            doc_ids = []
            contents = []
            metadata_list = []

            while True:
                results, next_offset = self.store.client.scroll(
                    collection_name=self.store.collection_name,
                    scroll_filter=rag_filter,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in results:
                    payload = point.payload or {}
                    content = payload.get("content", "")
                    if content:
                        doc_ids.append(str(point.id))
                        contents.append(content)
                        metadata_list.append(payload)

                if next_offset is None:
                    break
                offset = next_offset

            if doc_ids:
                self.bm25_index.add_documents(doc_ids, contents, metadata_list)
                print(f"✅ BM25 索引重建完成: {len(doc_ids)} 条文档")
            else:
                print("ℹ️  BM25 索引: 暂无文档数据")
        except Exception as e:
            print(f"⚠️  BM25 索引重建失败 (降级为纯向量检索): {e}")

    # ── 文档加载 ──────────────────────────────

    def load_document(self, file_path: str) -> Dict[str, Any]:
        """
        加载文档到知识库

        流程：解析 → 分块 → 预处理 → 嵌入 → 写入 Qdrant
        """
        if not os.path.exists(file_path):
            return {"success": False, "message": f"文件不存在: {file_path}"}

        start_time = time.time()
        filename = os.path.basename(file_path)

        try:
            # Step 1: 解析文档
            print(f"📄 解析文档: {filename}")
            text = self.parser.parse(file_path)
            if not text.strip():
                return {"success": False, "message": "文档内容为空"}

            # Step 2: 文本分块
            print(f"✂️ 文本分块 (策略: {self.config.chunk_strategy})")
            chunks = self.chunker.chunk(
                text,
                strategy=self.config.chunk_strategy,
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
                source_path=file_path,
            )
            print(f"   生成 {len(chunks)} 个分块")

            # Step 3: 预处理 + 嵌入
            print(f"🧮 向量化 {len(chunks)} 个分块...")
            processed_texts = [preprocess_for_embedding(c["content"]) for c in chunks]
            vectors = self.embedder.embed_texts(processed_texts)

            # Step 4: 构建元数据并写入 Qdrant
            ids = [c["id"] for c in chunks]
            payloads = []
            for c in chunks:
                payload = {
                    "content": c["content"],
                    "source_path": file_path,
                    "is_rag_data": True,
                    "data_source": "rag_pipeline",
                    "added_at": int(time.time()),
                    **c["metadata"],
                }
                payloads.append(payload)

            self.store.upsert(ids, vectors, payloads)

            # V3: 同步更新 BM25 索引
            contents = [c["content"] for c in chunks]
            self.bm25_index.add_documents(ids, contents, payloads)

            elapsed = time.time() - start_time
            self.stats["documents_loaded"] += 1

            return {
                "success": True,
                "message": f"加载成功！{len(chunks)} 个分块 (耗时: {elapsed:.1f}s)",
                "document": filename,
                "chunks": len(chunks),
            }

        except Exception as e:
            return {"success": False, "message": f"加载失败: {str(e)}"}

    # ── 智能问答 ──────────────────────────────

    def ask(self, question: str) -> Dict[str, Any]:
        """
        向知识库提问

        流程：混合检索（含 MQE + HyDE）→ 重排序 → LLM 生成回答
        返回结构化结果，包含管道各步骤信息
        """
        steps = []
        start_total = time.time()

        try:
            # Step 1: 混合检索 (Dense + BM25 + RRF)，内部自动处理 MQE/HyDE
            t0 = time.time()
            candidates = self.retriever.retrieve(
                query=question,
                top_k=self.config.rerank_candidates,
                enable_mqe=self.config.enable_mqe,
                mqe_expansions=self.config.mqe_expansions,
                enable_hyde=self.config.enable_hyde,
                enable_bm25=self.config.enable_bm25,
            )
            retrieval_time = time.time() - t0

            # 记录管道步骤信息
            step_detail = "Dense + BM25 + RRF 融合" if self.config.enable_bm25 else "纯 Dense 向量检索"
            if self.config.enable_mqe:
                step_detail += " + MQE"
            if self.config.enable_hyde:
                step_detail += " + HyDE"
            step_name = "混合检索" if self.config.enable_bm25 else "向量检索"
            steps.append({
                "name": step_name,
                "icon": "🔍",
                "detail": f"{step_detail}，召回 {len(candidates)} 条候选",
                "time": f"{retrieval_time:.1f}s",
            })

            if not candidates:
                return {
                    "answer": "❌ 知识库中未找到相关内容，请确保已上传相关文档。",
                    "steps": steps,
                }

            # Step 2: Cross-Encoder 重排序
            if self.config.enable_rerank:
                t0 = time.time()
                top_results = self.reranker.rerank(question, candidates, top_k=self.config.rerank_top_k)
                steps.append({
                    "name": "Cross-Encoder 重排序",
                    "icon": "🎯",
                    "detail": f"从 {len(candidates)} 条中精选 {len(top_results)} 条",
                    "time": f"{time.time()-t0:.1f}s",
                })
            else:
                top_results = candidates[: self.config.top_k]

            # Step 3: LLM 生成回答
            context_chunks = [r["content"] for r in top_results if r.get("content")]
            if not context_chunks:
                return {
                    "answer": "❌ 检索到的内容为空，请尝试换一种方式提问。",
                    "steps": steps,
                }

            t0 = time.time()
            answer = self.llm.generate_answer(question, context_chunks)
            steps.append({
                "name": "LLM 生成回答",
                "icon": "✨",
                "detail": f"基于 {len(context_chunks)} 个片段生成答案",
                "time": f"{time.time()-t0:.1f}s",
            })

            self.stats["questions_asked"] += 1
            total_time = time.time() - start_total

            return {
                "answer": answer,
                "steps": steps,
                "contexts": context_chunks,  # 供评估脚本使用，避免二次检索
                "total_time": f"{total_time:.1f}s",
            }

        except Exception as e:
            return {
                "answer": f"❌ 问答失败: {str(e)}",
                "steps": steps,
            }

    # ── 流式问答（SSE 实时推送管道步骤）──────────

    def ask_streaming(self, question: str):
        """
        流式问答生成器 — yield 每个管道步骤的实时事件

        事件格式:
            {"type": "step", "icon": "📝", "name": "...", "detail": "...", "time": "..."}
            {"type": "answer", "content": "..."}
            {"type": "done", "total_time": "..."}
        """
        import json
        start_total = time.time()

        try:
            # Step 1: 检索，内部自动处理 MQE/HyDE/BM25
            step_detail = "Dense + BM25 + RRF 融合" if self.config.enable_bm25 else "纯 Dense 向量检索"
            if self.config.enable_mqe:
                step_detail += " + MQE"
            if self.config.enable_hyde:
                step_detail += " + HyDE"
            step_name = "混合检索" if self.config.enable_bm25 else "向量检索"
            yield json.dumps({"type": "step", "icon": "🔍", "name": step_name, "detail": f"{step_detail}，正在搜索..."}, ensure_ascii=False)

            t0 = time.time()
            candidates = self.retriever.retrieve(
                query=question,
                top_k=self.config.rerank_candidates,
                enable_mqe=self.config.enable_mqe,
                mqe_expansions=self.config.mqe_expansions,
                enable_hyde=self.config.enable_hyde,
                enable_bm25=self.config.enable_bm25,
            )
            yield json.dumps({"type": "step", "icon": "🔍", "name": step_name, "detail": f"{step_detail}，召回 {len(candidates)} 条", "time": f"{time.time()-t0:.1f}s"}, ensure_ascii=False)

            if not candidates:
                yield json.dumps({"type": "answer", "content": "❌ 知识库中未找到相关内容，请确保已上传相关文档。"}, ensure_ascii=False)
                return

            # Step 2: Rerank
            if self.config.enable_rerank:
                yield json.dumps({"type": "step", "icon": "🎯", "name": "Cross-Encoder 重排序", "detail": "正在精排候选结果..."}, ensure_ascii=False)
                t0 = time.time()
                top_results = self.reranker.rerank(question, candidates, top_k=self.config.rerank_top_k)
                yield json.dumps({"type": "step", "icon": "🎯", "name": "Cross-Encoder 重排序", "detail": f"从 {len(candidates)} 条中精选 {len(top_results)} 条", "time": f"{time.time()-t0:.1f}s"}, ensure_ascii=False)
            else:
                top_results = candidates[: self.config.top_k]

            # Step 3: LLM 生成
            context_chunks = [r["content"] for r in top_results if r.get("content")]
            if not context_chunks:
                yield json.dumps({"type": "answer", "content": "❌ 检索到的内容为空。"}, ensure_ascii=False)
                return

            yield json.dumps({"type": "step", "icon": "✨", "name": "LLM 生成回答", "detail": "正在组织答案..."}, ensure_ascii=False)
            t0 = time.time()
            answer = self.llm.generate_answer(question, context_chunks)
            yield json.dumps({"type": "step", "icon": "✨", "name": "LLM 生成回答", "detail": f"基于 {len(context_chunks)} 个片段生成", "time": f"{time.time()-t0:.1f}s"}, ensure_ascii=False)

            self.stats["questions_asked"] += 1
            yield json.dumps({"type": "answer", "content": answer}, ensure_ascii=False)
            yield json.dumps({"type": "done", "total_time": f"{time.time()-start_total:.1f}s"}, ensure_ascii=False)

        except Exception as e:
            yield json.dumps({"type": "answer", "content": f"❌ 问答失败: {str(e)}"}, ensure_ascii=False)
    # ── 统计信息 ──────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """获取会话统计"""
        duration = (datetime.now() - self.stats["session_start"]).total_seconds()
        info = {}
        if self.store:
            try:
                info = self.store.get_collection_info()
            except Exception:
                pass

        return {
            "会话时长": f"{duration:.0f}秒",
            "加载文档": self.stats["documents_loaded"],
            "提问次数": self.stats["questions_asked"],
            "知识库向量数": info.get("vectors_count", "未知"),
        }
