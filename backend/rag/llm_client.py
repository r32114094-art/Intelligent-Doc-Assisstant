"""
LLM 客户端 — 封装 DeepSeek / OpenAI 兼容 API 的调用
"""

from typing import List, Dict, Optional
from openai import OpenAI
from .config import LLMConfig


class LLMClient:
    """大语言模型调用封装"""

    def __init__(self, config: LLMConfig = None):
        cfg = config or LLMConfig()
        self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.timeout)
        self.model = cfg.model

    # ── 通用调用 ────────────────────────────

    def generate(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 2048) -> str:
        """调用 LLM Chat Completion"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    # ── RAG 问答 ─────────────────────────────

    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """根据检索片段生成回答"""
        context = "\n\n---\n\n".join(context_chunks)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个专业的文档问答助手。请根据以下检索到的文档片段准确回答用户问题。\n"
                    "规则：\n"
                    "1. 只基于提供的文档内容作答，不要编造信息。\n"
                    "2. 如果文档中没有相关信息，请明确告知用户。\n"
                    "3. 回答要结构清晰，可使用列表或分段。\n"
                    "4. 引用关键信息时注明来源片段。"
                ),
            },
            {
                "role": "user",
                "content": f"参考文档片段：\n{context}\n\n用户问题：{query}",
            },
        ]
        return self.generate(messages, temperature=0.3, max_tokens=4096)

    # ── MQE 查询扩展 ─────────────────────────

    def expand_query(self, query: str, n: int = 2) -> List[str]:
        """Multi-Query Expansion：生成 N 个语义等价查询"""
        messages = [
            {
                "role": "system",
                "content": "你是检索查询扩展助手。生成语义等价或互补的多样化查询。使用中文，简短精炼，每行一个。",
            },
            {
                "role": "user",
                "content": f"原始查询：{query}\n请给出{n}个不同表述的查询，每行一个，不要编号：",
            },
        ]
        try:
            text = self.generate(messages, temperature=0.8, max_tokens=256)
            lines = [ln.strip("- \t0123456789.、） ") for ln in text.splitlines()]
            return [ln for ln in lines if ln and ln != query][:n]
        except Exception:
            return []

    # ── HyDE 假设文档生成 ─────────────────────

    def generate_hypothetical_doc(self, query: str) -> Optional[str]:
        """Hypothetical Document Embedding：生成假设性答案段落"""
        messages = [
            {
                "role": "system",
                "content": "根据用户问题，写一段可能包含答案的段落。这段文字将用于向量检索，请包含关键术语和细节。",
            },
            {
                "role": "user",
                "content": f"问题：{query}\n请直接写一段中等长度、客观、包含关键术语的段落：",
            },
        ]
        try:
            return self.generate(messages, temperature=0.5, max_tokens=512)
        except Exception:
            return None
