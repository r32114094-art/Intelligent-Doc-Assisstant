"""
文本分块器 — 将长文本切分为适合向量检索的片段

实现了两种分块策略：
1. 固定大小滑动窗口（基线方案）
2. Markdown 标题感知分块（推荐，保留文档结构）

选型理由：
- 固定大小简单但会切断语义完整的段落
- 标题感知能保留章节结构，检索时 heading_path 提供额外上下文
"""

import hashlib
import re
from typing import List, Dict, Optional


def _is_cjk(ch: str) -> bool:
    """判断字符是否是中日韩字符"""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0xF900 <= code <= 0xFAFF
    )


_CJK_RANGES = re.compile(
    r'[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\uf900-\ufaff]'
)


def _approx_token_len(text: str) -> int:
    """近似 token 计数（CJK 按 1 token/字，其他按空白分词）"""
    cjk = len(_CJK_RANGES.findall(text))
    non_cjk_text = _CJK_RANGES.sub('', text)
    non_cjk = len(non_cjk_text.split())
    return cjk + non_cjk


class TextChunker:
    """文本分块器"""

    def chunk(
        self,
        text: str,
        strategy: str = "heading_aware",
        chunk_size: int = 800,
        overlap: int = 100,
        source_path: str = "",
        enable_small_to_big: bool = False,
        parent_chunk_size: int = 1200,
    ) -> List[Dict]:
        """
        统一分块入口

        Args:
            text: 待分块的文本
            strategy: 分块策略 ("fixed" | "heading_aware")
            chunk_size: 每个块的目标 token 数
            overlap: 重叠 token 数
            source_path: 源文件路径（写入元数据）
            enable_small_to_big: V5 — 是否启用 Small-to-Big 分块
            parent_chunk_size: V5 — 父 chunk 大小 (仅 small_to_big 生效)

        Returns:
            分块列表，每个块包含 id, content, metadata
        """
        if not text.strip():
            return []

        if enable_small_to_big:
            print(f"   [S2B] Small-to-Big mode: child={chunk_size}, parent={parent_chunk_size}")
            raw_chunks = self._chunk_small_to_big(text, child_size=chunk_size, parent_size=parent_chunk_size, strategy=strategy)
        elif strategy == "fixed":
            raw_chunks = self._chunk_fixed_size(text, chunk_size, overlap)
        else:
            print(f"   文本长度: {len(text)} 字符", flush=True)
            raw_chunks = self._chunk_heading_aware(text, chunk_size, overlap)
            print(f"   标题感知分块完成: {len(raw_chunks)} 个原始块", flush=True)

        # 为每个块生成唯一 ID 和元数据
        doc_id = hashlib.md5(f"{source_path}|{len(text)}".encode()).hexdigest()
        results = []
        seen_hashes = set()

        for ch in raw_chunks:
            content = ch["content"].strip()
            if not content:
                continue

            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            chunk_id = hashlib.md5(
                f"{doc_id}|{ch['start']}|{ch['end']}|{content_hash}".encode()
            ).hexdigest()

            metadata = {
                "source_path": source_path,
                "doc_id": doc_id,
                "content_hash": content_hash,
                "start": ch["start"],
                "end": ch["end"],
                "heading_path": ch.get("heading_path"),
                "chunk_strategy": strategy if not enable_small_to_big else "small_to_big",
            }

            # V5: Small-to-Big 元数据
            if ch.get("parent_id"):
                metadata["parent_id"] = ch["parent_id"]
                metadata["parent_content"] = ch["parent_content"]

            results.append({
                "id": chunk_id,
                "content": content,
                "metadata": metadata,
            })

        return results

    # ── V5: Small-to-Big 分块 ─────────────────

    def _chunk_small_to_big(self, text: str, child_size: int = 400,
                            parent_size: int = 1200, strategy: str = "heading_aware") -> List[Dict]:
        """
        Small-to-Big 分块策略：

        1. 先按 parent_size 做大分块（父 chunk）
        2. 对每个父 chunk 做 child_size 的小分块（子 chunk）
        3. 子 chunk 用于检索（入向量库），父 chunk 用于生成（存 payload）

        子 chunk 的 metadata 中携带 parent_id 和 parent_content。
        """
        import time

        # Step 1: 做大分块（parent）
        t0 = time.time()
        if strategy == "heading_aware":
            paragraphs = self._split_by_headings(text)
            parent_chunks = self._merge_paragraphs(paragraphs, parent_size, overlap=0)
        else:
            parent_chunks = self._chunk_fixed_size(text, parent_size, overlap=0)
        print(f"   |-- Parent chunks: {len(parent_chunks)} (parent_size={parent_size}) ({time.time()-t0:.1f}s)", flush=True)

        # Step 2: 每个 parent 分成小 child
        t1 = time.time()
        all_children = []
        for pi, parent in enumerate(parent_chunks):
            parent_id = f"parent_{pi}"
            parent_content = parent["content"]

            # 对父 chunk 内容做小分块
            child_paragraphs = self._split_into_segments(parent_content, child_size)

            for ci, child_text in enumerate(child_paragraphs):
                if not child_text.strip():
                    continue
                all_children.append({
                    "content": child_text,
                    "start": parent["start"],
                    "end": parent["end"],
                    "heading_path": parent.get("heading_path"),
                    "parent_id": parent_id,
                    "parent_content": parent_content,
                })

        print(f"   +-- Child chunks: {len(all_children)} (child_size={child_size}) ({time.time()-t1:.1f}s)", flush=True)
        return all_children

    def _split_into_segments(self, text: str, target_size: int) -> List[str]:
        """
        将文本按 target_size tokens 切分成多个片段（句子边界友好）。
        """
        sentences = re.split(r'(?<=[。！？\n])', text)
        segments = []
        current = []
        current_tokens = 0

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            sent_tokens = _approx_token_len(sent)
            if current_tokens + sent_tokens > target_size and current:
                segments.append("".join(current))
                current = [sent]
                current_tokens = sent_tokens
            else:
                current.append(sent)
                current_tokens += sent_tokens

        if current:
            segments.append("".join(current))

        # 如果整段文本不够一个 segment，直接返回
        if not segments:
            segments = [text]

        return segments

    # ── 策略 1：固定大小滑动窗口 ─────────────────

    def _chunk_fixed_size(self, text: str, size: int, overlap: int) -> List[Dict]:
        """
        固定大小分块：按 token 数切分，相邻块有重叠

        优点：实现简单，块大小均匀
        缺点：可能切断句子或段落
        """
        words = text.split()
        chunks = []
        i = 0
        pos = 0

        while i < len(words):
            # 取一个窗口的 words
            window = []
            token_count = 0
            j = i
            while j < len(words) and token_count < size:
                w = words[j]
                token_count += _approx_token_len(w)
                window.append(w)
                j += 1

            content = " ".join(window)
            start = pos
            end = pos + len(content)

            chunks.append({"content": content, "start": start, "end": end})

            # 计算滑动步长
            step_tokens = max(1, size - overlap)
            step_words = 0
            count = 0
            for w in window:
                count += _approx_token_len(w)
                step_words += 1
                if count >= step_tokens:
                    break

            pos += len(" ".join(window[:step_words])) + 1
            i += step_words

        return chunks

    # ── 策略 2：Markdown 标题感知分块 ─────────────

    def _chunk_heading_aware(self, text: str, size: int, overlap: int) -> List[Dict]:
        """
        标题感知分块：按 Markdown 标题拆分段落 → 按 token 合并

        优点：保留文档的章节结构，heading_path 提供检索上下文
        缺点：依赖文档有清晰的标题标记
        """
        import time

        # Step 1: 按标题拆分为段落
        t0 = time.time()
        paragraphs = self._split_by_headings(text)
        print(f"   ├ 标题拆分: {len(paragraphs)} 个段落 ({time.time()-t0:.1f}s)", flush=True)

        # Step 2: 按 token 限制合并段落
        t1 = time.time()
        result = self._merge_paragraphs(paragraphs, size, overlap)
        print(f"   └ 段落合并: {len(result)} 个块 ({time.time()-t1:.1f}s)", flush=True)

        return result

    def _split_by_headings(self, text: str) -> List[Dict]:
        """按 Markdown 标题 (#) 拆分段落，记录标题路径"""
        lines = text.splitlines()
        heading_stack: List[str] = []
        paragraphs: List[Dict] = []
        buf: List[str] = []
        char_pos = 0

        def flush():
            nonlocal buf
            if not buf:
                return
            content = "\n".join(buf).strip()
            if content:
                paragraphs.append({
                    "content": content,
                    "heading_path": " > ".join(heading_stack) if heading_stack else None,
                    "start": max(0, char_pos - len(content)),
                    "end": char_pos,
                })
            buf = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                flush()
                level = len(stripped) - len(stripped.lstrip("#"))
                title = stripped.lstrip("#").strip()
                if level <= len(heading_stack):
                    heading_stack = heading_stack[: level - 1]
                heading_stack.append(title)
                char_pos += len(line) + 1
                continue

            if stripped == "":
                flush()
            else:
                buf.append(line)
            char_pos += len(line) + 1

        flush()

        if not paragraphs:
            paragraphs = [{"content": text, "heading_path": None, "start": 0, "end": len(text)}]
        return paragraphs

    def _merge_paragraphs(self, paragraphs: List[Dict], size: int, overlap: int) -> List[Dict]:
        """将小段落合并到不超过 size tokens 的块中"""
        chunks = []
        cur: List[Dict] = []
        cur_tokens = 0

        i = 0
        while i < len(paragraphs):
            p = paragraphs[i]
            p_tokens = _approx_token_len(p["content"]) or 1

            if cur_tokens + p_tokens <= size or not cur:
                cur.append(p)
                cur_tokens += p_tokens
                i += 1
            else:
                # 输出当前块
                content = "\n\n".join(x["content"] for x in cur)
                heading = next((x["heading_path"] for x in reversed(cur) if x.get("heading_path")), None)
                chunks.append({
                    "content": content,
                    "start": cur[0]["start"],
                    "end": cur[-1]["end"],
                    "heading_path": heading,
                })

                # 保留尾部作为重叠
                if overlap > 0:
                    kept = []
                    kept_tokens = 0
                    for x in reversed(cur):
                        t = _approx_token_len(x["content"])
                        if kept_tokens + t > overlap:
                            break
                        kept.append(x)
                        kept_tokens += t
                    cur = list(reversed(kept))
                    cur_tokens = kept_tokens

                    # 防止死循环：如果 overlap 后仍放不下，清空 overlap 强制推进
                    if cur_tokens + p_tokens > size and cur:
                        cur = []
                        cur_tokens = 0
                else:
                    cur = []
                    cur_tokens = 0

        # 输出最后一个块
        if cur:
            content = "\n\n".join(x["content"] for x in cur)
            heading = next((x["heading_path"] for x in reversed(cur) if x.get("heading_path")), None)
            chunks.append({
                "content": content,
                "start": cur[0]["start"],
                "end": cur[-1]["end"],
                "heading_path": heading,
            })

        return chunks
