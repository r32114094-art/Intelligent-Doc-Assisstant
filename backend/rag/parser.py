"""
文档解析器 — 将 PDF / Markdown / DOCX 等格式转为干净的纯文本

选型说明：
- 使用 MarkItDown 库统一处理各种文档格式，将其转为 Markdown
- PDF 做额外的后处理（去页码噪音、合并短行、重组段落）
- 比直接用 PyPDF2 更好，因为 MarkItDown 能保留标题层级和表格结构
"""

import os
import re
from typing import Optional


class DocumentParser:
    """文档解析器"""

    def __init__(self):
        self._md = None
        try:
            from markitdown import MarkItDown
            self._md = MarkItDown()
        except ImportError:
            print("[WARNING] MarkItDown 未安装，将使用纯文本回退方案")

    # ── 公开接口 ──────────────────────────────

    def parse(self, file_path: str) -> str:
        """
        统一解析入口：将文件转为纯文本

        Args:
            file_path: 文件路径

        Returns:
            解析后的文本内容
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return self._parse_pdf(file_path)
        elif ext in (".md", ".txt", ".csv", ".json", ".xml", ".log"):
            return self._read_text_file(file_path)
        else:
            return self._parse_with_markitdown(file_path)

    # ── PDF 解析 ──────────────────────────────

    def _parse_pdf(self, path: str) -> str:
        """PDF 专用解析：MarkItDown 提取 + 后处理清洗"""
        raw = self._parse_with_markitdown(path)
        if not raw.strip():
            return self._read_text_file(path)
        return self._post_process_pdf(raw)

    def _post_process_pdf(self, text: str) -> str:
        """
        PDF 文本后处理：
        1. 去除页码、页眉页脚噪音
        2. 合并因分页导致的短行
        3. 重组段落结构
        """
        lines = text.splitlines()
        cleaned = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 跳过纯数字行（页码）
            if re.match(r"^\d+$", line):
                continue
            # 跳过单字符行
            if len(line) <= 2 and not line.isdigit():
                continue
            cleaned.append(line)

        # 智能合并短行
        merged = []
        i = 0
        while i < len(cleaned):
            current = cleaned[i]
            if len(current) < 60 and i + 1 < len(cleaned):
                nxt = cleaned[i + 1]
                # 不合并标题和冒号结尾的行
                if (
                    not current.endswith(("：", ":"))
                    and not current.startswith("#")
                    and not nxt.startswith("#")
                    and len(nxt) < 120
                ):
                    merged.append(current + " " + nxt)
                    i += 2
                    continue
            merged.append(current)
            i += 1

        # 重组段落
        paragraphs = []
        current_para = []
        for line in merged:
            if line.startswith("#") or line.endswith(("：", ":")) or len(line) > 150:
                if current_para:
                    paragraphs.append(" ".join(current_para))
                    current_para = []
                paragraphs.append(line)
            else:
                current_para.append(line)
        if current_para:
            paragraphs.append(" ".join(current_para))

        return "\n\n".join(paragraphs)

    # ── MarkItDown 通用解析 ───────────────────

    def _parse_with_markitdown(self, path: str) -> str:
        """使用 MarkItDown 转换任意支持格式"""
        if self._md is None:
            return self._read_text_file(path)
        try:
            result = self._md.convert(path)
            text = getattr(result, "text_content", None)
            return text.strip() if isinstance(text, str) else ""
        except Exception as e:
            print(f"[WARNING] MarkItDown 处理失败: {e}")
            return self._read_text_file(path)

    # ── 纯文本回退 ────────────────────────────

    @staticmethod
    def _read_text_file(path: str) -> str:
        """纯文本读取（最终回退方案）"""
        for encoding in ("utf-8", "gbk", "latin-1"):
            try:
                with open(path, "r", encoding=encoding, errors="ignore") as f:
                    return f.read()
            except Exception:
                continue
        return ""
