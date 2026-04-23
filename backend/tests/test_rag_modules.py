"""
RAG 管道核心模块单元测试

覆盖：
- DocumentParser: 文本读取、PDF 后处理
- TextChunker: 固定分块、标题感知、Small-to-Big
- preprocess_for_embedding: Markdown 清洗
- BM25Index: 索引构建与检索
"""

import os
import sys
import tempfile

import pytest

# 确保 backend/ 在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.parser import DocumentParser
from rag.chunker import TextChunker, _approx_token_len
from rag.embedder import preprocess_for_embedding
from rag.bm25_index import BM25Index


# ─── Parser 测试 ───────────────────────────


class TestDocumentParser:
    """DocumentParser 单元测试"""

    def test_parse_txt_file(self, tmp_path):
        """解析纯文本文件"""
        txt = tmp_path / "test.txt"
        txt.write_text("Hello World\n这是一段测试文本", encoding="utf-8")

        parser = DocumentParser()
        result = parser.parse(str(txt))
        assert "Hello World" in result
        assert "测试文本" in result

    def test_parse_md_file(self, tmp_path):
        """解析 Markdown 文件"""
        md = tmp_path / "test.md"
        md.write_text("# 标题\n\n这是段落内容\n\n## 小标题\n\n更多内容", encoding="utf-8")

        parser = DocumentParser()
        result = parser.parse(str(md))
        assert "标题" in result
        assert "段落内容" in result

    def test_parse_nonexistent_file(self):
        """解析不存在的文件应抛出异常"""
        parser = DocumentParser()
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/file.txt")

    def test_post_process_pdf_removes_page_numbers(self):
        """PDF 后处理：去除纯数字页码行"""
        parser = DocumentParser()
        text = "第一行正文\n42\n第二行正文\n123\n最后一行"
        result = parser._post_process_pdf(text)
        assert "42" not in result
        assert "123" not in result
        assert "第一行正文" in result

    def test_post_process_pdf_skips_short_lines(self):
        """PDF 后处理：去除单字符噪音行"""
        parser = DocumentParser()
        text = "正文内容\n|\n另一段正文"
        result = parser._post_process_pdf(text)
        assert "|" not in result


# ─── Chunker 测试 ──────────────────────────


class TestTextChunker:
    """TextChunker 单元测试"""

    def test_approx_token_len_chinese(self):
        """中文按字计数"""
        assert _approx_token_len("你好世界") == 4

    def test_approx_token_len_english(self):
        """英文按空格分词计数"""
        assert _approx_token_len("hello world foo") == 3

    def test_approx_token_len_mixed(self):
        """中英文混合计数"""
        result = _approx_token_len("你好 world")
        assert result == 3  # 2 CJK + 1 English

    def test_chunk_empty_text(self):
        """空文本返回空列表"""
        chunker = TextChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_chunk_fixed_size(self):
        """固定大小分块：生成非空结果"""
        chunker = TextChunker()
        text = "This is a test. " * 100
        result = chunker.chunk(text, strategy="fixed", chunk_size=50)
        assert len(result) > 1
        for chunk in result:
            assert chunk["content"]
            assert chunk["id"]

    def test_chunk_heading_aware(self):
        """标题感知分块：保留标题路径元数据"""
        chunker = TextChunker()
        text = "# 第一章\n\n第一章内容段落\n\n## 1.1 节\n\n1.1 节内容\n\n# 第二章\n\n第二章内容"
        result = chunker.chunk(text, strategy="heading_aware", chunk_size=200)
        assert len(result) >= 1
        # 检查是否有标题路径元数据
        has_heading = any(c["metadata"].get("heading_path") for c in result)
        assert has_heading

    def test_chunk_small_to_big(self):
        """Small-to-Big 分块：子 chunk 携带 parent 信息"""
        chunker = TextChunker()
        text = ("这是一段很长的中文文本。" * 50 + "\n\n") * 5
        result = chunker.chunk(
            text,
            strategy="heading_aware",
            chunk_size=100,
            enable_small_to_big=True,
            parent_chunk_size=400,
        )
        assert len(result) >= 1
        # 检查 parent_id 和 parent_content
        for chunk in result:
            assert "parent_id" in chunk["metadata"]
            assert "parent_content" in chunk["metadata"]

    def test_chunk_deduplication(self):
        """分块结果去重：相同内容不重复"""
        chunker = TextChunker()
        text = "重复段落内容\n\n" * 3
        result = chunker.chunk(text, strategy="fixed", chunk_size=200)
        contents = [c["content"] for c in result]
        assert len(contents) == len(set(contents))


# ─── Embedding 预处理测试 ─────────────────


class TestPreprocessForEmbedding:
    """preprocess_for_embedding 单元测试"""

    def test_removes_heading_marks(self):
        """去除 Markdown 标题符号"""
        result = preprocess_for_embedding("## 这是标题")
        assert result == "这是标题"

    def test_removes_link_syntax(self):
        """去除链接语法，保留文字"""
        result = preprocess_for_embedding("[点击这里](https://example.com)")
        assert result == "点击这里"

    def test_removes_bold_italic(self):
        """去除粗体和斜体标记"""
        assert preprocess_for_embedding("**粗体**") == "粗体"
        assert preprocess_for_embedding("*斜体*") == "斜体"

    def test_removes_inline_code(self):
        """去除行内代码标记"""
        assert preprocess_for_embedding("`code`") == "code"

    def test_preserves_plain_text(self):
        """纯文本不受影响"""
        text = "这是普通文本，没有任何 Markdown 标记。"
        assert preprocess_for_embedding(text) == text


# ─── BM25 索引测试 ─────────────────────────


class TestBM25Index:
    """BM25Index 单元测试"""

    def test_empty_index(self):
        """空索引返回空结果"""
        idx = BM25Index()
        assert idx.size == 0
        assert idx.search("测试查询") == []

    def test_add_and_search(self):
        """添加文档后可以检索到"""
        idx = BM25Index()
        idx.add_documents(
            doc_ids=["doc1", "doc2", "doc3"],
            contents=[
                "深度学习是机器学习的一个分支",
                "自然语言处理是人工智能的重要方向",
                "计算机视觉用于图像识别",
            ],
        )
        assert idx.size == 3

        results = idx.search("深度学习", top_k=2)
        assert len(results) > 0
        assert results[0]["id"] == "doc1"

    def test_search_returns_correct_format(self):
        """检索结果格式与 VectorStore 一致"""
        idx = BM25Index()
        idx.add_documents(
            doc_ids=["d1"],
            contents=["Transformer 架构革新了自然语言处理"],
            metadata=[{"source": "test.md"}],
        )
        results = idx.search("Transformer")
        assert len(results) > 0
        result = results[0]
        assert "id" in result
        assert "score" in result
        assert "content" in result
        assert "metadata" in result

    def test_clear_index(self):
        """清空索引"""
        idx = BM25Index()
        idx.add_documents(["d1"], ["测试内容"])
        assert idx.size == 1
        idx.clear()
        assert idx.size == 0
        assert idx.search("测试") == []
