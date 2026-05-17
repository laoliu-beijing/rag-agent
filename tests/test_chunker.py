"""
文档分块器单元测试

验证不同内容类型的分块策略：正文切分、表格保留、重叠处理。
"""

import pytest

from app.parser.chunker import ChunkData, DocumentChunker


class TestDocumentChunker:
    """DocumentChunker 测试"""

    def test_short_text_no_split(self):
        """测试短文本不切分"""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)
        parsed = {
            "doc_id": "doc-1",
            "pages": [{"text": "短文本", "page": 1, "label": "text"}],
            "tables": [],
        }
        chunks = chunker.chunk_document(parsed)
        assert len(chunks) == 1
        assert chunks[0].text == "短文本"
        assert chunks[0].chunk_type == "text"

    def test_long_text_split_with_overlap(self):
        """测试长文本切分并保留重叠"""
        chunker = DocumentChunker(chunk_size=20, chunk_overlap=5)
        long_text = "0123456789" * 5  # 50 字符
        parsed = {
            "doc_id": "doc-1",
            "pages": [{"text": long_text, "page": 1, "label": "text"}],
            "tables": [],
        }
        chunks = chunker.chunk_document(parsed)
        assert len(chunks) > 1
        # 验证重叠：第二个块的前 5 个字符应等于第一个块的后 5 个字符
        assert chunks[1].text[:5] == chunks[0].text[-5:]

    def test_table_preserved_as_single_chunk(self):
        """测试表格作为单个块保留"""
        chunker = DocumentChunker()
        parsed = {
            "doc_id": "doc-1",
            "pages": [],
            "tables": [
                {
                    "page": 3,
                    "data": [{"A": "1", "B": "2"}, {"A": "3", "B": "4"}],
                    "caption": "测试表",
                }
            ],
        }
        chunks = chunker.chunk_document(parsed)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "table"
        assert "测试表" in chunks[0].text
        assert "1" in chunks[0].text

    def test_empty_document(self):
        """测试空文档返回空列表"""
        chunker = DocumentChunker()
        parsed = {"doc_id": "doc-1", "pages": [], "tables": []}
        chunks = chunker.chunk_document(parsed)
        assert chunks == []

    def test_multiple_pages(self):
        """测试多页文档"""
        chunker = DocumentChunker()
        parsed = {
            "doc_id": "doc-1",
            "pages": [
                {"text": "第一页内容", "page": 1, "label": "text"},
                {"text": "第二页内容", "page": 2, "label": "text"},
            ],
            "tables": [],
        }
        chunks = chunker.chunk_document(parsed)
        assert len(chunks) == 2
        assert chunks[0].page == 1
        assert chunks[1].page == 2
