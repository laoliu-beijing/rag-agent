"""
PDF 解析器单元测试

测试解析器的页数检测和结构提取功能。
使用 mock 避免实际调用 docling（依赖重）。
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.parser.pdf_parser import PDFParser, PDFParseError, quick_page_count


class TestQuickPageCount:
    """快速页数检测测试"""

    @patch("PyPDF2.PdfReader")
    def test_valid_pdf(self, mock_reader):
        """测试正常 PDF 页数读取"""
        mock_reader.return_value.pages = [1, 2, 3, 4, 5]
        count = quick_page_count("/tmp/test.pdf")
        assert count == 5

    def test_nonexistent_file(self):
        """测试文件不存在时返回默认值 1"""
        count = quick_page_count("/nonexistent/file.pdf")
        assert count == 1


class TestPDFParser:
    """PDFParser 测试"""

    @patch("app.parser.pdf_parser.DocumentConverter")
    def test_parse_success(self, mock_converter_cls):
        """测试正常解析流程"""
        # 构造 mock 转换结果
        mock_result = MagicMock()
        mock_page = MagicMock()
        mock_page.export_to_image.return_value = MagicMock()
        mock_result.pages = [mock_page]

        mock_doc = MagicMock()
        mock_doc.name = "test.pdf"
        # mock 文本项：避免 pages 为空触发 PyMuPDF fallback
        mock_text_item = MagicMock()
        mock_text_item.text = "这是一段正常的中文测试文本内容"
        mock_text_item.label = "text"
        mock_prov = MagicMock()
        mock_prov.page_no = 1
        mock_text_item.prov = [mock_prov]
        mock_doc.texts = [mock_text_item]
        mock_doc.tables = []
        mock_doc.export_to_markdown.return_value = "# Test"
        mock_result.document = mock_doc

        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result
        mock_converter_cls.return_value = mock_converter

        parser = PDFParser()

        # mock 保存 JSON 的方法，避免写文件
        with patch.object(parser, "_save_json"):
            result = parser.parse("/tmp/test.pdf", "doc-123")

        assert result["doc_id"] == "doc-123"
        assert result["metadata"]["title"] == "test.pdf"
        assert "pages" in result
        assert "tables" in result

    @patch("app.parser.pdf_parser.DocumentConverter")
    def test_parse_failure(self, mock_converter_cls):
        """测试解析失败时抛出异常"""
        mock_converter = MagicMock()
        mock_converter.convert.side_effect = Exception("OCR failed")
        mock_converter_cls.return_value = mock_converter

        parser = PDFParser()

        with pytest.raises(PDFParseError):
            parser.parse("/tmp/test.pdf", "doc-123")
