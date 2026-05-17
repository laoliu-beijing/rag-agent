"""
文档处理服务单元测试

测试上传、删除、列表功能，使用 mock 避免实际文件操作和 API 调用。
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import UploadFile

from app.core.document_service import DocumentService


class TestDocumentService:
    """DocumentService 测试"""

    @patch("app.core.document_service.parse_pdf")
    @patch("app.core.document_service.ChromaRetriever")
    @patch("app.core.document_service.quick_page_count")
    def test_sync_upload_small_file(
        self, mock_page_count, mock_retriever_cls, mock_parse
    ):
        """测试小文件同步上传"""
        mock_page_count.return_value = 5
        mock_retriever = MagicMock()
        mock_retriever_cls.return_value = mock_retriever

        mock_parse.return_value = {
            "doc_id": "test-id",
            "pages": [{"text": "内容", "page": 1}],
            "tables": [],
        }

        service = DocumentService()
        # mock chunker
        service.chunker = MagicMock()
        service.chunker.chunk_document.return_value = [
            MagicMock(text="chunk1", doc_id="test-id", page=1, chunk_type="text", metadata={})
        ]

        # 创建 mock UploadFile：file 属性需为 bytes-like 对象供 shutil.copyfileobj 使用
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "test.pdf"
        mock_file.file = io.BytesIO(b"fake pdf content")

        import asyncio
        result = asyncio.run(service.upload_document(mock_file, MagicMock()))

        assert result.status == "completed"
        assert result.page_count == 5

    @patch("app.core.document_service.quick_page_count")
    def test_async_upload_large_file(self, mock_page_count):
        """测试大文件异步上传"""
        mock_page_count.return_value = 20

        service = DocumentService()
        mock_file = MagicMock(spec=UploadFile)
        mock_file.filename = "large.pdf"
        mock_file.file = io.BytesIO(b"fake large pdf content")

        import asyncio
        bg_tasks = MagicMock()
        result = asyncio.run(service.upload_document(mock_file, bg_tasks))

        assert result.status == "processing"
        assert result.task_id is not None
        bg_tasks.add_task.assert_called_once()

    @patch("app.core.document_service.ChromaRetriever")
    def test_delete_document(self, mock_retriever_cls):
        """测试删除文档"""
        mock_retriever = MagicMock()
        mock_retriever_cls.return_value = mock_retriever

        service = DocumentService()
        # mock 目录不存在的情况
        with patch("pathlib.Path.exists", return_value=False):
            result = service.delete_document("doc-1")
            assert result is True
            mock_retriever.delete_document.assert_called_once_with("doc-1")
