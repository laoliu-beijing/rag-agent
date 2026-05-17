"""
API 端到端测试

使用 TestClient 测试所有 HTTP 接口，使用 mock 避免外部依赖。
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.models.schemas import DocumentUploadResponse


class TestDocumentsAPI:
    """文档接口测试"""

    @patch("app.api.documents.doc_service")
    def test_upload_pdf(self, mock_service):
        """测试 PDF 上传"""
        from app.main import app

        mock_service.upload_document = AsyncMock(
            return_value=DocumentUploadResponse(
                doc_id="test-id",
                filename="test.pdf",
                status="completed",
                page_count=5,
                chunks_count=10,
                task_id=None,
            )
        )

        client = TestClient(app)
        response = client.post(
            "/documents/upload",
            files={"file": ("test.pdf", b"fake pdf content", "application/pdf")},
        )

        assert response.status_code == 201
        assert response.json()["status"] == "completed"

    @patch("app.api.documents.doc_service")
    def test_upload_invalid_file(self, mock_service):
        """测试非 PDF 文件上传被拒"""
        from app.main import app

        client = TestClient(app)
        response = client.post(
            "/documents/upload",
            files={"file": ("test.txt", b"text content", "text/plain")},
        )

        assert response.status_code == 400

    @patch("app.api.documents.doc_service")
    def test_list_documents(self, mock_service):
        """测试文档列表"""
        from app.main import app

        mock_service.list_documents.return_value = [
            {"doc_id": "doc-1", "filename": "test.pdf", "status": "completed"}
        ]

        client = TestClient(app)
        response = client.get("/documents")

        assert response.status_code == 200
        assert len(response.json()) == 1

    @patch("app.api.documents.doc_service")
    def test_delete_document(self, mock_service):
        """测试删除文档"""
        from app.main import app

        mock_service.delete_document.return_value = True

        client = TestClient(app)
        response = client.delete("/documents/doc-1")

        assert response.status_code == 200


class TestQueryAPI:
    """问答接口测试"""

    @patch("app.api.query.agent")
    def test_query_success(self, mock_agent):
        """测试问答成功"""
        from app.main import app

        mock_agent.run.return_value = MagicMock(
            answer="答案是...",
            sources=[],
            confidence="high",
            has_evidence=True,
        )

        client = TestClient(app)
        response = client.post("/query", json={"question": "测试问题"})

        assert response.status_code == 200
        assert "答案是" in response.json()["answer"]

    @patch("app.api.query.agent")
    def test_query_rejection(self, mock_agent):
        """测试拒答场景"""
        from app.main import app

        mock_agent.run.return_value = MagicMock(
            answer="此问题已经超出我的范围...",
            sources=[],
            confidence="none",
            has_evidence=False,
        )

        client = TestClient(app)
        response = client.post("/query", json={"question": "无关问题"})

        assert response.status_code == 200
        assert "超出我的范围" in response.json()["answer"]


class TestHealthAPI:
    """健康检查接口测试"""

    @patch("app.api.health.ChromaRetriever")
    def test_health_ok(self, mock_retriever_cls):
        """测试健康检查通过"""
        from app.main import app

        mock_retriever = MagicMock()
        mock_retriever.health_check.return_value = True
        mock_retriever_cls.return_value = mock_retriever

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["chromadb_connected"] is True
