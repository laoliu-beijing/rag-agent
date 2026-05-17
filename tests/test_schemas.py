"""
数据模型单元测试

验证 Pydantic 模型的序列化、反序列化和校验逻辑。
"""

import pytest
from pydantic import ValidationError

from app.models.schemas import DocumentUploadResponse, QueryRequest, Source


class TestQueryRequest:
    """QueryRequest 模型测试"""

    def test_valid_request(self):
        """测试合法请求能正确创建"""
        req = QueryRequest(question="测试问题")
        assert req.question == "测试问题"
        assert req.top_k == 5
        assert req.doc_ids is None

    def test_question_required(self):
        """测试问题字段必填"""
        with pytest.raises(ValidationError):
            QueryRequest(question="")

    def test_top_k_bounds(self):
        """测试 top_k 边界校验"""
        with pytest.raises(ValidationError):
            QueryRequest(question="test", top_k=0)
        with pytest.raises(ValidationError):
            QueryRequest(question="test", top_k=25)


class TestDocumentUploadResponse:
    """DocumentUploadResponse 模型测试"""

    def test_response_creation(self):
        """测试响应对象创建"""
        resp = DocumentUploadResponse(
            doc_id="test-id",
            filename="test.pdf",
            status="completed",
            page_count=10,
            chunks_count=20,
        )
        assert resp.doc_id == "test-id"
        assert resp.chunks_count == 20
