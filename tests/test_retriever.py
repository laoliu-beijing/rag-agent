"""
ChromaDB 检索器单元测试

使用 mock 避免实际调用 Embedding API 和 ChromaDB。
"""

from unittest.mock import MagicMock, patch

import pytest

from app.core.retriever import ChromaRetriever


class TestChromaRetriever:
    """ChromaRetriever 测试"""

    @patch("app.core.retriever.chromadb.PersistentClient")
    @patch("app.core.retriever.HuggingFaceEmbeddings")
    @patch("app.core.retriever._resolve_model_path", return_value="dummy-model-path")
    def test_add_document(self, mock_resolve, mock_embeddings_cls, mock_client_cls):
        """测试添加文档到向量库"""
        # mock Embedding 模型
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        mock_embeddings_cls.return_value = mock_embeddings

        # mock ChromaDB collection
        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_client.create_collection.return_value = mock_collection
        mock_client_cls.return_value = mock_client

        retriever = ChromaRetriever()

        # 创建 mock chunk
        mock_chunk = MagicMock()
        mock_chunk.text = "测试文本"
        mock_chunk.doc_id = "doc-1"
        mock_chunk.page = 1
        mock_chunk.chunk_type = "text"
        mock_chunk.metadata = {}

        count = retriever.add_document("doc-1", [mock_chunk])
        assert count == 1
        mock_collection.add.assert_called_once()

    @patch("app.core.retriever.chromadb.PersistentClient")
    @patch("app.core.retriever.HuggingFaceEmbeddings")
    @patch("app.core.retriever._resolve_model_path", return_value="dummy-model-path")
    def test_delete_document(self, mock_resolve, mock_embeddings_cls, mock_client_cls):
        """测试删除文档"""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        retriever = ChromaRetriever()
        retriever.delete_document("doc-1")
        mock_client.delete_collection.assert_called_once_with(name="doc_doc-1")

    @patch("app.core.retriever.chromadb.PersistentClient")
    @patch("app.core.retriever.HuggingFaceEmbeddings")
    @patch("app.core.retriever._resolve_model_path", return_value="dummy-model-path")
    def test_health_check_success(self, mock_resolve, mock_embeddings_cls, mock_client_cls):
        """测试健康检查通过"""
        mock_client = MagicMock()
        mock_client.heartbeat.return_value = True
        mock_client_cls.return_value = mock_client

        retriever = ChromaRetriever()
        assert retriever.health_check() is True

    @patch("app.core.retriever.chromadb.PersistentClient")
    @patch("app.core.retriever.HuggingFaceEmbeddings")
    @patch("app.core.retriever._resolve_model_path", return_value="dummy-model-path")
    def test_health_check_failure(self, mock_resolve, mock_embeddings_cls, mock_client_cls):
        """测试健康检查失败"""
        mock_client = MagicMock()
        mock_client.heartbeat.side_effect = Exception("connection refused")
        mock_client_cls.return_value = mock_client

        retriever = ChromaRetriever()
        assert retriever.health_check() is False
