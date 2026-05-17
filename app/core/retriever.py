"""
ChromaDB 向量检索器

封装 ChromaDB 的集合管理和向量检索功能，
提供按文档隔离、元数据过滤、相似度检索等能力。
"""

import os
from pathlib import Path

import chromadb
from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config.settings import get_settings
from app.models.state import Chunk
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_model_path(model_name: str) -> str:
    """
    解析 embedding 模型路径。

    优先级：
    1. 本地绝对/相对路径
    2. HuggingFace Hub 模型 ID（含 '/'）
    3. ModelScope 模型 ID（通过 snapshot_download 下载到本地缓存）
    """
    if Path(model_name).exists():
        return model_name

    # HuggingFace Hub 模型 ID 通常含 '/'
    if "/" in model_name:
        return model_name

    # 尝试从 ModelScope 下载
    try:
        from modelscope import snapshot_download

        logger.info("从 ModelScope 下载模型", model_name=model_name)
        local_path = snapshot_download(model_name)
        logger.info("ModelScope 模型下载完成", local_path=local_path)
        return local_path
    except ImportError:
        logger.error(
            "未安装 modelscope，无法下载 ModelScope 模型。"
            "请运行: pip install modelscope"
        )
        raise RuntimeError(
            f"模型 {model_name} 未找到，且 modelscope 未安装。"
            "请先安装: pip install modelscope"
        ) from None
    except Exception as e:
        logger.error("ModelScope 模型下载失败", model_name=model_name, error=str(e))
        raise RuntimeError(
            f"无法从 ModelScope 下载模型 {model_name}，错误: {e}"
        ) from e


class ChromaRetriever:
    """
    ChromaDB 向量检索封装

    每个文档对应一个独立的 ChromaDB collection，
    支持动态创建、删除和跨文档检索。
    """

    def __init__(self):
        """初始化 ChromaDB 客户端和 Embedding 模型"""
        self.settings = get_settings()
        self.client = chromadb.PersistentClient(path=str(self.settings.chroma_persist_path))
        self.embedding_model = self._create_embedding_model()

    def _create_embedding_model(self) -> Embeddings:
        """
        创建本地 Embedding 模型

        支持 HuggingFace Hub 和 ModelScope 模型，无需调用外部 API。
        """
        model_path = _resolve_model_path(self.settings.EMBEDDING_MODEL)
        return HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": self.settings.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )

    def _get_collection_name(self, doc_id: str) -> str:
        """生成 collection 名称"""
        return f"doc_{doc_id}"

    def add_document(self, doc_id: str, chunks: list) -> int:
        """
        将文档 chunk 添加到向量库

        Args:
            doc_id: 文档唯一标识
            chunks: ChunkData 列表

        Returns:
            int: 添加的 chunk 数量
        """
        collection_name = self._get_collection_name(doc_id)

        # 获取或创建 collection
        try:
            collection = self.client.get_collection(name=collection_name)
            logger.info(f"使用已存在的 collection: {collection_name}")
        except Exception:
            collection = self.client.create_collection(name=collection_name)
            logger.info(f"创建新 collection: {collection_name}")

        # 准备数据
        texts = [chunk.text for chunk in chunks]
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "doc_id": chunk.doc_id,
                "page": chunk.page,
                "chunk_type": chunk.chunk_type,
                **(chunk.metadata or {}),
            }
            for chunk in chunks
        ]

        # 计算 embeddings 并添加
        embeddings = self.embedding_model.embed_documents(texts)
        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )

        logger.info(
            "文档向量已入库",
            doc_id=doc_id,
            chunks_added=len(chunks),
            collection=collection_name,
        )

        return len(chunks)

    def delete_document(self, doc_id: str) -> None:
        """
        删除文档及其向量数据

        Args:
            doc_id: 文档唯一标识
        """
        collection_name = self._get_collection_name(doc_id)
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"已删除 collection: {collection_name}")
        except Exception as e:
            logger.warning(
                "删除 collection 失败（可能不存在）",
                collection=collection_name,
                error=str(e),
            )

    def search(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[Chunk]:
        """
        向量检索

        Args:
            query: 用户查询文本
            doc_ids: 限定检索的文档 ID 列表，None 表示全部
            top_k: 返回的最大结果数

        Returns:
            list[Chunk]: 检索到的 chunk 列表，按相似度降序排列
        """
        query_embedding = self.embedding_model.embed_query(query)

        # 确定检索范围
        if doc_ids:
            collections = []
            for doc_id in doc_ids:
                try:
                    collection = self.client.get_collection(
                        name=self._get_collection_name(doc_id)
                    )
                    collections.append(collection)
                except Exception:
                    logger.warning(f"文档 collection 不存在: {doc_id}")
        else:
            # 获取所有 doc_ 前缀的 collection
            collections = []
            for coll in self.client.list_collections():
                coll_name = coll.name if hasattr(coll, "name") else coll
                if coll_name.startswith("doc_"):
                    collections.append(self.client.get_collection(coll_name))

        # 在所有目标 collection 中检索并合并结果
        all_results = []
        for collection in collections:
            try:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )

                for i in range(len(results["documents"][0])):
                    doc_text = results["documents"][0][i]
                    metadata = results["metadatas"][0][i]
                    distance = results["distances"][0][i]

                    # ChromaDB 使用 L2 距离，转换为相似度分数
                    # 使用简单的反向映射：score = 1 / (1 + distance)
                    similarity = 1.0 / (1.0 + distance)

                    all_results.append(
                        Chunk(
                            text=doc_text,
                            doc_id=metadata.get("doc_id", ""),
                            page=metadata.get("page", 1),
                            chunk_type=metadata.get("chunk_type", "text"),
                            score=similarity,
                        )
                    )
            except Exception as e:
                logger.error(
                    "检索失败",
                    collection=collection.name if hasattr(collection, "name") else "unknown",
                    error=str(e),
                )

        # 按相似度降序排列，取 top_k
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    def health_check(self) -> bool:
        """
        检查 ChromaDB 连接状态

        Returns:
            bool: 连接正常返回 True
        """
        try:
            self.client.heartbeat()
            return True
        except Exception as e:
            logger.error("ChromaDB 连接异常", error=str(e))
            return False
