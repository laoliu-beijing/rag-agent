"""
文档处理服务

协调 PDF 解析、分块、向量入库的完整流程，
实现混合处理模式（小文件同步 / 大文件异步）。
"""

import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import BackgroundTasks, UploadFile

from app.config.settings import get_settings
from app.core.retriever import ChromaRetriever
from app.models.schemas import DocumentUploadResponse
from app.parser.chunker import DocumentChunker
from app.parser.pdf_parser import PDFParseError, parse_pdf, quick_page_count
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentService:
    """
    文档处理服务

    负责：
    1. 接收上传的 PDF 文件
    2. 判断同步/异步处理
    3. 调用解析器、分块器、检索器完成入库
    4. 维护文档处理状态
    """

    def __init__(self):
        """初始化服务组件"""
        self.settings = get_settings()
        self.retriever = ChromaRetriever()
        self.chunker = DocumentChunker()

        # 确保目录存在
        self.settings.pdf_dir_path.mkdir(parents=True, exist_ok=True)
        self.settings.json_dir_path.mkdir(parents=True, exist_ok=True)

    async def upload_document(
        self,
        file: UploadFile,
        background_tasks: BackgroundTasks,
    ) -> DocumentUploadResponse:
        """
        处理文档上传

        Args:
            file: 上传的 PDF 文件
            background_tasks: FastAPI 后台任务对象

        Returns:
            DocumentUploadResponse: 上传结果
        """
        # 生成文档 ID
        doc_id = str(uuid.uuid4())
        filename = file.filename or "unknown.pdf"

        logger.info(
            "收到文档上传",
            doc_id=doc_id,
            filename=filename,
        )

        # 保存上传文件
        doc_dir = self.settings.pdf_dir_path / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        file_path = doc_dir / filename

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # 快速检测页数
        page_count = quick_page_count(file_path)

        # 判断同步或异步处理
        if page_count <= self.settings.PDF_SYNC_PAGE_THRESHOLD:
            # 小文件：同步处理
            logger.info(
                "小文件同步处理",
                doc_id=doc_id,
                page_count=page_count,
            )
            chunks_count = await self._process_document(doc_id, file_path, filename)

            return DocumentUploadResponse(
                doc_id=doc_id,
                filename=filename,
                status="completed",
                page_count=page_count,
                chunks_count=chunks_count,
            )
        else:
            # 大文件：异步处理
            logger.info(
                "大文件异步处理",
                doc_id=doc_id,
                page_count=page_count,
            )
            background_tasks.add_task(
                self._process_document_sync,
                doc_id,
                file_path,
                filename,
            )

            return DocumentUploadResponse(
                doc_id=doc_id,
                filename=filename,
                status="processing",
                page_count=page_count,
                task_id=f"task_{doc_id}",
            )

    async def _process_document(
        self,
        doc_id: str,
        file_path: Path,
        filename: str,
    ) -> int:
        """
        异步处理文档（可被 await）

        Args:
            doc_id: 文档 ID
            file_path: PDF 文件路径
            filename: 原始文件名

        Returns:
            int: 生成的 chunk 数量
        """
        return self._process_document_sync(doc_id, file_path, filename)

    def _process_document_sync(
        self,
        doc_id: str,
        file_path: Path,
        filename: str,
    ) -> int:
        """
        同步处理文档

        Args:
            doc_id: 文档 ID
            file_path: PDF 文件路径
            filename: 原始文件名

        Returns:
            int: 生成的 chunk 数量
        """
        try:
            logger.info(
                "开始处理文档",
                doc_id=doc_id,
                filename=filename,
            )

            # 1. PDF 解析
            parsed_data = parse_pdf(file_path, doc_id)

            # 2. 文档分块
            chunks = self.chunker.chunk_document(parsed_data)

            # 3. 向量入库
            if chunks:
                self.retriever.add_document(doc_id, chunks)

            logger.info(
                "文档处理完成",
                doc_id=doc_id,
                chunks_count=len(chunks),
            )

            return len(chunks)

        except PDFParseError as e:
            logger.error(
                "文档解析失败",
                doc_id=doc_id,
                error=str(e),
            )
            raise
        except Exception as e:
            logger.error(
                "文档处理异常",
                doc_id=doc_id,
                error=str(e),
            )
            raise

    def delete_document(self, doc_id: str) -> bool:
        """
        删除文档及其关联数据

        Args:
            doc_id: 文档 ID

        Returns:
            bool: 删除成功返回 True
        """
        try:
            # 删除向量数据
            self.retriever.delete_document(doc_id)

            # 删除原始 PDF
            pdf_dir = self.settings.pdf_dir_path / doc_id
            if pdf_dir.exists():
                shutil.rmtree(pdf_dir)

            # 删除解析后的 JSON
            json_path = self.settings.json_dir_path / f"{doc_id}.json"
            if json_path.exists():
                json_path.unlink()

            logger.info("文档已删除", doc_id=doc_id)
            return True

        except Exception as e:
            logger.error("删除文档失败", doc_id=doc_id, error=str(e))
            return False

    def list_documents(self) -> list[dict]:
        """
        列出所有已上传的文档

        Returns:
            list[dict]: 文档信息列表
        """
        documents = []
        pdf_dir = self.settings.pdf_dir_path

        if not pdf_dir.exists():
            return documents

        for doc_dir in pdf_dir.iterdir():
            if doc_dir.is_dir():
                doc_id = doc_dir.name
                pdf_files = list(doc_dir.glob("*.pdf"))
                filename = pdf_files[0].name if pdf_files else "unknown"

                # 检查是否有对应的 JSON 文件判断状态
                json_path = self.settings.json_dir_path / f"{doc_id}.json"
                status = "completed" if json_path.exists() else "processing"

                documents.append({
                    "doc_id": doc_id,
                    "filename": filename,
                    "status": status,
                    "created_at": datetime.fromtimestamp(doc_dir.stat().st_ctime),
                })

        return documents
