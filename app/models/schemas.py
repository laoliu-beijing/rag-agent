"""
API 请求/响应数据模型

使用 Pydantic 定义所有 API 的输入输出结构，
自动生成 OpenAPI 文档并提供运行时数据校验。
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """文档上传响应模型"""

    doc_id: str = Field(..., description="文档唯一标识 UUID")
    filename: str = Field(..., description="原始文件名")
    status: Literal["completed", "processing", "failed"] = Field(
        ..., description="处理状态"
    )
    page_count: int = Field(..., description="PDF 总页数")
    chunks_count: int = Field(0, description="切分后的 chunk 数量（同步完成时有效）")
    task_id: str | None = Field(None, description="异步任务 ID（异步处理时返回）")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")


class DocumentMetadata(BaseModel):
    """文档元数据模型"""

    doc_id: str
    filename: str
    status: Literal["completed", "processing", "failed"]
    page_count: int
    chunks_count: int = 0
    created_at: datetime
    updated_at: datetime | None = None
    error_message: str | None = None


class Source(BaseModel):
    """答案来源信息模型"""

    doc_id: str = Field(..., description="来源文档 ID")
    doc_name: str = Field(..., description="来源文档名称")
    page: int = Field(..., description="来源页码")
    chunk_text: str = Field(..., description="原始文本片段")
    score: float = Field(..., description="相似度分数 0-1")


class QueryRequest(BaseModel):
    """问答请求模型"""

    question: str = Field(..., min_length=1, description="用户问题")
    doc_ids: list[str] | None = Field(
        None, description="限定检索的文档 ID 列表，None 表示全部"
    )
    top_k: int = Field(5, ge=1, le=20, description="检索返回的最大片段数")


class QueryResponse(BaseModel):
    """问答响应模型"""

    answer: str = Field(..., description="生成的答案或拒答文案")
    sources: list[Source] = Field(default_factory=list, description="答案来源列表")
    confidence: Literal["high", "medium", "low", "none"] = Field(
        ..., description="答案可信度"
    )
    has_evidence: bool = Field(..., description="是否有检索证据支持")


class HealthResponse(BaseModel):
    """健康检查响应模型"""

    status: str = Field("ok", description="服务状态")
    chromadb_connected: bool = Field(..., description="ChromaDB 连接状态")
    version: str = Field("0.1.0", description="服务版本")
