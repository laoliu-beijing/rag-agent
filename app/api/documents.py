"""
文档管理 API 路由

提供文档上传、列表、删除、状态查询接口。
"""

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from app.core.document_service import DocumentService
from app.models.schemas import DocumentUploadResponse

router = APIRouter(prefix="/documents", tags=["documents"])

# 服务单例
doc_service = DocumentService()


@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    上传 PDF 文档

    根据页数自动选择同步或异步处理：
    - 小文件（<=10页）：同步完成，返回 completed 状态
    - 大文件（>10页）：异步处理，返回 processing 状态和 task_id
    """
    # 校验文件类型
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")

    try:
        result = await doc_service.upload_document(file, background_tasks)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}")


@router.get("", response_model=list[dict])
async def list_documents():
    """
    列出所有已上传的文档

    返回文档 ID、文件名、处理状态和创建时间。
    """
    return doc_service.list_documents()


@router.get("/{doc_id}")
async def get_document(doc_id: str):
    """
    获取单个文档的元数据
    """
    docs = doc_service.list_documents()
    for doc in docs:
        if doc["doc_id"] == doc_id:
            return doc
    raise HTTPException(status_code=404, detail="文档不存在")


@router.get("/{doc_id}/status")
async def get_document_status(doc_id: str):
    """
    查询文档处理状态

    对于异步处理的大文件，可通过此接口轮询处理进度。
    """
    docs = doc_service.list_documents()
    for doc in docs:
        if doc["doc_id"] == doc_id:
            return {"doc_id": doc_id, "status": doc["status"]}
    raise HTTPException(status_code=404, detail="文档不存在")


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """
    删除文档及其向量数据

    同时删除原始 PDF、解析后的 JSON 和 ChromaDB 中的向量数据。
    """
    success = doc_service.delete_document(doc_id)
    if success:
        return {"message": "文档已删除", "doc_id": doc_id}
    raise HTTPException(status_code=500, detail="删除失败")
