"""
健康检查 API 路由

提供服务运行状态和依赖组件连通性检查。
"""

from fastapi import APIRouter

from app.core.retriever import ChromaRetriever
from app.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    健康检查接口

    检查项：
    - 服务本身运行正常
    - ChromaDB 向量数据库可连接
    """
    retriever = ChromaRetriever()
    chromadb_ok = retriever.health_check()

    return HealthResponse(
        status="ok" if chromadb_ok else "degraded",
        chromadb_connected=chromadb_ok,
    )
