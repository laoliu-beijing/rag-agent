"""
FastAPI 应用主入口

注册所有路由、配置中间件、初始化日志系统。
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.query import router as query_router
from app.config.settings import get_settings
from app.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理

    启动时：初始化日志系统
    关闭时：清理资源
    """
    # 启动
    setup_logging()
    logger.info("应用启动完成")
    yield
    # 关闭
    logger.info("应用关闭")


def create_app() -> FastAPI:
    """
    创建 FastAPI 应用实例

    Returns:
        FastAPI: 配置好的应用实例
    """
    settings = get_settings()

    app = FastAPI(
        title="智能文档问答 Agent",
        description="基于 LangGraph + RAG 的文档问答系统",
        version="0.1.0",
        lifespan=lifespan,
    )

    # 注册 CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    app.include_router(documents_router)
    app.include_router(query_router)
    app.include_router(health_router)

    return app


# 创建应用实例（供 uvicorn 使用）
app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        workers=settings.APP_WORKERS,
        reload=True,
    )
