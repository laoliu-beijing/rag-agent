"""
应用配置管理模块

使用 Pydantic-Settings 从环境变量和 .env 文件加载配置，
支持类型校验和默认值，确保配置项在启动时即被验证。
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用全局配置类"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    DASHSCOPE_API_KEY: str = ""

    # 服务端配置
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    APP_WORKERS: int = 1

    # 业务配置
    PDF_SYNC_PAGE_THRESHOLD: int = 10
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_MIN_SCORE: float = 0.5

    # 模型配置
    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"
    EMBEDDING_DEVICE: str = "cpu"  # cpu 或 cuda
    EMBEDDING_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL: str = "qwen-plus"
    LLM_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"

    # 日志
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "./data/logs"

    @property
    def chroma_persist_path(self) -> Path:
        """返回 ChromaDB 持久化目录的 Path 对象"""
        return Path(self.CHROMA_PERSIST_DIR)

    @property
    def log_dir_path(self) -> Path:
        """返回日志目录的 Path 对象"""
        return Path(self.LOG_DIR)

    @property
    def pdf_dir_path(self) -> Path:
        """返回 PDF 存储目录的 Path 对象"""
        return Path("./data/pdfs")

    @property
    def json_dir_path(self) -> Path:
        """返回 JSON 解析结果存储目录的 Path 对象"""
        return Path("./data/json")


@lru_cache()
def get_settings() -> Settings:
    """
    获取配置单例

    使用 lru_cache 确保配置只被加载一次，避免重复读取环境变量。
    """
    return Settings()
