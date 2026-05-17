"""
结构化日志模块

使用 structlog 输出 JSON 格式的结构化日志，便于后续日志收集和分析。
每个请求分配唯一的 request_id，贯穿所有日志记录。
"""

import logging
import logging.handlers
import sys
from pathlib import Path

import structlog

from app.config.settings import get_settings


def setup_logging() -> None:
    """
    初始化日志系统

    配置 structlog 处理器，同时输出到控制台和文件。
    日志按天轮转，保留最近 7 天的日志文件。
    """
    settings = get_settings()
    log_dir = Path(settings.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 配置标准库 logging 的基础处理器
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    )

    # 配置 structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # 创建文件处理器
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_dir / "app.log",
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    # 将文件处理器添加到 structlog 的根 logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)


def get_logger(name: str):
    """
    获取一个命名 logger 实例

    Args:
        name: logger 名称，通常使用 __name__

    Returns:
        structlog 包装后的 logger 实例
    """
    return structlog.get_logger(name)
