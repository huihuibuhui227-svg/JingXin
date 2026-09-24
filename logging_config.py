"""
JingXin 统一日志配置

使用方式：
    from logging_config import setup_logging
    setup_logging()  # 在应用启动时调用一次

    import logging
    logger = logging.getLogger(__name__)
    logger.info("消息")
    logger.error("错误", exc_info=True)  # 自动附带堆栈
"""

import logging
import logging.config
import os
from pathlib import Path

LOG_DIR = Path(__file__).parent / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(level: str = None) -> None:
    """初始化全局日志配置。

    Args:
        level: 日志级别，优先读环境变量 LOG_LEVEL，默认 INFO。
    """
    log_level = level or os.getenv("LOG_LEVEL", "INFO").upper()

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level,
                "formatter": "standard",
                "filename": str(LOG_DIR / "jingxin.log"),
                "maxBytes": 10 * 1024 * 1024,  # 10 MB
                "backupCount": 5,
                "encoding": "utf-8",
            },
        },
        "root": {
            "level": log_level,
            "handlers": ["console", "file"],
        },
    }

    logging.config.dictConfig(config)
