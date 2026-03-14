# main/__init__.py
"""
JingXin Main Module
多模态面试评估系统 - 主模块
"""

__version__ = "1.0.0"
__author__ = "JingXin Team"

from .integrator import JingXinIntegrator
from .storage import FileLogger, SqlServerLogger

__all__ = [
    "JingXinIntegrator",
    "FileLogger",
    "SqlServerLogger"
]