# API模块初始化
"""
集成API模块

提供统一的RESTful API接口，协调三个子模块
"""

from .app import app

__all__ = ['app']
