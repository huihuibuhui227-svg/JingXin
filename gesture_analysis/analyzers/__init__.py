"""
分析器模块

提供手部和肩部分析功能
"""

from .hand_analyzer import HandAnalyzer
from .shoulder_analyzer import ShoulderAnalyzer

# 明确导出公共接口
__all__ = [
    'HandAnalyzer',
    'ShoulderAnalyzer'
]

# 可选：为静态类型检查提供支持（如 mypy）
# 便于在其他模块中进行类型注解
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .hand_analyzer import HandAnalysisResult
    from .shoulder_analyzer import ShoulderAnalysisResult