
# Gesture Analysis Module
"""
手势与肢体语言分析模块

提供实时手势检测、肩部动作分析和情绪评估功能
"""

__version__ = "1.0.0"
__author__ = "JingXin Team"

from .analyzers import HandAnalyzer, ShoulderAnalyzer
from .inference import EmotionInferencer

__all__ = [
    'HandAnalyzer',
    'ShoulderAnalyzer',
    'EmotionInferencer'
]
