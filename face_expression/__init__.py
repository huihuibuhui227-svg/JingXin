"""
Face Expression Analyzer Module

提供面部动作单元(AU)分析和情绪识别功能
"""

__version__ = "1.0.0"

from .pipeline import VideoPipeline, ImagePipeline
from .models import AUFeatures, EmotionResult, AnalysisFrameResult

__all__ = [
    'VideoPipeline',
    'ImagePipeline',
    'AUFeatures',
    'EmotionResult',
    'AnalysisFrameResult'
]