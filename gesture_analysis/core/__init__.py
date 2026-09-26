"""
核心分析引擎模块

提供手势分析的核心功能，包括特征提取和分析评估。
"""

# 2026-09-26 M3.0 Task 9:`from .feature_extraction import (...)` 那 4 个
# FeatureExtractor 已随模块一起删掉(死代码,不在 CSV 生产路径上)。
# 本包不再转发 `feature_extraction` 的任何名字;`feature_extraction.angles`
# 仍可按完整路径直接 import(见 `gesture_analysis/api/app.py`)。
from .analysis import (
    HandAnalyzer,
    ArmAnalyzer,
    ShoulderAnalyzer,
    UpperBodyAnalyzer,
    EmotionAnalyzer
)

__all__ = [
    # Analysis
    'HandAnalyzer',
    'ArmAnalyzer',
    'ShoulderAnalyzer',
    'UpperBodyAnalyzer',
    'EmotionAnalyzer'
]
