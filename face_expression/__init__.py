"""
Face Expression Analyzer Module

该模块提供面部动作单元(AU)分析和情绪识别功能，支持实时视频流和静态图片分析。
"""

__version__ = "1.0.0"
__author__ = "JingXin Team"

# 导出核心分析器类
from .analyzers import FaceAUAnalyzer, StaticFaceAnalyzer
from .inference import infer_emotion_from_au

# 不要在这里导出 MediaPipe 对象！
# 让用户自己 import mediapipe as mp

__all__ = [
    'FaceAUAnalyzer',
    'StaticFaceAnalyzer',
    'infer_emotion_from_au'
]