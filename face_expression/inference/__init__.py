
# Inference模块初始化
"""
情绪推断模块

基于AU特征进行情绪识别
"""

from .emotion_infer import infer_emotion_from_au

__all__ = ['infer_emotion_from_au']
