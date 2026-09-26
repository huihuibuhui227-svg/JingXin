
# Gesture Analysis Module
"""
手势与肢体语言分析模块

提供实时手势检测、肩部动作分析和情绪评估功能
"""

__version__ = "1.0.0"
__author__ = "JingXin Team"

# 导出分析器
from gesture_analysis.core.analysis.hand_analyzer import HandAnalyzer
from gesture_analysis.core.analysis.shoulder_analyzer import ShoulderAnalyzer
from gesture_analysis.core.analysis.arm_analyzer import ArmAnalyzer
from gesture_analysis.core.analysis.upper_body_analyzer import UpperBodyAnalyzer

# 导出推理模块
from gesture_analysis.core.analysis.emotion_inferencer import EmotionInferencer

__all__ = [
    # 新接口
    # ⚠️ 2026-09-26 M3.0 Task 9:这里原来还有 4 个 FeatureExtractor 名字
    # (HandFeatureExtractor / ArmFeatureExtractor / ShoulderFeatureExtractor /
    #  UpperBodyFeatureExtractor)。它们**从来没有在本文件 import 过**(纯悬空名字,
    # `from gesture_analysis import *` 会因此 AttributeError),对应的模块也已删。
    # 一并清掉。
    # ⚠️ 本文件下面那些**没有对应 import** 的名字(HandEmotionAnalyzer / HandFeatures /
    # GestureEmotionPipeline …)是**先前就存在**的同类悬空名字,**不在 Task 9 范围**,
    # 未改动。
    # 情绪分析器
    "HandEmotionAnalyzer",
    "ShoulderEmotionAnalyzer",
    "ArmEmotionAnalyzer",
    "UpperBodyEmotionAnalyzer",
    "EmotionFusionAnalyzer",
    
    # 特征模型
    "HandFeatures",
    "ShoulderFeatures",
    "ArmFeatures",
    "UpperBodyFeatures",
    "GestureFeatures",
    
    # 情绪结果模型
    "HandEmotionResult",
    "ShoulderEmotionResult",
    "ArmEmotionResult",
    "UpperBodyEmotionResult",
    "EmotionResult",
    "GestureEmotionResult",
    
    # 管道
    "GestureEmotionPipeline",
    
    # 旧接口（兼容）
    "HandAnalyzer",
    "ShoulderAnalyzer",
    "ArmAnalyzer",
    "UpperBodyAnalyzer",
    "EmotionInferencer"
]
