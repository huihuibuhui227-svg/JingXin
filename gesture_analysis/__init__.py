
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
    # 特征提取器
    "HandFeatureExtractor",
    "ShoulderFeatureExtractor",
    "ArmFeatureExtractor",
    "UpperBodyFeatureExtractor",
    
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
