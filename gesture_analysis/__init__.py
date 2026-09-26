
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

# 导出数据模型
# ⚠️ 2026-09-26 M3.0 整支复核(Important 5):下面这 5 个名字此前**挂在 `__all__` 上却没有 import**
# —— 它们是**悬空名字**,`from gesture_analysis import *` 会因此 `AttributeError`(改前实测直接炸)。
# 它们**确实存在**,由 `gesture_analysis/models/__init__.py:__all__` 导出(那是单一真源),
# 所以正确的处置是**把它们收进来**,而不是删名字。同一批里另外 12 个名字(见 `__all__` 的注释)
# 才是**全仓根本不存在**的,那些删。
from gesture_analysis.models import (
    HandFeatures,
    ShoulderFeatures,
    ArmFeatures,
    UpperBodyFeatures,
    EmotionResult,
)

__all__ = [
    # 新接口
    # ⚠️ 2026-09-26 M3.0 Task 9:这里原来还有 4 个 FeatureExtractor 名字
    # (HandFeatureExtractor / ArmFeatureExtractor / ShoulderFeatureExtractor /
    #  UpperBodyFeatureExtractor)。它们**从来没有在本文件 import 过**(纯悬空名字,
    # `from gesture_analysis import *` 会因此 AttributeError),对应的模块也已删。
    # 一并清掉。
    #
    # ⚠️ 2026-09-26 M3.0 整支复核(Important 5):这里原来还有 **12 个全仓不存在的名字** ——
    #   `HandEmotionAnalyzer` / `ShoulderEmotionAnalyzer` / `ArmEmotionAnalyzer` /
    #   `UpperBodyEmotionAnalyzer` / `EmotionFusionAnalyzer` / `GestureFeatures` /
    #   `HandEmotionResult` / `ShoulderEmotionResult` / `ArmEmotionResult` /
    #   `UpperBodyEmotionResult` / `GestureEmotionResult` / `GestureEmotionPipeline`。
    # 它们既**没有任何定义**(全仓 0 处代码定义),也**没有对应 import** ⟹ 同样是悬空名字,
    # 而且比上面那 4 个更毒:那 4 个至少有过模块文件,这 12 个**从来不存在**。
    # 已删。留下的 5 个特征/结果模型是**真的**(见上面的 import)。
    # ⟹ 现在 `from gesture_analysis import *` 真的可用 —— 由
    # `tests/test_gesture_dead_code_removed.py` 里**断言**守着(docstring 里那句"不替它们背书"
    # 没有约束力,2026-09-26 整支复核已把它升级成断言)。

    # 特征模型(来自 `gesture_analysis.models` —— 单一真源在这里再导出一次)
    "HandFeatures",
    "ShoulderFeatures",
    "ArmFeatures",
    "UpperBodyFeatures",

    # 情绪结果模型
    "EmotionResult",

    # 旧接口（兼容）
    "HandAnalyzer",
    "ShoulderAnalyzer",
    "ArmAnalyzer",
    "UpperBodyAnalyzer",
    "EmotionInferencer"
]
