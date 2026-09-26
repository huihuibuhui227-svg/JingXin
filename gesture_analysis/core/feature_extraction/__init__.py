"""
特征提取模块

从MediaPipe原始landmarks数据中提取手势特征。
⚠️ 2026-09-26 M3.0 Task 9 之后,本包只剩**无状态的纯函数**模块(`angles.py`);
原来那几个带历史缓冲的 `*_feature_extractor.py` 已删(死代码,见下)。
"""

# 2026-09-26 M3.0 Task 9:这里原来导出 4 个 FeatureExtractor
# (HandFeatureExtractor / ArmFeatureExtractor / ShoulderFeatureExtractor /
#  UpperBodyFeatureExtractor)。它们各自模块已删 —— 设计文档
# `docs/superpowers/specs/2026-09-21-jingxin-feature-redesign-design.md` 判它们
# 「删代码 —— 不在 CSV 生产路径上(死代码),与生效 analyzer 取值不同」。
# CSV 生产路径走的是 `gesture_analysis/core/analysis/*_analyzer.py`。
# 本包现在只剩不依赖 landmarks 分帧状态的纯函数模块(`angles.py`)。
__all__ = []
