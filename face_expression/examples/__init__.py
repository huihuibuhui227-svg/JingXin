"""
JingXin - Face Expression 模块
面部微表情分析核心模块，支持实时视频与静态图片分析。
"""

# 导出分析器类
from ..analyzers.face_au_analyzer import FaceAUAnalyzer
from ..analyzers.image_analyzer import StaticFaceAnalyzer

# 导出情绪推断函数
from ..inference.emotion_infer import infer_emotion_from_au

# 可选：如果你确实需要在外部使用 MediaPipe 的绘图工具，
# 建议让用户直接 import mediapipe，而不是通过本模块中转
# （保持依赖解耦）

__version__ = "1.0.0"
__all__ = [
    "FaceAUAnalyzer",
    "StaticFaceAnalyzer",
    "infer_emotion_from_au"
]