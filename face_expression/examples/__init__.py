"""
JingXin - Face Expression 模块
面部微表情分析核心模块，支持实时视频与静态图片分析。
"""

# 导出分析器类（使用绝对路径）
from face_expression.pipeline.video_pipeline import VideoPipeline as FaceAUAnalyzer
from face_expression.pipeline.image_pipeline import ImagePipeline as StaticFaceAnalyzer

# 导出情绪推断函数
from face_expression.core.analysis.emotion_engine import EmotionEngine

# 辅助函数：方便外部调用
def infer_emotion_from_au(au_features, temporal_stats=None, micro_expressions=None):
    """从 AU 特征推断情绪（兼容旧接口）"""
    engine = EmotionEngine()
    return engine.infer(au_features, temporal_stats, micro_expressions)

__version__ = "1.0.0"
__all__ = [
    "FaceAUAnalyzer",
    "StaticFaceAnalyzer",
    "infer_emotion_from_au",
    "EmotionEngine"  # 可选：暴露引擎供高级用户使用
]