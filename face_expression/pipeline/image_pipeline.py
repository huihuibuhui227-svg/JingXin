from ..core.feature_extraction.au_calculator import AUFeatureCalculator
from ..core.analysis.emotion_engine import EmotionEngine
from ..core.analysis.tension_engine import TensionEngine
from ..models.features import TemporalStats, MicroExpressionResult

class ImagePipeline:
    def __init__(self):
        self.feature_calculator = AUFeatureCalculator()
        self.emotion_engine = EmotionEngine()
        self.tension_engine = TensionEngine()

    def process_image(self, landmarks_norm, face_width, face_outline_height):
        # ⚠️ 本类**没有生产调用方**(`face_expression/examples/run_image_analyzer.py` 走的是
        # 一个不存在的 `face_expression.analyzers.image_analyzer`)。第三个参数 2026-09-27
        # 跟着 `video_pipeline` 一起改名:`face_height`(鼻尖→下巴)→ `face_outline_height`
        # (前额顶→下巴)—— 名字不跟着改的话,调用方会按旧含义去算一个错的尺度。
        au_features = self.feature_calculator.calculate(
            landmarks_norm, face_width, face_outline_height)
        emotion_result = self.emotion_engine.infer(au_features)
        tension_result = self.tension_engine.compute(au_features, TemporalStats({}))
        return {
            "au_features": au_features,
            "emotion_result": emotion_result,
            "tension_result": tension_result
        }