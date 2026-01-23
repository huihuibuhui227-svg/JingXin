from ..core.feature_extraction.au_calculator import AUFeatureCalculator
from ..core.analysis.emotion_engine import EmotionEngine
from ..core.analysis.tension_engine import TensionEngine
from ..models.features import TemporalStats, MicroExpressionResult

class ImagePipeline:
    def __init__(self):
        self.feature_calculator = AUFeatureCalculator()
        self.emotion_engine = EmotionEngine()
        self.tension_engine = TensionEngine()

    def process_image(self, landmarks_norm, face_width, face_height):
        au_features = self.feature_calculator.calculate(landmarks_norm, face_width, face_height)
        emotion_result = self.emotion_engine.infer(au_features)
        tension_result = self.tension_engine.compute(au_features, TemporalStats({}))
        return {
            "au_features": au_features,
            "emotion_result": emotion_result,
            "tension_result": tension_result
        }