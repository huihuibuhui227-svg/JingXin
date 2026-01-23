from ...models.features import AUFeatures, TemporalStats
from ...models.results import TensionResult

class TensionEngine:
    def compute(self, au_features: AUFeatures, temporal_stats: TemporalStats, emotion_vector=None):
        au4 = au_features.au4_frown
        au23 = au_features.au23_lip_compression
        eye_closed = min(au_features.eye_closed_sec / 2.0, 1.0)
        smile_vol = temporal_stats.data.get('au12_smile_volatility', 0)
        asymmetry = 0.0  # gaze_asymmetry 未在 AUFeatures 中，暂设为0

        base_tension = (
            0.3 * au4 +
            0.25 * au23 +
            0.2 * eye_closed +
            0.15 * smile_vol +
            0.1 * asymmetry
        )

        emotional_boost = 0.0
        if emotion_vector:
            anxiety = emotion_vector.get('anxiety', 0)
            anger = emotion_vector.get('anger', 0)
            fear = emotion_vector.get('fear', 0)
            moral_disgust = emotion_vector.get('moral_disgust', 0)
            emotional_boost = max(anxiety, anger, fear, moral_disgust)
            base_tension = min(base_tension + emotional_boost * 0.3, 1.0)

        tension = min(base_tension, 1.0)
        level = "low" if tension < 0.3 else "medium" if tension < 0.6 else "high"

        return TensionResult(
            tension_score=round(tension, 3),
            tension_level=level,
            tension_sources={
                'brow_furrow': round(au4, 3),
                'lip_compression': round(au23, 3),
                'eye_closure': round(eye_closed, 3),
                'expression_instability': round(smile_vol, 3),
                'emotional_influence': round(emotional_boost * 0.3, 3)
            }
        )