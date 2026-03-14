from ...models.features import AUFeatures, TemporalStats
from ...models.results import TensionResult


class TensionEngine:
    def compute(self, au_features: AUFeatures, temporal_stats: TemporalStats, emotion_vector=None):
        au4 = au_features.au4_frown
        au23 = au_features.au23_lip_compression
        eye_closed = min(au_features.eye_closed_sec / 2.0, 1.0)

        # === 表情不稳定性（增强版）===
        expression_instability = 0.0
        smile_vol = 0.0

        if temporal_stats and hasattr(temporal_stats, 'data') and temporal_stats.data:
            smile_vol = temporal_stats.data.get('au12_smile_volatility', 0.0)
            volatilities = [
                v for k, v in temporal_stats.data.items()
                if k.endswith('_volatility') and isinstance(v, (int, float)) and v > 0
            ]
            if volatilities:
                expression_instability = min(sum(volatilities) / len(volatilities) * 1.5, 1.0)
            else:
                # 备用：基于当前AU估算
                au_vals = [au4, au23, au_features.au1_inner_brow_raise, au_features.au6_cheek_raise]
                expression_instability = min(sum(au_vals) * 0.8, 1.0)
        else:
            expression_instability = 0.3  # 默认非零值

        # === 基础紧张度（放大微小信号 + 调整权重）===
        base_tension = (
            0.25 * min(au4 * 1.3, 1.0) +
            0.25 * min(au23 * 1.3, 1.0) +
            0.2 * eye_closed +
            0.15 * min(smile_vol * 1.5, 1.0) +
            0.15 * expression_instability
        )

        # === 情绪增强 ===
        emotional_boost = 0.0
        if emotion_vector:
            anxiety = emotion_vector.get('anxiety', 0)
            anger = emotion_vector.get('anger', 0)
            fear = emotion_vector.get('fear', 0)
            moral_disgust = emotion_vector.get('moral_disgust', 0)
            emotional_boost = max(anxiety, anger, fear, moral_disgust)
            base_tension = min(base_tension + emotional_boost * 0.4, 1.0)

        tension = min(base_tension, 1.0)
        level = "low" if tension < 0.3 else "medium" if tension < 0.6 else "high"

        return TensionResult(
            tension_score=round(tension, 3),
            tension_level=level,
            tension_sources={
                'brow_furrow': round(min(au4 * 1.3, 1.0), 3),
                'lip_compression': round(min(au23 * 1.3, 1.0), 3),
                'eye_closure': round(eye_closed, 3),
                'expression_instability': round(expression_instability, 3),
                'emotional_influence': round(emotional_boost * 0.4, 3)
            }
        )