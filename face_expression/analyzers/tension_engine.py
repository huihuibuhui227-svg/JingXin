# face_expression/analyzers/tension_engine.py

class TensionEngine:
    def compute(self, au_features, temporal_stats, emotion_vector=None):
        au4 = au_features.get('au4_frown', 0)
        au23 = au_features.get('au23_lip_compression', 0)
        eye_closed = min(au_features.get('eye_closed_sec', 0) / 2.0, 1.0)
        smile_vol = temporal_stats.get('au12_smile_volatility', 0)
        asymmetry = au_features.get('gaze_asymmetry', 0)

        base_tension = (
            0.3 * au4 +
            0.25 * au23 +
            0.2 * eye_closed +
            0.15 * smile_vol +
            0.1 * asymmetry
        )

        emotional_boost = 0.0
        if emotion_vector:
            # 高焦虑/愤怒/恐惧会提升紧张度
            anxiety = emotion_vector.get('anxiety', 0)
            anger = emotion_vector.get('anger', 0)
            fear = emotion_vector.get('fear', 0)
            moral_disgust = emotion_vector.get('moral_disgust', 0)
            emotional_boost = max(anxiety, anger, fear, moral_disgust)
            base_tension = min(base_tension + emotional_boost * 0.3, 1.0)

        tension = min(base_tension, 1.0)
        level = "low" if tension < 0.3 else "medium" if tension < 0.6 else "high"

        return {
            'tension_score': round(tension, 3),
            'tension_level': level,
            'tension_sources': {
                'brow_furrow': round(au4, 3),
                'lip_compression': round(au23, 3),
                'eye_closure': round(eye_closed, 3),
                'expression_instability': round(smile_vol, 3),
                'emotional_influence': round(emotional_boost * 0.3, 3)
            }
        }