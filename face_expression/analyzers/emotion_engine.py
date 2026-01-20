# face_expression/analyzers/emotion_engine.py

class EmotionEngine:
    def infer(self, au_features, temporal_stats=None, micro_expressions=None):
        # 基础情绪 + 复合情绪
        emotions = {
            "happy": 0.0, "sadness": 0.0, "anger": 0.0, "fear": 0.0, "surprise": 0.0,
            "disgust": 0.0, "contempt": 0.0, "anxiety": 0.0, "fatigue": 0.0,
            "polite_smile": 0.0, "distress": 0.0,
            # === 新增复合情绪 ===
            "forced_smile": 0.0,
            "startled_anxiety": 0.0,
            "cognitive_load": 0.0,
            "moral_disgust": 0.0
        }

        # 提取特征
        au1 = au_features.get('au1_inner_brow_raise', 0)
        au2 = au_features.get('au2_outer_brow_raise', 0)
        au4 = au_features.get('au4_frown', 0)
        au6 = au_features.get('au6_cheek_raise', 0)
        au7 = au_features.get('au7_eye_squeeze', 0)
        au9 = au_features.get('au9_nose_wrinkle', 0)
        au10 = au_features.get('au10_upper_lip_raise', 0)
        au12 = au_features.get('au12_smile', 0)
        au14 = au_features.get('au14_dimpler', 0)
        au15 = au_features.get('au15_mouth_down', 0)
        au20 = au_features.get('au20_lip_stretcher', 0)
        au23 = au_features.get('au23_lip_compression', 0)
        au25 = au_features.get('au25_mouth_open', 0)
        au26 = au_features.get('au26_jaw_drop', 0)
        symmetry = au_features.get('symmetry_score', 1.0)
        head_yaw = au_features.get('head_yaw', 0)
        avg_ear = au_features.get('avg_ear', 0)

        # === 基础情绪 ===
        if au12 > 0.1 and au6 > 0.1:
            emotions["happy"] = min((au12 + au6) / 2, 1.0)
        elif au12 > 0.1:
            emotions["polite_smile"] = au12 * 0.7

        if (au1 > 0.1 or au2 > 0.1) and au26 > 0.3:
            emotions["surprise"] = min((au1 + au2 + au26) / 3, 1.0)

        if au9 > 0.2 and au10 > 0.1:
            emotions["disgust"] = min((au9 + au10) / 2, 1.0)

        if au4 > 0.3 and au7 > 0.3:
            emotions["anger"] = (au4 + au7) / 2

        if au4 > 0.2 and au15 > 0.05:
            sadness = (au4 + au15) / 2
            emotions["sadness"] = sadness
            emotions["distress"] = sadness * 0.8

        if (au1 > 0.15 or au2 > 0.15) and au26 > 0.5 and head_yaw > 0.1:
            emotions["fear"] = min((au1 + au2 + au26) / 3, 1.0)

        if avg_ear < 0.15:
            emotions["fatigue"] = 1.0 - avg_ear

        # === 复合情绪 ===
        if au12 > 0.1 and au20 > 0.1 and au23 > 0.2:
            emotions["forced_smile"] = min((au12 + au20 + au23) / 3, 1.0)

        if temporal_stats and \
           temporal_stats.get('au1_inner_brow_raise_trend', 0) > 0.01 and \
           temporal_stats.get('au4_frown_trend', 0) > 0.01:
            emotions["startled_anxiety"] = 0.7

        if au4 > 0.2 and au7 > 0.3 and head_yaw < -0.1:
            emotions["cognitive_load"] = min((au4 + au7) / 2, 1.0)

        if au9 > 0.3 and au10 > 0.2 and au4 > 0.3:
            emotions["moral_disgust"] = min((au9 + au10 + au4) / 3, 1.0)

        if au12 > 0.1 and symmetry < 0.7:
            emotions["contempt"] = min(au12 * (1 - symmetry), 1.0)

        # Anxiety
        anxiety_score = 0.0
        if au4 > 0.1:
            anxiety_score += au4 * 0.5
        if au23 > 0.2:
            anxiety_score += (au23 - 0.2) * 1.5
        if head_yaw < -0.05:
            anxiety_score += 0.2
        emotions["anxiety"] = min(anxiety_score, 1.0)

        # 归一化
        total = sum(emotions.values())
        if total > 0:
            for k in emotions:
                emotions[k] = round(emotions[k] / total, 3)
        else:
            emotions["neutral"] = 1.0

        dominant = max(emotions, key=emotions.get)
        confidence = emotions[dominant]

        # === NLG: 生成心理状态摘要 ===
        summary = self._generate_psychological_summary(emotions, au_features, micro_expressions)

        return {
            "emotion_vector": emotions,
            "dominant_emotion": dominant,
            "confidence": round(confidence, 3),
            "composite_emotions": [k for k, v in emotions.items() if v > 0.3 and k not in [
                "happy", "sadness", "anger", "fear", "surprise", "disgust"
            ]],
            "psychological_summary": summary
        }

    def _generate_psychological_summary(self, emotions, au_features, micro_expressions):
        """生成自然语言心理状态描述"""
        summary_parts = []

        # 主导情绪
        dominant = max(emotions, key=emotions.get)
        conf = emotions[dominant]
        if conf > 0.5:
            summary_parts.append(f"主导情绪为{dominant}（置信度{conf:.2f}）")

        # 复合情绪
        composite = [k for k in ["forced_smile", "contempt", "cognitive_load", "moral_disgust"] if emotions.get(k, 0) > 0.3]
        if composite:
            summary_parts.append("检测到复合情绪：" + "、".join(composite))

        # 微表情
        if micro_expressions and any(micro_expressions.values()):
            summary_parts.append("发现微表情活动")

        # 紧张度
        tension = au_features.get('psychological_signals', {}).get('tension_score', 0)
        if tension > 0.6:
            summary_parts.append("处于高紧张状态")
        elif tension > 0.3:
            summary_parts.append("有中等紧张表现")

        if not summary_parts:
            return "当前情绪状态平稳"
        return "；".join(summary_parts) + "。"