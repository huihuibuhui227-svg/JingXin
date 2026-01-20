# face_expression/inference/emotion_infer.py

def infer_emotion_from_au(au_features, temporal_stats=None, micro_expressions=None):
    """
    基于 AU 特征推理多维情绪状态。

    Args:
        au_features (dict): 原始 AU 特征字典
        temporal_stats (dict, optional): 时序统计特征
        micro_expressions (dict, optional): 微表情检测结果

    Returns:
        dict: 包含情绪向量和元数据的结构体，例如：
        {
            "emotion_vector": {"happy": 0.85, "anxiety": 0.3, ...},
            "dominant_emotion": "happy",
            "confidence": 0.85,
            "is_neutral": False
        }
    """
    # 定义所有支持的情绪类别（仅用于概率分布）
    basic_emotions = [
        "happy", "sadness", "anger", "fear", "surprise",
        "disgust", "contempt", "anxiety", "fatigue",
        "polite_smile", "distress"
    ]

    emotions = {e: 0.0 for e in basic_emotions}

    # 提取特征
    au4_frown = au_features.get('au4_frown', 0)
    au12_smile = au_features.get('au12_smile', 0)
    au6_cheek = au_features.get('au6_cheek_raise', 0)
    au7_squeeze = au_features.get('au7_eye_squeeze', 0)
    au15_mouth_down = au_features.get('au15_mouth_down', 0)
    au23_lip_comp = au_features.get('au23_lip_compression', 0)
    au25_open = au_features.get('au25_mouth_open', 0)
    eye_closed_sec = au_features.get('eye_closed_sec', 0)
    tension_score = au_features.get('psychological_signals', {}).get('tension_score', 0)
    symmetry = au_features.get('symmetry_score', 1.0)
    au9_nose = au_features.get('au9_nose_wrinkle', 0)

    # === 1. Happy / Duchenne Smile ===
    if au12_smile > 1.0 and au6_cheek > 0.25:
        intensity = min(au12_smile - 1.0, au6_cheek) * 1.5
        emotions["happy"] = min(intensity, 1.0)
    elif au12_smile > 1.0:
        emotions["polite_smile"] = min((au12_smile - 1.0) * 1.2, 0.7)

    # === 2. Sadness / Distress ===
    if au4_frown > 0.4 and au15_mouth_down > 0.1:
        sadness_intensity = (au4_frown + au15_mouth_down) / 2
        emotions["sadness"] = min(sadness_intensity, 1.0)
        emotions["distress"] = min(sadness_intensity * 0.8, 1.0)

    # === 3. Anger ===
    if au4_frown > 0.5 and au7_squeeze > 0.4:
        anger_intensity = (au4_frown + au7_squeeze) / 2
        emotions["anger"] = min(anger_intensity, 1.0)

    # === 4. Anxiety / Tension ===
    anxiety_base = max(tension_score, au23_lip_comp)
    if micro_expressions and 'au4_frown' in micro_expressions:
        anxiety_base += 0.2
    emotions["anxiety"] = min(anxiety_base, 1.0)

    # === 5. Fatigue ===
    if eye_closed_sec > 1.0:
        emotions["fatigue"] = min(eye_closed_sec / 3.0, 1.0)

    # === 6. Surprise ===
    eyebrow_raise = au_features.get('au12_eyebrow_raise', 0)
    if au25_open > 0.4 and eyebrow_raise > 0.2:
        emotions["surprise"] = min((au25_open + eyebrow_raise) / 2, 1.0)

    # === 7. Contempt (asymmetric smile) ===
    if au12_smile > 1.0 and symmetry < 0.8:
        emotions["contempt"] = min((1.0 - symmetry) * 0.8, 0.6)

    # === 8. Disgust ===
    if au9_nose > 0.3 and au15_mouth_down > 0.2:
        emotions["disgust"] = min((au9_nose + au15_mouth_down) / 2, 1.0)

    # === 9. Fear ===
    blink_rate = au_features.get('blink_rate_per_min', 0)
    if au25_open > 0.5 and blink_rate > 40:
        emotions["fear"] = min(au25_open * 0.8, 1.0)

    # === 归一化（可选）===
    total = sum(emotions.values())
    if total > 0:
        for k in emotions:
            emotions[k] = round(emotions[k] / total, 3)
    else:
        emotions["neutral"] = 1.0

    # === 确定主导情绪 ===
    dominant = max(emotions, key=emotions.get)
    confidence = emotions[dominant]

    return {
        "emotion_vector": emotions,  # 纯情绪概率字典（仅 float）
        "dominant_emotion": dominant,
        "confidence": round(confidence, 3),
        "is_neutral": (confidence < 0.3 and dominant == "neutral")
    }