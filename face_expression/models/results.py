from dataclasses import dataclass
from typing import Dict, List
from .features import AUFeatures, TemporalStats, MicroExpressionResult


@dataclass
class EmotionResult:
    emotion_vector: Dict[str, float]
    dominant_emotion: str
    confidence: float
    composite_emotions: List[str]
    psychological_summary: str


@dataclass
class TensionResult:
    tension_score: float
    tension_level: str
    tension_sources: Dict[str, float]


@dataclass
class AnalysisFrameResult:
    session_id: str
    timestamp: float
    focus_score: float
    au_features: AUFeatures
    temporal_stats: TemporalStats
    micro_expressions: MicroExpressionResult
    emotion_result: EmotionResult
    tension_result: TensionResult

    def to_dict(self):
        """
        转换为扁平化字典，严格匹配 face_logs 表的 182 列顺序。
        所有字段均有默认值，避免写入 NULL。
        """
        # === 提取各部分数据 ===
        au = self.au_features
        ts_data = self.temporal_stats.data or {}
        me_dict = self.micro_expressions.data or {}
        ev = self.emotion_result.emotion_vector or {}
        tr = self.tension_result

        # === 微表情处理：取第一个有效项 ===
        if me_dict:
            first_key = next(iter(me_dict))
            me_info = me_dict[first_key]
            micro_exp_au_name = first_key
            micro_exp_intensity = me_info.get("intensity", 0.0)
            micro_exp_duration_frames = me_info.get("duration_frames", 0)
            micro_exp_onset_frame = me_info.get("onset_frame", 0)
        else:
            micro_exp_au_name = None
            micro_exp_intensity = None
            micro_exp_duration_frames = None
            micro_exp_onset_frame = None

        # === tension_sources 安全提取 ===
        tension_sources = tr.tension_sources or {}
        brow_furrow = tension_sources.get("brow_furrow", 0.0)
        lip_compression = tension_sources.get("lip_compression", 0.0)
        eye_closure = tension_sources.get("eye_closure", 0.0)
        expr_instability = tension_sources.get("expression_instability", 0.0)
        emotional_influence = tension_sources.get("emotional_influence", 0.0)

        # === 构建扁平化字典（严格按表结构顺序）===
        result = {
            # 基础信息
            "session_id": self.session_id,
            "timestamp": self.timestamp,

            # 基础评分
            "focus_score": round(float(self.focus_score), 3),
            "symmetry_score": round(float(au.symmetry_score), 3),

            # AU 特征
            "au1_inner_brow_raise": round(float(au.au1_inner_brow_raise), 3),
            "au2_outer_brow_raise": round(float(au.au2_outer_brow_raise), 3),
            "au4_frown": round(float(au.au4_frown), 3),
            "au6_cheek_raise": round(float(au.au6_cheek_raise), 3),
            "au7_eye_squeeze": round(float(au.au7_eye_squeeze), 3),
            "au9_nose_wrinkle": round(float(au.au9_nose_wrinkle), 3),
            "au10_upper_lip_raise": round(float(au.au10_upper_lip_raise), 3),
            "au12_smile": round(float(au.au12_smile), 3),
            "au14_dimpler": round(float(au.au14_dimpler), 3),
            "au15_mouth_down": round(float(au.au15_mouth_down), 3),
            "au20_lip_stretcher": round(float(au.au20_lip_stretcher), 3),
            "au23_lip_compression": round(float(au.au23_lip_compression), 3),
            "au25_mouth_open": round(float(au.au25_mouth_open), 3),
            "au26_jaw_drop": round(float(au.au26_jaw_drop), 3),
            "avg_ear": round(float(au.avg_ear), 3),

            # 头部姿态
            "head_yaw": round(float(au.head_yaw), 3),
            "head_pitch": round(float(au.head_pitch), 3),

            # 眨眼
            "blink_rate_per_min": round(float(au.blink_rate_per_min), 3),
            "eye_closed_sec": round(float(au.eye_closed_sec), 3),
            "is_blink": int(bool(au.is_blink)),

            # 视线追踪
            "left_iris_x": round(float(au.left_iris_x), 3),
            "left_iris_y": round(float(au.left_iris_y), 3),
            "right_iris_x": round(float(au.right_iris_x), 3),
            "right_iris_y": round(float(au.right_iris_y), 3),
            "gaze_direction_x": round(float(au.gaze_direction_x), 3),
            "gaze_direction_y": round(float(au.gaze_direction_y), 3),
            "gaze_deviation": round(float(au.gaze_deviation), 3),

            # 心理信号
            "tension_score": round(float(tr.tension_score), 3),
            "tension_level": str(tr.tension_level),

            # 紧张源（扁平化）
            "tension_sources_brow_furrow": round(float(brow_furrow), 3),
            "tension_sources_lip_compression": round(float(lip_compression), 3),
            "tension_sources_eye_closure": round(float(eye_closure), 3),
            "tension_sources_expression_instability": round(float(expr_instability), 3),
            "tension_sources_emotional_influence": round(float(emotional_influence), 3),

            # 微表情
            "micro_exp_au_name": micro_exp_au_name,
            "micro_exp_intensity": round(float(micro_exp_intensity), 3) if micro_exp_intensity is not None else None,
            "micro_exp_duration_frames": int(micro_exp_duration_frames) if micro_exp_duration_frames is not None else None,
            "micro_exp_onset_frame": int(micro_exp_onset_frame) if micro_exp_onset_frame is not None else None,
        }

        # === 时间动态统计（90+ 字段）===
        # 定义所有需要的时间统计字段前缀（✅ 补全 avg_ear）
        ts_prefixes = [
            "au1_inner_brow_raise", "au2_outer_brow_raise", "au4_frown", "au6_cheek_raise",
            "au7_eye_squeeze", "au9_nose_wrinkle", "au10_upper_lip_raise", "au12_smile",
            "au14_dimpler", "au15_mouth_down", "au20_lip_stretcher", "au23_lip_compression",
            "au25_mouth_open", "au26_jaw_drop", "avg_ear",  # ✅ 已包含
            "head_yaw", "head_pitch",
            "symmetry_score",
            "blink_rate_per_min", "eye_closed_sec", "is_blink",
            "left_iris_x", "left_iris_y", "right_iris_x", "right_iris_y",
            "gaze_direction_x", "gaze_direction_y", "gaze_deviation"
        ]

        for prefix in ts_prefixes:
            for suffix in ["_trend", "_volatility", "_change_rate"]:
                key = prefix + suffix
                value = ts_data.get(key, 0.0)
                result[key] = round(float(value), 3) if value is not None else 0.0

        # === 情绪向量（15 维）===
        emotion_keys = [
            "happy", "sadness", "anger", "fear", "surprise",
            "disgust", "contempt", "anxiety", "fatigue",
            "polite_smile", "distress", "forced_smile",
            "startled_anxiety", "cognitive_load", "moral_disgust"
        ]
        for key in emotion_keys:
            value = ev.get(key, 0.0)
            result[f"emotion_{key}"] = round(float(value), 3) if value is not None else 0.0

        # === 最终输出 ===
        result["dominant_emotion"] = str(self.emotion_result.dominant_emotion)
        result["confidence"] = round(float(self.emotion_result.confidence), 3)
        result["overall_score"] = round(float(getattr(self.emotion_result, "overall_score", 0.0)), 3)
        result["emotion_state"] = str(getattr(self.emotion_result, "emotion_state", "neutral"))
        result["is_valid"] = int(True)  # 始终有效

        return result