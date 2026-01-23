from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class AUFeatures:
    # 眉部
    au1_inner_brow_raise: float = 0.0
    au2_outer_brow_raise: float = 0.0
    au4_frown: float = 0.0

    # 眼部
    au6_cheek_raise: float = 0.0
    au7_eye_squeeze: float = 0.0
    avg_ear: float = 0.0

    # 鼻部
    au9_nose_wrinkle: float = 0.0

    # 嘴部
    au10_upper_lip_raise: float = 0.0
    au12_smile: float = 0.0
    au14_dimpler: float = 0.0
    au15_mouth_down: float = 0.0
    au20_lip_stretcher: float = 0.0
    au23_lip_compression: float = 0.0
    au25_mouth_open: float = 0.0
    au26_jaw_drop: float = 0.0

    # 头部与对称性
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    symmetry_score: float = 1.0

    # 生理指标
    blink_rate_per_min: float = 0.0
    eye_closed_sec: float = 0.0
    is_blink: bool = False  # 新增：当前帧是否眨眼

    # 新增：视线追踪
    left_iris_x: float = 0.0
    left_iris_y: float = 0.0
    right_iris_x: float = 0.0
    right_iris_y: float = 0.0
    gaze_direction_x: float = 0.0
    gaze_direction_y: float = 0.0
    gaze_deviation: float = 0.0

    # 可选：原始 landmarks（[x1,y1,x2,y2,...]）
    landmarks: Optional[List[float]] = None


@dataclass
class TemporalStats:
    data: Dict[str, float]  # ✅ 字段名为 data


@dataclass
class MicroExpressionResult:
    data: Dict[str, dict]  # ✅ 字段名为 data