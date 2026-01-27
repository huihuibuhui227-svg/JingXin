"""
日志工具模块

提供手势和肩部分析结果的日志记录功能
支持结构化 CSV 日志，便于后续分析与审计。
"""

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
from ..config import LOGS_DIR, LOG_CONFIG


class GestureLogger:
    """手势分析日志记录器"""

    def __init__(self, log_dir: Optional[str] = None, log_file_name: Optional[str] = None):
        """
        初始化日志记录器

        参数:
            log_dir: 日志目录路径，若为 None 则使用 config.LOGS_DIR
            log_file_name: 日志文件名模板，若为 None 则使用 config 中的模板
        """
        self.log_dir = Path(log_dir) if log_dir else Path(LOGS_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 生成带时间戳的日志文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_template = log_file_name or LOG_CONFIG['gesture_log_file']
        self.log_file = self.log_dir / file_template.format(timestamp=timestamp)

        # 定义字段名（包含所有原始特征数据和屏幕显示的角度）
        self.fieldnames = [
            # 时间戳
            "timestamp",                # Unix 时间戳（秒）
            "timestamp_iso",            # ISO 8601 格式时间（便于阅读）

            # 手部特征
            "left_hand_score",
            "left_hand_jitter",
            "left_hand_fist_status",
            "left_hand_spread",
            "right_hand_score",
            "right_hand_jitter",
            "right_hand_fist_status",
            "right_hand_spread",

            # 手指角度（屏幕显示）
            "left_thumb_angle",
            "left_index_angle",
            "left_middle_angle",
            "left_ring_angle",
            "left_pinky_angle",
            "right_thumb_angle",
            "right_index_angle",
            "right_middle_angle",
            "right_ring_angle",
            "right_pinky_angle",

            # 肩部特征
            "shoulder_score",
            "left_shoulder_jitter",
            "right_shoulder_jitter",
            "shrug_level",
            "is_calibrated",

            # 手臂特征
            "left_arm_score",
            "left_wrist_jitter",
            "left_elbow_jitter",
            "left_arm_angle",
            "left_arm_stability",
            "right_arm_score",
            "right_wrist_jitter",
            "right_elbow_jitter",
            "right_arm_angle",
            "right_arm_stability",

            # 手臂角度（屏幕显示）
            "left_elbow_angle",
            "right_elbow_angle",
            "left_shoulder_angle",
            "right_shoulder_angle",

            # 上半身特征
            "head_score",
            "head_jitter",
            "head_tilt",
            "torso_score",
            "torso_jitter",
            "torso_stability",

            # 头部角度（屏幕显示）
            "head_tilt_angle",
            "head_pitch_angle",

            # 肩部角度（屏幕显示）
            "shoulder_angle",

            # 躯干角度（屏幕显示）
            "torso_angle",

            # 情绪特征
            "overall_score",
            "emotion_state",
            "feedback",
            "used_features",
            "is_valid"
        ]

        # 写入文件头（仅一次）
        self._write_header()

    def _write_header(self) -> None:
        """写入 CSV 文件头（幂等操作）"""
        if not self.log_file.exists():
            with open(self.log_file, 'w', newline='', encoding=LOG_CONFIG['encoding']) as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(
        self,
        left_hand_result: Optional[Dict[str, Any]],
        right_hand_result: Optional[Dict[str, Any]],
        shoulder_result: Optional[Dict[str, Any]],
        left_arm_result: Optional[Dict[str, Any]] = None,
        right_arm_result: Optional[Dict[str, Any]] = None,
        upper_body_result: Optional[Dict[str, Any]] = None,
        emotion_result: Optional[Dict[str, Any]] = None,
        angles_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        记录分析结果到日志文件

        参数:
            left_hand_result: 左手分析结果
            right_hand_result: 右手分析结果
            shoulder_result: 肩部分析结果
            left_arm_result: 左手臂分析结果
            right_arm_result: 右手臂分析结果
            upper_body_result: 上半身分析结果
            emotion_result: 情绪评估结果
            angles_data: 屏幕显示的所有角度数据

        返回:
            是否成功写入
        """
        try:
            now = datetime.now()
            data = {
                # 时间戳
                "timestamp": now.timestamp(),
                "timestamp_iso": now.isoformat(),

                # 手部特征
                "left_hand_score": self._safe_get(left_hand_result, 'resilience_score', 0.0),
                "left_hand_jitter": self._safe_get(left_hand_result, 'jitter', 0.0),
                "left_hand_fist_status": int(self._safe_get(left_hand_result, 'fist_status', False)),
                "left_hand_spread": self._safe_get(left_hand_result, 'spread', 0.0),
                "right_hand_score": self._safe_get(right_hand_result, 'resilience_score', 0.0),
                "right_hand_jitter": self._safe_get(right_hand_result, 'jitter', 0.0),
                "right_hand_fist_status": int(self._safe_get(right_hand_result, 'fist_status', False)),
                "right_hand_spread": self._safe_get(right_hand_result, 'spread', 0.0),

                # 手指角度（屏幕显示）
                "left_thumb_angle": self._safe_get_angle(angles_data, 'left_finger_angles', 'thumb'),
                "left_index_angle": self._safe_get_angle(angles_data, 'left_finger_angles', 'index'),
                "left_middle_angle": self._safe_get_angle(angles_data, 'left_finger_angles', 'middle'),
                "left_ring_angle": self._safe_get_angle(angles_data, 'left_finger_angles', 'ring'),
                "left_pinky_angle": self._safe_get_angle(angles_data, 'left_finger_angles', 'pinky'),
                "right_thumb_angle": self._safe_get_angle(angles_data, 'right_finger_angles', 'thumb'),
                "right_index_angle": self._safe_get_angle(angles_data, 'right_finger_angles', 'index'),
                "right_middle_angle": self._safe_get_angle(angles_data, 'right_finger_angles', 'middle'),
                "right_ring_angle": self._safe_get_angle(angles_data, 'right_finger_angles', 'ring'),
                "right_pinky_angle": self._safe_get_angle(angles_data, 'right_finger_angles', 'pinky'),

                # 肩部特征
                "shoulder_score": self._safe_get(shoulder_result, 'shoulder_score', 0.0),
                "left_shoulder_jitter": self._safe_get(shoulder_result, 'left_jitter', 0.0),
                "right_shoulder_jitter": self._safe_get(shoulder_result, 'right_jitter', 0.0),
                "shrug_level": self._safe_get(shoulder_result, 'shrug_level', 0.0),
                "is_calibrated": int(self._safe_get(shoulder_result, 'is_calibrated', False)),

                # 手臂特征
                "left_arm_score": self._safe_get(left_arm_result, 'arm_score', 0.0),
                "left_wrist_jitter": self._safe_get(left_arm_result, 'wrist_jitter', 0.0),
                "left_elbow_jitter": self._safe_get(left_arm_result, 'elbow_jitter', 0.0),
                "left_arm_angle": self._safe_get(left_arm_result, 'arm_angle', 0.0),
                "left_arm_stability": self._safe_get(left_arm_result, 'arm_stability', 0.0),
                "right_arm_score": self._safe_get(right_arm_result, 'arm_score', 0.0),
                "right_wrist_jitter": self._safe_get(right_arm_result, 'wrist_jitter', 0.0),
                "right_elbow_jitter": self._safe_get(right_arm_result, 'elbow_jitter', 0.0),
                "right_arm_angle": self._safe_get(right_arm_result, 'arm_angle', 0.0),
                "right_arm_stability": self._safe_get(right_arm_result, 'arm_stability', 0.0),

                # 手臂角度（屏幕显示）
                "left_elbow_angle": self._safe_get(angles_data, 'left_elbow_angle', None),
                "right_elbow_angle": self._safe_get(angles_data, 'right_elbow_angle', None),
                "left_shoulder_angle": self._safe_get(angles_data, 'left_shoulder_angle', None),
                "right_shoulder_angle": self._safe_get(angles_data, 'right_shoulder_angle', None),

                # 上半身特征
                "head_score": self._safe_get(upper_body_result, 'head_score', 0.0),
                "head_jitter": self._safe_get(upper_body_result, 'head_jitter', 0.0),
                "head_tilt": self._safe_get(upper_body_result, 'head_tilt', 0.0),
                "torso_score": self._safe_get(upper_body_result, 'torso_score', 0.0),
                "torso_jitter": self._safe_get(upper_body_result, 'torso_jitter', 0.0),
                "torso_stability": self._safe_get(upper_body_result, 'torso_stability', 0.0),

                # 头部角度（屏幕显示）
                "head_tilt_angle": self._safe_get(angles_data, 'head_tilt_angle', None),
                "head_pitch_angle": self._safe_get(angles_data, 'head_pitch_angle', None),

                # 肩部角度（屏幕显示）
                "shoulder_angle": self._safe_get(angles_data, 'shoulder_angle', None),

                # 躯干角度（屏幕显示）
                "torso_angle": self._safe_get(angles_data, 'torso_angle', None),

                # 情绪特征
                "overall_score": self._safe_get(emotion_result, 'overall_score', 0.0),
                "emotion_state": self._safe_get(emotion_result, 'emotion_state', ''),
                "feedback": self._safe_get(emotion_result, 'feedback', ''),
                "used_features": self._safe_get(emotion_result, 'used_features', 'none'),
                "is_valid": bool(self._safe_get(emotion_result, 'is_valid', False))
            }

            with open(self.log_file, 'a', newline='', encoding=LOG_CONFIG['encoding']) as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(data)
            return True

        except Exception as e:
            print(f"❌ 日志记录失败: {str(e)}")
            return False

    def _safe_get(self, obj: Optional[Dict], key: str, default):
        """安全获取字典值，避免 KeyError 或 AttributeError"""
        if not isinstance(obj, dict):
            return default
        return obj.get(key, default)

    def _safe_get_angle(self, angles_data: Optional[Dict], finger_angles_key: str, finger_name: str):
        """安全获取手指角度数据"""
        if not isinstance(angles_data, dict):
            return None
        finger_angles = angles_data.get(finger_angles_key)
        if not isinstance(finger_angles, dict):
            return None
        angle = finger_angles.get(finger_name)
        return round(angle, 1) if angle is not None else None

    def get_log_path(self) -> str:
        """获取日志文件路径"""
        return str(self.log_file)