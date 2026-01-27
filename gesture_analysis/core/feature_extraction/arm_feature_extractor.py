
"""
手臂特征提取模块

从MediaPipe姿态landmarks中提取手臂原始特征数据
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from collections import deque


class ArmFeatureExtractor:
    """手臂特征提取器"""

    def __init__(self, arm_id: str = 'left', history_length: int = 30):
        """
        初始化手臂特征提取器

        参数:
            arm_id: 手臂标识 ('left' 或 'right')
            history_length: 历史数据长度（帧）
        """
        if arm_id not in ('left', 'right'):
            raise ValueError("arm_id 必须为 'left' 或 'right'")

        self.arm_id = arm_id
        self.history_length = history_length

        # 初始化历史数据
        self.wrist_history = deque(maxlen=history_length)
        self.elbow_history = deque(maxlen=history_length)

        # 特征数据
        self.features: Dict[str, Any] = {
            "wrist_position": (0.0, 0.0),
            "elbow_position": (0.0, 0.0),
            "shoulder_position": (0.0, 0.0),
            "wrist_jitter": 0.0,
            "elbow_jitter": 0.0,
            "arm_angle": 0.0,
            "is_valid": False
        }

        # 关键点索引映射
        if self.arm_id == 'left':
            self.wrist_idx = 15  # LEFT_WRIST
            self.elbow_idx = 13  # LEFT_ELBOW
            self.shoulder_idx = 11  # LEFT_SHOULDER
        else:
            self.wrist_idx = 16  # RIGHT_WRIST
            self.elbow_idx = 14  # RIGHT_ELBOW
            self.shoulder_idx = 12  # RIGHT_SHOULDER

    def extract(self, landmarks) -> Dict[str, Any]:
        """
        从landmarks中提取手臂特征

        参数:
            landmarks: MediaPipe 姿态关键点列表（需至少包含 33 个点）
                      或包含手腕和手肘关键点的字典

        返回:
            包含提取特征的字典
        """
        if landmarks is None:
            self.features["is_valid"] = False
            return self.features.copy()

        try:
            # 提取关键点坐标
            if hasattr(landmarks, '__getitem__'):
                # MediaPipe landmarks对象
                wrist = (float(landmarks[self.wrist_idx].x), float(landmarks[self.wrist_idx].y))
                elbow = (float(landmarks[self.elbow_idx].x), float(landmarks[self.elbow_idx].y))
                shoulder = (float(landmarks[self.shoulder_idx].x), float(landmarks[self.shoulder_idx].y))
            else:
                # 字典格式
                wrist = (float(landmarks.get('wrist_x', 0)), float(landmarks.get('wrist_y', 0)))
                elbow = (float(landmarks.get('elbow_x', 0)), float(landmarks.get('elbow_y', 0)))
                shoulder = (float(landmarks.get('shoulder_x', 0)), float(landmarks.get('shoulder_y', 0)))

            # 更新历史数据
            self.wrist_history.append(wrist)
            self.elbow_history.append(elbow)

            # 计算特征
            wrist_jitter = self._calculate_jitter(self.wrist_history)
            elbow_jitter = self._calculate_jitter(self.elbow_history)
            arm_angle = self._calculate_arm_angle(shoulder, elbow, wrist)

            # 更新特征数据
            self.features = {
                "wrist_position": wrist,
                "elbow_position": elbow,
                "shoulder_position": shoulder,
                "wrist_jitter": wrist_jitter,
                "elbow_jitter": elbow_jitter,
                "arm_angle": arm_angle,
                "is_valid": True
            }

            return self.features.copy()

        except (AttributeError, IndexError, TypeError, KeyError):
            self.features["is_valid"] = False
            return self.features.copy()

    def _calculate_jitter(self, history: deque) -> float:
        """
        计算抖动幅度

        参数:
            history: 位置历史记录

        返回:
            抖动幅度
        """
        if len(history) < min(10, history.maxlen // 3):
            return 0.0
        positions = np.array(history)
        jitter = np.std(positions, axis=0).mean()
        return float(jitter)

    def _calculate_arm_angle(self, shoulder: Tuple[float, float],
                            elbow: Tuple[float, float],
                            wrist: Tuple[float, float]) -> float:
        """
        计算手臂角度（肘关节角度）

        参数:
            shoulder: 肩关节坐标
            elbow: 肘关节坐标
            wrist: 手腕坐标

        返回:
            手臂角度（度）
        """
        try:
            # 计算向量
            vector_shoulder_elbow = np.array([elbow[0] - shoulder[0], elbow[1] - shoulder[1]])
            vector_elbow_wrist = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]])

            # 计算角度
            dot_product = np.dot(vector_shoulder_elbow, vector_elbow_wrist)
            norm_shoulder = np.linalg.norm(vector_shoulder_elbow)
            norm_wrist = np.linalg.norm(vector_elbow_wrist)

            if norm_shoulder == 0 or norm_wrist == 0:
                return 0.0

            cos_angle = dot_product / (norm_shoulder * norm_wrist)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))

            return float(angle)
        except Exception:
            return 0.0

    def get_features(self) -> Dict[str, Any]:
        """
        获取当前特征

        返回:
            包含当前特征的字典
        """
        return self.features.copy()

    def is_valid(self) -> bool:
        """返回当前特征是否基于有效输入"""
        return self.features.get("is_valid", False)

    def reset(self) -> None:
        """重置提取器状态"""
        self.wrist_history = deque(maxlen=self.history_length)
        self.elbow_history = deque(maxlen=self.history_length)
        self.features = {
            "wrist_position": (0.0, 0.0),
            "elbow_position": (0.0, 0.0),
            "shoulder_position": (0.0, 0.0),
            "wrist_jitter": 0.0,
            "elbow_jitter": 0.0,
            "arm_angle": 0.0,
            "is_valid": False
        }
