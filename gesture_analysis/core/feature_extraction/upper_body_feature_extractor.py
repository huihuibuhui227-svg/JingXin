
"""
上肢特征提取模块

从MediaPipe姿态landmarks中提取上肢（头部和躯干）原始特征数据
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from collections import deque


class UpperBodyFeatureExtractor:
    """上肢特征提取器（头部和躯干）"""

    def __init__(self, history_length: int = 30):
        """
        初始化上肢特征提取器

        参数:
            history_length: 历史数据长度（帧）
        """
        self.history_length = history_length

        # 初始化历史数据
        self.head_history = deque(maxlen=history_length)
        self.torso_history = deque(maxlen=history_length)

        # 特征数据
        self.features: Dict[str, Any] = {
            "head_position": (0.0, 0.0),
            "torso_position": (0.0, 0.0),
            "head_jitter": 0.0,
            "torso_jitter": 0.0,
            "head_tilt": 0.0,
            "torso_stability": 0.0,
            "is_valid": False
        }

    def extract(self, landmarks) -> Dict[str, Any]:
        """
        从landmarks中提取上肢特征

        参数:
            landmarks: MediaPipe 姿态关键点列表（需至少包含 33 个点）

        返回:
            包含提取特征的字典
        """
        if landmarks is None or len(landmarks) < 33:
            self.features["is_valid"] = False
            return self.features.copy()

        try:
            # 提取头部关键点（鼻子作为头部中心）
            nose = (float(landmarks[0].x), float(landmarks[0].y))

            # 提取躯干关键点（肩膀中点）
            left_shoulder = (float(landmarks[11].x), float(landmarks[11].y))
            right_shoulder = (float(landmarks[12].x), float(landmarks[12].y))
            torso_center = (
                (left_shoulder[0] + right_shoulder[0]) / 2.0,
                (left_shoulder[1] + right_shoulder[1]) / 2.0
            )

            # 更新历史数据
            self.head_history.append(nose)
            self.torso_history.append(torso_center)

            # 计算特征
            head_jitter = self._calculate_jitter(self.head_history)
            torso_jitter = self._calculate_jitter(self.torso_history)
            head_tilt = self._calculate_head_tilt(landmarks)
            torso_stability = self._calculate_torso_stability(torso_jitter)

            # 更新特征数据
            self.features = {
                "head_position": nose,
                "torso_position": torso_center,
                "head_jitter": head_jitter,
                "torso_jitter": torso_jitter,
                "head_tilt": head_tilt,
                "torso_stability": torso_stability,
                "is_valid": True
            }

            return self.features.copy()

        except (AttributeError, IndexError, TypeError):
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

    def _calculate_head_tilt(self, landmarks) -> float:
        """
        计算头部倾斜角度（左右耳连线与水平线的夹角）

        参数:
            landmarks: MediaPipe 姿态关键点列表

        返回:
            头部倾斜角度（度）
        """
        try:
            left_ear = (float(landmarks[7].x), float(landmarks[7].y))
            right_ear = (float(landmarks[8].x), float(landmarks[8].y))

            # 计算耳部连线角度
            dx = right_ear[0] - left_ear[0]
            dy = right_ear[1] - left_ear[1]
            angle = np.degrees(np.arctan2(dy, dx))

            # 转换为与水平线的夹角
            return float(abs(angle))
        except Exception:
            return 0.0

    def _calculate_torso_stability(self, torso_jitter: float) -> float:
        """
        计算躯干稳定性

        参数:
            torso_jitter: 躯干抖动

        返回:
            稳定性评分 (0-1)
        """
        # 抖动越小，稳定性越高
        stability = max(0.0, 1.0 - torso_jitter * 10.0)
        return float(stability)

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
        self.head_history = deque(maxlen=self.history_length)
        self.torso_history = deque(maxlen=self.history_length)
        self.features = {
            "head_position": (0.0, 0.0),
            "torso_position": (0.0, 0.0),
            "head_jitter": 0.0,
            "torso_jitter": 0.0,
            "head_tilt": 0.0,
            "torso_stability": 0.0,
            "is_valid": False
        }
