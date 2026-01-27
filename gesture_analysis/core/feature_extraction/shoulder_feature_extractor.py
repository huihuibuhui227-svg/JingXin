
"""
肩部特征提取模块

从MediaPipe姿态landmarks中提取肩部原始特征数据
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from collections import deque


class ShoulderFeatureExtractor:
    """肩部特征提取器"""

    def __init__(self, history_length: int = 30):
        """
        初始化肩部特征提取器

        参数:
            history_length: 历史数据长度（帧）
        """
        self.history_length = history_length

        self.shoulder_history = {
            'left': deque(maxlen=history_length),
            'right': deque(maxlen=history_length)
        }

        # 校准状态
        self.shoulder_baseline_y: Optional[float] = None
        self.baseline_frames_collected = 0
        self.baseline_frames_needed = 30  # 初始校准所需帧数

        # 特征数据
        self.features: Dict[str, Any] = {
            "left_position": (0.0, 0.0),
            "right_position": (0.0, 0.0),
            "left_jitter": 0.0,
            "right_jitter": 0.0,
            "height_diff": 0.0,
            "shrug_level": 0.0,
            "is_valid": False,
            "is_calibrated": False
        }

    def extract(self, landmarks) -> Dict[str, Any]:
        """
        从landmarks中提取肩部特征

        参数:
            landmarks: MediaPipe 姿态关键点列表（需至少包含 33 个点）

        返回:
            包含提取特征的字典
        """
        if landmarks is None or len(landmarks) < 33:
            self.features["is_valid"] = False
            return self.features.copy()

        try:
            left_shoulder = (float(landmarks[11].x), float(landmarks[11].y))
            right_shoulder = (float(landmarks[12].x), float(landmarks[12].y))

            self.shoulder_history['left'].append(left_shoulder)
            self.shoulder_history['right'].append(right_shoulder)

            # 计算特征
            left_jitter, right_jitter = self._calculate_shoulder_jitter()
            height_diff = abs(left_shoulder[1] - right_shoulder[1])
            shrug = self._calculate_shrug_level(left_shoulder, right_shoulder)

            # 更新特征数据
            self.features = {
                "left_position": left_shoulder,
                "right_position": right_shoulder,
                "left_jitter": left_jitter,
                "right_jitter": right_jitter,
                "height_diff": height_diff,
                "shrug_level": shrug,
                "is_valid": True,
                "is_calibrated": self.is_calibrated()
            }

            return self.features.copy()

        except (AttributeError, IndexError, TypeError):
            self.features["is_valid"] = False
            return self.features.copy()

    def _calculate_shoulder_jitter(self) -> Tuple[float, float]:
        """
        计算肩部抖动

        返回:
            (左肩抖动, 右肩抖动)
        """
        def calc_jitter(history):
            if len(history) < min(10, history.maxlen // 3):
                return 0.0
            positions = np.array(history)
            return float(np.std(positions, axis=0).mean())

        return calc_jitter(self.shoulder_history['left']), calc_jitter(self.shoulder_history['right'])

    def _calculate_shrug_level(self, left_shoulder: Tuple[float, float],
                              right_shoulder: Tuple[float, float]) -> float:
        """
        计算耸肩程度

        参数:
            left_shoulder: 左肩坐标
            right_shoulder: 右肩坐标

        返回:
            耸肩程度 (0-1)
        """
        try:
            # 更新基准线（如果未校准）
            if not self.is_calibrated():
                avg_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
                if self.shoulder_baseline_y is None:
                    self.shoulder_baseline_y = avg_y
                else:
                    # 使用指数移动平均平滑基准线
                    self.shoulder_baseline_y = (
                        self.shoulder_baseline_y * 0.9 + avg_y * 0.1
                    )
                self.baseline_frames_collected += 1

            # 计算当前肩部高度与基准线的差异
            avg_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
            diff = abs(avg_y - self.shoulder_baseline_y) if self.shoulder_baseline_y else 0.0

            # 归一化耸肩程度（假设最大差异为0.1）
            max_diff = 0.1
            shrug_level = min(diff / max_diff, 1.0)

            return float(shrug_level)

        except Exception:
            return 0.0

    def is_calibrated(self) -> bool:
        """返回是否已完成校准"""
        return self.baseline_frames_collected >= self.baseline_frames_needed

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
        self.shoulder_history = {
            'left': deque(maxlen=self.history_length),
            'right': deque(maxlen=self.history_length)
        }
        self.shoulder_baseline_y = None
        self.baseline_frames_collected = 0
        self.features = {
            "left_position": (0.0, 0.0),
            "right_position": (0.0, 0.0),
            "left_jitter": 0.0,
            "right_jitter": 0.0,
            "height_diff": 0.0,
            "shrug_level": 0.0,
            "is_valid": False,
            "is_calibrated": False
        }
