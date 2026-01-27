
"""
手部特征提取模块

从MediaPipe手部landmarks中提取原始特征数据
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from collections import deque


class HandFeatureExtractor:
    """手部特征提取器"""

    def __init__(self, hand_id: int = 0, history_length: int = 30):
        """
        初始化手部特征提取器

        参数:
            hand_id: 手部ID (0或1)
            history_length: 历史数据长度（帧）
        """
        if hand_id not in (0, 1):
            raise ValueError("hand_id 必须为 0 或 1")

        self.hand_id = hand_id
        self.history_length = history_length

        # 初始化历史数据：仅跟踪食指指尖（ID=8）
        self.tip_history = {8: deque(maxlen=history_length)}

        # 特征数据
        self.features: Dict[str, Any] = {
            "jitter": 0.0,
            "fist_status": False,
            "spread": 0.0,
            "finger_positions": {},
            "is_valid": False
        }

    def extract(self, landmarks) -> Dict[str, Any]:
        """
        从landmarks中提取手部特征

        参数:
            landmarks: MediaPipe 手部关键点列表（需至少包含 21 个点）

        返回:
            包含提取特征的字典
        """
        if landmarks is None or len(landmarks) < 21:
            self.features["is_valid"] = False
            return self.features.copy()

        try:
            # 更新食指指尖历史
            self.tip_history[8].append((float(landmarks[8].x), float(landmarks[8].y)))

            # 计算特征
            jitter = self._calculate_jitter()
            is_fist = self._is_fist(landmarks)
            spread = self._calculate_finger_spread(landmarks)
            finger_positions = self._extract_finger_positions(landmarks)

            # 更新特征数据
            self.features = {
                "jitter": jitter,
                "fist_status": is_fist,
                "spread": spread,
                "finger_positions": finger_positions,
                "is_valid": True
            }

            return self.features.copy()

        except (AttributeError, IndexError, TypeError):
            self.features["is_valid"] = False
            return self.features.copy()

    def _calculate_jitter(self) -> float:
        """计算手指抖动幅度"""
        if len(self.tip_history[8]) < min(10, self.tip_history[8].maxlen // 3):
            return 0.0
        positions = np.array(self.tip_history[8])
        jitter = np.std(positions, axis=0).mean()
        return float(jitter)

    def _is_fist(self, landmarks, threshold: Optional[float] = None) -> bool:
        """
        判断是否握拳

        参数:
            landmarks: 手部关键点
            threshold: 握拳阈值

        返回:
            是否握拳的布尔值
        """
        if threshold is None:
            threshold = 0.08  # 默认握拳阈值

        tips = [8, 12, 16, 20]
        dips = [7, 11, 15, 19]
        try:
            distances = [
                np.linalg.norm(
                    np.array([landmarks[tip].x, landmarks[tip].y]) -
                    np.array([landmarks[dip].x, landmarks[dip].y])
                )
                for tip, dip in zip(tips, dips)
            ]
            return bool(np.mean(distances) < threshold)
        except (AttributeError, IndexError):
            return False

    def _calculate_finger_spread(self, landmarks) -> float:
        """
        计算手指张开度

        参数:
            landmarks: 手部关键点

        返回:
            手指张开度值
        """
        try:
            palm = np.array([landmarks[0].x, landmarks[0].y])
            tips = [4, 8, 12, 16, 20]
            spread = np.mean([
                np.linalg.norm(np.array([landmarks[i].x, landmarks[i].y]) - palm)
                for i in tips
            ])
            return float(spread)
        except (AttributeError, IndexError):
            return 0.0

    def _extract_finger_positions(self, landmarks) -> Dict[str, Tuple[float, float]]:
        """
        提取所有手指关键点位置

        参数:
            landmarks: 手部关键点

        返回:
            包含所有手指关键点位置的字典
        """
        try:
            positions = {}
            # 拇指
            positions['thumb'] = [
                (float(landmarks[i].x), float(landmarks[i].y)) 
                for i in range(1, 5)
            ]
            # 食指
            positions['index'] = [
                (float(landmarks[i].x), float(landmarks[i].y)) 
                for i in range(5, 9)
            ]
            # 中指
            positions['middle'] = [
                (float(landmarks[i].x), float(landmarks[i].y)) 
                for i in range(9, 13)
            ]
            # 无名指
            positions['ring'] = [
                (float(landmarks[i].x), float(landmarks[i].y)) 
                for i in range(13, 17)
            ]
            # 小指
            positions['pinky'] = [
                (float(landmarks[i].x), float(landmarks[i].y)) 
                for i in range(17, 21)
            ]
            return positions
        except (AttributeError, IndexError):
            return {}

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
        self.tip_history = {8: deque(maxlen=self.history_length)}
        self.features = {
            "jitter": 0.0,
            "fist_status": False,
            "spread": 0.0,
            "finger_positions": {},
            "is_valid": False
        }
