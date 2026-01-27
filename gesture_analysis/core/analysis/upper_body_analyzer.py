"""
上肢分析模块

提供头部和躯干特征提取与评估功能
"""

import numpy as np
from collections import deque
from typing import Dict, Any, Optional, Tuple
from gesture_analysis.config import ARM_CONFIG
import math


UpperBodyAnalysisResult = Dict[str, Any]


class UpperBodyAnalyzer:
    """上肢分析器（头部和躯干）"""

    def __init__(self, config: Optional[Dict[str, float]] = None):
        """
        初始化上肢分析器

        参数:
            config: 配置字典，如果为None则使用默认配置
        """
        self.config = config or ARM_CONFIG.copy()

        # 初始化历史数据
        self.head_history = deque(maxlen=int(self.config['history_length']))
        self.torso_history = deque(maxlen=int(self.config['history_length']))

        # 分析状态
        self._is_valid = False
        self.reset()

    def reset(self) -> None:
        """重置分析状态"""
        self.results: UpperBodyAnalysisResult = {
            "head_score": 50.0,
            "torso_score": 50.0,
            "head_jitter": 0.0,
            "torso_jitter": 0.0,
            "head_tilt": 0.0,
            "torso_stability": 0.0,
            "is_valid": False
        }
        self._is_valid = False

    def update(self, landmarks) -> None:
        """
        更新上肢数据

        参数:
            landmarks: MediaPipe 姿态关键点列表（需至少包含 33 个点）

        异常:
            ValueError: 当 landmarks 无效或长度不足时
        """
        if landmarks is None:
            self._handle_invalid_input()
            return

        if len(landmarks) < 33:
            raise ValueError(f"姿态 landmarks 长度不足，期望 >=33，实际: {len(landmarks)}")

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
            head_score = self._compute_head_score(head_jitter, head_tilt)
            torso_score = self._compute_torso_score(torso_jitter, torso_stability)

            # 更新结果
            self.results = {
                "head_score": float(np.clip(head_score, 0.0, 100.0)),
                "torso_score": float(np.clip(torso_score, 0.0, 100.0)),
                "head_jitter": head_jitter,
                "torso_jitter": torso_jitter,
                "head_tilt": head_tilt,
                "torso_stability": torso_stability,
                "is_valid": True
            }
            self._is_valid = True

        except (AttributeError, IndexError, TypeError) as e:
            self._handle_invalid_input()

    def _handle_invalid_input(self) -> None:
        """处理无效输入"""
        self.results["is_valid"] = False
        self._is_valid = False

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
        计算头部倾斜角度

        参数:
            landmarks: MediaPipe姿态关键点

        返回:
            头部倾斜角度（度）
        """
        try:
            # 使用左右耳连线作为参考
            left_ear = np.array([landmarks[7].x, landmarks[7].y])
            right_ear = np.array([landmarks[8].x, landmarks[8].y])

            # 计算水平角度
            ear_vector = right_ear - left_ear
            angle = math.degrees(math.atan2(ear_vector[1], ear_vector[0]))

            return float(abs(angle))
        except (AttributeError, IndexError):
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
        stability = max(0.0, 1.0 - torso_jitter * self.config['jitter_multiplier'] / 100.0)
        return float(stability)

    def _compute_head_score(self, jitter: float, tilt: float) -> float:
        """
        计算头部评分

        参数:
            jitter: 头部抖动
            tilt: 头部倾斜角度

        返回:
            头部评分 (0-100)
        """
        # 基础分
        base_score = 70.0

        # 抖动惩罚
        jitter_penalty = jitter * self.config['jitter_multiplier']

        # 倾斜惩罚（理想角度在0-10度）
        if tilt <= 10:
            tilt_penalty = 0.0
        elif tilt <= 20:
            tilt_penalty = 5.0
        else:
            tilt_penalty = 10.0

        score = base_score - jitter_penalty - tilt_penalty
        return float(score)

    def _compute_torso_score(self, jitter: float, stability: float) -> float:
        """
        计算躯干评分

        参数:
            jitter: 躯干抖动
            stability: 躯干稳定性

        返回:
            躯干评分 (0-100)
        """
        # 基础分
        base_score = 70.0

        # 抖动惩罚
        jitter_penalty = jitter * self.config['jitter_multiplier']

        # 稳定性奖励
        stability_bonus = stability * self.config['stability_bonus']

        score = base_score - jitter_penalty + stability_bonus
        return float(score)

    def get_results(self) -> UpperBodyAnalysisResult:
        """
        获取分析结果

        返回:
            包含分析结果的字典
        """
        return self.results.copy()

    def is_valid(self) -> bool:
        """返回当前分析状态是否基于有效输入"""
        return self._is_valid
