"""
肩部分析模块

提供肩部特征提取和紧张度评估功能
"""

import numpy as np
from collections import deque
from typing import Dict, Any, Optional, Tuple
from ..config import SHOULDER_CONFIG


ShoulderAnalysisResult = Dict[str, Any]


class ShoulderAnalyzer:
    """肩部分析器"""

    def __init__(self, config: Optional[Dict[str, float]] = None):
        """
        初始化肩部分析器

        参数:
            config: 配置字典，如果为None则使用默认配置
        """
        self.config = config or SHOULDER_CONFIG.copy()

        self.shoulder_history = {
            'left': deque(maxlen=int(self.config['history_length'])),
            'right': deque(maxlen=int(self.config['history_length']))
        }

        # 校准状态
        self.shoulder_baseline_y: Optional[float] = None
        self.baseline_frames_collected = 0

        # 分析状态
        self._is_valid = False
        self.reset()

    def reset(self) -> None:
        """重置分析状态"""
        self.results: ShoulderAnalysisResult = {
            "left_jitter": 0.0,
            "right_jitter": 0.0,
            "shrug_level": 0.0,
            "shoulder_score": 50.0,
            "is_valid": False,
            "is_calibrated": False
        }
        self.shoulder_baseline_y = None
        self.baseline_frames_collected = 0
        self._is_valid = False

    def update(self, landmarks) -> None:
        """
        更新肩部数据

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
            left_shoulder = (float(landmarks[11].x), float(landmarks[11].y))
            right_shoulder = (float(landmarks[12].x), float(landmarks[12].y))
            self.shoulder_history['left'].append(left_shoulder)
            self.shoulder_history['right'].append(right_shoulder)

            left_jitter, right_jitter = self._calculate_shoulder_jitter()
            shrug = self._calculate_shrug_level(landmarks)
            score = self._compute_shoulder_score(left_jitter, right_jitter, shrug)

            is_calibrated = self.is_calibrated()
            self.results = {
                "left_jitter": left_jitter,
                "right_jitter": right_jitter,
                "shrug_level": shrug,
                "shoulder_score": float(np.clip(score, 0.0, 100.0)),
                "is_valid": True,
                "is_calibrated": is_calibrated
            }
            self._is_valid = True

        except (AttributeError, IndexError, TypeError) as e:
            self._handle_invalid_input()
            # print(f"[WARN] ShoulderAnalyzer: landmarks 格式异常 - {e}")

    def _handle_invalid_input(self) -> None:
        """处理无效输入"""
        self.results.update({
            "is_valid": False,
            "is_calibrated": self.is_calibrated()
        })
        self._is_valid = False

    def _calculate_shoulder_jitter(self) -> Tuple[float, float]:
        """
        计算肩部抖动幅度

        返回:
            (左肩抖动, 右肩抖动)
        """
        min_len = min(10, self.shoulder_history['left'].maxlen // 3)
        if (len(self.shoulder_history['left']) < min_len or
            len(self.shoulder_history['right']) < min_len):
            return 0.0, 0.0

        left_jitter = np.std(np.array(self.shoulder_history['left']), axis=0).mean()
        right_jitter = np.std(np.array(self.shoulder_history['right']), axis=0).mean()
        return float(left_jitter), float(right_jitter)

    def _calculate_shrug_level(self, landmarks) -> float:
        """
        计算耸肩程度

        参数:
            landmarks: MediaPipe姿态关键点

        返回:
            耸肩程度 (0-1)
        """
        try:
            left_y = landmarks[11].y
            right_y = landmarks[12].y
            avg_y = (left_y + right_y) / 2.0
        except (AttributeError, IndexError):
            return 0.0

        # 校准阶段
        if self.baseline_frames_collected < self.config['baseline_frames_needed']:
            if self.shoulder_baseline_y is None:
                self.shoulder_baseline_y = avg_y
            else:
                self.shoulder_baseline_y = (
                    self.shoulder_baseline_y * self.config['baseline_smoothing'] +
                    avg_y * (1 - self.config['baseline_smoothing'])
                )
            self.baseline_frames_collected += 1
            return 0.0

        # 耸肩判断：y 值越小表示位置越高（图像坐标系）
        if avg_y < self.shoulder_baseline_y:
            shrug_diff = self.shoulder_baseline_y - avg_y
            shrug_norm = min(shrug_diff, self.config['max_shrug_diff']) / self.config['max_shrug_diff']
            return float(shrug_norm)
        return 0.0

    def _compute_shoulder_score(self, left_jitter: float, right_jitter: float, shrug: float) -> float:
        """
        计算肩部评分

        参数:
            left_jitter: 左肩抖动
            right_jitter: 右肩抖动
            shrug: 耸肩程度

        返回:
            肩部评分 (0-100)
        """
        avg_jitter = (left_jitter + right_jitter) / 2.0
        jitter_penalty = avg_jitter * self.config['jitter_multiplier']
        shrug_penalty = shrug * self.config['shrug_penalty']
        score = 70.0 - jitter_penalty - shrug_penalty
        return float(score)

    def get_results(self) -> ShoulderAnalysisResult:
        """
        获取分析结果

        返回:
            包含分析结果的字典
        """
        return self.results.copy()

    def is_calibrated(self) -> bool:
        """
        检查是否已完成校准

        返回:
            是否已校准的布尔值
        """
        return self.baseline_frames_collected >= self.config['baseline_frames_needed']

    def is_valid(self) -> bool:
        """返回当前分析状态是否基于有效输入"""
        return self._is_valid