"""
手部分析模块

提供手部特征提取和抗压能力评估功能
"""

import numpy as np
from collections import deque
from typing import Dict, Any, Optional, Tuple
from ..config import HAND_CONFIG

# 定义明确的返回类型，便于类型检查和文档生成
HandAnalysisResult = Dict[str, Any]


class HandAnalyzer:
    """手部分析器"""

    def __init__(self, hand_id: int = 0, config: Optional[Dict[str, float]] = None):
        """
        初始化手部分析器

        参数:
            hand_id: 手部ID (0或1)
            config: 配置字典，如果为None则使用默认配置
        """
        if hand_id not in (0, 1):
            raise ValueError("hand_id 必须为 0 或 1")

        self.hand_id = hand_id
        self.config = config or HAND_CONFIG.copy()  # 避免外部修改污染

        # 初始化历史数据：仅跟踪食指指尖（ID=8）
        self.tip_history = {8: deque(maxlen=int(self.config['history_length']))}

        # 分析状态标志
        self._is_valid = False  # 是否有有效 landmarks 输入
        self.reset()

    def reset(self) -> None:
        """重置分析状态"""
        self.results: HandAnalysisResult = {
            "resilience_score": 50.0,
            "jitter": 0.0,
            "fist_status": False,
            "spread": 0.0,
            "is_valid": False  # 新增：标识本次分析是否基于有效输入
        }
        self._is_valid = False

    def update(self, landmarks) -> None:
        """
        更新手部数据

        参数:
            landmarks: MediaPipe 手部关键点列表（需至少包含 21 个点）

        异常:
            ValueError: 当 landmarks 无效或长度不足时
        """
        if landmarks is None:
            self._handle_invalid_input()
            return

        if len(landmarks) < 21:
            raise ValueError(f"手部 landmarks 长度不足，期望 >=21，实际: {len(landmarks)}")

        try:
            # 更新食指指尖历史
            self.tip_history[8].append((float(landmarks[8].x), float(landmarks[8].y)))

            # 计算特征
            jitter = self._calculate_jitter()
            is_fist = self._is_fist(landmarks)
            spread = self._calculate_finger_spread(landmarks)
            score = self._compute_resilience_score(jitter, is_fist, spread)

            # 更新结果
            self.results = {
                "resilience_score": float(np.clip(score, 0.0, 100.0)),
                "jitter": jitter,
                "fist_status": bool(is_fist),
                "spread": spread,
                "is_valid": True
            }
            self._is_valid = True

        except (AttributeError, IndexError, TypeError) as e:
            # 捕获 landmarks 格式错误（如缺少 .x/.y 属性）
            self._handle_invalid_input()
            # 可选：记录警告（未来可接入 logger）
            # print(f"[WARN] HandAnalyzer: landmarks 格式异常 - {e}")

    def _handle_invalid_input(self) -> None:
        """处理无效输入，保持状态一致"""
        self.results["is_valid"] = False
        self._is_valid = False

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
            threshold: 握拳阈值，如果为None则使用配置中的值

        返回:
            是否握拳的布尔值
        """
        threshold = threshold or self.config['fist_threshold']
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

    def _compute_resilience_score(self, jitter: float, is_fist: bool, spread: float) -> float:
        """
        计算抗压能力评分

        参数:
            jitter: 抖动幅度
            is_fist: 是否握拳
            spread: 手指张开度

        返回:
            抗压能力评分 (0-100)
        """
        jitter_penalty = jitter * self.config['jitter_multiplier']
        jitter_score = max(0.0, 70.0 - jitter_penalty)
        fist_penalty = self.config['fist_penalty'] if is_fist else 0.0
        spread_bonus = 0.0
        if spread > self.config['spread_threshold']:
            spread_bonus = min(
                self.config['spread_bonus_multiplier'],
                (spread - self.config['spread_threshold']) * self.config['spread_bonus_multiplier']
            )
        score = jitter_score - fist_penalty + spread_bonus
        return float(score)

    def get_results(self) -> HandAnalysisResult:
        """
        获取分析结果

        返回:
            包含分析结果的字典，包含 is_valid 字段标识有效性
        """
        return self.results.copy()  # 防止外部修改内部状态

    def is_valid(self) -> bool:
        """返回当前分析状态是否基于有效输入"""
        return self._is_valid