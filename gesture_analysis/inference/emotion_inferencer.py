"""
情绪推断模块

提供基于手势和肩部特征的情绪评估功能
融合手部抗压能力与肩部紧张度，输出结构化情绪状态。
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from ..config import EMOTION_CONFIG


EmotionInferenceResult = Dict[str, Any]


class EmotionInferencer:
    """情绪推断器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化情绪推断器

        参数:
            config: 配置字典，如果为None则使用默认配置
        """
        self.config = config or EMOTION_CONFIG.copy()

    def infer_emotion(
        self,
        hand_results: Optional[Dict[str, Any]],
        shoulder_results: Optional[Dict[str, Any]]
    ) -> EmotionInferenceResult:
        """
        推断情绪状态

        参数:
            hand_results: 手部分析结果（来自 HandAnalyzer.get_results()）
            shoulder_results: 肩部分析结果（来自 ShoulderAnalyzer.get_results()）

        返回:
            情绪评估结果字典，包含：
            - overall_score: 综合评分 (0-100)
            - emotion_state: 情绪状态文本
            - emoji: 对应 emoji
            - feedback: 建议反馈
            - color: BGR 颜色元组 (用于可视化)
            - is_valid: 是否基于有效输入
            - used_features: 使用了哪些有效特征（'hand', 'shoulder', 'both', 'none'）
        """
        # 判断输入有效性
        hand_valid = self._is_valid_result(hand_results, required_key='resilience_score')
        shoulder_valid = self._is_valid_result(shoulder_results, required_key='shoulder_score')

        # 提取分数，无效时使用默认值但标记来源
        hand_score = float(hand_results.get('resilience_score', 50.0)) if hand_valid else 50.0
        shoulder_score = float(shoulder_results.get('shoulder_score', 50.0)) if shoulder_valid else 50.0

        # 确定使用了哪些特征
        if hand_valid and shoulder_valid:
            used_features = "both"
            overall_score = (
                hand_score * self.config['hand_weight'] +
                shoulder_score * self.config['shoulder_weight']
            )
        elif hand_valid:
            used_features = "hand"
            overall_score = hand_score  # 仅用手部
        elif shoulder_valid:
            used_features = "shoulder"
            overall_score = shoulder_score  # 仅用肩部
        else:
            used_features = "none"
            overall_score = 50.0  # 完全无效，返回中性

        overall_score = float(np.clip(overall_score, 0.0, 100.0))

        # 映射情绪状态
        emotion_state, emoji = self._map_score_to_emotion(overall_score)
        feedback, color = self._get_emotion_feedback(overall_score)

        return {
            "overall_score": overall_score,
            "emotion_state": emotion_state,
            "emoji": emoji,
            "feedback": feedback,
            "color": color,
            "is_valid": used_features != "none",
            "used_features": used_features
        }

    def _is_valid_result(self, result: Optional[Dict], required_key: str) -> bool:
        """
        判断分析结果是否有效

        参数:
            result: 分析结果字典
            required_key: 必需的键名

        返回:
            是否有效
        """
        if not isinstance(result, dict):
            return False
        if not result.get('is_valid', False):
            return False
        if required_key not in result:
            return False
        try:
            float(result[required_key])
            return True
        except (TypeError, ValueError):
            return False

    def _map_score_to_emotion(self, score: float) -> Tuple[str, str]:
        """
        将评分映射到情绪状态

        参数:
            score: 综合评分 (0-100)

        返回:
            (情绪状态文本, emoji 表情)
        """
        ranges = self.config['score_ranges']

        # 注意：区间是 >=，从高到低判断
        if score >= ranges['very_relaxed']:
            return "非常放松", "🟢"
        elif score >= ranges['relaxed']:
            return "放松", "🟢"
        elif score >= ranges['neutral']:
            return "中性", "🟡"
        elif score >= ranges['slightly_nervous']:
            return "轻微紧张", "🟠"
        elif score >= ranges['nervous']:
            return "紧张", "🔴"
        else:
            return "高度焦虑", "🔴"

    def _get_emotion_feedback(self, score: float) -> Tuple[str, Tuple[int, int, int]]:
        """
        获取情绪反馈文本和颜色

        参数:
            score: 综合评分 (0-100)

        返回:
            (反馈文本, BGR 颜色元组)
        """
        emotion_state, _ = self._map_score_to_emotion(score)

        if "放松" in emotion_state:
            return "你看起来很放松，状态很棒！", (0, 255, 0)      # 绿色
        elif "中性" in emotion_state:
            return "状态平稳，保持自然呼吸", (255, 255, 0)       # 黄色
        elif "轻微紧张" in emotion_state:
            return "注意放松手部和肩膀，深呼吸～", (255, 165, 0)  # 橙色
        else:
            return "你可能处于紧张或焦虑状态，建议暂停休息", (0, 0, 255)  # 红色