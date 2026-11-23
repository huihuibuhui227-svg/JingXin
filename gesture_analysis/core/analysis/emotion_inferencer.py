"""
情绪推断模块

提供基于手势和肩部特征的情绪评估功能
融合手部抗压能力与肩部紧张度，输出结构化情绪状态。
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from gesture_analysis.config import EMOTION_CONFIG

EmotionInferenceResult = Dict[str, Any]

import statistics


class EmotionInferencer:
    """情绪推断器"""
    import statistics

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
            shoulder_results: Optional[Dict[str, Any]],
            left_arm_results: Optional[Dict[str, Any]] = None,
            right_arm_results: Optional[Dict[str, Any]] = None
    ) -> EmotionInferenceResult:
        """
        推断情绪状态

        参数:
            hand_results: 手部分析结果（来自 HandAnalyzer.get_results()）
            shoulder_results: 肩部分析结果（来自 ShoulderAnalyzer.get_results()）
            left_arm_results: 左手臂分析结果（来自 ArmAnalyzer.get_results()）
            right_arm_results: 右手臂分析结果（来自 ArmAnalyzer.get_results()）

        返回:
            情绪评估结果字典，包含：
            - overall_score: 综合评分 (0-100)
            - emotion_state: 情绪状态文本
            - emoji: 对应 emoji
            - feedback: 建议反馈
            - color: BGR 颜色元组 (用于可视化)
            - is_valid: 是否基于有效输入
            - used_features: 使用了哪些有效特征
        """
        # 判断输入有效性
        hand_valid = self._is_valid_result(hand_results, required_key='resilience_score')
        shoulder_valid = self._is_valid_result(shoulder_results, required_key='shoulder_score')
        left_arm_valid = self._is_valid_result(left_arm_results, required_key='arm_score')
        right_arm_valid = self._is_valid_result(right_arm_results, required_key='arm_score')

        # 提取分数，无效时使用默认值但标记来源
        hand_score = float(hand_results.get('resilience_score', 50.0)) if hand_valid else 50.0
        shoulder_score = float(shoulder_results.get('shoulder_score', 50.0)) if shoulder_valid else 50.0
        left_arm_score = float(left_arm_results.get('arm_score', 50.0)) if left_arm_valid else 50.0
        right_arm_score = float(right_arm_results.get('arm_score', 50.0)) if right_arm_valid else 50.0

        # 计算手臂平均分
        arm_valid = left_arm_valid or right_arm_valid
        if left_arm_valid and right_arm_valid:
            arm_score = (left_arm_score + right_arm_score) / 2.0
        elif left_arm_valid:
            arm_score = left_arm_score
        elif right_arm_valid:
            arm_score = right_arm_score
        else:
            arm_score = 50.0

        # 确定使用了哪些特征
        used_features_list = []
        if hand_valid:
            used_features_list.append("hand")
        if shoulder_valid:
            used_features_list.append("shoulder")
        if arm_valid:
            used_features_list.append("arm")

        used_features = "+".join(used_features_list) if used_features_list else "none"

        # 计算综合评分
        if used_features != "none":
            total_weight = 0.0
            weighted_score = 0.0

            if hand_valid:
                weighted_score += hand_score * self.config['hand_weight']
                total_weight += self.config['hand_weight']

            if shoulder_valid:
                weighted_score += shoulder_score * self.config['shoulder_weight']
                total_weight += self.config['shoulder_weight']

            if arm_valid:
                weighted_score += arm_score * self.config['arm_weight']
                total_weight += self.config['arm_weight']

            overall_score = weighted_score / total_weight if total_weight > 0 else 50.0
        else:
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
        将上肢运动平稳度指数映射为客观活动水平档位（不做情绪或心理诊断）。

        参数:
            score: 上肢平稳度参考值 (0-100)

        返回:
            (客观运动状态文本, 状态标识)
        """
        ranges = self.config['score_ranges']

        # 注意：区间是 >=，从高到低判断
        if score >= ranges['very_relaxed']:
            return "动作极平稳", "🟢"
        elif score >= ranges['relaxed']:
            return "动作平稳", "🟢"
        elif score >= ranges['neutral']:
            return "动作适中", "🟡"
        elif score >= ranges['slightly_nervous']:
            return "动作略活跃", "🟠"
        elif score >= ranges['nervous']:
            return "动作较活跃", "🔴"
        else:
            return "动作高频波动", "🔴"

    def _get_emotion_feedback(self, score: float) -> Tuple[str, Tuple[int, int, int]]:
        """
        获取客观上肢运动观测提示和颜色（不输出情绪/心理诊断措辞）。

        参数:
            score: 上肢平稳度参考值 (0-100)

        返回:
            (观测提示文本, BGR 颜色元组)
        """
        motion_state, _ = self._map_score_to_emotion(score)

        if "平稳" in motion_state:
            return "上肢与肩部姿态保持平稳", (0, 255, 0)
        elif "适中" in motion_state:
            return "上肢动作幅度适中，姿态稳定", (255, 255, 0)
        elif "略活跃" in motion_state:
            return "上肢位移略有增加，请保持在摄像头取景范围内", (255, 165, 0)
        else:
            return "检测到较高频的上肢位移波动，建议保持坐姿稳定以便采集", (0, 0, 255)