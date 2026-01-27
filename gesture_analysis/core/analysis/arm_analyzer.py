
"""
手臂分析模块

提供手腕和手肘特征提取与评估功能
"""

import numpy as np
from collections import deque
from typing import Dict, Any, Optional, Tuple
from gesture_analysis.config import ARM_CONFIG


ArmAnalysisResult = Dict[str, Any]


class ArmAnalyzer:
    """手臂分析器"""

    def __init__(self, arm_id: str = 'left', config: Optional[Dict[str, float]] = None):
        """
        初始化手臂分析器

        参数:
            arm_id: 手臂标识 ('left' 或 'right')
            config: 配置字典，如果为None则使用默认配置
        """
        if arm_id not in ('left', 'right'):
            raise ValueError("arm_id 必须为 'left' 或 'right'")

        self.arm_id = arm_id
        self.config = config or ARM_CONFIG.copy()

        # 初始化历史数据
        self.wrist_history = deque(maxlen=int(self.config['history_length']))
        self.elbow_history = deque(maxlen=int(self.config['history_length']))

        # 分析状态标志
        self._is_valid = False
        self.reset()

    def reset(self) -> None:
        """重置分析状态"""
        self.results: ArmAnalysisResult = {
            "arm_score": 50.0,
            "wrist_jitter": 0.0,
            "elbow_jitter": 0.0,
            "arm_angle": 0.0,
            "arm_stability": 0.0,
            "is_valid": False
        }
        self._is_valid = False

    def update(self, landmarks) -> None:
        """
        更新手臂数据

        参数:
            landmarks: MediaPipe 姿态关键点列表（需至少包含 33 个点）
                      或包含手腕和手肘关键点的字典

        异常:
            ValueError: 当 landmarks 无效或缺少必要关键点时
        """
        if landmarks is None:
            self._handle_invalid_input()
            return

        try:
            # 根据arm_id选择对应的关键点索引
            if self.arm_id == 'left':
                wrist_idx = 15  # LEFT_WRIST
                elbow_idx = 13  # LEFT_ELBOW
                shoulder_idx = 11  # LEFT_SHOULDER
            else:
                wrist_idx = 16  # RIGHT_WRIST
                elbow_idx = 14  # RIGHT_ELBOW
                shoulder_idx = 12  # RIGHT_SHOULDER

            # 提取关键点坐标
            if hasattr(landmarks, '__getitem__'):
                # MediaPipe landmarks对象
                wrist = (float(landmarks[wrist_idx].x), float(landmarks[wrist_idx].y))
                elbow = (float(landmarks[elbow_idx].x), float(landmarks[elbow_idx].y))
                shoulder = (float(landmarks[shoulder_idx].x), float(landmarks[shoulder_idx].y))
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
            arm_stability = self._calculate_arm_stability(wrist_jitter, elbow_jitter)
            arm_score = self._compute_arm_score(wrist_jitter, elbow_jitter, arm_angle, arm_stability)

            # 更新结果
            self.results = {
                "arm_score": float(np.clip(arm_score, 0.0, 100.0)),
                "wrist_jitter": wrist_jitter,
                "elbow_jitter": elbow_jitter,
                "arm_angle": arm_angle,
                "arm_stability": arm_stability,
                "is_valid": True
            }
            self._is_valid = True

        except (AttributeError, IndexError, TypeError, KeyError) as e:
            self._handle_invalid_input()

    def _handle_invalid_input(self) -> None:
        """处理无效输入，保持状态一致"""
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

    def _calculate_arm_stability(self, wrist_jitter: float, elbow_jitter: float) -> float:
        """
        计算手臂稳定性

        参数:
            wrist_jitter: 手腕抖动
            elbow_jitter: 肘关节抖动

        返回:
            稳定性评分 (0-1)
        """
        avg_jitter = (wrist_jitter + elbow_jitter) / 2.0
        # 抖动越小，稳定性越高
        stability = max(0.0, 1.0 - avg_jitter * self.config['jitter_multiplier'] / 100.0)
        return float(stability)

    def _compute_arm_score(self, wrist_jitter: float, elbow_jitter: float, 
                          arm_angle: float, arm_stability: float) -> float:
        """
        计算手臂评分

        参数:
            wrist_jitter: 手腕抖动
            elbow_jitter: 肘关节抖动
            arm_angle: 手臂角度
            arm_stability: 手臂稳定性

        返回:
            手臂评分 (0-100)
        """
        # 基础分
        base_score = 70.0

        # 抖动惩罚
        avg_jitter = (wrist_jitter + elbow_jitter) / 2.0
        jitter_penalty = avg_jitter * self.config['jitter_multiplier']

        # 角度评分（使用配置中的角度参数）
        if self.config['ideal_angle_min'] <= arm_angle <= self.config['ideal_angle_max']:
            angle_bonus = 10.0
        elif self.config['acceptable_angle_min'] <= arm_angle <= self.config['acceptable_angle_max']:
            angle_bonus = 5.0
        else:
            angle_bonus = 0.0

        # 稳定性奖励
        stability_bonus = arm_stability * self.config['stability_bonus']

        # 综合评分
        score = base_score - jitter_penalty + angle_bonus + stability_bonus
        return float(score)

    def get_results(self) -> ArmAnalysisResult:
        """
        获取分析结果

        返回:
            包含分析结果的字典
        """
        return self.results.copy()

    def is_valid(self) -> bool:
        """返回当前分析状态是否基于有效输入"""
        return self._is_valid
