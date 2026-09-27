
"""
手臂分析模块

提供手腕和手肘特征提取与评估功能
"""

import numpy as np
from collections import deque
from typing import Dict, Any, Optional, Tuple
from gesture_analysis.config import ARM_CONFIG
from gesture_analysis.core.analysis.jitter import windowed_jitter


ArmAnalysisResult = Dict[str, Any]


class ArmAnalyzer:
    """手臂分析器"""

    def __init__(self, arm_id: str = 'left', config: Optional[Dict[str, float]] = None,
                 metric: bool = False):
        """
        初始化手臂分析器

        参数:
            arm_id: 手臂标识 ('left' 或 'right')
            config: 配置字典，如果为None则使用默认配置
            metric: 喂进来的是**米制 3D 坐标**(`pose_world_landmarks`)时为 `True`。
                ⚠️ 它只影响**分母**:米制已是解剖尺度,`*_jitter_world` 明写「**不**除肩宽」
                (l0_columns.json 那 6 行);画面坐标那批要 ÷ 肩宽。两批用的是**同一个类**
                (公式同一套、只换输入空间),所以这个开关必须显式给 —— 让调用方说明
                "我喂的是哪一种坐标",而不是让类去猜。
        """
        if arm_id not in ('left', 'right'):
            raise ValueError("arm_id 必须为 'left' 或 'right'")

        self.arm_id = arm_id
        self.metric = bool(metric)
        self.config = config or ARM_CONFIG.copy()

        # 初始化历史数据(每一条是 `(timestamp_ms, x, y)`:jitter 的分母之一要用窗内真实秒数)
        self.wrist_history = deque(maxlen=int(self.config['history_length']))
        self.elbow_history = deque(maxlen=int(self.config['history_length']))
        # 同长的肩宽窗 —— 只有画面坐标那批拿它当中位数分母(见 `self.metric`)
        self.shoulder_width_history = deque(maxlen=int(self.config['history_length']))

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

    def update(self, landmarks, timestamp_ms: Optional[int] = None,
               shoulder_width: Optional[float] = None) -> None:
        """
        更新手臂数据

        参数:
            landmarks: MediaPipe 姿态关键点列表（需至少包含 33 个点）
                      或包含手腕和手肘关键点的字典
            timestamp_ms: 本帧的**会话相对毫秒时间戳**（jitter 的「÷ 窗内真实秒数」那层
                分母的原料；见 `l0_columns.json` 的 `left/right_wrist_jitter` 行）。
            shoulder_width: 本帧的双肩**归一化图像距离**。**画面坐标**那批 jitter 的
                分母（米制那批不除 —— 米制已是解剖尺度）。缺 ⟹ 那一帧交空。

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

            # 更新历史数据(时间戳与位置一起进窗)
            self.wrist_history.append(
                (timestamp_ms, wrist[0], wrist[1]))
            self.elbow_history.append(
                (timestamp_ms, elbow[0], elbow[1]))
            self.shoulder_width_history.append(shoulder_width)

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

    def _calculate_jitter(self, history: deque) -> Optional[float]:
        """
        计算抖动**率** = 逐轴标准差均值 ÷ 窗内真实秒数(画面坐标那批再 ÷ 窗内肩宽中位数)

        参数:
            history: `(timestamp_ms, x, y)` 的位置历史

        返回:
            抖动率;缺分母/没走过时间 ⟹ `None`(由 logger 落**空**)
        """
        return windowed_jitter(history, self.shoulder_width_history,
                               divide_by_shoulder=not self.metric)

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

    def _calculate_arm_stability(self, wrist_jitter: Optional[float],
                                 elbow_jitter: Optional[float]) -> float:
        """
        计算手臂稳定性

        参数:
            wrist_jitter: 手腕抖动率;`None` = 没测到
            elbow_jitter: 肘关节抖动率;`None` = 没测到

        ⚠️ `None` 的那一路**不进均值**,而不是当 0 用(当 0 = 把"没测到"算成"很稳");
        两路都没有 ⟹ 交 `None`(稳定性这一帧没有依据)。

        返回:
            稳定性评分 (0-1)
        """
        seen = [j for j in (wrist_jitter, elbow_jitter) if j is not None]
        if not seen:
            return None
        avg_jitter = sum(seen) / len(seen)
        # 抖动越小，稳定性越高
        stability = max(0.0, 1.0 - avg_jitter * self.config['jitter_multiplier'] / 100.0)
        return float(stability)

    def _compute_arm_score(self, wrist_jitter: Optional[float], elbow_jitter: Optional[float],
                          arm_angle: float, arm_stability: Optional[float]) -> float:
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

        # 抖动惩罚(`None` 的那一路不算进来 —— 不拿"没测到"去算分)
        seen = [j for j in (wrist_jitter, elbow_jitter) if j is not None]
        jitter_penalty = (sum(seen) / len(seen) * self.config['jitter_multiplier']
                          if seen else 0.0)

        # 角度评分（使用配置中的角度参数）
        if self.config['ideal_angle_min'] <= arm_angle <= self.config['ideal_angle_max']:
            angle_bonus = 10.0
        elif self.config['acceptable_angle_min'] <= arm_angle <= self.config['acceptable_angle_max']:
            angle_bonus = 5.0
        else:
            angle_bonus = 0.0

        # 稳定性奖励(None = 没测到 ⟹ 不给奖励)
        stability_bonus = (arm_stability or 0.0) * self.config['stability_bonus']

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
