"""
手部分析模块

提供手部特征提取和抗压能力评估功能
"""

import numpy as np
from collections import deque
from typing import Dict, Any, Optional, Tuple
from gesture_analysis.config import HAND_CONFIG
from gesture_analysis.core.analysis.jitter import windowed_jitter

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
        # ⚠️ 每一条是 `(timestamp_ms, x, y)` —— 时间戳必须**在窗口里**:
        #    jitter 的分母之一是「窗内真实秒数」(l0_columns.json 的 `*_hand_jitter` 行),
        #    而"帧数 ÷ 假定帧率"等于假设帧率恒定,帧率正是被污染的那个量。
        self.tip_history = {8: deque(maxlen=int(self.config['history_length']))}
        # 同长的肩宽窗:画面坐标 jitter 的**另一层分母**(窗内中位数)。缺肩/无姿态的帧
        # 占位 `None` —— 它们对中位数没有发言权,但也不让窗整体作废。
        self.shoulder_width_history = deque(maxlen=int(self.config['history_length']))

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

    def update(self, landmarks, timestamp_ms: Optional[int] = None,
               shoulder_width: Optional[float] = None) -> None:
        """
        更新手部数据

        参数:
            landmarks: MediaPipe 手部关键点列表（需至少包含 21 个点）
            timestamp_ms: 本帧的**会话相对毫秒时间戳**（`session_clock.SessionClock.stamp_ms()`）。
                jitter 的定义里有「÷ 窗内真实秒数」这一层，它的原料就是这个值 ——
                窗口里必须存得下时间，否则那层分母只能靠"帧数 ÷ 假定帧率"编出来，
                而帧率正是被污染的那个量（l0_columns.json 的 `*_jitter` 行）。
            shoulder_width: 本帧的双肩**归一化图像距离**（`angles.shoulder_width(pose)`）。
                **画面坐标**那批 jitter 的分母（同坐标系，米制那批不除）。缺（本帧没有
                姿态 / 缺一只肩）⟹ `None` ⟹ 那一帧的 jitter 交**空**，不拿未归一化的
                分子顶上 —— 那等于把单位从「肩宽/秒」静默换成「归一化图像单位」。

        异常:
            ValueError: 当 landmarks 无效或长度不足时
        """
        if landmarks is None:
            self._handle_invalid_input()
            return

        if len(landmarks) < 21:
            raise ValueError(f"手部 landmarks 长度不足，期望 >=21，实际: {len(landmarks)}")

        try:
            # 更新食指指尖历史(时间戳与位置一起进窗:分母要用窗内真实秒数)
            self.tip_history[8].append(
                (timestamp_ms, float(landmarks[8].x), float(landmarks[8].y)))
            self.shoulder_width_history.append(shoulder_width)

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

    def _calculate_jitter(self) -> Optional[float]:
        """食指指尖的抖动**率** = 逐轴标准差均值 ÷ 窗内肩宽中位数 ÷ 窗内真实秒数。

        交 `None` = 这一帧没有这个量(窗内没有肩宽 / 没走过时间),由 logger 落**空**。
        口径见 `l0_columns.json` 的 `left/right_hand_jitter` 两行。
        """
        return windowed_jitter(self.tip_history[8], self.shoulder_width_history,
                               divide_by_shoulder=True)

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

    def _calculate_finger_spread(self, landmarks) -> Optional[float]:
        """
        计算手指张开度 = `mean(dist(指尖_i, lm[0])) ÷ dist(lm[0], lm[9])`(掌长)

        分子与现行实现**逐字一致**(指尖 = 4/8/12/16/20、参考点是腕 lm[0])—— L0 行
        basis 明写「一次只改一处,否则『改坏了是分子还是分母』不可回答」。本批加的
        只有那个分母(掌长,同坐标系)。

        为什么要除:归一化图像坐标下的"指尖到腕的距离"同时编码**张开**与**离镜头远近**
        (手靠近镜头一倍,该值也大一倍)。取**掌长**而不是肩宽:手前伸时与肩不在同一
        深度,除肩宽会把深度差算成张开度变化;掌长是同一只手自己的骨性长度、同深度。
        口径见 `l0_columns.json` 的 `left/right_hand_spread` 两行。

        参数:
            landmarks: 手部关键点

        返回:
            张开度比值;掌长取不出来或为 0(退化)⟹ `None`(没有分母就没有这个比值,
            **不补 0** —— 0 的意思是"指尖全贴在腕上",那是个测量结果)
        """
        try:
            palm = np.array([landmarks[0].x, landmarks[0].y])
            palm_length = float(np.linalg.norm(
                np.array([landmarks[9].x, landmarks[9].y]) - palm))
            tips = [4, 8, 12, 16, 20]
            spread = np.mean([
                np.linalg.norm(np.array([landmarks[i].x, landmarks[i].y]) - palm)
                for i in tips
            ])
            if palm_length <= 0:
                return None
            return float(spread / palm_length)
        except (AttributeError, IndexError):
            return None

    def _compute_resilience_score(self, jitter: Optional[float], is_fist: bool,
                                  spread: Optional[float]) -> float:
        """
        计算抗压能力评分

        参数:
            jitter: 抖动**率**(肩宽/秒);`None` = 这一帧没测到
            is_fist: 是否握拳
            spread: 手指张开度比值;`None` = 这一帧没测到

        返回:
            抗压能力评分 (0-100)

        ⚠️ `None` ⟹ **不加那一项惩罚/奖励**,而不是当 0 用:当 0 用等于拿"完全静止"
        去算分(那是把"没测到"说成一个好结果)。分数列本身已被报告层封停
        (`evidence_gate.QUARANTINE` 里的 `hand_score`),口径另批处置 —— 这里只保证
        "没测到"不会伪装成"测到一个好数"。
        """
        jitter_penalty = 0.0 if jitter is None else jitter * self.config['jitter_multiplier']
        jitter_score = max(0.0, 70.0 - jitter_penalty)
        fist_penalty = self.config['fist_penalty'] if is_fist else 0.0
        spread_bonus = 0.0
        if spread is not None and spread > self.config['spread_threshold']:
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