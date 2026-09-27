"""
肩部分析模块

提供肩部特征提取和紧张度评估功能
"""

import numpy as np
from collections import deque
from typing import Dict, Any, Optional, Tuple
from gesture_analysis.config import SHOULDER_CONFIG
from gesture_analysis.core.analysis.jitter import window_median, windowed_jitter


ShoulderAnalysisResult = Dict[str, Any]


class ShoulderAnalyzer:
    """肩部分析器"""

    def __init__(self, config: Optional[Dict[str, float]] = None, metric: bool = False):
        """
        初始化肩部分析器

        参数:
            config: 配置字典，如果为None则使用默认配置
            metric: 喂进来的是**米制 3D 坐标**(`pose_world_landmarks`)时为 `True`。
                只影响**分母**:米制已是解剖尺度,`*_jitter_world` 明写「**不**除肩宽」
                (l0_columns.json 那 6 行),画面坐标那两列要除。同一个类两处用,所以
                这个开关由调用方显式给。
        """
        self.metric = bool(metric)
        self.config = config or SHOULDER_CONFIG.copy()

        self.shoulder_history = {
            'left': deque(maxlen=int(self.config['history_length'])),
            'right': deque(maxlen=int(self.config['history_length']))
        }
        # 每一条是 `(timestamp_ms, x, y)` —— jitter 的分母之一要用窗内真实秒数
        self.shoulder_width_history = deque(maxlen=int(self.config['history_length']))
        # `shrug_level` 的**基线**原料:双肩平均 y,**全程**(不设上限)。
        # 基线 = 它的中位数(`_calculate_shrug_level` 里有为什么是因果中位数的一整段)
        self.shrug_y_history: deque = deque()

        # 校准状态
        # ⚠️ 旧口径的 `shoulder_baseline_y`(开头 30 帧的指数滑动均值)**已删**:基线
        #    现在是 `shrug_y_history` 的中位数。`baseline_frames_collected` 还在,但只喂
        #    `is_calibrated` 那一列(它的语义 = "这一场已经看够 30 帧了",与基线不再是
        #    一回事)—— 留着一个没人读的基线字段,下一个读代码的人会以为它还在用。
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
        self.baseline_frames_collected = 0
        # 基线窗口**清空**:新会话从零开始攒(基线是"这场会话的中位数",不是跨会话的)。
        # ⚠️ 位置窗口(`shoulder_history`)历来**不**清 —— 那是现行行为,本批不动它。
        self.shrug_y_history.clear()
        self._is_valid = False

    def update(self, landmarks, timestamp_ms: Optional[int] = None,
               shoulder_width: Optional[float] = None) -> None:
        """
        更新肩部数据

        参数:
            landmarks: MediaPipe 姿态关键点列表（需至少包含 33 个点）
            timestamp_ms: 本帧的**会话相对毫秒时间戳**（jitter 的「÷ 窗内真实秒数」那层
                分母的原料；见 `l0_columns.json` 的 `left/right_shoulder_jitter` 行）。
            shoulder_width: 本帧的双肩**归一化图像距离**。两处用得上：
                jitter 的**窗内中位数**分母、`shrug_level` 的现帧分母。
                缺 ⟹ 这一帧那两个量都交空（没有分母就没有那个比值）。

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
            self.shoulder_history['left'].append((timestamp_ms,) + left_shoulder)
            self.shoulder_history['right'].append((timestamp_ms,) + right_shoulder)
            self.shoulder_width_history.append(shoulder_width)

            left_jitter, right_jitter = self._calculate_shoulder_jitter()
            shrug = self._calculate_shrug_level(landmarks, shoulder_width)
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

    def _calculate_shoulder_jitter(self) -> Tuple[Optional[float], Optional[float]]:
        """
        计算肩部抖动**率** = 逐轴标准差均值 ÷ 窗内肩宽中位数 ÷ 窗内真实秒数

        返回:
            (左肩抖动, 右肩抖动);缺分母/没走过时间 ⟹ `None`(由 logger 落**空**)。
            米制那一路(`metric=True`)**不除肩宽** —— 米制已是解剖尺度。
        """
        divide = not self.metric
        return (windowed_jitter(self.shoulder_history['left'],
                                self.shoulder_width_history, divide_by_shoulder=divide),
                windowed_jitter(self.shoulder_history['right'],
                                self.shoulder_width_history, divide_by_shoulder=divide))

    def _calculate_shrug_level(self, landmarks, shoulder_width: Optional[float]) -> Optional[float]:
        """
        计算耸肩程度 = `max(0, 基线 − 双肩平均 y) ÷ 肩宽`,**不截顶**。

        参数:
            landmarks: MediaPipe姿态关键点
            shoulder_width: 本帧的双肩归一化图像距离(分母)

        返回:
            耸肩程度;缺肩宽 ⟹ `None`(没有分母就没有这个比值)

        ★ 两处口径由 `l0_columns.json` 的 `shrug_level` 行定(basis ①②):

        ① **基线 = 双肩平均 y 的全程中位数**,不再是"开头 30 帧的指数滑动均值"。
           旧口径下这一列**成了时间的函数**:基线冻在开头,人只要慢慢沉肩,后半场就
           一路恒 1.0 或恒 0(§4.2 依据栏点名的「时间漂移量」)。中位数只用到"哪一半
           更高",对缓慢漂移不敏感。
           ⚠️ 实时流上"全程中位数"要用到**会话未来** ⟹ 这里取**因果近似**:每次只用
           **已经见过**的帧算中位数(`shrug_y_history`,按帧推进)。这层近似不改本行的
           定义(与「禁止用视频内分位定义**事件**」那条上游规矩无关:本列的分位是
           **基线**,不是事件门限 —— 与 `au12_smile` 的 p10 同一性质)。

        ② **分母 = 肩宽**(归一化图像单位,同坐标系),不再是裸常量
           `max_shrug_diff = 0.1` —— 那个常量量纲上与"肩抬了多少"无关,而且像所有
           画面坐标量一样随取景缩放变(§4.5 规矩 1)。

        不截顶:截顶会把幅度藏起来;饱和交给 L1 的分位映射(与 `au26_jaw_drop` 同一处置)。

        ⚠️ 图像坐标里 **y 越小 = 位置越高**,所以"抬起"= 当前 y 低于基线。
        """
        try:
            left_y = landmarks[11].y
            right_y = landmarks[12].y
            avg_y = (left_y + right_y) / 2.0
        except (AttributeError, IndexError):
            return None

        # 基线原料:本帧的双肩平均 y。`baseline_frames_collected` 只为 `is_calibrated`
        # 那一列继续计数(它的语义是"这一场已经看够 30 帧了",与基线不再是一回事)。
        self.shrug_y_history.append(avg_y)
        if self.baseline_frames_collected < self.config['baseline_frames_needed']:
            self.baseline_frames_collected += 1

        baseline = window_median(self.shrug_y_history)
        if baseline is None or not shoulder_width:
            return None
        if avg_y < baseline:
            return float((baseline - avg_y) / shoulder_width)
        return 0.0

    def _compute_shoulder_score(self, left_jitter: Optional[float],
                                right_jitter: Optional[float],
                                shrug: Optional[float]) -> float:
        """
        计算肩部评分

        参数:
            left_jitter: 左肩抖动率;`None` = 没测到
            right_jitter: 右肩抖动率;`None` = 没测到
            shrug: 耸肩程度;`None` = 没测到

        返回:
            肩部评分 (0-100)

        ⚠️ `None` 的那几项**不进惩罚**(当 0 用 = 拿"没测到"算出一个好分);
        三项都没测到 ⟹ 交基础分 70.0(与"没有任何可罚的东西"同形,而不是编一个
        更高/更低的分)。分数列本身已被报告层封停(`evidence_gate.QUARANTINE`),
        口径另批处置 —— 这里只保证不会因为 `None` 崩掉。
        """
        seen = [j for j in (left_jitter, right_jitter) if j is not None]
        jitter_penalty = (sum(seen) / len(seen) * self.config['jitter_multiplier']
                          if seen else 0.0)
        shrug_penalty = (shrug or 0.0) * self.config['shrug_penalty']
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