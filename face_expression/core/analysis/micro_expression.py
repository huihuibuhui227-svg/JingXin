from collections import deque
import numpy as np
from ...models.features import AUFeatures, MicroExpressionResult

# 窗长按**秒**定义,不按帧数定义(M2.5 spec §5.3)。
#
# 为什么必须这样:实时路径的 fps 是实测且逐会话可变的。旧的 `maxlen=15` 在 1 fps 下
# 是 15 秒、在 30 fps 下是 0.5 秒 —— 同一个常数两种含义。时间窗在两条路径下语义唯一。
#
# ⚠️ 副作用(账本已记):实时 1 fps 时 1.5 秒窗只有 ~2 帧,于是 `_MIN_FRAMES` 那道门
# 永远过不去 —— 微表情在实时路径**不再产出**。那不是本改动的 bug,而是采集率太低
# 被暴露出来。提高采集率见 spec §8.1 / §10。
_WINDOW_MS = 1500

# 窗内至少要有这么多帧才评估。保持与改动前同量级(旧实现要求 10 帧)。
_MIN_FRAMES = 10

# 校准期:丢掉最前面这么多**毫秒**的帧(数据还没稳)。按时间给,理由同上。
_CALIBRATION_MS = 1500


class MicroExpressionDetector:
    def __init__(self, window_ms: int = _WINDOW_MS):
        self.window_ms = window_ms
        # 值为 (timestamp_ms, au_value);按时间修剪,所以 deque 不设 maxlen。
        self.au_history = {au: deque() for au in [
            'au4_frown', 'au7_eye_squeeze', 'au15_mouth_down'
        ]}
        self._first_ts = None

    def _window(self, au_name: str) -> list:
        """窗内这一路的 AU 取值(按时间升序)。测试直接用,故用这个下划线名。"""
        return [v for _, v in self.au_history[au_name]]

    def _prune(self, timestamp_ms: int) -> None:
        cutoff = timestamp_ms - self.window_ms
        for dq in self.au_history.values():
            while dq and dq[0][0] < cutoff:
                dq.popleft()

    def detect(self, current_au_values: AUFeatures, timestamp_ms: int) -> MicroExpressionResult:
        if self._first_ts is None:
            self._first_ts = timestamp_ms

        au_dict = {k: v for k, v in current_au_values.__dict__.items() if k in self.au_history}

        self._prune(timestamp_ms)
        for au_name in self.au_history:
            if au_name in au_dict:
                self.au_history[au_name].append((timestamp_ms, au_dict[au_name]))

        if timestamp_ms - self._first_ts < _CALIBRATION_MS:
            return MicroExpressionResult(data={})

        micro_exps = {}
        for au_name in self.au_history:
            series = self._window(au_name)
            if len(series) < _MIN_FRAMES:
                continue

            baseline = np.mean(series[:-5])
            std_dev = np.std(series[:-5]) or 0.01
            current_val = series[-1]

            dynamic_threshold = baseline + 1.5 * std_dev
            min_activation = 0.1

            if current_val > dynamic_threshold and current_val > min_activation:
                recent_series = series[-8:]
                duration = sum(1 for v in recent_series if v > dynamic_threshold)
                if 2 <= duration <= 8:
                    micro_exps[au_name] = {
                        'intensity': round(float(current_val), 3),
                        'duration_frames': duration,
                        'onset_frame': len(series) - duration
                    }

        return MicroExpressionResult(data=micro_exps)
