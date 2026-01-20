# face_expression/analyzers/micro_expression.py

from collections import deque
import numpy as np

class MicroExpressionDetector:
    def __init__(self, fps=30):
        self.window_size = int(0.5 * fps)  # 0.5秒窗口
        self.au_history = {au: deque(maxlen=15) for au in [
            'au4_frown', 'au7_eye_squeeze', 'au15_mouth_down'
        ]}
        self.calibration_frames = 0
        self.max_calibration = 10

    def detect(self, current_au_values: dict) -> dict:
        # 更新历史
        for au_name in self.au_history:
            if au_name in current_au_values:
                self.au_history[au_name].append(current_au_values[au_name])

        # 校准阶段：前10帧建立基线
        if self.calibration_frames < self.max_calibration:
            self.calibration_frames += 1
            return {}

        micro_exps = {}
        for au_name in self.au_history:
            if len(self.au_history[au_name]) < 10:
                continue

            series = list(self.au_history[au_name])
            baseline = np.mean(series[:-5])  # 前5帧作为基线
            std_dev = np.std(series[:-5]) or 0.01
            current_val = series[-1]

            # 自适应阈值：基线 + 1.5 * 标准差
            dynamic_threshold = baseline + 1.5 * std_dev
            min_activation = 0.1  # 最小激活值

            if current_val > dynamic_threshold and current_val > min_activation:
                # 检测持续时间（最近8帧中有多少超过阈值）
                recent_series = series[-8:]
                duration = sum(1 for v in recent_series if v > dynamic_threshold)
                if 2 <= duration <= 8:  # 2~8帧（0.07~0.27秒）
                    micro_exps[au_name] = {
                        'intensity': round(float(current_val), 3),
                        'duration_frames': duration,
                        'onset_frame': len(series) - duration
                    }

        return micro_exps