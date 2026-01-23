from collections import deque
import numpy as np
from ...models.features import AUFeatures, MicroExpressionResult

class MicroExpressionDetector:
    def __init__(self, fps=30):
        self.window_size = int(0.5 * fps)
        self.au_history = {au: deque(maxlen=15) for au in [
            'au4_frown', 'au7_eye_squeeze', 'au15_mouth_down'
        ]}
        self.calibration_frames = 0
        self.max_calibration = 10

    def detect(self, current_au_values: AUFeatures) -> MicroExpressionResult:
        au_dict = {k: v for k, v in current_au_values.__dict__.items() if k in self.au_history}
        
        for au_name in self.au_history:
            if au_name in au_dict:
                self.au_history[au_name].append(au_dict[au_name])

        if self.calibration_frames < self.max_calibration:
            self.calibration_frames += 1
            return MicroExpressionResult(data={})

        micro_exps = {}
        for au_name in self.au_history:
            if len(self.au_history[au_name]) < 10:
                continue

            series = list(self.au_history[au_name])
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