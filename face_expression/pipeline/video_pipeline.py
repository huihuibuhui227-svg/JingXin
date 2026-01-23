import time
import collections
import numpy as np
from scipy.spatial import distance as dist

from ..core.feature_extraction.au_calculator import AUFeatureCalculator
from ..core.analysis.micro_expression import MicroExpressionDetector
from ..core.analysis.tension_engine import TensionEngine
from ..core.analysis.emotion_engine import EmotionEngine
from ..models.features import TemporalStats
from ..models.results import AnalysisFrameResult

class VideoPipeline:
    def __init__(self, fps=30, session_id="default", save_landmarks=False):
        self.fps = fps
        self.session_id = session_id
        self.blink_times = []  # 记录眨眼时间戳
        self.eye_closed_duration = 0
        self.last_blink_time = 0
        self.EAR_THRESHOLD = 0.21

        self._face_mesh = None
        self.feature_calculator = AUFeatureCalculator(save_landmarks=save_landmarks)
        self.micro_detector = MicroExpressionDetector(fps=fps)
        self.tension_engine = TensionEngine()
        self.emotion_engine = EmotionEngine()
        self.au_history = collections.deque(maxlen=int(3 * fps))

    @property
    def face_mesh(self):
        if self._face_mesh is None:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.8,
                min_tracking_confidence=0.8
            )
        return self._face_mesh

    def process_frame(self, image_rgb):
        results = self.face_mesh.process(image_rgb)
        h, w = image_rgb.shape[:2]

        if not results.multi_face_landmarks:
            return None, None, {"emotion": "no_face"}

        lm = results.multi_face_landmarks[0].landmark
        landmarks_norm = [(pt.x, pt.y) for pt in lm]

        nose_tip = np.array(landmarks_norm[1])
        chin = np.array(landmarks_norm[152])
        cheek_left = np.array(landmarks_norm[234])
        cheek_right = np.array(landmarks_norm[455])
        face_height = dist.euclidean(nose_tip, chin)
        face_width = dist.euclidean(cheek_left, cheek_right)
        if face_height < 1e-5 or face_width < 1e-5:
            face_height = face_width = 1.0

        current_au = self.feature_calculator.calculate(landmarks_norm, face_width, face_height)

        # === 优化眨眼检测 ===
        ear = current_au.avg_ear
        current_time = time.time()
        is_blink = ear < self.EAR_THRESHOLD
        current_au.is_blink = is_blink

        if is_blink and (current_time - self.last_blink_time) > 0.3:
            self.blink_times.append(current_time)
            self.last_blink_time = current_time

        # 计算过去 1 分钟内的眨眼次数
        one_minute_ago = current_time - 60
        blink_count = sum(1 for t in self.blink_times if t > one_minute_ago)
        current_au.blink_rate_per_min = blink_count

        if ear < 0.18:
            self.eye_closed_duration += 1 / self.fps
        else:
            self.eye_closed_duration = 0
        current_au.eye_closed_sec = self.eye_closed_duration

        # === 时间序列统计 ===
        self.au_history.append(current_au)
        temporal_stats_dict = {}
        for field in current_au.__dataclass_fields__:
            au_name = field
            series = [getattr(frame, au_name) for frame in self.au_history]
            if len(series) >= 2:
                trend = np.polyfit(range(len(series)), series, 1)[0]
                volatility = np.std(series)
                change_rate = current_au.__getattribute__(au_name) - series[-1]
                temporal_stats_dict[f'{au_name}_trend'] = round(float(trend), 3)
                temporal_stats_dict[f'{au_name}_volatility'] = round(float(volatility), 3)
                temporal_stats_dict[f'{au_name}_change_rate'] = round(float(change_rate), 3)

        temporal_stats = TemporalStats(data=temporal_stats_dict)

        micro_exps = self.micro_detector.detect(current_au)
        emotion_result = self.emotion_engine.infer(current_au, temporal_stats, micro_exps)
        tension_result = self.tension_engine.compute(current_au, temporal_stats, emotion_result.emotion_vector)

        focus_score = self._calculate_focus_score(current_au)

        result = AnalysisFrameResult(
            session_id=self.session_id,
            timestamp=current_time,
            focus_score=round(float(focus_score), 2),
            au_features=current_au,
            temporal_stats=temporal_stats,
            micro_expressions=micro_exps,
            emotion_result=emotion_result,
            tension_result=tension_result
        )

        return result, results, result.to_dict()

    def _calculate_focus_score(self, au_features):
        yaw = au_features.head_yaw
        blink_rate = au_features.blink_rate_per_min
        if abs(yaw) < 0.03 and blink_rate < 30:
            return 0.8
        elif abs(yaw) > 0.08:
            return 0.3
        return 0.5