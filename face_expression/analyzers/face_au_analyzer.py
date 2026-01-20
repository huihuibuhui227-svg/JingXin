# face_expression/analyzers/face_au_analyzer.py

import cv2
import time
import collections
import numpy as np
from scipy.spatial import distance as dist

from .feature_extractor import FeatureExtractor
from .micro_expression import MicroExpressionDetector
from .tension_engine import TensionEngine
from .emotion_engine import EmotionEngine


class FaceAUAnalyzer:
    def __init__(self, fps=30, session_id="default"):
        self.fps = fps
        self.session_id = session_id
        self.blink_buffer = collections.deque(maxlen=int(5 * fps))
        self.eye_closed_duration = 0
        self.last_blink_time = 0
        self.EAR_THRESHOLD = 0.21

        self._face_mesh = None
        self.feature_extractor = FeatureExtractor()
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

        # === 标准面部尺寸（使用颧骨）===
        nose_tip = np.array(landmarks_norm[1])
        chin = np.array(landmarks_norm[152])
        cheek_left = np.array(landmarks_norm[234])
        cheek_right = np.array(landmarks_norm[455])
        face_height = dist.euclidean(nose_tip, chin)
        face_width = dist.euclidean(cheek_left, cheek_right)
        if face_height < 1e-5 or face_width < 1e-5:
            face_height = face_width = 1.0

        # === 特征提取 ===
        current_au = self.feature_extractor.extract_all_features(landmarks_norm, face_width, face_height)

        # === 眨眼检测 ===
        ear = current_au.get('avg_ear', 0)
        current_time = time.time()
        is_blink = ear < self.EAR_THRESHOLD
        self.blink_buffer.append(is_blink)

        if is_blink and (current_time - self.last_blink_time) > 0.3:
            self.last_blink_time = current_time

        blink_rate = (sum(self.blink_buffer) / len(self.blink_buffer)) * self.fps * (60 / len(self.blink_buffer))
        current_au['blink_rate_per_min'] = blink_rate

        if ear < 0.18:
            self.eye_closed_duration += 1 / self.fps
        else:
            self.eye_closed_duration = 0
        current_au['eye_closed_sec'] = self.eye_closed_duration

        # === 时间序列统计 ===
        self.au_history.append(current_au)
        temporal_stats = {}
        for au_name in current_au:
            series = [frame[au_name] for frame in self.au_history]
            if len(series) >= 2:
                trend = np.polyfit(range(len(series)), series, 1)[0]
                volatility = np.std(series)
                change_rate = current_au[au_name] - series[-1] if len(series) > 1 else 0
                temporal_stats[f'{au_name}_trend'] = round(float(trend), 3)
                temporal_stats[f'{au_name}_volatility'] = round(float(volatility), 3)
                temporal_stats[f'{au_name}_change_rate'] = round(float(change_rate), 3)

        # === 微表情 & 情绪 & 紧张度 ===
        micro_exps = self.micro_detector.detect(current_au)
        emotion_result = self.emotion_engine.infer(current_au, temporal_stats, micro_exps)
        psychological_signals = self.tension_engine.compute(current_au, temporal_stats, emotion_result["emotion_vector"])

        focus_score = self._calculate_focus_score(current_au)

        features = {
            "session_id": self.session_id,
            "timestamp": current_time,
            "focus_score": round(float(focus_score), 2),
            "symmetry_score": round(float(current_au.get('symmetry_score', 1.0)), 3),
            **{k: round(float(v), 3) for k, v in current_au.items()},
            "psychological_signals": psychological_signals,
            "micro_expressions": micro_exps,
            "temporal_stats": temporal_stats,
            "emotion_vector": emotion_result["emotion_vector"],
            "dominant_emotion": emotion_result["dominant_emotion"],
            "confidence": emotion_result["confidence"]
        }

        return features, results, features

    def _calculate_focus_score(self, au_features):
        yaw = au_features.get('head_yaw', 0)
        blink_rate = au_features.get('blink_rate_per_min', 0)
        if abs(yaw) < 0.03 and blink_rate < 30:
            return 0.8
        elif abs(yaw) > 0.08:
            return 0.3
        return 0.5