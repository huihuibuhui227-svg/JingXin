import time
import collections
import copy
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
        self.blink_times = []
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

        # === 眨眼检测 ===
        ear = current_au.avg_ear
        current_time = time.time()
        is_blink = ear < self.EAR_THRESHOLD

        if is_blink and (current_time - self.last_blink_time) > 0.3:
            self.blink_times.append(current_time)
            self.last_blink_time = current_time

        one_minute_ago = current_time - 60
        blink_count = sum(1 for t in self.blink_times if t > one_minute_ago)
        eye_closed_sec = self.eye_closed_duration
        if ear < 0.18:
            self.eye_closed_duration += 1 / self.fps
        else:
            self.eye_closed_duration = 0

        # === 关键修复：深拷贝 + 强制初始化历史帧 ===
        au_for_history = copy.deepcopy(current_au)
        au_for_history.is_blink = is_blink
        au_for_history.blink_rate_per_min = blink_count
        au_for_history.eye_closed_sec = self.eye_closed_duration

        # 强制确保至少2帧（避免时间序列计算被跳过）
        if len(self.au_history) == 0:
            # 创建一个“零帧”作为历史起点
            zero_au = copy.deepcopy(au_for_history)
            for attr in [
                'au1_inner_brow_raise', 'au2_outer_brow_raise', 'au4_frown',
                'au6_cheek_raise', 'au7_eye_squeeze', 'au9_nose_wrinkle',
                'au10_upper_lip_raise', 'au12_smile', 'au14_dimpler',
                'au15_mouth_down', 'au20_lip_stretcher', 'au23_lip_compression',
                'au25_mouth_open', 'au26_jaw_drop', 'avg_ear'
            ]:
                if hasattr(zero_au, attr):
                    setattr(zero_au, attr, 0.0)
            zero_au.head_yaw = 0.0
            zero_au.head_pitch = 0.0
            zero_au.symmetry_score = 1.0
            zero_au.blink_rate_per_min = 0
            zero_au.eye_closed_sec = 0
            zero_au.is_blink = False
            self.au_history.append(zero_au)

        self.au_history.append(au_for_history)

        # === 时间序列统计（增强敏感度）===
        temporal_stats_dict = {}
        if len(self.au_history) >= 2:
            au_fields = [
                name for name, typ in current_au.__annotations__.items()
                if typ in (float, int) and name != 'landmarks'
            ]
            for field_name in au_fields:
                try:
                    series = [getattr(frame, field_name, 0.0) for frame in self.au_history]
                    y = np.array(series, dtype=np.float32)
                    x = np.arange(len(y))

                    # 趋势（放大100倍）
                    if np.all(y == y[0]):
                        trend = 0.0
                    else:
                        coeffs = np.polyfit(x, y, 1)
                        trend = float(coeffs[0]) * 100

                    # 波动性（微小值用 std*10，正常值用变异系数）
                    mean_val = np.mean(y)
                    std_val = np.std(y)
                    if mean_val < 0.01:
                        volatility = std_val * 10.0
                    else:
                        volatility = std_val / mean_val

                    # 变化率（相对变化 ×2）
                    if len(y) >= 2:
                        prev_mean = np.mean(y[:-1])
                        current = y[-1]
                        if prev_mean > 0:
                            change_rate = abs(current - prev_mean) / prev_mean * 2.0
                        else:
                            change_rate = abs(current) * 10.0
                    else:
                        change_rate = 0.0

                    # 限制范围
                    trend = max(-1.0, min(1.0, trend))
                    volatility = min(volatility, 1.0)
                    change_rate = min(change_rate, 1.0)

                    temporal_stats_dict[f'{field_name}_trend'] = round(trend, 3)
                    temporal_stats_dict[f'{field_name}_volatility'] = round(volatility, 3)
                    temporal_stats_dict[f'{field_name}_change_rate'] = round(change_rate, 3)

                except Exception as e:
                    print(f"Temporal stats error for {field_name}: {e}")
                    continue

        temporal_stats = TemporalStats(data=temporal_stats_dict)

        # === 微表情、情绪、紧张度 ===
        micro_exps = self.micro_detector.detect(current_au)
        emotion_result = self.emotion_engine.infer(current_au, temporal_stats, micro_exps)
        tension_result = self.tension_engine.compute(
            current_au,
            temporal_stats,
            emotion_result.emotion_vector
        )

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