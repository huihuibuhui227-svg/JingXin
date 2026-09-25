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
    def __init__(self, session_id="default", save_landmarks=False, detector=None):
        # `fps` 参数已删除(M2.5 spec §5.1):它以前同时是「元数据」和「计时依据」,
        # 而那个计时依据实测错了 30 倍(§3.1)。现在时间只剩一个来源 —— 入参 timestamp_ms。
        # 元信息里仍报 fps,但那是**滑动实测值**(见 `measured_fps()`)。
        self.session_id = session_id
        self.blink_times = []          # 单位:秒(会话相对),不是挂钟
        self.eye_closed_duration = 0.0
        self.last_blink_time = None
        self.history_last_ms = None
        self.EAR_THRESHOLD = 0.21

        # 历史窗按**秒**定,不按帧数定(旧的是 `int(3 * fps)` = 90 帧)。
        # 实时 1 fps 下 90 帧是 90 秒,与「最近 3 秒」差了 30 倍(spec §3.3)。
        self.history_window_ms = 3000
        self.n_submitted = 0
        self.first_ts = None
        self.last_ts = None

        # 探测器**按会话**注入(不是惰性属性,也不是模块级单例):tasks 的 VIDEO 模式
        # 把跟踪状态挂在实例上,单例会让并发会话互相污染跟踪(spec §3.5 / D3)。
        # 缺省自己造一个真的;测试注入假件。
        if detector is None:
            from .detector import FaceDetector
            from ..config import FACE_MODEL
            detector = FaceDetector(FACE_MODEL)
        self.detector = detector

        self.feature_calculator = AUFeatureCalculator(save_landmarks=save_landmarks)
        # M2.5 Task 3:窗长改按**秒**定之后,这个探测器不再需要 fps(spec §5.3)。
        self.micro_detector = MicroExpressionDetector()
        self.tension_engine = TensionEngine()
        self.emotion_engine = EmotionEngine()
        self.au_history = collections.deque()      # 按时间修剪,不设 maxlen

    def process_frame(self, image_rgb, timestamp_ms: int):
        if self.first_ts is None:
            self.first_ts = timestamp_ms
        self.last_ts = timestamp_ms
        self.n_submitted += 1

        h, w = image_rgb.shape[:2]
        landmarks_norm = self.detector.detect(image_rgb, timestamp_ms)

        if not landmarks_norm:
            # ★ 审查 F2:没脸的帧也要推进「上一帧时刻」。否则 `_update_blink_state` 里那个
            # `timestamp_ms - history_last_ms` 会跨过整段无人脸的时间,把它算成**一个闭眼帧**
            # —— 实测:有脸闭眼帧 + 10 秒无脸 + 有脸闭眼帧 → eye_closed_sec = 10.0 秒。
            self.history_last_ms = timestamp_ms
            # ⚠️ 这个 dict **必须**带 timestamp:face 的 logger 在缺这个键时会用
            # `datetime.now().timestamp()` 编一个墙钟(`face_expression/utils/logger.py:90-91`)。
            # M2.5 的 E2E 实测抓到的:5 帧里有 1 帧没检出脸,那一行的 timestamp 于是成了
            # 1.790318e+09(绝对 epoch 秒),而其余四行是 0.00 / 0.09 / 0.15 / 0.19 ——
            # 整列时间戳被混进了第二个时间源,正是 spec §5.1 要根除的东西。
            return None, None, {"emotion": "no_face", "timestamp": timestamp_ms / 1000.0}

        nose_tip = np.array(landmarks_norm[1])
        chin = np.array(landmarks_norm[152])
        cheek_left = np.array(landmarks_norm[234])
        cheek_right = np.array(landmarks_norm[455])
        face_height = dist.euclidean(nose_tip, chin)
        face_width = dist.euclidean(cheek_left, cheek_right)
        if face_height < 1e-5 or face_width < 1e-5:
            face_height = face_width = 1.0

        current_au = self.feature_calculator.calculate(landmarks_norm, face_width, face_height)

        # === 眨眼检测(全部按真实时间,不再碰挂钟 —— M2.5 spec §3.3)===
        ear = current_au.avg_ear
        is_blink, blink_count, eye_closed_sec = self._update_blink_state(ear, timestamp_ms)

        # === 关键修复：深拷贝 + 强制初始化历史帧 ===
        au_for_history = copy.deepcopy(current_au)
        au_for_history.is_blink = is_blink
        au_for_history.blink_rate_per_min = blink_count
        au_for_history.eye_closed_sec = eye_closed_sec
        # 历史修剪要靠它,所以挂一个不在 `__annotations__` 里的属性 ——
        # 于是它不会被当成特征字段卷进时序统计(下面那个列表按 `__annotations__` 取)。
        au_for_history.timestamp_ms = timestamp_ms

        # ★ M2.5 修复:序列化用的是 `current_au`,而这三个字段以前只写在深拷贝上
        # → CSV 里 is_blink 三列恒 0(spec §3.4)。
        current_au.is_blink = is_blink
        current_au.blink_rate_per_min = blink_count
        current_au.eye_closed_sec = eye_closed_sec

        # 历史窗按时间修剪(旧的是 `deque(maxlen=90)` 那种帧数上限)
        cutoff = timestamp_ms - self.history_window_ms
        while self.au_history and getattr(self.au_history[0], "timestamp_ms", 0) < cutoff:
            self.au_history.popleft()

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
        micro_exps = self.micro_detector.detect(current_au, timestamp_ms)
        emotion_result = self.emotion_engine.infer(current_au, temporal_stats, micro_exps)
        tension_result = self.tension_engine.compute(
            current_au,
            temporal_stats,
            emotion_result.emotion_vector
        )

        focus_score = self._calculate_focus_score(current_au)

        result = AnalysisFrameResult(
            session_id=self.session_id,
            # 单位仍是**秒**,与旧列同量纲,只是基准从挂钟改成会话相对时间。
            # 这是 M2.5 唯一改变 timestamp 列语义的地方,下游影响已核(spec §3.6)。
            timestamp=timestamp_ms / 1000.0,
            focus_score=round(float(focus_score), 2),
            au_features=current_au,
            temporal_stats=temporal_stats,
            micro_expressions=micro_exps,
            emotion_result=emotion_result,
            tension_result=tension_result
        )

        # 第二槽原是 mediapipe 的原始 results 对象(`mp.solutions` 那代);换成 tasks 之后
        # 探测器交出的就是摊平后的 landmarks,所以这里跟着换成 `landmarks_norm` —— 沿用
        # 原名 `results` 会 NameError。真值语义(有真值 ⟺ 本帧检出脸)与迁移前一致。
        return result, landmarks_norm, result.to_dict()

    def reset(self):
        """`/session/{sid}/reset` 调:帧计数一起归零,否则下一帧时间戳回退(spec §6.4)。"""
        self.detector.reset()

    def close(self):
        """TTL 回收时调:释放探测器的 native 句柄(spec §6.3)。"""
        self.detector.close()

    def measured_fps(self) -> float:
        """到当前为止的**滑动实测**帧率 —— 元信息用,不是特征列(spec §5.5)。

        旧的 `self.fps` 是客户端申报的 30,而实发 1 帧/秒。现在报的是量出来的。
        """
        if self.first_ts is None or self.last_ts is None:
            return 0.0
        elapsed_sec = (self.last_ts - self.first_ts) / 1000.0
        if elapsed_sec <= 0:
            return 0.0
        return round(self.n_submitted / elapsed_sec, 3)

    def _update_blink_state(self, ear: float, timestamp_ms: int):
        """眨眼/闭眼记账。全部按 `timestamp_ms` 算 —— 不再碰挂钟。

        为什么必须抽出来:旧代码把这件事和 `time.time()` 缠在 `process_frame` 里,
        离线批处理下那个挂钟是**处理时间**(实测中位是真时长的 2.32 倍),
        于是「每分钟眨眼次数」算的是处理时间里的次数(spec §3.3)。
        """
        now_sec = timestamp_ms / 1000.0
        is_blink = ear < self.EAR_THRESHOLD

        if is_blink and (self.last_blink_time is None
                         or (now_sec - self.last_blink_time) > 0.3):
            self.blink_times.append(now_sec)
            self.last_blink_time = now_sec

        one_minute_ago = now_sec - 60.0
        blink_count = sum(1 for t in self.blink_times if t > one_minute_ago)

        if ear < 0.18:
            # 真实间隔 = 与上一帧的时间差;第一帧没有上一帧,记 0
            if self.history_last_ms is not None:
                self.eye_closed_duration += (timestamp_ms - self.history_last_ms) / 1000.0
        else:
            self.eye_closed_duration = 0.0

        self.history_last_ms = timestamp_ms
        return is_blink, blink_count, self.eye_closed_duration

    def _calculate_focus_score(self, au_features):
        yaw = au_features.head_yaw
        blink_rate = au_features.blink_rate_per_min
        if abs(yaw) < 0.03 and blink_rate < 30:
            return 0.8
        elif abs(yaw) > 0.08:
            return 0.3
        return 0.5

    def get_summary(self) -> dict:
        """返回当前会话的聚合统计数据。遍历 au_history 计算均值/波动/情绪分布等。"""
        if len(self.au_history) == 0:
            return {
                "session_id": self.session_id,
                "frame_count": 0,
                "fps": self.measured_fps(),
                "duration_sec": 0,
            }

        au_fields = [
            "au1_inner_brow_raise", "au2_outer_brow_raise", "au4_frown",
            "au6_cheek_raise", "au7_eye_squeeze", "au9_nose_wrinkle",
            "au10_upper_lip_raise", "au12_smile", "au14_dimpler",
            "au15_mouth_down", "au20_lip_stretcher", "au23_lip_compression",
            "au25_mouth_open", "au26_jaw_drop", "avg_ear",
            "head_yaw", "head_pitch", "symmetry_score",
            "blink_rate_per_min", "eye_closed_sec", "gaze_deviation",
        ]

        # AU 特征聚合
        au_summary = {}
        for field in au_fields:
            values = [getattr(f, field, 0.0) for f in self.au_history if hasattr(f, field)]
            if not values:
                continue
            arr = np.array(values, dtype=np.float32)
            trend = 0.0
            if len(arr) >= 2:
                coeffs = np.polyfit(np.arange(len(arr)), arr, 1)
                trend = "increasing" if coeffs[0] > 0.001 else ("decreasing" if coeffs[0] < -0.001 else "stable")
            au_summary[field] = {
                "mean": round(float(np.mean(arr)), 4),
                "std": round(float(np.std(arr)), 4),
                "min": round(float(np.min(arr)), 4),
                "max": round(float(np.max(arr)), 4),
                "trend": trend,
            }

        # 情绪分布
        emotion_counts = {}
        for f in self.au_history:
            if hasattr(f, "dominant_emotion"):
                emotion_counts[f.dominant_emotion] = emotion_counts.get(f.dominant_emotion, 0) + 1
        dominant = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "unknown"

        # 紧张度
        tension_vals = [getattr(f, "tension_score", 0.0) for f in self.au_history if hasattr(f, "tension_score")]
        avg_tension = float(np.mean(tension_vals)) if tension_vals else 0.0
        tension_level = "low" if avg_tension < 0.3 else ("medium" if avg_tension < 0.6 else "high")

        # 注视稳定性
        gaze_vals = [getattr(f, "gaze_deviation", 0.0) for f in self.au_history if hasattr(f, "gaze_deviation")]
        avg_gaze = float(np.mean(gaze_vals)) if gaze_vals else 0.0
        gaze_stability = 1.0 - min(avg_gaze * 10, 1.0) if gaze_vals else 0.8

        # 眨眼统计 —— 时长一律由时间戳导出,不再 `帧数 / fps`(spec §3.3)
        first = getattr(self.au_history[0], "timestamp_ms", 0)
        last = getattr(self.au_history[-1], "timestamp_ms", 0)
        duration_sec = (last - first) / 1000.0
        duration_min = duration_sec / 60.0
        recent_blinks = sum(1 for t in self.blink_times
                            if (last / 1000.0 - t) < 60.0)
        total_blinks = len(self.blink_times)
        avg_blink_rate = (recent_blinks / max(duration_min, 0.01)
                          if duration_min >= 0.1 else recent_blinks / 1.0)

        frame_count = len(self.au_history)

        # 专注度
        focus_vals = [getattr(f, "focus_score", 0.5) for f in self.au_history if hasattr(f, "focus_score")]
        avg_focus = float(np.mean(focus_vals)) if focus_vals else 0.5

        return {
            "session_id": self.session_id,
            "frame_count": frame_count,
            "fps": self.measured_fps(),
            "duration_sec": round(duration_sec, 2),
            "au_features": au_summary,
            "emotion": {
                "dominant": dominant,
                "distribution": emotion_counts,
            },
            "tension": {
                "avg_score": round(avg_tension, 4),
                "avg_level": tension_level,
            },
            "gaze": {
                "avg_deviation": round(avg_gaze, 4),
                "stability": round(gaze_stability, 4),
            },
            "blink": {
                "total_blinks": total_blinks,
                "recent_blinks_per_min": round(avg_blink_rate, 1),
            },
            "focus": {
                "avg_score": round(avg_focus, 4),
            },
        }