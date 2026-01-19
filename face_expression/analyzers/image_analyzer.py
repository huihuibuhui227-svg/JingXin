import cv2
import numpy as np
from scipy.spatial import distance as dist
import time
from .landmarks import *
from ..inference.emotion_infer import infer_emotion_from_au


def eye_aspect_ratio(eye_pts):
    """计算眼睛纵横比(EAR)"""
    A = dist.euclidean(eye_pts[1], eye_pts[5])
    B = dist.euclidean(eye_pts[2], eye_pts[4])
    C = dist.euclidean(eye_pts[0], eye_pts[3])
    ear = (A + B) / (2.0 * C)
    return ear


class StaticFaceAnalyzer:
    """静态图片面部AU分析器"""

    def __init__(self):
        # 静态模式下无需帧率缓冲区
        self.EAR_THRESHOLD = 0.18  # 可根据实际调整

        # ====== 延迟初始化 MediaPipe ======
        self._face_mesh = None

    @property
    def face_mesh(self):
        """延迟初始化 MediaPipe FaceMesh"""
        if self._face_mesh is None:
            try:
                import mediapipe as mp
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.8,
                    min_tracking_confidence=0.8
                )
            except ImportError:
                raise ImportError(
                    "MediaPipe 未安装！请运行：pip install mediapipe==0.10.21\n"
                    "如需使用面部分析功能，请确保已正确安装 MediaPipe。"
                )
        return self._face_mesh

    def infer_emotion(self, features):
        """推断情绪状态"""
        # 使用共享的情绪推断模块
        return infer_emotion_from_au(features)

    def analyze_image(self, image_path):
        """分析单张图片"""
        # 读取图像
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"❌ 错误：无法加载图片 '{image_path}'")
            return None

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        results = self.face_mesh.process(image_rgb)  # 使用属性访问，触发延迟初始化
        if not results.multi_face_landmarks:
            print("❌ 未检测到人脸")
            return None

        lm = results.multi_face_landmarks[0].landmark
        landmarks_norm = [(pt.x, pt.y) for pt in lm]

        # === 基础特征 ===
        left_eye = np.array([landmarks_norm[i] for i in LEFT_EYE])
        right_eye = np.array([landmarks_norm[i] for i in RIGHT_EYE])
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # 眨眼检测（静态图不计算频率，只判断是否闭眼）
        is_blink = ear < self.EAR_THRESHOLD
        blink_status = "闭眼" if is_blink else "睁眼"

        left_eye_center = np.mean([landmarks_norm[i] for i in [33, 133]], axis=0)
        right_eye_center = np.mean([landmarks_norm[i] for i in [362, 263]], axis=0)
        mouth_center = np.mean([landmarks_norm[MOUTH_CORNER_LEFT], landmarks_norm[MOUTH_CORNER_RIGHT]], axis=0)
        face_center = (left_eye_center + right_eye_center + mouth_center) / 3.0
        nose = np.array(landmarks_norm[NOSE_TIP])
        yaw_offset = nose[0] - face_center[0]

        mouth_width = dist.euclidean(landmarks_norm[MOUTH_CORNER_LEFT], landmarks_norm[MOUTH_CORNER_RIGHT])
        face_height = dist.euclidean(landmarks_norm[NOSE_TIP], landmarks_norm[CHIN])
        smile_ratio = mouth_width / face_height if face_height > 0 else 0

        # === AU 特征计算 ===
        brow_left = np.array(landmarks_norm[BROW_CENTER_LEFT])
        brow_right = np.array(landmarks_norm[BROW_CENTER_RIGHT])
        brow_distance = dist.euclidean(brow_left, brow_right)
        face_width = dist.euclidean(landmarks_norm[MOUTH_CORNER_LEFT], landmarks_norm[MOUTH_CORNER_RIGHT])
        au4_frown = 1.0 - (brow_distance / face_width)

        left_brow_top = np.mean([landmarks_norm[i][1] for i in LEFT_EYEBROW_UPPER])
        left_eye_top = np.mean([landmarks_norm[i][1] for i in LEFT_EYE_UPPER])
        right_brow_top = np.mean([landmarks_norm[i][1] for i in RIGHT_EYEBROW_UPPER])
        right_eye_top = np.mean([landmarks_norm[i][1] for i in RIGHT_EYE_UPPER])
        brow_eye_dist_left = left_eye_top - left_brow_top
        brow_eye_dist_right = right_eye_top - right_brow_top
        au12_eyebrow_raise = (brow_eye_dist_left + brow_eye_dist_right) / (2 * face_height)

        mouth_center_y = (landmarks_norm[MOUTH_CENTER_TOP][1] + landmarks_norm[MOUTH_CENTER_BOTTOM][1]) / 2
        mouth_corner_left_y = landmarks_norm[MOUTH_CORNER_LEFT][1]
        mouth_corner_right_y = landmarks_norm[MOUTH_CORNER_RIGHT][1]
        au15_mouth_down = ((mouth_corner_left_y - mouth_center_y) + (mouth_corner_right_y - mouth_center_y)) / 2

        au12_smile = mouth_width / dist.euclidean(landmarks_norm[EYE_CORNER_LEFT], landmarks_norm[EYE_CORNER_RIGHT])

        nose_tip = np.array(landmarks_norm[NOSE_TIP])
        nose_wing_left = np.array(landmarks_norm[NOSE_WING_LEFT])
        nose_wing_right = np.array(landmarks_norm[NOSE_WING_RIGHT])
        au9_left = dist.euclidean(nose_tip, nose_wing_left)
        au9_right = dist.euclidean(nose_tip, nose_wing_right)
        au9_nose_wrinkle = 1.0 - (au9_left + au9_right) / (2 * face_width)

        lip_top_y = landmarks_norm[LIP_TOP][1]
        lip_bottom_y = landmarks_norm[LIP_BOTTOM][1]
        au25_mouth_open = (lip_bottom_y - lip_top_y) / face_height

        focus_score = 0.5
        if abs(yaw_offset) < 0.03:
            if smile_ratio > 0.35:
                focus_score = 0.8
            else:
                focus_score = 0.7
        elif abs(yaw_offset) > 0.08:
            focus_score = 0.3

        # 静态图不计算闭眼持续时间
        eye_closed_sec = 0.0

        features = {
            "timestamp": time.time(),
            "focus_score": round(float(focus_score), 2),
            "blink_status": blink_status,
            "au4_frown": round(float(au4_frown), 3),
            "au12_eyebrow_raise": round(float(au12_eyebrow_raise), 3),
            "au12_smile": round(float(au12_smile), 3),
            "au9_nose_wrinkle": round(float(au9_nose_wrinkle), 3),
            "au15_mouth_down": round(float(au15_mouth_down), 3),
            "au25_mouth_open": round(float(au25_mouth_open), 3),
            "eye_closed_sec": round(float(eye_closed_sec), 1)
        }

        emotion = self.infer_emotion(features)
        features["emotion"] = emotion

        return features