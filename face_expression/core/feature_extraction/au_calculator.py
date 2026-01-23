import numpy as np
from scipy.spatial import distance as dist
from face_expression.models.features import AUFeatures
from .landmarks import *


class EyeFeatureExtractor:
    @staticmethod
    def extract(lm):
        left_eye_pts = [lm[i] for i in [33, 160, 159, 133, 153, 144]]
        right_eye_pts = [lm[i] for i in [362, 387, 386, 263, 380, 373]]
        left_ear = EyeFeatureExtractor._eye_aspect_ratio(left_eye_pts)
        right_ear = EyeFeatureExtractor._eye_aspect_ratio(right_eye_pts)
        avg_ear = (left_ear + right_ear) / 2.0
        au7_eye_squeeze = max(0.0, 1.0 - avg_ear)

        # 标准化
        avg_ear = max(0.0, min(avg_ear, 1.0))
        au7_eye_squeeze = max(0.0, min(au7_eye_squeeze, 1.0))

        return {'avg_ear': avg_ear, 'au7_eye_squeeze': au7_eye_squeeze}

    @staticmethod
    def _eye_aspect_ratio(eye_pts):
        A = dist.euclidean(eye_pts[1], eye_pts[5])
        B = dist.euclidean(eye_pts[2], eye_pts[4])
        C = dist.euclidean(eye_pts[0], eye_pts[3])
        return (A + B) / (2.0 * C)


class MouthFeatureExtractor:
    def __init__(self):
        self.rest_mouth_width = None
        self.rest_lip_thickness = None
        self.rest_jaw_drop = None
        self.rest_upper_lip_y = None
        self.rest_mouth_height = None
        self.rest_mouth_corner_y = None
        self.calibration_frames = 0
        self.max_calibration = 10

    def extract(self, lm, face_width, face_height):
        mouth_left = np.array(lm[61])
        mouth_right = np.array(lm[291])
        upper_lip = np.array(lm[13])
        lower_lip = np.array(lm[14])
        mouth_top = np.array(lm[0])
        mouth_bottom = np.array(lm[17])

        current_mouth_width = dist.euclidean(mouth_left, mouth_right)
        current_lip_thickness = dist.euclidean(upper_lip, lower_lip)
        current_jaw_drop = abs(np.array(lm[152])[1] - upper_lip[1])
        current_upper_lip_y = np.array(lm[164])[1]
        current_mouth_height = dist.euclidean(mouth_top, mouth_bottom)
        current_mouth_corner_y = (np.array(lm[61])[1] + np.array(lm[291])[1]) / 2

        if self.calibration_frames < self.max_calibration:
            if self.rest_mouth_width is None:
                self.rest_mouth_width = current_mouth_width
                self.rest_lip_thickness = current_lip_thickness
                self.rest_jaw_drop = current_jaw_drop
                self.rest_upper_lip_y = current_upper_lip_y
                self.rest_mouth_height = current_mouth_height
                self.rest_mouth_corner_y = current_mouth_corner_y
            else:
                alpha = 1.0 / (self.calibration_frames + 1)
                self.rest_mouth_width = (1 - alpha) * self.rest_mouth_width + alpha * current_mouth_width
                self.rest_lip_thickness = (1 - alpha) * self.rest_lip_thickness + alpha * current_lip_thickness
                self.rest_jaw_drop = (1 - alpha) * self.rest_jaw_drop + alpha * current_jaw_drop
                self.rest_upper_lip_y = (1 - alpha) * self.rest_upper_lip_y + alpha * current_upper_lip_y
                self.rest_mouth_height = (1 - alpha) * self.rest_mouth_height + alpha * current_mouth_height
                self.rest_mouth_corner_y = (1 - alpha) * self.rest_mouth_corner_y + alpha * current_mouth_corner_y
            self.calibration_frames += 1

        rest_mouth_width = self.rest_mouth_width or current_mouth_width
        rest_lip_thickness = self.rest_lip_thickness or current_lip_thickness
        rest_jaw_drop = self.rest_jaw_drop or current_jaw_drop
        rest_upper_lip_y = self.rest_upper_lip_y or current_upper_lip_y
        rest_mouth_height = self.rest_mouth_height or current_mouth_height
        rest_mouth_corner_y = self.rest_mouth_corner_y or current_mouth_corner_y

        au12_smile = max(0.0, (current_mouth_width - rest_mouth_width) / (rest_mouth_width + 1e-6))
        au25_mouth_open = max(0.0, (current_mouth_height - rest_mouth_height) / (rest_mouth_height + 1e-6))
        au25_mouth_open = min(au25_mouth_open, 1.0)
        lip_compression_ratio = current_lip_thickness / (rest_lip_thickness + 1e-6)
        au23_lip_compression = max(0.0, 1.0 - lip_compression_ratio)
        mouth_corner_down = max(0.0, current_mouth_corner_y - rest_mouth_corner_y)
        au15_mouth_down = min(mouth_corner_down / face_height, 1.0)
        au10_upper_lip_raise = max(0.0, (rest_upper_lip_y - current_upper_lip_y) / face_height)
        dimple_left = np.array(lm[202])
        dimple_right = np.array(lm[422])
        left_dimple_depth = dist.euclidean(dimple_left, mouth_left)
        right_dimple_depth = dist.euclidean(dimple_right, mouth_right)
        au14_dimpler = min((left_dimple_depth + right_dimple_depth) / (2 * face_width) * 5.0, 1.0)
        au20_lip_stretcher = max(0.0, (current_mouth_width - rest_mouth_width) / (rest_mouth_width + 1e-6))
        jaw_drop_change = current_jaw_drop - rest_jaw_drop
        au26_jaw_drop = max(0.0, jaw_drop_change / (0.1 * face_height))

        # 标准化
        au12_smile = max(0.0, min(au12_smile, 1.0))
        au25_mouth_open = max(0.0, min(au25_mouth_open, 1.0))
        au23_lip_compression = max(0.0, min(au23_lip_compression, 1.0))
        au15_mouth_down = max(0.0, min(au15_mouth_down, 1.0))
        au10_upper_lip_raise = max(0.0, min(au10_upper_lip_raise, 1.0))
        au14_dimpler = max(0.0, min(au14_dimpler, 1.0))
        au20_lip_stretcher = max(0.0, min(au20_lip_stretcher, 1.0))
        au26_jaw_drop = max(0.0, min(au26_jaw_drop, 1.0))

        return {
            'au12_smile': au12_smile,
            'au25_mouth_open': au25_mouth_open,
            'au23_lip_compression': au23_lip_compression,
            'au15_mouth_down': au15_mouth_down,
            'au10_upper_lip_raise': au10_upper_lip_raise,
            'au14_dimpler': au14_dimpler,
            'au20_lip_stretcher': au20_lip_stretcher,
            'au26_jaw_drop': au26_jaw_drop,
        }


class BrowFeatureExtractor:
    @staticmethod
    def extract(lm, face_width):
        brow_left = np.array(lm[276])
        brow_right = np.array(lm[33])
        brow_distance = dist.euclidean(brow_left, brow_right)
        au4_frown = max(0.0, 1.0 - (brow_distance / face_width))
        au4_frown = max(0.0, min(au4_frown, 1.0))
        return {'au4_frown': au4_frown}


class BrowRaiserExtractor:
    @staticmethod
    def extract(lm, face_width, face_height):
        inner_brow_left = np.array(lm[52])
        inner_brow_right = np.array(lm[55])
        outer_brow_left = np.array(lm[70])
        outer_brow_right = np.array(lm[63])
        brow_center = np.array(lm[168])

        inner_lift_left = brow_center[1] - inner_brow_left[1]
        inner_lift_right = brow_center[1] - inner_brow_right[1]
        outer_lift_left = brow_center[1] - outer_brow_left[1]
        outer_lift_right = brow_center[1] - outer_brow_right[1]

        au1_inner_brow_raise = max(0.0, (inner_lift_left + inner_lift_right) / (2 * face_height))
        au2_outer_brow_raise = max(0.0, (outer_lift_left + outer_lift_right) / (2 * face_height))

        au1_inner_brow_raise = max(0.0, min(au1_inner_brow_raise, 1.0))
        au2_outer_brow_raise = max(0.0, min(au2_outer_brow_raise, 1.0))

        return {
            'au1_inner_brow_raise': au1_inner_brow_raise,
            'au2_outer_brow_raise': au2_outer_brow_raise
        }


class NoseFeatureExtractor:
    @staticmethod
    def extract(lm, face_width, face_height):
        nose_tip = np.array(lm[1])
        nose_wing_left = np.array(lm[234])
        nose_wing_right = np.array(lm[455])
        au9_nose_wrinkle = max(0.0, 1.0 - (
                dist.euclidean(nose_tip, nose_wing_left) +
                dist.euclidean(nose_tip, nose_wing_right)
        ) / (2 * face_width))
        au9_nose_wrinkle = max(0.0, min(au9_nose_wrinkle, 1.0))
        return {'au9_nose_wrinkle': au9_nose_wrinkle}


class HeadPoseExtractor:
    @staticmethod
    def extract(lm, face_width, face_height):
        nose_tip = np.array(lm[1])
        cheek_left = np.array(lm[234])
        cheek_right = np.array(lm[455])
        ear_mid = (cheek_left + cheek_right) / 2
        head_yaw = (nose_tip[0] - ear_mid[0]) / face_width
        chin = np.array(lm[152])
        head_pitch = (nose_tip[1] - chin[1]) / face_height
        return {
            'head_yaw': head_yaw,
            'head_pitch': head_pitch
        }


class CheekFeatureExtractor:
    @staticmethod
    def extract(lm, face_width, face_height):
        cheek_top_left = np.array(lm[205])
        cheek_top_right = np.array(lm[425])
        eye_bottom_left = np.array(lm[145])
        eye_bottom_right = np.array(lm[374])
        left_cheek_lift = dist.euclidean(cheek_top_left, eye_bottom_left)
        right_cheek_lift = dist.euclidean(cheek_top_right, eye_bottom_right)
        au6_cheek_raise = 1.0 - (left_cheek_lift + right_cheek_lift) / (2 * face_height)
        au6_cheek_raise = max(0.0, min(au6_cheek_raise, 1.0))
        return {'au6_cheek_raise': au6_cheek_raise}


class SymmetryExtractor:
    @staticmethod
    def extract(lm):
        left_eye_center = np.mean([lm[i] for i in [33, 133]], axis=0)
        right_eye_center = np.mean([lm[i] for i in [362, 263]], axis=0)
        mouth_left = np.array(lm[61])
        mouth_right = np.array(lm[291])
        eye_y_diff = abs(left_eye_center[1] - right_eye_center[1])
        mouth_y_diff = abs(mouth_left[1] - mouth_right[1])
        symmetry_score = max(0.0, min(1.0, 1.0 - (eye_y_diff + mouth_y_diff)))
        return {'symmetry_score': symmetry_score}


# === 新增：眼球特征提取器 ===
class IrisFeatureExtractor:
    @staticmethod
    def extract(lm, face_width, face_height):
        # 左眼虹膜中心（MediaPipe 索引 468, 469, 470, 471）
        left_iris = np.array([lm[468], lm[469], lm[470], lm[471]])
        right_iris = np.array([lm[473], lm[474], lm[475], lm[476]])

        # 计算平均位置
        left_iris_center = np.mean(left_iris, axis=0)
        right_iris_center = np.mean(right_iris, axis=0)

        # 归一化坐标
        left_iris_x = left_iris_center[0]
        left_iris_y = left_iris_center[1]
        right_iris_x = right_iris_center[0]
        right_iris_y = right_iris_center[1]

        # 计算视线方向（基于瞳孔与鼻尖的相对位置）
        nose_tip = np.array(lm[1])
        eye_mid = (left_iris_center + right_iris_center) / 2

        # 水平偏移：正数表示向右看，负数表示向左看
        gaze_direction_x = (eye_mid[0] - nose_tip[0]) / face_width
        # 垂直偏移：正数表示向上看，负数表示向下看
        gaze_direction_y = (eye_mid[1] - nose_tip[1]) / face_height

        # 视线偏离度（综合水平+垂直偏移）
        gaze_deviation = np.sqrt(gaze_direction_x ** 2 + gaze_direction_y ** 2)

        return {
            'left_iris_x': left_iris_x,
            'left_iris_y': left_iris_y,
            'right_iris_x': right_iris_x,
            'right_iris_y': right_iris_y,
            'gaze_direction_x': gaze_direction_x,
            'gaze_direction_y': gaze_direction_y,
            'gaze_deviation': gaze_deviation
        }


class AUFeatureCalculator:
    def __init__(self, save_landmarks=False):
        self.mouth_extractor = MouthFeatureExtractor()
        self.save_landmarks = save_landmarks

    def calculate(self, landmarks_norm, face_width, face_height) -> AUFeatures:
        features = {}
        features.update(EyeFeatureExtractor.extract(landmarks_norm))
        features.update(self.mouth_extractor.extract(landmarks_norm, face_width, face_height))
        features.update(BrowFeatureExtractor.extract(landmarks_norm, face_width))
        features.update(BrowRaiserExtractor.extract(landmarks_norm, face_width, face_height))
        features.update(NoseFeatureExtractor.extract(landmarks_norm, face_width, face_height))
        features.update(CheekFeatureExtractor.extract(landmarks_norm, face_width, face_height))
        features.update(HeadPoseExtractor.extract(landmarks_norm, face_width, face_height))
        features.update(SymmetryExtractor.extract(landmarks_norm))
        features.update(IrisFeatureExtractor.extract(landmarks_norm, face_width, face_height))

        # 补充缺失字段（默认0）
        all_fields = {f.name for f in AUFeatures.__dataclass_fields__.values()}
        for field in all_fields:
            if field not in features:
                features[field] = 0.0

        # 可选：保存原始 landmarks
        if self.save_landmarks:
            features['landmarks'] = [pt[0] for pt in landmarks_norm] + [pt[1] for pt in landmarks_norm]

        return AUFeatures(**features)