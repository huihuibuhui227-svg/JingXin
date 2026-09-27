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
        mouth_top = np.array(lm[0])
        mouth_bottom = np.array(lm[17])

        current_mouth_width = dist.euclidean(mouth_left, mouth_right)
        current_jaw_drop = abs(np.array(lm[152])[1] - upper_lip[1])
        current_upper_lip_y = np.array(lm[164])[1]
        current_mouth_height = dist.euclidean(mouth_top, mouth_bottom)
        current_mouth_corner_y = (np.array(lm[61])[1] + np.array(lm[291])[1]) / 2

        if self.calibration_frames < self.max_calibration:
            if self.rest_mouth_width is None:
                self.rest_mouth_width = current_mouth_width
                self.rest_jaw_drop = current_jaw_drop
                self.rest_upper_lip_y = current_upper_lip_y
                self.rest_mouth_height = current_mouth_height
                self.rest_mouth_corner_y = current_mouth_corner_y
            else:
                alpha = 1.0 / (self.calibration_frames + 1)
                self.rest_mouth_width = (1 - alpha) * self.rest_mouth_width + alpha * current_mouth_width
                self.rest_jaw_drop = (1 - alpha) * self.rest_jaw_drop + alpha * current_jaw_drop
                self.rest_upper_lip_y = (1 - alpha) * self.rest_upper_lip_y + alpha * current_upper_lip_y
                self.rest_mouth_height = (1 - alpha) * self.rest_mouth_height + alpha * current_mouth_height
                self.rest_mouth_corner_y = (1 - alpha) * self.rest_mouth_corner_y + alpha * current_mouth_corner_y
            self.calibration_frames += 1

        rest_mouth_width = self.rest_mouth_width or current_mouth_width
        rest_jaw_drop = self.rest_jaw_drop or current_jaw_drop
        rest_upper_lip_y = self.rest_upper_lip_y or current_upper_lip_y
        rest_mouth_height = self.rest_mouth_height or current_mouth_height
        rest_mouth_corner_y = self.rest_mouth_corner_y or current_mouth_corner_y

        au12_smile = max(0.0, (current_mouth_width - rest_mouth_width) / (rest_mouth_width + 1e-6))
        au25_mouth_open = max(0.0, (current_mouth_height - rest_mouth_height) / (rest_mouth_height + 1e-6))
        au25_mouth_open = min(au25_mouth_open, 1.0)
        # `au23_lip_compression` = **唇红厚度 ÷ face_height**(L0 行 `au23_lip_compression`)。
        # ⚠️ 改前是 `1 − 当前唇厚/静息唇厚`,而"当前唇厚"取的是 `lm[13] − lm[14]`
        # —— 那是**口内开合度**,与 `au25_mouth_open` 语义重叠(实测会话内相关 −0.50,
        # 两者互为镜像)。改后只取**上唇两点** `lm[0]`(唇上缘)→ `lm[13]`(唇缝):
        # 张口时 `lm[14]` 下沉、这两点不动 ⟹ 本列与 au25 解耦。
        # 同时它不再依赖静息值 ⟹ 第一帧就有定义(改前第一帧恒为 0.0,一个假值)。
        au23_lip_compression = dist.euclidean(mouth_top, upper_lip) / face_height
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
    def extract(lm, eye_distance):
        """`au4_frown` = 双眉**内侧**点距 ÷ 眼距(L0 行 `au4_frown`)。

        ⚠️ 改前读的是 `lm[276]`(眉尾)与 `lm[33]`(眼内角),分母是**面宽**,
        而且返回 `1 − 比值` —— 实测 86~92% 的帧触地板(spec §1 的症状),
        报告里的「皱眉」几乎是个常数。
        下标依据:眉内端 = `landmarks.py` 的 `LEFT_EYEBROW_UPPER[0]`(107)与
        `RIGHT_EYEBROW_UPPER[0]`(336);眼距 = `EYE_CORNER_LEFT`(130)/`EYE_CORNER_RIGHT`(359)。
        **是比值本身,不取 `1 −`**、也不夹到 [0,1] —— 定义要的是那个比值,
        夹一刀就是把测量值改成另一个量(「皱眉」的事件门限留给 L1)。
        """
        brow_inner_left = np.array(lm[107])
        brow_inner_right = np.array(lm[336])
        return {'au4_frown': float(dist.euclidean(brow_inner_left, brow_inner_right) / eye_distance)}


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
        """`au9_nose_wrinkle` = 1 − 鼻尖到两鼻翼均距 ÷ face_width(L0 行 `au9_nose_wrinkle`)。

        ⚠️ 改前鼻翼读的是 `lm[234]`/`lm[455]` —— 那是**脸颊轮廓点**,不是鼻翼
        (§4.1:166 依据栏里的 234/455 描述的就是这个 bug)。改用 `landmarks.py`
        声明的 `NOSE_WING_LEFT`(129)/`NOSE_WING_RIGHT`(358)。
        **分母照旧 ÷ face_width**(`video_pipeline` 的 `dist(lm[234], lm[455])`):
        L0 行 basis 写明"沿用现行分母,只换下标"。
        """
        nose_tip = np.array(lm[1])
        nose_wing_left = np.array(lm[129])
        nose_wing_right = np.array(lm[358])
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
    def extract(lm, eye_distance):
        """`au6_cheek_raise` = 1 − (颊-眼竖直距两段之和) ÷ (2 × **眼距**)。

        ⚠️ 改前分母是 `face_height`。颊抬起是**竖直位移**,除竖直尺度本就同向;
        但 `face_height`(鼻尖→下巴)会随**低头/仰头**大幅变化,于是该列的会话内方差
        随 yaw/pitch 单向漂移 —— 量到的是头姿,不是颊。改除**眼距**:眼球到眼眶的距离
        不随头姿变,且它本身就是"脸的尺度"里最稳的那个量。

        配对与左右反转见 L0 行:颊 205 ↔ 145、颊 425 ↔ 374 在解剖上是对的;
        `landmarks.py` 把 145 声明在 `RIGHT_EYE` 与代码里的命名相反,但本列**取两段均值**,
        该反转不影响取值 —— 下一个人不要按名字去反推这是哪只眼。
        """
        cheek_top_left = np.array(lm[205])
        cheek_top_right = np.array(lm[425])
        eye_bottom_left = np.array(lm[145])
        eye_bottom_right = np.array(lm[374])
        left_cheek_lift = dist.euclidean(cheek_top_left, eye_bottom_left)
        right_cheek_lift = dist.euclidean(cheek_top_right, eye_bottom_right)
        au6_cheek_raise = 1.0 - (left_cheek_lift + right_cheek_lift) / (2 * eye_distance)
        au6_cheek_raise = max(0.0, min(au6_cheek_raise, 1.0))
        return {'au6_cheek_raise': au6_cheek_raise}


class SymmetryExtractor:
    @staticmethod
    def extract(lm, face_height):
        """`symmetry_score` = 1 − (|Δ眼中心y| + |Δ嘴角y|) ÷ **face_height**。

        ⚠️ 改前**什么都不除**:分子是归一化图像坐标里的竖直差,差多少就直接减多少,
        于是该列量的是"脸在画面里多大/多斜"—— 取景代理(§4.1:208 依据栏
        「未除人脸尺度 → 取景代理」)。实测会话内 mean 0.9825、std ≈0.012,近常量。
        除 `face_height` 后,人脸整体缩放时分子分母同步变 ⟹ 值不动
        (钉子 `test_symmetry_score_is_invariant_to_face_scale`)。
        分母取竖直尺度是因为分子是**竖直**差 —— 与 au6 那条"分子分母同向"的理由同源。
        """
        left_eye_center = np.mean([lm[i] for i in [33, 133]], axis=0)
        right_eye_center = np.mean([lm[i] for i in [362, 263]], axis=0)
        mouth_left = np.array(lm[61])
        mouth_right = np.array(lm[291])
        eye_y_diff = abs(left_eye_center[1] - right_eye_center[1])
        mouth_y_diff = abs(mouth_left[1] - mouth_right[1])
        symmetry_score = max(0.0, min(1.0, 1.0 - (eye_y_diff + mouth_y_diff) / face_height))
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

        # ★ 四列 = **眼内相对坐标**(L0 行 iris 四列;§4.1:188):
        #       `(虹膜中心 − 眼中心) ÷ 眼宽`
        # ⚠️ 改前直接吐**归一化图像坐标**(`left_iris_center[0]` 之类)——
        #    那编码的是"脸在画面里的位置",是**取景代理**;`eye_contact` 的封停理由
        #    原文就是「iris 图像坐标到画面中心距离,是取景代理」,而它的 unblock 就是这四列。
        # x、y **同除眼宽**:同一分母才是各向同性,虹膜这个"圆"不会被拉扁。
        # 减眼中心 ⟹ 人脸在画面里平移时四列不变(钉子
        # `test_iris_columns_are_invariant_to_where_the_face_sits_in_frame`)。
        left_eye_center = (np.array(lm[33]) + np.array(lm[133])) / 2.0
        right_eye_center = (np.array(lm[362]) + np.array(lm[263])) / 2.0
        left_eye_width = dist.euclidean(np.array(lm[33]), np.array(lm[133]))
        right_eye_width = dist.euclidean(np.array(lm[362]), np.array(lm[263]))

        left_iris_x = (left_iris_center[0] - left_eye_center[0]) / left_eye_width
        left_iris_y = (left_iris_center[1] - left_eye_center[1]) / left_eye_width
        right_iris_x = (right_iris_center[0] - right_eye_center[0]) / right_eye_width
        right_iris_y = (right_iris_center[1] - right_eye_center[1]) / right_eye_width

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

        # ★ **眼距算一次、传下去**(L0 行 iris 四列的 normalization 与 face_scale 验收①
        #   「不许两处各算一遍同一个 dist」)。au4 与 au6 都要它 —— 各自重算一遍
        #   在数学上等价,但一旦有人只改其中一处(比如换下标),两列就会**静默地**
        #   用两个不同的尺度,而这种不一致不会有任何报错。
        eye_distance = float(dist.euclidean(np.array(landmarks_norm[130]),
                                            np.array(landmarks_norm[359])))

        features.update(EyeFeatureExtractor.extract(landmarks_norm))
        features.update(self.mouth_extractor.extract(landmarks_norm, face_width, face_height))
        features.update(BrowFeatureExtractor.extract(landmarks_norm, eye_distance))
        features.update(BrowRaiserExtractor.extract(landmarks_norm, face_width, face_height))
        features.update(NoseFeatureExtractor.extract(landmarks_norm, face_width, face_height))
        features.update(CheekFeatureExtractor.extract(landmarks_norm, eye_distance))
        features.update(HeadPoseExtractor.extract(landmarks_norm, face_width, face_height))
        features.update(SymmetryExtractor.extract(landmarks_norm, face_height))
        features.update(IrisFeatureExtractor.extract(landmarks_norm, face_width, face_height))

        # ★ `face_scale`:该帧的**面部尺度本身**,作为**协变量**显式输出
        #   (§4.5:272 对这类量的两条路之一:要么按解剖尺度归一,要么显式输出为协变量)。
        #   它**就是**各列当分母用的那个 `face_height` —— 传进来的那个,不是在这里
        #   重算一遍 `dist(lm[1], lm[152])`(验收①:两处各算一遍就会静默分岔)。
        #   ⚠️ 必须显式赋值:下面那条"补充缺失字段(默认0)"的循环会把漏掉的字段填成 0.0,
        #   而 0 是个**看着像测量值**的数(钉子 `test_face_scale_is_not_silently_defaulted_to_zero`)。
        features['face_scale'] = float(face_height)

        # 补充缺失字段（默认0）
        all_fields = {f.name for f in AUFeatures.__dataclass_fields__.values()}
        for field in all_fields:
            if field not in features:
                features[field] = 0.0

        # 可选：保存原始 landmarks
        if self.save_landmarks:
            features['landmarks'] = [pt[0] for pt in landmarks_norm] + [pt[1] for pt in landmarks_norm]

        return AUFeatures(**features)