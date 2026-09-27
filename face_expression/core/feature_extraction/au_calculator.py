import numpy as np
from collections import deque
from typing import Optional
from scipy.spatial import distance as dist
from face_expression.models.features import AUFeatures
from .landmarks import *


def causal_p10(values) -> Optional[float]:
    """`values` 的**累积(从头到最后)第 10 百分位** —— 「全程 p10」的**因果近似**。

    为什么是累积而不是滚动窗:与手势那边
    `shoulder_analyzer.ShoulderAnalyzer._calculate_shrug_level` 的全程中位数**同一条理由** ——
    「全程」这个统计量的因果估计就是「只用**已经见过**的帧」,而滚动窗是**另一个**统计量
    (窗长一选,它就成了别的量)。两者共用同一条形状:不设窗口上限、每一帧把**本帧**也放进去。

    为什么不是「开头 N 帧的冻结均值」(改前的口径):基线冻在开头 ⟹ 被试的口宽一旦在中途
    改变(笑肌疲劳、换姿势、挪机位),比值就一路贴着 0 或 1 —— 那正是 L0 行
    `au12_smile` 验收②点名的「基线跳变后比值恒定」形态。低分位只用到「哪一端更低」,
    对缓慢漂移不敏感。

    ⚠️ 这层近似**不改本列的定义**,也不要与「禁止用视频内分位定义**事件**」那条上游规矩
    混起来:本列的分位是**基线**,不是事件门限 —— 与 `shrug_level` 的全程中位数同一性质。
    """
    if not values:
        return None
    return float(np.percentile(np.asarray(list(values), dtype=float), 10.0))


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
        # ⚠️ 旧口径的 `rest_mouth_width`(开头 `max_calibration` 帧的滑动均值)**已删**
        #    (2026-09-27):`au12_smile` 的基线换成了**全程口宽 p10** 的因果近似
        #    (`causal_p10` + `mouth_width_history`,口径见 L0 行 `au12_smile`),
        #    那个字段再没人读 —— 留着一个没人读的基线字段,下一个读代码的人会以为它还在用。
        # 留下的两条 y 基线**仍是**「开头 10 帧的滑动均值」:它们属于 au10/au15 两列,而那两个
        # L0 行**明文写着**这个机制(`au10_upper_lip_raise` 的 normalization:「基线是会话内
        # 自采的开头 10 帧均值」),本批不动 —— 要动它们得连着改那两行。
        self.rest_upper_lip_y = None
        self.rest_mouth_corner_y = None
        self.calibration_frames = 0
        self.max_calibration = 10
        # `au12_smile` 的基线原料:口宽,**全程**(不设 maxlen,与肩部的 `shrug_y_history` 同形)。
        self.mouth_width_history: deque = deque()

    def extract(self, lm, face_width, face_outline_height):
        mouth_left = np.array(lm[61])
        mouth_right = np.array(lm[291])
        upper_lip = np.array(lm[13])
        lower_lip = np.array(lm[14])
        mouth_top = np.array(lm[0])

        current_mouth_width = dist.euclidean(mouth_left, mouth_right)
        # 口内开合度 = **唇内缘**两点(y 向开度,`au25_mouth_open`)。见下面 au25 的说明。
        mouth_inner_open = dist.euclidean(upper_lip, lower_lip)
        current_upper_lip_y = np.array(lm[164])[1]
        current_mouth_corner_y = (np.array(lm[61])[1] + np.array(lm[291])[1]) / 2

        if self.calibration_frames < self.max_calibration:
            if self.rest_upper_lip_y is None:
                self.rest_upper_lip_y = current_upper_lip_y
                self.rest_mouth_corner_y = current_mouth_corner_y
            else:
                alpha = 1.0 / (self.calibration_frames + 1)
                self.rest_upper_lip_y = (1 - alpha) * self.rest_upper_lip_y + alpha * current_upper_lip_y
                self.rest_mouth_corner_y = (1 - alpha) * self.rest_mouth_corner_y + alpha * current_mouth_corner_y
            self.calibration_frames += 1

        rest_upper_lip_y = self.rest_upper_lip_y or current_upper_lip_y
        rest_mouth_corner_y = self.rest_mouth_corner_y or current_mouth_corner_y

        # `au12_smile` = `max(0, (口宽 − 基线) ÷ 基线)`,基线 = **全程口宽 p10**(L0 行
        # `au12_smile`)。两处与改前不同,都是那一行的 `definition`/`l0_output` 定的:
        #   ① 基线不再冻结在开头 10 帧 ⟹ 中途改变的口宽不会被当成「一直存在的偏离」
        #      (`causal_p10` 的说明写了为什么);
        #   ② **不截顶** —— 截顶把幅度藏起来,饱和交给 L1 的分位映射(与 `au26_jaw_drop`、
        #      `shrug_level` 同一处置)。`max(0, …)` 留着:definition 里就写着它。
        # 分母那个 `+ 1e-6` 是**除零守卫**(不是门限、不改变量纲):p10 基线理论上可以是 0。
        self.mouth_width_history.append(current_mouth_width)
        baseline = causal_p10(self.mouth_width_history)
        au12_smile = max(0.0, (current_mouth_width - baseline) / (baseline + 1e-6))
        # `au25_mouth_open` = **口内开合度的原始比值** `dist(lm[13], lm[14]) ÷ 面部轮廓高`
        # (L0 行 `au25_mouth_open`)。两处与改前不同,都是 L0 行的口径定的:
        #   ① 读**唇内缘** `lm[13]`/`lm[14]`(`landmarks.py` 的 MOUTH_CENTER_TOP /
        #      MOUTH_CENTER_BOTTOM),不再读唇外缘 `lm[0]`/`lm[17]`;
        #   ② **不减静息基线、不截顶** —— §4.1:173 对"基线改滑动分位"**没给分位点与窗长**
        #      ⟹ 按 spec §4.2 的灰区规则,基线留给 L1。于是本列变成一个**无状态**的比值:
        #      同一帧在第 1 帧与第 100 帧的值逐位相等(钉子 `test_au25_mouth_open_carries_no_rest_baseline`)。
        au25_mouth_open = mouth_inner_open / face_outline_height
        # `au23_lip_compression` = **唇红厚度 ÷ 面部轮廓高**(L0 行 `au23_lip_compression`)。
        # ⚠️ 改前是 `1 − 当前唇厚/静息唇厚`,而"当前唇厚"取的是 `lm[13] − lm[14]`
        # —— 那是**口内开合度**,与 `au25_mouth_open` 语义重叠(实测会话内相关 −0.50,
        # 两者互为镜像)。改后只取**上唇两点** `lm[0]`(唇上缘)→ `lm[13]`(唇缝):
        # 张口时 `lm[14]` 下沉、这两点不动 ⟹ 本列与 au25 解耦。
        # 同时它不再依赖静息值 ⟹ 第一帧就有定义(改前第一帧恒为 0.0,一个假值)。
        au23_lip_compression = dist.euclidean(mouth_top, upper_lip) / face_outline_height
        mouth_corner_down = max(0.0, current_mouth_corner_y - rest_mouth_corner_y)
        au15_mouth_down = min(mouth_corner_down / face_outline_height, 1.0)
        au10_upper_lip_raise = max(0.0, (rest_upper_lip_y - current_upper_lip_y) / face_outline_height)
        dimple_left = np.array(lm[202])
        dimple_right = np.array(lm[422])
        left_dimple_depth = dist.euclidean(dimple_left, mouth_left)
        right_dimple_depth = dist.euclidean(dimple_right, mouth_right)
        au14_dimpler = min((left_dimple_depth + right_dimple_depth) / (2 * face_width) * 5.0, 1.0)
        # ⚠️ `au20_lip_stretcher` 与 `au12_smile` 是**同一行代码**的复制粘贴
        # (§4.1:171;`l0_columns.json` 的 `legacy_allowlist` 里登记为 `why` = 「与 au12
        # 逐字符相同的同一行代码」、`planned` = 「M3.2 删列」)。所以它读**同一个**基线、
        # 同样不截顶 —— 两列必须继续**逐位相等**。一旦让它们分岔,那句登记就变成假话,
        # 而且是**静默**的(钉子 `test_au20_lip_stretcher_stays_bit_identical_to_au12`)。
        au20_lip_stretcher = max(0.0, (current_mouth_width - baseline) / (baseline + 1e-6))
        # `au26_jaw_drop` = `|下巴 y − 上唇内缘 y| ÷ 面部轮廓高`(L0 行 `au26_jaw_drop`)。
        # 三处与改前不同:① 分母不再是 `0.1 × face_height` —— 那个 `0.1` 在文档里**查不到出处**;
        # ② **不减静息基线**(它的作用被 L1 的 p1/p99 映射取代);③ **不截顶** ——
        # 截顶把幅度藏起来,饱和交给 L1 的分位映射(§4.1:174「饱和即消失」)。
        # 实测改前截顶命中 92/124/40 帧(三场正式素材),改后 0。
        au26_jaw_drop = abs(np.array(lm[152])[1] - upper_lip[1]) / face_outline_height

        # 标准化 —— ⚠️ `au12_smile`/`au20_lip_stretcher` **不在**这一批里:它们**不截顶**
        # (见上面的口径说明)。夹一刀就是把测量值改成另一个量。
        au23_lip_compression = max(0.0, min(au23_lip_compression, 1.0))
        au15_mouth_down = max(0.0, min(au15_mouth_down, 1.0))
        au10_upper_lip_raise = max(0.0, min(au10_upper_lip_raise, 1.0))
        au14_dimpler = max(0.0, min(au14_dimpler, 1.0))

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
    def extract(lm, face_width, face_outline_height):
        """`au1_inner_brow_raise` / `au2_outer_brow_raise` = 眉抬起的**竖直**位移 ÷ 面部轮廓高。

        ⚠️ 分母 2026-09-27 换过端点:旧的是 `face_height = dist(lm[1], lm[152])`(鼻尖→下巴),
        现在这个参数里装的是**面部轮廓高** `dist(lm[10], lm[152])`(见 `video_pipeline.py`
        与 L0 行 `au1_inner_brow_raise` 的 normalization)⟹ 两列的比值整体变大
        (实测三场改前→改后见该行的 acceptance)。
        """
        inner_brow_left = np.array(lm[52])
        inner_brow_right = np.array(lm[55])
        outer_brow_left = np.array(lm[70])
        outer_brow_right = np.array(lm[63])
        brow_center = np.array(lm[168])

        inner_lift_left = brow_center[1] - inner_brow_left[1]
        inner_lift_right = brow_center[1] - inner_brow_right[1]
        outer_lift_left = brow_center[1] - outer_brow_left[1]
        outer_lift_right = brow_center[1] - outer_brow_right[1]

        au1_inner_brow_raise = max(0.0, (inner_lift_left + inner_lift_right) / (2 * face_outline_height))
        au2_outer_brow_raise = max(0.0, (outer_lift_left + outer_lift_right) / (2 * face_outline_height))

        au1_inner_brow_raise = max(0.0, min(au1_inner_brow_raise, 1.0))
        au2_outer_brow_raise = max(0.0, min(au2_outer_brow_raise, 1.0))

        return {
            'au1_inner_brow_raise': au1_inner_brow_raise,
            'au2_outer_brow_raise': au2_outer_brow_raise
        }


class NoseFeatureExtractor:
    @staticmethod
    def extract(lm, face_width, face_outline_height):
        """`au9_nose_wrinkle` = 1 − 鼻尖到两鼻翼均距 ÷ face_width(L0 行 `au9_nose_wrinkle`)。

        ⚠️ 改前鼻翼读的是 `lm[234]`/`lm[455]` —— 那是**脸颊轮廓点**,不是鼻翼
        (§4.1:166 依据栏里的 234/455 描述的就是这个 bug)。改用 `landmarks.py`
        声明的 `NOSE_WING_LEFT`(129)/`NOSE_WING_RIGHT`(358)。
        **分母照旧 ÷ face_width**(`video_pipeline` 的 `dist(lm[234], lm[455])`):
        L0 行 basis 写明"沿用现行分母,只换下标"。
        ⚠️ 本列**不读**第三个参数(它是给同族签名用的);`face_outline_height` 换端点
        (2026-09-27)对本列**无影响** —— 钉子 `test_the_columns_that_do_not_take_this_divisor_are_untouched`。
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
    def extract(lm, face_width, face_outline_height):
        """`head_yaw` / `head_pitch` —— 两个都是 **2D 代理**(真 3D 版被 L0 行判 `blocked`)。

        ⚠️ `head_pitch` 的分母 2026-09-27 换过端点:旧的是 `dist(lm[1], lm[152])`(鼻尖→下巴),
        现在装的是**面部轮廓高** `dist(lm[10], lm[152])`。它的分子**仍**是
        `鼻尖 y − 下巴 y`(那是它的定义,不是分母的口径)⟹ 两处后果:
          · 值域整体上移(三场实测均值 −0.999/−0.991/−0.996 → −0.722/−0.717/−0.720,
            见 L0 行 `head_pitch`);
          · 「分子分母同源」这个病**没有**被治 —— 分子还是鼻尖,而分母的端点已经换成前额顶,
            两者的相关性下降,但 `head_pitch` 仍是 2D 比值,不是俯仰角。L0 行 `head_pitch`
            的 `status` 仍是 `blocked`(解封条件 = 相机内参),本批不改变这一点。
        """
        nose_tip = np.array(lm[1])
        cheek_left = np.array(lm[234])
        cheek_right = np.array(lm[455])
        ear_mid = (cheek_left + cheek_right) / 2
        head_yaw = (nose_tip[0] - ear_mid[0]) / face_width
        chin = np.array(lm[152])
        head_pitch = (nose_tip[1] - chin[1]) / face_outline_height
        return {
            'head_yaw': head_yaw,
            'head_pitch': head_pitch
        }


class CheekFeatureExtractor:
    @staticmethod
    def extract(lm, face_outline_height):
        """`au6_cheek_raise` = 1 − (颊-眼竖直距两段之和) ÷ (2 × **面部轮廓高**)。

        分母的两次更换(L0 行 `au6_cheek_raise` 的 basis 写了完整依据 + 实测数):

        * 改前是 `face_height`(= `dist(lm[1], lm[152])`,鼻尖→下巴);
        * B2 换成**眼距** `dist(lm[130], lm[359])` —— 实测**反而更差**(该列与 `head_yaw`
          的会话内相关 −0.017/+0.260/−0.012 → −0.236/+0.508/+0.300,三场正式素材重放);
        * 现用 `dist(lm[10], lm[152])`(`landmarks.py` 的 FOREHEAD_TOP → CHIN)。

        为什么它该稳(不是"试到哪个好就用哪个"):
          ① **竖直** —— yaw 是绕竖直轴的旋转,画面的**横向**跨度按 cos(yaw) 缩短,
             竖直跨度一阶不变;分子(颊↔眼下缘)实测 |Δy|/(|Δx|+|Δy|) = 0.90,是竖直量。
             横向下场最差(面宽 +0.851/+0.673/+0.988),竖直的都更好。
          ② **两端都在面部中轴轮廓上** —— `lm[1]` 是**突出的鼻尖**(在脸平面之外),
             yaw 下它的投影横向滑动,给一个本该竖直的量塞进一个横向项。实测
             `r(head_yaw, |Δx(nose→chin)|)` = −0.437/−0.266/−0.069,而同一条
             `|Δx(lm[10]→chin)|` 只有 −0.040/−0.206/−0.186 ⟹ 端点换成轮廓点后
             这个横向污染项大幅变小。

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
        au6_cheek_raise = 1.0 - (left_cheek_lift + right_cheek_lift) / (2 * face_outline_height)
        au6_cheek_raise = max(0.0, min(au6_cheek_raise, 1.0))
        return {'au6_cheek_raise': au6_cheek_raise}


class SymmetryExtractor:
    @staticmethod
    def extract(lm, face_outline_height):
        """`symmetry_score` = 1 − (|Δ眼中心y| + |Δ嘴角y|) ÷ **面部轮廓高**。

        ⚠️ 改前**什么都不除**:分子是归一化图像坐标里的竖直差,差多少就直接减多少,
        于是该列量的是"脸在画面里多大/多斜"—— 取景代理(§4.1:208 依据栏
        「未除人脸尺度 → 取景代理」)。实测会话内 mean 0.9825、std ≈0.012,近常量。
        除尺度后,人脸整体缩放时分子分母同步变 ⟹ 值不动
        (钉子 `test_symmetry_score_is_invariant_to_face_scale`)。
        分母取竖直尺度是因为分子是**竖直**差 —— 与 au6 那条"分子分母同向"的理由同源。
        ⚠️ 2026-09-27:那个竖直尺度的**端点**从鼻尖换成前额顶(`dist(lm[10], lm[152])`,
        L0 行 `symmetry_score` 的 normalization 由本行定 ⟹ 本行的文本随本批改)。
        """
        left_eye_center = np.mean([lm[i] for i in [33, 133]], axis=0)
        right_eye_center = np.mean([lm[i] for i in [362, 263]], axis=0)
        mouth_left = np.array(lm[61])
        mouth_right = np.array(lm[291])
        eye_y_diff = abs(left_eye_center[1] - right_eye_center[1])
        mouth_y_diff = abs(mouth_left[1] - mouth_right[1])
        symmetry_score = max(0.0, min(1.0, 1.0 - (eye_y_diff + mouth_y_diff) / face_outline_height))
        return {'symmetry_score': symmetry_score}


# === 新增：眼球特征提取器 ===
class IrisFeatureExtractor:
    @staticmethod
    def extract(lm, face_width, face_outline_height):
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
        # ⚠️ 只影响 `gaze_direction_y` / `gaze_deviation` —— 两列在 L0 白名单里
        #    (`legacy_allowlist`,planned = M3.2 删列),`evidence_gate.QUARANTINE` 也封着它们。
        #    2026-09-27 起分母是**面部轮廓高**。
        gaze_direction_y = (eye_mid[1] - nose_tip[1]) / face_outline_height

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

    def calculate(self, landmarks_norm, face_width, face_outline_height) -> AUFeatures:
        features = {}

        # ★ **眼距算一次、传下去**(L0 行 iris 四列的 normalization 与 face_scale 验收①
        #   「不许两处各算一遍同一个 dist」)。au4 与 au6 各自重算一遍在数学上等价,
        #   但一旦有人只改其中一处(比如换下标),两列就会**静默地**用两个不同的尺度,
        #   而这种不一致不会有任何报错。
        #   ⚠️ B2 之后 **au6 不再用它**:它的分母换成了眼距→**面部轮廓高**
        #   (眼距是横向跨度,实测把 yaw 带进 au6)。现在只剩 au4 用它。
        eye_distance = float(dist.euclidean(np.array(landmarks_norm[130]),
                                            np.array(landmarks_norm[359])))
        # ★ **面部轮廓高**:前额顶(`FOREHEAD_TOP`)→ 下巴(`CHIN`),两端都在面部中轴
        #   轮廓上。它是 `au6_cheek_raise` 的分母 —— 为什么不是眼距/鼻尖→下巴,见
        #   `CheekFeatureExtractor.extract` 的说明与 L0 行 `au6_cheek_raise` 的 basis。
        #   同样只算一次、传下去(理由同上)。
        #   ⚠️ 这个**参数**装的就是它(`video_pipeline.process_frame` 里算的那一个),
        #   本函数**不**重算 —— 验收① 钉的正是"两处各算一遍就会静默分岔"。

        features.update(EyeFeatureExtractor.extract(landmarks_norm))
        features.update(self.mouth_extractor.extract(landmarks_norm, face_width, face_outline_height))
        features.update(BrowFeatureExtractor.extract(landmarks_norm, eye_distance))
        features.update(BrowRaiserExtractor.extract(landmarks_norm, face_width, face_outline_height))
        features.update(NoseFeatureExtractor.extract(landmarks_norm, face_width, face_outline_height))
        features.update(CheekFeatureExtractor.extract(landmarks_norm, face_outline_height))
        features.update(HeadPoseExtractor.extract(landmarks_norm, face_width, face_outline_height))
        features.update(SymmetryExtractor.extract(landmarks_norm, face_outline_height))
        features.update(IrisFeatureExtractor.extract(landmarks_norm, face_width, face_outline_height))

        # ★ `face_scale`:该帧的**面部尺度本身**,作为**协变量**显式输出
        #   (§4.5:272 对这类量的两条路之一:要么按解剖尺度归一,要么显式输出为协变量)。
        #   它**就是**各列当分母用的那个数 —— 传进来的那个,不是在这里重算一遍
        #   `dist(lm[10], lm[152])`(验收①:两处各算一遍就会静默分岔)。
        #   ⚠️ 必须显式赋值:下面那条"补充缺失字段(默认0)"的循环会把漏掉的字段填成 0.0,
        #   而 0 是个**看着像测量值**的数(钉子 `test_face_scale_is_not_silently_defaulted_to_zero`)。
        #   ⚠️ 2026-09-27:这个数与上一批的**不同名同值关系**已随分母换端点而变
        #   (旧:鼻尖→下巴;现:前额顶→下巴)—— 见 L0 行 `face_scale`。
        features['face_scale'] = float(face_outline_height)

        # 补充缺失字段（默认0）
        all_fields = {f.name for f in AUFeatures.__dataclass_fields__.values()}
        for field in all_fields:
            if field not in features:
                features[field] = 0.0

        # 可选：保存原始 landmarks
        if self.save_landmarks:
            features['landmarks'] = [pt[0] for pt in landmarks_norm] + [pt[1] for pt in landmarks_norm]

        return AUFeatures(**features)