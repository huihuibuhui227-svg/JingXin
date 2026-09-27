# tests/test_face_outline_height.py
"""面部尺度分母 = **面部轮廓高** `dist(lm[10], lm[152])`(前额顶→下巴)。

**为什么改**(2026-09-27,三场正式素材实测):旧分母
`face_height = dist(lm[1], lm[152])`(鼻尖→下巴)的**一端是突出在脸平面之外的鼻尖**,
yaw 下它的投影横向滑动,给一个本该竖直的尺度塞进一个横向项 ——
`r(head_yaw, |Δx(鼻尖→下巴)|)` = −0.437/−0.266/−0.135,而同一条换成**前额顶**只有
−0.040/−0.206/−0.142。`au6_cheek_raise` 早已改用轮廓高(那一行的 basis 写了完整依据),
本批把**源头的那一次**改动补齐:**所有**以它为分母的列一起换尺度
(au1/au2/au10/au15/au23/au25/au26、`head_pitch`、`symmetry_score`、`gaze_direction_y`),
以及协变量 `face_scale` **本身** —— 它就是这个分母,不是另一个量。

⚠️ **这两条钉子必须走真产出方**:分母是**在管线里算的**
(`video_pipeline.process_frame` 的 `dist.euclidean(..., chin)`),不在
`AUFeatureCalculator` 里。只喂 calculator 的话,管线里那个 `dist` 的端点改不改都绿 ——
本仓已 5 次栽在「验证跑错了对象」。所以这里注入假探测器跑 `process_frame`,读它交出的
序列化字典(与落盘那条路同一个 `to_dict()`)。

合成分岔的几何(**两个候选分母故意不同**):前额顶 lm[10] (0.50, 0.05)、
鼻尖 lm[1] (0.50, 0.30)、下巴 lm[152] (0.50, 0.95)
⟹ 轮廓高 **0.90**、鼻尖→下巴 **0.65**。
"""
import numpy as np
import pytest

from face_expression.pipeline.video_pipeline import VideoPipeline

_FRAME = np.zeros((48, 48, 3), dtype=np.uint8)


def _lm(points):
    """造 478 点:未点名的点按**下标铺成一个极小的稀疏网格**(不是全 0)。

    为什么默认值不能全等(与 `test_face_l0_columns._lm` 同一条理由):眼宽
    `dist(lm[33], lm[133])` 是分母,全等 ⟹ **ZeroDivisionError** —— 红的理由就成了
    "崩了"而不是"口径错了"。铺开后任意两点距离 ≥ 1e-4 > 0,而量级(≤0.002)相对下面
    断言里的坐标(≥0.1)可忽略。
    """
    lm = np.zeros((478, 2), dtype=float)
    for i in range(478):
        lm[i] = ((i % 20) * 1e-4, (i // 20) * 1e-4)
    for i, (x, y) in points.items():
        lm[i] = (x, y)
    return lm


class _FakeDetector:
    """只交一帧固定的 landmarks。不 import mediapipe。

    ⚠️ 交出的是 `[(x, y), …]`(**列表**,不是 ndarray)—— 真探测器
    (`face_expression/pipeline/detector.py`)交的就是这个形状,而 `process_frame` 里
    有一句 `if not landmarks_norm:`:喂 ndarray 会在那里**炸**
    (`ValueError: truth value of an array … is ambiguous`),红的理由就成了"崩了"
    而不是"口径错了"(TDD 的"看着它失败"要求红的理由与预期一致)。
    """

    def __init__(self, landmarks):
        self._landmarks = [(float(x), float(y)) for x, y in landmarks]

    def detect_with_blendshapes(self, image_rgb, timestamp_ms):
        return self._landmarks, {}

    def detect(self, image_rgb, timestamp_ms):
        return self._landmarks

    def reset(self):
        pass

    def close(self):
        pass


# 「每个消费列都算得出非零值」的合成脸。分两段:几何设定 + 由设定推出的期望值。
_POINTS = {
    # ── 面部中轴轮廓(分母的三个端点)────────────────────────────
    10: (0.50, 0.05),      # 前额顶  FOREHEAD_TOP
    1: (0.50, 0.30),       # 鼻尖    NOSE_TIP(旧分母的端点)
    152: (0.50, 0.95),     # 下巴    CHIN
    # ── 面宽(不改)──────────────────────────────────────────────
    # ⚠️ 两点**故意不对称**地摆(中点 0.45,不是 0.50):head_yaw 靠它拿到 +0.125,
    #    同时 face_width 仍是 0.40 —— 鼻尖留在中轴上,au9 那条期望值才是干净的 0.75。
    234: (0.25, 0.50), 455: (0.65, 0.50),          # face_width = 0.40, ear_mid.x = 0.45
    # ── 嘴部 ──────────────────────────────────────────────────
    0: (0.50, 0.40),       # 唇上缘   → au23 = 0.10/分母
    13: (0.50, 0.50),      # 唇内缘上 → au26 = |0.95−0.50|/分母
    14: (0.50, 0.54),      # 唇内缘下
    61: (0.45, 0.60), 291: (0.55, 0.62),           # 嘴角:Δy = 0.02 给 symmetry
    # ── 眉毛:眉中心在下、眉端点在上 ⟹ 抬起为正 ────────────────
    168: (0.50, 0.48),
    52: (0.45, 0.30), 55: (0.55, 0.30),            # 眉内端,各抬 0.18 → au1
    70: (0.35, 0.30), 63: (0.65, 0.30),            # 眉外端,各抬 0.18 → au2
    # ── 眼睛(形状给 avg_ear ≈ 0.20;不改)────────────────────
    33: (0.35, 0.45), 133: (0.45, 0.45),           # 左眼角,宽 0.10
    160: (0.37, 0.44), 153: (0.37, 0.46),
    159: (0.40, 0.44), 144: (0.40, 0.46),
    362: (0.65, 0.45), 263: (0.55, 0.45),          # 右眼角,宽 0.10
    387: (0.63, 0.44), 380: (0.63, 0.46),
    386: (0.60, 0.44), 373: (0.60, 0.46),
    # ── 虹膜:眼内相对坐标(不改)──────────────────────────────
    **{i: (0.43, 0.47) for i in (468, 469, 470, 471)},
    **{i: (0.57, 0.43) for i in (473, 474, 475, 476)},
    # ── au6(已经是轮廓高,本批**不该动**)────────────────────
    205: (0.40, 0.50), 145: (0.40, 0.60),          # 颊-眼下缘 0.10
    425: (0.60, 0.50), 374: (0.60, 0.60),
    # ── au9(用 face_width,本批**不该动**):鼻翼与鼻尖同高 ⟹ 两段各 0.10
    129: (0.40, 0.30), 358: (0.60, 0.30),
    # ── au4(用眼距,本批**不该动**)─────────────────────────
    130: (0.30, 0.40), 359: (0.70, 0.40),          # 眼距 0.40
    107: (0.45, 0.35), 336: (0.55, 0.35),          # 眉内端 0.10 → au4 = 0.25
    # ── au14(用 face_width,本批**不该动**)─────────────────
    202: (0.40, 0.60), 422: (0.60, 0.60),
}


def _run(points=None, ts=1000):
    """跑**真产出方**:管线一帧 → 交回落盘用的那个扁平字典。"""
    p = VideoPipeline(session_id="s", detector=_FakeDetector(_lm(points or _POINTS)))
    _result, _mesh, dumped = p.process_frame(_FRAME, ts)
    return dumped


def test_the_divisor_is_the_outline_span_not_the_nose_to_chin_span():
    """★ 源头改的那一次:分母 = `dist(lm[10], lm[152])`(**轮廓高 0.90**)。

    旧分母 `dist(lm[1], lm[152])` 在这份几何里是 **0.65** ⟹ 两个数分岔,任一处改回去
    立刻红。逐列的红法(括号里是旧分母会给的值):
      · `face_scale`(= 分母本身,4 位小数)  0.90(**0.65**)
      · `head_pitch` = (0.30−0.95)/0.90 = −0.722(**−1.0**:分子分母同源,旧口径下恒 ≈ −1)
      · `au26_jaw_drop` = 0.45/0.90 = 0.5(**0.6923**)
      · `au1_inner_brow_raise` = 0.36/(2×0.90) = 0.20(**0.2769**)
      · `symmetry_score` = 1 − 0.02/0.90 = 0.9778(**0.9692**)
    """
    d = _run()
    assert d["face_scale"] == pytest.approx(0.90), (
        f"face_scale 该是**面部轮廓高** dist(lm[10], lm[152]) = 0.90,实为 {d['face_scale']} —— "
        f"0.65 说明分母还是鼻尖→下巴(dist(lm[1], lm[152]))")
    assert d["head_pitch"] == pytest.approx(-0.722, abs=1e-3), (
        f"head_pitch 该是 (0.30−0.95)/轮廓高 0.90 = −0.722,实为 {d['head_pitch']} —— "
        f"−1.0 说明它还在除鼻尖→下巴(分子分母同源那个形态)")
    assert d["au26_jaw_drop"] == pytest.approx(0.5, abs=1e-3), (
        f"au26 该是 0.45/0.90 = 0.5,实为 {d['au26_jaw_drop']} —— 0.692 说明分母是 0.65")
    assert d["au1_inner_brow_raise"] == pytest.approx(0.20, abs=1e-3), (
        f"au1 该是 0.36/(2×0.90) = 0.20,实为 {d['au1_inner_brow_raise']} —— 0.277 说明分母是 0.65")
    assert d["symmetry_score"] == pytest.approx(1 - 0.02 / 0.90, abs=1e-3), (
        f"symmetry 该是 1 − 0.02/0.90 = 0.978,实为 {d['symmetry_score']} —— 0.969 说明分母是 0.65")


def test_the_columns_that_do_not_take_this_divisor_are_untouched():
    """反向那半:**只**换了分母的端点 —— 别的尺度(front 宽、眼距)一个都不许动。

    这条是"改动是定点手术、不是散弹枪"的判据。若上面那条改了、这条也红,说明有人顺手
    改了别的分母(或把 `face_width` 也算成了轮廓高)。
      · `au6_cheek_raise` = 1 − 0.20/(2×0.90) = 8/9 ≈ 0.889(它**早已**是轮廓高)
      · `au9_nose_wrinkle` = 1 − 0.20/(2×0.40) = 0.75(÷ face_width)
      · `au4_frown`       = 0.10/0.40 = 0.25(÷ 眼距)
      · `left_iris_x`     = (0.43−0.40)/0.10 = 0.30(÷ 眼宽)
    """
    d = _run()
    assert d["au6_cheek_raise"] == pytest.approx(8 / 9, abs=1e-3), d["au6_cheek_raise"]
    assert d["au9_nose_wrinkle"] == pytest.approx(0.75, abs=1e-3), d["au9_nose_wrinkle"]
    assert d["au4_frown"] == pytest.approx(0.25, abs=1e-3), d["au4_frown"]
    assert d["left_iris_x"] == pytest.approx(0.30, abs=1e-3), d["left_iris_x"]


def test_the_scale_and_au6_are_invariant_to_horizontal_foreshortening():
    """★ **yaw 病的判据**(从 `test_face_l0_columns.py` 移来,2026-09-27):横向整体压缩
    (转头造成的透视缩短)不该改变**尺度**与 `au6_cheek_raise`。

    为什么移到这里:分母现在**由管线从 landmarks 算**(calculator 只收一个数)⟹ 在
    calculator 层喂一个固定尺度时,"压缩横向"对分母没有任何影响,那条判据在那里**恒真**。
    要钉的性质是「分母是**竖直**跨度」,而它发生在这一层。

    构造:分母两端(lm[10]/lm[152])与 au6 的分子两段**都是纯竖直段**(Δx = 0)
    ⟹ 压缩横向对它们**没有影响**;而 `face_width`(lm[234]/lm[455])是纯横向段 ⟹ 按比例缩短。
    ⚠️ 鼻尖**刻意摆在中轴之外**(0.70):旧分母 `dist(lm[1], lm[152])` 正是靠它把横向项吃进来
    —— 实测这份几何上压缩 0.6 倍会让**旧**分母从 0.680 变成 0.661(**−2.8%**),
    而**新**分母一动都不动(0%)。⟹ 这条在改前是红的。

    红法:分母改回含横向分量的量(鼻尖→下巴、眼距)⟹ 立刻红。
    """
    base = {**_POINTS, 1: (0.70, 0.30),
            # 鼻翼与鼻尖**错开 y**(0.50 vs 0.30)⟹ au9 的分子是**斜**段,横向压缩下不按
            # 0.6 缩短 ⟹ 它变。这是"这一帧真的被压缩了"的对照(纯横/纯竖的量都按比例缩,
            # 拿它们当对照会**恒等**,零区分力 —— 实测踩过)。
            129: (0.40, 0.50), 358: (0.60, 0.50)}
    wide = _run(base)
    squeezed = _run({i: (x * 0.6, y) for i, (x, y) in base.items()})
    assert wide["face_scale"] == pytest.approx(0.90, abs=1e-4), wide["face_scale"]
    assert squeezed["face_scale"] == pytest.approx(wide["face_scale"]), (
        f"压缩横向不该改变面部尺度:{wide['face_scale']} -> {squeezed['face_scale']} —— "
        f"变小说明分母里含横向分量(它随转头缩短)")
    assert wide["au6_cheek_raise"] == pytest.approx(8 / 9, abs=1e-3), wide["au6_cheek_raise"]
    assert squeezed["au6_cheek_raise"] == pytest.approx(wide["au6_cheek_raise"], abs=1e-3), (
        f"压缩横向不该改变 au6:{wide['au6_cheek_raise']} -> {squeezed['au6_cheek_raise']}")
    # 对照组:这一帧**确实**被横向压缩了(否则上面两条可能是"整条管线都没动")
    assert abs(squeezed["au9_nose_wrinkle"] - wide["au9_nose_wrinkle"]) > 0.1, (
        f"au9 的分子是斜段,压缩后该明显变:{wide['au9_nose_wrinkle']} -> "
        f"{squeezed['au9_nose_wrinkle']} —— 没变说明这一帧根本没被压缩")


def test_every_consumer_divides_by_the_number_that_goes_out_as_face_scale():
    """★ 「不许两处各算一遍同一个 dist」的**管线级**钉子。

    断言写成**以 `face_scale` 为自变量**的形式:分子是各列自己的几何量(本例里是常数),
    分母必须**逐字就是**交出去的那个 `face_scale`。哪一个消费方自己重算一遍分母
    (例如 `HeadPoseExtractor` 内部再写一次 `dist(lm[1], lm[152])`),这条就红 ——
    而 `face_scale` 那条断言**抓不到**它(协变量仍是对的,只是消费者用了另一个数)。
    """
    d = _run()
    scale = d["face_scale"]
    assert d["au26_jaw_drop"] == pytest.approx(0.45 / scale, abs=1e-3), (
        f"au26 ({d['au26_jaw_drop']}) 不等于 0.45 ÷ face_scale ({0.45/scale}) —— "
        f"它的分母与交出去的 face_scale 不是同一个数")
    assert d["au1_inner_brow_raise"] == pytest.approx(0.36 / (2 * scale), abs=1e-3), (
        f"au1 ({d['au1_inner_brow_raise']}) 不等于 0.36 ÷ (2×face_scale) —— 分母分岔了")
    assert abs(d["head_pitch"]) == pytest.approx(0.65 / scale, abs=1e-3), (
        f"head_pitch ({d['head_pitch']}) 的绝对值不等于 0.65 ÷ face_scale —— 分母分岔了")
