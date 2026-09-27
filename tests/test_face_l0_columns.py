# tests/test_face_l0_columns.py
"""M3.2 面部 10 列的 L0 逐列重构(B 档)。

这 10 列的**共同病根**只有两个(L0 表逐行写明了):
  · **下标错** —— 拿错了 landmark(au4 用 276/33、au9 用 234/455);
  · **分母错/没有分母** —— 除以裸常量或干脆不除,于是量到的是**取景**不是脸
    (au6 除 face_height、au23 除静息值、symmetry 不除、iris 四列直接吐归一化图像坐标)。

**为什么每一条都配一个"旧口径会给别的数"的断言**:本仓栽过 5 次「验证跑错了对象」——
只断言"值在合理范围内"这类判据,新旧口径**都能过**,等于没验。所以下面每条的红法都是
**构造一份让新旧口径必然分岔的几何**,再把新口径的数钉死。合成分岔不出来的,
就用**性质**(平移不变 / 缩放不变)来分岔:取景代理的定义就是"随取景变",所以
"取景变了值不变"是一条只有对的口径才过得了的判据。

合成 landmarks 的约定:477 个点先铺成同一个默认点,再逐点覆盖。凡是本列不读的点
留在默认点上也**不影响**本列的值 —— 这本身就是"它只读它该读的那几个点"的证明。
"""
import numpy as np
import pytest

from face_expression.core.feature_extraction.au_calculator import AUFeatureCalculator


def _lm(points):
    """造 478 点:未点名的点按**下标铺成一个极小的稀疏网格**(不是全 0),点名的字典覆盖。

    为什么默认值不能全是同一个点(踩过):`EyeFeatureExtractor._eye_aspect_ratio` 除以
    `dist(lm[33], lm[133])`。默认点若全等,那个距离是 0 ⟹ **ZeroDivisionError** ——
    于是测试红的理由是"崩了",而不是"口径错了"。红的理由不对,这条测试就什么都没证明
    (TDD 的"看着它失败"要求失败理由与预期一致)。铺开后任意两点距离 ≥ 1e-4 > 0,
    而量级(≤0.002)相对断言里的坐标(≥0.1)可忽略,不会污染任何一条断言。
    """
    lm = np.zeros((478, 2), dtype=float)
    for i in range(478):
        lm[i] = ((i % 20) * 1e-4, (i // 20) * 1e-4)
    for i, (x, y) in points.items():
        lm[i] = (x, y)
    return lm


def _f(points, face_width=0.40, face_height=0.40):
    return AUFeatureCalculator().calculate(_lm(points), face_width, face_height)


# ── au4_frown ────────────────────────────────────────────────────────────────
def test_au4_frown_is_the_ratio_itself_over_eye_distance():
    """`au4_frown` = `dist(lm[107], lm[336]) ÷ dist(lm[130], lm[359])`,**比值本身**(不再 1−)。

    几何(眉内距 0.10、眼距 0.40):新口径 = 0.10/0.40 = **0.25**。
    旧口径 `1 − dist(lm[276], lm[33]) ÷ face_width`:把 276/33 摆在 0.20 之外,
    `1 − 0.20/0.40` = **0.50** ⟹ 两个数必然分岔。

    红法:把眉内端换回 276/33、或把返回值改成 `1 − 比值` ⟹ 立刻红(症状是 86~92% 触地板)。
    """
    f = _f({107: (0.45, 0.30), 336: (0.55, 0.30),     # 眉内端:相距 0.10
            130: (0.30, 0.40), 359: (0.70, 0.40),     # 眼距:0.40
            276: (0.20, 0.30), 33: (0.40, 0.30)})     # 旧口径读的点(相距 0.20)
    assert f.au4_frown == pytest.approx(0.25), (
        f"该是 眉内距÷眼距 = 0.25(比值本身),实为 {f.au4_frown} —— "
        f"0.50 说明还在读 276/33,0.75 说明还在算 1−比值")


def test_au4_frown_is_not_a_floor_constant():
    """反向那半:眉内距在解剖上会变 ⟹ 该列必须跟着变,不许贴地板。

    红法:任何把它写成常量的改动(例如再套一层 max(0, ...) 把负值吃掉)⟹ 红。
    """
    near = _f({107: (0.48, 0.30), 336: (0.52, 0.30),
               130: (0.30, 0.40), 359: (0.70, 0.40)}).au4_frown
    far = _f({107: (0.40, 0.30), 336: (0.60, 0.30),
              130: (0.30, 0.40), 359: (0.70, 0.40)}).au4_frown
    assert far > near * 3, f"眉内距翻倍该让本列明显变大:{near} -> {far}"


# ── au6_cheek_raise ──────────────────────────────────────────────────────────
def test_au6_cheek_raise_divides_by_eye_distance_not_face_height():
    """分母由 `face_height` 改成**眼距**(L0 行 normalization 栏)。

    几何:颊-眼距两段各 0.10(和 0.20)、眼距 0.40、`face_height` **故意给 0.80**。
      · 新口径:1 − 0.20/(2×0.40) = **0.75**
      · 旧口径:1 − 0.20/(2×0.80) = **0.875**
    ⟹ 分岔。红法:分母改回 `face_height` ⟹ 立刻红。
    """
    f = _f({205: (0.40, 0.50), 145: (0.40, 0.60),     # 左颊-眼下缘:0.10
            425: (0.60, 0.50), 374: (0.60, 0.60),     # 右颊-眼下缘:0.10
            130: (0.30, 0.40), 359: (0.70, 0.40)},    # 眼距:0.40
           face_height=0.80)
    assert f.au6_cheek_raise == pytest.approx(0.75), (
        f"该是 1 − 0.20/(2×眼距 0.40) = 0.75,实为 {f.au6_cheek_raise} —— "
        f"0.875 说明分母还是 face_height")


# ── au9_nose_wrinkle ─────────────────────────────────────────────────────────
def test_au9_nose_wrinkle_uses_the_declared_nose_wing_landmarks():
    """鼻翼改用 `landmarks.py` 声明的 129/358(不再用脸颊轮廓点 234/455)。

    几何:鼻尖 0.50 → 两鼻翼各 0.10(和 0.20)、`face_width` 0.40 ⟹ **0.75**。
    旧口径读 234/455(摆在 0.10/0.90,相距 1.60)⟹ `1 − 1.60/0.80` 被夹成 **0.0**。
    ⟹ 分岔。红法:鼻翼改回 234/455 ⟹ 立刻红(症状:每视频 max 恰为 0.5)。
    """
    f = _f({1: (0.50, 0.50),
            129: (0.40, 0.50), 358: (0.60, 0.50),     # 声明的鼻翼:各距 0.10
            234: (0.10, 0.50), 455: (0.90, 0.50)})    # 旧口径读的脸颊点
    assert f.au9_nose_wrinkle == pytest.approx(0.75), (
        f"该是 1 − 0.20/(2×face_width 0.40) = 0.75,实为 {f.au9_nose_wrinkle} —— "
        f"0.0 说明还在读 234/455")


# ── au23_lip_compression ─────────────────────────────────────────────────────
def test_au23_lip_compression_is_a_pure_geometric_ratio():
    """`au23` = `dist(lm[0], lm[13]) ÷ face_height` —— **不再**吃静息值。

    几何:上唇上缘 (0.50,0.40) → 唇缝 (0.50,0.50),相距 0.10;`face_height` 0.40 ⟹ **0.25**。
    旧口径在**第一帧**把静息值取成本帧 ⟹ `1 − 1/1` = **0.0**。⟹ 分岔。

    红法:改回 `1 − 当前唇厚/静息唇厚` ⟹ 第一帧变 0.0,立刻红。
    """
    f = _f({0: (0.50, 0.40), 13: (0.50, 0.50)})
    assert f.au23_lip_compression == pytest.approx(0.25), (
        f"该是 唇红厚度÷face_height = 0.25,实为 {f.au23_lip_compression} —— 0.0 说明还在用静息值")


def test_au23_does_not_move_when_the_mouth_opens():
    """★ 该列量的是**唇厚**,不是**张口**。张口时 `lm[14]` 下沉、`lm[13]` 不动 ⟹ 本列不变。

    这正是它与 `au25_mouth_open` 从"互为镜像"(实测会话内相关 −0.50)解耦的机制:
    两者读的点不再重合。
    红法:改回读 `lm[13] − lm[14]`(口内开合度)⟹ 张口那一下本列就变,立刻红。
    """
    closed = _f({0: (0.50, 0.40), 13: (0.50, 0.50), 14: (0.50, 0.52)}).au23_lip_compression
    open_ = _f({0: (0.50, 0.40), 13: (0.50, 0.50), 14: (0.50, 0.70)}).au23_lip_compression
    assert closed == pytest.approx(open_), (
        f"张口不该改变唇厚:闭口 {closed} vs 张口 {open_} —— "
        f"两者不等说明还在读 lm[13]−lm[14](与 au25 互为镜像)")


# ── symmetry_score ───────────────────────────────────────────────────────────
def test_symmetry_score_divides_by_face_height():
    """`symmetry_score` = `1 − (|Δ眼中心y| + |Δ嘴角y|) ÷ face_height`。

    几何:两眼中心同高(Δ=0)、嘴角 y 差 0.02、`face_height` 0.40 ⟹ `1 − 0.02/0.40` = **0.95**。
    旧口径不除 ⟹ **0.98**。⟹ 分岔。红法:去掉除法 ⟹ 立刻红。
    """
    f = _f({33: (0.40, 0.40), 133: (0.45, 0.40),      # 左眼中心 y=0.40
            362: (0.55, 0.40), 263: (0.60, 0.40),     # 右眼中心 y=0.40
            61: (0.45, 0.60), 291: (0.55, 0.62)})     # 嘴角 y 差 0.02
    assert f.symmetry_score == pytest.approx(0.95), (
        f"该是 1 − 0.02/0.40 = 0.95,实为 {f.symmetry_score} —— 0.98 说明还没除人脸尺度")


def test_symmetry_score_is_invariant_to_face_scale():
    """★ **取景代理病的判据**:把人脸整体放大 2 倍(连同它的尺度),该列必须**不变**。

    未除尺度的口径量到的是"脸在画面里多大",放大 2 倍值就跟着变(0.98 → 0.96);
    除了 `face_height` 之后,分子分母同步放大 ⟹ 值不动。
    红法:去掉除法 ⟹ 立刻红。
    """
    pts = {33: (0.40, 0.40), 133: (0.45, 0.40),
           362: (0.55, 0.40), 263: (0.60, 0.40),
           61: (0.45, 0.60), 291: (0.55, 0.62)}
    small = _f(pts, face_height=0.40).symmetry_score
    big = _f({i: (x * 2, y * 2) for i, (x, y) in pts.items()}, face_height=0.80).symmetry_score
    assert small == pytest.approx(big), (
        f"人脸放大 2 倍该让对称度不动:{small} vs {big} —— 变了说明量的是取景")


# ── iris 四列 ────────────────────────────────────────────────────────────────
def test_iris_columns_are_eye_relative_coordinates():
    """四列 = `(虹膜中心 − 眼中心) ÷ 眼宽`,**x、y 同除眼宽**(保持各向同性)。

    几何:左眼 33→133 横跨 0.10(中心 0.40,y=0.45);左虹膜 468-471 落在 (0.42,0.47)
      ⟹ `x=(0.42−0.40)/0.10 = +0.20`、`y=(0.47−0.45)/0.10 = +0.20`。
    右眼 362→263 中心 0.60;右虹膜 473-476 落在 (0.58,0.43) ⟹ 两列都是 **−0.20**。

    旧口径直接吐归一化图像坐标(0.42 / 0.47 / 0.58 / 0.43)⟹ 分岔。
    红法:去掉减眼中心或去掉除眼宽 ⟹ 立刻红。
    """
    f = _f({33: (0.35, 0.45), 133: (0.45, 0.45),
            362: (0.55, 0.45), 263: (0.65, 0.45),
            **{i: (0.42, 0.47) for i in (468, 469, 470, 471)},
            **{i: (0.58, 0.43) for i in (473, 474, 475, 476)}})
    assert f.left_iris_x == pytest.approx(+0.20), f.left_iris_x
    assert f.left_iris_y == pytest.approx(+0.20), f.left_iris_y
    assert f.right_iris_x == pytest.approx(-0.20), f.right_iris_x
    assert f.right_iris_y == pytest.approx(-0.20), f.right_iris_y


def test_iris_columns_are_invariant_to_where_the_face_sits_in_frame():
    """★ **取景代理病的判据**:人脸整体在画面里平移,四列必须**不变**。

    旧口径吐的是绝对归一化坐标 ⟹ 人往右挪 0.1,四列全跟着 +0.1(实测 `eye_contact`
    的封停理由正是「iris 图像坐标到画面中心距离,是取景代理」)。
    红法:去掉减眼中心 ⟹ 立刻红。
    """
    base = {33: (0.35, 0.45), 133: (0.45, 0.45),
            362: (0.55, 0.45), 263: (0.65, 0.45),
            **{i: (0.42, 0.47) for i in (468, 469, 470, 471)},
            **{i: (0.58, 0.43) for i in (473, 474, 475, 476)}}
    moved = {i: (x + 0.10, y + 0.07) for i, (x, y) in base.items()}
    a = _f(base)
    b = _f(moved)
    for col in ("left_iris_x", "left_iris_y", "right_iris_x", "right_iris_y"):
        assert getattr(a, col) == pytest.approx(getattr(b, col)), (
            f"{col} 随人脸在画面里的位置变了({getattr(a, col)} -> {getattr(b, col)})"
            f"—— 那说明它还在吐绝对坐标")


# ── face_scale ───────────────────────────────────────────────────────────────
def test_face_scale_is_the_same_number_the_other_columns_divide_by():
    """★ `face_scale` 必须**就是**那些列当分母用的那个数 —— 不许两处各算一遍同一个 dist。

    这条是本列的验收①。判据取法:**喂一个与 landmarks 算不出来的尺度**。
    `lm[1]`/`lm[152]` 之间的距离是 0.90,而传进去的 `face_height` 是 0.47 ——
    一个"各处自己重算一遍"的实现会吐 0.90;单一来源的实现吐 0.47。
    红法:`face_scale` 改成自己算一遍 `dist(lm[1], lm[152])` ⟹ 立刻红。
    """
    f = _f({1: (0.50, 0.00), 152: (0.50, 0.90)},
           face_width=0.31, face_height=0.47)
    assert f.face_scale == pytest.approx(0.47), (
        f"该吐传进来的尺度 0.47(与各列分母同源),实为 {f.face_scale} —— "
        f"0.90 说明它自己重算了一遍 dist(lm[1], lm[152])")
    assert f.face_scale != pytest.approx(0.90), "不许回落到自己重算的那份"


def test_face_scale_is_not_silently_defaulted_to_zero():
    """★ 反向那半:`AUFeatures` 的默认值是 0.0,而 0 是个**看着像测量值**的数。

    `calculate` 末尾那条"补充缺失字段(默认0)"的循环会把没被赋值的字段填成 0.0 ——
    新列一旦忘了赋值,落盘就是 0.0 而**不报错**。
    红法:把 `face_scale` 从 `calculate` 里去掉(只留 dataclass 字段)⟹ 立刻红。
    """
    f = _f({}, face_height=0.42)
    assert f.face_scale == pytest.approx(0.42), (
        f"face_scale 被填成了默认值 {f.face_scale} —— 忘了赋值(0.0 是个假值)")
