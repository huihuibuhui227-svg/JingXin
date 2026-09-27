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

⚠️ 2026-09-27 的两处补充(本文件末尾那两节):
  · 第三个参数由 `face_height` 改名为 **`face_outline_height`**(面部轮廓高 =
    `dist(lm[10], lm[152])`,前额顶→下巴;旧的是鼻尖→下巴)。下面若干 docstring 里出现的
    `face_height` 是**它的旧名或历史口径**(例如 au26 那条函数名 —— L0 行按名字引它,
    故**不改名**)。管线那一层的钉子见 `tests/test_face_outline_height.py`。
  · 新增 `au12_smile` 一节(A 档:基线由「开头 10 帧冻结均值」改为**全程口宽 p10**,
    并去掉截顶)。
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


def _f(points, face_width=0.40, face_outline_height=0.40):
    """⚠️ 第三个参数 2026-09-27 改名:`face_height`(鼻尖→下巴)→ `face_outline_height`
    (**面部轮廓高**,前额顶→下巴)。**它一直是"传进去的那个尺度"** —— `calculate` 不自己算,
    所以本文件这些用例喂的仍是**任意一个数**(0.40/0.80/0.42…),那正是它们要的:
    它们钉的是"某一列拿这个数当分母",不是"这个数怎么来的"(后者由
    `test_face_outline_height.py` 在管线那一层钉)。"""
    return AUFeatureCalculator().calculate(_lm(points), face_width, face_outline_height)


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
def test_au6_cheek_raise_divides_by_the_vertical_outline_span():
    """分母 = **传进来的那个面部尺度** —— 也就是 `video_pipeline` **只算一次**的那个
    **面部轮廓高** `dist(lm[10], lm[152])`(前额顶→下巴),不是眼距、不是鼻尖→下巴。

    为什么是它(L0 行 basis 写了完整依据,这里是它的机器钉子):**竖直**且**两端都在
    面部中轴轮廓上**。眼距是横向跨度、鼻尖→下巴的一端是突出的鼻尖 —— 两者都把
    头姿(yaw)带进比值。

    ⚠️ 2026-09-27 本条的**形状**随分母换端点一起改了:分母不再由 `calculate` 自己从
    landmarks 算(那是与管线**第二处算同一个 dist**),而是管线算一次传下来。所以这里
    对比的是「用传进来的数」vs「自己按 landmarks 重算一个」:
      几何:颊-眼两段各 0.10(和 0.20);**传进来的尺度 0.90**;
            眼距 `dist(lm[130], lm[359])` = 0.40;
            鼻尖→下巴 `dist(lm[1], lm[152])` = 0.80(刻意摆成 0.80)。
      · 用传进来的 0.90:1 − 0.20/(2×0.90) = **8/9 ≈ 0.8889**
      · 重算成眼距 0.40:1 − 0.20/(2×0.40) = **0.75**
      · 重算成鼻尖→下巴 0.80:1 − 0.20/(2×0.80) = **0.875**
    ⟹ 三个数互不相等,任一处改回旧口径立刻红。
    """
    f = _f({205: (0.40, 0.50), 145: (0.40, 0.60),     # 左颊-眼下缘:0.10
            425: (0.60, 0.50), 374: (0.60, 0.60),     # 右颊-眼下缘:0.10
            130: (0.30, 0.40), 359: (0.70, 0.40),     # 眼距:0.40
            1: (0.50, 0.15), 152: (0.50, 0.95)},      # 鼻尖→下巴:0.80
           face_outline_height=0.90)
    assert f.au6_cheek_raise == pytest.approx(8 / 9), (
        f"该是 1 − 0.20/(2×传进来的 0.90) = 0.8889,实为 {f.au6_cheek_raise} —— "
        f"0.75 说明它自己重算成了眼距、0.875 说明重算成了鼻尖→下巴")


# ⚠️ 原 `test_au6_cheek_raise_is_invariant_to_horizontal_foreshortening`(横向压缩不改变
#    au6)**已移到 `tests/test_face_outline_height.py`**,改名
#    `test_the_scale_and_au6_are_invariant_to_horizontal_foreshortening`。
#    为什么移:分母现在**不由本层的 landmarks 算**(见上一条的说明)⟹ 在 calculator 层
#    喂一个固定的尺度,"压缩横向"对分母没有任何影响,那条判据在这里**变成了恒真**;
#    它要钉的是「分母是从 landmarks 导出的**竖直**跨度」,而那件事发生在管线里。


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
    """`au23` = `dist(lm[0], lm[13]) ÷ 面部尺度`(**参数名 2026-09-27 改为 `face_outline_height`**;下面几条 docstring 里出现的 `face_height` 都是它的旧名)—— **不再**吃静息值。

    几何:上唇上缘 (0.50,0.40) → 唇缝 (0.50,0.50),相距 0.10;传进来的尺度 0.40 ⟹ **0.25**。
    旧口径在**第一帧**把静息值取成本帧 ⟹ `1 − 1/1` = **0.0**。⟹ 分岔。

    红法:改回 `1 − 当前唇厚/静息唇厚` ⟹ 第一帧变 0.0,立刻红。
    """
    f = _f({0: (0.50, 0.40), 13: (0.50, 0.50)})
    assert f.au23_lip_compression == pytest.approx(0.25), (
        f"该是 唇红厚度÷尺度 = 0.25,实为 {f.au23_lip_compression} —— 0.0 说明还在用静息值")


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


# ── au25_mouth_open ──────────────────────────────────────────────────────────
def test_au25_mouth_open_is_the_raw_inner_lip_ratio():
    """`au25` = `dist(lm[13], lm[14]) ÷ 面部尺度`(**唇内缘**两点),不吃静息值。

    几何:上唇内缘 (0.50,0.40) → 下唇内缘 (0.50,0.50),相距 0.10;传进来的尺度 0.40 ⟹ **0.25**。
    改前口径在**第一帧**把静息值取成本帧 ⟹ `max(0, (唇外缘开合 − 它自己)/它自己)` = **0.0**;
    而且它读的是**唇外缘** `lm[0]`/`lm[17]`(这里给 0.30→0.55,相距 0.25)⟹ 也是 0.0。

    红法:改回减基线、或改回读唇外缘 ⟹ 立刻红(0.25 → 0.0)。
    """
    f = _f({0: (0.50, 0.30), 17: (0.50, 0.55),        # 唇外缘(改前口径读的点):0.25
            13: (0.50, 0.40), 14: (0.50, 0.50)})      # 唇内缘(本列该读的):0.10
    assert f.au25_mouth_open == pytest.approx(0.25), (
        f"该是 唇内缘开合÷尺度 = 0.25,实为 {f.au25_mouth_open} —— "
        f"0.0 说明还在减静息基线(或还在读唇外缘 lm[0]/lm[17])")


def test_au25_mouth_open_carries_no_rest_baseline():
    """★ L0 只吐**原始**比值:**同一帧**在会话里的位置不该改变它的值。

    喂三帧:张口 → 闭口 → **再喂与第一帧逐格相同的那一帧**。有无基线在这里分岔:
      · 无基线(本列该有的形态):第 1 帧与第 3 帧的值**逐位相等**;
      · 有基线(改前):第 3 帧的基线已被第 2 帧改过 ⟹ 值不同。
    红法:把静息基线加回来 ⟹ 立刻红。
    """
    calc = AUFeatureCalculator()
    open_pts = {13: (0.50, 0.35), 14: (0.50, 0.55), 0: (0.50, 0.30), 17: (0.50, 0.65)}
    shut_pts = {13: (0.50, 0.45), 14: (0.50, 0.47), 0: (0.50, 0.42), 17: (0.50, 0.50)}
    first = calc.calculate(_lm(open_pts), 0.40, 0.40).au25_mouth_open
    calc.calculate(_lm(shut_pts), 0.40, 0.40)
    again = calc.calculate(_lm(open_pts), 0.40, 0.40).au25_mouth_open
    assert first == pytest.approx(again), (
        f"同一帧两次的值必须相同(第一帧 {first} vs 第三帧 {again})—— "
        f"不等说明 L0 里还在做基线减法(那是 L1 的活)")


def test_au25_mouth_open_is_not_capped_at_one():
    """「只吐**原始比值**」的另一半:不截顶 —— 截顶把幅度藏起来(饱和交给 L1 的分位映射)。

    几何:唇内缘相距 0.60、传进来的尺度 0.40 ⟹ 比值 **1.5**(> 1)。
    红法:把上限截回 1.0 ⟹ 立刻红。
    """
    f = _f({13: (0.50, 0.20), 14: (0.50, 0.80)})
    assert f.au25_mouth_open == pytest.approx(1.5), (
        f"该是 0.60/0.40 = 1.5(不截顶),实为 {f.au25_mouth_open} —— 1.0 说明还截了顶")


# ── au26_jaw_drop ────────────────────────────────────────────────────────────
def test_au26_jaw_drop_divides_by_face_height_not_a_tenth_of_it():
    """`au26` = `|lm[152].y − lm[13].y| ÷ 面部尺度`(**不减基线、不截顶**)。

    几何:下巴 y=0.90、上唇内缘 y=0.40 ⟹ 竖直距 0.50;传进来的尺度 0.40 ⟹ **1.25**。
      · 改前口径:减开头基线 ⟹ 第一帧 **0.0**;
      · 若保留 `÷ (0.1 × face_height)`:比值 12.5,被截顶成 **1.0**。
    ⟹ 三个数互不相等。红法:任一改回 ⟹ 立刻红。
    """
    f = _f({152: (0.50, 0.90), 13: (0.50, 0.40)})
    assert f.au26_jaw_drop == pytest.approx(1.25), (
        f"该是 0.50/0.40 = 1.25,实为 {f.au26_jaw_drop} —— "
        f"0.0 说明还在减基线,1.0 说明分母还是 0.1×face_height(或还截顶)")


def test_au26_jaw_drop_carries_no_rest_baseline():
    """★ 同 `au25`:**同一帧**的值不该因为它出现在会话第几帧而变。

    红法:把开头 10 帧的静息基线加回来 ⟹ 立刻红。
    """
    calc = AUFeatureCalculator()
    wide = {152: (0.50, 0.95), 13: (0.50, 0.40)}
    shut = {152: (0.50, 0.70), 13: (0.50, 0.42)}
    first = calc.calculate(_lm(wide), 0.40, 0.40).au26_jaw_drop
    calc.calculate(_lm(shut), 0.40, 0.40)
    again = calc.calculate(_lm(wide), 0.40, 0.40).au26_jaw_drop
    assert first == pytest.approx(again), (
        f"同一帧两次的值必须相同(第一帧 {first} vs 第三帧 {again})—— "
        f"不等说明 L0 里还在做基线减法(那是 L1 的活)")


# ── symmetry_score ───────────────────────────────────────────────────────────
def test_symmetry_score_divides_by_face_height():
    """`symmetry_score` = `1 − (|Δ眼中心y| + |Δ嘴角y|) ÷ 面部尺度`。

    几何:两眼中心同高(Δ=0)、嘴角 y 差 0.02、传进来的尺度 0.40 ⟹ `1 − 0.02/0.40` = **0.95**。
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
    除了那个尺度之后,分子分母同步放大 ⟹ 值不动。
    红法:去掉除法 ⟹ 立刻红。
    """
    pts = {33: (0.40, 0.40), 133: (0.45, 0.40),
           362: (0.55, 0.40), 263: (0.60, 0.40),
           61: (0.45, 0.60), 291: (0.55, 0.62)}
    small = _f(pts, face_outline_height=0.40).symmetry_score
    big = _f({i: (x * 2, y * 2) for i, (x, y) in pts.items()}, face_outline_height=0.80).symmetry_score
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
    landmarks 里 `lm[1]`→`lm[152]` 与 `lm[10]`→`lm[152]` 都在 0.90 上下,
    而**传进去的**尺度(2026-09-27 起参数名 `face_outline_height`)是 0.47 ——
    一个"各处自己重算一遍"的实现会吐 0.90;单一来源的实现吐 0.47。
    红法:`face_scale` 改成自己重算一遍 `dist(lm[10], lm[152])` ⟹ 立刻红。
    """
    f = _f({1: (0.50, 0.00), 152: (0.50, 0.90)},
           face_width=0.31, face_outline_height=0.47)
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
    f = _f({}, face_outline_height=0.42)
    assert f.face_scale == pytest.approx(0.42), (
        f"face_scale 被填成了默认值 {f.face_scale} —— 忘了赋值(0.0 是个假值)")


# ── 审计脚本与活列的口径一致性 ───────────────────────────────────────────────
def test_the_audit_scripts_head_yaw_is_the_column_itself():
    """★ 纪律 1 的钉子:`experiments/au6_denominator_audit.py` 里那个 `_head_yaw`
    必须**逐字等于**本仓 `head_yaw` 列的口径 —— 否则那份"候选分母 vs yaw"的对照表
    量的是**另一个 yaw**,结论整张作废(本仓已 5 次栽在"验证跑错了对象")。

    红法:改 `_head_yaw` 的任一处(换成 lm[33]、去掉除 face_width、`ear_mid` 用别的点)⟹ 红。
    """
    import importlib
    audit = importlib.import_module("experiments.au6_denominator_audit")
    from face_expression.core.feature_extraction.au_calculator import HeadPoseExtractor

    lm = _lm({1: (0.55, 0.45), 234: (0.30, 0.50), 455: (0.70, 0.50), 152: (0.50, 0.85)})
    column = HeadPoseExtractor.extract(lm, face_width=0.40, face_outline_height=0.35)["head_yaw"]
    assert audit._head_yaw(lm) == pytest.approx(column), (
        f"审计脚本的 head_yaw {audit._head_yaw(lm)} 与列的 {column} 不等 —— "
        f"那张分母对照表量的是另一个头姿量,不能拿它下结论")


# ── au12_smile(A 档:基线改「全程 p10」;2026-09-27)─────────────────────────
def _width_frame(width):
    """一帧只需要口宽:嘴角 lm[61]/lm[291] 拉开 `width`,y 同高。"""
    return {61: (0.50 - width / 2, 0.60), 291: (0.50 + width / 2, 0.60)}


def test_au12_smile_baseline_is_the_whole_session_p10_not_the_opening_window():
    """★ 基线 = **全程口宽 p10** 的因果近似,不是「开头 10 帧冻结均值」。

    造法:前十帧口宽 0.20 → 随后十帧 0.10 → 探针帧 0.15。两个口径在这一帧上必然分岔:
      · **冻结基线**(改前):baseline 停在 0.20 ⟹ `max(0, (0.15−0.20)/0.20)` = **0.0**,
        而且从这里往后**恒 0**(正是判据②点名的「基线跳变后比值恒定」形态);
      · **全程 p10**(本列该有的):21 个样本里第 10 百分位 = **0.10**
        ⟹ `(0.15−0.10)/0.10` = **0.5**。
    """
    calc = AUFeatureCalculator()
    for _ in range(10):
        calc.calculate(_lm(_width_frame(0.20)), 0.40, 0.40)
    for _ in range(10):
        calc.calculate(_lm(_width_frame(0.10)), 0.40, 0.40)
    got = calc.calculate(_lm(_width_frame(0.15)), 0.40, 0.40).au12_smile
    assert got == pytest.approx(0.5, abs=1e-4), (
        f"该是 (0.15−0.10)/0.10 = 0.5,实为 {got} —— "
        f"0.0 说明基线还是「开头 10 帧的冻结均值」(它停在 0.20)")


def test_au12_smile_baseline_is_the_low_decile_and_the_ratio_is_not_capped():
    """★ 两件事一起钉:基线是**低分位**(不是均值/中位数),并且**不截顶**。

    十帧口宽 = [0.10, 0.10, 0.20×7, 0.30] ⟹ p10 = **0.10**、中位数 = 0.20、均值 = 0.19;
    探针帧 0.25:
      · p10 基线(本列该有的):(0.25−0.10)/0.10 = **1.5**(> 1 ⟹ 同时证明没截顶);
      · 中位数基线:0.25;· 均值基线(改前的冻结均值):0.316;
      · 截顶版(p10 但夹到 1.0):**1.0**。
    四个数互不相等。
    ⚠️ 容差取 `abs=1e-4`:分母里那个 `+1e-6` 是**除零守卫**(不是门限),它让值偏 5e-6。
    """
    calc = AUFeatureCalculator()
    for width in [0.10, 0.10] + [0.20] * 7 + [0.30]:
        calc.calculate(_lm(_width_frame(width)), 0.40, 0.40)
    got = calc.calculate(_lm(_width_frame(0.25)), 0.40, 0.40).au12_smile
    assert got == pytest.approx(1.5, abs=1e-4), (
        f"该是 (0.25−0.10)/0.10 = 1.5,实为 {got} —— "
        f"1.0 说明还截着顶、0.25 说明基线是中位数、0.316 说明基线是均值")


def test_au20_lip_stretcher_stays_bit_identical_to_au12():
    """⚠️ **回归守卫**(它改前改后都该绿 —— 与"看着它失败"那批不同,这条守的是**不许分岔**)。

    `au20_lip_stretcher` 与 `au12_smile` 是**同一行代码**的复制粘贴
    (§4.1:171;`l0_columns.json` 的 `legacy_allowlist` 里登记着 `why` = 「与 au12 逐字符
    相同的同一行代码」、`planned` = 「M3.2 删列」)。本次只改 au12 的基线口径 ——
    若 au20 留在旧基线上,那两句登记**当场变成假话**,而且是**静默**的。
    所以两者必须继续读同一个基线、同样不截顶。

    红法:把 au20 的基线改回冻结均值(或给它单独夹一刀)⟹ 立刻红。
    """
    calc = AUFeatureCalculator()
    for width in (0.20, 0.22, 0.10, 0.30, 0.18):
        f = calc.calculate(_lm(_width_frame(width)), 0.40, 0.40)
        assert f.au20_lip_stretcher == f.au12_smile, (
            f"口宽 {width} 时 au20 {f.au20_lip_stretcher} != au12 {f.au12_smile} —— "
            f"两列的分岔是静默的,而 allowlist 里「逐位相等」那句会变成假话")
