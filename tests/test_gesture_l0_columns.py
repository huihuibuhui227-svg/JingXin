# tests/test_gesture_l0_columns.py
"""M3.3 手势 19 列的 L0 逐列重构(B 档)。

这 19 列的病根只有一个:**没有分母**(外加 `shrug_level` 的基线冻在开头 30 帧)。
`l0_columns.json` 那 19 行的 `definition` / `normalization` / `acceptance` 是判据,
本文件把判据里**可机检**的那部分钉住:

| 族 | 判据(L0 行) | 本文件怎么分岔 |
|---|---|---|
| 10 个画面坐标 jitter | 分子 ÷ 肩宽(窗内中位数)÷ 窗内真实秒数 | 时间轴 ×k ⟹ 值 ÷k;取景 ×k(点与肩宽同乘 k)⟹ 不变;缺肩宽 ⟹ **空** |
| 6 个米制 jitter | 分子 ÷ 窗内真实秒数,**不**除肩宽 | 时间轴 ×k ⟹ 值 ÷k;传不传肩宽都一样 |
| 2 个 hand_spread | 分子(逐字不变)÷ 掌长 | 取景 ×k ⟹ 不变;指尖在分子里、掌长在分母里 |
| 1 个 shrug_level | 基线 = 全程中位数(因果近似)÷ 肩宽,**不截顶** | 冻结基线后半场恒 1.0 而中位数基线会归零;取景 ×k ⟹ 不变;抬过肩宽 ⟹ >1 |

**为什么每条都构造"新旧口径必然分岔"的几何**:本仓栽过 5 次「验证跑错了对象」——
"值在合理范围内"这类判据新旧都能过,等于没验。分岔不出来的(例如"分子逐字不变"
这种**本来就不该变**的)写明它是**钉子**,不当分岔判据用。

合成 landmarks 的约定:33 点(姿态)/ 21 点(手)先铺成**同一个默认点**,再逐点覆盖;
凡本列不读的点留在默认点上也不影响本列的值 —— 这本身就是"它只读它该读的那几个点"的证明。
⚠️ 默认点**不取原点**(踩过别处的坑):肩宽/掌长这类分母会变成 0 ⟹ 红法是"除零崩了"
而不是"口径错了",那条测试就什么都没证明。默认点摆在 (0.95, 0.95) 且在断言用到的
坐标之外。
"""
import asyncio
import csv
import importlib
import types

import pytest

import media_retention                                                # noqa: E402
gesture_app = importlib.import_module("gesture_analysis.api.app")     # noqa: E402

from gesture_analysis.core.analysis.arm_analyzer import ArmAnalyzer               # noqa: E402
from gesture_analysis.core.analysis.hand_analyzer import HandAnalyzer             # noqa: E402
from gesture_analysis.core.analysis.shoulder_analyzer import ShoulderAnalyzer     # noqa: E402
from gesture_analysis.core.analysis.upper_body_analyzer import UpperBodyAnalyzer  # noqa: E402

_FAR = (0.95, 0.95)          # 「本列不读的点」的占位;不等于原点


class _Pt:
    """mediapipe 的 landmarks:分析器只读 `.x` / `.y`(姿态那批不看 `visibility`)。"""
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


def _pose(overrides=None, k=1.0):
    """33 点姿态。`k` = 取景缩放(绕原点:任意两点距离同乘 k)。"""
    pts = [_Pt(*_FAR) for _ in range(33)]
    for i, p in (overrides or {}).items():
        pts[i] = _Pt(p[0] * k, p[1] * k)
    return pts


def _hand(overrides=None, k=1.0):
    """21 点的手。默认点同样避开原点(否则掌长会是 0)。"""
    pts = [_Pt(*_FAR) for _ in range(21)]
    for i, p in (overrides or {}).items():
        pts[i] = _Pt(p[0] * k, p[1] * k)
    return pts


# ── 手的标准几何 ─────────────────────────────────────────────────────────────
# 腕 lm[0]=(0.30,0.50)、中指 MCP lm[9]=(0.40,0.50) ⟹ **掌长 0.10**。
# 五个指尖到腕:0.20 / 0.20 / 0.30 / 0.40 / 0.40 ⟹ 均值 **0.30**。
# ⟹ 张开度 = 0.30 ÷ 0.10 = **3.0**(旧口径给 0.30 —— 差一个量级,必然分岔)。
_HAND = {0: (0.30, 0.50), 9: (0.40, 0.50),
         4: (0.30, 0.70), 8: (0.50, 0.50), 12: (0.60, 0.50),
         16: (0.30, 0.10), 20: (0.70, 0.50)}

# ── 姿态的标准几何 ───────────────────────────────────────────────────────────
# 双肩相距 0.40(这就是 `shoulder_width`),躯干中点在两者中间,鼻/肘/腕/髋各自摆开。
_POSE = {0: (0.50, 0.20),                       # nose(上半身分析器的"头")
         11: (0.30, 0.50), 12: (0.70, 0.50),    # 双肩 ⟹ 肩宽 0.40
         13: (0.30, 0.70), 14: (0.70, 0.70),    # 双肘
         15: (0.30, 0.90), 16: (0.70, 0.90),    # 双腕
         23: (0.35, 0.95), 24: (0.65, 0.95)}    # 双髋
_SH = 0.40                                       # _POSE 里那对肩的距离


def _wiggle(i: int, amp: float = 0.01):
    """第 i 帧的位移:确定性锯齿(不是随机),使窗内逐轴标准差 ≠ 0。"""
    return amp * (0, 0.5, 1.0)[i % 3], 0.0


def _drive(make, kind="pose", *, k=1.0, dt=1000, n=12, shoulder_width=_SH,
           shoulder_from=0, hand=None, pose=None):
    """建一个分析器、喂 n 帧、交**最后一帧**的结果字典。

    `dt` = 相邻帧的毫秒间隔;`shoulder_width=None` = 这一帧没有尺度(缺肩 / 无姿态);
    `shoulder_from` = 从第几帧起才有尺度(默认 0 = 一直都有)。
    """
    a = make()
    for i in range(n):
        # ⚠️ 位移要加在**缩放之前**:取景 ×k 时整幅画面(含运动幅度)一起 ×k。
        #    加在缩放之后 ⟹ 分子不随 k 变,而分母 ×k ⟹ 值 ÷k,那条不变性断言会
        #    把"构造错了"读成"口径错了"(第一版就是这么写错的)。
        dx, dy = _wiggle(i)
        base = _pose(pose or _POSE) if kind == "pose" else _hand(hand or _HAND)
        lm = [_Pt((p.x + dx) * k, (p.y + dy) * k) for p in base]
        a.update(lm, timestamp_ms=i * dt,
                 shoulder_width=shoulder_width if i >= shoulder_from else None)
    return a.get_results()


# ══════════════════════════════════════════════════════════════════════════════
# 10 个画面坐标 jitter:÷ 肩宽(窗内中位数)÷ 窗内真实秒数
# ══════════════════════════════════════════════════════════════════════════════
SCREEN_JITTER = [
    ("left_hand_jitter", lambda: HandAnalyzer(hand_id=0), "jitter", "hand"),
    ("right_hand_jitter", lambda: HandAnalyzer(hand_id=1), "jitter", "hand"),
    ("left_wrist_jitter", lambda: ArmAnalyzer(arm_id="left"), "wrist_jitter", "pose"),
    ("left_elbow_jitter", lambda: ArmAnalyzer(arm_id="left"), "elbow_jitter", "pose"),
    ("right_wrist_jitter", lambda: ArmAnalyzer(arm_id="right"), "wrist_jitter", "pose"),
    ("right_elbow_jitter", lambda: ArmAnalyzer(arm_id="right"), "elbow_jitter", "pose"),
    ("left_shoulder_jitter", lambda: ShoulderAnalyzer(), "left_jitter", "pose"),
    ("right_shoulder_jitter", lambda: ShoulderAnalyzer(), "right_jitter", "pose"),
    ("head_jitter", lambda: UpperBodyAnalyzer(), "head_jitter", "pose"),
    ("torso_jitter", lambda: UpperBodyAnalyzer(), "torso_jitter", "pose"),
]

METRIC_JITTER = [
    ("left_wrist_jitter_world",
     lambda: ArmAnalyzer(arm_id="left", metric=True), "wrist_jitter", "pose"),
    ("left_elbow_jitter_world",
     lambda: ArmAnalyzer(arm_id="left", metric=True), "elbow_jitter", "pose"),
    ("right_wrist_jitter_world",
     lambda: ArmAnalyzer(arm_id="right", metric=True), "wrist_jitter", "pose"),
    ("right_elbow_jitter_world",
     lambda: ArmAnalyzer(arm_id="right", metric=True), "elbow_jitter", "pose"),
    ("left_shoulder_jitter_world",
     lambda: ShoulderAnalyzer(metric=True), "left_jitter", "pose"),
    ("right_shoulder_jitter_world",
     lambda: ShoulderAnalyzer(metric=True), "right_jitter", "pose"),
]


@pytest.mark.parametrize("name,make,key,kind", SCREEN_JITTER)
def test_screen_jitter_is_a_rate_not_a_window_std(name, make, key, kind):
    """★ 时间轴 ×k ⟹ 值 ÷k。`definition` 的第二个分母是**窗内真实秒数**。

    同一串位置样本、只把时间戳整体乘 2(等价于"这两帧之间过去了 2 倍的时间"):
      · 新口径 = 分子 ÷ 秒数 ⟹ 减半;
      · 旧口径(只算 `np.std`,不看时间)⟹ **一毫不变**。
    ⟹ 断言 `v(1 s 步长) == 2 × v(2 s 步长)`;旧口径给 1.0 倍,必然红。

    为什么这条是**帧率污染**的正身:窗长是 30 **帧**,帧率一变、窗跨的**时间**就变;
    "÷ 窗内真实秒数"把那个时间**显式写进单位**(肩宽/秒),而不是让它藏在值里。
    ⚠️ 为什么是"窗内真实秒数"而不是 `30 ÷ fps`:后者等于假设帧率恒定,而帧率正是被
    污染的那个量(L0 行 basis ②)—— 本测试改的**只是时间戳**,正好绕开那个假设。
    """
    slow = _drive(make, kind, dt=1000)[key]
    fast = _drive(make, kind, dt=2000)[key]
    assert slow is not None and fast is not None, f"{name} 交空了:{slow!r} / {fast!r}"
    assert slow == pytest.approx(2 * fast, rel=0.02), (
        f"{name}:时间轴 ×2 时该列该**减半**(÷ 窗内秒数),实测 {slow!r} vs {fast!r} —— "
        f"两者相等说明那一层分母没做(旧口径)")


@pytest.mark.parametrize("name,make,key,kind", SCREEN_JITTER)
def test_screen_jitter_is_invariant_under_a_framing_zoom(name, make, key, kind):
    """★ 取景缩放不变(acceptance ②):人离镜头近一倍,该列**不变**。

    构造:整份 landmarks ×2(画面里大一倍)⟹ 分子(位置标准差)也 ×2;
    肩宽同时 ×2 ⟹ 两层一起变、比值不变。旧口径只跟着分子走 ⟹ **×2**,必然红。

    为什么这条就是"取景代理"的定义:旧值随取景变 ⟹ 它量的是"手/肩在画面里多大",
    不是"抖得多厉害"(见各 L0 行的 ⚠️ 段)。
    """
    a = _drive(make, kind)[key]
    b = _drive(make, kind, k=2.0, shoulder_width=_SH * 2)[key]
    assert a is not None and b is not None, f"{name} 交空了:{a!r} / {b!r}"
    assert a == pytest.approx(b, rel=0.05), (
        f"{name}:整份坐标 ×2(肩宽同乘 2)后该列该**不变**,实测 {a!r} → {b!r} —— "
        f"×2 说明分母里没有肩宽")


@pytest.mark.parametrize("name,make,key,kind", SCREEN_JITTER)
def test_screen_jitter_is_empty_when_the_shoulder_scale_is_missing(name, make, key, kind):
    """★ 缺肩宽 ⟹ **空**(`None`),不是"未归一化的分子顶上"。

    没有分母就没有这个**率**:交别的量上去 = 把单位从「肩宽/秒」静默换成
    「归一化图像单位」,而下游从这一行看不出单位换了(本项目禁止的形态)。

    红法:把缺失分支写成 `return std / span`(漏掉肩宽那层)⟹ 本条红。
    """
    exact = _drive(make, kind, shoulder_width=_SH)[key]
    missing = _drive(make, kind, shoulder_width=None)[key]
    assert exact is not None, f"{name}:前提不成立(有肩宽时它该有值)"
    assert missing is None, (
        f"{name}:缺肩宽时交的是 {missing!r} —— 该交**空**:没有分母就没有这个率")


@pytest.mark.parametrize("name,make,key,kind", SCREEN_JITTER)
def test_screen_jitter_window_median_ignores_frames_without_a_scale(name, make, key, kind):
    """★ 分母取的是**窗内中位数**,不是"这一帧的肩宽"、也不是"缺一帧就整窗作废"。

    构造:12 帧里前 6 帧没有肩宽、后 6 帧肩宽 0.40 ⟹ 窗内中位数仍是 0.40,
    所以结果该与"12 帧全有肩宽"**逐字相同**。
    """
    a = _drive(make, kind, shoulder_width=_SH)[key]
    b = _drive(make, kind, shoulder_width=_SH, shoulder_from=6)[key]
    assert a is not None and b is not None, f"{name} 交空了:{a!r} / {b!r}"
    assert a == pytest.approx(b, rel=1e-12), (
        f"{name}:窗内有一半帧没有肩宽时,分母该取**有值那些帧的中位数**"
        f"({a!r} vs {b!r})—— 不等说明用的是「这一帧」或「缺一帧就作废」")


# ══════════════════════════════════════════════════════════════════════════════
# 6 个米制 jitter:÷ 窗内真实秒数,**不**除肩宽
# ══════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("name,make,key,kind", METRIC_JITTER)
def test_metric_jitter_is_a_rate_too(name, make, key, kind):
    """★ 米制那批同样 ÷ 窗内真实秒数(`normalization` 栏:「÷ 窗内真实秒数」)。

    旧口径不看时间 ⟹ 时间轴 ×2 时**一毫不变** ⟹ 本条红
    (与画面坐标那 10 条同一形态)。
    """
    slow = _drive(make, kind, dt=1000)[key]
    fast = _drive(make, kind, dt=2000)[key]
    assert slow is not None and fast is not None, f"{name} 交空了:{slow!r} / {fast!r}"
    assert slow == pytest.approx(2 * fast, rel=0.02), (
        f"{name}:时间轴 ×2 时该减半,实测 {slow!r} vs {fast!r}")


@pytest.mark.parametrize("name,make,key,kind", METRIC_JITTER)
def test_metric_jitter_does_not_divide_by_shoulder_width(name, make, key, kind):
    """钉子(改前改后都该绿):米制那批**不**除肩宽 —— 米制已是解剖尺度。

    传 `shoulder_width=0.40` 与传 `None` 必须逐字相同。红法:哪天有人把画面坐标那层
    分母也套到米制上(表里那 6 行的 `normalization` 明写「**不**除肩宽」)⟹ 红。
    """
    with_sh = _drive(make, kind, shoulder_width=_SH)[key]
    without = _drive(make, kind, shoulder_width=None)[key]
    assert with_sh is not None and without is not None, f"{name} 交空了"
    assert with_sh == pytest.approx(without, rel=1e-12), (
        f"{name}:肩宽不该进米制那批的分母({with_sh!r} vs {without!r})")


@pytest.mark.parametrize("name,make,key,kind", SCREEN_JITTER + METRIC_JITTER)
def test_jitter_warmup_still_returns_zero_for_the_first_frames(name, make, key, kind):
    """钉子:**窗还没攒够 10 帧时仍交 `0.0`** —— 那是**现行**行为,本批没动它。

    为什么钉住:本批改的是分母,不是"窗没攒够怎么办";顺手把它改成 `None` 会让每场
    会话前 9 行的含义一起变(那是另一次口径决定,要先在表里立字据)。
    实测(3 场正式素材):每列前 9 行都是 `0.0`,改前改后一致。
    """
    assert _drive(make, kind, n=9)[key] == 0.0, (
        f"{name}:9 帧时该交 0.0(现行行为),交别的说明热身那条分支被顺手改了")


# ══════════════════════════════════════════════════════════════════════════════
# 2 个 hand_spread:分子逐字不变,只加 ÷ 掌长
# ══════════════════════════════════════════════════════════════════════════════
SPREAD = [("left_hand_spread", lambda: HandAnalyzer(hand_id=0)),
          ("right_hand_spread", lambda: HandAnalyzer(hand_id=1))]


@pytest.mark.parametrize("name,make", SPREAD)
def test_hand_spread_divides_by_palm_length(name, make):
    """★ `mean(dist(指尖, lm[0])) ÷ dist(lm[0], lm[9])`。

    `_HAND` 那份几何:指尖到腕的均值 **0.30**、掌长 **0.10** ⟹ 新口径 **3.0**;
    旧口径(不除任何尺度)给 **0.30** —— 差一个量级,必然分岔。
    """
    v = _drive(make, "hand")["spread"]
    assert v == pytest.approx(3.0, rel=1e-6), (
        f"{name}:该是 0.30 ÷ 0.10 = 3.0,实为 {v!r} —— 0.30 说明分母没做(旧口径)")


@pytest.mark.parametrize("name,make", SPREAD)
def test_hand_spread_is_invariant_under_a_framing_zoom(name, make):
    """★ 手在画面里大/小一倍,该列**不变**(acceptance ①)。

    整只手 ×2:分子(指尖到腕)与分母(掌长)**同乘 2** ⟹ 比值不变;
    旧口径 **×2**(它量的是"手在画面里多大")⟹ 本条红。
    """
    a = _drive(make, "hand")["spread"]
    b = _drive(make, "hand", k=2.0)["spread"]
    assert a == pytest.approx(b, rel=1e-9), (
        f"{name}:整只手 ×2 后该列该不变,实测 {a!r} → {b!r}")


@pytest.mark.parametrize("name,make", SPREAD)
def test_hand_spread_molecule_is_still_the_five_tips_to_the_wrist(name, make):
    """钉子(改前改后都该绿):分子**逐字**是 `mean(dist(指尖{4,8,12,16,20}, lm[0]))`。

    L0 行 basis 明写「分子里的指尖集合与现行实现逐字一致,只加分母 —— 一次只改一处」。
    红法(都是"顺手把分子也改了"):
      · 把拇指 lm[4] 从指尖集合里去掉 ⟹ 第一段红;
      · 把参考点从 lm[0](腕)换成 lm[9] ⟹ 第一段也红;
      · 让某个不相干的点进分子 ⟹ 第二段红。
    """
    base = _drive(make, "hand")["spread"]
    far_thumb = _drive(make, "hand", hand={**_HAND, 4: (0.30, 0.90)})["spread"]
    assert far_thumb > base, (
        f"{name}:把拇指指尖挪远,张开度没变大({base!r} → {far_thumb!r})—— "
        f"说明 lm[4] 不在分子里了")
    # 移动一个**既不是指尖、也不是 lm[0]/lm[9]** 的点 ⟹ 分子分母都不该动
    other = _drive(make, "hand", hand={**_HAND, 6: (0.10, 0.90)})["spread"]
    assert other == pytest.approx(base, rel=1e-12), (
        f"{name}:挪一个与分子分母都无关的点却改了值({base!r} → {other!r})")


# ══════════════════════════════════════════════════════════════════════════════
# shrug_level:基线 = 全程中位数(因果近似),分母 = 肩宽,不截顶
# ══════════════════════════════════════════════════════════════════════════════
# 图像坐标里 **y 越小 = 抬得越高**;数值取整数好手算:耸起 d 时 = d ÷ 肩宽。
_Y0 = 0.50          # 低位(基线所在)
_SH2 = 0.20         # 这一族的肩宽


def _shrug_series(seq, shoulder_width=_SH2, k=1.0):
    """`seq` = 每帧的双肩 y(直接给 y,好构造漂移/阶跃);交**逐帧**的 `shrug_level`。"""
    a = ShoulderAnalyzer()
    out = []
    for i, y in enumerate(seq):
        a.update(_pose({**_POSE, 11: (0.30, y), 12: (0.70, y)}, k=k),
                 timestamp_ms=i * 1000,
                 shoulder_width=None if shoulder_width is None else shoulder_width * k)
        out.append(a.get_results()["shrug_level"])
    return out


def test_shrug_baseline_is_the_running_median_not_the_first_thirty_frames():
    """★ 基线不是"开头 30 帧冻住",而是**截至当前的全程中位数**(`basis` ①)。

    序列:前 60 帧 y=0.50(低位),后 180 帧 y=0.30(抬起 = 抬高 0.20 = 一个肩宽)。
      · 旧口径:基线冻在 ~0.50 ⟹ 从第 31 帧起**恒 1.0**,再也不回来;
      · 新口径:抬起的帧成为**多数**之后,基线跟到 0.30 ⟹ 值**回到 0**。
    ⟹ 第 150 帧:旧口径 1.0、新口径 **0.0**。

    为什么这条是本列的正身:§4.2 依据栏点名的病是「基准线取开头 30 帧 → **时间漂移量**」
    —— 旧口径下这一列**成了时间的函数**(后半场恒 1.0),与"耸没耸肩"脱钩。
    """
    vals = _shrug_series([_Y0] * 60 + [_Y0 - _SH2] * 180)
    assert vals[60] == pytest.approx(1.0, rel=1e-6), (
        f"第 61 帧(抬起仍是窗内少数)该是满幅 1.0,实为 {vals[60]!r}")
    assert vals[149] == pytest.approx(0.0, abs=1e-12), (
        f"第 150 帧该回到 0 —— 抬起的帧已是**多数**、基线跟到了 0.30,当前帧不再高于基线;"
        f"实为 {vals[149]!r}(=1.0 说明基线还冻在开头那 30 帧)")


def test_shrug_is_invariant_under_a_framing_zoom():
    """★ 取景缩放不变(acceptance ②):分子(竖直位移)与分母(肩宽)同乘 k。

    整份姿态 ×2(肩宽 0.20 → 0.40)、抬起的位移也从 0.10 变成 0.20 ⟹ 值不变。
    旧口径除的是裸常量 0.1 ⟹ **×2**,本条红。
    """
    seq = [_Y0] * 20 + [_Y0 - 0.10] * 20
    a = _shrug_series(seq, k=1.0)[-1]
    b = _shrug_series(seq, k=2.0)[-1]
    assert a == pytest.approx(b, rel=0.05), (
        f"整份姿态 ×2(肩宽同乘 2)后该列该不变:{a!r} vs {b!r}")


def test_shrug_is_not_capped_at_one():
    """★ `不截顶`(`l0_output` 栏):抬得比肩宽还高 ⟹ 值 **> 1**。

    肩宽 0.20、抬起 0.40(2 倍肩宽)⟹ 2.0 那一档。
    旧口径 `min(shrug_diff, 0.1) / 0.1` ⟹ 最多 1.0(且前 30 帧恒交 0.0)⟹ 本条红。
    截顶会把"幅度"藏起来;饱和交给 L1 的分位映射(与 `au26_jaw_drop` 同一处置)。
    """
    vals = _shrug_series([_Y0] * 20 + [_Y0 - 0.40] * 10)
    assert max(vals) > 1.0, (
        f"抬起 2 倍肩宽时该列该 > 1(不截顶),实测最大 {max(vals)!r} —— "
        f"≤1 说明还在截顶(或分母还是那个裸常量 0.1)")


def test_shrug_needs_a_shoulder_width_or_it_is_empty():
    """没有肩宽 ⟹ **空**:分母没了,这个比值就不存在(不拿裸常量顶上)。"""
    vals = _shrug_series([_Y0] * 20 + [_Y0 - 0.10] * 5, shoulder_width=None)
    assert vals[-1] is None, (
        f"缺肩宽时该列该交空,实为 {vals[-1]!r} —— 交数说明分母还在用别的东西")


# ══════════════════════════════════════════════════════════════════════════════
# 端到端:端点 → 真分析器 → 真 logger(接线那一层)
# ══════════════════════════════════════════════════════════════════════════════
SID_A = "20260927_120000_l0aa"
SID_B = "20260927_120000_l0bb"


class _Upload:
    def __init__(self, data=b"jpeg"):
        self._data, self.filename, self.content_type = data, "f.jpg", "image/jpeg"

    async def read(self):
        return self._data


class _Req:
    async def form(self):
        raise RuntimeError("no multipart")


class _Img:
    shape = (720, 1280, 3)


def _wired(monkeypatch, tmp_path, hands, pose, *, k=1.0):
    """把端点接到**真分析器**上(不是替身):本文件要验的正是"端点喂进去的东西
    真的到了那一列上"。被替身掉的只有 cv2/np 的解码、探测器、时钟与落盘目录。

    ⚠️ 探测器替身**每帧交一份新的 landmarks**(带 `_wiggle` 的位移):交同一批对象
    等于一帧都没动 ⟹ 窗内标准差恒 0,而"0 与 0 相等"能让任何不变性断言**假绿**。
    """
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.delenv("JINGXIN_RETAIN_MEDIA", raising=False)
    media_retention._reset_for_tests()
    monkeypatch.setattr(gesture_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(gesture_app, "cv2", types.SimpleNamespace(
        imdecode=lambda buf, flag: _Img(), cvtColor=lambda i, c: i,
        IMREAD_COLOR=1, COLOR_BGR2RGB=1))
    monkeypatch.setattr(gesture_app, "np", types.SimpleNamespace(
        frombuffer=lambda b, t: b, uint8="u8"))

    frame_no = iter(range(10 ** 6))

    def _next_frame(base):
        # ⚠️ 位移加在**缩放之前**:取景 ×k 时整幅画面(含运动幅度)一起 ×k
        dx, dy = _wiggle(next(frame_no))
        return [_Pt((p.x + dx) * k, (p.y + dy) * k) for p in base]

    hand_tpl = _hand(_HAND) if hands else []
    pose_tpl = _pose(_POSE) if pose else None
    monkeypatch.setattr(gesture_app, "get_or_create_detectors", lambda sid: {
        "hands": types.SimpleNamespace(
            # 模型说 "Right" ⟹ 非镜像输入要**翻**过来 = 人的**左手** ⟹ 进 left 槽
            # (见 tests/test_hand_handedness_routing.py;写 "Left" 会填进 right 槽)
            detect_with_handedness=lambda img, ts: (
                [_next_frame(hand_tpl)] if hands else [], [("Right", 0.9)]),
            detect=lambda img, ts: [_next_frame(hand_tpl)] if hands else []),
        "pose": types.SimpleNamespace(
            detect=lambda img, ts: _next_frame(pose_tpl) if pose_tpl else None,
            detect_with_world=lambda img, ts: (
                _next_frame(pose_tpl) if pose_tpl else None,
                _next_frame(pose_tpl) if pose_tpl else None)),
    })
    # 每帧 1000 ms 的假时钟:"窗内真实秒数"在测试里因此是确定的
    # (真墙钟跑 12 帧只跨几毫秒 ⟹ 所有 jitter 变成巨数,断言没法读)
    ticks = iter(range(0, 10 ** 6, 1000))
    monkeypatch.setattr(gesture_app, "_clock_for",
                        lambda sid: types.SimpleNamespace(stamp_ms=lambda: next(ticks)))
    # ⚠️ **必须清**:分析器是模块级的、按会话存,上一支测试喂过的帧会留在窗口里
    gesture_app.session_analyzers.clear()
    gesture_app.session_loggers.clear()
    gesture_app.session_clocks.clear()


def _post(sid):
    return asyncio.run(gesture_app.analyze_image(
        request=_Req(), file=_Upload(), session_id=sid))


def _last_row(tmp_path, sid):
    p = tmp_path / "logs" / f"gesture_emotion_log_{sid}.csv"
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    assert rows, "一行都没写"
    return rows[-1]


def test_endpoint_zooming_the_subject_leaves_the_screen_jitter_unchanged(monkeypatch, tmp_path):
    """★ 走**端点 → 真分析器 → 真 logger** 整条路:取景 ×2 后 `left_hand_jitter` 不变。

    为什么要单跑一遍整条路:分析器的单测全绿**不代表**列上是对的 —— 喂 ts、喂肩宽、
    `get_results()` 的键、`world_results` 的键名、logger 的 `_safe_get`,任何一处漏了,
    值就在半路丢了。这里构造的是与单测同一条性质(缩放不变),但穿过**真产出方**。

    红法:端点里 `update(...)` 不传 `shoulder_width` ⟹ 该格**空**;传了而分析器不除 ⟹ ×2。
    """
    _wired(monkeypatch, tmp_path, [_hand(_HAND)], _pose(_POSE))
    for _ in range(12):
        _post(SID_A)
    row_a = _last_row(tmp_path, SID_A)

    _wired(monkeypatch, tmp_path, [_hand(_HAND)], _pose(_POSE), k=2.0)
    for _ in range(12):
        _post(SID_B)
    row_b = _last_row(tmp_path, SID_B)

    assert row_a["left_hand_jitter"] not in ("", "None"), (
        f"前提不成立:有姿态、有肩宽时该列该有值:{row_a['left_hand_jitter']!r}")
    # 先证明两次输入**确实**不同(否则"值不变"可能只是因为我喂了同一份数据)
    assert float(row_b["shoulder_width"]) == pytest.approx(
        2 * float(row_a["shoulder_width"]), rel=1e-6), (
        f"前提不成立:第二场该是取景 ×2(肩宽 0.40 → 0.80):"
        f"{row_a['shoulder_width']!r} → {row_b['shoulder_width']!r}")
    assert float(row_a["left_hand_jitter"]) > 0, "前提不成立:取值该为正"
    assert float(row_a["left_hand_jitter"]) == pytest.approx(
        float(row_b["left_hand_jitter"]), rel=0.05), (
        f"取景 ×2 后 left_hand_jitter 该不变(肩宽同乘 2):"
        f"{row_a['left_hand_jitter']!r} → {row_b['left_hand_jitter']!r}")


def test_endpoint_leaves_the_screen_jitter_empty_when_there_is_no_pose(monkeypatch, tmp_path):
    """★ 本帧**没有姿态** ⟹ 那一格是**空**,不是"未归一化的分子"(也不是 0)。

    活路径上"有手但没有姿态"是真会发生的(pose 漏检而 hand 检出)。那时没有肩宽 ——
    没有分母就没有这个率。旧口径照样吐一个数(它不设分母)⟹ 本条红。
    """
    _wired(monkeypatch, tmp_path, [_hand(_HAND)], None)      # 有手,没姿态
    for _ in range(12):
        _post(SID_A)
    row = _last_row(tmp_path, SID_A)
    assert row["left_hand_score"] != "", (
        f"前提不成立:这一帧左手槽该收到手:{row['left_hand_score']!r}")
    assert row["shoulder_width"] == "", "前提不成立:没有姿态时肩宽该是空"
    assert row["left_hand_jitter"] == "", (
        f"缺肩宽那一帧写进了值:{row['left_hand_jitter']!r} —— 空才是对的")


def test_endpoint_keeps_the_columns_working_when_the_pose_is_there(monkeypatch, tmp_path):
    """★ 对照臂:有姿态时那几格**必须有值** —— 否则上面"空"的断言可以靠"永远写空"骗过去。

    顺带钉住 `resilience_score` 仍在:jitter 交空或交率,都不该让分数字段消失或崩掉。
    """
    _wired(monkeypatch, tmp_path, [_hand(_HAND)], _pose(_POSE))
    for _ in range(12):
        _post(SID_A)
    row = _last_row(tmp_path, SID_A)
    for col in ("left_hand_jitter", "left_hand_spread",
                "left_wrist_jitter", "left_elbow_jitter", "left_shoulder_jitter",
                "right_shoulder_jitter", "head_jitter", "torso_jitter", "shrug_level",
                "left_wrist_jitter_world", "left_shoulder_jitter_world"):
        assert row[col] not in ("", "None"), f"{col} 这一格是空的:{row[col]!r}"
    assert float(row["left_hand_score"]) >= 0.0


def test_world_columns_keep_enough_decimals_for_a_rate():
    """★ world 那 8 列的落盘小数位:6 位够不够、4 位为什么不够(纯函数,直接钉)。

    这 6 个 jitter 列现在是**率**(米/秒):实测 3 场正式素材的中位数落在 2e-4…8e-4。
    4 位小数(步长 1e-4)在那个量级上只剩 1 位有效数字 —— 相邻两档就差一倍,下游拿它
    算相关/分位时被量化噪声吃掉。列名与列序一个都没动(双向钉子另有一条守着)。

    红法:把 `_WORLD_DECIMALS` 改回 4(或写成 `round(v, 4)`)⟹ 本条红。
    """
    import importlib
    glog = importlib.import_module("gesture_analysis.utils.logger")
    cells = glog.GestureLogger._world_cells(
        {"left_wrist_jitter_world": 0.0001234, "left_arm_angle_world": 33.125})
    assert cells["left_wrist_jitter_world"] == 0.000123, (
        f"米/秒 量级的率被 4 位小数截成了 {cells['left_wrist_jitter_world']!r} —— "
        f"1e-4 的步长在那个量级上是 1 位有效数字")
    # 缺 ⟹ 空,不写 0(与 `_world_cells` 一贯的约定一致)
    assert glog.GestureLogger._world_cells({})["left_wrist_jitter_world"] == ""
