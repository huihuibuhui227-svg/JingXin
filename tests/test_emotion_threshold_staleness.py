# tests/test_emotion_threshold_staleness.py
"""★ `emotion_engine` 里那些**照旧口径写的**硬编码门限 —— 让「已过期」不可能被静默忽略。

**账**(2026-09-27):`au6_cheek_raise` / `au25_mouth_open` / `au26_jaw_drop` 的口径在
2026-09-27 那一批里改过(分母、基线、截顶);同一天又把面部尺度的分母端点从**鼻尖**换成
**前额顶**(面部轮廓高)—— 后者还改掉了 au1/au2/au10/au15/au23/au12/symmetry 等的尺度。
而 `emotion_engine.infer()` 里的门限(`au26 > 0.3`、`au26 > 0.5`、`au6 > 0.1`、`au12 > 0.1` …)
是**照旧口径的数字范围**写的。实测下游分支的命中帧数当场就变了(三场正式素材:
`emotion_surprise` 189/269/121 → 271/413/416;`emotion_fear` 123/76/43 → 203/132/219)。

**口径(使用者已裁定)**:这些门限**不许**重新标定 —— 那需要标定集,**本仓没有**
(M3.1 已被裁定跳过),而「编一个阈值」正是本表反复禁止的形态;门限重登记按 spec 归
**M3.5**。本文件要做的只有一件事:**让「它们已过期」不可能被静默忽略**。做法三件:
① 常量处与模块里写清它是照哪个旧口径写的、现在为什么对不上(`STALE_THRESHOLDS`);
② 钉子把当前行为钉住(下面两条边界用例 —— **任何**对那几列口径的后续改动都会让它变红);
③ 账登记在本文件与 `l0_columns.json` 对应行的 `acceptance` 里。

⚠️ **边界(如实说)**:这些用例钉的是「门限 + 当前口径」这个**组合**,不是「门限是对的」。
它们红,只说明口径又动了 ⟹ 下一个人必须去重登记门限;**它们绿不代表门限被标定过**。
"""
import ast
import inspect
import pathlib

import numpy as np
import pytest

from face_expression.core.analysis.emotion_engine import (
    STALE_THRESHOLDS, UNAFFECTED_THRESHOLDS, EmotionEngine,
)
from face_expression.pipeline.video_pipeline import VideoPipeline

_FRAME = np.zeros((48, 48, 3), dtype=np.uint8)


def _lm(points):
    """478 点:未点名的点铺成极小的稀疏网格(与 `test_face_outline_height.py` 同一手法 ——
    全 0 会让 `dist(lm[33], lm[133])` 这类分母为 0,红的理由就变成「崩了」)。"""
    lm = np.zeros((478, 2), dtype=float)
    for i in range(478):
        lm[i] = ((i % 20) * 1e-4, (i // 20) * 1e-4)
    for i, (x, y) in points.items():
        lm[i] = (x, y)
    return [(float(x), float(y)) for x, y in lm]


class _FakeDetector:
    """只交一帧固定的 landmarks(**列表**,与真探测器同形;见 `test_face_outline_height.py`)。"""

    def __init__(self, landmarks):
        self._landmarks = landmarks

    def detect_with_blendshapes(self, image_rgb, timestamp_ms):
        return self._landmarks, {}

    def detect(self, image_rgb, timestamp_ms):
        return self._landmarks

    def reset(self):
        pass

    def close(self):
        pass


# 「每个门限都能算出落在边界附近的值」的合成脸。
#   · 面部中轴:前额顶 0.05、鼻尖 0.30、下巴 0.95 ⟹ **轮廓高 0.90**、鼻尖→下巴 0.65
#   · 下巴 y − 上唇内缘 y = 0.95 − 0.545 = **0.405**
#       ⟹ 现口径 au26 = 0.405/0.90 = **0.450**(旧分母 0.65 会给 **0.623**)
#       ⚠️ 上唇内缘**刻意不取 0.50**:那样 0.45/0.90 恰好落在 `0.5` 这个浮点刀口上,
#          钉子的红绿就不再由口径决定,而由最后一位比特决定。
#   · 眉抬起 0.18×2 ⟹ au1 = au2 = 0.36/(2×0.90) = **0.20**(旧分母 0.65 给 0.277)
#   · 颊-眼竖直距 0.10×2、轮廓高 0.90 ⟹ au6 = 1 − 0.20/(2×0.90) = **0.889**
#   · head_yaw = (鼻尖 0.50 − 双颊中点 0.45)/face_width 0.40 = **0.125**
_POINTS = {
    10: (0.50, 0.05), 1: (0.50, 0.30), 152: (0.50, 0.95),
    234: (0.25, 0.50), 455: (0.65, 0.50),            # face_width = 0.40
    0: (0.50, 0.40), 13: (0.50, 0.545), 14: (0.50, 0.60),
    168: (0.50, 0.48),
    52: (0.45, 0.30), 55: (0.55, 0.30),
    70: (0.35, 0.30), 63: (0.65, 0.30),
    205: (0.40, 0.50), 145: (0.40, 0.60),
    425: (0.60, 0.50), 374: (0.60, 0.60),
    33: (0.35, 0.45), 133: (0.45, 0.45),
    160: (0.37, 0.44), 153: (0.37, 0.46),
    159: (0.40, 0.44), 144: (0.40, 0.46),
    362: (0.65, 0.45), 263: (0.55, 0.45),
    387: (0.63, 0.44), 380: (0.63, 0.46),
    386: (0.60, 0.44), 373: (0.60, 0.46),
    **{i: (0.43, 0.47) for i in (468, 469, 470, 471)},
    **{i: (0.57, 0.43) for i in (473, 474, 475, 476)},
    129: (0.40, 0.30), 358: (0.60, 0.30),
    130: (0.30, 0.40), 359: (0.70, 0.40),
    107: (0.45, 0.35), 336: (0.55, 0.35),
    202: (0.40, 0.60), 422: (0.60, 0.60),
    61: (0.45, 0.60), 291: (0.55, 0.60),
}


def _mouth(width):
    return {61: (0.50 - width / 2, 0.60), 291: (0.50 + width / 2, 0.60)}


def _run_pipeline(frames, ts0=1000):
    """★ 走**真产出方**:`VideoPipeline.process_frame`(喂假探测器),读它交出的扁平字典。

    为什么不是「自己造一个 `AUFeatures` 再喂 engine」:那样测的只是「给定一个数,门限怎么比」——
    分母/基线/截顶**怎么改都绿**(本仓已 5 次栽在「验证跑错了对象」)。这里连
    `emotion_engine.infer` 都是管线**自己**调的(端点上就是这一条路),所以任何对那几列口径的
    改动,都会经由「算出来的 AU 值」翻转分支,钉子才会红。
    """
    p = VideoPipeline(session_id="s", detector=_FakeDetector(_lm(frames[0])))
    dumped = None
    for i, pts in enumerate(frames):
        p.detector = _FakeDetector(_lm(pts))
        _r, _mesh, dumped = p.process_frame(_FRAME, ts0 + i * 1000)
    return dumped


def test_the_fear_branch_reads_the_current_au26_scale():
    """★ `au26 > 0.5` 这条门限的**边界**:au26 的口径一改,这条就红。

    几何:下巴 0.95、上唇内缘 0.545 ⟹ 竖直距 0.405。
      · **现口径**(÷ 面部轮廓高 0.90):au26 = **0.450** ⟹ `au26 > 0.5` 不成立 ⟹ fear **不发**;
      · **旧口径**(÷ 鼻尖→下巴 0.65):au26 = **0.623** ⟹ 同一帧 fear **会发**
        (另两个条件都刻意留在满足侧:au1 = au2 = 0.20 > 0.15、head_yaw = 0.125 > 0.1)。

    ⟹ 这条的红法就是「某一列的分母又动了」,而且红的时候**同时**告诉你 `au26 > 0.5`
    这个常量是按旧口径的数值范围定的、现在落在哪儿。它**不**声称 0.5 是对的。
    """
    d = _run_pipeline([_POINTS])
    assert d["au26_jaw_drop"] == pytest.approx(0.450, abs=1e-3), (
        f"au26 该是 0.405/0.90 = 0.450,实为 {d['au26_jaw_drop']} —— "
        f"0.623 说明分母是鼻尖→下巴 0.65")
    assert d["au1_inner_brow_raise"] == pytest.approx(0.20, abs=1e-3), d["au1_inner_brow_raise"]
    assert d["head_yaw"] == pytest.approx(0.125, abs=1e-3), d["head_yaw"]
    assert d["emotion_surprise"] > 0, (
        "au26 = 0.450 > 0.3 且眉抬 > 0.1 ⟹ surprise 分支该命中")
    assert d["emotion_fear"] == 0, (
        f"au26 = 0.450 不满足 `au26 > 0.5` ⟹ fear 分支不该命中,实为 {d['emotion_fear']} —— "
        f"若它命中了,说明 au26 又变大回去了(口径/尺度被改),"
        f"那正是「门限照旧口径写」这件事要被翻出来的信号")


def test_the_happy_branch_reads_the_current_au12_scale():
    """★ `au12 > 0.1` 这条门限的**边界**:基线口径一改,这条就红。

    同一会话喂两帧,口宽 0.10 → 0.115:
      · **现口径**(基线 = 全程 p10 的因果近似):p10({0.10, 0.115}) = 0.1015
        ⟹ au12 = 0.0135/0.1015 = **0.133** > 0.1 ⟹ happy 发(au6 = 0.889 > 0.1);
      · **旧口径**(基线 = 开头 10 帧的滑动均值):基线 = 0.1075 ⟹ au12 = **0.0698** < 0.1
        ⟹ happy **不发**。
    """
    d = _run_pipeline([{**_POINTS, **_mouth(0.10)}, {**_POINTS, **_mouth(0.115)}])
    assert d["au12_smile"] == pytest.approx(0.133, abs=2e-3), (
        f"au12 该是 (0.115−0.1015)/0.1015 = 0.133,实为 {d['au12_smile']} —— "
        f"0.070 说明基线还是「开头 10 帧的滑动均值」")
    assert d["au6_cheek_raise"] > 0.5, (
        f"au6 = {d['au6_cheek_raise']} —— `au6 > 0.1` 这条门限在现口径下离实测值域"
        f"(三场 0.73~0.81)极远,基本不可能是决定性的那一项(见 STALE_THRESHOLDS)")
    assert d["emotion_happy"] > 0, (
        f"au12 = 0.133 > 0.1 且 au6 > 0.1 ⟹ happy 分支该命中,实为 {d['emotion_happy']}")


# ── 账本的完整性 ─────────────────────────────────────────────────────────────
_COMPARE_OPS = {ast.Gt: ">", ast.Lt: "<", ast.GtE: ">=", ast.LtE: "<=",
                ast.Eq: "==", ast.NotEq: "!="}


def _thresholds_in_infer():
    """从 `EmotionEngine.infer` 的**源码**里推出全部门限(变量 > 浮点常量)。

    用 AST 而不是正则:门限的写法会变(换行、加括号、`and`/`or` 重排),正则一漏就是
    「账本没覆盖到而钉子看不见」—— 那正是本文件要防的静默。判据的**边界**也一并说清:
    只看「左 = 变量、右 = 浮点字面量」的比较 ⟹ `total > 0`(整数,归一化守卫)、
    `temporal_stats.data.get(...) > 0.01`(左边是调用)不在本判据内 —— 后者在账本里
    单列成 `indirect` 项。
    """
    path = inspect.getsourcefile(EmotionEngine)
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    infer = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "EmotionEngine":
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == "infer":
                    infer = sub
    assert infer is not None, "没找到 EmotionEngine.infer —— 判据本身失效了"
    found: dict[str, int] = {}
    for node in ast.walk(infer):
        if not (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
                and len(node.ops) == 1):
            continue
        try:
            value = ast.literal_eval(node.comparators[0])
        except (ValueError, TypeError):
            continue
        if not isinstance(value, float):
            continue
        key = f"{node.left.id} {_COMPARE_OPS[type(node.ops[0])]} {value!r}"
        found[key] = found.get(key, 0) + 1
    return found


def test_the_ledger_covers_every_hard_coded_comparison_in_infer():
    """★ 双向钉子:源码里每一条门限都要在账本里,账本里每一条也都要在源码里。

    红法(三条都实测过):
      · 改一个门限的数(`0.3` → `0.35`)⟹ 源码那边多一条、账本这边少一条 ⟹ 红;
      · 新加一条门限而不登记 ⟹ 红;
      · 删掉一条门限 ⟹ 账本里那条成了孤儿 ⟹ 红;
      · 解析不出任何门限(判据自己被改坏)⟹ 红。
    为什么需要它:**这就是「不可能被静默忽略」的机械保证** —— 门限可以改,但改的人
    **必须**同时动这个账本,而账本每一行都写着「它照哪个旧口径写的、待 M3.5 重登记」。
    """
    found = _thresholds_in_infer()
    registered = set(STALE_THRESHOLDS) | set(UNAFFECTED_THRESHOLDS)
    missing = sorted(set(found) - registered)
    # 账本里允许有**不来自 AST** 的条目:唯一一个是 `indirect` —— 那两条 trend 门限的
    # 左边是**调用**不是变量,AST 看不见(见下一条测试)。除它以外的孤儿都是"账本说了
    # 一个不存在的世界"。
    orphan = sorted(registered - set(found) - {"indirect"})
    assert not missing, (
        f"这些门限在 `infer()` 里,却不在账本里:{missing} —— 把它们登记进 STALE_THRESHOLDS"
        f"(写清照哪个旧口径写的)或 UNAFFECTED_THRESHOLDS(写清为什么不受口径影响)")
    assert not orphan, (
        f"账本里这些条目在 `infer()` 里已经找不到了:{orphan} —— "
        f"门限删了/改了就要同步账本,否则账本说的是一个不存在的世界")
    assert found, "一条门限都没解析出来 —— 判据本身坏了(不是「通过」)"


def test_every_stale_entry_says_which_old_scale_it_was_written_against():
    """账本每一行都要能回答四个问题 —— 缺一个,它就退化成一个「编出来的阈值」或一句
    「假装没过期」。"""
    assert STALE_THRESHOLDS, "账本空了 —— 那条完整性钉子会当场红,但这条也说明账没了"
    for key, rec in STALE_THRESHOLDS.items():
        assert rec.get("column"), f"{key}: 没写是哪一列的门限"
        assert rec.get("written_against"), (
            f"{key}: 没写它**照哪个旧口径**写的 —— 这正是「已过期」这件事的全部内容")
        assert rec.get("now"), f"{key}: 没写该列现在的口径"
        assert rec.get("measured"), f"{key}: 没写实测(三场正式素材的分支命中/门限生效帧数)"
        assert "M3.5" in rec.get("status", ""), (
            f"{key}: status 里没写「M3.5」—— 账要落到一个明确的里程碑上,"
            f"否则「待重登记」等于「永远不登记」")
    for key, why in UNAFFECTED_THRESHOLDS.items():
        assert str(why).strip(), f"{key}: 说了不受影响却没说理由"


def test_the_two_trend_thresholds_are_registered_as_indirect():
    """那两条 `*_trend > 0.01` 的比较左边是**调用**(不是变量),AST 判据看不见它们。

    它们同样是过期的:trend 是 `au1_inner_brow_raise` 的斜率 ×100,而 au1 的尺度本批也变了
    (分母换了端点)⟹ `startled_anxiety` 的命中率跟着变。这里把它们**单独**登记,
    免得「AST 抓不到」被读成「它们没问题」。
    """
    assert "indirect" in STALE_THRESHOLDS, (
        "账本里要有一条 `indirect` 项,把 AST 判据看不见的那两条 trend 门限记下来")
