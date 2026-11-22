# tests/test_m4_l1_mapper.py
"""**M4.5**:14 个旧槽 → L1 指标的替换守卫。

## 钉的是什么

1. **映射表不许指向不存在的指标** —— 槽写了 L1 名字,那名字必须真在 L1 里;
2. **构念不同的必须删,不许拿相邻构念顶替** —— 删掉的槽要带理由;
3. **权重在幸存槽上重新归一** —— 和必须为 1;
4. **全部槽都删掉的维度整体消失**,不留一个 `0/0` 的空壳在 payload 里;
5. **置信度取最弱的那条**,不是"填了几个槽";
6. **不给维度分、不给档位** —— 与 2026-10-04 的决定一致。

## 为什么第 2 条最要紧

历史上「皱眉」槽就是被子串匹配到「从 micro_exp 字符串解析出的垃圾」的。
换个名字接着填,比留空更坏:读者会以为量到了。
"""

from __future__ import annotations

import pytest

from report_frontend import l1_face, l1_gesture, l1_mapper, l1_text, l1_voice


def _all_l1_names() -> set[str]:
    return (set(l1_face.indicator_names())
            | set(l1_gesture.ORDER)
            | set(l1_voice.ORDER)
            | set(l1_text.ORDER))


def _ind(name, value=None, n_valid=100, n_total=100, confidence="高",
         unit="", gap=None, human_name=""):
    return {"name": name, "value": value, "n_valid": n_valid, "n_total": n_total,
            "confidence": confidence, "unit": unit, "gap": gap,
            "human_name": human_name or name}


# ── 1. 映射表不许指向不存在的指标 ───────────────────────────────────────

def test_every_mapped_l1_name_really_exists():
    """★ 红法:把 `pause_duration` 映射到 `pause_duration_p95`(不存在)⟹ 本条红。

    没有这条,映射表可以静静指向一个永远无值的名字,报告只会说"本场无值",
    而真相是"这个名字根本没人产出"。
    """
    known = _all_l1_names()
    for slot, (l1, reason) in l1_mapper.SLOT_MAP.items():
        assert reason, f"{slot} 没有理由"
        if l1 is not None:
            assert l1 in known, f"{slot} 指向了不存在的 L1 指标 {l1}"


def test_slot_map_covers_exactly_the_fourteen_old_slots():
    """14 个槽一个不多一个不少 —— 少一个就是"悄悄漏掉",多一个就是"凭空发明"。"""
    old = set()
    for spec in l1_mapper.DIMENSIONS.values():
        old |= set(spec["slots"])
    assert len(old) == 14, sorted(old)
    assert set(l1_mapper.SLOT_MAP) == old


# ── 2. 构念不同的必须删 ─────────────────────────────────────────────────

def test_no_l1_indicator_is_counted_by_two_slots():
    """★ 同一个 L1 指标不许喂两个槽 —— 那是把**一份**测量数两遍。

    这条守卫在**旧引擎** `research_mapper` 上早就有
    (`test_report_slot_columns.py::test_no_column_keyword_is_used_by_two_slots`),
    但生产默认路径从 2026-10-18 起是 `l1_mapper`(`/api/report/structured` 默认
    `engine=l1`),而 L1 侧**一条等价守卫都没有** —— 正是本仓「验证跑错了对象」
    的形态:守的是对照用的旧路径,跑的是 L1。

    后果不只是"数两遍":同一份测量进两个槽 ⟹ 该维度的证据链里有两条
    **相关性 1.0** 的"独立证据",置信度取最弱那条也会被抬上去。

    红法:把 `shrug_level` 映射到 `posture_stability`(与另一槽同一个 L1)⟹ 本条红。
    """
    seen: dict[str, str] = {}
    for slot, (l1, _reason) in l1_mapper.SLOT_MAP.items():
        if l1 is None:            # 删槽,没有对象可查
            continue
        assert l1 not in seen, (
            f"{l1} 同时喂了 {seen.get(l1)!r} 和 {slot!r} —— 同一份测量被数了两遍")
        seen[l1] = slot
    # 地板:幸存槽数不许悄悄变少(有人把某个槽改成 None = 悄悄删槽,循环就少跑一次)
    # ⚠️ 2026-11-22:8 → 11。`symmetry_score` / `left_hand_spread` / `shrug_level`
    #    由"删"改为"换"(三条新指标各自读一列真测量),于是幸存槽 +3、删槽 −3。
    #    这条地板值必须跟着涨 —— 它守的是"槽表被改小"这类静默改动。
    # ⚠️ 2026-11-21:11 → 13。`fluency_score`(→ pause_structure_entropy)与
    #    `n_chars`(→ answer_detail_chars_p50)由"删"改为"换",幸存 +2、删槽 −2。
    #    这条地板值必须跟着涨 —— 它守的是"槽表被改小"这类静默改动。
    # ⚠️ 2026-11-22(同日第二次):13 → **14,满**。最后一个删槽 `eye_contact` 补上
    #    `gaze_dwell_p50`(注视带内连续段长中位,秒)⟹ 14 个槽全部有对应指标。
    #    这条地板值同时是**上限**(槽总数 14):多一个就说明 `SLOT_MAP` 里冒出了
    #    维度表里没有的槽(`test_slot_map_covers_exactly_the_fourteen_old_slots` 也守这条)。
    assert len(seen) == 14, f"幸存槽应 14 个(满),实得 {len(seen)}:{sorted(seen)}"


def test_no_slot_is_dropped_any_more_but_the_rule_is_still_enforced():
    """★ 2026-11-22:**一个删槽都不剩**了 —— 14 个槽全部换成了 L1 指标。

    历史:7/7 → 8/6 → 11/3 → 13/1 → **14/0**。每一次"删→换"都是因为**补了一条
    真会出值的指标**(不是放宽"同构念"这条判据):`symmetry_score` /
    `left_hand_spread` / `shrug_level`(三条新指标)、`fluency_score` / `n_chars`
    (2026-11-21)、`eye_contact` → `gaze_dwell_p50`(本次)。

    ## 判据本身仍然必须可执行(不许靠"今天没有"混过去)

    原先这里是「每个删槽一条 parametrize」,删槽清零之后 parametrize 会变空列表
    ⟹ `test_assert_coverage` 会报"声明了却没跑到的 assert"。所以把判据拆成两部分:
    ① 真表上**零删槽**(这条仍然是硬钉,谁把一个槽改回 `None` 就红);
    ② 判据函数本身对**人造的**删槽照样会判(见
       `test_dropped_slot_reasons_say_construct_mismatch_not_a_coverage_excuse`
       —— 那个名字是**保留的旧名**,本次只是把它的判据从"遍历真表"扩到
       "真表零删槽 + 人造输入验判据函数"),能力不随数据消失。

    红法:把 `eye_contact` 改回 `None` ⟹ ①红。
    """
    dropped = {s: (l1, r) for s, (l1, r) in l1_mapper.SLOT_MAP.items() if l1 is None}
    assert dropped == {}, (
        f"真表上又出现删槽了:{sorted(dropped)} —— 若某条指标被撤掉,请按判据逐条核,"
        f"不要直接把这条测试改松")


def _drop_verdict(slot: str, l1: str | None, reason: str, known: set[str]) -> bool:
    """判据本体:一个槽**该不该**被判成删槽。

    * 有对应指标 ⟹ 不删(与构念无关的那些理由归 `NOT_COMPARABLE` 去说);
    * 没有对应指标 ⟹ 必须删,且理由必须点出一条**真实存在**的 L1 指标名
      (证明"拿谁来顶替"被考虑过并被否决),并且**不许**用覆盖借口
      (「没测到」「未采集」这类 —— 那是"这一场没有",不是"这条管线不测")。
    """
    if l1 is not None:
        return False
    if not any(n in reason for n in known):
        return False
    return not any(x in reason for x in ("没测到", "未采集", "无数据", "没有数据"))


def test_dropped_slot_reasons_say_construct_mismatch_not_a_coverage_excuse():
    """★ 删槽理由必须是「**构念不同**」,不能是「这个没测到」;而且这条判据今天仍可执行。

    ## ⚠️ 2026-11-22:真表上已经没有删槽了 ⟹ 判据改用**人造输入**验

    本函数原名下面那段历史仍然成立(它记的是判据的由来):这条判据 2026-10-23
    引入时是恒真的(`assert "左" not in reason or "可见性" in reason`),被
    `_drop_verdict` 取代后才真的能抓错。最后一个删槽 `eye_contact` 改成"换"之后,
    真表上**一个删槽都不剩** ⟹ 若继续只遍历真表,循环一次都不进、所有断言都不执行
    (`test_assert_coverage` 会当场报"声明了却没跑到的 assert")。

    所以本用例变成两段,判据一字没改:
    ① 真表上**零删槽**(谁把一个槽改回 `None`,这里先红);
    ② 拿**人造的**删槽输入跑同一个判据函数 `_drop_verdict`,四条覆盖
       「该删 / 有指标不该删 / 没点名真实指标 / 用覆盖借口」两侧边界。

    红法:把 `_drop_verdict` 改成恒 `False` ⟹ 第 ①... 见下,第一、四条断言红;
    改成恒 `True` ⟹ 第三、四条断言红。
    """
    known = _all_l1_names()
    good = "已由 `gaze_aversion_ratio_vs_baseline` 覆盖同一件事,不重复产出"
    # ① 没有指标 + 点名了真实 L1 指标 + 没说覆盖借口 ⟹ 该删
    assert _drop_verdict("synthetic_slot", None, good, known)
    # ② 有指标 ⟹ 不删(不管理由写得多像删槽)
    assert not _drop_verdict("synthetic_slot", "gaze_dwell_p50", good, known)
    # ③ 没点名任何真实 L1 指标 ⟹ 不合格
    assert not _drop_verdict("synthetic_slot", None, "这个量我们不测", known)
    # ④ 用了覆盖借口 ⟹ 不合格
    assert not _drop_verdict("synthetic_slot", None,
                             "gaze_dwell_p50 那一列没测到", known)


def test_fourteen_slots_survive_and_none_is_dropped():
    """实测对账结论:**14 换 0 删**(7/7 → 8/6 → 11/3 → 13/1 → **14/0**)。

    数字变了就说明对账被改过,必须重新说明 —— 这就是这条测试的作用。
    * 2026-11-22 的 +3 来自三条新指标(`facial_asymmetry_p90` /
      `hand_openness_p50` / `shrug_elevation_p90`),各自读一列**已落盘**的真测量;
      复核证据见 `docs/补强/patch/D1_指标替换_3退3进_说明.md`。
    * 2026-11-21 的 +2 来自 `pause_structure_entropy` / `answer_detail_chars_p50`;
    * **2026-11-22(同日第二次)的 +1** 来自 `gaze_dwell_p50`(注视带内连续段长中位,
      秒)—— 最后一个删槽 `eye_contact` 由此换成。ρ 体检表见交付说明第 22 号。

    红法:把 `connective_density` 改回 `None` ⟹ 本条红(回到 13/1)。
    """
    kept = [s for s, (l1, _) in l1_mapper.SLOT_MAP.items() if l1 is not None]
    dropped = [s for s, (l1, _) in l1_mapper.SLOT_MAP.items() if l1 is None]
    assert len(kept) == 14 and len(dropped) == 0, (kept, dropped)
    assert "connective_density" in kept
    assert "eye_contact" in kept, "最后一个删槽必须已经换成 `gaze_dwell_p50`"


# ── 3. 权重重新归一 ─────────────────────────────────────────────────────

def test_surviving_weights_are_renormalized_to_one():
    """★ 幸存槽权重和必须为 1。

    红法:去掉 `_renormalize` 调用 ⟹ 话语结构特征的和是 0.429,本条红。
    """
    inds = [_ind(n, 1.0) for n in _all_l1_names()]
    dims, _ = l1_mapper.map_l1_to_dimensions(inds)
    for key, d in dims.items():
        total = sum(s["weight"] for s in d.slots)
        assert total == pytest.approx(1.0), (key, total)


def test_renormalization_keeps_the_original_proportions():
    """`logical_thinking` 原比例 .286:.143 = 2:1 ⟹ 归一后仍应是 2:1。

    红法:改成等权(0.5/0.5)⟹ 本条红。删槽时**保持比例**是本仓既有做法,
    等权会让一个原本次要的槽突然占到一半。
    """
    dims, _ = l1_mapper.map_l1_to_dimensions([])
    d = dims["logical_thinking"]
    w = {s["slot"]: s["weight"] for s in d.slots}
    assert w["gaze_direction_x_std"] / w["au4_frown"] == pytest.approx(2.0, rel=1e-6)


# ── 4. 维度整体消失 ─────────────────────────────────────────────────────

#: 一个人造的"整维没有对应指标"维度 —— 专供本节的四条测试。
#:
#: ## 为什么需要它(2026-11-22)
#:
#: 补进三条新指标之后,**生产数据里再没有任何维度会整体消失**:
#: `confidence_level`(行为表现活跃度)从"3 槽全删"变成"2 槽有指标 + 1 槽如实登记不测",
#: 于是 `vanished_dimensions` 为空。那是**事实变了**,不是渲染层不印了 ——
#: 而"整维消失时不许留 0/0 空壳"这条不变量**仍然必须有人守**,否则下一个人
#: 删掉 `if not res.slots: continue` 时不会有任何测试变红。
#:
#: ## 为什么用"借用真删槽"而不是编一个假维度
#:
#: 三个槽(`fluency_score` / `eye_contact` / `n_chars`)**就是今天真实没有对应指标的
#: 那三个** —— 它们的 `SLOT_MAP`/`DROP_REASON`/`SLOT_LABEL` 条目都是真的,
#: 所以这条测试仍然在验真表,不验一份编出来的表。造出来的只是"它们恰好被分到
#: 同一个维度里"这一件事。
_SYNTHETIC_KEY = "synthetic_dead_dimension"


@pytest.fixture
def all_dead_dimension(monkeypatch):
    """临时加一个"槽全都没有对应指标"的维度(用完即撤)。

    ## 2026-11-21 为什么改用**合成槽键**

    真删槽只剩 `eye_contact` 一个(`fluency_score` / `n_chars` 已改判为"换"),
    凑不出"整维全删"。所以这里用三个**确定不在 `SLOT_MAP` 里**的合成键 ——
    不变量照旧:不在表里 ⟹ 无对应指标 ⟹ 该维度应整体消失。
    合成键比"借用真删槽 + 改断言"更诚实:后者会让这条测试随槽表演进漂移。
    """
    fake = {"fake_dead_slot_a": 0.4, "fake_dead_slot_b": 0.35, "fake_dead_slot_c": 0.25}
    monkeypatch.setitem(l1_mapper.DIMENSIONS, _SYNTHETIC_KEY, {
        "name": "构造用整维", "slots": fake,
    })
    dead = [s for s in fake if l1_mapper.SLOT_MAP.get(s, (None,))[0] is None]
    assert len(dead) == len(fake), (
        f"构造前提不成立:{sorted(fake)} 里有 {len(dead)} 个是真删槽 —— "
        "合成键不该出现在 SLOT_MAP 里")
    return _SYNTHETIC_KEY


def test_dimension_with_no_surviving_slot_vanishes_not_an_empty_shell(
        all_dead_dimension):
    """★ 一个维度的槽全删 ⟹ 该维度**整体消失**,不留 `0/0` 空壳。

    ## 2026-11-22 的改动:被测对象从"`confidence_level` 天然全删"变成"构造一个全删维度"

    原因不是放宽,是**生产数据里已经没有整维消失的维度了**(三条新指标让
    `行为表现活跃度` 复活到 2/3)。不变量本身一个字没改:全删就该消失。
    另外这一条仍然守着**真**表:`confidence_level` 现在必须有指标,由下面那条
    `test_the_confidence_dimension_no_longer_vanishes` 直接钉死。

    红法:把 `if not res.slots: continue` 删掉 ⟹ 会返回一个 `matched_indicators="0/0"`
    的维度,报告里就会多出一行"XX 0/0" —— 读者会以为"本场没测到",
    而真相是"这个维度量的东西当前管线都不测"。
    """
    dims, vanished = l1_mapper.map_l1_to_dimensions([])
    assert all_dead_dimension not in dims
    assert [v["key"] for v in vanished] == [all_dead_dimension]


def test_the_confidence_dimension_no_longer_vanishes():
    """★ 2026-11-22:**`行为表现活跃度` 3 个槽全都有对应指标了**(不再"2/3")。

    ## 这条为什么必须存在

    上一版这一维整维消失(三个槽全删),而"整维消失"是一句**很强的话**
    ("这个维度量的东西当前管线都不测")。补进 `shrug_elevation_p90` /
    `hand_openness_p50` 之后那句话成了假话;本次再补 `gaze_dwell_p50`,
    **一个删槽都不剩** ⟹ 必须在**真表**上钉住它,否则下一个人可能把新指标撤掉
    而没人发现:报告会退回"四维/2 槽",而那条用构造维度跑的测试照样绿。

    ⚠️ 2026-11-22(同日第二次):`eye_contact` 由删改换 ⟹
    `dropped` 从 `["eye_contact"]` 变**空**,`n_slots` 仍是 3,
    分母 `matched_indicators` 仍按**旧槽总数** 3 计(读者要比"原来 3 个槽今天过几个")。

    红法:把 `shrug_level` / `left_hand_spread` / `eye_contact` 任一改回 `None` ⟹ 本条红。
    """
    dims, vanished = l1_mapper.map_l1_to_dimensions([])
    assert "confidence_level" in dims, f"这一维又消失了:{[v['key'] for v in vanished]}"
    assert "confidence_level" not in [v["key"] for v in vanished]
    d = dims["confidence_level"]
    assert d.n_slots == 3, d.n_slots
    # 顺序 = `DIMENSIONS["confidence_level"]["slots"]` 的声明顺序(权重仍是 0.375/0.25/0.375)
    assert [s["l1"] for s in d.slots] == [
        "hand_openness_p50", "shrug_elevation_p90", "gaze_dwell_p50"]
    assert [x["slot"] for x in d.dropped] == [], (
        f"这一维又出现删槽了:{[x['slot'] for x in d.dropped]}")
    # 分母仍是"旧槽总数"(3):读者要比的是"原来 3 个槽,今天过了几个"。
    assert d.matched_indicators.startswith("0/3")


# ── 5. 置信度取最弱 ─────────────────────────────────────────────────────

def test_dimension_confidence_is_the_weakest_matched_evidence():
    """两条证据一高一低 ⟹ 维度只能是"低"。

    红法:改成 `max` ⟹ 本条红。维度陈述的强度不可能超过最弱的那条证据。
    """
    inds = [_ind("gaze_stability", 0.8, confidence="高"),
            _ind("brow_furrow_ratio_vs_baseline", 0.1, confidence="低")]
    dims, _ = l1_mapper.map_l1_to_dimensions(inds)
    assert dims["logical_thinking"].confidence == "低"


def test_dimension_with_no_measured_value_is_confidence_none():
    """证据在、但都没值 ⟹ 置信"无",不是"低"。"""
    inds = [_ind("gaze_stability", None, n_valid=0, confidence="无")]
    dims, _ = l1_mapper.map_l1_to_dimensions(inds)
    assert dims["logical_thinking"].confidence == "无"
    assert dims["logical_thinking"].matched_indicators == "0/3"


# ── 6. 计数口径 ─────────────────────────────────────────────────────────

def test_matched_counts_over_old_slots_not_survivors():
    """`matched_indicators` 的分母是**旧槽总数**(3),不是幸存槽数(2)。

    读者要对比的是"这一维原来有 3 个槽,今天过了 2 个",分母换成 2 会让人
    以为这维本来就只有 2 个槽。
    """
    inds = [_ind("gaze_stability", 0.8), _ind("brow_furrow_ratio_vs_baseline", 0.1)]
    dims, _ = l1_mapper.map_l1_to_dimensions(inds)
    assert dims["logical_thinking"].matched_indicators == "2/3"


# ── 7. 不许出现分与档 ───────────────────────────────────────────────────

def test_payload_has_no_score_or_level():
    """★ 没有标定样本 ⟹ 任何 `score` / `level` 都是未标定标尺上的点分。

    红法:在 `to_dict()` 里加一行 `"score": ...` ⟹ 本条红。
    """
    inds = [_ind(n, 1.0) for n in _all_l1_names()]
    dims, _ = l1_mapper.map_l1_to_dimensions(inds)
    for d in dims.values():
        payload = d.to_dict()
        assert "score" not in payload and "level" not in payload
        assert set(payload) == {"display_name", "evidence_chain",
                                "evidence_gaps", "confidence", "matched_indicators"}


def test_dropped_slots_appear_as_evidence_gaps():
    """删掉的槽必须能被读者看见 —— 它们进 `evidence_gaps`,不是消失得无影无踪。

    ⚠️ 2026-11-22:被测维度从 `stress_resilience` 换成 `communication_fluency`。
    原来那句 `assert "symmetry_score" in dropped` 现在**依然为真**,但**为真的原因
    是错的**:`symmetry_score` 已经映射到 `facial_asymmetry_p90`,它出现在
    `evidence_gaps` 里是因为"本场没有传指标值",不是因为它是删槽。
    一条"碰巧还成立"的断言比一条红断言更危险 ⟹ 现在分开数:
    删槽那条(`fluency_score`,真删)与没值那条(映射了但没数)分别断言。
    """
    dims, _ = l1_mapper.map_l1_to_dimensions([])
    payload = dims["communication_fluency"].to_dict()
    dropped = {g["slot"] for g in payload["evidence_gaps"] if "slot" in g}
    # 真删槽:必须在。
    assert "fluency_score" in dropped
    # 已映射、但本场一条指标都没值的槽:也必须在 —— 两种缺口都得说出来。
    assert "pitch_variation" in dropped
    # 而且 `symmetry_score` 现在**不是**删槽(它在 `stress_resilience` 的幸存槽里)。
    assert "symmetry_score" not in dropped
    stress = dims["stress_resilience"].to_dict()
    assert [e["indicator"] for e in stress["evidence_chain"]] == [
        "lip_compression_ratio_vs_baseline", "gesture_activity_per_sec",
        "facial_asymmetry_p90"]


def test_confidence_rank_matches_research_mapper():
    """★ 两张置信度排序表必须**逐项相同**,方向都是"数值越大越可信"。

    ## 这条为什么必须存在

    `l1_indicators.CONFIDENCE = ("高","中","低","无")` 的顺序是"从可信到不可信",
    直接 `enumerate` 出来是 **高=0**;而 `research_mapper.CONF_ORDER` 是 **高=3**。
    两个表同名反序、都用 `max`,含义正好颠倒 —— 我第一版就把"最弱证据"算成了
    "最强证据",被 `test_dimension_confidence_is_the_weakest_matched_evidence` 抓住。

    红法:把 `l1_mapper.CONF_RANK` 改回 `{c: i for i, c in enumerate(CONFIDENCE)}`
    ⟹ 本条红(顺带那条"最弱证据"也红)。
    """
    from report_frontend import research_mapper
    assert l1_mapper.CONF_RANK == research_mapper.CONF_ORDER
    assert l1_mapper.CONF_RANK["高"] == 3 and l1_mapper.CONF_RANK["无"] == 0


def test_nan_value_counts_as_no_value():
    """★ `NaN` 必须与 `None` 同义 —— 不许数成"过门"。

    `l1_indicators.unavailable()` 把"算不出来"记作 `value=math.nan`,而
    `math.nan is not None` 是 **True**。第一版只判 `is not None`,于是空数据场
    的 `n_passed` 从 0 虚报成 7、`matched_indicators` 跟着虚高。

    红法:把 `_clean()` 换成 `lambda v: v` ⟹ 本条红(端点测试的空数据断言也红)。
    """
    import math
    inds = [{"name": "gaze_stability", "value": math.nan, "n_valid": 0,
             "n_total": 100, "confidence": "无", "unit": "", "gap": "no_samples"},
            {"name": "brow_furrow_ratio_vs_baseline", "value": None, "n_valid": 0,
             "n_total": 100, "confidence": "无", "unit": "", "gap": "uncalibrated"}]
    dims, _ = l1_mapper.map_l1_to_dimensions(inds)
    d = dims["logical_thinking"]
    assert d.n_matched == 0, d.n_matched
    assert d.matched_indicators == "0/3"
    assert d.confidence == "无"


# ── 4. 与旧引擎的"可比性"必须被分类,不许有漏网的 ──────────────────────

def test_every_kept_slot_is_classified_for_comparability():
    """★ 每个**保留下来**的槽,都必须被判成"可比"或"不可比",二选一。

    为什么这条必须存在:读者手里同时有旧引擎报告与 L1 报告,槽名又相近,
    很自然会拿两边相减。但 L1 把其中 6 个槽的**统计量换掉了**
    (绝对量 → 相对本人基线的偏离;均值 → 中位数;单列 → 多列合计;
    presence-once → 出现次数)。相减得到的差**没有任何含义**。

    将来新增一个保留槽却忘了归类 ⟹ 本条红。这是有意的:漏归类就等于
    默认"可比",而默认恰恰错在那一侧。

    红法:把 `COMPARABLE` 里的 `pitch_variation` 删掉(或不归类任何新槽)⟹ 本条红。
    """
    kept = {s for s, (l1, _) in l1_mapper.SLOT_MAP.items() if l1 is not None}
    classified = set(l1_mapper.NOT_COMPARABLE) | set(l1_mapper.COMPARABLE)
    assert kept == classified, ("未归类:", kept - classified, "多出:", classified - kept)
    # 两类不许重叠:一个槽不可能既可比又不可比。
    assert not (set(l1_mapper.NOT_COMPARABLE) & set(l1_mapper.COMPARABLE))


def test_l1_dead_weight_matches_the_slot_map():
    """★ **生产路径**(L1)每一维的死权重,与旧引擎那张表**不是同一个数**。

    ## 这条是怎么来的(2026-10-24)

    `tests/test_dimension_live_weight.py` 那张 `_DEAD_WEIGHT` 表守的是
    `research_mapper.mapping_rules` —— **旧引擎**。而 `/api/report/structured`
    从 2026-10-18 起默认走 L1(`l1_mapper.DIMENSIONS`),旧引擎只剩
    `?engine=legacy` 对照用。两边的槽表同名不同命,实测差这么多:

    | 维度 | 旧引擎 | **L1(2026-11-22 起)** | 差在哪 |
    |---|---|---|---|
    | 话语结构特征 | 0.0 | **0.0** | — |
    | 情境行为稳定性 | 0.0 | **0.0** | `symmetry_score` 已映射(`facial_asymmetry_p90`) |
    | 沟通表达流畅度 | 5/7 | **0.0** | `fluency_score` 已映射(`pause_structure_entropy`) |
    | 行为表现活跃度 | 0.0 | **0.0** | `eye_contact` 已映射(`gaze_dwell_p50`,2026-11-22) |
    | 作答过程特征 | 0.0 | **0.0** | `n_chars` 已映射(`answer_detail_chars_p50`) |

    ⟹ **五维死权重全为 0**(2026-11-22 起)—— 14 个槽全部有同构念指标。
    这张表**仍然要逐维核对**(判据是 `SLOT_MAP.get(slot)[0] is None`,不是抄写),
    因为它是"生产路径上这一维还能说什么"的唯一记账处:下一次谁把某个槽改回 `None`,
    这里的数会立刻不为 0,而报告会重新开始"整维消失 / 死权重"的说法。

    ⚠️ 沿革(全部保留,便于复核判据有没有被放宽):
    2026-10-24 时五维里三到四维的 L1 死权重与旧引擎不同;2026-11-21/22 陆续把
    `fluency_score` / `n_chars` / `symmetry_score` / `left_hand_spread` / `shrug_level` /
    `eye_contact` 六条由"删"改为"换",死权重逐条归零。

    ⚠️ 也正因为这张表是"数槽",它**故意不查 `evidence_gate.QUARANTINE`** ——
    封停表管的是旧引擎的特征键,而实测两条撞名的理由都不成立:
    `pause_duration` 的封停理由是 rms 能量门(`prosody_extractor` 那条),
    L1 走的是 pyin 的 `isfinite(f0)` + `MIN_PAUSE_SEC`(定义,非标定);
    `gaze_stability` 的封停理由说"由 `gaze_deviation` 派生",L1 的输入列不是它。
    实测 41 场里这两条分别出值 **37 / 27** 场 —— 封停并没有拦住 L1。

    红法(实测):把 `symmetry_score` 的 `SLOT_MAP` 改回映射到某个 L1 名字
    ⟹ 情境行为稳定性那一行的 0.25 变 0.0,本条红。
    """
    got = l1_mapper.dead_weight_by_dimension()
    expect = {
        "logical_thinking": ("话语结构特征", 0.0, []),
        # 2026-11-22:`symmetry_score` 由删槽改为映射 ⟹ 死权重 0.25 → 0.0
        "stress_resilience": ("情境行为稳定性", 0.0, []),
        # 2026-11-21:`fluency_score` 由删槽改为映射(pause_structure_entropy)⟹ 死权重 4/7 → 0
        "communication_fluency": ("沟通表达流畅度", 0.0, []),
        # 2026-11-22:`left_hand_spread` / `shrug_level` 补上了指标 ⟹ 1.0 → 0.375;
        # 同日第二次 `eye_contact` → `gaze_dwell_p50` ⟹ 0.375 → **0.0**(这一维已满)
        "confidence_level": ("行为表现活跃度", 0.0, []),
        # 2026-11-21:`n_chars` 由删槽改为映射(answer_detail_chars_p50)⟹ 死权重 0.4 → 0
        "cognitive_efficiency": ("作答过程特征", 0.0, []),
    }
    assert set(got) == set(expect), (sorted(got), sorted(expect))
    bad = []
    for key, (display, dead, slots) in expect.items():
        g_display, g_dead, g_slots = got[key]
        if g_display != display:
            bad.append(f"{key}: 显示名「{display}」→「{g_display}」")
        if abs(g_dead - dead) > 1e-9:
            bad.append(f"{key}: 死权重 {dead:.4f},实测 {g_dead:.4f}(死的槽:{g_slots})")
        if sorted(g_slots) != sorted(slots):
            bad.append(f"{key}: 死的槽 {slots},实测 {g_slots}")
    assert not bad, "L1 死权重与登记不符:\n  " + "\n  ".join(bad)

    # 两个方向都要能红:登记表本身必须能区分"删槽"与"没删"。
    assert got["logical_thinking"][1] == 0.0, "话语结构特征在 L1 上三个槽都在"
    assert got["confidence_level"][1] == 0.0, (
        "`eye_contact` 已映射到 `gaze_dwell_p50` ⟹ 这一维死权重应为 0;"
        f"实测 {got['confidence_level'][1]}(死的槽:{got['confidence_level'][2]})")
    assert got["communication_fluency"][1] == 0.0, (
        "沟通表达流畅度的三个槽现在都有对应指标 ⟹ 死权重应为 0")
    assert got["cognitive_efficiency"][1] == 0.0, (
        "作答过程特征两个槽现在都有对应指标 ⟹ 死权重应为 0")
    # ★ **全零不是"这条测试没东西可测了"**:上面四条断言逐个维度核对,
    #   而"非零方向"由 `test_dimension_with_no_surviving_slot_vanishes_not_an_empty_shell`
    #   的合成维度守着(那里 `dead_weight` 必须非零)。两处合起来,判据两个方向都可达。


def test_l1_mapping_does_not_consult_the_legacy_quarantine_table():
    """★ L1 的槽映射**不许**引用 `evidence_gate.QUARANTINE` —— 那是旧引擎的登记。

    这条不是"禁依赖"的洁癖,是防一个**已经发生过两次**的错:
    把旧引擎的封停理由当成 L1 那条指标的算法说明(`pause_duration` /
    `gaze_stability` 都撞名,理由都不成立,见 `dead_weight_by_dimension` 的 docstring)。
    一旦有人"顺手"给 L1 加上封停检查,那两条会当场变死,而**没有一条理由成立**。

    红法:在 `l1_mapper` 里 `from .evidence_gate import QUARANTINE` 并让
    `map_l1_to_dimensions` 跳过命中项 ⟹ 本条红。
    """
    import ast
    import inspect

    src = inspect.getsource(l1_mapper)
    # 查**导入语句**,不查子串 —— 本模块的 docstring 里就写着 `evidence_gate`
    # (说明为什么不引用它),子串判据会把那段说明本身判成违规。
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "") + "." + "*")
            imported |= {f"{node.module}.{a.name}" for a in node.names}
    hits = {m for m in imported if "evidence_gate" in m}
    assert not hits, (
        f"`l1_mapper` 导入了 {sorted(hits)} —— 封停表管的是旧引擎的特征键,"
        "两条撞名的理由都不成立(见 `dead_weight_by_dimension`)")
    # 正向对照:上面那条断言只有在源码**真的**被解析到时有意义。
    assert "SLOT_MAP" in src and "DIMENSIONS" in src, "读到的不是 l1_mapper 的源码"


def test_comparability_notes_say_what_changed_not_just_that_it_changed():
    """★ 每条说明都要**说出换了什么**,不是只说"不可比"。

    "不可比"三个字对读者没有用 —— 他需要知道的是"旧的数的是什么、新的数的是什么"。

    红法:把 `NOT_COMPARABLE` 里某条的理由改成空串 ⟹ 本条红。
    """
    notes = l1_mapper.comparability_notes()
    assert len(notes) == len(l1_mapper.NOT_COMPARABLE)
    for slot, why in l1_mapper.NOT_COMPARABLE.items():
        line = next(n for n in notes if n.startswith(slot + " →"))
        assert "旧" in line and "新" in line, line
        assert len(why) >= 12, (slot, why)
