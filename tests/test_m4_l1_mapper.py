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
    assert len(seen) == 13, f"幸存槽应 13 个,实得 {len(seen)}:{sorted(seen)}"


@pytest.mark.parametrize("slot,why_fragment", [
    # ⚠️ 2026-11-21:只剩**一个**真删槽。`fluency_score` 与 `n_chars` 已改判为
    #    "换"(见 SLOT_MAP);`eye_contact` 仍然删,但理由从"关系未确认"改成
    #    "已由注视偏离占比覆盖(补集关系),不重复产出" —— 原话"这套采集没有测法"
    #    是错的,`gaze_aversion_ratio_vs_baseline` 数的就是同一件事的另一面。
    ("eye_contact", "补集"),
])
def test_construct_mismatched_slots_are_dropped_with_a_reason(slot, why_fragment):
    """★ 仍然**没有**同构念 L1 指标的槽,必须删,且理由要说到点子上。

    ## 2026-11-22:由 6 个变 3 个(**这是被实测推翻,不是放宽标准**)

    退出去的三条(`symmetry_score` / `left_hand_spread` / `shrug_level`)当年判
    "没有同构念的 L1 指标"是对的 —— 那时确实没人给那三列写过指标。
    但它们要的列**一直在**(`symmetry_score` 44/55 份日志有数、
    `shrug_level` 50/52、`hand_spread` 38/52),所以判定的对象错了:
    缺的是"指标",不是"数据"。补上三条指标之后这三条**不再是删槽**,
    删槽的理由(「L1 没有面部对称性指标」)同时成了假话,所以一并删掉。

    留下来这三条各自仍然因为**构念不同**而删(不是"这一场没测到"):
    流畅度=零额外信息、眼神接触=关系未确认、详尽度=速率不是总量。
    判据一字未改。

    红法:把 `eye_contact` 映射到 `gaze_aversion_ratio_vs_baseline`
    (看起来"都是看哪儿")⟹ 本条红。那正是本仓历史上把"皱眉"配到垃圾值的那类错误。
    """
    l1, reason = l1_mapper.SLOT_MAP[slot]
    assert l1 is None, f"{slot} 不该被映射到 {l1}"
    assert why_fragment in reason


def test_thirteen_slots_survive_and_one_is_dropped():
    """实测对账结论:**13 换 1 删**(2026-10-18 由 7/7 变 8/6;2026-11-22 变 11/3;
    2026-11-21 变 13/1 —— `fluency_score` 换成 `pause_structure_entropy`、
    `n_chars` 换成 `answer_detail_chars_p50`,只剩 `eye_contact` 还是删)。

    数字变了就说明对账被改过,必须重新说明 —— 这就是这条测试的作用。
    2026-11-22 这 +3 来自三条新指标(`facial_asymmetry_p90` / `hand_openness_p50` /
    `shrug_elevation_p90`),它们各自读一列**已落盘**的真测量;复核证据见
    `docs/补强/patch/D1_指标替换_3退3进_说明.md`。

    红法:把 `connective_density` 改回 `None` ⟹ 本条红(回到 10/4)。
    """
    kept = [s for s, (l1, _) in l1_mapper.SLOT_MAP.items() if l1 is not None]
    dropped = [s for s, (l1, _) in l1_mapper.SLOT_MAP.items() if l1 is None]
    assert len(kept) == 13 and len(dropped) == 1, (kept, dropped)
    assert "connective_density" in kept


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
    """★ 2026-11-22:**`行为表现活跃度` 复活了** —— 3 个槽里 2 个有了对应指标。

    ## 这条为什么必须存在

    上一版这一维整维消失(三个槽全删),而"整维消失"是一句**很强的话**
    ("这个维度量的东西当前管线都不测")。现在补进 `shrug_elevation_p90` 与
    `hand_openness_p50` 之后那句话成了假话 ⟹ 必须在**真表**上钉住它已经复活,
    否则下一个人可能把两条新指标撤掉而没人发现:报告会退回"四维",
    而那条测试(用构造维度)照样绿。

    红法:把 `shrug_level` 或 `left_hand_spread` 改回 `None` ⟹ 本条红。
    """
    dims, vanished = l1_mapper.map_l1_to_dimensions([])
    assert "confidence_level" in dims, f"这一维又消失了:{[v['key'] for v in vanished]}"
    assert "confidence_level" not in [v["key"] for v in vanished]
    d = dims["confidence_level"]
    assert d.n_slots == 3, d.n_slots
    # 顺序 = `DIMENSIONS["confidence_level"]["slots"]` 的声明顺序(权重仍是 0.375/0.25/0.375)
    assert [s["l1"] for s in d.slots] == ["hand_openness_p50", "shrug_elevation_p90"]
    assert [x["slot"] for x in d.dropped] == ["eye_contact"]
    # 分母仍是"旧槽总数"(3):读者要比的是"原来 3 个槽,今天过了几个"。
    assert d.matched_indicators.startswith("0/3")


def test_dropped_slot_reasons_say_construct_mismatch_not_a_coverage_excuse():
    """删槽理由必须是「**构念不同**」,不能是「这个没测到」。

    ⚠️ 这条原先写成

        assert "左" not in vanished[0]["reason"] or "可见性" in vanished[0]["reason"]

    —— **恒真**,而且被删槽理由自己证伪了:理由串里既有「左」(来自
    `left_hand_spread`)又有「可见性」,于是 `or` 右边永远成立,任何实现都通过。
    它声称守的那条判据一次都没生效过(2026-10-23 `488c004` 引入至今)。

    区别为什么重要:读者看到「没测到」会等下一次采集;真相是这条管线**不测这个构念**,
    再采也不会有。两种话在报告里长得一样,含义相反。

    红法:把任一槽的 reason 改成 `"左手没测到"` ⟹ 第一条断言红(没点名任何真实 L1);
    改成 `"hand_visible_ratio_l 未采集"` ⟹ 第二条断言红。
    """
    # 2026-11-21:真删槽只剩 `eye_contact`(另两条已改判为"换"),所以直接验真表,
    # 不再借用合成维度 —— 合成键的 reason 是"未登记的槽",验不了这条判据。
    dropped = [{"slot": s, "reason": r}
               for s, (l1, r) in l1_mapper.SLOT_MAP.items() if l1 is None]
    assert len(dropped) == 1 and dropped[0]["slot"] == "eye_contact", (
        f"真删槽集合变了:{dropped} —— 若某槽恢复映射,请同步这条测试")

    known = _all_l1_names()
    for d in dropped:
        reason = d["reason"]
        assert any(n in reason for n in known), (
            f"{d['slot']} 的删槽理由没点名任何真实 L1 指标 —— "
            f"读者无从知道「拿谁来顶替被否决了」:{reason!r}")
        for excuse in ("没测到", "未采集", "无数据", "没有数据"):
            assert excuse not in reason, (
                f"{d['slot']} 用**覆盖**借口冒充构念理由:{reason!r}")


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
    | 情境行为稳定性 | 0.0 | **0.0** | `symmetry_score` 已映射(`facial_asymmetry_p90`)⟹ 死权重归零 |
    | 沟通表达流畅度 | 5/7 | **4/7** | L1 只删 `fluency_score` 一条 |
    | 行为表现活跃度 | 0.0 | **0.375** | 只剩 `eye_contact` 死;另两个槽补上了指标(**维度不再整体消失**) |
    | 作答过程特征 | 0.0 | **0.4** | `n_chars` 在 L1 没有同构念 |

    ⟹ 五维里**两维**不能拿旧表回答"生产路径上这一维能说什么"(沟通表达 / 作答过程)。
    ⚠️ 2026-11-22 之前是三到四维 —— 三条新指标把「情境行为稳定性」与
    「行为表现活跃度」的死权重压了下去,这张表的"差在哪"因此变窄了。**变窄也是事实,
    要跟着改**,否则读者会按一张过期的差值表去解释生产路径。

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
        # 2026-11-22:`left_hand_spread` / `shrug_level` 补上了指标 ⟹
        # 死权重 1.0(整维消失)→ 0.375(只剩 `eye_contact` 那一份)
        "confidence_level": ("行为表现活跃度", 0.375, ["eye_contact"]),
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
    assert got["confidence_level"][1] == 0.375, (
        "行为表现活跃度在 L1 上还剩 `eye_contact` 一个删槽 ⟹ 死权重必须是 0.375;"
        f"实测 {got['confidence_level'][1]}")
    assert got["logical_thinking"][1] == 0.0, "话语结构特征在 L1 上三个槽都在"
    # 反方向对照:现在**只剩一个**非零死权重的维度(`confidence_level` 的 eye_contact)。
    # 2026-11-21 之后 `communication_fluency` / `cognitive_efficiency` 都已归零
    # (fluency_score → pause_structure_entropy、n_chars → answer_detail_chars_p50)。
    assert got["communication_fluency"][1] == 0.0, (
        "沟通表达流畅度的三个槽现在都有对应指标 ⟹ 死权重应为 0")
    assert got["cognitive_efficiency"][1] == 0.0, (
        "作答过程特征两个槽现在都有对应指标 ⟹ 死权重应为 0")


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
