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

@pytest.mark.parametrize("slot,why_fragment", [
    ("symmetry_score", "对称"),
    # ⚠️ 2026-10-24 由「产出方」改为「零额外信息」:这一条的结论从
    #    "全系统没人算它"升级成"**不该有人算它**" —— VAD 流畅度的三个分量
    #    在本仓都已有现成指标,其中「平均连续发声长度」与 `pause_rate_per_min`
    #    是**代数恒等**(实测残差 4.4e-16)。理由的完整推导见「甲、3」。
    ("fluency_score", "零额外信息"),
    ("left_hand_spread", "可见性"),
    ("shrug_level", "稳定性"),
    ("eye_contact", "未确认"),
    ("n_chars", "速率"),
])
def test_construct_mismatched_slots_are_dropped_with_a_reason(slot, why_fragment):
    """★ 这 6 个槽**没有**同构念的 L1 指标,必须删,且理由要说到点子上。

    (`connective_density` 原在本列表里,2026-10-18 T 族落地后改为映射到
    `discourse_marker_rate` —— 口径有差异,见 `SLOT_MAP` 里的 ⚠️。)

    红法:把 `left_hand_spread` 映射到 `hand_visible_ratio_l`(看起来"都是手")
    ⟹ 本条红。那正是本仓历史上把"皱眉"配到垃圾值的那类错误。
    """
    l1, reason = l1_mapper.SLOT_MAP[slot]
    assert l1 is None, f"{slot} 不该被映射到 {l1}"
    assert why_fragment in reason


def test_eight_slots_survive_and_six_are_dropped():
    """实测对账结论:**8 换 6 删**(2026-10-18 T 族落地后由 7/7 变为 8/6)。

    数字变了就说明对账被改过,必须重新说明 —— 这就是这条测试的作用。

    红法:把 `connective_density` 改回 `None` ⟹ 本条红(回到 7/7)。
    """
    kept = [s for s, (l1, _) in l1_mapper.SLOT_MAP.items() if l1 is not None]
    dropped = [s for s, (l1, _) in l1_mapper.SLOT_MAP.items() if l1 is None]
    assert len(kept) == 8 and len(dropped) == 6, (kept, dropped)
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

def test_dimension_with_no_surviving_slot_vanishes_not_an_empty_shell():
    """★ `confidence_level` 三个槽全删 ⟹ 维度**整体消失**,不留 `0/0` 空壳。

    红法:把 `if not res.slots: continue` 删掉 ⟹ 会返回一个 `matched_indicators="0/0"`
    的维度,报告里就会多出一行"行为表现活跃度 0/0" —— 读者会以为"本场没测到",
    而真相是"这个维度量的东西当前管线都不测"。
    """
    dims, vanished = l1_mapper.map_l1_to_dimensions([])
    assert "confidence_level" not in dims
    assert [v["key"] for v in vanished] == ["confidence_level"]
    assert "左" not in vanished[0]["reason"] or "可见性" in vanished[0]["reason"]


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
    """删掉的槽必须能被读者看见 —— 它们进 `evidence_gaps`,不是消失得无影无踪。"""
    dims, _ = l1_mapper.map_l1_to_dimensions([])
    gaps = dims["stress_resilience"].to_dict()["evidence_gaps"]
    dropped = {g["slot"] for g in gaps if "slot" in g}
    assert "symmetry_score" in dropped


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

    | 维度 | 旧引擎 | **L1** | 差在哪 |
    |---|---|---|---|
    | 话语结构特征 | 0.0 | **0.0** | — |
    | 情境行为稳定性 | 0.0 | **0.25** | `symmetry_score` 在 L1 没有同构念 |
    | 沟通表达流畅度 | 5/7 | **4/7** | L1 只删 `fluency_score` 一条 |
    | 行为表现活跃度 | 0.0 | **1.0**(维度整体消失) | 三个槽在 L1 全删 |
    | 作答过程特征 | 0.0 | **0.4** | `n_chars` 在 L1 没有同构念 |

    ⟹ 五维里**四维**不能拿旧表回答"生产路径上这一维能说什么"。

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
        "stress_resilience": ("情境行为稳定性", 0.25, ["symmetry_score"]),
        "communication_fluency": ("沟通表达流畅度", 4 / 7, ["fluency_score"]),
        "confidence_level": ("行为表现活跃度", 1.0,
                             ["left_hand_spread", "shrug_level", "eye_contact"]),
        "cognitive_efficiency": ("作答过程特征", 0.4, ["n_chars"]),
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
    assert got["confidence_level"][1] == 1.0, (
        "行为表现活跃度在 L1 上三个槽全删 ⟹ 死权重必须是 1.0(维度整体消失);"
        f"实测 {got['confidence_level'][1]}")
    assert got["logical_thinking"][1] == 0.0, "话语结构特征在 L1 上三个槽都在"


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
