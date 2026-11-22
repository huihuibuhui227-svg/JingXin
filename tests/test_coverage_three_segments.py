# tests/test_coverage_three_segments.py
r"""覆盖数**三段式** —— `8/14` 这个会误导读者的聚合数,拆成读者能据以判断的三段。

## 失效现场(本仓实测)

真实那一场(`20260927_212143_95a8`,用 `LogDataLoader` 装配日志后跑 L1)算出来是
`n_passed = 8`、`n_slots = 14`,报告头印:

    8 / 14
    个指标槽通过证据门

读者只会读出一个意思:「只覆盖了 57%」。而 14 里有 **6 个槽永远填不上**:

* 3 个「这套采集不测」—— 面部对称性 / 语音流畅度 / 回答详尽度
  (`dimensions.*.evidence_gaps_not_measured`,**不是**这一场没量到);
* 3 个「整维没有对应指标」—— 整维「行为表现活跃度」被删(`vanished_dimensions`),
  这一维要的量当前没有任何 L1 指标承接。

⟹ 可测量的那 8 个槽其实是 **8/8 全过**。把 14 当分母,等于用"这套采集缺的东西"
给这一场扣分。

## ⚠️ 2026-11-22(本次):分母变成 14,第三段归零

补进最后一条指标 `gaze_dwell_p50`(`eye_contact` 槽的时长面)之后,
**14 个槽全部可测**:三段从 `11 可测 + 3 不测 + 0 无指标` 变成
**`14 可测 + 0 不测 + 0 无指标`**。上面那 8/14 的失效现场成了历史 ——
但三段式这套**口径**不能因为"今天没有缺口"就删掉:下一场缺指标时,
读者仍需要看见"不测"与"没量到"分开写。所以本文件继续钉三段之和 = `n_slots`、
分母为 0 时给 `None`、以及三个出口同一句话。

## 这个文件守什么

1. **三段之和恒等于 `n_slots`** —— 不是靠维护,是同一个加法换了个顺序
   (`DimensionResult.n_slots ≡ len(slots) + len(dropped)`);
2. `n_passed` 语义未变;"全都不测"(可测量槽 = 0)时 `pass_rate_of_measurable`
   是 `None` —— **不是 0,也不是 1**(0 会被读成"覆盖率 0%",1 会被读成"全部通过",
   两个都是假话);
3. HTML 里三段式那句话**出现**,而旧的那句(「8 / 14 项指标量到了可用的数」/
   「N / 14」)不得再单独出现 —— 两个出口口径分叉过一次(`n_passed` 曾虚报成 7);
4. `demo_run.py` 投屏那一行同口径;
5. 计数只有一处(`l1_report.coverage_counts()`),渲染层不重算。

## 盲区

本文件只测**计数与文字**,不测"哪些槽该被删" —— 那是 `l1_mapper.SLOT_MAP` 的事
(`tests/test_m4_l1_mapper.py` 守)。口径若在 `SLOT_MAP` 上变了,这里照样绿。
"""
from __future__ import annotations

import re

import pytest

import demo_run
from report_frontend import l1_mapper
from report_frontend.l1_report import (
    COVERAGE_EXPLANATION,
    build_l1_report,
    coverage_counts,
    coverage_segments,
    coverage_summary_line,
)
import report_frontend.l1_report as l1_report
from report_frontend.report_generator import ReportGenerator

_SID = "20260101_000000_0000"

#: 三段式那句话里的三个词。HTML / 命令行 / 载荷三处必须用同一套词,
#: 否则同一个数在三处有三种说法(`coverage.subscription` 之外的分叉本仓记过多次)。
_W_MEASURABLE = "可测量"
_W_NOT_MEASURED = "这套采集不测"
_W_NO_INDICATOR = "整维没有对应指标"


# ---------------------------------------------------------------- 造维度
def _dim(n_slots: int, n_dropped: int = 0) -> l1_mapper.DimensionResult:
    """一个维度,`n_slots` 个槽里 `n_dropped` 个没有对应指标。

    直接用 `DimensionResult` 而不是手写 dict:`n_slots` 是它的 property
    (`len(slots) + len(dropped)`),三段之和恒等于 `n_slots` 这条不变式正是靠它成立 ——
    自己编一个形状就把被测对象换掉了。
    """
    return l1_mapper.DimensionResult(
        key=f"d{n_slots}_{n_dropped}",
        display_name="人造维度",
        slots=[{"slot": f"s{i}", "l1": "x", "value": None} for i in range(n_slots)],
        dropped=[{"slot": f"d{i}", "reason": "这套采集不测"} for i in range(n_dropped)],
    )


def _real_shape() -> tuple[dict, list]:
    """真实的那一场**当前**的形状:5 维 14 槽,**全部可测**,整维不消失。

    沿革(全部保留,便于复核判据没被放宽):
    * 2026-11-21:补进 `shrug_elevation_p90` / `hand_openness_p50` /
      `facial_asymmetry_p90` ⟹ 三段 `8/3/3` → `11/3/0`;
    * 2026-11-22(本次):补进 `gaze_dwell_p50`(`eye_contact` 的时长面)⟹
      三段 `11/3/0` → **`14/0/0`**,`n_dropped` 3 → **0**;
    * `n_slots` 始终 14(槽总数没变,变的是"14 里有多少可测")。
    """
    # `_dim(存活槽, 被删槽)`:槽总数 = 存活 + 被删。14 存活 + 0 被删 = 14。
    dims = {
        "logical_thinking": _dim(3, 0),         # 话语结构 3/3
        "stress_resilience": _dim(3, 0),        # 情境行为 3/3
        "communication_fluency": _dim(3, 0),    # 沟通表达 3/3(流畅度已换成停顿时长熵)
        "confidence_level": _dim(3, 0),         # 行为表现活跃度 3/3(眼神接触已换成注视停留时长)
        "cognitive_efficiency": _dim(2, 0),     # 作答过程 2/2(详尽度已换成逐段字数中位)
    }
    vanished: list = []
    return dims, vanished


# ---------------------------------------------------------------- ① 三段之和
def test_the_three_segments_always_add_up_to_n_slots():
    """★ 三段之和 == `n_slots`,在四种极端形状上都成立。

    形状覆盖:全部可测 / 维度表内全部没有指标 / 整维全消失 / 真实形状。
    """
    shapes = {
        # 全部可测:8 个槽都有对应指标
        "全可测": ({"a": _dim(8, 0)}, []),
        # 维度表里的槽**全都**没有对应指标(可测量 = 0)
        "全不测": ({"a": _dim(0, 14)}, []),
        # 整维消失(可测量 = 0)
        "整维消失": ({}, [{"n_slots": 14, "dropped": []}]),
        # 真实那一场的形状
        "真实": _real_shape(),
    }
    for name, (dims, vanished) in shapes.items():
        counts = coverage_counts(dims, vanished, n_passed=0)
        total = (counts["n_measurable"] + counts["n_not_measured"]
                 + counts["n_no_indicator"])
        assert total == counts["n_slots"], (
            f"「{name}」三段相加 {total} != n_slots {counts['n_slots']} —— 分母口径分叉了")
        assert counts["n_dropped"] == (counts["n_not_measured"]
                                       + counts["n_no_indicator"]), (
            f"「{name}」n_dropped 与两段缺口之和对不上 —— 原有键的语义变了")


def test_the_real_shape_splits_into_14_0_0():
    """真实形状:可测量 **14** · 这套采集不测 **0** · 整维没有对应指标 **0**。

    合起来仍是 **14**(槽总数没变)。2026-11-21 之前是 `8/3/3`,2026-11-21 是 `11/3/0`,
    本次(补 `gaze_dwell_p50`)是 `14/0/0` —— 三次都是**事实变了**,不是判据放宽。

    ⚠️ **"可测 14"不等于"这一场 14 全过"**:可测只是说"有指标承接",指标这一场
    有没有值要另看(参考场次实测 `13/14`,`answer_detail_chars_p50` 在无转写时
    报 `not_stored`)。所以下面用 `n_passed=14` 只验"全过时通过率是 1.0"这一侧,
    "未必全过"那一侧由 `test_pass_rate_follows_the_data_not_the_denominator` 单独钉。
    """
    dims, vanished = _real_shape()
    counts = coverage_counts(dims, vanished, n_passed=14)
    assert counts["n_measurable"] == 14
    assert counts["n_not_measured"] == 0
    assert counts["n_no_indicator"] == 0
    assert counts["n_slots"] == 14
    assert counts["n_dropped"] == 0, "本次之后没有删槽 —— n_dropped 必须是 0"
    assert counts["n_measurable_no_data"] == 0, "14 个可测槽全过时不该有「没量到」的槽"
    assert counts["pass_rate_of_measurable"] == 1.0


def test_pass_rate_follows_the_data_not_the_denominator():
    """★ 可测 14 ≠ 全过 14:分母不缩水,分子照实数。

    参考场次实测就是这个形状(`20260927_212143_95a8` / `20261026_002930_cb79`
    都是 13 可测槽通过 + 1 个 `not_stored`)。若哪天有人把 `n_passed` 直接改成
    `n_measurable`("反正都可测了"),本条当场红 —— 那是把"可测"说成了"通过"。
    """
    dims, vanished = _real_shape()
    counts = coverage_counts(dims, vanished, n_passed=13)
    assert counts["n_measurable"] == 14
    assert counts["n_measurable_no_data"] == 1, (
        f"13/14 时「可测但没量到」应是 1 槽,实得 {counts['n_measurable_no_data']}")
    assert counts["pass_rate_of_measurable"] == pytest.approx(13 / 14)
    line = coverage_summary_line(14, 13, 0, 0)
    assert "13/14" in line, f"那句话没把 13/14 说出来:{line}"
    assert "全部通过" not in line, f"13/14 被说成了「全部通过」:{line}"


# ---------------------------------------------------------------- ② 极端载荷
def test_when_nothing_is_measurable_the_pass_rate_is_none_not_zero():
    """★ 分母为 0 ⟹ `None`。给 0 或 1 都是在替读者下结论。

    这一条是本轮的**行为判据**:真实场次不会出现"一个可测槽都没有",所以它只能
    在造出来的载荷上验 —— 抽 `coverage_counts()` 出来正是为了这个。
    """
    for name, (dims, vanished) in {
        "全不测": ({"a": _dim(0, 14)}, []),
        "整维消失": ({}, [{"n_slots": 14, "dropped": []}]),
        "两条都是零": ({}, []),
    }.items():
        counts = coverage_counts(dims, vanished, n_passed=0)
        assert counts["n_measurable"] == 0, f"「{name}」的可测量槽不是 0"
        assert counts["pass_rate_of_measurable"] is None, (
            f"「{name}」分母为 0 时通过率是 {counts['pass_rate_of_measurable']!r} —— "
            f"必须是 None(0 会被读成 0%,1 会被读成 100%)")
        assert counts["n_passed"] == 0, "n_passed 在极端载荷下必须是真值 0"


def test_the_summary_line_says_the_denominator_is_zero_instead_of_a_percentage():
    """分母为 0 那句话:说「不适用」,不印百分比、不说「全部通过」。"""
    line = coverage_summary_line(0, 0, 14, 0)
    assert "不适用" in line, f"分母为 0 时那句话没有说「不适用」:{line}"
    assert "%" not in line, f"分母为 0 的那句话里出现了百分比:{line}"
    assert "全部通过" not in line, f"分母为 0 时说了「全部通过」—— 那是 0/0 的假话:{line}"


def test_a_partial_measurable_count_keeps_the_denominator_honest():
    """可测 14 槽只过 11 槽:分母**不缩水**,不许偷偷改成 11/11。

    ⚠️ 2026-11-22:分母由 11 变 14(14 槽全可测),被「不缩水分母」这条判据本身没动。
    """
    line = coverage_summary_line(14, 11, 0, 0)
    assert "11/14" in line, f"部分通过时没有把 14 留在分母里:{line}"
    assert "全部通过" not in line, f"11/14 被说成了「全部通过」:{line}"
    assert "11/11" not in line, f"分母被改成了过门数:{line}"


def test_coverage_segments_give_none_to_the_two_gap_segments():
    """后两段没有"通过"这回事 ⟹ `n_passed` 是 `None`,不是 0。"""
    segs = coverage_segments(8, 8, 0, 3, 3)
    by_key = {s["key"]: s for s in segs}
    assert set(by_key) == {"measurable", "not_measured", "no_indicator"}, (
        f"三段的键不对:{sorted(by_key)}")
    assert by_key["measurable"]["n_passed"] == 8
    assert by_key["not_measured"]["n_passed"] is None, (
        "「这套采集不测」那段给了 n_passed —— 前端会印出「通过 0 / 3」")
    assert by_key["no_indicator"]["n_passed"] is None
    assert sum(s["n"] for s in segs) == 14, "三段 n 相加不是 14"


# ---------------------------------------------------------------- 载荷接线
class _StubQuality:
    """`build_quality_report` 的替身 —— 只给 `build_l1_report` 真正读的那几个属性。"""

    n_total = 0
    n_measured = 0
    by_confidence: dict = {}
    by_gap: dict = {}
    indicators: list = []
    coverage: dict = {}
    timebase: dict = {}
    windows: dict = {}
    # `modality_summary()` 按族汇总 —— 没有族就是"这一份没有逐模态产出"。
    families: dict = {}


@pytest.fixture
def empty_payload(monkeypatch) -> dict:
    """**空数据**下走一遍真的 `build_l1_report`(L1 与质量报告换成替身)。

    为什么不用真实场次:`data/logs/` 不进仓库(只有 `.gitkeep`),拿它当测试前提,
    换一台机器就红。上游换成替身之后,`l1_mapper.map_l1_to_dimensions` 与
    `coverage_counts` 都还是**真的** —— 被测的正是这两处。
    """
    class _Result:
        indicators: list = []
        # `gaps` / `dispersion` 原样进载荷(空值即可,本文件不测这两块)。
        gaps: list = []
        dispersion: dict = {}

    monkeypatch.setattr(l1_report, "compute_session_l1",
                        lambda *a, **k: _Result())
    monkeypatch.setattr(l1_report, "build_quality_report", lambda *a, **k: _StubQuality())
    return build_l1_report(_SID)


def test_the_payload_carries_the_three_segments_and_keeps_the_old_keys(empty_payload):
    """载荷:新键齐了,旧键(`n_slots` / `n_passed` / `n_dropped`)语义没动。

    空数据下(一条指标都没有)真实算出来是 `0 / 11`,三段 `11 + 3 + 0 = 14`。
    """
    cov = empty_payload["coverage"]
    for key in ("n_slots", "n_passed", "n_dropped", "confidence_cap", "passed_slots"):
        assert key in cov, f"原有键 {key} 不见了 —— 前端与材料都在读它"
    assert cov["n_slots"] == 14, f"空数据下 n_slots 不是 14:{cov['n_slots']}"
    assert cov["n_passed"] == 0, f"空数据下 n_passed 不是 0:{cov['n_passed']}"
    # ⚠️ 2026-11-22:3 → 0。最后一个删槽 `eye_contact` 已换成 `gaze_dwell_p50`,
    #    14 个槽全部有对应指标 ⟹ 空数据下"这套采集不测"是 0 槽。
    assert cov["n_dropped"] == 0, f"空数据下 n_dropped 不是 0:{cov['n_dropped']}"
    assert (cov["n_measurable"] + cov["n_not_measured"] + cov["n_no_indicator"]
            == cov["n_slots"]), "载荷里三段相加不等于 n_slots"
    assert cov["n_measurable"] == 14
    assert cov["n_not_measured"] == 0
    assert cov["n_no_indicator"] == 0
    # 空数据是"可测但没量到",**不是**"没有可测量的槽":通过率是真 0.0,不是 None。
    assert cov["n_measurable_no_data"] == 14
    assert cov["pass_rate_of_measurable"] == 0.0, (
        f"可测 14 槽一个都没过时通过率应是 0.0,实际 {cov['pass_rate_of_measurable']!r}")
    assert cov["explanation"] == COVERAGE_EXPLANATION
    assert len(cov["segments"]) == 3


def test_the_dispatcher_and_the_segments_are_the_same_numbers(empty_payload):
    """`summary_line` / `segments` / 三个分段数**说的必须是同一批数**。

    这一条守的是"三个出口各拼一遍"的旧形态:措辞可以不同,数字不能不同。
    """
    cov = empty_payload["coverage"]
    segs = {s["key"]: s for s in cov["segments"]}
    assert segs["measurable"]["n"] == cov["n_measurable"]
    assert segs["not_measured"]["n"] == cov["n_not_measured"]
    assert segs["no_indicator"]["n"] == cov["n_no_indicator"]
    for word in (_W_MEASURABLE, _W_NOT_MEASURED, _W_NO_INDICATOR):
        assert word in cov["summary_line"], f"那句话里没有「{word}」:{cov['summary_line']}"


# ---------------------------------------------------------------- ③ HTML
def _payload(coverage: dict) -> dict:
    """一份能渲染的载荷。`coverage` 由调用方给 —— 结论句必须跟着它变。"""
    return {
        "engine": "l1",
        "dimensions": {
            "logical_thinking": {
                "display_name": "话语结构特征", "description": "", "algorithm": "",
                "evidence_chain": [
                    {"feature": "discourse_marker_rate", "human_name": "连接词密度",
                     "raw_value": 3.4, "n_valid": 301, "observed_interval": []},
                ],
                "evidence_gaps": [],
                "evidence_gaps_not_measured": [],
                "confidence": "中", "matched_indicators": "1/3", "stats": {},
            },
            "cognitive_efficiency": {
                "display_name": "作答过程特征", "description": "", "algorithm": "",
                "evidence_chain": [],
                "evidence_gaps": ["作答时长:本场没有可用数"],
                "evidence_gaps_not_measured": ["回答详尽度:这一条当前这套采集不测"],
                "confidence": "无", "matched_indicators": "0/2", "stats": {},
            },
        },
        "vanished_dimensions": [{
            "key": "confidence_level", "display_name": "行为表现活跃度", "n_slots": 3,
            "dropped": [{"slot": "left_hand_spread", "human_name": "手掌张开度",
                         "plain_reason": "这套采集只测手有没有被看到", "reason": "x"}],
            "reason": "x",
        }],
        "summary_narrative": "未产出综合评分,也不给评级。",
        "coverage": coverage,
        "model_metadata": {"version": "l1-2", "timestamp": "2026-11-21T00:00:00"},
        "l1": {"n_total": 38, "n_measured": 35, "by_confidence": {}, "by_gap": {},
               "modality_coverage": {}, "modality_summary": {},
               "timebase": {}, "windows": {"granularity": "answer", "n_questions": 8},
               "indicators": [], "gaps": [], "comparability_notes": [],
               "comparability_notes_plain": [], "low_confidence_notes": [],
               "low_confidence_notes_plain": [], "dispersion": {}},
    }


def _full_coverage(**over) -> dict:
    """三段齐全的 `coverage`(默认 = 真实那一场 **14/0/0**;14 槽全可测且全过)。"""
    base = {
        "n_slots": 14, "n_passed": 14, "n_dropped": 0,
        "n_measurable": 14, "n_measurable_no_data": 0,
        "n_not_measured": 0, "n_no_indicator": 0,
        "pass_rate_of_measurable": 1.0,
        "summary_line": coverage_summary_line(14, 14, 0, 0),
        "explanation": COVERAGE_EXPLANATION,
        "segments": coverage_segments(14, 14, 0, 0, 0),
        "confidence_cap": "高",
        "passed_slots": [],
    }
    base.update(over)
    return base


def _render(tmp_path, coverage: dict) -> str:
    gen = ReportGenerator(output_dir=str(tmp_path))
    return gen._build_html_report(_payload(coverage), {}, {}, {}, [],
                                  sources={}, target=_SID)


def _text_of(html: str) -> str:
    body = re.sub(r"(?s)<(script|style).*?</\1>", "", html)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body))


def test_the_html_states_the_three_segments(tmp_path):
    """★ HTML 里三段式那句话在,而且解释那句也在。"""
    html = _render(tmp_path, _full_coverage())
    text = _text_of(html)
    for word in (_W_MEASURABLE, _W_NOT_MEASURED, _W_NO_INDICATOR):
        assert word in text, f"报告正文里没有「{word}」:{text[:400]}"
    assert "14/14" in text, "三段式那句话里没有 14/14(可测量 14 槽全过)"
    # 给人看的那句解释:说清"不测 ≠ 这一场没量到"
    assert "不是这一场没量到" in text, (
        f"没有那句「不测的原因不是这一场没量到」:{text[:600]}")
    assert "没有任何一条指标承接" in text, "没解释「整维没有对应指标」是什么意思"


def test_the_html_no_longer_prints_the_old_aggregate(tmp_path):
    """★ 旧那句(「N / 14 项指标量到了可用的数」)不得再单独出现。

    它和新那三段并存过一次就是口径分叉 —— 两个数都在,读者只会信更醒目的那个。

    ## ⚠️ 2026-11-22:判据拆成两半,因为"分母恰好等于槽总数"了

    原判据是「正文里不许出现 `N / 14`」。14 个槽全部可测之后,`14 / 14`
    是**正确**的那个数,再把这条写成"出现 14 就红"就是拿判据去反对事实。
    但"分母必须是可测量槽、不是槽总数"这条判据本身**没变**,于是拆成:
    ① 今天的真形状(14/0/0)⟹ 大数必须是 `14 / 14`,旧措辞不许出现;
    ② 一个人造形状(可测 11 + 不测 3,即 2026-11-22 之前的样子)⟹ 大数必须是
       `11 / 11`,正文里**不许**出现「N / 14」。②这一半才是原来那条判据的实体。
    """
    html = _render(tmp_path, _full_coverage())
    text = _text_of(html)
    assert "项指标量到了可用的数" not in text, (
        "报告头还是旧的措辞(拿 n_slots 当分母)")
    assert re.search(r'cover-number">14 / 14<', html), (
        "报告头的大数不是「过门 / 可测量」")

    # ② 分母是可测量槽(11),不是槽总数(14) —— 旧判据的实体。
    cov11 = _full_coverage(
        n_passed=11, n_dropped=3, n_measurable=11, n_not_measured=3,
        summary_line=coverage_summary_line(11, 11, 3, 0),
        segments=coverage_segments(11, 11, 0, 3, 0),
    )
    html11 = _render(tmp_path / "eleven", cov11)
    text11 = _text_of(html11)
    assert not re.search(r"\d+\s*/\s*14", text11), (
        f"可测 11 槽的载荷里出现了「N / 14」—— 那是把槽总数当分母:{text11[:400]}")
    assert re.search(r'cover-number">11 / 11<', html11), (
        "报告头的大数不是「过门 / 可测量」")


def test_the_html_says_the_rate_is_not_applicable_when_nothing_is_measurable(tmp_path):
    """★ 极端载荷(全都不测):大数印「—」,那句话说不适用,不许出现 0%/100%。"""
    cov = _full_coverage(
        n_slots=14, n_passed=0, n_dropped=14,
        n_measurable=0, n_measurable_no_data=0,
        n_not_measured=14, n_no_indicator=0,
        pass_rate_of_measurable=None,
        summary_line=coverage_summary_line(0, 0, 14, 0),
        segments=coverage_segments(0, 0, 0, 14, 0),
        confidence_cap="无",
    )
    html = _render(tmp_path, cov)
    text = _text_of(html)
    assert re.search(r'cover-number">—<', html), (
        "可测量槽为 0 时报告头没有印「—」—— 会印出「0 / 0」")
    assert "不适用" in text, "分母为 0 时没有说「不适用」"
    assert "0%" not in text and "100%" not in text, (
        "分母为 0 的报告里出现了百分比")
    assert "全部通过" not in text, "分母为 0 时说了「全部通过」"
    assert "这套采集不测 14 槽" in text, "「全都不测」那份载荷没有报 14 槽"


def test_the_html_tolerates_an_old_payload_without_the_segments(tmp_path):
    """旧载荷(`l1-1`,前端 store 里可能还躺着)不许崩,也不许编分段数。"""
    old = {"n_slots": 14, "n_passed": 8, "n_dropped": 6, "confidence_cap": "高",
           "passed_slots": []}
    html = _render(tmp_path, old)
    text = _text_of(html)
    assert "没带分段覆盖数" in text, (
        "旧载荷下既没分段也不说「没带」 —— 读者会以为这一段本来就不该有")
    assert "整维没有对应指标" not in text, "旧载荷下编出了分段数"


# ---------------------------------------------------------------- ④ demo_run
def test_demo_run_prints_the_same_three_segments(capsys):
    """`demo_run.py` 投屏那一行同口径 —— 它原来印「过证据门 8/14 槽」。"""
    demo_run.stage_numbers(_payload(_full_coverage()))
    out = capsys.readouterr().out
    assert "过证据门 8/14" not in out, f"命令行还在印旧的聚合数:{out}"
    for word in (_W_MEASURABLE, _W_NOT_MEASURED, _W_NO_INDICATOR):
        assert word in out, f"命令行那一行没有「{word}」:{out}"
    assert "14/14" in out, f"命令行没有把可测量的 14/14 说出来:{out}"
