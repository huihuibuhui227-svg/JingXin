# tests/test_m4_quality.py
"""**M4.4 对齐质量报告**的守卫。

## 钉的是什么

1. **模态缺失 ⟹ 依赖它的指标必须无证据。** 这是 spec 里写死的一条硬不变量,
   本文件用变异钉住(删掉检查 ⟹ 红)。
2. **`gap` 必须分类统计**,不能糊成一句"证据不足" —— 四种原因对应四种补救动作。
3. **"未提供" ≠ "缺失"。** 没把日志传进来,和这一场根本没录,是两件事;
   混在一起会让人去补一份其实已经在盘上的日志。
4. **报告里的中文解释不许直接印英文常量。**
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from report_frontend import l1_quality as lq
from report_frontend.event_windows import AnswerWindow, WindowSet
from report_frontend.l1_indicators import (
    GAP_MODALITY_MISSING, GAP_UNCALIBRATED, L1Indicator,
)
from report_frontend.session_timeline import Timebase, TimebaseStatus


def _ind(name="blink_rate_per_min", family="F3_blink", window="answer:0",
         value=math.nan, n_valid=0, n_total=10, confidence="无", gap=GAP_MODALITY_MISSING):
    return L1Indicator(name=name, family=family, formula="f", required_l0=("a",),
                       window=window, unit="", value=value, n_valid=n_valid,
                       n_total=n_total, confidence=confidence, gap=gap)


class _FakeResult:
    """最小替身 —— 只补 `L1Result` 的四个属性,其余用**真的** dataclass。

    刻意不用手搓的假对象:假对象一旦与真 dataclass 漂开,测试就变成在测假对象。
    """

    def __init__(self, modalities, indicators, windows=None, gaps=None):
        self.session_id = "s"
        self.timebase = type("T", (), {"modalities": modalities})()
        self.windows = windows or WindowSet(
            session_id="s", granularity="answer",
            windows=[AnswerWindow(index=0, t_start=0.0, t_end=10.0)],
            n_questions=1, n_answer_segments=1, wall_reliable=True)
        self.indicators = indicators
        self.gaps = gaps or []


def _tb(status, usable, back=0, drift=0.0, n=10):
    return Timebase(modality="face", status=TimebaseStatus(status),
                    t_sec=pd.Series([0.0, 1.0]), n_frames=n,
                    wall_back_steps=back, drift_sec=drift)


# ── 1. 硬不变量:模态缺失 ⟹ 不许有值 ────────────────────────────────────

def test_indicator_with_value_but_missing_modality_is_rejected():
    """★ 核心不变量。红法:把 `validate()` 里那段 missing 检查删掉 ⟹ 本条红。"""
    res = _FakeResult(
        modalities={"face": _tb("unreconstructible", usable=False)},
        indicators=[_ind(value=14.0, n_valid=100, n_total=100,
                         confidence="中", gap=None)],
    )
    with pytest.raises(ValueError, match="依赖的模态"):
        lq.build_quality_report(res)


def test_indicator_with_neither_value_nor_gap_is_rejected():
    """两者皆无 ⟹ 读者会以为它是 0。"""
    res = _FakeResult(
        modalities={"face": _tb("ok", usable=True)},
        indicators=[_ind(value=math.nan, gap=None, confidence="无")],
    )
    with pytest.raises(ValueError, match="既没有值也没有 gap"):
        lq.build_quality_report(res)


# ── 2. gap 分类统计 ──────────────────────────────────────────────────────

def test_gap_kinds_are_counted_separately_and_explained_in_chinese():
    """四种原因必须分开计数,而且解释是中文,不是英文常量。"""
    res = _FakeResult(
        modalities={"face": _tb("ok", usable=True)},
        indicators=[
            _ind(name="a", gap=GAP_UNCALIBRATED),
            _ind(name="b", gap=GAP_UNCALIBRATED),
            _ind(name="c", gap=GAP_MODALITY_MISSING),
        ],
    )
    q = lq.build_quality_report(res)
    assert q.by_gap[GAP_UNCALIBRATED] == 2
    assert q.by_gap[GAP_MODALITY_MISSING] == 1

    lines = lq.explain_gaps(q)
    joined = " ".join(lines)
    assert "标定集" in joined and "模态" in joined
    for english in (GAP_UNCALIBRATED, GAP_MODALITY_MISSING):
        assert english not in joined, f"英文常量 {english} 漏进了给读者看的句子"


def test_explain_gaps_omits_zero_counts():
    res = _FakeResult(modalities={"face": _tb("ok", usable=True)},
                      indicators=[_ind(gap=GAP_UNCALIBRATED)])
    q = lq.build_quality_report(res)
    assert len(lq.explain_gaps(q)) == 1


# ── 3. "未提供" ≠ "缺失" ─────────────────────────────────────────────────

def test_not_provided_is_distinct_from_absent():
    """★ 只传了 face ⟹ gesture/voice 是 `not_provided`,不是"缺失"。

    红法:把 `build_quality_report` 里那段 `setdefault` 改回 `status="absent"` ⟹ 本条红。
    """
    res = _FakeResult(modalities={"face": _tb("ok", usable=True)},
                      indicators=[_ind(gap=GAP_UNCALIBRATED)])
    q = lq.build_quality_report(res)

    assert q.timebase["face"]["provided"] is True
    assert q.timebase["gesture"]["provided"] is False
    assert q.timebase["gesture"]["status"] == "not_provided"

    line = lq.modality_line(q)
    assert "未提供" in line and "gesture" in line
    assert "缺失" not in line, "不能把'没传进来'说成'缺失'"


# ── 4. 计数自洽 ──────────────────────────────────────────────────────────

def test_confidence_and_gap_counts_are_consistent():
    res = _FakeResult(
        modalities={"face": _tb("ok", usable=True)},
        indicators=[
            _ind(name="a", value=1.0, n_valid=100, n_total=100, confidence="高", gap=None),
            _ind(name="b", value=1.0, n_valid=30, n_total=100, confidence="中", gap=None),
            _ind(name="c", gap=GAP_UNCALIBRATED),
        ],
    )
    q = lq.build_quality_report(res)

    assert sum(q.by_confidence.values()) == q.n_total == 3
    assert q.n_measured == 2
    assert sum(q.by_gap.values()) == 1
    # 有值的与有 gap 的互补
    assert q.n_measured + sum(q.by_gap.values()) == q.n_total


def test_family_summary_attributes_each_family_to_its_modality():
    res = _FakeResult(
        modalities={"face": _tb("ok", usable=True)},
        indicators=[_ind(name="a", family="F3_blink", gap=GAP_UNCALIBRATED),
                    _ind(name="b", family="F7_micro", gap=GAP_UNCALIBRATED)],
    )
    q = lq.build_quality_report(res)
    assert q.families["F3_blink"]["modality"] == "face"
    assert q.families["F7_micro"]["modality"] == "face"
    assert q.families["F3_blink"]["n"] == 1


# ── 5. 端到端:真链路走一遍(不读 data/) ────────────────────────────────

def test_end_to_end_with_no_face_log_reports_modality_missing(tmp_path, monkeypatch):
    """没有 face 日志 ⟹ 16 条全部 `modality_missing`,且报告能生成、能自检通过。"""
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    from report_frontend import l1_face, l1_quality, l1_session
    from report_frontend.event_windows import AnswerWindow, WindowSet

    ws = WindowSet("s", "answer", [AnswerWindow(index=0, t_start=0.0, t_end=10.0)],
                   n_questions=1)
    inds = l1_face.compute_face_indicators(None, ws, None)
    assert len(inds) == 16
    assert all(i.gap == GAP_MODALITY_MISSING for i in inds)

    res = _FakeResult(modalities={}, indicators=inds, windows=ws)
    q = l1_quality.build_quality_report(res)
    q.validate()                       # 不抛 ⟹ 自洽
    assert q.by_gap[GAP_MODALITY_MISSING] == 16
    assert q.n_measured == 0


# ── 5. 「日志存在」≠「数据存在」──────────────────────────────────────────
#
# 实测背景(2026-10-18,49 场):面部日志 21861 行里 **25.5% 整行为空**,
# **13 场会话的面部日志每一行都是空的**;手势侧 12 场肘角 0%(臂不可见);
# 研究模式 11 个日志文件全部 0 行。
# ⟹ 一场会话可以「有 face 日志 + 时间轴成立 + 16 条 F 指标全部 n_valid=0」,
#    而报告只说「face 已提供」。读者会把「提供」读成「有数据」。

def test_modality_coverage_counts_only_rows_that_carry_data():
    """4 行里只有 2 行有 AU 值 ⟹ `usable_ratio == 0.5`。

    红法:把 `notna().any(axis=1)` 改成 `len(df)` 之类"按行数算" ⟹ 得 1.0,本条红。
    """
    logs = {"face": pd.DataFrame({
        "au12_smile": [0.1, np.nan, 0.3, np.nan],
        "avg_ear": [0.3, np.nan, 0.3, np.nan],
    })}
    cov = lq.modality_coverage(logs)
    assert cov["face"]["rows"] == 4
    assert cov["face"]["usable_rows"] == 2
    assert cov["face"]["usable_ratio"] == pytest.approx(0.5)
    assert cov["face"]["provided"] is True


def test_fully_empty_modality_reports_zero_coverage():
    """整行为空的日志 ⟹ `provided=True` 但 `usable_ratio == 0`。

    ★ 两个字段必须**分开**:`provided` 说的是"盘上有没有这份日志",
    `usable_ratio` 说的是"里面有没有数据"。合成一个布尔就再也分不开了。
    """
    logs = {"face": pd.DataFrame({"au12_smile": [np.nan] * 5,
                                  "avg_ear": [np.nan] * 5})}
    cov = lq.modality_coverage(logs)
    assert cov["face"]["provided"] is True
    assert cov["face"]["usable_ratio"] == 0.0


def test_missing_log_is_not_provided():
    cov = lq.modality_coverage({})
    assert cov["face"]["provided"] is False
    assert cov["face"]["usable_ratio"] == 0.0


def test_present_but_empty_modality_cannot_carry_a_value():
    """★★ 新硬不变量:**名义存在、实际为零**的模态,也不许有值。

    它比「模态缺失」更危险:时间轴成立、`timebase.usable` 为真,
    所以原来那条 `missing` 检查**抓不到它**。

    红法:把 `validate()` 里 `empty` 那段删掉 ⟹ 本条红(会静默通过)。
    """
    res = _FakeResult(
        modalities={"face": _tb("ok", usable=True)},
        indicators=[_ind(value=14.0, n_valid=100, n_total=100,
                         confidence="中", gap=None)],
    )
    res.coverage = {"face": {"rows": 100, "usable_rows": 0,
                             "usable_ratio": 0.0, "provided": True}}
    with pytest.raises(ValueError, match="一行数据都没有"):
        lq.build_quality_report(res)


def test_coverage_flows_into_the_report():
    res = _FakeResult(
        modalities={"face": _tb("ok", usable=True)},
        indicators=[_ind(value=math.nan, gap=GAP_MODALITY_MISSING)],
    )
    res.coverage = {"face": {"rows": 100, "usable_rows": 25,
                             "usable_ratio": 0.25, "provided": True}}
    rep = lq.build_quality_report(res)
    assert rep.coverage["face"]["usable_ratio"] == pytest.approx(0.25)


# ── 「为什么只有这么信」必须真的进报告 ───────────────────────────────────

def test_low_confidence_notes_reach_the_report_payload():
    """★ `explain_confidence()` 必须**被调用**,且理由进载荷。

    ## 这条测试的由来

    `LOW_CONFIDENCE_REASON`(一张"为什么置信低"的表)与 `explain_confidence()`
    写好了,却**没有任何调用点** —— 生产和测试都没用过。后果是报告上只写
    「置信 低」,不写"为什么只有这么信"。置信度本身不回答这个问题,
    而后者才是读者要的那句话。这属于"写好了没接线"的死代码。

    红法两条(都已实测):
      ① 删掉 `l1_report.build_l1_report` 里 `"low_confidence_notes"` 那一行 ⟹ ③ 红;
      ② 把 `LOW_CONFIDENCE_REASON` 里 `speech_rate` 那条删掉 ⟹ ① 红。
    """
    import inspect

    from report_frontend import l1_report

    low = _ind(name="speech_rate", value=7.1, n_valid=1, n_total=1,
               confidence="低", gap=None)
    high = _ind(name="blink_rate_per_min", value=14.0, n_valid=100, n_total=100,
                confidence="高", gap=None)
    q = lq.build_quality_report(
        _FakeResult(modalities={"face": _tb("ok", usable=True)},
                    indicators=[low, high]))

    # ① 「低 + 有值」的那条被翻译出来;高置信的不进说明。
    notes = lq.explain_confidence(q)
    assert any(n.startswith("speech_rate:") for n in notes), notes
    assert not any(n.startswith("blink_rate_per_min:") for n in notes), notes
    # 说明里必须讲清"为什么"是整场只测了一次,而不是只说"低"。
    assert "一次" in notes[0], notes[0]

    # ② 没有值的指标不进说明(它该去 gap 那边解释,不是置信度)。
    assert lq.explain_confidence(
        lq.build_quality_report(_FakeResult(
            modalities={"face": _tb("ok", usable=True)},
            indicators=[_ind(name="speech_rate", value=math.nan, n_valid=0,
                             n_total=1, confidence="无",
                             gap=GAP_MODALITY_MISSING)]))) == []

    # ③ 载荷里真的有这个字段(否则函数再好也到不了报告)。
    src = inspect.getsource(l1_report.build_l1_report)
    assert '"low_confidence_notes"' in src, "载荷里没有 low_confidence_notes —— 又变成死代码了"
