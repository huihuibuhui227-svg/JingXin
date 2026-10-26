"""报告里的「证据缺口」必须是人话 —— 不许把开发者的键名和缺口码印给读者。

## 为什么

2026-10-25 把 HTML 报告切到 L1 引擎后(`test_html_report_uses_the_l1_engine.py`),
真实产出的报告里出现了这样的行:

    话语结构特征
    证据不足
    —— 本次未采集到足以评估该行为线索的有效样本。
    connective_density → discourse_marker_rate:not_stored
    gaze_direction_x_std → gaze_stability:no_samples
    au4_frown → brow_furrow_ratio_vs_baseline:uncalibrated
    symmetry_score:删槽 —— L1 没有面部对称性指标

读者读到的是三个**开发者才认识的**东西:

* `connective_density` / `symmetry_score` —— `l1_mapper.DIMENSIONS` 里的**内部槽键**;
* `not_stored` / `no_samples` —— `l1_indicators.KINDS` 里的**缺口码常量**;
* `→` —— 旧槽到新指标的一次**迁移记号**,只在核对口径时才有意义。

而这三样,报告层早就各有一份人话对应:

* 槽的中文名在旧引擎 `research_mapper` 的槽表里就有(「连接词密度」「面部对称性」);
* `l1_quality.GAP_EXPLANATION` 把 9 个缺口码逐条翻成一句中文,模块注释原话是
  「**不许直接渲染 `gap` 的英文常量**」。

⟹ 这不是"信息不够",是**摆错了层**:内部对账用的记号漏进了交付物。

## 这个文件守什么

1. 缺口行里不出现任何缺口码常量(`KINDS` 里那 9 个)。
2. 缺口行里不出现任何内部槽键(`SLOT_MAP` 的键)。
3. 缺口行用 `l1_quality.GAP_EXPLANATION` 那张表 —— **改表就改报告**,不是另抄一份。
4. 同一个槽在两份报告(HTML / 旧引擎)里的中文名必须一致,免得读者以为是两个东西。
"""
from __future__ import annotations

import pytest
import pandas as pd

from report_frontend import l1_indicators, l1_mapper, l1_quality
from report_frontend.l1_mapper import DIMENSIONS, SLOT_MAP
from report_frontend.l1_report import build_l1_report

_SID = "20260101_000000_0000"


def _labels() -> dict[str, str]:
    """槽键 → 中文名。没有这张表本身就是红 —— 报告里就只能露出英文键。"""
    labels = getattr(l1_mapper, "SLOT_LABEL", None)
    assert labels is not None, (
        "`l1_mapper` 里没有 `SLOT_LABEL` —— 14 个槽没有中文名的登记处,"
        "报告只能把 `connective_density` 这样的内部键印给读者")
    return labels


def _gap_lines() -> list[str]:
    """空数据下每个维度的缺口行。

    ⚠️ 必须把 L1 会读的**三条旁路**都指到空目录(`session_timeline` 的帧账本、
    `voice_series` 的序列缓存、转写):否则本机盘上的真实日志会让某些槽出值,
    缺口行就不再全了 —— 那是**测试环境泄漏**,不是被测行为。
    见 `tests/test_structured_report_endpoint.py` 的 `isolated` fixture。
    """
    payload = build_l1_report(_SID, logs={})
    lines = [g for d in payload["dimensions"].values() for g in d["evidence_gaps"]]
    assert lines, "一个缺口行都没有 —— 本测试退化为空断言"
    return lines


@pytest.fixture
def gaps(monkeypatch, tmp_path):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path / "no_rec"))
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path / "no_series"))
    return _gap_lines()


def test_no_gap_code_reaches_the_reader(gaps):
    """★ 红法:把 `_gap_lines` 改回 `f"{s['slot']} → {s['l1']}:{s['gap']}"` ⟹ 这里红。

    `not_stored` 这类常量是给代码用的;读者要读的是
    「上游只存了聚合量,这一层需要的粒度没落盘」(见 `GAP_EXPLANATION`)。
    """
    leaked = [(line, code) for line in gaps
              for code in l1_indicators.KINDS if code in line]
    assert not leaked, "缺口码直接印给了读者:\n  " + "\n  ".join(
        f"「{code}」出现在「{line}」" for line, code in leaked)


def test_no_internal_slot_key_reaches_the_reader(gaps):
    """★ 红法:同上。`connective_density` 是 `DIMENSIONS` 的内部键,报告里该写「连接词密度」。"""
    leaked = [(line, slot) for line in gaps
              for slot in SLOT_MAP if slot in line]
    assert not leaked, "内部槽键直接印给了读者:\n  " + "\n  ".join(
        f"「{slot}」出现在「{line}」" for line, slot in leaked)


def test_gap_lines_use_the_one_shared_translation_table(monkeypatch, tmp_path):
    """★ 红法:在 `_gap_lines` 里另抄一张中文表 ⟹ 这里红。

    抄第二张表的后果是可预见的:新增一个缺口码时改了一处、漏了一处,
    界面上就冒出一个英文单词 —— `l1_quality` 与前端 `GAP_LABEL` 都吃过这个亏
    (`test_every_gap_kind_has_a_human_explanation` 记着)。

    判据:把 `GAP_EXPLANATION` 里的每一句都换成同一个可识别的标记,
    报告里必须跟着变 —— 变了就证明它读的是**这张表**,不是自己抄的一份。
    """
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path / "no_rec"))
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path / "no_series"))

    marker = "只可能来自GAP_EXPLANATION"
    patched = {code: marker for code in l1_indicators.KINDS}
    # 报告层若 `from ... import GAP_EXPLANATION`,它有自己的一份名字;
    # 若 `from ... import l1_quality` 再点出来,改 `l1_quality` 那份就够。两处都改。
    monkeypatch.setattr(l1_quality, "GAP_EXPLANATION", patched)
    import report_frontend.l1_report as l1_report
    if hasattr(l1_report, "GAP_EXPLANATION"):
        monkeypatch.setattr(l1_report, "GAP_EXPLANATION", patched)

    lines = _gap_lines()
    assert any(marker in g for g in lines), (
        "缺口行没读 `GAP_EXPLANATION` —— 它自己抄了一张表。"
        f"实际内容:{lines}")


def test_gap_lines_carry_no_markup(gaps):
    """★ 红法:让理由里的 `**字/秒**` 原样进缺口行 ⟹ 这里红。

    `SLOT_MAP` 的删槽理由是**给开发者读的注释体**,里面写着 `**字/秒**`
    (markdown 粗体)。它被原样拼进缺口行后:

    * HTML 报告把它印成字面的 `**字/秒**`;
    * 前端 `ReportViewer` 用 `{g}` 渲染,React 转义后同样是字面的星号。

    ⟹ 缺口行要么是纯文本,要么两边都得先过一遍渲染器 —— 后者不可能
    (前端没有 markdown 依赖),所以只能在产出处去掉记号。
    """
    import re

    markup = [(g, m.group(0)) for g in gaps
              for m in re.finditer(r"\*\*|`|__|~~|\[[^\]]*\]\([^)]*\)", g)]
    assert not markup, "缺口行里带着标记语法,会原样印给读者:\n  " + "\n  ".join(
        f"「{tok}」出现在「{g}」" for g, tok in markup)


def test_dropped_slot_lines_say_the_layer_does_not_measure_it(gaps):
    """★ 红法:把删槽行改回 `f"{label}:{reason}"` ⟹ 这里红。

    读者必须能一眼分出**两件完全不同的事**:

    * 「本层不测这个构念」—— 再采一百次也不会有,别等;
    * 「测了,这次没数据」—— 下次采集/补缓存就有。

    混成一句,读者会去追一个不存在的数据。所以删槽行必须**显式**带上前者那句话。

    ⚠️ 2026-10-26 换措辞:原来是「本层不测这一条」——「本层」是分层架构的内部叫法
    (L1 层 / L0 层),读者不知道有几层。现在说「这一条当前这套采集不测」,
    判据本身没变:仍然是**采集方式决定的不可能有**,不是"这次没采到"。
    """
    dropped = [g for g in gaps if "这套采集不测" in g]
    # 地板**从槽表推导**,不写死数字:写死会过期,而且这里有个易漏点 ——
    # `confidence_level` 三个槽在 L1 全删 ⟹ 整个维度消失(`vanished_dimensions`),
    # 它的删槽行**不在** `dimensions` 里。所以"删槽总数"≠"报告里出现的删槽行数"。
    expected = sum(1 for spec in DIMENSIONS.values()
                   if any(SLOT_MAP.get(s, (None,))[0] is not None for s in spec["slots"])
                   for s in spec["slots"] if SLOT_MAP.get(s, (None,))[0] is None)
    assert expected, "一条删槽都没有 —— 本测试退化为空断言"
    assert len(dropped) == expected, (
        f"删槽行数({len(dropped)})与幸存的维度里的删槽数({expected})不符:"
        f"{dropped}")
    for g in dropped:
        assert g.split(":")[0].strip(), f"删槽行没有槽名:{g!r}"


@pytest.fixture
def one_passed(monkeypatch, tmp_path):
    """一份**真有一条过门指标**的载荷 —— 空数据下 `evidence_chain` 是空的,
    拿它做断言就是空转(本仓 `test_no_assert_is_dead` 守这条)。

    注入的 `gaze_stability` 是 `gaze_direction_x_std` 槽的 L1 对应(同构念),
    构造方式与 `tests/test_structured_report_endpoint.py` 里那条一致。
    """
    from report_frontend import l1_report
    from report_frontend.event_windows import WindowSet
    from report_frontend.l1_indicators import L1Indicator
    from report_frontend.l1_session import L1Result
    from report_frontend.session_timeline import Timebase, TimebaseStatus

    face = Timebase(modality="face", status=TimebaseStatus.OK,
                    t_sec=pd.Series([0.0, 1.0]), n_frames=2,
                    wall_back_steps=0, drift_sec=0.0)
    result = L1Result(
        session_id=_SID,
        timebase=type("T", (), {"modalities": {"face": face}})(),
        windows=WindowSet(session_id=_SID, granularity="session",
                          windows=[], n_questions=0),
        indicators=[L1Indicator(
            name="gaze_stability", family="F2_gaze",
            formula="1 − std(gaze_direction_x) / 视频内极差（⚠️ 注释体记号）",
            required_l0=("gaze_direction_x",), window="session", unit="",
            value=0.83, n_valid=628, n_total=700, confidence="高", gap=None)],
    )
    monkeypatch.setattr(l1_report, "compute_session_l1", lambda *a, **k: result)
    return l1_report.build_l1_report(_SID)


def test_passed_slots_name_the_indicator_in_chinese(one_passed):
    """★ 红法:把 `indicator` 填回内部槽键(`s["slot"]`)⟹ 这里红。

    这是同一个病的**另一处**:缺口行说人话了,而「通过证据门的指标」那张表
    还印着 `connective_density`。报告里同一个指标在一处叫「连接词密度」、
    另一处叫 `connective_density`,读者会当成两个东西。
    """
    passed = one_passed["coverage"]["passed_slots"]
    assert passed, "一个过门槽都没有 —— 本测试退化为空断言"
    leaked = [(p["indicator"], s) for p in passed for s in SLOT_MAP
              if s in p["indicator"]]
    assert not leaked, "过门指标表里露出了内部槽键:\n  " + "\n  ".join(
        f"「{s}」出现在「{name}」" for name, s in leaked)


def test_evidence_chain_names_are_not_raw_formulas(one_passed):
    """★ 红法:把 `human_name` 填回 `l1_indicators` 的 `formula` ⟹ 这里红。

    2026-10-25 实测产出过这样一行(报告「指标」列):

        话语标记**出现次数** ÷ 汉字数 × 100(与 connective_density 的
        「不同标记出现过几次」不同口径;分母只数汉字 —— 标记表是中文的,
        英文词不构成可命中的材料);⚠️ 全场混音,含面试官

    `formula` 是**给开发者核对算法**用的,它有两百多字、带着 `⚠️` 和 markdown
    粗体、还点了另一个内部键名。它该待在 `l1.indicators`(审计链)里,
    不该占着「指标」这一列 —— 那一列读者只想知道"这一行量的是什么"。
    """
    names = [e["human_name"] for d in one_passed["dimensions"].values()
             for e in d["evidence_chain"]]
    assert names, "证据链一条都没有 —— 本测试退化为空断言"
    for n in names:
        assert len(n) <= 20, (
            f"「指标」列里是一整条公式(长度 {len(n)}):{n[:60]}…")
        assert "**" not in n and "⚠️" not in n, f"指标名里带着注释体记号:{n!r}"
        assert not any(s in n for s in SLOT_MAP), f"指标名里露着内部槽键:{n!r}"


def test_every_slot_has_a_chinese_label():
    """★ 红法:从 `SLOT_LABEL` 里删掉一条 ⟹ 这里红。

    地板:`DIMENSIONS` 里**全部** 14 个槽都要有名字。没有这句,槽表被清空时
    下面的循环一次都不进,本测试静默全绿。
    """
    declared = {s for spec in DIMENSIONS.values() for s in spec["slots"]}
    assert len(declared) == 14, f"声明的槽数变了:{sorted(declared)}"
    missing = sorted(declared - set(_labels()))
    assert not missing, f"这些槽没有中文名,报告里会露出英文键:{missing}"


def test_slot_labels_agree_with_the_legacy_report():
    """★ 红法:把 `SLOT_LABEL["symmetry_score"]` 改成别的 ⟹ 这里红。

    同一个槽在「历史报告」(旧引擎)与「本次报告」(L1)里必须同名 ——
    否则读者会以为 11/14 与 7/14 那两份报告量的是两批不同的指标。
    (两边的**数**不可比,理由登记在 `l1_mapper.NOT_COMPARABLE`;但**名字**必须一致。)
    """
    from report_frontend.research_mapper import ResearchCapabilityMapper

    legacy = {}
    for rule in ResearchCapabilityMapper().mapping_rules.values():
        for key, _w, _pos, label, _role in rule["indicators"]:
            # 旧表的标签常带括注(「皱眉 (双眉内侧点距 ÷ 眼距)」),只比前导名。
            legacy[key] = label.split(" (")[0].strip()

    assert set(legacy) == set(SLOT_MAP), (
        f"旧引擎与 L1 的槽表已经不是同一批:{set(legacy) ^ set(SLOT_MAP)}")
    labels = _labels()
    diff = {k: (labels.get(k), v) for k, v in legacy.items() if labels.get(k) != v}
    assert not diff, "同一个槽在两份报告里叫不同的名字:\n  " + "\n  ".join(
        f"{k}: L1={a!r} 旧引擎={b!r}" for k, (a, b) in diff.items())
