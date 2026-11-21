# tests/test_response_latency_gap_text.py
"""反应延迟报缺口时,句子必须**按实测证据说**,不许再"多数场次"一刀切。

## 为什么单开一个文件

`tests/test_gap_explanations.py` 守的是"同一码同一句"的**通用**规矩。这一条不一样:
它是全仓**唯一一个"同一个缺口码有两种病因"**的指标 —— 规矩也单独成立。

2026-11-21 实测(45 场真实录制,逐场核过):

| 情形 | 场次 | 台账 `questions.jsonl` | 语音日志 | 旧文案说什么 | 真实病因 |
|---|---|---|---|---|---|
| 一 | 25 | 无 | 0 行 | 「没存题目时刻表」 | 没走题库作答流程(只当录像机用) |
| 二 | 6 | **有** | **0 行** | 「没存题目时刻表」 | 采集端没把应答时刻落盘 |
| 三 | 0 | 有 | 有行、无那一列 | 「粒度太粗」 | 应答时刻那一列没落盘 |
| 四 | 10 | 有 | 有行、有列 | (不该出现缺口) | —— |

旧文案对情形二说「补上题目时刻就能算」是**当众说错话**:台账就在盘上。
评审只要 `ls questions.jsonl` 一下就会问"那你说没存是什么意思"。

## 这个文件守什么

1. 三种病因各有一句**互不相同**的话,且每句都以实测事实为准;
2. 拿不到事实(`has_ledger` 是 `None`、没有 `gap_detail`、传了非法值)⟹ 退回
   **在任何一场都为真**的兜底句,**不许**把"没证据"读成"实测没有";
3. 有值的那 10 场**不出现**这个缺口(防御性分支:不写死句子);
4. 缺口码一个都没变(`window_none` / `not_stored`,都在既有 9 类里);
5. 诊断事实**不进载荷**:明细行键集不长新键,只用来选句;
6. 三处出口(产出方 → 明细行 → 报告正文)拿到的是**同一句**。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from report_frontend import gap_explanations as gapx
from report_frontend import l1_indicators, l1_mapper, l1_quality, l1_report, l1_voice
from report_frontend import voice_series

#: 交付物正文的行话/符号规矩 —— 与 `tests/test_gap_explanations.py` 同一套。
_JARGON = ("指标槽", "证据门", "过门", "同构念", "效标", "本层", "当前管线")
_SYMBOLS = ("⟹", "→", "⇒")

#: 旧文案里那句"对 31 场一律印"的判词。新句子**一个都不许**带它。
_OLD_CLAIM = "多数场次没存题目时刻表"


# ---------------------------------------------------------------------------
# 造三种情形的输入(事实由产出方给,不在这里猜)
# ---------------------------------------------------------------------------

def _detail(has_ledger, voice_rows, rt_col_present=False) -> dict:
    return {"has_ledger": has_ledger, "voice_rows": voice_rows,
            "rt_col_present": rt_col_present}


def _text(detail) -> str | None:
    """与 `l1_report._gap_lines` 同一条取值路径(`explain_specific` 优先)。"""
    return gapx.explain_specific({"name": "response_latency_sec",
                                  "gap": "window_none", "gap_detail": detail})


def _voice_df(n_rows: int, *, with_rt: bool) -> pd.DataFrame:
    """一段回答一行的语音日志。`with_rt=False` 时**没有**应答时刻那一列。"""
    data = {
        "duration_sec": [30.0] * n_rows,
        "speech_duration_sec": [20.0] * n_rows,
        "n_chars": [50] * n_rows,
        "pause_frequency": [3.0] * n_rows,
    }
    if with_rt:
        data["reaction_time"] = [8.0] * n_rows
    return pd.DataFrame(data)


def _series(sid: str) -> list:
    """一条最小的帧序列 —— 有了它,`response_latency_sec` 才走 `window_none` 那条路。"""
    n = 300
    return [voice_series.VoiceSeries(
        session_id=sid, index=None, frame_sec=0.1,
        f0=np.where(np.arange(n) % 9 == 0, np.nan,
                    120.0 + 10.0 * np.sin(np.arange(n) / 5.0)),
        voiced_prob=np.ones(n), rms=np.full(n, 0.05), source="camera.webm")]


def _rt_indicator(voice, sid: str, *, has_ledger: bool, monkeypatch):
    """跑**产出方**拿到那条指标(`_ledger_present` 换成给定事实,不依赖盘上语料)。"""
    monkeypatch.setattr(l1_voice, "_ledger_present", lambda _sid: has_ledger)
    monkeypatch.setattr(l1_voice, "_source_on_disk", lambda _sid: True)
    inds = {i.name: i for i in l1_voice.compute_voice_indicators(
        voice, series=_series(sid), session_id=sid)}
    return inds["response_latency_sec"]


# ---------------------------------------------------------------------------
# ① 三种病因各一句,且必须互不相同
# ---------------------------------------------------------------------------

def test_a_session_that_never_used_the_question_bank_says_so():
    """★ 红法:让第一类也印"没存题目时刻表" ⟹ 红。

    这类场次(25 场)既没有题目台账、也没有逐题语音日志 —— 它没走题库作答流程,
    只当录像机用。对它说"补上题目时刻就能算"同样是错的:没有作答流程,补出来的
    是一张空台账。
    """
    text = _text(_detail(has_ledger=False, voice_rows=0))
    assert text, "第一类没有给出任何说法"
    assert "没走题库作答流程" in text, f"没点明这一场没走作答流程:{text}"
    assert _OLD_CLAIM not in text, f"还在印那句一刀切的判词:{text}"


def test_a_ledger_that_exists_is_never_called_missing():
    """★ 红法:把第二类并回第一类(都按"没台账"说)⟹ 红。**这是本次要修的那个缺口**。

    6 场(`a971`/`0dc7`/`253a`/`4579`/`90e7`/`87c9`)的 `questions.jsonl` 就在盘上,
    真实病因是"语音作答日志 0 行"。判据只认实测:台账在 ⟹ 不许说没台账,
    也不许说"没走题库作答流程"(它走了,只是采集端没落盘)。
    """
    text = _text(_detail(has_ledger=True, voice_rows=0))
    assert text, "第二类没有给出任何说法"
    assert "台账在" in text, f"没有承认台账在盘上:{text}"
    assert "日志是空的" in text, f"没点明真实病因是语音日志空:{text}"
    assert "没走题库作答流程" not in text, f"把有台账的一场说成没走流程:{text}"
    assert _OLD_CLAIM not in text, f"还在印那句一刀切的判词:{text}"


def test_voice_rows_without_the_timestamp_column_blame_the_column():
    """第三类:日志**有行**,缺的是应答时刻那一列(缺失或整列空着)。

    这一类的码本来就不是 `window_none`(产出方记 `not_stored`),但病因与"没台账"
    毫无关系 —— 拿台账说事同样会误导。钉住它说的是"那一列没落盘"。
    """
    text = _text(_detail(has_ledger=True, voice_rows=8))
    assert text, "第三类没有给出任何说法"
    assert "应答时刻" in text and "那一列" in text, (
        f"没点明缺的是应答时刻那一列:{text}")
    assert "台账" not in text, f"有语音行的时候还拿台账说事:{text}"


def test_the_three_causes_get_three_different_sentences():
    """★ 红法:两类的句子写成一样 ⟹ 红。三句相同就等于没分情形。

    这是这个补丁的全部价值:同一个码,三种病因,三种补救动作(重录 / 查采集端 /
    改落盘),句子必须能把读者引到对的那一条上。
    """
    texts = {
        "没有台账也没有语音行": _text(_detail(False, 0)),
        "有台账但语音日志空": _text(_detail(True, 0)),
        "有语音行但没有那一列": _text(_detail(True, 8)),
    }
    assert len(set(texts.values())) == 3, f"三种病因没有三句不同的话:{texts}"


def test_the_sentences_obey_the_house_rules():
    """三句 + 兜底句都要进交付物正文 ⟹ 逐句过措辞规矩(`test_gap_explanations` 同一套)。

    行话/符号/英文键名一律不许:这些句子不经过任何渲染层,产出处就是终稿。
    """
    sentences = {
        "第一类": _text(_detail(False, 0)),
        "第二类": _text(_detail(True, 0)),
        "第三类": _text(_detail(True, 8)),
        "兜底": _text(None),
    }
    assert all(sentences.values()), f"有情形没有句子:{sentences}"
    for name, text in sentences.items():
        assert len(text) <= gapx.LIMIT, f"{name} 超过一句话上限({len(text)} 字):{text}"
        assert not any(ch.isascii() and ch.isalpha() for ch in text), (
            f"{name} 里带着英文/指标键:{text}")
        assert not any(s in text for s in _SYMBOLS), f"{name} 里带着推理符号:{text}"
        assert not any(w in text for w in _JARGON), f"{name} 里带着行话:{text}"


# ---------------------------------------------------------------------------
# ② 拿不到事实 ⟹ 退回"在任何一场都为真"的兜底句
# ---------------------------------------------------------------------------

def test_an_unknown_ledger_is_not_read_as_no_ledger():
    """★ 红法:把 `has_ledger is None` 当成 `False` ⟹ 红。

    `None` 与 `False` 是两件事:前者是"这一层问不到"(没有 session_id、id 形状
    不合法、读盘出错),后者是"盘上确实没有"。把问不到说成没有,正是本仓一直在
    消灭的"判词比事实更狠"。
    """
    text = _text(_detail(has_ledger=None, voice_rows=0))
    assert text == gapx.SPECIAL["response_latency_sec"]["window_none"], (
        f"事实不明时没有退回兜底句:{text}")
    assert "没走题库作答流程" not in text, f"事实不明却下了结论:{text}"
    assert _OLD_CLAIM not in text, f"兜底句还在印那句一刀切的判词:{text}"


def test_no_evidence_at_all_still_says_something_true():
    """没有 `gap_detail`(老调用方、单测直调)也必须有话说 —— 且每场都为真。

    兜底句只说两件全语料都成立的事:①这条指标只读语音日志里的应答时刻,
    不按题目时刻重算;②没落盘就补不回来。它**不猜**是哪一种没落盘。
    """
    for payload in ({"name": "response_latency_sec", "gap": "window_none"},
                    {"name": "response_latency_sec", "gap": "window_none",
                     "gap_detail": "has_ledger=true"},
                    {"name": "response_latency_sec", "gap": "window_none",
                     "gap_detail": {"has_ledger": True, "voice_rows": None}}):
        text = gapx.explain_specific(payload)
        assert text == gapx.SPECIAL["response_latency_sec"]["window_none"], (
            f"坏证据没有退回兜底句:{payload} → {text}")
        assert "语音日志" in text, f"兜底句没说明这个数只从语音日志来:{text}"
        assert _OLD_CLAIM not in text, f"兜底句还在印那句一刀切的判词:{text}"


def test_a_computed_value_gets_no_gap_sentence():
    """第四类:有值(10 场,如 `20260927_212143_95a8` 8.155 秒)⟹ **不出现**缺口句子。

    防御性分支:这一层不许为"没有缺口"的指标编一句话。两个入口都要返回 `None`。
    """
    row = {"name": "response_latency_sec", "gap": None, "value": 8.155,
           "n_valid": 8, "confidence": "中",
           "gap_detail": _detail(True, 8, True)}
    assert gapx.explain(row) is None, "有值却给出了缺口解释"
    assert gapx.explain_specific(row) is None, "有值却给出了特例句"


def test_the_evidence_only_ever_overrides_the_latency_indicator():
    """★ 红法:让证据句按名字之外的东西命中 ⟹ 红。

    证据只对 `response_latency_sec` 有意义。别的指标带着同样的 `gap_detail`,
    必须原样拿通用句/别家特例句 —— 否则"一个字段能吃遍全表"。
    """
    detail = _detail(True, 0)
    row = {"name": "pause_ratio", "gap": "not_produced", "gap_detail": detail}
    assert gapx.explain(row) == gapx.SPECIAL["pause_ratio"]["not_produced"]
    row = {"name": "blink_rate_per_min", "gap": "modality_missing", "gap_detail": detail}
    assert gapx.explain(row) == gapx.GENERIC["modality_missing"]
    # 反应延迟换一个码也不许套这组句子(它只有这两个码会产生"没有应答时刻")
    row = {"name": "response_latency_sec", "gap": "no_samples", "gap_detail": detail}
    assert gapx.explain(row) == gapx.GENERIC["no_samples"]


# ---------------------------------------------------------------------------
# ③ 判据来自实测:台账在不在,读的是盘上那个文件
# ---------------------------------------------------------------------------

def test_has_ledger_reads_the_questions_file_under_the_resolved_dir(
        tmp_path, monkeypatch):
    """★ 红法:把 `has_ledger` 换成"目录名里有没有问号"之类的间接判断 ⟹ 红。

    判据只有一条:`<这一场的录像目录>/questions.jsonl` 在不在。而录像目录名有
    **两种形态** —— 纯 id 与 `<标签>__<短id>`(`87c9` 就是后者),短 id 才是
    `session_id`。自己拼 `root()/sid` 会在存过标注的场次上判错(实测那会让
    29/45 场取不到日志),所以解析只走 `media_retention.resolve_recording_dir`。
    """
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    plain, labelled, empty = "20260101_000000_aaaa", "20260101_000000_bbbb", "20260101_000000_cccc"
    for name in (plain, f"标签-1-某人__{labelled}", empty):
        (tmp_path / name).mkdir()
    for name in (plain, f"标签-1-某人__{labelled}"):
        (tmp_path / name / "questions.jsonl").write_text("{}\n", encoding="utf-8")

    assert l1_voice._ledger_present(plain) is True, "纯 id 目录里的台账没认出来"
    assert l1_voice._ledger_present(labelled) is True, (
        "`<标签>__<短id>` 形态的目录里的台账没认出来 —— 6 场踩雷里就有一场是这种命名")
    assert l1_voice._ledger_present(empty) is False, "没有台账的目录被判成了有"
    assert l1_voice._ledger_present("20260101_000000_dddd") is False, (
        "盘上根本没有这一场的目录时,该说'没有台账'(这是实测结论,不是'问不到')")


def test_an_unusable_session_id_yields_unknown_not_false():
    """★ 红法:让非法 id 抛错(报告炸掉)或返回 `False`(下结论)⟹ 红。

    `session_id` 同时是目录名,库里有白名单正则;`../evil` 这类值必须被挡在门外。
    挡下来的结果是"这一层问不到"(`None`),不是"这一场没有台账"。
    """
    assert l1_voice._ledger_present(None) is None, "没有 session_id 却说'没有台账'"
    assert l1_voice._ledger_present("") is None
    assert l1_voice._ledger_present("../evil") is None, "非法 id 被当成了一场会话"


# ---------------------------------------------------------------------------
# ④ 产出方:码不变、值不变,只多带一份事实
# ---------------------------------------------------------------------------

def test_the_gap_code_is_unchanged_and_the_facts_ride_along(monkeypatch):
    """★ 红法:为了带事实而把码改成第 10 类、或顺手改个值 ⟹ 红。

    「只增不改」的落地处:三条码仍然只能是既有 9 类里的那两个(`window_none` /
    `not_stored`),`value` 仍然是 NaN、`n_valid` 仍然是 0;多出来的只有事实。
    """
    # 有台账 + 语音日志 0 行 ⟹ 仍是 window_none
    ind = _rt_indicator(None, "20260101_000000_eeee", has_ledger=True, monkeypatch=monkeypatch)
    assert ind.gap == l1_indicators.GAP_WINDOW_NONE, f"缺口码变了:{ind.gap}"
    assert ind.gap in l1_indicators.KINDS, "缺口码不在既有 9 类里"
    assert math.isnan(ind.value) and ind.n_valid == 0, "报缺口却带了值"
    assert ind.gap_detail == _detail(True, 0), f"事实没带上或带错了:{ind.gap_detail}"

    # 没有台账 + 语音日志 0 行 ⟹ 同样还是 window_none,只是事实不同
    ind = _rt_indicator(None, "20260101_000000_ffff", has_ledger=False, monkeypatch=monkeypatch)
    assert ind.gap == l1_indicators.GAP_WINDOW_NONE, f"缺口码变了:{ind.gap}"
    assert ind.gap_detail == _detail(False, 0), f"事实没带上或带错了:{ind.gap_detail}"

    # 有语音行、但没有应答时刻那一列 ⟹ not_stored(与改之前一致)
    ind = _rt_indicator(_voice_df(8, with_rt=False), "20260101_000000_gggg",
                        has_ledger=True, monkeypatch=monkeypatch)
    assert ind.gap == l1_indicators.GAP_NOT_STORED, f"缺口码变了:{ind.gap}"
    assert ind.gap_detail["voice_rows"] == 8, f"语音行数没带上:{ind.gap_detail}"
    assert ind.gap_detail["rt_col_present"] is False


def test_a_session_with_a_value_carries_no_gap_and_no_excuse(monkeypatch):
    """反例阵:语音日志 8 行、有应答时刻 ⟹ 出值、无缺口、无事实。

    实测这一场是 `20260927_212143_95a8`(8.155 秒)。防御性分支的另一半:
    "有值"时不许带 `gap`,更不许带一份"为什么没有值"的事实。
    """
    ind = _rt_indicator(_voice_df(8, with_rt=True), "20260101_000000_hhhh",
                        has_ledger=True, monkeypatch=monkeypatch)
    assert ind.gap is None, f"有值却报了缺口:{ind.gap}"
    assert not math.isnan(ind.value), "该出值的那一场没有值"
    assert abs(ind.value - 8.0) < 1e-9, f"值不对:{ind.value}"
    assert ind.gap_detail is None, "有值却带了缺口归因事实"


# ---------------------------------------------------------------------------
# ⑤ 三个出口说同一句:明细行 / 报告正文
# ---------------------------------------------------------------------------

def _quality_row(ind) -> dict:
    """`build_quality_report` 的明细行(与报告页脚 / 前端标签同一份)。"""
    from report_frontend.event_windows import WindowSet
    from report_frontend.l1_session import L1Result
    from report_frontend.session_timeline import Timebase, TimebaseStatus
    import types

    face = Timebase(modality="face", status=TimebaseStatus.OK,
                    t_sec=pd.Series([0.0, 1.0]), n_frames=2,
                    wall_back_steps=0, drift_sec=0.0)
    result = L1Result(
        session_id="20260101_000000_iiii",
        timebase=types.SimpleNamespace(modalities={"face": face}),
        windows=WindowSet(session_id="20260101_000000_iiii",
                          granularity="session", windows=[], n_questions=0),
        indicators=[ind],
    )
    return l1_quality.build_quality_report(result).indicators[0]


def test_the_audit_row_explains_by_evidence_and_gains_no_new_key(monkeypatch):
    """★ 红法:把事实塞进明细行(多一个 `gap_detail` 键)⟹ 红。

    事实是**选句的依据**,不是要发的载荷。明细行的键集是既有契约 ——
    `tests/test_gap_explanations.py` 里那条"新增键只允许仪器来源两键"就钉在这里,
    本条守的是它在这个指标上同样成立。
    """
    ind = _rt_indicator(None, "20260101_000000_jjjj", has_ledger=True, monkeypatch=monkeypatch)
    row = _quality_row(ind)
    assert row["gap_explanation"] == _text(_detail(True, 0)), (
        f"明细行没有按证据选句:{row['gap_explanation']}")
    assert "gap_detail" not in row, "诊断事实漏进了明细行(键集被悄悄改了)"
    assert row["gap"] == l1_indicators.GAP_WINDOW_NONE and row["value"] is None


def test_the_report_body_line_says_the_same_thing(monkeypatch):
    """★ 红法:把 `_gap_lines` 里传 `gap_detail` 的那一处删掉 ⟹ 红。

    这是**报告正文**那条通路(slot 由 `l1_mapper` 装、句由 `l1_report` 选)。
    实测踩雷的 6 场就死在这里:正文印「多数场次没存题目时刻表」,而台账在盘上。
    """
    ind = _rt_indicator(None, "20260101_000000_kkkk", has_ledger=True, monkeypatch=monkeypatch)
    dims, _ = l1_mapper.map_l1_to_dimensions([ind])
    dim = dims["cognitive_efficiency"]
    label = l1_mapper.slot_label("reaction_time")
    lines = l1_report._gap_lines(dim)
    want = f"{label}:{_text(_detail(True, 0))}"
    assert want in lines, f"正文没按证据说:{lines}"
    assert all(_OLD_CLAIM not in line for line in lines), (
        f"正文里还留着那句一刀切的判词:{lines}")
