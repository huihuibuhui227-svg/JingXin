# tests/test_report_header_agrees_with_the_payload.py
"""报告头说的那句,不许与同一份载荷里的缺口**互相打架**。

## 实测缺陷(2026-10-25 深夜,服务器上真跑的一场)

`20260928_180131_8102` 的 HTML 报告头写着:

    语音（面试） · 未读到数据
    （日志文件在，但一行数据都没有。常见原因是本场没有提交过逐题回答
     —— 例如从「实时分析」页录的，那条路没有题库；若本场确实答过题，那才是文件损坏。）

而同一份报告的「本场报告的限制」里写着:

    voice 模态来自原始音频重算(日志 0 行),时间轴未与墙钟核对
    本场没有 voice 日志 —— V 族 12 条里 10 条由整场录音重算得出,其余 2 条不可出

并且「沟通表达流畅度」那一维**真的印出了两条指标**(语调变化 22.868、平均停顿时长 0.6720)。

⟹ 同一个模态,报告头说「读不到、可能文件损坏」,报告正文说「算出来了 10 条」。
读者先看到头,会以为这一场废了 —— 而它其实是本场最完整的一个模态。

## 根因:两个出口各自只看得到一半

* 报告头看的是 `LogDataLoader.selected_sessions` —— **只描述日志文件**;
* L1 层知道语音有**第二个数据源**(`voice_series`,整场录音逐帧重算)。

日志确实 0 行,这句是真的;假的是它引出的**后果**("所以这个模态没有数据")。
`l1_quality.QualityReport.series_available` 早就把第二数据源记下来了,
`validate()` 甚至专门为它开了豁免 —— 只是**没有人把这件事告诉报告头**。

## 修法:载荷给出「每个模态到底算出了几条」,报告头照抄

按第 16 条(分层)的做法:数字与措辞都由载荷给,渲染层不自己推。
`l1.modality_summary` = `{模态: {n_total, n_measured, source_note}}`,
由 `l1_quality.modality_summary()` 从 `families` 汇总 —— 那是已经算好的东西,不重算。

## 这个文件守什么

1. 有值的时候,报告头**不许**再说「未读到数据 / 文件损坏」;
2. 一条都没算出来时,**保留**原措辞(那是真的:该去查采集或补提取);
3. 载荷真的带上了每个模态的条数(不是渲染层自己数的);
4. 走完整 `generate_report` 路径时,头与缺口说的是同一件事。
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

import report_frontend.report_generator as report_generator
from report_frontend import l1_session, voice_series
from report_frontend.data_loader import LogDataLoader
from report_frontend.report_generator import ReportGenerator, sources_disclosure

_SID = "20260924_221811_9212"


def _voice_summary(n_measured: int, n_total: int = 12) -> dict:
    return {"voice": {"n_total": n_total, "n_measured": n_measured,
                      "source_note": "由整场录音重算得出"}}


def test_a_modality_that_produced_values_is_not_reported_as_missing_data():
    """★ 红法:让 `sources_disclosure` 忽略 `modality_summary`(回到只看日志三态)⟹ 红。

    「日志 0 行」是真的,但由此说「未读到数据」并暗示文件损坏,与载荷里的
    「V 族 12 条里 10 条有值」直接矛盾。
    """
    html = sources_disclosure(
        {"voice_interview": {"session_id": _SID, "status": "unreadable", "rows": 0}},
        _SID, modality_summary=_voice_summary(10))

    assert "未读到数据" not in html, (
        f"语音算出了 10 条,报告头却说未读到数据:{html}")
    assert "文件损坏" not in html, f"不该暗示文件损坏:{html}"
    assert "10" in html and "12" in html, (
        f"头里必须写出「12 条里有 10 条算出了值」,读者才知道这一维不是空的:{html}")
    assert "整场录音重算" in html, f"必须说清这些值是哪来的:{html}"


def test_a_modality_with_zero_values_keeps_the_old_wording():
    """★ 回归地板:一条都没算出来时,原措辞**必须留着**。

    那是真的:日志空 + 没有序列缓存 ⟹ 这个模态本场确实没有量,
    读者该去查采集或补一次提取。把它也换成温和措辞,就是把「真缺」说没了。
    """
    html = sources_disclosure(
        {"voice_interview": {"session_id": _SID, "status": "unreadable", "rows": 0}},
        _SID, modality_summary=_voice_summary(0))

    assert "未读到数据" in html, f"0 条有值却说成没事:{html}"


def test_a_missing_log_with_recomputed_values_says_so():
    """`missing`(盘上连日志文件都没有)与 `unreadable` 是两句话,但结论相同。

    红法:只处理 `unreadable` 分支 ⟹ 这一条红。
    """
    html = sources_disclosure(
        {"voice_interview": {"session_id": _SID, "status": "missing", "rows": 0}},
        _SID, modality_summary=_voice_summary(9))

    assert "缺失" not in html, f"算出了 9 条,却说这个模态缺失:{html}"
    assert "9" in html and "12" in html, html


def test_the_payload_carries_the_per_modality_counts(tmp_path, monkeypatch):
    """★ 红法:把 `l1.modality_summary` 从载荷里删掉 ⟹ 红。

    渲染层不许自己数(它拿不到 `families` 的模态归属),所以这件事必须由载荷说。
    走真路径:`build_l1_report` → `result["l1"]["modality_summary"]`。
    """
    from report_frontend import l1_report, voice_series

    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path))
    voice_series.save_series([voice_series.VoiceSeries(
        session_id=_SID, index=None, frame_sec=0.1,
        f0=np.where(np.arange(200) % 7 == 0, np.nan,
                    120.0 + 10.0 * np.sin(np.arange(200) / 5.0)),
        voiced_prob=np.ones(200), rms=np.full(200, 0.05), source="camera.webm")])

    payload = l1_report.build_l1_report(_SID, logs={})
    summary = payload["l1"]["modality_summary"]

    assert "voice" in summary, f"载荷没给语音的条数:{summary.keys()}"
    v = summary["voice"]
    n_val = sum(1 for i in payload["l1"]["indicators"]
                if i["family"].startswith("V") and i["value"] is not None)
    assert v["n_measured"] == n_val > 0, (
        f"载荷里的条数与指标明细对不上:{v} vs {n_val}")
    assert v["n_total"] == sum(1 for i in payload["l1"]["indicators"]
                               if i["family"].startswith("V"))
    assert v["source_note"], "没说这些值是哪来的 —— 读者会以为是日志里的"


def test_the_recompute_note_only_appears_when_the_log_really_was_empty(tmp_path, monkeypatch):
    """★ 红法:去掉 `modality_summary` 里 `not log_rows` 那个条件 ⟹ 红。

    有语音日志的场次,值来自日志。此时说「由整场录音重算得出」是**反向**骗人:
    读者会以为日志没被用上,而重算出的序列其实只是同一份录音的另一个读法。
    这条与上一条成对 —— 上一条守"没有日志时要说出重算",这条守"有日志时别乱说"。
    """
    import pandas as pd

    from report_frontend import l1_quality

    voice = pd.DataFrame({
        "n_chars": [30.0, 40.0], "speech_duration_sec": [6.0, 8.0],
        "duration_sec": [9.0, 12.0], "pause_frequency": [20.0, 30.0],
        "reaction_time": [5.0, 8.0], "pitch_mean": [100.0, 110.0],
        "pitch_p10": [80.0, 85.0], "pitch_p90": [120.0, 130.0],
        "energy_mean": [0.05, 0.06],
    })
    # 序列缓存**也在**(很多场同时有日志和序列)—— 正是这条测试要的形状。
    # ⚠️ 必须在 `compute_session_l1` **之前**放好:它决定 `series_available`,
    #    而 `validate()` 靠这个豁免"日志没有、序列有"的场次。顺序反了会红在
    #    「自称有值但模态不存在」上,测的就不是措辞了。
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path))
    voice_series.save_series([voice_series.VoiceSeries(
        session_id=_SID, index=None, frame_sec=0.1,
        f0=np.where(np.arange(200) % 7 == 0, np.nan, 120.0),
        voiced_prob=np.ones(200), rms=np.full(200, 0.05), source="camera.webm")])
    result = l1_session.compute_session_l1(_SID, logs={"voice_interview": voice})
    summary = l1_quality.modality_summary(l1_quality.build_quality_report(result))

    assert summary["voice"]["n_measured"] > 0, "前提不成立:语音没算出值"
    assert not summary["voice"]["source_note"], (
        f"日志有 {len(voice)} 行,却说值是重算来的:{summary['voice']}")


def test_the_produced_line_is_printed_once_per_modality_not_once_per_log_file():
    """★ 红法:去掉按模态去重那一步 ⟹ 红。

    ## 实测(2026-10-25 深夜,服务器上重生成的真报告)

    修好上面那条之后,报告头变成:

        语音（面试） · 日志没有数据，但这个模态算出了 10 / 12 条指标（由整场录音重算得出）
        语音（科研） · 日志没有数据，但这个模态算出了 10 / 12 条指标（由整场录音重算得出）

    **同一批 10 条指标被报了两次**,读起来像两个模态各算出了 10 条。

    根因:日志按「面试 / 科研」分两个文件键(`voice_interview` / `voice_research`),
    而 L1 的 V 族**不分** —— 12 条指标是同一套,`modality_summary` 只有一个 `voice`。
    三态是**逐文件**的事实(两个文件各说各的没错),但"算出了几条"是**逐模态**的,
    说两遍就是把同一份证据数了两次 —— 本仓对这件事一向是标出来(见
    `evidence_gate` 里 `pause_ratio` 的处置)。

    ⟹ 有值那句按**模态**去重,且标签用模态名(`语音`),不用文件名。
    """
    html = sources_disclosure(
        {"voice_interview": {"session_id": _SID, "status": "unreadable", "rows": 0},
         "voice_research": {"session_id": _SID, "status": "unreadable", "rows": 0}},
        _SID, modality_summary=_voice_summary(10))

    n = html.count("10 / 12")
    assert n == 1, (
        f"同一批 10 条指标报了 {n} 次 —— 读者会以为有两个模态各算出了 10 条:{html}")
    # 标签必须是模态名:V 族不区分面试/科研,挂着文件名会让人以为两者各有 12 条。
    assert "语音（面试） · <strong>日志没有数据" not in html, html
    assert "语音 · <strong>日志没有数据" in html, f"标签没用模态名:{html}"


def test_two_log_files_with_different_statuses_still_report_their_own_status():
    """★ 回归地板:三态是**逐文件**的事实,去重只许去重"算出了几条"那句。

    面试日志读到了、科研日志没有 —— 这是两条不同的事实,都得说。
    红法:把整个循环按模态去重(而不是只去重 produced 那句)⟹ 红。
    """
    html = sources_disclosure(
        {"voice_interview": {"session_id": _SID, "status": "loaded", "rows": 40},
         "voice_research": {"session_id": _SID, "status": "missing", "rows": 0}},
        _SID, modality_summary=_voice_summary(10))

    assert "语音（面试） · 已读入（40 行）" in html, html
    assert "语音（科研） · 缺失" in html, html


# ── 走完整路径:头与缺口必须说同一件事 ──────────────────────────────────

def _write_log(log_dir: Path, filename: str, rows: list) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    fields = ["session_id", "timestamp", "connective_density", "connective_density_std",
              "n_rows", "focus_score", "symmetry_score"]
    p = log_dir / filename
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


def _row(sid: str) -> dict:
    return {"session_id": sid, "timestamp": "2026-09-24T22:18:11",
            "connective_density": None, "connective_density_std": 0.5, "n_rows": 1,
            "focus_score": 0.5, "symmetry_score": 0.9}


def test_the_whole_report_tells_one_story_about_voice(tmp_path, monkeypatch):
    """★ 红法:同上(载荷不带 `modality_summary`)⟹ 头会说未读到数据,这里红。

    这条**故意走 `generate_report`**(真加载器 → 真渲染):单元测绿而整条路
    什么都不显示,是本仓栽过的形态(`test_report_sources.py` 的注释记过)。
    """
    from report_frontend import voice_series

    log_dir = tmp_path / "logs"
    # 语音日志只有表头(0 行)—— 24 场正式场全是这个形状。
    _write_log(log_dir, f"interview_emotion_log_{_SID}.csv", [])
    _write_log(log_dir, f"face_au_log_{_SID}.csv", [_row(_SID)])
    _write_log(log_dir, f"gesture_emotion_log_{_SID}.csv", [_row(_SID)])

    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path / "series"))
    voice_series.save_series([voice_series.VoiceSeries(
        session_id=_SID, index=None, frame_sec=0.1,
        f0=np.where(np.arange(200) % 7 == 0, np.nan,
                    120.0 + 10.0 * np.sin(np.arange(200) / 5.0)),
        voiced_prob=np.ones(200), rms=np.full(200, 0.05), source="camera.webm")])

    monkeypatch.setattr(report_generator, "LogDataLoader",
                        lambda *a, **k: LogDataLoader(str(log_dir)))
    monkeypatch.setattr(report_generator.webbrowser, "open", lambda *a, **k: None)

    path = ReportGenerator(output_dir=str(tmp_path / "out")).generate_report(_SID)
    assert path, "报告没生成 —— 本测试的前提不成立"
    html = Path(path).read_text(encoding="utf-8")

    assert "语音（面试）" in html, "连模态名都没有"
    assert "未读到数据" not in html, (
        "报告头说语音未读到数据,而同一份载荷里 V 族算出了值 —— 两处自相矛盾")
