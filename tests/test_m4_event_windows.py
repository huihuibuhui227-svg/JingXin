# tests/test_m4_event_windows.py
"""**M4.1 回答窗口**的守卫。

## 钉的是什么

窗口粒度不定死,G3 的阈值就没有意义 —— `evidence_thresholds.json` 的
`n_valid_units` 登记了 `frame` / `answer`,而 `answer` 长期没有对应的实体。
这块以前是"一整场一个窗口",于是 `speech_ratio` 实测 **87.9% 恒为 1.0**。

本文件钉五件事:
1. 有台账 ⟹ 切成 N 个回答窗口,边界是 `ask_end`(不是 `ask_start`);
2. 窗口**首尾相接、不重不漏**(左闭右开);
3. 落在窗口外的时间点返回 `None`,**不四舍五入到最近**;
4. 没有台账 ⟹ 只有一个 `session` 窗口,并写明原因(戊-15 那 26 场);
5. `answer_audio` 段数与题数对不上 ⟹ 仍按 `ask_end` 切,但把矛盾写进 `gap`
   —— **不按顺序硬配**。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from report_frontend import event_windows as ew

SID = "20261018_130000_cd34"
ORIGIN = 1_790_400_000.0


@pytest.fixture()
def rec_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    return tmp_path


def _questions(root: Path, ask_ends: list[float], ask_len: float = 3.0) -> None:
    d = root / SID
    d.mkdir(parents=True, exist_ok=True)
    (d / "questions.jsonl").write_text("\n".join(json.dumps({
        "qid": f"题目{i}", "index": i,
        "ask_start": e - ask_len, "ask_end": e,
        "source": "/session/question", "session_id": SID,
    }) for i, e in enumerate(ask_ends)) + "\n", encoding="utf-8")


def _audio_segments(root: Path, arrivals: list[float]) -> None:
    d = root / SID
    d.mkdir(parents=True, exist_ok=True)
    (d / "retention.audio.jsonl").write_text("\n".join(json.dumps({
        "kind": "raw", "modality": None, "seq": i + 1,
        "file": f"media/audio/{i + 1:04d}.webm", "bytes": 10, "sha256": "x",
        "received_at_wall": a, "declared_ts": None,
        "source_endpoint": "/interview/answer_audio", "session_id": SID,
    }) for i, a in enumerate(arrivals)) + "\n", encoding="utf-8")


# ── 1. 切成 N 个回答窗口 ─────────────────────────────────────────────────

def test_windows_are_cut_at_ask_end(rec_dir):
    """★ 边界必须是 `ask_end`。用 `ask_start` 会把提问时长算进"反应延迟"。

    红法:把 `t_start` 改成 `ask_start - origin` ⟹ 每条窗口的起点左移 3 s,本条红。
    """
    _questions(rec_dir, [ORIGIN + 10, ORIGIN + 40, ORIGIN + 70])
    _audio_segments(rec_dir, [ORIGIN + 12, ORIGIN + 44, ORIGIN + 75])

    ws = ew.build_windows(SID, ORIGIN, session_end_sec=100.0)

    assert ws.granularity == ew.ANSWER_GRANULARITY
    assert len(ws) == 3
    assert [w.index for w in ws.windows] == [0, 1, 2]
    assert np.allclose([w.t_start for w in ws.windows], [10.0, 40.0, 70.0])
    assert ws.gap is None


# ── 2. 首尾相接、不重不漏 ────────────────────────────────────────────────

def test_windows_partition_the_session_without_gaps_or_overlap(rec_dir):
    """红法:把 `t_end` 写成 `ask_end + 1` ⟹ 相邻窗口之间漏掉 1 秒,本条红。"""
    _questions(rec_dir, [ORIGIN + 10, ORIGIN + 40, ORIGIN + 70])
    ws = ew.build_windows(SID, ORIGIN, session_end_sec=100.0)

    for a, b in zip(ws.windows, ws.windows[1:]):
        assert a.t_end == b.t_start, f"题{a.index} 与题{b.index} 之间断开了"
    assert ws.windows[-1].t_end == 100.0, "最后一个窗口必须收到会话结束"


# ── 3. 窗口外返回 None,不四舍五入 ────────────────────────────────────────

def test_timestamps_outside_every_window_are_none_not_snapped(rec_dir):
    """左闭右开:边界值归右窗口;场前(-5)与会后(200)都不属于任何窗口。

    红法:把 `contains` 改成 `abs(t - 中心) < 半径` 这种"就近归入" ⟹ 场外值被硬塞进去,本条红。
    """
    _questions(rec_dir, [ORIGIN + 10, ORIGIN + 40]);
    ws = ew.build_windows(SID, ORIGIN, session_end_sec=100.0)

    assert ws.assign(10.0) == 0, "左闭:边界值属于右边那个窗口"
    assert ws.assign(40.0) == 1
    assert ws.assign(39.999) == 0
    assert ws.assign(-5.0) is None, "场前的观测不属于任何回答窗口"
    assert ws.assign(200.0) is None, "场后的观测不属于任何回答窗口"
    assert ws.assign(float("nan")) is None, "缺失不是'落在某个窗口'"

    s = pd.Series([-1.0, 15.0, 45.0, 500.0, np.nan])
    got = ws.assign_series(s).tolist()
    assert got == [None, 0, 1, None, None]


# ── 4. 没有台账 ⟹ 会话级兜底,并写明原因 ─────────────────────────────────

def test_no_question_ledger_falls_back_to_session_granularity(rec_dir):
    """戊-15 那 26 场自然对话的形态。红法:把这一支改成 `ANSWER_GRANULARITY` + 假窗口 ⟹ 红。"""
    rec_dir.joinpath(SID).mkdir(parents=True, exist_ok=True)

    ws = ew.build_windows(SID, ORIGIN, session_end_sec=60.0)

    assert ws.granularity == ew.SESSION_GRANULARITY
    assert len(ws) == 0
    assert ws.gap and "提问台账" in ws.gap


# ── 5. 段数对不上 ⟹ 明说,不硬配 ─────────────────────────────────────────

def test_segment_count_mismatch_is_reported_not_zipped(rec_dir):
    """★ 题 3 个、音频段 5 段(重传)。窗口仍按 `ask_end` 切,矛盾进 `gap`。

    红法:按"第 i 段配第 i 题"硬配 ⟹ 多出的段被静默丢弃,本条红(它会断言 gap 非空)。
    """
    _questions(rec_dir, [ORIGIN + 10, ORIGIN + 40, ORIGIN + 70])
    _audio_segments(rec_dir, [ORIGIN + 12, ORIGIN + 20, ORIGIN + 44, ORIGIN + 50, ORIGIN + 75])

    ws = ew.build_windows(SID, ORIGIN, session_end_sec=100.0)

    assert len(ws) == 3, "窗口数只由台账决定,不被段数带偏"
    assert ws.n_questions == 3 and ws.n_answer_segments == 5
    assert ws.gap and "对不上" in ws.gap


# ── 6. 空窗口要能被看见 ─────────────────────────────────────────────────

def test_zero_length_window_is_flagged(rec_dir):
    """两题 `ask_end` 相同 ⟹ 前一题退化成空窗口。不是错误,但必须看得见。"""
    _questions(rec_dir, [ORIGIN + 10, ORIGIN + 10, ORIGIN + 40])
    _audio_segments(rec_dir, [ORIGIN + 12, ORIGIN + 13, ORIGIN + 44])

    ws = ew.build_windows(SID, ORIGIN, session_end_sec=100.0)

    assert len(ws) == 3
    assert ws.windows[0].duration_sec == 0.0
    assert ws.gap and "0" in ws.gap
