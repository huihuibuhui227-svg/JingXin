# tests/test_m4_per_question_windows.py
"""**M4.17 逐题窗口**的三个缺口。

## 钉的是什么

1. **落在窗口外的帧必须数出来。** 有台账时原点 = `min(ask_start)`,而第一个窗口从
   第一个 `ask_end` 才开始 ⟹ 提问期(以及首题之前的一切帧)**不属于任何窗口**,
   `assign_series` 对它们静默返回 `None`。不数就等于分母悄悄变小而没人知道 ——
   而 F/G 两族的比率类指标全靠这个分母。
2. **没有逐题回答音频是一种采集模式,不是"段数对不上"的缺陷。**
   原先那条分支会给"只有提问标记、没有每题音频"的场次**每场永久加一条假缺陷**,
   把真缺陷稀释掉。
3. **合成台账 → `build_windows` → F/G 逐题指标** 这条链**此前没有任何测试串起来**:
   `tests/test_m4_l1_indicators.py` 与 `tests/test_m4_l1_voice_gesture.py` 都直接
   构造 `WindowSet`,`build_windows` 只在 `tests/test_m4_event_windows.py` 里单独测。
   两段各自绿、接起来错,是最典型的漏法。

## 不依赖 `data/`

全部合成:台账写进 `tmp_path`,`JINGXIN_RECORDINGS_DIR` 由 fixture 兜住。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from report_frontend import event_windows as ew
from report_frontend import l1_face, l1_gesture

SID = "20261018_130000_cd34"
ORIGIN = 1_790_400_000.0


@pytest.fixture()
def rec_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    return tmp_path


def _questions(root: Path, ask_ends: list[float], ask_len: float = 3.0) -> None:
    """写一份合成台账。`ask_ends` 是**距原点秒**,落盘时加上 `ORIGIN` 变成墙钟。

    ⚠️ 台账里必须是**墙钟秒**(`session_meta._validate_window` 会拦),而
    `build_windows` 做的是 `t = ask_end - origin_wall`。所以两者必须同一量纲 ——
    把相对秒当墙钟写进去,窗口会落在 1970 年,而**测试仍可能绿**(只要 `session_end_sec`
    也跟着错)。本文件第一条测试专门断言窗口坐标,就是为了钉住这个减法。
    """
    d = root / SID
    d.mkdir(parents=True, exist_ok=True)
    (d / "questions.jsonl").write_text("\n".join(json.dumps({
        "qid": f"题目{i}", "index": i,
        "ask_start": ORIGIN + e - ask_len, "ask_end": ORIGIN + e,
        "source": "/session/question", "session_id": SID,
    }) for i, e in enumerate(ask_ends)) + "\n", encoding="utf-8")


# ── 1. 窗口外的帧 ────────────────────────────────────────────────────────

def test_count_outside_counts_frames_before_the_first_ask_end(rec_dir):
    """★ 首题 `ask_end` 之前的帧不属于任何窗口,必须**数出来**。

    两题:40 秒念完、60 秒念完,会话到 100 秒 ⟹ 窗口是 [40,60) 和 [60,100)。
    1 Hz 的 100 帧里,`t=0..39` 这 **40 帧**落在窗口外。

    红法:把 `count_outside` 改成恒返回 0 ⟹ 本条得 0 ≠ 40。
    """
    _questions(rec_dir, [40.0, 60.0])
    ws = ew.build_windows(SID, ORIGIN, session_end_sec=100.0)
    assert ws.granularity == ew.ANSWER_GRANULARITY
    assert len(ws) == 2
    # 窗口坐标必须是"距原点秒"(40/60),不是墙钟 1.79e9 —— 钉住那次减法。
    assert [(w.t_start, w.t_end) for w in ws.windows] == [(40.0, 60.0), (60.0, 100.0)]

    t = pd.Series(np.arange(100, dtype=float))
    assert ws.count_outside(t) == 40

    # NaN 不算"落在窗口外" —— 缺失不是位置。
    assert ws.count_outside(pd.Series([0.0, np.nan, 50.0])) == 1


def test_count_outside_is_zero_when_the_whole_session_is_one_window(rec_dir):
    """会话级(没有台账)时整场就是一个窗口,没有东西被排除 ⟹ 恒 0。

    红法:去掉 `count_outside` 开头那个 `if not self.windows: return 0`,
    本条会把 100 帧全数成"窗口外"。
    """
    ws = ew.build_windows(SID, ORIGIN, session_end_sec=100.0)   # 无台账
    assert ws.granularity == ew.SESSION_GRANULARITY
    assert ws.count_outside(pd.Series(np.arange(100, dtype=float))) == 0


# ── 2. 没有逐题音频 ≠ 缺陷 ───────────────────────────────────────────────

def test_no_answer_audio_is_a_capture_mode_not_a_mismatch(rec_dir):
    """★ 只有提问标记、没有每题音频 ⟹ 中性说明,不是"段数对不上"。

    原先 `len(segments) != len(usable)` 在 `segments == 0` 时也成立,于是这种场次
    **每一场**都得到"answer_audio 段 0 条 vs 可用题窗口 N 个 —— 对不上;…要复核"。
    那是把**采集模式**报成了**缺陷**,而且它会稀释真正的段数错配。

    红法:删掉 `elif not segments:` 那一支 ⟹ gap 变成"对不上",第一条断言红。
    """
    _questions(rec_dir, [40.0, 60.0])
    ws = ew.build_windows(SID, ORIGIN, session_end_sec=100.0)
    gap = ws.gap or ""
    assert "对不上" not in gap, f"没有逐题音频不该报成对不上:{gap}"
    assert "没有逐题回答音频" in gap, f"必须说清是哪种模式:{gap}"
    assert "response_latency_sec" in gap, f"必须说清后果(反应延迟不可出):{gap}"


def test_answer_audio_count_mismatch_is_still_reported(rec_dir):
    """反过来:真有逐题音频但段数对不上 ⟹ 仍要报"对不上"(别把守卫一起删了)。

    红法:把 `if segments and …` 里的 `segments and` 去掉 ⟹ 本条仍绿(说明红法无效),
    所以这条的红法只能是**删掉整个 if 分支** ⟹ gap 变成 None。
    """
    _questions(rec_dir, [40.0, 60.0])
    d = rec_dir / SID
    (d / "retention.audio.jsonl").write_text("\n".join(json.dumps({
        "kind": "raw", "modality": None, "seq": i + 1,
        "file": f"media/audio/{i + 1:04d}.webm", "bytes": 10, "sha256": "x",
        "received_at_wall": ORIGIN + 50 + i, "declared_ts": None,
        "source_endpoint": "/interview/answer_audio", "session_id": SID,
    }) for i in range(3)) + "\n", encoding="utf-8")   # 3 段 vs 2 题
    ws = ew.build_windows(SID, ORIGIN, session_end_sec=100.0)
    assert "对不上" in (ws.gap or "")


# ── 3. 台账 → 窗口 → F/G,整条链 ──────────────────────────────────────────

def test_synthetic_ledger_drives_per_answer_face_and_gesture_rows(rec_dir):
    """★ 真台账 → 真 `build_windows` → 真 F/G 函数,窗口标签必须是 `answer:k`。

    这条链此前**没有任何测试**串起来:上游测 `build_windows`,下游测
    `compute_*_indicators(WindowSet(...))`,中间那段(窗口的时间量是不是同一量纲、
    `t_sec` 要不要减原点)从来没有被同时验过。

    红法:把台账写到别的目录(或 `SID` 写错)⟹ `build_windows` 退回 `session` 粒度,
    下面 `"answer:0" in wins` 立刻红。
    """
    _questions(rec_dir, [40.0, 60.0])
    ws = ew.build_windows(SID, ORIGIN, session_end_sec=100.0)
    assert ws.granularity == ew.ANSWER_GRANULARITY and len(ws) == 2

    t = pd.Series(np.arange(100, dtype=float))
    face = pd.DataFrame({
        "is_blink": np.zeros(100, dtype=bool),
        "gaze_direction_x": np.linspace(0.0, 1.0, 100),
        "head_yaw": np.zeros(100), "head_pitch": np.zeros(100),
    })
    f_wins = {i.window for i in l1_face.compute_face_indicators(face, ws, t)}
    assert "answer:0" in f_wins and "answer:1" in f_wins, f"F 族没按题切:{sorted(f_wins)}"
    assert "session" not in f_wins, "有台账就不该再出现会话级行"

    gest = pd.DataFrame({
        "timestamp": t.to_numpy(dtype=float),
        **{c: np.zeros(100) for c in l1_gesture.ACTIVITY_COLS},
        **{c: np.zeros(100) for c in l1_gesture.POSTURE_COLS},
    })
    g_wins = {i.window for i in l1_gesture.compute_gesture_indicators(gest, ws, t)}
    assert "answer:0" in g_wins and "answer:1" in g_wins, f"G 族没按题切:{sorted(g_wins)}"


# ── 4. "被排除的帧多到值得说一句"的阈值判断 ───────────────────────────────

def test_frames_outside_note_speaks_only_above_five_percent():
    """★ 纯函数:`>5%` 才出一行说明,否则 None。

    为什么单独测它:`compute_session_l1` 那条管道要真时间轴才跑得动(合成日志过不了
    face 时间轴校验),而这条判断是纯算术 —— 抽出来才能钉住。
    与 `segment_gap_note`(丙-37)同一做法。

    红法:把 `frac` 的默认值改成 0.0 ⟹ 第一条断言(None)红。
    """
    from report_frontend.l1_session import frames_outside_note

    # 40/100 = 40% ⟹ 说话
    note = frames_outside_note({"face": 40}, {"face": 100})
    assert note and "face 40/100 帧(40%)" in note

    # 1/100 = 1% ⟹ 不说
    assert frames_outside_note({"face": 1}, {"face": 100}) is None

    # 5% 边界:严格大于才说
    assert frames_outside_note({"face": 5}, {"face": 100}) is None
    assert frames_outside_note({"face": 6}, {"face": 100}) is not None

    # 分母为 0(模态没有帧)不许除零,也不许当成"全被排除"
    assert frames_outside_note({"face": 3}, {"face": 0}) is None

    # 两个模态都超标 ⟹ 都要点到
    both = frames_outside_note({"face": 40, "gesture": 50},
                               {"face": 100, "gesture": 100})
    assert both and "face" in both and "gesture" in both
