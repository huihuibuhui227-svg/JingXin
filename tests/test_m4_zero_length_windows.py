# tests/test_m4_zero_length_windows.py
"""**零长窗口必须能看见**(规格 §7 第 3 条)。

## 钉的是什么

两题的 `ask_end` 撞在一起(或倒序)时,前一个窗口时长为 0 ⟹ 那一题**不会有任何观测**。
`assign()` 对窗口外的帧静默返回 `None`,所以不报就永远不知道。

原实现把这条消息挂在 `if/elif` 链的**第三位**:

```python
if segments and len(segments) != len(usable):   ...
elif not segments:                              ...
elif empty:                                     ...   # ← 只有前两条都不成立才到这里
```

⟹ 只要"段数与窗口数对不上"(最常见的现场情况)或"没有逐题音频",零长窗口
就被那条更响的消息**盖掉**,而它才是"这一题没有任何观测"的直接原因。
两条消息现在各自独立拼接,谁也盖不住谁。

## 不依赖 `data/`

台账写进 `tmp_path`;`JINGXIN_RECORDINGS_DIR` 由 fixture 兜住。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from report_frontend import event_windows as ew

SID = "20261024_101010_zero"
ORIGIN = 1_790_000_000.0


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    return tmp_path


def _questions(tmp_path: Path, ask_ends: list[float], ask_start: float = 0.0) -> None:
    d = tmp_path / SID
    d.mkdir(parents=True, exist_ok=True)
    with (d / "questions.jsonl").open("w", encoding="utf-8") as f:
        for i, ae in enumerate(ask_ends):
            f.write(json.dumps({
                "qid": f"题{i}", "index": i,
                "ask_start": ORIGIN + ask_start + i, "ask_end": ORIGIN + ae,
                "source": "/session/question",
            }, ensure_ascii=False) + "\n")


def _audio(tmp_path: Path, n: int) -> None:
    """造 n 条 `/interview/answer_audio` raw 记录 —— 只为让段数参与分支判断。"""
    d = tmp_path / SID
    d.mkdir(parents=True, exist_ok=True)
    with (d / "retention.audio.jsonl").open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"kind": "raw", "seq": i + 1,
                                "source_endpoint": "/interview/answer_audio",
                                "file": f"media/audio/{i+1:04d}.webm"}) + "\n")


def test_zero_length_window_survives_a_segment_count_mismatch(tmp_path):
    """★ 段数对不上时,零长窗口**仍然**要被报出来。

    这是原实现漏掉的那条路径:两题 ask_end 相同 + 音频段数也对不上,
    `elif empty` 被 `if segments and ...` 盖住 ⟹ 静默。

    红法:把 `if empty:` 改回 `elif empty:` 挂在链尾 ⟹ 本条红。
    """
    _questions(tmp_path, [40.0, 40.0])          # 两题同一时刻问完 ⟹ 第一窗长 0
    _audio(tmp_path, 5)                          # 段数 ≠ 窗口数,触发第一条消息
    ws = ew.build_windows(SID, ORIGIN, session_end_sec=100.0)
    assert ws.gap is not None
    assert "时长为 0" in ws.gap, ws.gap
    assert "对不上" in ws.gap, "两条消息应当同时在,不能互相盖住"


def test_zero_length_window_survives_missing_audio(tmp_path):
    """没有逐题音频时,零长窗口同样要被报出来(原实现这里也漏)。"""
    _questions(tmp_path, [40.0, 40.0])
    ws = ew.build_windows(SID, ORIGIN, session_end_sec=100.0)
    assert "时长为 0" in ws.gap
    assert "没有逐题回答音频" in ws.gap


def test_zero_length_window_alone(tmp_path):
    """段数正好对得上、无零长窗口时,只有那条中性说明,不该冒出"时长为 0"。"""
    _questions(tmp_path, [40.0, 60.0])
    _audio(tmp_path, 2)
    ws = ew.build_windows(SID, ORIGIN, session_end_sec=100.0)
    assert "时长为 0" not in (ws.gap or "")

# ── 末窗右边界早于最后一题 ask_end ──────────────────────────────────────
#
# 这一条与"相邻两题撞在一起"是**两个不同的缺陷**,补救动作也不同:
# 前者的根在台账,后者的根在**调用方给的 session_end_sec**。
# 2026-10-24 实测:三个模态全都不可用时 `l1_session` 兜底传 `session_end_sec=0.0`,
# 而台账末题的 ask_end 是距原点 288.7 秒 ⟹ 末窗变成 `[288.7, 0.0)`、时长 −288.7 秒。


def test_end_before_last_ask_end_does_not_produce_a_negative_window(tmp_path):
    """★ 右边界早于末题 ask_end 时,末窗**不许**是倒置区间。

    红法:删掉 `build_windows` 里的 `end_clamped` 收口 ⟹ 本条红
    (末窗时长 −288.7)。
    """
    _questions(tmp_path, [10.0, 40.0, 288.7])
    ws = ew.build_windows(SID, ORIGIN, session_end_sec=0.0)
    assert all(w.duration_sec >= 0 for w in ws.windows), \
        [(w.index, w.t_start, w.t_end, w.duration_sec) for w in ws.windows]
    assert "早于最后一题的" in (ws.gap or ""), ws.gap


def test_inverted_window_would_have_emptied_the_union_denominator(tmp_path):
    """★ 倒置窗口的**真实后果**:并集口径下 F/G 的分母静默变空。

    这不是"多一个负数"的观感问题 —— `contains()` 左闭右开,
    `t >= 288.7 and t < 0.0` 恒假,末窗一帧都取不到;而并集右边界取的就是
    末窗的 `t_end` ⟹ 整个并集跟着倒置,一场会话的 F/G 全部分母为 0。

    红法:同上(删 `end_clamped`) ⟹ 并集 `t_end < t_start`,本条红。
    """
    _questions(tmp_path, [10.0, 40.0, 288.7])
    ws = ew.build_windows(SID, ORIGIN, session_end_sec=0.0)
    u = ws.answer_union()
    assert len(u.windows) == 1
    assert u.windows[0].t_end >= u.windows[0].t_start, u.windows[0]
    # 并集必须仍覆盖末窗起点 —— 收口到 ask_end 而不是砍掉这一题。
    assert u.windows[0].t_end >= 288.7


def test_clip_session_end_never_inverts_the_last_window():
    """`clip_session_end` 是同一类缺陷的第二处入口,一并守住下界。"""
    ws = ew.WindowSet(SID, ew.ANSWER_GRANULARITY,
                      [ew.AnswerWindow(index=0, t_start=100.0, t_end=200.0)])
    out = ew.clip_session_end(ws, 50.0)
    assert out.windows[-1].duration_sec >= 0
    assert out.windows[-1].t_end == 100.0
