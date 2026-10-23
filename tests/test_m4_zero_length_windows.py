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
