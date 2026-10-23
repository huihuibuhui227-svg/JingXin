# tests/test_m4_window_labels.py
"""**窗口标签不能比数先撒谎**(审计发现,2026-10-24)。

## 钉的是什么

G 族的"不可出"占位值把窗口标签**写死**成 `"answer:0"`:

    def _un(name, gap):
        return unavailable(..., "answer:0", "", gap)

后果:整场口径(没有台账,只有一个 `session` 窗口)下,G 族 7 条带着
`answer:0` 进报告,而同一份报告里 F 族 16 条带的是 `session`。
读者看到"同一场里两个族的窗口不同",会以为逐题窗口生效了 ——
实际那是没算出来的占位值,连一题都没对上。

实测 `20260926_153202_2b11` 的标签分布:`{'answer:-1': 16, 'answer:0': 7, 'session': 15}`。

## 红法

把 `_un` 的 `window` 参数默认值改回 `"answer:0"` ⟹ 前两条红。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from report_frontend import event_windows as ew
from report_frontend import l1_face, l1_gesture

SID = "20261024_101010_labels"
ORIGIN = 1_790_000_000.0
N = 400


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    return tmp_path


def _face_df(n: int = N) -> pd.DataFrame:
    return pd.DataFrame({c: np.zeros(n) for c in (
        "gaze_direction_x", "is_blink", "head_pitch", "head_yaw", "head_roll", "avg_ear")})


def _gest_df(n: int = N) -> pd.DataFrame:
    return pd.DataFrame({c: np.zeros(n) for c in (
        "torso_jitter", "left_shoulder_jitter", "right_shoulder_jitter",
        "shoulder_width", "hand_visible_left", "hand_visible_right")})


def _t(n: int = N) -> pd.Series:
    return pd.Series(np.arange(n, dtype=float))


def test_gesture_unavailable_uses_session_label_when_there_are_no_windows():
    """★ 没有窗口 ⟹ G 族的"不可出"必须标 `session`,不能标 `answer:0`。

    红法:`_un` 的 `window` 默认值改回 `"answer:0"` ⟹ 本条红。
    """
    ws = ew.WindowSet(SID, ew.SESSION_GRANULARITY, [])
    out = l1_gesture.compute_gesture_indicators(None, ws, None)
    assert {i.window for i in out} == {"session"}, {i.window for i in out}


def test_face_and_gesture_agree_on_the_label_in_session_caliber():
    """★ 同一份报告里两族口径必须一致 —— 否则读者会以为逐题窗口生效了。"""
    ws = ew.WindowSet(SID, ew.SESSION_GRANULARITY, [])
    f = l1_face.compute_face_indicators(None, ws, None)
    g = l1_gesture.compute_gesture_indicators(None, ws, None)
    assert {i.window for i in f} == {i.window for i in g} == {"session"}


def test_gesture_unavailable_uses_answer_labels_when_windows_exist():
    """有窗口时反过来:必须逐题标,不能退回 `session`。

    红法:把 `_all_unavailable` 的 `labels` 改成永远 `[_SESSION]` ⟹ 本条红。
    """
    ws = ew.WindowSet(SID, ew.ANSWER_GRANULARITY,
                      [ew.AnswerWindow(0, 0.0, 100.0), ew.AnswerWindow(1, 100.0, 200.0)],
                      n_questions=2)
    out = l1_gesture.compute_gesture_indicators(None, ws, None)
    assert {i.window for i in out} == {"answer:0", "answer:1"}
    # 每个窗口都要有整族 7 条 —— 不能只给第一个窗口
    from collections import Counter
    assert Counter(i.window for i in out) == {"answer:0": len(l1_gesture.ORDER),
                                              "answer:1": len(l1_gesture.ORDER)}


def test_union_window_labels_every_family_the_same_way():
    """并集窗口(`index=-1`)下,F/G 两族必须给同一个标签。

    红法:`answer_union()` 里的 `index=-1` 改成 0 ⟹ 标签变成 `answer:0`,
    本条仍绿(它只断言两族**一致**)—— 所以另加一条断言 index 是 -1。
    """
    ws = ew.WindowSet(SID, ew.ANSWER_GRANULARITY,
                      [ew.AnswerWindow(0, 0.0, 100.0), ew.AnswerWindow(1, 100.0, 200.0)],
                      n_questions=2)
    u = ws.answer_union()
    assert u.windows[0].index == -1, "并集窗口的 index 必须是 -1(不是 0,会与第 0 题撞)"
    f = l1_face.compute_face_indicators(_face_df(), u, _t())
    g = l1_gesture.compute_gesture_indicators(_gest_df(), u, _t())
    assert {i.window for i in f} == {i.window for i in g} == {"answer:-1"}
