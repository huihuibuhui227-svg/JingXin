"""`l1_session.capture_segments` / `segment_gap_note` —— 拼接场的切段与告警。

## 为什么要有这组测试

13 场 A 档正式语料里 **候选B一场是 9 段录制拼起来的**,而其余 12 场是单段。
查证:`retention.camera.jsonl` 9 条记录全指向同一个 `media/camera.webm`、`seq` 都是 1,
但 9 个不同的大小与 sha256 ⟹ 每段结束就**整文件重传并覆盖**;9 次上传都紧跟某段结束后
26~87 秒;每份字节数 ÷ 该段时长恒为 ~315 KB/s ⟹ 每次上传的就是该段完整录像。
于是盘上只剩**最后一段**(camera.webm 实测 434.7 秒),而 F/G 两族覆盖全部 9 段
共 6959 秒 ⟹ **同一场里不同族的观察窗口不一样长**。

这条如果不查,"这场有多长"就会被算成首末帧之差(7427 秒),把 468 秒的重启空档
也算进去,而 V/T 只覆盖最后 7.2 分钟这件事没有任何地方会说。

全部用**合成时间轴**,不依赖 `data/`。
"""
from __future__ import annotations

import numpy as np
import pytest

from report_frontend.l1_session import (SEGMENT_GAP_SEC, capture_segments,
                                        segment_gap_note)


def _ts(n, gaps=None):
    """1 Hz 的 n 帧;`gaps={下标: 该帧到下一帧的间隔秒数}`(默认 1 秒)。

    用"间隔"而不是"插入的空档"表述,是为了让边界测试能精确写出 diff = 30。
    """
    gaps = gaps or {}
    t, cur = [], 0.0
    for i in range(n):
        t.append(cur)
        cur += gaps.get(i, 1.0)
    return np.array(t)


# ── 1. 切段 ────────────────────────────────────────────────────────────

def test_empty_and_single_segment():
    assert capture_segments([]) == []
    assert capture_segments(np.array([np.nan])) == []
    assert capture_segments(np.arange(100, dtype=float)) == [(0.0, 99.0)]


def test_a_long_gap_splits_the_capture():
    """一段 60 秒的空档把 100 帧切成两段,且**空档不计入段长**。"""
    t = _ts(100, gaps={50: 61.0})   # 第 50→51 帧之间空 61 秒
    segs = capture_segments(t)
    assert len(segs) == 2, segs
    assert segs[0] == (0.0, 50.0)
    assert segs[1] == (111.0, 159.0)
    # 段长合计 98 秒,而首末差 159 秒 —— 差的那 61 秒正是空档。
    assert sum(b - a for a, b in segs) == pytest.approx(98.0)
    assert segs[-1][1] - segs[0][0] == pytest.approx(159.0)


def test_the_split_threshold_is_strict_greater_than():
    """缺口**恰好**等于 `SEGMENT_GAP_SEC` 时不切,超过才切。

    为什么卡这个边界:13 场单段场的最大相邻间隔是 1~2 秒,拼接场是 26~105 秒,
    两者差一个数量级,所以界在哪不重要 —— 但"恰好等于"必须有个确定的答案,
    否则同一个时间轴换个浮点表示就会得到不同段数。

    红法:把 `np.diff(t) > SEGMENT_GAP_SEC` 改成 `>=` ⟹ 本条红。
    """
    t = _ts(10, gaps={5: SEGMENT_GAP_SEC})
    assert len(capture_segments(t)) == 1
    t = _ts(10, gaps={5: SEGMENT_GAP_SEC + 1})
    assert len(capture_segments(t)) == 2


def test_unsorted_input_is_sorted_first():
    """乱序时间轴必须先排序再切 —— 否则会切出 `起 > 止` 的假段。

    用 `[0, 100, 1, 2]` 而不是温和的 `[0,1,3,2,4]`:后者不排序也恰好得到
    `[(0,4)]`(因为只取首末值),**测不出**少了排序;前者不排序会得到
    `[(0,0), (100,2)]` —— 第二段起 100 止 2,是个物理上不存在的段。

    红法:去掉 `np.sort(...)` ⟹ 本条红。
    """
    t = np.array([0.0, 100.0, 1.0, 2.0])
    segs = capture_segments(t)
    assert all(b >= a for a, b in segs), segs
    assert segs == [(0.0, 2.0), (100.0, 100.0)], segs


# ── 2. 措辞 ────────────────────────────────────────────────────────────

def test_note_is_silent_for_a_single_segment_matching_the_audio():
    """单段且音频与观测等长 ⟹ **不出声**。12/13 场就是这种,不该有噪音。"""
    assert segment_gap_note([(0.0, 400.0)], 400.0) is None
    assert segment_gap_note([], 400.0) is None
    assert segment_gap_note([(0.0, 400.0)], None) is None


def test_note_names_the_segment_count_and_both_windows():
    """拼接场必须**同时**说出两件事:段数/各段合计,以及 V/T 只覆盖最后一段。

    只说"不可比"对读者没有用 —— 他需要知道"哪一族看的是哪一段"。

    红法:删掉 `V/T 两族因此只覆盖最后` 那句 ⟹ 本条红。
    """
    note = segment_gap_note([(0.0, 795.0), (900.0, 1831.0)], 434.7)
    assert note is not None
    assert "2 段录制拼接" in note
    assert "1726 秒" in note           # 795 + 931
    assert "105 秒" in note            # 段间缺口
    assert "最后一段(434.7 秒)" in note
    assert "最后 7.2 分钟" in note
    assert "不可互相印证" in note


def test_single_segment_with_truncated_audio_still_warns():
    """单段但音频明显短于观测(转写只覆盖一部分)也要告警 —— 这是另一类窗口错配。

    红法:把单段分支的 `return` 改成 `return None` ⟹ 本条红。
    """
    note = segment_gap_note([(0.0, 900.0)], 300.0)
    assert note is not None
    assert "F/G 两族覆盖 900 秒" in note
    assert "音频只有 300.0 秒" in note
