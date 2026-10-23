# tests/test_m4_l1_voice_gesture.py
"""**V 族(语音)** 与 **G 族(手势)** 的守卫。

## 钉的是什么

V 族:
1. 规格 12 条里**只有 4 条**能按原公式产出,另外 8 条必须是 `not_stored`
   (不是 `uncalibrated`,更不是 0)—— 两种缺口的补救动作完全不同;
2. `speech_rate` 必须**按有声时长加权**合并,不是"各段比值的算术平均";
3. `pause_ratio` 由 `(整段 − 有声) ÷ 有声` 导出,不依赖已丢失的停顿明细。

G 族:
4. `gesture_activity_per_sec` **不再除肩宽** —— jitter 列已经除过,再除一次是重复归一化;
5. ★ `pose_valid_ratio` **不许用 `is_valid`** —— 实测 49 场全部为 False,
   取它的均值会恒为 0,把满帧的躯干数据说成"一帧有效姿态都没有";
6. `hand_visible_ratio` 只能算**下界**(空串 ≠ 0);
7. 两条率量(2026-10-18 起可出)的**帧率必须从时间戳读**,读不到就报 `no_samples`,
   **不许默认成 1.0** —— 写死帧率会让"2 秒一段"在换帧率时悄悄变成"2 帧一段";
8. 最短段长按**秒**定义(2 s):1 Hz 下 1 帧 = 1 s,单帧超基线不算一段。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from report_frontend import l1_gesture, l1_voice
from report_frontend.event_windows import AnswerWindow, WindowSet
from report_frontend.l1_indicators import GAP_NOT_STORED, GAP_NO_SAMPLES, GAP_UNCALIBRATED


def _ws(*bounds):
    return WindowSet("s", "answer",
                     [AnswerWindow(index=i, t_start=a, t_end=b)
                      for i, (a, b) in enumerate(bounds)],
                     n_questions=len(bounds))


# ── V 族 ─────────────────────────────────────────────────────────────────

def test_voice_reports_exactly_four_values_and_eight_not_stored():
    """★ 粒度不够的 8 条必须报 `not_stored`,且值必须是 NaN。

    红法:给 `pitch_median` 改成用 `pitch_mean` 顶替 ⟹ 它会出值,本条红
    (名字一样、含义不同,比缺着更坏)。
    """
    voice = pd.DataFrame({
        "n_chars": [30.0, 40.0, 50.0], "speech_duration_sec": [6.0, 8.0, 10.0],
        "duration_sec": [9.0, 12.0, 15.0], "pause_frequency": [20.0, 30.0, 40.0],
        "reaction_time": [5.0, 8.0, 11.0],
        "pitch_mean": [100.0, 110.0, 120.0], "pitch_p10": [80.0, 85.0, 90.0],
        "pitch_p90": [120.0, 130.0, 140.0], "energy_mean": [0.05, 0.06, 0.07],
    })
    inds = l1_voice.compute_voice_indicators(voice)
    assert len(inds) == 12

    measured = [i for i in inds if i.has_value]
    assert sorted(i.name for i in measured) == [
        "pause_rate_per_min", "pause_ratio", "response_latency_sec", "speech_rate"]

    gaps = {i.name: i.gap for i in inds if not i.has_value}
    assert len(gaps) == 8
    for name, gap in gaps.items():
        assert gap == GAP_NOT_STORED, (name, gap)
    assert "pitch_median" in gaps and "voiced_ratio" in gaps


def test_speech_rate_is_weighted_by_voiced_seconds_not_a_mean_of_ratios():
    """★ 一段 3 秒答 30 字(10 字/秒)、一段 30 秒答 30 字(1 字/秒)。

    加权 = 60 字 ÷ 33 秒 ≈ **1.82**;若按"比值的算术平均" = (10+1)/2 = **5.5**。
    短答的噪声会把答案抬高三倍 —— 这正是"每段等权"的老毛病。

    红法:把 `n_chars.sum()/speech.sum()` 换成 `(n_chars/speech).mean()` ⟹ 本条红。
    """
    voice = pd.DataFrame({"n_chars": [30.0, 30.0],
                          "speech_duration_sec": [3.0, 30.0],
                          "duration_sec": [4.0, 32.0]})
    inds = {i.name: i for i in l1_voice.compute_voice_indicators(voice)}
    sr = inds["speech_rate"]
    assert sr.value == pytest.approx(60.0 / 33.0, rel=1e-9)
    assert sr.value < 2.0, "加权后应贴近长答的 1 字/秒,不是 5.5"


def test_pause_ratio_comes_from_duration_minus_speech():
    """停顿总时长 = 整段 − 有声;比值 = 停顿 ÷ 有声。

    两段:9−6=3 与 12−8=4 ⟹ 7 ÷ 14 = 0.5。红法:分母改成整段时长 ⟹ 7/21≈0.33,本条红。
    """
    voice = pd.DataFrame({"duration_sec": [9.0, 12.0],
                          "speech_duration_sec": [6.0, 8.0],
                          "n_chars": [30.0, 40.0], "pause_frequency": [10.0, 10.0]})
    inds = {i.name: i for i in l1_voice.compute_voice_indicators(voice)}
    assert inds["pause_ratio"].value == pytest.approx(7.0 / 14.0)


def test_voice_missing_reports_modality_missing_for_all_twelve():
    inds = l1_voice.compute_voice_indicators(None)
    assert len(inds) == 12
    assert all(i.gap == "modality_missing" for i in inds)


# ── G 族 ─────────────────────────────────────────────────────────────────

def _gesture_df(n=40, **over):
    base = {
        "left_wrist_jitter": np.full(n, 0.01), "right_wrist_jitter": np.full(n, 0.01),
        "left_elbow_jitter": np.zeros(n), "right_elbow_jitter": np.zeros(n),
        "left_shoulder_jitter": np.full(n, 0.001), "right_shoulder_jitter": np.full(n, 0.001),
        "torso_jitter": np.full(n, 0.002), "shoulder_width": np.full(n, 0.4),
        "is_valid": np.zeros(n, dtype=bool),
        "hand_visible_left": [np.nan] * n, "hand_visible_right": [np.nan] * n,
    }
    base.update(over)
    return pd.DataFrame(base)


def test_activity_sums_jitter_columns_without_dividing_by_shoulder_width_again():
    """★ jitter 列已经"÷秒、÷肩宽"。四列各 0.01+0.01+0.001+0.001 = 0.022。

    若再除一次肩宽(0.4)会得 0.055 —— 相差 2.5 倍,且把取景远近的影响平方了。

    红法:在 `total` 上再除 `shoulder_width` ⟹ 本条红。
    """
    df = _gesture_df()
    t = pd.Series(np.arange(len(df), dtype=float))
    inds = {i.name: i for i in l1_gesture.compute_gesture_indicators(
        df, _ws((-1.0, 999.0)), t)}
    assert inds["gesture_activity_per_sec"].value == pytest.approx(0.022, rel=1e-9)


def test_pose_valid_ratio_ignores_the_stuck_is_valid_flag():
    """★★ 最重要的一条:实测 49 场 `is_valid` **全部为 False**。

    本用例构造 `is_valid` 全 False、而躯干/肩/肩宽四列全有值的日志:
    正确的 `pose_valid_ratio` 是 **1.0**(数据真的在),
    若取 `is_valid` 均值会得 **0.0** —— 把满帧数据说成"一帧有效姿态都没有"。

    红法:把 `ok` 改成从 `is_valid` 算 ⟹ 得 0.0,本条红。
    """
    df = _gesture_df()
    assert df["is_valid"].sum() == 0, "前提:标志位全 False"
    t = pd.Series(np.arange(len(df), dtype=float))
    inds = {i.name: i for i in l1_gesture.compute_gesture_indicators(
        df, _ws((-1.0, 999.0)), t)}
    assert inds["pose_valid_ratio"].value == pytest.approx(1.0)


def test_hand_visible_ratio_is_a_lower_bound_because_empty_is_not_zero():
    """★ 写入方明确"没有依据时写空串,永不写 0" ⟹ 空串只能当"不知道"。

    40 帧里 10 帧写了 1 ⟹ 下界 0.25。若把空串当 0 也仍是 0.25 ——
    但若把 `1` 之外的一切(含空串)**都当 0**,那与"下界"的读法一致;
    差别在于**不许**把它当成"可见率"去解释。这里钉住数值与公式里的"下界"字样。
    """
    n = 40
    left = [np.nan] * n
    for i in range(10):
        left[i] = 1.0
    df = _gesture_df(n=n, hand_visible_left=left)
    t = pd.Series(np.arange(n, dtype=float))
    inds = {i.name: i for i in l1_gesture.compute_gesture_indicators(
        df, _ws((-1.0, 999.0)), t)}
    hv = inds["hand_visible_ratio_l"]
    assert hv.value == pytest.approx(0.25)
    assert "下界" in hv.formula


def test_rates_without_a_timestamp_column_are_no_samples_not_a_guessed_fps():
    """★ 读不到帧间隔 ⟹ `no_samples`,**不许默认成 1.0**。

    帧间隔是"2 秒一段"换算成帧数的唯一依据。写死 1.0 的话,将来换成 15 fps 采集,
    同一条定义会悄悄变成"2 帧 = 0.13 秒一段",而报告上什么都不会变。

    红法:把 `_frame_sec` 的 `return None` 改成 `return 1.0` ⟹ 本条红。
    """
    df = _gesture_df()          # 这个 fixture 故意没有 timestamp 列
    t = pd.Series(np.arange(len(df), dtype=float))
    inds = {i.name: i for i in l1_gesture.compute_gesture_indicators(
        df, _ws((-1.0, 999.0)), t)}
    for name in ("gesture_burst_rate_per_min_vs_baseline",
                 "postural_shift_rate_per_min_vs_baseline"):
        assert inds[name].gap == GAP_NO_SAMPLES, name
        assert not inds[name].has_value


def _gesture_df_with_clock(n=60, spikes=(), torso_spikes=(), **over):
    """1 Hz 的手势帧:0.10~0.20 的基线噪声 + 指定帧位置抬到 1.0。

    ★ `spikes` 只抬**活动列**(腕/肘/肩),`torso_spikes` 只抬 `torso_jitter`。
    两者**刻意分开**:原夹具让它们同帧一起抬,于是"运动段"与"位移段"必然同段数,
    `burst` 就算量错了信号(拿姿态列当活动列)也测不出来 —— 实测过,9/13 场的段数是错的。
    """
    rng = np.random.RandomState(3)
    act = {c: rng.uniform(0.10, 0.20, n) for c in l1_gesture.ACTIVITY_COLS}
    tor = rng.uniform(0.10, 0.20, n)
    for i in spikes:
        for c in l1_gesture.ACTIVITY_COLS:
            act[c][i] = 1.0
    for i in torso_spikes:
        tor[i] = 1.0
    df = _gesture_df(n=n, torso_jitter=tor, timestamp=np.arange(n, dtype=float), **act)
    return df


def test_burst_needs_two_consecutive_frames_because_one_frame_is_one_second():
    """★ 最短段长 2 s = 1 Hz 下 2 帧;单帧尖峰(1 s)**不算一段**。

    `spikes=(10,)` 是一个孤立单帧 ⟹ 0 段 ⟹ 率 0.0;
    `spikes=(10, 11)` 是连续两帧 ⟹ 1 段 ⟹ 率 = 1 ÷ (60/60) = 1.0 次/分。

    红法:把 `MIN_EVENT_SEC` 改成 1.0 ⟹ 单帧也算一段,前半条红。
    """
    def rate(idx):
        df = _gesture_df_with_clock(spikes=idx)
        t = pd.Series(np.arange(len(df), dtype=float))
        inds = {i.name: i for i in l1_gesture.compute_gesture_indicators(
            df, _ws((-1.0, 999.0)), t)}
        return inds["gesture_burst_rate_per_min_vs_baseline"]

    one = rate((10,))
    assert one.has_value and one.value == pytest.approx(0.0), one.value

    two = rate((10, 11))
    assert two.has_value and two.value == pytest.approx(1.0), two.value


def test_burst_rate_divides_by_the_observed_minutes():
    """分母是**本窗口的观测时长**(60 帧 × 1 s = 1 分钟),不是帧数。

    两个互不相邻的连续段 ⟹ 2 段 ÷ 1 分钟 = 2.0 次/分。

    红法:把 `episodes / duration_min` 换成 `episodes / n_total` ⟹ 得 0.033,本条红。
    """
    df = _gesture_df_with_clock(spikes=(10, 11, 40, 41))
    t = pd.Series(np.arange(len(df), dtype=float))
    inds = {i.name: i for i in l1_gesture.compute_gesture_indicators(
        df, _ws((-1.0, 999.0)), t)}
    ind = inds["gesture_burst_rate_per_min_vs_baseline"]
    assert ind.value == pytest.approx(2.0)


def test_zero_episodes_is_a_measured_zero_not_a_missing_value():
    """★ 一次都没超基线 ⟹ **有值的 0.0**,不是 `no_samples`。

    "这场他一次都没超过自己的基线"是一条真实的观测结论;报成缺失会让人以为
    手势没测到。`n_valid` 报观测帧数(60),所以置信不是「无」。

    (试过把 `n_valid` 报成事件数 —— 0 段时 `n_valid=0` 且 value=0.0,
    直接违反 `L1Indicator` 的不变量 "confidence='无' 却有值"。)

    红法:在 `episodes == 0` 时返回 `_un(name, GAP_NO_SAMPLES)` ⟹ 本条红。
    """
    df = _gesture_df_with_clock(spikes=())
    t = pd.Series(np.arange(len(df), dtype=float))
    inds = {i.name: i for i in l1_gesture.compute_gesture_indicators(
        df, _ws((-1.0, 999.0)), t)}
    ind = inds["gesture_burst_rate_per_min_vs_baseline"]
    assert ind.has_value and ind.value == pytest.approx(0.0)
    # `n_valid` 报**段数**(0 段取 1,表示"看完了一次观测"),`n_total` 报帧数。
    assert ind.n_valid == 1 and ind.n_total == 60
    assert ind.confidence == "低"


def test_gesture_missing_reports_modality_missing():
    inds = l1_gesture.compute_gesture_indicators(None, _ws((-1.0, 10.0)), None)
    assert len(inds) == 7
    assert all(i.gap == "modality_missing" for i in inds)


# ── 两条率量必须各用各的信号 ────────────────────────────────────────────

def test_burst_uses_activity_columns_and_shift_uses_torso():
    """★ 只让**活动列**尖峰 ⟹ `burst` 有段、`shift` 为 0;反过来也一样。

    ## 这条测试的由来(一级 bug)

    `_for_window` 里 G1 段用 `any_val`/`total` 装活动六列的和,**G2 段用同样的
    变量名**装姿态列的和 —— 于是 G2 把 G1 的结果覆盖了。后面算"运动段"时
    `act_sig = np.where(any_val, total, nan)` 拿到的是**姿态信号**,
    而 `required_l0` 对外仍报活动六列。实测 9/13 场的段数是错的(最多差 3 倍),
    而且不会报错、不会告警。

    旧夹具让活动列与躯干**同帧一起**尖峰,两种口径必然同段数 ⟹ 这个 bug 测不出来。

    红法:把 G2 的 `pos_any`/`pos_tot` 改回 `any_val`/`total`(即恢复覆盖)⟹ 本条红。
    """
    def two(**kw):
        df = _gesture_df_with_clock(**kw)
        t = pd.Series(np.arange(len(df), dtype=float))
        return {i.name: i for i in l1_gesture.compute_gesture_indicators(
            df, _ws((-1.0, 999.0)), t)}

    only_act = two(spikes=(10, 11))
    assert only_act["gesture_burst_rate_per_min_vs_baseline"].value > 0
    assert only_act["postural_shift_rate_per_min_vs_baseline"].value == pytest.approx(0.0)

    only_torso = two(torso_spikes=(20, 21))
    assert only_torso["postural_shift_rate_per_min_vs_baseline"].value > 0
    assert only_torso["gesture_burst_rate_per_min_vs_baseline"].value == pytest.approx(0.0)


def test_frame_interval_survives_out_of_order_timestamps():
    """★ 时间戳乱序时帧间隔仍是 1.0,不是 2.0。

    踩过的坑:直接差分后丢掉负值,`[0,1,3,2,4]` 得到 2.0 ⟹ `min_frames` 由 2 变 1、
    时长翻倍、率减半。修法是**先排序再差分**。

    红法:把 `np.diff(np.sort(t))` 改回 `np.diff(t)` ⟹ 本条红。
    """
    df = pd.DataFrame({"timestamp": [0.0, 1.0, 3.0, 2.0, 4.0]})
    assert l1_gesture._frame_sec(df) == pytest.approx(1.0)
