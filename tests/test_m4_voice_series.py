# tests/test_m4_voice_series.py
"""**语音序列重建**与**序列派生的 8 条 V 指标**的守卫。

## 钉的是什么

1. 停顿区间**含首尾静音**,且只保留 ≥ `MIN_PAUSE_SEC` 的(定义,不是标定);
2. `voiced_ratio` **跨回答汇总**,不是"每题比值的平均";
3. `pitch_range_semitone` 严格按规格的**首末差**,不偷换成分位数;
4. `*_per_sec` 的分母是**浊音秒**,不是整段秒;
5. 没有序列 ⟹ 8 条全部 `not_stored`,**不猜**;
6. 缓存能来回读,读不出时返回 `None` 而不是炸掉。
"""

from __future__ import annotations

import numpy as np
import pytest

from report_frontend import l1_voice, l1_voice_series as lvs, voice_series as vs
from report_frontend.l1_indicators import GAP_NOT_STORED


def _series(index=0, pattern="", frame_sec=0.1, f0_hz=100.0, rms=0.05, sid="s"):
    """按字符串造序列:`V` 浊音、`.` 静音。"""
    voiced = np.array([c == "V" for c in pattern])
    f0 = np.where(voiced, f0_hz, np.nan).astype(float)
    return vs.VoiceSeries(session_id=sid, index=index, frame_sec=frame_sec,
                          f0=f0, voiced_prob=np.where(voiced, 0.9, 0.01),
                          rms=np.full(len(pattern), rms), source="synthetic")


# ── 1. 停顿区间的定义 ────────────────────────────────────────────────────

def test_pause_runs_count_leading_and_trailing_silence():
    """`...VVV....`(帧 0.1 s)⟹ 前导 0.3 s、尾部 0.4 s 都要算。

    红法:去掉"收尾"那一支 ⟹ 尾部 0.4 s 丢掉,本条红。
    """
    s = _series(pattern="...VVV....", frame_sec=0.1)
    runs = lvs.pause_runs(s.voiced_flag, s.frame_sec, 0.2)
    assert runs == pytest.approx([0.3, 0.4])


def test_pause_runs_drop_gaps_shorter_than_the_minimum():
    """`.V.V` 的间隙只有 0.1 s < 0.2 s ⟹ 不算停顿。

    红法:把 `>= min_pause_sec` 改成 `> 0` ⟹ 会数出 2 段,本条红。
    """
    s = _series(pattern=".V.V.V", frame_sec=0.1)
    assert lvs.pause_runs(s.voiced_flag, s.frame_sec, 0.2) == []


# ── 2. voiced_ratio 要汇总,不是平均 ─────────────────────────────────────

def test_voiced_ratio_pools_frames_across_answers():
    """答 A:10 帧全浊音;答 B:90 帧全静音。

    汇总 = 10/100 = **0.1**;若"每题比值再平均" = (1.0+0.0)/2 = **0.5**。
    红法:改成 `np.mean([s.voiced_flag.mean() for s in series])` ⟹ 本条红。
    """
    a = _series(index=0, pattern="V" * 10)
    b = _series(index=1, pattern="." * 90)
    got = lvs.compute_from_series([a, b], {"speech": 2}, n_total=2)
    assert got["voiced_ratio"].value == pytest.approx(0.1)


# ── 3. pitch_range_semitone 必须是首末差 ────────────────────────────────

def test_pitch_range_uses_first_and_last_voiced_frame_in_semitones():
    """首帧 200 Hz、末帧 100 Hz ⟹ 12·log2(200/100) = **12 半音**(一个八度)。

    红法:把首末改成 p10/p90 ⟹ 数值不同,本条红。
    """
    f0 = np.array([200.0, 150.0, 120.0, 100.0])
    s = vs.VoiceSeries("s", 0, 0.1, f0, np.full(4, 0.9), np.full(4, 0.05))
    got = lvs.compute_from_series([s], {"pitch": 5}, n_total=1)
    assert got["pitch_range_semitone"].value == pytest.approx(12.0, abs=1e-9)


# ── 4. 率量的分母是浊音秒 ────────────────────────────────────────────────

def test_energy_dynamics_divides_by_voiced_seconds_not_total():
    """rms 从 0.0 线性升到 0.3,共 11 帧(1.0 s),其中只有 5 帧浊音(0.5 s)。

    Σ|Δrms| = 0.3(10 段差、每段 0.03)。分母取浊音秒 0.5 ⟹ **0.6**;
    取整段 1.0 ⟹ 0.3。两者差一倍,正是"分母用错"的典型症状。

    红法:把 `s.voiced_sec` 换成 `s.duration_sec` ⟹ 得 0.3,本条红。
    """
    n = 11
    rms = np.linspace(0.0, 0.3, n)
    f0 = np.full(n, np.nan)
    f0[:5] = 100.0                      # 前 5 帧浊音 = 0.5 s
    s = vs.VoiceSeries("s", 0, 0.1, f0, np.full(n, 0.9), rms)
    got = lvs.compute_from_series([s], {"energy": 5}, n_total=1)
    assert got["energy_dynamics_per_sec"].value == pytest.approx(0.6, rel=1e-9)


def test_pitch_variability_also_divides_by_voiced_seconds():
    """★ 音高那条的分母**同样**必须是浊音秒 —— 只测能量那条会漏掉它(踩过)。

    10 帧 × 0.1 s = 1.0 s,其中 5 帧浊音 = 0.5 s。
    f0 在 100/200 Hz 之间交替 ⟹ 4 段差、每段 12 半音,Σ = 48。
    ÷ 浊音秒 0.5 ⟹ **96**;÷ 整段 1.0 ⟹ 48。

    红法:把 `sec = s.voiced_sec` 改成 `s.duration_sec` ⟹ 得 48,本条红。
    """
    n = 10
    f0 = np.array([100.0, 200.0, 100.0, 200.0, 100.0, np.nan, np.nan, np.nan, np.nan, np.nan])
    s = vs.VoiceSeries("s", 0, 0.1, f0, np.full(n, 0.9), np.full(n, 0.05))
    got = lvs.compute_from_series([s], {"pitch": 5}, n_total=1)
    assert got["pitch_variability_per_sec"].value == pytest.approx(96.0, rel=1e-9)


# ── 5. 没有序列 ⟹ 全部 not_stored ───────────────────────────────────────

def test_without_series_all_eight_are_not_stored():
    """红法:让空序列走进"有序列"分支并补 0 ⟹ 会出现 `value=0.0`,本条红。"""
    got = lvs.compute_from_series(None, {}, n_total=8)
    assert set(got) == set(lvs.SERIES_NAMES)
    for name, ind in got.items():
        assert ind.gap == GAP_NOT_STORED, (name, ind.gap)
        assert not ind.has_value


def test_empty_audio_yields_not_stored_not_zero():
    """空音频(0 帧)⟹ 也是 `not_stored`,不许算出"完全静音"的读数。"""
    empty = vs.VoiceSeries("s", 0, 0.0, np.array([]), np.array([]), np.array([]))
    got = lvs.compute_from_series([empty], {}, n_total=1)
    assert all(not i.has_value for i in got.values())


# ── 6. 缓存往返 ─────────────────────────────────────────────────────────

def test_cache_roundtrip_and_corrupt_cache_returns_none(tmp_path, monkeypatch):
    """缓存读不出时返回 `None`(派生品损坏不该让上层炸掉)。"""
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path))
    s = _series(pattern="VVV...VVV", frame_sec=0.05)
    vs.save_series([s])
    back = vs.load_series("s")
    assert back is not None and len(back) == 1
    assert np.allclose(back[0].f0, s.f0, equal_nan=True)

    vs._cache_path("s").write_bytes(b"not a npz")
    assert vs.load_series("s") is None


# ── 7. 端到端:12 条全部可产 ────────────────────────────────────────────

def test_all_twelve_voice_indicators_produce_values_when_series_present():
    """★ 给了日志 + 序列 ⟹ 12 条**全部**出值,一条 not_stored 都不该剩。

    红法:在 `compute_voice_indicators` 里不把 `series` 传下去 ⟹ 8 条退回
    `not_stored`,本条红。
    """
    import pandas as pd
    voice = pd.DataFrame({
        "n_chars": [30.0] * 8, "speech_duration_sec": [6.0] * 8,
        "duration_sec": [9.0] * 8, "pause_frequency": [20.0] * 8,
        "reaction_time": [5.0] * 8,
    })
    series = [_series(index=i, pattern="VVV..VVV..VVVV") for i in range(8)]
    inds = l1_voice.compute_voice_indicators(voice, series=series)
    assert len(inds) == 12
    missing = [i.name for i in inds if not i.has_value]
    assert missing == [], f"仍有未出值:{missing}"


# ── 7. `pause_rate_per_min`(2026-10-18 新增:从序列侧补出来)─────────────
#
# 背景:这一条原属"日志侧 4 条",正式场语音日志 0 行 ⟹ 一直报 modality_missing。
# 但它的两个输入(停顿区间、有声秒)都只在帧序列里,序列侧本来就算得出来。

def test_pause_rate_divides_by_voiced_minutes_not_total_minutes():
    """`pause_rate_per_min` 的分母是**有声**分钟,不是整段分钟。

    `..VVVVVVVV..`:16 帧 × 0.1 s = 1.6 s,有声 8 帧 = 0.8 s;
    停顿区间 2 个(各 0.2 s)⟹ 2 ÷ (0.8/60) = 150 次/分。

    红法:分母换成 `duration_sec`(1.6 s)⟹ 得 75,本条红。
    """
    s = _series(pattern="..VVVVVVVV..", frame_sec=0.1)
    got = lvs.compute_from_series([s])
    assert got["pause_rate_per_min"].value == pytest.approx(150.0)


def test_pause_rate_counts_only_pauses_at_least_min_pause_sec():
    """短于 `MIN_PAUSE_SEC`(0.2 s)的抖动**不算停顿**。

    `.V.VV.` 帧 0.1 s:三个静音段各只有 0.1 s ⟹ 停顿次数 0 ⟹ 率为 0.0。

    红法:把 `MIN_PAUSE_SEC` 传成 0 ⟹ 三个 0.1 s 都算停顿,率变正,本条红。
    """
    s = _series(pattern=".V.VV.", frame_sec=0.1)
    got = lvs.compute_from_series([s])
    assert got["pause_rate_per_min"].value == pytest.approx(0.0)


def test_pause_ratio_is_deliberately_not_produced():
    """★ `pause_ratio` **故意不产出** —— 它与 `voiced_ratio` 是恒等变换。

    实测 13 场 A 档:`pause_ratio` vs `1/voiced_ratio − 1` 的 **r = 0.9990**;
    按 ≥0.2s 口径算也还是 0.999(平均差 0.0585)。两条路都是 `voiced_ratio`
    的单调变换 ⟹ 零额外信息。本仓对这种东西是标出来,不是留着。

    红法:在 `compute_from_series` 里把 `pause_ratio` 加回去 ⟹ 本条红。
    """
    s = _series(pattern="..VVVVVVVV..", frame_sec=0.1)
    got = lvs.compute_from_series([s])
    assert "pause_ratio" not in got
    assert "pause_ratio" not in lvs.SERIES_NAMES
