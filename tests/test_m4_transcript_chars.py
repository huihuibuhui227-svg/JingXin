# tests/test_m4_transcript_chars.py
"""M4.15:`speech_rate` 的 `n_chars` 从批量转写里取。

## 为什么会有这一轮

正式语料 24/24 场的**语音日志是 0 行**,`speech_rate` 的 `n_chars` 来自语音日志的
文本列 ⟹ 一度被判成"不可出"。但 `tools/jx_transcribe_recordings.py` 早就把整场录音
跑过 ASR,产出落在 `data/transcripts/<sid>.json`,实测 **13/13 场 A 档全都有**
(`n_chars` 320~3899)。⟹ 这条指标不是缺数据,是**我先前只在一个目录里找**。

## 本文件钉住的三件事

1. **读不出 ≠ 0 字。** 没有转写就报缺,不许拿 0 顶替 —— 0 字是"一场没人说话"。
2. **公式必须写明"全场混音"。** 音频没有说话人分离,字数是面试官 + 考生一起数的,
   量的是"这一场说话有多密",不是"考生语速"。不写清楚就是无声地改构念。
3. **置信度恒为低。** 整场只有一个合并值(`n_valid=1`),低于 `speech` 族门槛 2。
"""

import json
from pathlib import Path

import numpy as np
import pytest

from report_frontend import l1_voice, transcript_chars
from report_frontend.l1_indicators import GAP_NOT_STORED, GAP_NO_SAMPLES
from report_frontend.voice_series import VoiceSeries

FRAME_SEC = 0.032


def _series(voiced_frames=100, total_frames=200, index=None):
    f0 = np.full(total_frames, np.nan)
    f0[:voiced_frames] = 200.0
    return VoiceSeries(session_id="S", index=index, frame_sec=FRAME_SEC, f0=f0,
                       voiced_prob=np.ones(total_frames), rms=np.full(total_frames, 0.1))


@pytest.fixture
def store(tmp_path, monkeypatch):
    """把转写缓存指到 tmp,并按需写一份。"""
    monkeypatch.setenv("JINGXIN_TRANSCRIPTS_DIR", str(tmp_path))

    def _write(sid, n_chars, **extra):
        (tmp_path / f"{sid}.json").write_text(
            json.dumps({"session_id": sid, "n_chars": n_chars, **extra}),
            encoding="utf-8")

    return _write


def _speech_rate(inds):
    return {i.name: i for i in inds}["speech_rate"]


# ── 读取层 ────────────────────────────────────────────────────────────────

def test_missing_transcript_is_none_not_zero(store):
    """★ 没有转写 ⟹ `None`。**不是 0**。

    红法:把 `n_chars()` 的 `return None` 改成 `return 0` ⟹ 本条红。
    0 会让下游算出 `speech_rate = 0.00 字/秒`,一个看着像测量、其实是
    "这场没人说话"的数 —— 而真相是"没转写过"。
    """
    assert transcript_chars.n_chars("nope") is None
    assert transcript_chars.meta("nope") is None


def test_zero_chars_is_treated_as_no_transcript(store):
    """★ `n_chars = 0` 视同没有可用转写。

    红法:去掉 `if v <= 0: return None` 那行 ⟹ 本条红。
    """
    store("S", 0)
    assert transcript_chars.n_chars("S") is None


def test_chars_are_read_from_the_store(store):
    store("S", 1234, audio_seconds=99.5)
    assert transcript_chars.n_chars("S") == 1234
    assert transcript_chars.audio_seconds("S") == 99.5


def test_broken_json_is_none_not_an_exception(store, tmp_path):
    (tmp_path / "S.json").write_text("{not json", encoding="utf-8")
    assert transcript_chars.n_chars("S") is None


# ── 指标层 ────────────────────────────────────────────────────────────────

def test_speech_rate_is_chars_over_voiced_seconds(store):
    """`speech_rate` = 字数 ÷ **有声秒**(不是总时长)。

    100 个浊音帧 × 0.032 s = 3.2 s;32 字 ÷ 3.2 s = 10.0 字/秒。

    红法:把 `voiced` 换成 `s.duration_sec`(200 帧 = 6.4 s)⟹ 得 5.0,本条红。
    """
    store("S", 32)
    ind = _speech_rate(l1_voice.compute_voice_indicators(
        None, series=[_series()], session_id="S"))
    assert ind.has_value
    assert ind.value == pytest.approx(10.0)


def test_speech_rate_formula_admits_the_audio_is_mixed(store):
    """★ 公式必须写明"全场混音 / 含面试官 / 不是考生语速"。

    红法:把 `_TRANSCRIPT_FORMULA` 换回规格原话 `"字数 ÷ 语音秒数"` ⟹ 本条红。
    没有说话人分离,这个数就**不是**考生语速;不写清楚,读者会当考生语速用。
    """
    store("S", 32)
    ind = _speech_rate(l1_voice.compute_voice_indicators(
        None, series=[_series()], session_id="S"))
    assert "全场混音" in ind.formula
    assert "含面试官" in ind.formula
    assert "不是考生语速" in ind.formula


def test_speech_rate_confidence_is_low_because_it_is_one_measurement(store):
    """★ 整场只有一个合并值 ⟹ `n_valid=1` < 门槛 2 ⟹ 置信「低」。

    红法:把 `n_valid=1, n_total=1` 改成 `n_valid=2, n_total=2` ⟹ 本条红。
    那是把"一次测量"谎报成"两次",置信度会跳到「中」。
    """
    store("S", 32)
    ind = _speech_rate(l1_voice.compute_voice_indicators(
        None, series=[_series()], session_id="S"))
    assert (ind.n_valid, ind.n_total) == (1, 1)
    assert ind.confidence == "低"


def test_no_transcript_leaves_speech_rate_unstored(store):
    """没有转写 ⟹ `not_stored`,且**没有值**。

    红法:把 `_un(name, GAP_NOT_STORED)` 改成返回 `measured(..., value=0.0)` ⟹ 本条红。
    """
    ind = _speech_rate(l1_voice.compute_voice_indicators(
        None, series=[_series()], session_id="S"))
    assert not ind.has_value
    assert ind.gap == GAP_NOT_STORED


# ⚠️ 已移一条:`test_without_session_id_speech_rate_stays_unstored`。
#   它钉的是 `_speech_rate_from_transcript` 里那句 `if session_id is None: return _un(...)`。
#   变异验证时**删掉那句,测试照样绿** —— 因为 `transcript_chars.n_chars(None)`
#   本来就走 `data/transcripts/None.json`(不存在)返回 `None`,结果完全一样。
#   ⟹ 那句是死代码,已连测试一起删(本仓有 `test_no_assert_is_dead` 守着这条规矩)。
#   "调用方没给 session_id" 这个场景本身仍被
#   `test_no_transcript_leaves_speech_rate_unstored` 覆盖到(同样是 not_stored)。


def test_no_voiced_frames_means_no_samples(store):
    """有转写但没有浊音帧 ⟹ `no_samples`,不是 0 字/秒(那是除零)。

    红法:去掉 `if not voiced > 0` 那行 ⟹ 除零 `ZeroDivisionError`,本条红。
    """
    store("S", 32)
    ind = _speech_rate(l1_voice.compute_voice_indicators(
        None, series=[_series(voiced_frames=0)], session_id="S"))
    assert not ind.has_value
    assert ind.gap == GAP_NO_SAMPLES


def test_speech_rate_sums_voiced_seconds_across_segments(store):
    """多段序列要**先合并秒数再除**,不是各段比值的均值。

    两段各 100 浊音帧 = 6.4 s;64 字 ÷ 6.4 s = 10.0。

    红法:把 `sum(...)` 换成 `series[0].voiced_sec` ⟹ 得 20.0,本条红。
    """
    store("S", 64)
    ind = _speech_rate(l1_voice.compute_voice_indicators(
        None, series=[_series(), _series()], session_id="S"))
    assert ind.value == pytest.approx(10.0)
