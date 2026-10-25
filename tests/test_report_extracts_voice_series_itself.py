# tests/test_report_extracts_voice_series_itself.py
r"""报告**自己**把语音序列提取出来,而不是让读者去跑一个工具。

## 失效现场(2026-10-26,使用者新录一场之后的报告)

同一场会话 `20261026_002930_cb79`,报告头写着语音已读入,而正文:

    沟通表达流畅度 · 证据不足（0/3）
    语调变化、平均停顿时长、停顿频率 …… 全部没有值

原因是 `data/l0_series/<sid>.npz` 这个派生缓存没生成 —— V 族 12 条里有 8 条
只吃这份序列。报告里那句提示是「跑一次提取就有,不用重录」,**指向一个
读者并不知道怎么跑的工具**。

实测:手动跑一次 `voice_series.extract_session`(34 秒),同一场立刻变成

    覆盖 6 / 14  →  8 / 14
    语音 4/12    →  12/12
    沟通表达流畅度 0/3 → 2/3

⟹ 缺的不是数据,是**那一步没人做**。让读者去做一件系统自己能做的事,
就是把系统的活推给了使用者。

## 与「报告是读侧、不许有副作用」的关系

`tools/jx_backfill_series.py` 的注释写着「不在报告路径上跑」。那条约束的本意是
**报告不许偷偷改写素材**(日志、CSV、录像)。`data/l0_series/*.npz` 是
`voice_series` 自己声明的**派生缓存**:「可随时删、可随时从音频重建」。
生成一个可以随时重建的缓存,不改变任何事实来源。

所以这里的边界是:

* 允许:从**已在盘上的音频**生成派生缓存(纯函数式,输入相同 ⟹ 输出相同);
* 禁止:改日志、改 CSV、改录像、改任何记录事实的东西。

## 这个文件守什么

1. 缓存缺失、音频在盘上 ⟹ 报告路径**当场提取**;
2. 缓存已在 ⟹ **不重算**(一次 34 秒,不能每份报告都付);
3. 没有音频 ⟹ **不试**(那是真的模态缺失,提取一万次也没有);
4. 提取失败 ⟹ 报告照出,缺口如实报(不许因为这一步炸掉整份报告);
5. 两个出口(HTML 与 `/api/report/structured`)共用同一条路。
"""
from __future__ import annotations

import numpy as np
import pytest

import report_frontend.voice_series as voice_series
from report_frontend.l1_report import build_l1_report

_SID = "20261026_002930_cb79"


@pytest.fixture
def calls(monkeypatch):
    """替掉提取与"音频在不在",记下每次调用。"""
    rec = {"extract": [], "force": [], "audio": []}

    def _fake_extract(session_id, force=False):
        rec["extract"].append(session_id)
        rec["force"].append(force)
        return [voice_series.VoiceSeries(
            session_id=session_id, index=None, frame_sec=0.032,
            f0=np.full(100, 120.0), voiced_prob=np.ones(100),
            rms=np.full(100, 0.05), source="camera.webm")]

    monkeypatch.setattr(voice_series, "extract_session", _fake_extract)
    return rec


def _set_audio(monkeypatch, has_audio: bool, calls):
    def _audio_source(session_id):
        calls["audio"].append(session_id)
        return ("whole", [(None, "fake.webm")]) if has_audio else ("none", [])
    monkeypatch.setattr(voice_series, "audio_source", _audio_source)


def test_missing_cache_is_extracted_on_the_report_path(monkeypatch, calls, tmp_path):
    """★ 红法:把 `ensure_series` 里那次 `extract_session` 删掉 ⟹ 这里红。

    这就是使用者那一场的现场:缓存没有、音频在,报告只印"缺"。
    """
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path / "series"))
    _set_audio(monkeypatch, True, calls)
    monkeypatch.setattr(voice_series, "load_series", lambda sid: None)

    got = voice_series.ensure_series(_SID)

    assert calls["extract"] == [_SID], (
        f"音频在盘上、缓存没有,却没提取 —— 报告会照旧说「缺」:{calls}")
    assert got, "提取了却没把序列返回给上层"


def test_an_existing_cache_is_not_recomputed(monkeypatch, calls, tmp_path):
    """★ 红法:把"先读缓存"那一步删掉、每次都提取 ⟹ 这里红。

    单场 34 秒。每份报告都重算一遍,面板点十次就是六分钟。
    """
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path / "series"))
    _set_audio(monkeypatch, True, calls)
    cached = [voice_series.VoiceSeries(
        session_id=_SID, index=None, frame_sec=0.032, f0=np.full(10, 100.0),
        voiced_prob=np.ones(10), rms=np.full(10, 0.1), source="camera.webm")]
    monkeypatch.setattr(voice_series, "load_series", lambda sid: cached)

    got = voice_series.ensure_series(_SID)

    assert got is cached, "有缓存却没返回它"
    assert calls["extract"] == [], f"有缓存仍重算了:{calls}"


def test_no_audio_means_no_attempt(monkeypatch, calls, tmp_path):
    """★ 红法:把 `audio_source` 那一关删掉 ⟹ 这里红。

    没有音频的场次是真的模态缺失,提取一万次也不会出东西 ——
    而且 `extract_session` 在没有音频时会抛/返回空,白白拖慢报告。
    """
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path / "series"))
    _set_audio(monkeypatch, False, calls)
    monkeypatch.setattr(voice_series, "load_series", lambda sid: None)

    got = voice_series.ensure_series(_SID)

    assert got is None
    assert calls["extract"] == [], f"没有音频仍然尝试提取:{calls}"


def test_a_failing_extraction_does_not_break_the_report(monkeypatch, calls, tmp_path):
    """★ 红法:把 `ensure_series` 里的 try/except 删掉 ⟹ 这里红。

    提取会调 ffmpeg/librosa,失败原因很多(音频损坏、依赖缺失)。那不该让
    **整份报告**出不来 —— 缺 8 条序列指标是可报的缺口,没有报告不可报。
    """
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path / "series"))
    _set_audio(monkeypatch, True, calls)
    monkeypatch.setattr(voice_series, "load_series", lambda sid: None)

    def _boom(session_id, force=False):
        raise RuntimeError("ffmpeg 不在 PATH 里")

    monkeypatch.setattr(voice_series, "extract_session", _boom)

    assert voice_series.ensure_series(_SID) is None


def test_the_shared_payload_entry_uses_it(monkeypatch, calls, tmp_path):
    """★ 红法:把 `compute_session_l1` 里的 `ensure_series` 换回 `load_series` ⟹ 红。

    ⚠️ 这一条测的是**生产路径真的调用了它**。本仓记过六次的形态:
    只断言"函数行为对",证明不了生产路径用了它。
    `build_l1_report` 是 HTML 报告与 `/api/report/structured` **共用的**入口 ——
    修在这里,两个出口一起好。
    """
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path / "rec"))
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path / "series"))
    _set_audio(monkeypatch, True, calls)
    monkeypatch.setattr(voice_series, "load_series", lambda sid: None)

    build_l1_report(_SID, logs={})

    assert calls["extract"] == [_SID], (
        f"载荷入口没有走 `ensure_series` —— 报告仍会说「缺」:{calls}")
