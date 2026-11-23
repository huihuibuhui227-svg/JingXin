# tests/test_voice_session_isolation_and_research_series.py
"""语音多会话隔离、科研音频序列识别及 FunASR 降级提示回归测试。"""
from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path

import pytest
from fastapi import HTTPException

import media_retention
from report_frontend import voice_series


def test_voice_series_recognizes_research_answer_audio(tmp_path, monkeypatch):
    sid = "20261123_121000_ef56"
    sdir = tmp_path / sid
    audio_dir = sdir / "media" / "audio"
    audio_dir.mkdir(parents=True)
    wav1 = audio_dir / "0001_converted.wav"
    wav2 = audio_dir / "0002_converted.wav"
    wav1.write_bytes(b"RIFF0000")
    wav2.write_bytes(b"RIFF0000")
    ledger = sdir / f"{media_retention.LEDGER_PREFIX}.audio{media_retention.LEDGER_SUFFIX}"
    ledger.write_text(
        "\n".join([
            json.dumps({"kind": "raw", "seq": 1, "source_endpoint": "/research/answer_audio"}),
            json.dumps({"kind": "raw", "seq": 2, "source_endpoint": "/research/answer_audio"}),
        ]) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    media_retention._reset_for_tests()

    found = voice_series.answer_audio_files(sid)
    assert len(found) == 2
    assert found[0] == (0, wav1)
    assert found[1] == (1, wav2)


def test_voice_sessions_are_isolated_by_session_id(tmp_path, monkeypatch):
    voice_app = importlib.import_module("voice_interaction.api.app")
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.setenv("JX_SERVER_TTS", "0")
    media_retention._reset_for_tests()

    s1 = asyncio.run(voice_app.start_interview())
    s2 = asyncio.run(voice_app.start_interview())
    sid1, sid2 = s1["session_id"], s2["session_id"]
    assert sid1 != sid2
    assert s1["discard_token"] and s2["discard_token"]

    asyncio.run(voice_app.submit_answer(voice_app.AnswerRequest(answer="会话一专属回答内容"), session_id=sid1))
    asyncio.run(voice_app.submit_answer(voice_app.AnswerRequest(answer="会话二专属回答内容"), session_id=sid2))

    sum1 = asyncio.run(voice_app.get_session_summary(sid1, type="interview"))
    sum2 = asyncio.run(voice_app.get_session_summary(sid2, type="interview"))
    assert sum1["data"]["valid_answers"] == ["会话一专属回答内容"]
    assert sum2["data"]["valid_answers"] == ["会话二专属回答内容"]


def test_voice_health_reports_asr_reachability_and_503_on_asr_down(monkeypatch):
    voice_app = importlib.import_module("voice_interaction.api.app")

    monkeypatch.setattr(voice_app, "_tcp_probe", lambda h, p, timeout=1.0: False)
    h_down = asyncio.run(voice_app.health_check())
    assert h_down["status"] == "degraded"
    assert h_down["asr_reachable"] is False

    monkeypatch.setattr(voice_app, "_tcp_probe", lambda h, p, timeout=1.0: True)
    h_up = asyncio.run(voice_app.health_check())
    assert h_up["status"] == "healthy"
    assert h_up["asr_reachable"] is True

    async def _boom(_pcm):
        raise ConnectionRefusedError("FunASR offline")

    monkeypatch.setattr(voice_app, "_transcribe_async", _boom)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(voice_app._transcribe_or_503(b"\x00" * 3200))
    assert exc.value.status_code == 503
    assert "FunASR" in exc.value.detail
