# tests/test_media_retention_voice.py
"""M2.6:voice 的回答端点把 WAV 原样交给留存。"""
import asyncio
import importlib
import io
import struct
import subprocess
import wave
from pathlib import Path

import pytest

# ⚠️ 不装 python_multipart 替身:真包装着(见 test_media_retention_face.py 的注释)。
import media_retention                                                # noqa: E402
voice_app = importlib.import_module("voice_interaction.api.app")      # noqa: E402


def _wav_bytes(n_bytes: int = 3200) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<%dh" % (n_bytes // 2), *([0] * (n_bytes // 2))))
    return buf.getvalue()


class _FakeUpload:
    def __init__(self, data: bytes, name: str = "a.wav"):
        self._data, self.filename, self.content_type = data, name, "audio/wav"

    async def read(self) -> bytes:
        return self._data


class _FakeRequest:
    async def form(self):
        raise RuntimeError("no multipart")


class _FakeUtt:
    text = "我 觉得 这个 问题 很 有意思"


async def _async_return(v):
    return v


@pytest.fixture
def wired(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.delenv("JINGXIN_RETAIN_MEDIA", raising=False)
    media_retention._reset_for_tests()
    monkeypatch.setattr(voice_app, "_transcribe_async",
                        lambda audio_data: _async_return(_FakeUtt()))
    monkeypatch.setattr(voice_app.transcript_store, "append_utterance",
                        lambda sid, utt, **kw: None)
    monkeypatch.setattr(voice_app, "log_recognition", lambda utt: None)
    monkeypatch.setattr(voice_app.voice_logger, "log_prosody", lambda *a, **kw: None)
    monkeypatch.setattr(voice_app.interview_assessment, "add_answer", lambda t: None)
    monkeypatch.setattr(voice_app.interview_assessment, "save_log", lambda: None)
    monkeypatch.setattr(voice_app.interview_assessment, "qa_pairs", [])
    return tmp_path


def _answer(sid: str, payload: bytes | None = None):
    return asyncio.run(voice_app.submit_answer_audio(
        request=_FakeRequest(), audio=_FakeUpload(payload or _wav_bytes()),
        session_id=sid))


def test_interview_answer_audio_is_retained(wired):
    """红法:删掉端点里的 retain_audio —— 音频从此不留,而特征照样算得出来。"""
    payload = _wav_bytes()
    _answer("20260925_120000_ffff", payload)
    f = wired / "20260925_120000_ffff" / "media" / "audio" / "0001.wav"
    assert f.read_bytes() == payload


def test_two_answers_get_sequence_one_and_two(wired):
    for _ in range(2):
        _answer("20260925_120000_gggg")
    d = wired / "20260925_120000_gggg" / "media" / "audio"
    assert sorted(p.name for p in d.iterdir()) == ["0001.wav", "0002.wav"]


def test_asr_retains_raw_and_converted_under_the_same_seq(wired, monkeypatch):
    """`/asr` 是**唯一**同时产出 raw 与 converted 的端点 —— 两个端点级测试都没有过它。

    (独立审查指出的测试缺口:模块层钉了"共用序号",但模块层的顺序调用恰好是唯一
    不会出错的那种调用;这里走真端点,把 raw 的 seq 传下去才算钉住。)

    红法:把 `seq=(_raw_retained or {}).get("seq")` 去掉 → converted 取"此刻最新值",
    本次请求的 raw 与 converted 仍会配成一对(单请求下看不出差别),
    所以本测试还额外断言**两个文件同名同号**。
    """
    # 造一个非 WAV(webm 头)让 /asr 走 ffmpeg 那条路;ffmpeg 换成"复制输入"
    webm = b"\x1a\x45\xdf\xa3" + b"W" * 200

    def fake_run(cmd, **kw):
        # 造一个**真的** 16k 单声道 WAV —— 端点下面会用 wave.open 校验它
        Path(cmd[-1]).write_bytes(_wav_bytes())

        class _R:
            returncode = 0
            stderr = b""
        return _R()

    monkeypatch.setattr(subprocess, "run", fake_run)   # /asr 里是函数内 import subprocess
    asyncio.run(voice_app.speech_to_text(
        request=_FakeRequest(), audio=_FakeUpload(webm, name="a.webm"),
        session_id="20260925_120000_iiii"))

    d = wired / "20260925_120000_iiii" / "media" / "audio"
    names = sorted(p.name for p in d.iterdir())
    assert names == ["0001.webm", "0001_converted.wav"], names


def test_research_answer_audio_is_retained(wired):
    """科研那条路也要留 —— 它与面试是**对称的两个端点**,漏一个就漏一半素材。

    (M2.1 的教训:第 19 条就是"先面试再科研时被写成上一场的 id",两条路必须一起看。)
    """
    payload = _wav_bytes()
    asyncio.run(voice_app.submit_research_answer_audio(
        request=_FakeRequest(), audio=_FakeUpload(payload),
        session_id="20260925_120000_hhhh"))
    f = wired / "20260925_120000_hhhh" / "media" / "audio" / "0001.wav"
    assert f.read_bytes() == payload
