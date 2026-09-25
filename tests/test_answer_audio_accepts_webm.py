# tests/test_answer_audio_accepts_webm.py
"""语音**回答**那条路必须收得下前端真正录出来的东西。

**这个文件来自一个真事故**(2026-09-25,使用者第一场真会话):

那场报告 **0 / 20**,语音族整族空。逐层查下来的链条是 ——

1. 前端"语音转文字"(`AssessmentPage.tsx:197`)拿到文字后**把音频 blob 丢掉了**,
   `useAssessment.ts:71` 的 `if (audioFile) submitAudioAnswer(...)` 于是从不触发;
2. 就算它触发了也**会被拒**:`/interview/answer_audio` 原先只认 `RIFF`,
   而浏览器 `MediaRecorder` 录的是 **webm/opus**(`useAudioRecorder.ts:19`);
3. 而语音特征(`log_prosody`)只在 `answer_audio` 里 —— 文字回答那条路一行都不写。

于是"用语音回答"这条**唯一的语音特征来源**,在真实使用里必然是空的。
(`/asr` 早就收 webm 并在内部用 ffmpeg 转 —— 同一件事,`answer_audio` 也要做。)
"""
import asyncio
import importlib
import io
import struct
import subprocess
import wave
from pathlib import Path

import pytest

import media_retention                                                    # noqa: E402
voice_app = importlib.import_module("voice_interaction.api.app")          # noqa: E402


def _wav_bytes(n: int = 3200) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(struct.pack("<%dh" % (n // 2), *([0] * (n // 2))))
    return buf.getvalue()


WEBM = b"\x1a\x45\xdf\xa3" + b"fake-opus-payload" * 20       # EBML 头


def _fake_ffmpeg(monkeypatch, *, ok=True):
    """装一个假 ffmpeg:把输出文件写成合法的 16k 单声道 WAV。"""
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        out = cmd[-1]
        if ok:
            Path(out).write_bytes(_wav_bytes())

        class _R:
            returncode = 0 if ok else 1
            stderr = b"" if ok else b"boom"
        return _R()

    monkeypatch.setattr(subprocess, "run", fake_run)
    return seen


class _FakeUpload:
    def __init__(self, data: bytes, name="a.webm"):
        self._data, self.filename, self.content_type = data, name, "audio/webm"

    async def read(self) -> bytes:
        return self._data


class _FakeRequest:
    async def form(self):
        raise RuntimeError("no multipart")


class _FakeUtt:
    text = "我 觉得 这个 问题 很 有意思"


async def _ret(v):
    return v


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.delenv("JINGXIN_RETAIN_MEDIA", raising=False)
    media_retention._reset_for_tests()
    return tmp_path


@pytest.fixture
def wired(_isolated, monkeypatch):
    logged = {}
    monkeypatch.setattr(voice_app, "_transcribe_async", lambda a: _ret(_FakeUtt()))
    monkeypatch.setattr(voice_app.transcript_store, "append_utterance", lambda *a, **k: None)
    monkeypatch.setattr(voice_app.voice_logger, "log_prosody",
                        lambda *a, **k: logged.update(k))
    monkeypatch.setattr(voice_app.interview_assessment, "add_answer", lambda t: None)
    monkeypatch.setattr(voice_app.interview_assessment, "save_log", lambda: None)
    monkeypatch.setattr(voice_app.interview_assessment, "qa_pairs", [])
    return _isolated, logged


def test_helper_converts_webm_to_16k_wav(monkeypatch):
    """`_to_wav16k_bytes` 把任意容器转成 16k/单声道/s16 WAV 字节。

    红法:去掉这个函数(或让它原样返回输入)—— 下面的端点测试会红在 wave.Error。
    """
    _fake_ffmpeg(monkeypatch)
    out = voice_app._to_wav16k_bytes(WEBM)
    assert out.startswith(b"RIFF")
    with wave.open(io.BytesIO(out), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2


def test_helper_cleans_up_its_temp_files(monkeypatch, tmp_path):
    """临时文件必须删干净 —— 它是**使用者上传的原始音频**,不该留在 /tmp。"""
    _fake_ffmpeg(monkeypatch)
    before = set(Path("/tmp").glob("tmp*"))
    voice_app._to_wav16k_bytes(WEBM)
    after = {p for p in Path("/tmp").glob("tmp*") if p not in before}
    assert not [p for p in after if p.is_file() and p.stat().st_size > 0]


def test_helper_raises_loudly_when_ffmpeg_fails(monkeypatch):
    """ffmpeg 失败要**抛**,不能安静地返回垃圾 —— 否则后面 wave.open 报出的是
    一个与真实原因无关的错(wave.Error: not a WAVE file),排查方向全错。"""
    _fake_ffmpeg(monkeypatch, ok=False)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        voice_app._to_wav16k_bytes(WEBM)


def test_answer_audio_accepts_webm_and_logs_prosody(wired, monkeypatch):
    """**这是那条真事故的直接回归钉子**:前端录的 webm 交给 answer_audio,
    必须 200 并且真的写下一行语音特征。

    红法:把 answer_audio 里那句 `if not contents.startswith(b'RIFF')` 恢复成硬 400。
    """
    root, logged = wired
    _fake_ffmpeg(monkeypatch)
    r = asyncio.run(voice_app.submit_answer_audio(
        request=_FakeRequest(), audio=_FakeUpload(WEBM),
        session_id="20260925_203826_2449"))
    assert r["status"] == "success"
    assert "connective_density" in r
    assert logged, "log_prosody 没被调到 —— 语音特征又不会记了"
    # 原始 webm 与转换后的 wav 都要留下
    names = sorted(p.name for p in (root / "20260925_203826_2449" / "media" / "audio").iterdir())
    assert names == ["0001.webm", "0001_converted.wav"], names


def test_answer_audio_still_accepts_plain_wav(wired, monkeypatch):
    """原本支持的 WAV 不能因为这次改动而坏掉(而且**不该**多跑一次 ffmpeg)。"""
    _fake_ffmpeg(monkeypatch, ok=False)          # 一旦走 ffmpeg 就会抛
    r = asyncio.run(voice_app.submit_answer_audio(
        request=_FakeRequest(), audio=_FakeUpload(_wav_bytes(), name="a.wav"),
        session_id="20260925_120000_wwww"))
    assert r["status"] == "success"
