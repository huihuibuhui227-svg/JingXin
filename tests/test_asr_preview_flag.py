# tests/test_asr_preview_flag.py
"""`/asr` 的一声识别,是**预览**还是**回答**?—— 两者都会写会话账本,是 2026-09-26 的缺陷。

实测(场 `20260926_124439_0f5b`,真浏览器真会话):前端录音后先调一次 `/asr`
把识别文本显示给面试官看,点提交时**同一份音频**(逐字节相同,sha256 8 对全等)
再走一次 `/interview/answer_audio`。两个端点在服务端**各自**都
`transcript_store.append_utterance` 一次 ⟹ `transcript.json` 每句话存两遍
(该场 24 段 = 12 段 × 2,连时间戳都一模一样)。

为什么必须修:M3 要用 transcript 的逐字时间戳算语速/停顿/反应潜伏期 ——
每句两遍会让这些量**整体翻倍**,而且是静默的(账本看不出哪里不对)。

修法(`record` 开关,2026-09-26 使用者裁定):`/asr` 收 `record: bool = True`,
预览方显式传 `False` ⟹ 只做事、不记账。**默认不变**,所以纯 ASR 调用
(不属任何回答)照旧留 transcript —— 那是既有契约,本文件钉住它不许被顺手改掉。

⚠️ 留存(原始字节落盘)**两种都照做、不随本开关变**:预览过又被重录的那一版
只在 `/asr` 下留过底,抹掉它就等于抹掉"客户端到底发过什么"—— 那正是 M2.6 存在的理由。
"""
import asyncio
import io
import json
import struct
import wave

import pytest

import media_retention                                                # noqa: E402
import importlib                                                      # noqa: E402
voice_app = importlib.import_module("voice_interaction.api.app")      # noqa: E402
from voice_interaction.asr.funasr_engine import AsrUtterance          # noqa: E402


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


def _utt() -> AsrUtterance:
    """一句真形状的识别结果(段形状与 transcript_store._segment_records 的契约一致)。"""
    seg = {"index": 0, "text": "我觉得这个问题很有意思。", "n_chars": 12,
           "timestamps_ms": [[0, 900], [900, 1800]], "punc_array": [1]}
    return AsrUtterance(text=seg["text"], n_chars=12, n_segments=1,
                        vad_split=False, segments=[seg])


async def _async_return(v):
    return v


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """真 transcript_store(不替身)—— 本文件要验的正是"账本里到底记了几条"。"""
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.delenv("JINGXIN_RETAIN_MEDIA", raising=False)
    media_retention._reset_for_tests()
    monkeypatch.setattr(voice_app, "_transcribe_async",
                        lambda audio_data: _async_return(_utt()))
    monkeypatch.setattr(voice_app, "log_recognition", lambda utt: None)
    monkeypatch.setattr(voice_app.voice_logger, "log_prosody", lambda *a, **kw: None)
    monkeypatch.setattr(voice_app.interview_assessment, "add_answer", lambda t: None)
    monkeypatch.setattr(voice_app.interview_assessment, "save_log", lambda: None)
    monkeypatch.setattr(voice_app.interview_assessment, "qa_pairs", [])
    return tmp_path


def _segments(root, sid):
    p = root / sid / "transcript.json"
    assert p.exists(), f"transcript.json 没写出来:{p}"
    return json.loads(p.read_text(encoding="utf-8"))["segments"]


def test_preview_then_submit_records_the_answer_once(wired):
    """红法:去掉 `/asr` 里对 `record` 的判断(重回到无条件 append)⟹ 这里得 2 段。

    这就是 2026-09-26 那一场的形态:同一份音频,预览一次 + 提交一次。
    """
    sid = "20260926_120000_pppp"
    payload = _wav_bytes()

    asyncio.run(voice_app.speech_to_text(
        request=_FakeRequest(), audio=_FakeUpload(payload), session_id=sid,
        record=False))                                    # ← 预览:别记账
    asyncio.run(voice_app.submit_answer_audio(
        request=_FakeRequest(), audio=_FakeUpload(payload), session_id=sid))

    assert len(_segments(wired, sid)) == 1, "预览不该在账本上留第二条"


def test_preview_still_returns_text(wired):
    """不记账 ≠ 不干活:预览仍要把识别文本回给调用方(界面就靠它显示)。"""
    r = asyncio.run(voice_app.speech_to_text(
        request=_FakeRequest(), audio=_FakeUpload(_wav_bytes()),
        session_id="20260926_120000_qqqq", record=False))
    assert r["text"] == "我觉得这个问题很有意思。"


def test_preview_still_retains_the_upload(wired):
    """留存不随开关变:预览过又被重录的那一版,只在 `/asr` 下留过底。"""
    asyncio.run(voice_app.speech_to_text(
        request=_FakeRequest(), audio=_FakeUpload(_wav_bytes()),
        session_id="20260926_120000_rrrr", record=False))
    d = wired / "20260926_120000_rrrr" / "media" / "audio"
    assert sorted(p.name for p in d.iterdir()) == ["0001.wav"], "预览的原始字节也该留"


def test_default_records_like_before(wired):
    """契约守卫:`record` 缺省时行为**不变**(纯 ASR 调用照旧留 transcript)。

    红法:把默认值写成 False,或让开关反过来 —— 纯 ASR 的调用方会**静默**
    不再留任何原句,而它们没有任何办法察觉。
    """
    sid = "20260926_120000_ssss"
    asyncio.run(voice_app.speech_to_text(
        request=_FakeRequest(), audio=_FakeUpload(_wav_bytes()), session_id=sid))
    assert len(_segments(wired, sid)) == 1
