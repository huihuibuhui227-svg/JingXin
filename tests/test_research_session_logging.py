# tests/test_research_session_logging.py
"""I2 的收口(2026-09-25 M2.1 独立复审提出,使用者当日授权按推荐方案执行):

**科研会话必须产出报告侧读得到的特征行,否则它在读侧永远描述不出来。**

失效现场(评审者用真日志目录实测):
* `/research/start` 铸了号,但科研的 `VoiceLogger` 是**不带 session_id** 现造的
  (`research_logger = VoiceLogger(log_type='research')`)→ 文件名恒为
  `research_emotion_log_NONE_<时间戳>.csv`;
* 科研回答那条路**不调 `log_prosody`**(与面试侧不对称);
* 于是盘上**不存在**任何带科研号的 `research_*_log_<sid>` 特征行,而 `ensure_manifest`
  给科研会话声明的 `expected_file` 一个都不可能被满足 → 报告里 `语音（科研）` 恒为 `missing`。

后果:「先面试 → 再科研 → 看报告」拿到的是科研会话那份(语音两栏全缺),
而它本该是一场**能被完整描述**的会话(`voice_research` 在 `feature_engine.py` 里
是真的会被消费的模态)。

红法:去掉 `/research/answer_audio` 里的 `log_prosody` 调用,或把 `/research/start` 里
带号的 logger 换回不带号的那个。
"""

import io
import wave

import importlib

import pytest
from fastapi.testclient import TestClient

from voice_interaction.api import app as voice_api

_voice_app_module = importlib.import_module("voice_interaction.api.app")

_ANSWER = "我先做了需求分析，然后我们讨论了方案，最后花了三天把原型做出来"


def _wav_bytes(seconds: float = 0.5, rate: int = 16000) -> bytes:
    """一个合法的 16 kHz / 16 bit / 单声道 WAV(端点会对格式做硬校验)。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x01" * int(rate * seconds))
    return buf.getvalue()


def _fake_utt(text: str):
    """形状够真:`append_utterance` 会真的读它的 `segments`(所以这里不替身掉它)。"""
    from voice_interaction.asr.funasr_engine import AsrUtterance

    return AsrUtterance(
        text=text, n_chars=len(text), n_segments=1, vad_split=False,
        segments=[{"text": text, "n_chars": len(text), "timestamps_ms": [], "punc_array": []}],
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    """仓库外录制目录与仓库内日志目录都挪到 tmp;TTS 换掉(本文件不测发声)。"""
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path / "rec"))
    monkeypatch.setattr(_voice_app_module.tts_engine, "speak", lambda *a, **k: None)
    import voice_interaction.utils.logger as voice_logger_module
    monkeypatch.setattr(voice_logger_module, "LOGS_DIR", str(tmp_path / "logs"))
    return TestClient(voice_api)


def test_research_answer_audio_writes_a_prosody_row(client, tmp_path, monkeypatch):
    """★ 科研回答要像面试回答一样,给**科研自己的号**写下一条特征行。

    红法:去掉 `/research/answer_audio` 里的 `log_prosody` 调用 →
    盘上没有 `research_emotion_log_<sid>.csv` → 红。
    """
    sid = client.post("/research/start").json()["session_id"]

    async def _fake_transcribe(pcm: bytes):
        return _fake_utt(_ANSWER)

    monkeypatch.setattr(_voice_app_module, "_transcribe_async", _fake_transcribe)
    # 评估管线本身不是本测试的对象(它还会另写一份人类可读的 note)
    monkeypatch.setattr(_voice_app_module.research_assessment, "add_answer", lambda text: None)
    monkeypatch.setattr(_voice_app_module.research_assessment, "save_log", lambda: "")

    resp = client.post(
        f"/research/answer_audio?session_id={sid}",
        files={"audio": ("answer.wav", _wav_bytes(), "audio/wav")},
    )
    assert resp.status_code == 200, resp.text

    csv_path = tmp_path / "logs" / f"research_emotion_log_{sid}.csv"
    assert csv_path.exists(), (
        f"科研会话没有产出特征行 —— 盘上是 "
        f"{sorted(p.name for p in (tmp_path / 'logs').glob('*.csv'))}")
    text = csv_path.read_text(encoding="utf-8")
    assert "connective_density" in text, "特征行里没有连接词密度列"
    assert sid in text, "首列不是这次会话的号"
    assert "NONE" not in text, "行落进了 NONE 桶"


def test_research_evaluation_logs_under_the_research_session_id(client, monkeypatch):
    """★ 评估行也要落在科研会话自己那份文件里(而不是 `..._NONE_<时间戳>.csv`)。

    红法:把 `/research/evaluation` 里那个 `VoiceLogger(log_type='research')` 加回来
    (不带号)→ csv_log 路径不含 sid → 红。
    """
    sid = client.post("/research/start").json()["session_id"]
    monkeypatch.setattr(_voice_app_module.research_assessment,
                        "get_comprehensive_evaluation", lambda: "评估文本")
    monkeypatch.setattr(_voice_app_module.research_assessment, "save_log", lambda: "")

    body = client.get("/research/evaluation").json()

    assert sid in body["csv_log"], f"评估行没落到本会话文件:{body['csv_log']}"
    assert "NONE" not in body["csv_log"], f"落进了 NONE 桶:{body['csv_log']}"
