# tests/test_prosody_wiring.py
"""语音活路径必须**真的算**语调 —— 不能再拿空字典冒充「有效」。

背景(§0.1 第 1 件,2026-09-25 深夜):

`/interview/answer_audio` 与 `/research/answer_audio` 把 `log_prosody({}, …)` 的空字典
写进日志,`log_prosody` 对缺的键一律 `.get(..., 0)`,于是
`pitch_mean` / `pitch_variation` / `energy_mean` / `energy_variation` /
`speech_ratio` / `pause_*` / `duration_sec` **全是 0**,而那一行 `is_valid` 恒 True。
`ProsodyFeatureExtractor` 一直在仓库里躺着,只被两个 `examples/` 引用。

这个文件守两件**不同**的事,故意分成两组、各自钉各自的坏法,
免得一条红了看不出是哪一处接线坏了:

1. 真音频进来,那几个列必须拿到**真值** —— 红法只有一条:恢复 `{}`;
2. `pitch_std` / `energy_std` → `pitch_variation` / `energy_variation` 的**改名映射**
   必须真的发生过 —— 红法:把映射表改空。少了它**只有那两列**静默留 0,
   其余每列都对,是最难发现的那种。

另有两条守"外围":真实 webm 那条路、以及提取不许卡住事件循环。
"""
import asyncio
import importlib
import io
import subprocess
import threading
import wave
from pathlib import Path

import numpy as np
import pytest

import media_retention                                                    # noqa: E402
voice_app = importlib.import_module("voice_interaction.api.app")          # noqa: E402
prosody_mod = importlib.import_module(
    "voice_interaction.core.feature_extraction.prosody_extractor")       # noqa: E402

SAMPLE_RATE = 16000
SECONDS = 3.0
GAP = (1.0, 1.5)          # 0.5 s 静音 —— 下限就是 0.5,理由见 _tone_pcm

# §0.1 点名的那几列里,**名字对名字直接过去**的那些(外加 duration_sec:它是
# "音频真的走到提取器了"的最直接证据)。改名的那两列**故意不列在这里** ——
# 它们由下面单独一条守,这样"哪条红了"能直接指出是哪一处接线坏了。
WIRED_NUMERIC = ["pitch_mean", "energy_mean", "speech_ratio", "duration_sec",
                 "pause_duration_mean", "pause_frequency"]
# 走改名表才落得进日志的两列(提取器叫 *_std,日志列叫 *_variation)。
RENAMED_NUMERIC = ["pitch_variation", "energy_variation"]


def _tone_pcm(seconds: float = SECONDS) -> np.ndarray:
    """一段**特征确定**的 16k 单声道信号,专为"能验出真值"而造。

    为什么造它,而不是拿静音或噪声糊:
      - 前半 150 Hz / 后半 200 Hz → `pitch_std`、`pitch_trend` 非 0;
      - 幅度 8000 → 20000 渐强 → `energy_std` 非 0;
      - 中间 0.5 s 静音 → `pause_*` 非 0。

    **中间那段静音的下限是 0.5 s,不是随便挑的**(实测):`librosa.feature.rms`
    窗长 2048(= 0.128 s @16k),0.2 s 的静音只有 3 个「整窗都安静」的帧,
    按提取器 `(i - pause_start) * duration/len(rms)` 算出来是 0.095 s,
    过不了它自己那道 `pause_duration > 0.1` 的闸 → 停顿列仍然是 0,
    于是这条测试会**假绿**(看着接好了,其实停顿那族一行都没验到)。
    """
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    f0 = np.where(t < seconds / 2, 150.0, 200.0)
    phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
    x = np.linspace(8000, 20000, n) * np.sin(phase)
    x[int(GAP[0] * SAMPLE_RATE):int(GAP[1] * SAMPLE_RATE)] = 0
    return x.astype(np.int16)


def _wav_bytes(pcm: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


WEBM = b"\x1a\x45\xdf\xa3" + b"fake-opus-payload" * 20       # EBML 头


class _FakeUpload:
    def __init__(self, data: bytes, name="a.wav"):
        self._data, self.filename, self.content_type = data, name, "audio/wav"

    async def read(self) -> bytes:
        return self._data


class _FakeRequest:
    async def form(self):
        raise RuntimeError("no multipart")


class _FakeUtt:
    text = "我 觉得 这个 问题 很 有 意思"


async def _ret(v):
    return v


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.delenv("JINGXIN_RETAIN_MEDIA", raising=False)
    media_retention._reset_for_tests()
    return tmp_path


@pytest.fixture
def logged(monkeypatch):
    """抓住端点**真正交给** `log_prosody` 的那本字典(第 1 个位置参数)。

    只抓 kwargs 是不够的 —— 特征字典是**位置参数**,而那正是本次要验的东西。
    """
    seen: dict = {}

    def _capture(prosody_data, **kw):
        seen.clear()
        seen.update(prosody_data)
        seen["_kwargs"] = kw
        return True

    monkeypatch.setattr(voice_app, "_transcribe_async", lambda a: _ret(_FakeUtt()))
    monkeypatch.setattr(voice_app.transcript_store, "append_utterance", lambda *a, **k: None)
    monkeypatch.setattr(voice_app.voice_logger, "log_prosody", _capture)
    monkeypatch.setattr(voice_app.research_logger, "log_prosody", _capture)
    monkeypatch.setattr(voice_app.interview_assessment, "add_answer", lambda t: None)
    monkeypatch.setattr(voice_app.interview_assessment, "save_log", lambda: None)
    monkeypatch.setattr(voice_app.interview_assessment, "qa_pairs", [])
    monkeypatch.setattr(voice_app.research_assessment, "add_answer", lambda t: None)
    monkeypatch.setattr(voice_app.research_assessment, "save_log", lambda: None)
    monkeypatch.setattr(voice_app.research_assessment, "qa_pairs", [])
    return seen


def _fake_ffmpeg_writing(pcm: np.ndarray, monkeypatch) -> dict:
    """假 ffmpeg:把输出文件写成**带真特征**的 16k 单声道 WAV(不是静音)。"""
    seen: dict = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        Path(cmd[-1]).write_bytes(_wav_bytes(pcm))

        class _R:
            returncode = 0
            stderr = b""
        return _R()

    monkeypatch.setattr(subprocess, "run", fake_run)
    return seen


# ---------------------------------------------------------------- 第 1 件:真值
def test_interview_answer_audio_logs_real_prosody(logged):
    """§0.1 第 1 件的正面验收:一段真音频进来,那些列必须是**真值**。

    红法(唯一一条):把 `submit_answer_audio` 里 `log_prosody(...)` 的第 1 个参数
    恢复成 `{}` → 本测试的 6 列全部回 0(`log_prosody` 对缺键 `.get(..., 0)`)。
    """
    r = asyncio.run(voice_app.submit_answer_audio(
        request=_FakeRequest(), audio=_FakeUpload(_wav_bytes(_tone_pcm())),
        session_id="20260925_203826_2449"))

    assert r["status"] == "success"
    for name in WIRED_NUMERIC:
        assert logged.get(name, 0) > 0, f"{name} 还是 0 —— 语调特征没真算"
    # 时长对得上,才说明提取器拿到的是**这段**音频(e(0.5 s 静音也在里面)
    assert logged["duration_sec"] == pytest.approx(SECONDS, abs=0.05)


def test_research_answer_audio_also_logs_real_prosody(logged):
    """科研侧与面试侧**同口径**(§0.1:「两处 `log_prosody(...)`」)。

    红法:只接面试侧那一处 → 本测试红,面试侧那条仍绿。
    """
    r = asyncio.run(voice_app.submit_research_answer_audio(
        request=_FakeRequest(), audio=_FakeUpload(_wav_bytes(_tone_pcm())),
        session_id="20260925_203826_2449"))

    assert r["status"] == "success"
    for name in WIRED_NUMERIC:
        assert logged.get(name, 0) > 0, f"{name} 还是 0 —— 科研侧没接线"


def test_prosody_is_computed_on_the_webm_path_the_browser_actually_uses(logged, monkeypatch):
    """**真实使用走的是这条路**:前端 `MediaRecorder` 发的是 webm/opus,不是 WAV
    (见 `test_answer_audio_accepts_webm.py` 记的那场真事故)。

    只在 WAV 上验会漏掉整条真实路径 —— ffmpeg 转出来的那份字节有没有走到提取器,
    只有这条测试说得上话。
    """
    _fake_ffmpeg_writing(_tone_pcm(), monkeypatch)
    r = asyncio.run(voice_app.submit_answer_audio(
        request=_FakeRequest(), audio=_FakeUpload(WEBM, name="a.webm"),
        session_id="20260925_203826_2449"))

    assert r["status"] == "success"
    for name in WIRED_NUMERIC:
        assert logged.get(name, 0) > 0, f"{name} 还是 0 —— webm 那条路没接上"


# ------------------------------------------------- 第 2 件:改名映射(单独钉)
def test_extractor_std_columns_are_renamed_to_the_log_column_names(logged):
    """提取器吐 `pitch_std` / `energy_std`,而日志列叫 `pitch_variation` / `energy_variation`。

    口径来源是仓库自己的定义:`ProsodyAnalyzer.analyze_pitch` 里就是
    `"pitch_variation": pitch_std`(`analyze_energy` 同理 `energy_variation: energy_std`)。

    **少了改名映射,只有这两列静默留 0,其余每一列都对** —— 所以它单独一条,
    别并进上面那两条里,否则红了看不出是这个原因。
    红法:把 `app.py` 里 `_EXTRACTOR_TO_LOG_COLUMNS` 改成 `{}`。
    """
    asyncio.run(voice_app.submit_answer_audio(
        request=_FakeRequest(), audio=_FakeUpload(_wav_bytes(_tone_pcm())),
        session_id="20260925_203826_2449"))

    for name in RENAMED_NUMERIC:
        assert logged.get(name, 0) > 0, f"{name} 还是 0 —— 改名映射没生效"


def test_rename_map_has_no_silent_gap_between_extractor_and_logger():
    """照**上游**点名:提取器产出的每个键,要么本身就是日志列,要么在改名表里。

    这条不验数值,验的是**契约完整**:将来提取器多产一个键、或日志多要一个键,
    这条会红 —— 而上面那些数值断言不会(新键没人断言,老键也仍然非 0)。
    """
    from voice_interaction.utils.logger import VoiceLogger

    produced = set(prosody_mod.ProsodyFeatureExtractor(SAMPLE_RATE)
                   .extract_all_features(_tone_pcm().astype(np.float32) / 32768.0))
    landed = set(VoiceLogger(log_type="interview", log_dir="/tmp").fieldnames)

    renamed = set(getattr(voice_app, "_EXTRACTOR_TO_LOG_COLUMNS", {}))
    # 提取器产出的键:已经在日志列里,或明确登记进改名表
    assert produced <= (landed | renamed), (
        f"提取器产出了日志层不认识、也没登记改名的键: {sorted(produced - landed - renamed)}")


# ---------------------------------------------------- 第 3 件:不许卡住事件循环
def test_extraction_runs_off_the_event_loop(monkeypatch):
    """pyin 是**同步重活**:实测 3 s 音频 ≈ 0.3 s、29.5 s ≈ 3.7 s(与 ASR 的 1.75 s 同量级)。

    端点都是 `async def`,直接在协程里算会**卡住整个事件循环** —— 期间服务连
    `/health` 都不回。这文件顶上 `_transcribe_async` 的注释已经为 ASR 记过同一条,
    语调必须走同一条路(`asyncio.to_thread`)。

    红法:把 `_prosody_async` 里的 `asyncio.to_thread` 去掉,改成同步调用 →
    本测试红在「跑在主线程上」。
    """
    where: dict = {}
    real = prosody_mod.ProsodyFeatureExtractor.extract_all_features

    def spy(self, audio):
        where["thread"] = threading.current_thread()
        return real(self, audio)

    monkeypatch.setattr(prosody_mod.ProsodyFeatureExtractor, "extract_all_features", spy)
    asyncio.run(voice_app.submit_answer_audio(
        request=_FakeRequest(), audio=_FakeUpload(_wav_bytes(_tone_pcm())),
        session_id="20260925_203826_2449"))

    assert where.get("thread") is not None, "提取器根本没被调到"
    assert where["thread"] is not threading.main_thread(), (
        "语调提取跑在事件循环线程上 —— 会卡住整个服务")
