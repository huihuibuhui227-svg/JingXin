"""`listen_for_speech` 的停录判定 —— 本文件之前**这条路径没有任何测试**。

背景(审查 Critical):上一版把它写成"队列里有没有数据"来判静音。但 `sd.RawInputStream`
每 0.5 s(16 kHz / blocksize 8000)就送一块 —— **不管有没有人在说话**,静音也是音频数据,
全零的 `bytes` 是真值。于是静音计时被每一块重置,永远到不了 `pause_threshold`,
每次录音都跑满 `timeout`(实测 5.0 s),并把这段全喂给引擎。三个示例从"说完就停"
变成"每个回答录满 30 s"。

修法:停录判定抽成纯函数 `should_stop`(按**真实能量**判),这段路径才第一次可测。
本文件两类测试:纯函数的离散场景 + 一个假麦克风的端到端复现。
"""

from __future__ import annotations

import queue
import threading
import time
import types

import numpy as np
import pytest

from voice_interaction.pipeline import speech_recognition_pipeline as srp
from voice_interaction.asr.funasr_engine import AsrUtterance

CHUNK_SAMPLES = 8000                      # 16 kHz 下 0.5 s,与 blocksize 一致
CHUNK_SECONDS = CHUNK_SAMPLES / 16000.0   # 0.5
FLOOR = 0.01                              # asr_config.json 里的静音能量下限


# ---------------------------------------------------------------- 信号构造

def _silence() -> bytes:
    return b"\x00" * (CHUNK_SAMPLES * 2)


def _speech(level: float = 0.2) -> bytes:
    """RMS ≈ 0.707 × level 的正弦:默认 0.14,远高于下限 0.01。"""
    t = np.arange(CHUNK_SAMPLES) / 16000.0
    return (level * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16).tobytes()


def _hiss(level: float = 0.002) -> bytes:
    """低幅白噪:模拟安静房间底噪,RMS < 0.01(应被判为静音)。"""
    rng = np.random.default_rng(0)
    return (rng.normal(0, level, CHUNK_SAMPLES) * 32767).astype(np.int16).tobytes()


# ---------------------------------------------------------------- 纯函数:chunk_energy

def test_chunk_energy_measures_signal_level_not_presence():
    """能量是**幅值**的函数:全零块与低幅底噪都必须远低于说话。

    这一条是修法的地基 —— 旧实现把"收到一块"当"还在说话",因为它手里根本没有
    任何幅值量。把 `chunk_energy` 改成"非空即 1.0"(即退化成旧的"有数据即有声"),
    本测试立刻红:`_hiss()` 会变成 1.0 而不是 ~0.002。
    """
    assert srp.chunk_energy(_silence()) == 0.0
    assert srp.chunk_energy(b"") == 0.0
    assert srp.chunk_energy(_hiss()) < FLOOR
    assert srp.chunk_energy(_speech()) > FLOOR


# ---------------------------------------------------------------- 纯函数:should_stop

def test_should_stop_when_trailing_silence_reaches_threshold():
    """说两块 → 静三块(1.5 s ≥ 1.2 s):该停。

    红法(审查给的例子):把 `should_stop` 里的"末尾连续静音时长"改回"只要收到过块就算
    还在说"(即每块都重置静音计数),末尾静音恒为 0 → 返回 False → 本测试红。
    """
    energies = [0.14, 0.14, 0.0, 0.0, 0.0]
    assert srp.should_stop(energies, pause_threshold_s=1.2,
                           chunk_seconds=CHUNK_SECONDS, energy_floor=FLOOR) is True


def test_should_stop_ignores_a_brief_pause_inside_speech():
    """说话中的短暂停顿(0.5 s < 1.2 s)不算说完 —— 否则一句话会被从中间截断。

    红法:把判定改成"出现任意一个静音块就停"(丢掉阈值比较) → 本测试红。
    """
    energies = [0.14, 0.0, 0.14]
    assert srp.should_stop(energies, pause_threshold_s=1.2,
                           chunk_seconds=CHUNK_SECONDS, energy_floor=FLOOR) is False


def test_should_stop_threshold_is_inclusive_and_granular_to_one_chunk():
    """边界:恰好等于阈值算到(≥);差一点算没到;阈值小于一块时按一块的粒度停。

    粒度就是一块 —— `pause_threshold` 低于 0.5 s(一块)时,实际效果等同于 0.5 s
    (第一块静音就停)。这是已知边界,写在 docstring 里,也在这一条钉住。
    """
    assert srp.should_stop([0.14, 0.0, 0.0], 1.0, CHUNK_SECONDS, FLOOR) is True     # 恰好 1.0
    assert srp.should_stop([0.14, 0.0, 0.0], 1.2, CHUNK_SECONDS, FLOOR) is False    # 1.0 < 1.2
    assert srp.should_stop([0.14, 0.0], 0.1, CHUNK_SECONDS, FLOOR) is True          # 阈值 < 一块
    assert srp.should_stop([0.14], 0.1, CHUNK_SECONDS, FLOOR) is False              # 还没静过


def test_should_stop_never_fires_before_the_user_has_spoken():
    """全程静音(还没开口)不该停:否则一进循环就返回空,用户来不及说第一句。

    红法:去掉"曾经有过高于下限的块"这道前置判断 → 全静音序列会被当成"说完了"→ 红。
    这条也是把 timeout 留给"用户一直没说话"的兜底依据。
    """
    assert srp.should_stop([0.0] * 10, 1.2, CHUNK_SECONDS, FLOOR) is False
    assert srp.should_stop([], 1.2, CHUNK_SECONDS, FLOOR) is False


# ---------------------------------------------------------------- 假麦克风端到端

class _FakeEngine:
    """记录收到的字节,返回固定文本(只被 `.text` 用到)。"""

    def __init__(self, text: str = "录音里的回答"):
        self.text = text
        self.calls: list[bytes] = []

    def transcribe_pcm(self, pcm: bytes) -> AsrUtterance:
        self.calls.append(pcm)
        return AsrUtterance(text=self.text, n_chars=len(self.text), n_segments=1,
                            vad_split=False, segments=[])


def _fake_mic(monkeypatch, script: list[bytes]):
    """把 `sd.RawInputStream` 换成假设备:按**消费节奏**供货,一次只放一块在队列里。

    为什么要"消费一块才补一块"而不是按墙上时间喂:后者的断言只能写成"耗时/缓冲大概
    小于某个数",而机器一变慢(这套测试会在 `test_assert_coverage` 的内层 `settrace`
    重跑里被执行)就会假红 —— 本项目已经栽过一次(4 位 id 的生日问题)。改成消费驱动后,
    **缓冲的块数是确定的**(停录规则只数块数,与快慢无关),断言可以钉死在一个窄区间里;
    墙钟只留一条"没有跑满 timeout"的宽松上界(15 s 的 timeout,期望值 ~0.1 s)。

    另一件必须保真的:真设备不管有没有人说话都持续送块(这正是审查那个 Critical 的根源),
    所以脚本走完后设备继续空转(不再供货,但也不退出),于是"没停"的实现在这里表现成
    "把脚本里的块全吃光 + 一直等到 timeout"。
    """
    state = {"delivered": 0}

    class TracingQueue(queue.Queue):
        """记下自己,好让假设备知道往哪个队列供货。"""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            state["queue"] = self

    monkeypatch.setattr(srp.queue, "Queue", TracingQueue)

    class FakeRawInputStream:
        def __init__(self, samplerate, blocksize, dtype, channels, callback):
            self.callback = callback
            self.thread = None
            self.stopped = threading.Event()

        def __enter__(self):
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            return self

        def __exit__(self, *exc):
            self.stopped.set()
            if self.thread:
                self.thread.join(timeout=10)
            return False

        def _run(self):
            while not state.get("queue"):
                if self.stopped.wait(0.01):
                    return
            for chunk in script:
                if self.stopped.is_set():
                    return
                # 等消费者把上一块取走(真设备的缓冲上限在这里用"队列空"代表)
                while not self.stopped.is_set() and not state["queue"].empty():
                    if self.stopped.wait(0.005):
                        return
                state["delivered"] += 1
                self.callback(chunk, len(chunk) // 2, None, None)
            while not self.stopped.wait(0.05):     # 设备还在,但脚本已尽
                pass

    monkeypatch.setattr(srp, "sd", types.SimpleNamespace(RawInputStream=FakeRawInputStream))
    return state


def test_listen_for_speech_stops_on_trailing_silence_instead_of_running_to_timeout(monkeypatch):
    """审查 Critical 的直接复现:末尾静音已远超阈值 → 必须停,不是录满 timeout。

    场景:说两块 → 静下来(阈值用**默认的 1.2 s**)。
    - 修好后:第 3 块静音到齐(1.5 s ≥ 1.2)就停 → 缓冲 5 块(2.5 s)。
    - 旧实现(任何队列项都重置静音计时,静音块也是"数据"):静音计时恒被重置,
      永远到不了 1.2 → 把脚本里的 32 块全吃光,再一直等到 `timeout=15` s
      (审查用真节奏实测:`timeout=5` 时返回耗时 5.0 s、缓冲 5.0 s)。

    两条断言都会红:缓冲块数(5–7 vs 32)与墙钟(≈0.1 s vs ≈15 s)。
    """
    pipe = srp.SpeechRecognitionPipeline()
    engine = _FakeEngine("录音里的回答")
    pipe.engine = engine
    _fake_mic(monkeypatch, [_speech(), _speech()] + [_silence()] * 30)

    started = time.monotonic()
    result, audio = pipe.listen_for_speech(timeout=15, pause_threshold=1.2)
    elapsed = time.monotonic() - started

    assert result.text == "录音里的回答"
    # 停录规则数的是**块数**(2 块说话 + 末尾 3 块静音,1.5s ≥ 1.2s = 5 块),与机器快慢无关,
    # 所以这里可以钉死;上界留一块在途的余量。旧实现会把脚本里的 32 块全吃光。
    assert 5 * CHUNK_SAMPLES <= len(audio) <= 7 * CHUNK_SAMPLES, \
        f"缓冲了 {len(audio) / 16000:.2f}s 音频({len(audio) // CHUNK_SAMPLES} 块)"
    # 墙钟只做宽松上界:期望 ~0.1s,旧实现必然 ≈ timeout(15s)
    assert elapsed < 5.0, f"没在静音处停下,录了 {elapsed:.2f}s(timeout=15)"
    # 喂给引擎的就是缓冲下来的那一段(旧实现会把整场都喂过去)
    assert [len(c) for c in engine.calls] == [len(audio) * 2]


def test_listen_for_speech_quiet_room_hiss_does_not_count_as_speech(monkeypatch):
    """安静房间的底噪(RMS≈0.002,低于下限 0.01)必须算静音。

    红法:把能量判定换成"非空块即有声"(变异 M6) → 底噪变成"说话"→ 末尾静音恒为 0 →
    不停 → 把 21 块全吃光并等到 `timeout=15` s → 两条断言都红。
    """
    pipe = srp.SpeechRecognitionPipeline()
    pipe.engine = _FakeEngine("回答")
    _fake_mic(monkeypatch, [_speech(), _hiss()] + [_hiss()] * 20)

    started = time.monotonic()
    _, audio = pipe.listen_for_speech(timeout=15, pause_threshold=1.2)
    elapsed = time.monotonic() - started

    # 1 块说话 + 末尾 3 块底噪(1.5s ≥ 1.2s = 4 块),上界留一块在途余量
    assert 4 * CHUNK_SAMPLES <= len(audio) <= 6 * CHUNK_SAMPLES, \
        f"缓冲了 {len(audio) / 16000:.2f}s 音频({len(audio) // CHUNK_SAMPLES} 块)"
    assert elapsed < 5.0, f"底噪被当成说话,录了 {elapsed:.2f}s(timeout=15)"


def test_silence_energy_floor_comes_from_asr_config(monkeypatch):
    """能量下限只能来自 asr_config.json(代码里不许写裸常量)。

    把下限抬到比"说话"还高 → 同一段音频全部算静音 → "还没开口"的前置判断不成立 →
    只能等 timeout 收。若实现把下限写死在代码里(不读配置),抬高配置不会改变行为,本测试红。
    """
    raised = dict(srp.load_config())
    raised["silence_energy_floor"] = {"value": 0.9}      # 高于 _speech() 的 RMS(≈0.14)
    monkeypatch.setattr(srp, "load_config", lambda: raised)

    pipe = srp.SpeechRecognitionPipeline()
    pipe.engine = _FakeEngine("回答")
    assert pipe.silence_energy_floor == 0.9, "下限没有从配置读进来"
    _fake_mic(monkeypatch, [_speech(), _silence(), _silence(), _silence()])

    started = time.monotonic()
    _, audio = pipe.listen_for_speech(timeout=1, pause_threshold=1.2)
    elapsed = time.monotonic() - started

    assert elapsed >= 1.0, "下限没生效:说话被当静音却仍提前停了"
    assert len(audio) >= 4 * CHUNK_SAMPLES
