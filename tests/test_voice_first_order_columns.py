# tests/test_voice_first_order_columns.py
"""语音补一阶量(2026-09-26):回答字数、有声时长、语速、能量/基频分位。

为什么补这几个 —— 每一个都对着一条**已知的缺陷**:
  · `n_chars`      → 报告里「回答详尽度」此前**全系统没有产出方**(只有映射表里那两行);
  · `speech_duration_sec` → `speech_ratio` 的**绝对量版本**。后者是自指阈值
    (封停理由:实测 87.9% 恰为 1.0)。同一段真音频上:绝对量说 12.07s/17.28s(70%),
    而 `speech_ratio` 那一列写 1.0 —— 这就是"自指"长什么样;
  · `energy_p10/p90`、`pitch_p10/p90` → 「会话内归一」的原料(energy/pitch 封停理由
    都写着要会话内归一,而现在只有 mean/std,受极端帧拖拽)。

不变量:取不到就**留空**,不补 0(0 是个合法的字数/秒数)。
"""
import csv
import importlib

import numpy as np
import pytest

vl = importlib.import_module("voice_interaction.utils.logger")


def _wav_like(seconds=3.0, sr=16000):
    """一段"前 1 秒静音 + 后 2 秒正弦"的假语音:停顿/有声时长都算得出来。"""
    t = np.arange(int(seconds * sr)) / sr
    x = np.zeros_like(t)
    x[int(1.0 * sr):] = 0.3 * np.sin(2 * np.pi * 180 * t[int(1.0 * sr):])
    return x.astype(np.float32)


def test_extractor_reports_speech_seconds_and_percentiles():
    """红法:去掉 `speech_duration_sec` / `*_p10/p90` —— 键不在,下游全空。"""
    from voice_interaction.core.feature_extraction.prosody_extractor import (
        ProsodyFeatureExtractor)

    f = ProsodyFeatureExtractor().extract_all_features(_wav_like())
    for key in ("speech_duration_sec", "energy_p10", "energy_p90",
                "pitch_p10", "pitch_p90"):
        assert key in f, f"{key} 没有产出"

    # 前 1 秒是静音 ⟹ 有声时长该明显小于总时长(而 speech_ratio 那种自指量看不出来)
    assert 0 < f["speech_duration_sec"] < f["duration_sec"], f
    assert f["energy_p90"] > f["energy_p10"], "分位数该拉开(静音 vs 有声)"


def test_voice_log_row_carries_the_new_columns(tmp_path, monkeypatch):
    """★ 列要真的落进 CSV。红法:去掉 fieldnames 里那七列 / 去掉行体那几行。"""
    monkeypatch.setattr(vl, "LOGS_DIR", str(tmp_path))
    log = vl.VoiceLogger(log_type="interview", session_id="20260926_120000_vvvv")

    for col in ("n_chars", "speech_duration_sec", "chars_per_sec",
                "energy_p10", "energy_p90", "pitch_p10", "pitch_p90"):
        assert col in log.fieldnames, f"{col} 没进表头"

    log.log_prosody({"pitch_mean": 150.0, "pitch_variation": 30.0,
                     "energy_mean": 0.05, "energy_variation": 0.01,
                     "speech_ratio": 1.0, "duration_sec": 12.0,
                     "pitch_p10": 100.0, "pitch_p90": 220.0,
                     "energy_p10": 0.001, "energy_p90": 0.09,
                     "speech_duration_sec": 9.5, "n_chars": 42,
                     "chars_per_sec": 4.42},
                    question_index=0, emotion="", feedback="")

    rows = list(csv.DictReader(open(log.csv_file, encoding="utf-8")))
    assert rows, "一行都没写"
    r = rows[-1]
    assert r["n_chars"] == "42" and r["speech_duration_sec"] == "9.5"
    assert r["chars_per_sec"] == "4.42" and r["pitch_p90"] == "220.0"


def test_missing_values_stay_empty_not_zero(tmp_path, monkeypatch):
    """取不到就**留空**:0 是个合法的字数/秒数,补 0 会把"没算出来"写成"算出来是 0"。

    红法:把行体改成 `prosody_data.get("n_chars", 0)` —— 这条立刻红。
    """
    monkeypatch.setattr(vl, "LOGS_DIR", str(tmp_path))
    log = vl.VoiceLogger(log_type="interview", session_id="20260926_120000_wwww")
    log.log_prosody({"pitch_mean": 150.0, "energy_mean": 0.05, "duration_sec": 1.0},
                    question_index=0, emotion="", feedback="")
    r = list(csv.DictReader(open(log.csv_file, encoding="utf-8")))[-1]
    assert r["n_chars"] == "" and r["chars_per_sec"] == "", \
        f"缺值写成了 {r['n_chars']!r} / {r['chars_per_sec']!r}"


def test_reaction_time_needs_all_three_ingredients():
    """反应延迟 = 首次开口墙钟 − ask_end;三个成分缺一个就**整组不写**(不猜、不补 0)。

    红法:把任一个 `return {}` 改成返回带 0 的字典 —— "算不出来"立刻变成一个看着正常的反应时间。
    """
    from voice_interaction.api.app import reaction_time_features as rt

    q = {"index": 0, "ask_end": 1000.0}
    ok = rt({"speech_onset_sec": 1.8, "duration_sec": 17.28}, 1020.0, q)
    # 录音开始 = 1020 − 17.28 = 1002.72;首次开口 = +1.8 = 1004.52;反应 = 4.52
    assert ok == {"speech_onset_sec": 1.8, "answer_onset_wall": 1004.52,
                  "reaction_time": 4.52}, ok
    assert rt({"duration_sec": 17.28}, 1020.0, q) == {}          # 缺首次开口
    assert rt({"speech_onset_sec": 1.8}, 1020.0, q) == {}        # 缺时长
    assert rt({"speech_onset_sec": 1.8, "duration_sec": 17.28}, None, q) == {}   # 缺到达时刻
    assert rt({"speech_onset_sec": 1.8, "duration_sec": 17.28}, 1020.0, None) == {}  # 缺题目窗口


def test_reaction_time_can_be_negative_and_is_not_clamped():
    """抢答(题还没念完就开口)⟹ **负值是真实现象,不截断**。

    红法:加一句 `max(0.0, ...)` —— 于是"抢答"与"0 秒反应"再也分不开。
    """
    from voice_interaction.api.app import reaction_time_features as rt
    out = rt({"speech_onset_sec": 0.2, "duration_sec": 3.0}, 1000.0,
             {"index": 0, "ask_end": 1000.0})
    assert out["reaction_time"] == -2.8, out


def test_voice_row_carries_the_reaction_time_columns(tmp_path, monkeypatch):
    """三个成分都要落盘 —— 只给结论的数复核不了,只能信。"""
    monkeypatch.setattr(vl, "LOGS_DIR", str(tmp_path))
    log = vl.VoiceLogger(log_type="interview", session_id="20260926_120000_rrrr")
    for col in ("speech_onset_sec", "answer_onset_wall", "reaction_time"):
        assert col in log.fieldnames, f"{col} 没进表头"
    log.log_prosody({"pitch_mean": 150.0, "energy_mean": 0.05, "duration_sec": 17.28,
                     "speech_onset_sec": 1.8, "answer_onset_wall": 1790401004.52,
                     "reaction_time": 4.52},
                    question_index=0, emotion="", feedback="")
    r = list(csv.DictReader(open(log.csv_file, encoding="utf-8")))[-1]
    assert r["reaction_time"] == "4.52" and r["speech_onset_sec"] == "1.8", r


def test_endpoint_writes_reaction_time_from_the_question_ledger(tmp_path, monkeypatch):
    """★ 端到端接线:题目台账里有 ask_end + 音频有到达时刻 ⟹ 反应延迟进得了日志。

    这条钉的是"**端点到底有没有把三样凑起来**" —— 纯函数单测与 logger 单测都
    各自绿着,却可能中间那根线根本没人接(本项目反复栽的形态)。

    红法:删掉端点里 `prosody.update(reaction_time_features(...))` 那两行。
    """
    import asyncio
    import io
    import struct
    import wave

    import media_retention
    voice_app = importlib.import_module("voice_interaction.api.app")
    from voice_interaction.asr.funasr_engine import AsrUtterance

    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.delenv("JINGXIN_RETAIN_MEDIA", raising=False)
    media_retention._reset_for_tests()

    # 一段"静音 + 正弦"的 16k 单声道 wav:前导静音 1.5s,总长 4s
    sr = 16000
    t = np.arange(4 * sr) / sr
    samples = np.zeros_like(t)
    samples[int(1.5 * sr):] = 0.3 * np.sin(2 * np.pi * 180 * t[int(1.5 * sr):])
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(struct.pack("<%dh" % len(samples), *(samples * 32767).astype(int)))

    class _Up:
        filename, content_type = "a.wav", "audio/wav"
        async def read(self): return buf.getvalue()

    class _Req:
        async def form(self): raise RuntimeError("no multipart")

    seg = {"index": 0, "text": "这是一段回答。", "n_chars": 7,
           "timestamps_ms": [[0, 500]], "punc_array": [1]}
    async def _fake_transcribe(_pcm):
        return AsrUtterance(text="这是一段回答。", n_chars=7, n_segments=1,
                            vad_split=False, segments=[seg])

    monkeypatch.setattr(voice_app, "_transcribe_async", _fake_transcribe)
    monkeypatch.setattr(voice_app, "log_recognition", lambda utt: None)
    monkeypatch.setattr(voice_app.interview_assessment, "add_answer", lambda t: None)
    monkeypatch.setattr(voice_app.interview_assessment, "save_log", lambda: None)
    monkeypatch.setattr(voice_app.interview_assessment, "qa_pairs", [])
    # 题目台账:该题在 1000.0 问完
    monkeypatch.setattr(voice_app.session_meta, "read_questions",
                        lambda sid: [{"index": 0, "ask_start": 990.0, "ask_end": 1000.0}])
    # 到达时刻 = 1002.0(录音 4s ⟹ 录音开始 998.0;+1.5s 前导静音 = 999.5 开口)
    monkeypatch.setattr(voice_app.media_retention, "retain_audio",
                        lambda *a, **k: {"seq": 1, "received_at_wall": 1002.0})

    seen = {}
    monkeypatch.setattr(voice_app.voice_logger, "log_prosody",
                        lambda prosody, **kw: seen.update(prosody))

    asyncio.run(voice_app.submit_answer_audio(
        request=_Req(), audio=_Up(), session_id="20260926_120000_tttt"))

    assert "reaction_time" in seen, f"端点没把反应延迟凑出来:{sorted(seen)}"
    # 起音检测有 ~1 帧(32ms)的颗粒度,RMS 窗还会跨过边界 ⟹ **不硬编码那一秒**,
    # 断的是"三个成分按定义拼出来的那个值"(这才是接线要保证的东西)。
    assert abs(seen["speech_onset_sec"] - 1.5) < 0.15, seen["speech_onset_sec"]
    expected = (1002.0 - 4.0 + seen["speech_onset_sec"]) - 1000.0
    assert abs(seen["reaction_time"] - expected) < 0.01, (seen["reaction_time"], expected)
    # 这段是"题念完前就开口"的构造 ⟹ 该是**负值**,且没有被截断成 0
    assert seen["reaction_time"] < 0, seen["reaction_time"]
