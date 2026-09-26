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


def _pcm_bytes(x: np.ndarray) -> bytes:
    """float 波形 → 16k/16bit 小端 PCM 字节(**活路径**的入口口径)。

    活路径 `voice_interaction/api/app.py:140` 的 `prosody_features_from_pcm(audio_data: bytes)` 吃的就是
    这个格式,所以端到端那一条测试必须从这里进 —— 直接调提取器会绕过改名映射表。
    """
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _extractor_features(audio: np.ndarray) -> dict:
    from voice_interaction.core.feature_extraction.prosody_extractor import (
        ProsodyFeatureExtractor)
    return ProsodyFeatureExtractor().extract_all_features(audio)


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


# ─────────────────────────────────────────────────────────────────────────────
# `voiced_prob`(2026-09-26,Task 5):`librosa.pyin` 的**第 3 个返回值**此前被 `_` 丢掉
# (`prosody_extractor.py:47` 的 `f0, voiced_flag, voiced_prob = librosa.pyin(...)`;改前那一行是 `_`)。
# `docs/superpowers/specs/2026-09-21-jingxin-feature-redesign-design.md:235` 的依据栏点名它是 pitch 修正的原料
# (「`pyin` 无能量门限且丢 `voiced_prob`」)。
#
# ★ 全静音是这条列的**分水岭**,两个量必须**分开**处理(plan
# `docs/superpowers/plans/2026-09-26-m3-0-l0-column-table.md:38` 把这条划给 Task 5):
#     · `voiced_prob_mean` = **0.0 是真值**(确实一个浊音帧都没有,pyin 的概率恒 0);
#       不许"顺手"把它特殊处理成空 —— 那会把"量到了 0"改成"没测到";
#     · `pitch_mean` = **空** —— f0 全 `nan`,均值无定义。写 0 就是"把没测到写成 0"。
# ─────────────────────────────────────────────────────────────────────────────


def test_voiced_prob_reaches_the_log(tmp_path, monkeypatch):
    """★ 红法:去掉 `fieldnames` 里那一列(或把它插到中间而不是末尾)。

    ⚠️ **位置也是契约**:必须是**末列**。插中间会让历史 CSV 的列序对不上 ——
    报告层按列名读,错位后会**静默读到别的列的值**。实测历史语音日志的末列是
    `reaction_time`(`data/logs/interview_emotion_log_20260926_155559_caf0.csv` 表头)。
    (将来若在它后面再追加新列,这条要**跟着改**,而不是删 —— 它保证的是
    "新列只往末尾加"。)
    """
    monkeypatch.setattr(vl, "LOGS_DIR", str(tmp_path))
    log = vl.VoiceLogger(log_type="interview", session_id="20260926_120000_vvp1")
    assert "voiced_prob_mean" in log.fieldnames, "voiced_prob_mean 没进表头"
    assert log.fieldnames[-1] == "voiced_prob_mean", "新列必须在列序末尾"


def test_voiced_prob_is_registered_in_the_extractor_map():
    """★ 坑 1(`docs/superpowers/sdd/2026-09-25-n1-prosody-wiring/progress.md:115` §4「本轮最大的静默陷阱」;同一件事也写在 `docs/下一步.md:722` 第 10 条):
    **上游产出名 ≠ 下游列名**是静默零值的根源。

    这条是**声明式**的,老实说清它抓什么、抓不到什么:
      · 删掉 `_EXTRACTOR_TO_LOG_COLUMNS` 里那一行,**今天**的值照样落得进 CSV
        (两边同名,`{map}.get(k, k)` 原样透传)⟹ **任何"值"断言都不会红**;
      · 它防的是**将来**:上游一改名("同名所以自动对上"不再成立),那一列就静默留空,
        而值断言只会看到"没红"。
    所以同名映射只能靠"**显式登记**"本身来钉 —— 与 `api/app.py:91-96` 那段注释同一件事。
    红法:删掉映射表里 `"voiced_prob_mean": "voiced_prob_mean"` 这一行。
    """
    voice_app = importlib.import_module("voice_interaction.api.app")
    mapping = voice_app._EXTRACTOR_TO_LOG_COLUMNS
    assert mapping.get("voiced_prob_mean") == "voiced_prob_mean", (
        "同名也要显式登记 —— 少了这一条,上游一改名本列就静默留空,"
        f"而其余每一列都对。当前映射表:{mapping}")


def test_voiced_prob_is_zero_on_all_silence_but_pitch_mean_is_empty():
    """全静音:两个量**必须分开处理**,不能一起写成 0(plan:38)。

      · `voiced_prob_mean` = **0.0 是真值**(pyin 对全静音给全 `False` 的 `voiced_flag`
        与全 0 的概率,确实一个浊音帧都没有)⟹ 不许特殊处理成空,那会把真值改成"没测到";
      · `pitch_mean` = **空** —— f0 全 `nan`,`mean` 无定义。写 0 就是"把没测到写成 0"。

    红法:① 把全静音分支的 `voiced_prob_mean` 改成 `None` → 第一句红;
          ② 把 `pitch_mean` 那格改回 `0.0` → 第二句红。
    """
    f = _extractor_features(np.zeros(3 * 16000, dtype=np.float32))
    # 先证明"这确实是那条分支":没有浊音帧 ⟹ 方向判不出来(该分支自己的标记)
    assert f["pitch_direction"] == "无法判断", f"这不是'零浊音帧'那条分支:{f}"
    assert f["voiced_prob_mean"] == 0.0, (
        f"全静音的概率均值是真值 0.0,不是空:{f['voiced_prob_mean']!r}")
    assert f["pitch_mean"] is None, (
        f"f0 全 nan ⟹ `pitch_mean` 该留空,实为 {f['pitch_mean']!r}(0 是「量到 0 Hz」)")


def test_zero_voiced_frames_leave_every_pitch_statistic_empty():
    """★ 零浊音帧:`f0_voiced` 上的**五个统计量一个都不许写 0**(2026-09-26 使用者裁定)。

    为什么把这四个跟 `pitch_mean` 一起改(裁定原话的意识):`pitch_p90 = 0.0` 读起来是
    「音高的 90 分位是 0 Hz」—— **一个看着像测量值、其实什么都没量到的数**。只修均值、
    留四个不修,读的人**无法判断这一支到底可不可信**。它们与 `pitch_mean` 是同一个判据
    (`f0` 全 `nan` ⟹ 该量无定义),所以复用同一支、不在别处复制逻辑。

    红法:把那一支里任一个改回 `0.0`(`pitch_std` / `pitch_trend` / `pitch_p10` / `pitch_p90`)⟹ 立刻红。
    """
    f = _extractor_features(np.zeros(3 * 16000, dtype=np.float32))
    assert f["pitch_direction"] == "无法判断", f"这不是'零浊音帧'那条分支:{f}"
    for col in ("pitch_mean", "pitch_std", "pitch_trend", "pitch_p10", "pitch_p90"):
        assert col in f, f"{col} 连键都没有 —— 下游按列名读会读到'缺列',不是'空值'"
        assert f[col] is None, (
            f"{col} 该留空(f0 全 nan、无定义),实为 {f[col]!r}"
            f" —— 0 是个会被下游当成真值的数")


def test_too_few_voiced_frames_have_no_trend_but_keeps_the_other_statistics(monkeypatch):
    """★ **浊音帧少于 3 帧**时:`pitch_trend` 该留空,不写 0.0、更不许写 `nan`。

    与零浊音帧**同一条判据**:切「首/末三分之一」要 `n//3 ≥ 1`,少于 3 帧切出来是**空切片**
    ⟹ `np.mean([])` = `nan`。改前实测(2026-09-26):**1 帧 → `0.0`**(一个看着像测量值、
    其实什么都没量到的数);**2 帧 → `nan`**(落盘成字符串 `"nan"`,同一类假值),而且
    `pitch_direction` 会写 `"平稳"` —— 那是一句**假话**(根本没量到趋势)。现在 <3 帧一律留空 +
    `"无法判断"`。

    ⚠️ 同一支里 `pitch_mean` / `pitch_std` / `pitch_p10` / `pitch_p90` **照常有值** ——
    它们对 1 帧**是**有定义的(均值就是那个数、单样本标准差按定义 0.0、分位也是那个数),
    所以这一条与"零浊音帧五个一起留空"**不冲突**:判据是"该量有没有定义",不是"帧数够不够多"。

    红法:把门槛改回 `> 1`(或把 `else` 支改回 `pitch_trend = 0.0`)⟹ 第一句红。
    用假 `pyin` 造确定的帧数,因为靠真音频凑不出。
    """
    import types
    prosody_mod = importlib.import_module(
        "voice_interaction.core.feature_extraction.prosody_extractor")
    ext = prosody_mod.ProsodyFeatureExtractor()          # 先构造(init 里要用真的 librosa)

    def _fake_pyin(audio, fmin=None, fmax=None, sr=None, **kw):
        f0 = np.array([180.0, 180.0, np.nan, np.nan])
        vf = np.array([True, True, False, False])        # 恰好 2 帧浊音
        vp = np.array([0.9, 0.9, 0.01, 0.01])
        return f0, vf, vp

    monkeypatch.setattr(prosody_mod, "librosa", types.SimpleNamespace(pyin=_fake_pyin))
    f = ext.extract_pitch_features(np.ones(16000, dtype=np.float32))

    assert f["pitch_trend"] is None, (
        f"2 帧浊音算不出趋势(空切片 → nan),该留空,实为 {f['pitch_trend']!r}")
    assert f["pitch_direction"] == "无法判断", (
        f"趋势没量到就不许写方向,实为 {f['pitch_direction']!r}(改前这里会写「平稳」)")
    # 同支的其余四个统计量对 2 帧**有**定义,照常有值(别把它一起写成空)
    assert f["pitch_mean"] == 180.0, f["pitch_mean"]
    assert f["pitch_std"] == 0.0, f["pitch_std"]
    assert f["pitch_p10"] == 180.0 and f["pitch_p90"] == 180.0, (f["pitch_p10"], f["pitch_p90"])


def test_three_voiced_frames_do_have_a_trend(monkeypatch):
    """反向那半:`n//3 ≥ 1`(3 帧)时趋势**有**定义,照常出值 —— 别把门槛修成"永远空"。

    红法:把门槛提到 `>= 4` ⟹ 立刻红。
    """
    import types
    prosody_mod = importlib.import_module(
        "voice_interaction.core.feature_extraction.prosody_extractor")
    ext = prosody_mod.ProsodyFeatureExtractor()

    def _fake_pyin(audio, fmin=None, fmax=None, sr=None, **kw):
        return (np.array([100.0, 150.0, 200.0, np.nan]),
                np.array([True, True, True, False]),
                np.array([0.9, 0.9, 0.9, 0.01]))

    monkeypatch.setattr(prosody_mod, "librosa", types.SimpleNamespace(pyin=_fake_pyin))
    f = ext.extract_pitch_features(np.ones(16000, dtype=np.float32))
    # 3 帧:首三分之一 = 第 1 帧(100)、末三分之一 = 最后一帧(200)⟹ 差 100 Hz
    assert f["pitch_trend"] == 100.0, f["pitch_trend"]
    assert f["pitch_direction"] == "上扬", f["pitch_direction"]


def test_a_voiced_clip_still_fills_every_pitch_statistic():
    """★ 反向那半:有浊音帧时这五个统计量**照常有值** —— 别把"留空"修成"永远空"。

    红法:把那一支的 `return` 之上去掉 `if`(或让正常分支也返回 `None`)⟹ 立刻红。
    """
    f = _extractor_features(_wav_like())          # 前 1s 静音 + 后 2s 180 Hz 正弦
    assert f["pitch_direction"] != "无法判断", f"这段有浊音帧,不该走那一支:{f}"
    for col in ("pitch_mean", "pitch_std", "pitch_trend", "pitch_p10", "pitch_p90"):
        assert f[col] is not None, f"{col} 在有浊音帧时也该有值,实为 {f[col]!r}"
    # 与这段构造对得上的量:音高在 180 Hz 一带、p10 < p90、std > 0
    assert 150 < f["pitch_mean"] < 210, f["pitch_mean"]
    assert f["pitch_p10"] < f["pitch_p90"], (f["pitch_p10"], f["pitch_p90"])
    assert f["pitch_std"] > 0, f["pitch_std"]


def test_voiced_prob_mean_is_always_a_probability():
    """`docs/superpowers/specs/2026-09-21-jingxin-feature-redesign-design.md:520` §8 验证策略第 3 条:归一化量的实测取值域必须落在物理范围内 ⟹ 概率 ∈ [0,1]。

    三条路都要过:正弦语音段(实测 0.6245)、全静音(0.0)、白噪声(0.01)。
    红法:去掉 `np.mean`(变成逐帧概率之和)⟹ 越界;或乘 100 当百分数 ⟹ 越界。
    """
    for name, audio in (("tone", _wav_like(2.0)),
                        ("silence", np.zeros(2 * 16000, dtype=np.float32)),
                        ("noise", (0.2 * np.random.default_rng(0).standard_normal(2 * 16000)
                                   ).astype(np.float32))):
        v = _extractor_features(audio)["voiced_prob_mean"]
        assert 0.0 <= v <= 1.0, f"{name}:voiced_prob 是概率,实测 {v!r} 在 [0,1] 之外"


def test_voiced_prob_and_pitch_mean_land_differently_on_all_silence(tmp_path, monkeypatch):
    """★ 全静音那一行**落盘后**长什么样:两类量、两种处置(plan:38 的正面验收)。

      · `pitch_mean` / `pitch_std` / `pitch_trend` / `pitch_p10` / `pitch_p90` → `""`(**空串**,不是 `"0.0"`);
      · `voiced_prob_mean` → `"0.0"`(**真值**,不是 `""`)。

    为什么必须在 CSV 那一层再钉一遍:两类量在提取器里就已经分开了,但**落盘**还要过
    logger 行体那一层(`.get(...)` 的缺省值:`pitch_mean` 那格写的是 `.get("pitch_mean", 0)`,
    若它给的是 `None` 就落成空串 —— 一旦有人把它改成 `or 0`,空串会**静默变回 0.0**)。
    红法:① 任一 pitch 列恢复写 0.0 → 第一组红;② `voiced_prob_mean` 特殊处理成空 → 第二句红。
    """
    voice_app = importlib.import_module("voice_interaction.api.app")
    feats = voice_app.prosody_features_from_pcm(
        _pcm_bytes(np.zeros(3 * 16000, dtype=np.float32)))
    monkeypatch.setattr(vl, "LOGS_DIR", str(tmp_path))
    log = vl.VoiceLogger(log_type="interview", session_id="20260926_120000_vvps")
    log.log_prosody(feats, question_index=0, emotion="", feedback="")
    r = list(csv.DictReader(open(log.csv_file, encoding="utf-8")))[-1]
    # ⚠️ 这里用的是**日志列名**:提取器的 `pitch_std` 经改名表(`app.py:97-100`)落成 `pitch_variation`
    for col in ("pitch_mean", "pitch_variation", "pitch_trend", "pitch_p10", "pitch_p90"):
        assert r[col] == "", f"全静音的 {col} 该留空,实为 {r[col]!r}"
    assert r["voiced_prob_mean"] == "0.0", (
        f"全静音的概率均值是**真值** 0.0,实为 {r['voiced_prob_mean']!r}")


def test_voiced_prob_travels_from_the_extractor_into_the_csv_cell(tmp_path, monkeypatch):
    """★ 端到端那一格:提取器 → 改名映射表 → logger 行体 → CSV 单元格。

    N1 的教训正是"中间那根线没人接,而每一段单独看都是绿的":提取器吐得出键、
    logger 表头里也有那一列,**两边的键名却对不上** ⟹ 列在、值是空的/0。

    红法:① 删掉 logger 行体里 `voiced_prob_mean` 那一行 → 空串;
          ② 把 logger 的键名改成别的(或把提取器的键名改掉而不同步)→ 空串;
          ③ 让 logger 对它 `.get(..., 0)` → 写成一个假的 0.0 概率。
    """
    voice_app = importlib.import_module("voice_interaction.api.app")
    feats = voice_app.prosody_features_from_pcm(_pcm_bytes(_wav_like()))
    assert "voiced_prob_mean" in feats, (
        f"活路径交出来的字典里没有这一列:{sorted(feats)}")
    v = feats["voiced_prob_mean"]
    assert 0.0 < v <= 1.0, f"这段是半静音半正弦,概率该在 (0,1]:{v!r}"

    monkeypatch.setattr(vl, "LOGS_DIR", str(tmp_path))
    log = vl.VoiceLogger(log_type="interview", session_id="20260926_120000_vvpe")
    log.log_prosody(feats, question_index=0, emotion="", feedback="")
    r = list(csv.DictReader(open(log.csv_file, encoding="utf-8")))[-1]
    assert r["voiced_prob_mean"] != "", "列在表头里,值却没落进去(键名对不上/行体缺了这一行)"
    assert float(r["voiced_prob_mean"]) == pytest.approx(v), (
        f"落盘的值与提取器算出的一致:{r['voiced_prob_mean']} vs {v}")


def test_voiced_prob_missing_stays_empty_not_zero(tmp_path, monkeypatch):
    """取不到就**留空**:0.0 是全静音时的**真值** ⟹ 它不能同时兼任"缺值"的占位。

    红法:把行体改成 `prosody_data.get("voiced_prob_mean", 0)` → 第一句立刻红。
    """
    monkeypatch.setattr(vl, "LOGS_DIR", str(tmp_path))
    log = vl.VoiceLogger(log_type="interview", session_id="20260926_120000_vvpm")
    log.log_prosody({"pitch_mean": 150.0, "duration_sec": 1.0},
                    question_index=0, emotion="", feedback="")
    r = list(csv.DictReader(open(log.csv_file, encoding="utf-8")))[-1]
    assert r["voiced_prob_mean"] == "", f"缺值写成了 {r['voiced_prob_mean']!r}"
