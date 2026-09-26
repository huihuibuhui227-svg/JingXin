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
