# tests/test_voice_instrument_upgrade.py
"""Tests for P0 Speech Instrument Upgrade:
1. Silero VAD ONNX pure-NumPy/ORT wrapper + hysteresis + RMS fallback
2. Parselmouth F0 accuracy, speedup vs librosa.pyin, HNR/Jitter/Shimmer
3. FasterWhisperConfidenceEngine aggregation + CUDA float32 safety rule
4. FunASREngine spk_name / spk_score propagation
5. jx_backfill_whisper_conf sidecar generation (non-destructive, zero PII)
"""
from __future__ import annotations

import io
import json
import sys
import time
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from report_frontend import voice_series
from tools import jx_backfill_whisper_conf
from voice_interaction.asr.funasr_engine import FunASREngine
from voice_interaction.asr.whisper_confidence_engine import FasterWhisperConfidenceEngine
from voice_interaction.core.feature_extraction import silero_vad_wrapper as svw
from voice_interaction.core.feature_extraction.prosody_extractor import ProsodyFeatureExtractor

SR = 16000


def _speech_like_burst(duration_sec: float = 1.2, f0_base: float = 150.0, sr: int = SR) -> np.ndarray:
    """Create a harmonic, FM-modulated, syllabic-envelope signal that excites Silero VAD."""
    t = np.arange(int(duration_sec * sr), dtype=np.float64) / sr
    mod = 0.65 + 0.35 * np.sin(2.0 * np.pi * 4.0 * t)
    f_inst = f0_base + 110.0 * np.sin(2.0 * np.pi * 2.5 * t)
    phase = 2.0 * np.pi * np.cumsum(f_inst) / sr
    sig = np.zeros_like(t)
    for k in range(1, 8):
        sig += (0.5 / k) * np.sin(k * phase)
    return (0.4 * mod * sig).astype(np.float32)


def _harmonic_vowel(duration_sec: float = 3.0, f0_hz: float = 150.0, sr: int = SR) -> np.ndarray:
    """Create a steady 150 Hz vowel-like harmonic waveform for F0/HNR/jitter verification."""
    t = np.arange(int(duration_sec * sr), dtype=np.float64) / sr
    sig = np.zeros_like(t)
    for k in range(1, 8):
        sig += (0.5 / k) * np.sin(2.0 * np.pi * f0_hz * k * t)
    peak = np.max(np.abs(sig)) or 1.0
    return (0.35 * sig / peak).astype(np.float32)


def _write_wav(path: Path, audio: np.ndarray, sr: int = SR) -> Path:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


# ── 1. Silero VAD ONNX wrapper ───────────────────────────────────────────────

def test_silero_vad_detects_speech_pauses_and_silence_without_torch():
    svw._reset_cached_session_for_tests()
    burst1 = _speech_like_burst(1.2, f0_base=150.0)
    burst2 = _speech_like_burst(1.2, f0_base=165.0)
    lead = np.zeros(int(0.4 * SR), dtype=np.float32)
    gap = np.zeros(int(0.6 * SR), dtype=np.float32)
    tail = np.zeros(int(0.4 * SR), dtype=np.float32)
    composite = np.concatenate([lead, burst1, gap, burst2, tail])

    res = svw.detect_speech_segments(composite, sr=SR)
    assert "torch" not in sys.modules
    assert res["vad_backend"] == "silero_vad_onnx"
    assert len(res["speech_segments"]) >= 2
    assert len(res["pause_durations_sec"]) >= 1
    assert any(p >= 0.4 for p in res["pause_durations_sec"])
    assert 0.2 <= res["speech_ratio"] <= 0.85
    assert len(res["frame_probs"]) > 0

    silence = np.zeros(int(2.0 * SR), dtype=np.float32)
    res_sil = svw.detect_speech_segments(silence, sr=SR)
    assert res_sil["vad_backend"] == "silero_vad_onnx"
    assert res_sil["speech_segments"] == []
    assert res_sil["speech_ratio"] == 0.0

    empty = svw.detect_speech_segments(np.zeros(0, dtype=np.float32), sr=SR)
    assert empty["speech_segments"] == []
    assert empty["speech_ratio"] == 0.0
    assert empty["speech_duration_sec"] == 0.0


def test_silero_vad_hysteresis_merge_and_min_speech_filtering():
    # 512 samples/frame at 16kHz -> 32 ms/frame
    # Burst 1: 10 frames (320ms >= 250ms) -> short dip (<150ms: 3 frames = 96ms) -> Burst 2: 10 frames
    # Gap: 10 frames of 0.1 (< neg_threshold 0.35, 320ms >= 150ms and >= 200ms)
    # Short blip: 4 frames (128ms < 250ms) -> dropped!
    probs = (
        [0.8] * 10
        + [0.40] * 3   # stays above neg_threshold (0.35), must stay merged
        + [0.8] * 10
        + [0.1] * 10   # real silence pause (320ms)
        + [0.8] * 4    # too short (<250ms), dropped
        + [0.1] * 8
    )
    segs = svw._segments_from_probs(
        probs,
        audio_length_samples=len(probs) * 512,
        sr=16000,
        threshold=0.5,
        neg_threshold=0.35,
        min_speech_duration_ms=250,
        min_silence_duration_ms=150,
    )
    assert len(segs) == 1
    assert segs[0] == (0, 23 * 512)


def test_silero_vad_rms_fallback_on_onnx_failure(monkeypatch):
    def _boom():
        raise RuntimeError("ONNX unavailable")

    monkeypatch.setattr(svw, "_get_onnx_session", _boom)
    burst = _speech_like_burst(1.0)
    gap = np.zeros(int(0.6 * SR), dtype=np.float32)
    sig = np.concatenate([burst, gap, burst])

    res = svw.detect_speech_segments(sig, sr=SR)
    assert res["vad_backend"] == "rms_fallback"
    assert len(res["speech_segments"]) >= 2
    assert len(res["pause_durations_sec"]) >= 1
    assert res["speech_ratio"] > 0.3


# ── 2. Parselmouth F0, Speedup, HNR/Jitter/Shimmer & Pause VAD Integration ──

def test_parselmouth_f0_accuracy_quality_and_speedup_vs_librosa(tmp_path):
    vowel = _harmonic_vowel(duration_sec=3.0, f0_hz=150.0, sr=SR)
    wav_path = _write_wav(tmp_path / "vowel_150.wav", vowel, sr=SR)

    ext = ProsodyFeatureExtractor(sample_rate=SR)
    pm_feats = ext.extract_prosody_parselmouth(vowel)
    assert pm_feats["pitch_backend"] == "parselmouth"
    assert pm_feats["pitch_mean"] is not None
    assert abs(pm_feats["pitch_mean"] - 150.0) < 2.0
    assert pm_feats["hnr_db"] is not None and pm_feats["hnr_db"] > 15.0
    assert pm_feats["jitter_local"] is not None and 0.0 <= pm_feats["jitter_local"] < 0.05
    assert pm_feats["shimmer_local"] is not None and 0.0 <= pm_feats["shimmer_local"] < 0.1

    empty_pm = ext.extract_prosody_parselmouth(np.zeros(0, dtype=np.float32))
    assert empty_pm["pitch_backend"] == "parselmouth"
    assert empty_pm["pitch_mean"] is None
    assert empty_pm["hnr_db"] is None

    # Warm up lazy Praat/librosa initialization before comparing steady-state latency.
    _ = voice_series.extract_file_parselmouth(wav_path)
    t0 = time.perf_counter()
    s_pm = voice_series.extract_file_parselmouth(wav_path)
    t_pm = time.perf_counter() - t0

    _ = voice_series.extract_file(wav_path, backend="librosa")
    t0 = time.perf_counter()
    s_lib = voice_series.extract_file(wav_path, backend="librosa")
    t_lib = time.perf_counter() - t0

    assert np.isfinite(s_pm.f0).any()
    assert abs(float(np.nanmedian(s_pm.f0)) - 150.0) < 2.0
    assert abs(float(np.nanmedian(s_lib.f0)) - 150.0) < 2.0
    assert t_pm < t_lib


def test_voice_series_auto_backend_falls_back_to_librosa(tmp_path, monkeypatch):
    vowel = _harmonic_vowel(duration_sec=1.5, f0_hz=180.0, sr=SR)
    wav_path = _write_wav(tmp_path / "vowel_180.wav", vowel, sr=SR)

    def _boom(*args, **kwargs):
        raise RuntimeError("parselmouth failed")

    monkeypatch.setattr(voice_series, "extract_file_parselmouth", _boom)
    s = voice_series.extract_file(wav_path, backend="auto")
    assert len(s.f0) > 0
    assert abs(float(np.nanmedian(s.f0)) - 180.0) < 3.0


def test_prosody_extractor_pause_silero_vs_legacy_compatibility():
    burst = _speech_like_burst(1.0)
    gap = np.zeros(int(0.5 * SR), dtype=np.float32)
    sig = np.concatenate([burst, gap, burst])

    legacy_ext = ProsodyFeatureExtractor(sample_rate=SR)
    legacy_pause = legacy_ext.extract_pause_features(sig)
    assert set(legacy_pause) == {
        "pause_duration_mean",
        "pause_duration_max",
        "pause_frequency",
        "speech_duration_sec",
        "speech_onset_sec",
    }

    silero_pause = legacy_ext.extract_pause_features(sig, use_silero_vad=True)
    assert silero_pause["vad_backend"] == "silero_vad_onnx"
    assert silero_pause["speech_duration_sec"] > 0.5
    assert silero_pause["pause_duration_max"] >= 0.35


# ── 3. FasterWhisperConfidenceEngine ─────────────────────────────────────────

def test_faster_whisper_confidence_engine_aggregation_and_cuda_float32_rule(monkeypatch):
    # Hardware rule: NEVER allow float16 on CUDA
    eng_cuda = FasterWhisperConfidenceEngine(model_size="small", device="cuda", compute_type="float16")
    assert eng_cuda.compute_type == "float32"

    eng_cuda0 = FasterWhisperConfidenceEngine(model_size="small", device="cuda:0", compute_type="float16")
    assert eng_cuda0.compute_type == "float32"

    eng_cpu = FasterWhisperConfidenceEngine(model_size="small")
    assert eng_cpu.device == "cpu"
    assert eng_cpu.compute_type == "int8"

    # Mocked faster-whisper model output
    words1 = [
        types.SimpleNamespace(word="你", probability=0.90),
        types.SimpleNamespace(word="好", probability=0.80),
        types.SimpleNamespace(word="啊", probability=0.30),
    ]
    words2 = [
        types.SimpleNamespace(word="谢", probability=0.60),
        types.SimpleNamespace(word="谢", probability=0.40),
    ]
    segments = [
        types.SimpleNamespace(text="你好啊", words=words1),
        types.SimpleNamespace(text="谢谢", words=words2),
    ]

    class _FakeModel:
        def __init__(self):
            self.calls = []

        def transcribe(self, audio, language=None, word_timestamps=False):
            self.calls.append((len(audio), language, word_timestamps))
            return iter(segments), types.SimpleNamespace(language=language)

    fake_model = _FakeModel()
    eng = FasterWhisperConfidenceEngine(model_size="small", model=fake_model)

    pcm = (np.ones(1600, dtype=np.int16) * 1000).tobytes()
    out = eng.analyze_pcm(pcm, audio_scope="qa_segment")

    assert fake_model.calls == [(1600, "zh", True)]
    assert out["asr_confidence_source"] == "word_prob_aligned_same_engine"
    assert out["engine"] == "faster-whisper"
    assert out["model_size"] == "small"
    assert out["audio_scope"] == "qa_segment"
    assert out["text"] == "你好啊谢谢"

    probs = np.array([0.90, 0.80, 0.30, 0.60, 0.40])
    summary = out["word_conf_summary"]
    assert summary["n_words"] == 5
    assert summary["median"] == pytest.approx(float(np.median(probs)), abs=1e-4)
    assert summary["p20"] == pytest.approx(float(np.percentile(probs, 20)), abs=1e-4)
    assert summary["low_word_ratio_lt_0_5"] == pytest.approx(0.4, abs=1e-4)

    empty_out = eng.analyze_pcm(b"")
    assert empty_out["text"] == ""
    assert empty_out["word_conf_summary"] == {
        "median": None,
        "p20": None,
        "low_word_ratio_lt_0_5": None,
        "n_words": 0,
    }


def test_faster_whisper_sets_hf_mirror_default(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    captured = {}

    class _DummyWhisperModel:
        def __init__(self, model_size, device=None, compute_type=None, **kw):
            captured["args"] = (model_size, device, compute_type)
            captured["hf_endpoint"] = __import__("os").environ.get("HF_ENDPOINT")

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        types.SimpleNamespace(WhisperModel=_DummyWhisperModel),
    )
    eng = FasterWhisperConfidenceEngine(model_size="tiny", device="cuda", compute_type="float16")
    eng._get_model()
    assert captured["args"] == ("tiny", "cuda", "float32")
    assert captured["hf_endpoint"] == "https://hf-mirror.com"


# ── 4. FunASREngine spk_name / spk_score propagation ─────────────────────────

def test_funasr_engine_propagates_spk_name_and_spk_score_only_when_present():
    seg_with_spk = types.SimpleNamespace(
        text="我认同这个观点",
        raw={
            "timestamp": [[0, 200]],
            "punc_array": [1],
            "spk_name": "candidate",
            "spk_score": 0.88,
        },
    )
    seg_without_spk = types.SimpleNamespace(
        text="继续补充",
        raw={
            "timestamp": [[0, 150]],
            "punc_array": [1],
        },
    )
    seg_null_spk = types.SimpleNamespace(
        text="好的",
        raw={
            "timestamp": [[0, 100]],
            "punc_array": [],
            "spk_name": "",
            "spk_score": None,
        },
    )
    res = types.SimpleNamespace(
        text="我认同这个观点继续补充好的",
        segments=[seg_with_spk, seg_without_spk, seg_null_spk],
    )
    utt = FunASREngine._to_utterance(res)
    assert utt.segments[0]["spk_name"] == "candidate"
    assert utt.segments[0]["spk_score"] == pytest.approx(0.88)
    assert "spk_name" not in utt.segments[1]
    assert "spk_score" not in utt.segments[1]
    assert "spk_name" not in utt.segments[2]
    assert "spk_score" not in utt.segments[2]


# ── 5. Backfill sidecar writes <sid>.whisper_conf.json without overwriting ──

def test_backfill_whisper_conf_writes_sidecar_only_without_pii(tmp_path, monkeypatch):
    sid = "20261123_230000_test"
    rec_dir = tmp_path / "recordings" / sid / "media" / "audio"
    rec_dir.mkdir(parents=True)
    wav_path = _write_wav(rec_dir / "0001_converted.wav", _harmonic_vowel(1.0, 160.0))

    out_dir = tmp_path / "transcripts"
    out_dir.mkdir()
    orig_json = out_dir / f"{sid}.json"
    orig_content = json.dumps({"session_id": sid, "text": "原始文本"}, ensure_ascii=False)
    orig_json.write_text(orig_content, encoding="utf-8")

    monkeypatch.setattr(
        voice_series,
        "audio_source",
        lambda s: ("answer", [(0, wav_path)]) if s == sid else ("none", []),
    )

    class _FakeModel:
        def transcribe(self, audio, language=None, word_timestamps=False):
            words = [
                types.SimpleNamespace(word="测", probability=0.92),
                types.SimpleNamespace(word="试", probability=0.45),
            ]
            return iter([types.SimpleNamespace(text="测试", words=words)]), None

    eng = FasterWhisperConfidenceEngine(model_size="tiny", model=_FakeModel())
    res = jx_backfill_whisper_conf.backfill_session(sid, eng, out_dir=out_dir)

    assert res["status"] == "ok"
    assert orig_json.read_text(encoding="utf-8") == orig_content

    sidecar = out_dir / f"{sid}.whisper_conf.json"
    assert sidecar.is_file()
    doc = json.loads(sidecar.read_text(encoding="utf-8"))
    assert doc["session_id"] == sid
    assert doc["asr_confidence_source"] == "word_prob_aligned_same_engine"
    assert doc["audio_scope"] == "qa_segment"
    assert doc["word_conf_summary"]["n_words"] == 2
    assert doc["word_conf_summary"]["low_word_ratio_lt_0_5"] == pytest.approx(0.5)
    assert "name" not in doc and "student_id" not in doc
