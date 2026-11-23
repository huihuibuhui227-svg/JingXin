"""Wave 1 · Live Voice & Gesture Capture Layer Wiring 回归与契约测试。"""
from __future__ import annotations

import asyncio
import csv
import importlib
import types

import cv2
import numpy as np
import pytest

import media_retention
from gesture_analysis.core.rtmpose_backend import (
    RTMWHandAdapter,
    RTMWPoseAdapter,
    RTMWWholeBodyEngine,
)

gesture_app = importlib.import_module("gesture_analysis.api.app")
gesture_logger_mod = importlib.import_module("gesture_analysis.utils.logger")
voice_app = importlib.import_module("voice_interaction.api.app")
voice_logger_mod = importlib.import_module("voice_interaction.utils.logger")


class _FakeUpload:
    def __init__(self, data: bytes, filename: str = "frame.jpg", content_type: str = "image/jpeg"):
        self._data = data
        self.filename = filename
        self.content_type = content_type

    async def read(self) -> bytes:
        return self._data


class _FakeRequest:
    def __init__(self, fields=None):
        self._fields = {} if fields is None else fields

    async def form(self):
        return dict(self._fields)


def _jpeg_bytes() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.zeros((64, 64, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def _make_coco133() -> tuple[np.ndarray, np.ndarray]:
    kpts = np.zeros((133, 2), dtype=float)
    scores = np.full((133,), 0.9, dtype=float)
    body = {
        0: (32.0, 12.0),
        5: (42.0, 24.0),
        6: (22.0, 24.0),
        7: (46.0, 34.0),
        8: (18.0, 34.0),
        9: (48.0, 40.0),
        10: (16.0, 40.0),
        11: (40.0, 48.0),
        12: (24.0, 48.0),
    }
    for idx, (x, y) in body.items():
        kpts[idx] = [x, y]
    for start, bx in ((91, 48.0), (112, 16.0)):
        for i in range(21):
            kpts[start + i] = [bx + (i % 5) * 0.5, 40.0 - (i // 5) * 1.0]
    return kpts, scores


def _speech_like_pcm(sr: int = 16000) -> bytes:
    """生成含停顿、双音高谐波元音与调频语音突发的 16k/16bit PCM（同时激发 pyin 与 Silero VAD）。"""
    def _vowel(dur: float, f0: float) -> np.ndarray:
        t = np.arange(int(dur * sr), dtype=np.float64) / sr
        sig = np.zeros_like(t)
        for k in range(1, 8):
            sig += (0.5 / k) * np.sin(2.0 * np.pi * f0 * k * t)
        peak = np.max(np.abs(sig)) or 1.0
        return (0.35 * sig / peak).astype(np.float32)

    def _burst(dur: float, f0: float) -> np.ndarray:
        t = np.arange(int(dur * sr), dtype=np.float64) / sr
        mod = 0.65 + 0.35 * np.sin(2.0 * np.pi * 4.0 * t)
        f_inst = f0 + 110.0 * np.sin(2.0 * np.pi * 2.5 * t)
        phase = 2.0 * np.pi * np.cumsum(f_inst) / sr
        sig = np.zeros_like(t)
        for k in range(1, 8):
            sig += (0.5 / k) * np.sin(k * phase)
        return (0.4 * mod * sig).astype(np.float32)

    lead = np.zeros(int(0.3 * sr), dtype=np.float32)
    gap1 = np.zeros(int(0.3 * sr), dtype=np.float32)
    gap2 = np.zeros(int(0.4 * sr), dtype=np.float32)
    tail = np.zeros(int(0.3 * sr), dtype=np.float32)
    wave = np.concatenate(
        [lead, _vowel(0.7, 150.0), gap1, _vowel(0.7, 195.0), gap2, _burst(1.5, 165.0), tail]
    )
    return (np.clip(wave, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


@pytest.fixture
def isolated_gesture(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path / "rec"))
    monkeypatch.setenv("JINGXIN_RETAIN_MEDIA", "0")
    media_retention._reset_for_tests()
    monkeypatch.setattr(gesture_app, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(gesture_logger_mod, "LOGS_DIR", str(tmp_path / "logs"))

    gesture_app.session_analyzers.clear()
    gesture_app.session_loggers.clear()
    gesture_app.session_clocks.clear()
    gesture_app.detectors.clear()
    gesture_app.detectors_backend.clear()
    yield tmp_path
    gesture_app.session_analyzers.clear()
    gesture_app.session_loggers.clear()
    gesture_app.session_clocks.clear()
    gesture_app.detectors.clear()
    gesture_app.detectors_backend.clear()


# ── Gesture Backend Wiring Tests ─────────────────────────────────────────────


def test_gesture_mediapipe_backend_and_analyze_response(isolated_gesture, monkeypatch):
    import gesture_analysis.core.detectors as det

    class _FakeMP:
        def __init__(self, *a, **k):
            self.closed = False

        def detect_with_handedness(self, _rgb, _ts):
            return [], []

        def detect_with_world(self, _rgb, _ts):
            return None, None

        def reset(self):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setattr(det, "HandDetector", _FakeMP)
    monkeypatch.setattr(det, "PoseDetector", _FakeMP)
    monkeypatch.setenv("JX_GESTURE_BACKEND", "mediapipe")

    sid = "20261124_010000_mp01"
    resp = asyncio.run(
        gesture_app.analyze_image(
            request=_FakeRequest(),
            file=_FakeUpload(_jpeg_bytes()),
            session_id=sid,
        )
    )
    assert resp["status"] == "success"
    assert resp["session_id"] == sid
    assert resp["detector_backend"] == "mediapipe"
    assert gesture_app.detectors_backend[sid] == "mediapipe"
    assert "result" in resp and "emotion" in resp["result"]

    health = asyncio.run(gesture_app.health_check())
    assert health["backend"] == "mediapipe"
    assert health["supported_backends"] == ["mediapipe", "rtmw"]


def test_gesture_rtmw_backend_wiring_with_mock_and_cleanup_on_reset(isolated_gesture, monkeypatch):
    import gesture_analysis.core.detectors as det

    kpts, scores = _make_coco133()

    class _FakeWholebody:
        def __init__(self):
            self.calls = 0
            self.closed = False

        def __call__(self, _bgr):
            self.calls += 1
            return kpts[None, ...], scores[None, ...]

        def close(self):
            self.closed = True

    wb = _FakeWholebody()

    def _fake_create_detectors(*args, **kwargs):
        assert kwargs.get("backend") == "rtmw"
        eng = RTMWWholeBodyEngine(mode="lightweight", wholebody_factory=lambda **_kw: wb)
        eng.ensure_ready()
        return RTMWHandAdapter(eng), RTMWPoseAdapter(eng)

    monkeypatch.setattr(det, "create_gesture_detectors", _fake_create_detectors)
    monkeypatch.setenv("JX_GESTURE_BACKEND", "rtmw")

    sid = "20261124_010000_rt01"
    resp = asyncio.run(
        gesture_app.analyze_image(
            request=_FakeRequest(),
            file=_FakeUpload(_jpeg_bytes()),
            session_id=sid,
        )
    )
    assert resp["status"] == "success"
    assert resp["detector_backend"] == "rtmw"
    assert gesture_app.detectors_backend[sid] == "rtmw"
    assert resp["result"]["detected_hands"] == 2
    assert wb.calls == 1

    health = asyncio.run(gesture_app.health_check())
    assert health["backend"] == "rtmw"

    # /reset?session_id=... 清理 detectors 与 detectors_backend 并后台关闭句柄
    reset_out = asyncio.run(gesture_app.reset_analyzers(session_id=sid))
    assert reset_out["status"] == "success"
    assert sid not in gesture_app.detectors
    assert sid not in gesture_app.detectors_backend
    assert sid not in gesture_app.session_analyzers


def test_gesture_rtmw_fallback_and_ttl_expiry_cleanup(isolated_gesture, monkeypatch):
    import time
    import gesture_analysis.core.detectors as det

    class _FakeMPFallback:
        def __init__(self, *a, **k):
            self.closed = False

        def detect_with_handedness(self, _rgb, _ts):
            return [], []

        def detect_with_world(self, _rgb, _ts):
            return None, None

        def reset(self):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setattr(det, "HandDetector", _FakeMPFallback)
    monkeypatch.setattr(det, "PoseDetector", _FakeMPFallback)
    # 故意让 wholebody_factory 失败，触发 create_gesture_detectors 回退到 MediaPipe
    orig_create = det.create_gesture_detectors

    def _create_with_broken_rtmw(*args, **kwargs):
        kwargs["wholebody_factory"] = lambda **_kw: (_ for _ in ()).throw(RuntimeError("no weights"))
        kwargs["factory"] = lambda *_a, **_k: _FakeMPFallback()
        return orig_create(*args, **kwargs)

    monkeypatch.setattr(det, "create_gesture_detectors", _create_with_broken_rtmw)
    monkeypatch.setenv("JX_GESTURE_BACKEND", "rtmw")

    sid_old = "20261124_010000_fb01"
    gesture_app.get_or_create_analyzers(sid_old)
    dets_old = gesture_app.get_or_create_detectors(sid_old)
    assert gesture_app.detectors_backend[sid_old] == "mediapipe"

    # 推到 TTL 之外，创建新会话触发过期回收，验证 detectors_backend 一并被清
    analyzers_old, _ = gesture_app.session_analyzers[sid_old]
    gesture_app.session_analyzers[sid_old] = (
        analyzers_old,
        time.time() - gesture_app.SESSION_TIMEOUT - 5,
    )
    sid_new = "20261124_010000_fb02"
    gesture_app.get_or_create_analyzers(sid_new)
    assert sid_old not in gesture_app.detectors
    assert sid_old not in gesture_app.detectors_backend
    assert dets_old["hands"] is not None

    # 无参 /reset 清空所有表
    asyncio.run(gesture_app.reset_analyzers(session_id=None))
    assert gesture_app.detectors == {}
    assert gesture_app.detectors_backend == {}


# ── Voice Backend Wiring Tests ───────────────────────────────────────────────


def test_resolve_live_voice_backends_env_parsing(monkeypatch):
    monkeypatch.delenv("JX_LIVE_PITCH_BACKEND", raising=False)
    monkeypatch.delenv("JX_LIVE_USE_SILERO_VAD", raising=False)
    assert voice_app._resolve_live_voice_backends() == ("librosa", False)

    monkeypatch.setenv("JX_LIVE_PITCH_BACKEND", "Parselmouth")
    monkeypatch.setenv("JX_LIVE_USE_SILERO_VAD", "1")
    assert voice_app._resolve_live_voice_backends() == ("parselmouth", True)

    monkeypatch.setenv("JX_LIVE_USE_SILERO_VAD", "true")
    assert voice_app._resolve_live_voice_backends() == ("parselmouth", True)

    monkeypatch.setenv("JX_LIVE_PITCH_BACKEND", "unknown_backend")
    monkeypatch.setenv("JX_LIVE_USE_SILERO_VAD", "0")
    assert voice_app._resolve_live_voice_backends() == ("librosa", False)


def test_voice_health_reports_live_backends_cleanly(monkeypatch):
    monkeypatch.setattr(voice_app, "_tcp_probe", lambda h, p, timeout=1.0: True)

    monkeypatch.delenv("JX_LIVE_PITCH_BACKEND", raising=False)
    monkeypatch.delenv("JX_LIVE_USE_SILERO_VAD", raising=False)
    h_default = asyncio.run(voice_app.health_check())
    assert h_default["status"] == "healthy"
    assert h_default["asr_reachable"] is True
    assert "pitch_backend" in h_default
    assert "vad_backend" in h_default
    assert "whisper_conf_ready" in h_default
    assert h_default["live_pitch_backend"] == "librosa"
    assert h_default["live_vad_backend"] == "rms_threshold"

    monkeypatch.setenv("JX_LIVE_PITCH_BACKEND", "parselmouth")
    monkeypatch.setenv("JX_LIVE_USE_SILERO_VAD", "1")
    h_upgraded = asyncio.run(voice_app.health_check())
    assert h_upgraded["live_pitch_backend"] == "parselmouth"
    assert h_upgraded["live_vad_backend"] == "silero_vad_onnx"


def test_voice_prosody_from_pcm_upgraded_vs_default_and_csv_compat(tmp_path, monkeypatch):
    pcm = _speech_like_pcm()

    # 1. 默认路径（librosa + RMS）保留原行为，且调用模块级 prosody_extractor
    monkeypatch.delenv("JX_LIVE_PITCH_BACKEND", raising=False)
    monkeypatch.delenv("JX_LIVE_USE_SILERO_VAD", raising=False)
    default_feats = voice_app.prosody_features_from_pcm(pcm)
    assert default_feats["pitch_mean"] is not None
    assert "pitch_variation" in default_feats
    assert "energy_variation" in default_feats
    assert "voiced_prob_mean" in default_feats
    assert "pitch_std" not in default_feats
    assert "energy_std" not in default_feats
    assert "hnr_db" not in default_feats
    assert "vad_backend" not in default_feats

    # 2. 升级路径（parselmouth + silero_vad_onnx）保留改名映射并附加声学质量量
    monkeypatch.setenv("JX_LIVE_PITCH_BACKEND", "parselmouth")
    monkeypatch.setenv("JX_LIVE_USE_SILERO_VAD", "1")
    upgraded_feats = voice_app.prosody_features_from_pcm(pcm)
    assert upgraded_feats["pitch_backend"] == "parselmouth"
    assert upgraded_feats["vad_backend"] == "silero_vad_onnx"
    assert upgraded_feats["pitch_mean"] is not None and 100.0 <= upgraded_feats["pitch_mean"] <= 250.0
    assert upgraded_feats["pitch_variation"] is not None and upgraded_feats["pitch_variation"] > 0.0
    assert upgraded_feats["energy_variation"] > 0.0
    assert 0.0 < upgraded_feats["voiced_prob_mean"] <= 1.0
    assert upgraded_feats["hnr_db"] is not None
    assert upgraded_feats["jitter_local"] is not None
    assert upgraded_feats["shimmer_local"] is not None
    assert upgraded_feats["speech_duration_sec"] > 0.5
    assert "pitch_std" not in upgraded_feats
    assert "energy_std" not in upgraded_feats

    # 3. 写入 VoiceLogger 32 列 CSV 契约时过滤附加键，无任何列错位
    monkeypatch.setattr(voice_logger_mod, "LOGS_DIR", str(tmp_path))
    vlog = voice_logger_mod.VoiceLogger(
        log_type="interview", log_dir=str(tmp_path), session_id="20261124_010000_v201"
    )
    vlog.log_prosody(upgraded_feats, question_index=0, emotion="", feedback="")
    with open(vlog.csv_file, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert list(rows[0].keys()) == vlog.fieldnames
    assert float(rows[0]["pitch_mean"]) == pytest.approx(upgraded_feats["pitch_mean"])

    # 4. 空音频与全静音在升级路径下仍严格遵循留空 / 0.0 契约
    assert voice_app.prosody_features_from_pcm(b"") == {}
    silence_pcm = np.zeros(3 * 16000, dtype=np.int16).tobytes()
    sil_feats = voice_app.prosody_features_from_pcm(silence_pcm)
    for col in ("pitch_mean", "pitch_variation", "pitch_trend", "pitch_p10", "pitch_p90"):
        assert sil_feats[col] is None
    assert sil_feats["voiced_prob_mean"] == 0.0
    assert sil_feats["speech_duration_sec"] == 0.0
    assert sil_feats["pause_frequency"] is None


def test_gesture_auto_backend_routes_according_to_cache(isolated_gesture, monkeypatch):
    import gesture_analysis.core.detectors as det
    import gesture_analysis.core.rtmpose_backend as rtb

    kpts, scores = _make_coco133()

    class _FakeMP:
        def __init__(self, *a, **k):
            pass

        def close(self):
            pass

    class _FakeWB:
        def __call__(self, _bgr):
            return kpts[None, ...], scores[None, ...]

    monkeypatch.setattr(det, "HandDetector", _FakeMP)
    monkeypatch.setattr(det, "PoseDetector", _FakeMP)
    monkeypatch.setenv("JX_GESTURE_BACKEND", "auto")

    # 无缓存时走 MediaPipe
    monkeypatch.setattr(rtb, "has_cached_rtmw_weights", lambda _m=None: False)
    orig_create = det.create_gesture_detectors
    monkeypatch.setattr(
        det,
        "create_gesture_detectors",
        lambda *a, **kw: orig_create(*a, **{**kw, "factory": lambda *_a, **_k: _FakeMP()}),
    )
    sid_uncached = "20261124_010000_au01"
    gesture_app.get_or_create_detectors(sid_uncached)
    assert gesture_app.detectors_backend[sid_uncached] == "mediapipe"

    # 有缓存（注入 wholebody_factory）时走 RTMW
    monkeypatch.setattr(
        det,
        "create_gesture_detectors",
        lambda *a, **kw: orig_create(*a, **{**kw, "wholebody_factory": lambda **_k: _FakeWB()}),
    )
    sid_cached = "20261124_010000_au02"
    gesture_app.get_or_create_detectors(sid_cached)
    assert gesture_app.detectors_backend[sid_cached] == "rtmw"


def test_voice_parselmouth_import_error_falls_back_to_librosa(monkeypatch):
    import voice_interaction.core.feature_extraction.prosody_extractor as pm_mod

    orig_extract_all = pm_mod.ProsodyFeatureExtractor.extract_all_features

    def _boom_on_parselmouth(self, audio):
        if self.pitch_backend == "parselmouth":
            raise ImportError("parselmouth missing")
        return orig_extract_all(self, audio)

    monkeypatch.setattr(pm_mod.ProsodyFeatureExtractor, "extract_all_features", _boom_on_parselmouth)
    monkeypatch.setattr(
        voice_app.prosody_extractor,
        "extract_prosody_parselmouth",
        lambda _audio: (_ for _ in ()).throw(ImportError("parselmouth missing")),
    )
    monkeypatch.setenv("JX_LIVE_PITCH_BACKEND", "parselmouth")
    monkeypatch.setenv("JX_LIVE_USE_SILERO_VAD", "1")

    feats = voice_app.prosody_features_from_pcm(_speech_like_pcm())
    assert feats["pitch_mean"] is not None
    assert feats["vad_backend"] == "silero_vad_onnx"
    assert "hnr_db" not in feats

