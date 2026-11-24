# tests/test_feature_layer_v2.py
"""Tests for report_frontend.feature_layer_v2."""

from __future__ import annotations

import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from report_frontend.feature_layer_v2 import extract_feature_layer_v2
from tests.test_report_layer import BANNED

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "report_frontend" / "feature_layer_v2.py"
EXTRA_BANNED = ("心理", "情绪", "得分", "评级", "评分")


def test_empty_and_none_logs_strictly_deterministic(tmp_path: Path):
    r_empty = extract_feature_layer_v2({}, session_id=None, l0_series_dir=tmp_path)
    r_none = extract_feature_layer_v2(None, session_id=None, l0_series_dir=tmp_path)
    r_empty_df = extract_feature_layer_v2(
        {"face": pd.DataFrame(), "gesture": pd.DataFrame(), "voice_interview": pd.DataFrame()},
        session_id="missing_sid",
        l0_series_dir=tmp_path,
    )

    assert r_empty == r_none
    assert r_empty == r_empty_df
    assert r_empty["available"] is False
    assert r_empty["pose_mode"] == "unavailable"
    assert r_empty["inference_ms"] == 0.0
    assert r_empty["framework_tags"] == []
    assert r_empty["highlights"] == []
    assert r_empty["tracks"]["catch22_dynamics"]["n_streams"] == 0
    dumped = json.dumps(r_empty, allow_nan=False)
    assert "NaN" not in dumped and "Infinity" not in dumped


def test_face_3d_kinematics_and_quality_sources(tmp_path: Path):
    t = np.arange(20, dtype=float) * 0.1
    face_3d = pd.DataFrame({
        "timestamp": t,
        "head_yaw_3d": 10.0 * t,
        "head_pitch_3d": np.zeros_like(t),
        "head_roll_3d": np.zeros_like(t),
        "frame_quality_ok": [True, True, False, True] * 5,
    })

    res = extract_feature_layer_v2({"face": face_3d}, l0_series_dir=tmp_path)
    fk = res["tracks"]["face_kinematics"]

    assert res["available"] is True
    assert res["pose_mode"] == "3d_matrix"
    assert fk["pose_mode"] == "3d_matrix"
    assert fk["unit"] == "度/秒"
    assert abs(fk["head_angular_velocity_3d_p50"] - 10.0) < 1e-3
    assert abs(fk["head_angular_velocity_3d_p90"] - 10.0) < 1e-3
    assert abs(fk["frame_quality_pass_rate"] - 0.75) < 1e-6
    assert fk["quality_source"] == "frame_quality_ok"
    assert "RTMW/MediaPipe-3D" in res["framework_tags"]

    face_blur = pd.DataFrame({
        "timestamp": t,
        "head_yaw_3d": 5.0 * t,
        "head_pitch_3d": np.zeros_like(t),
        "head_roll_3d": np.zeros_like(t),
        "frame_blur_laplacian": [10.0, 15.0, 25.0, 14.9] * 5,
    })
    res_blur = extract_feature_layer_v2({"face": face_blur}, l0_series_dir=tmp_path)
    fk_blur = res_blur["tracks"]["face_kinematics"]
    assert abs(fk_blur["frame_quality_pass_rate"] - 0.5) < 1e-6
    assert fk_blur["quality_source"] == "frame_blur_laplacian"


def test_face_2d_legacy_fallback_and_dt_guards(tmp_path: Path):
    # Fewer than 3 finite 3D rows -> must fall back to 2d_legacy
    face_clamp = pd.DataFrame({
        "timestamp": [0.0, 0.01, 0.02, 0.03, 0.04],
        "head_yaw_3d": [1.0, 2.0, np.nan, np.nan, np.nan],
        "head_pitch_3d": [0.0, 0.0, np.nan, np.nan, np.nan],
        "head_roll_3d": [0.0, 0.0, np.nan, np.nan, np.nan],
        "head_yaw": [0.0, 0.05, 0.10, 0.15, 0.20],
        "head_pitch": [0.0, 0.0, 0.0, 0.0, 0.0],
        "avg_ear": [0.3, 0.3, np.nan, 0.3, 0.3],
    })
    res_clamp = extract_feature_layer_v2({"face": face_clamp}, l0_series_dir=tmp_path)
    fk_clamp = res_clamp["tracks"]["face_kinematics"]

    assert res_clamp["pose_mode"] == "2d_legacy"
    assert fk_clamp["unit"] == "归一化位移/秒"
    # dt=0.01s is clamped to 0.05s -> 0.05 / 0.05 = 1.0
    assert abs(fk_clamp["head_angular_velocity_3d_p50"] - 1.0) < 1e-4
    assert fk_clamp["quality_source"] == "finite_landmark"
    assert abs(fk_clamp["frame_quality_pass_rate"] - 1.0) < 1e-4
    assert "MediaPipe-2D姿态代理" in res_clamp["framework_tags"]

    # Non-monotonic timestamps -> fallback to 1.0s
    face_nonmono = pd.DataFrame({
        "timestamp": [0.0, 2.0, 1.0, 3.0, 4.0],
        "head_yaw": [0.0, 0.2, 0.4, 0.6, 0.8],
        "head_pitch": [0.0, 0.0, 0.0, 0.0, 0.0],
    })
    res_nonmono = extract_feature_layer_v2({"face": face_nonmono}, l0_series_dir=tmp_path)
    assert abs(res_nonmono["tracks"]["face_kinematics"]["head_angular_velocity_3d_p50"] - 0.2) < 1e-4


def test_voice_praat_and_npz_cache_no_audio_decode(tmp_path: Path):
    sid = "sess_voice_test"
    # Two equal-length voiced bursts (each 40 frames) separated by 20 unvoiced frames
    dt = 0.025
    t = np.arange(120, dtype=float) * dt
    vf = np.zeros(120, dtype=bool)
    vf[10:50] = True
    vf[70:110] = True

    f0 = np.full(120, np.nan, dtype=float)
    # 0.5 octave per voiced second = +6.0 semitones/sec across voiced frames
    voiced_t = np.arange(int(vf.sum()), dtype=float) * dt
    f0[vf] = 120.0 * (2.0 ** (0.5 * voiced_t))
    rms = np.full(120, 0.001, dtype=float)
    rms[vf] = 0.05 + 0.03 * np.sin(2.0 * np.pi * voiced_t)

    np.savez_compressed(tmp_path / f"{sid}.npz", t=t, f0=f0, rms=rms, voiced_flag=vf, sr=16000)

    voice_df = pd.DataFrame({
        "hnr_db": [12.0, 14.0, 16.0],
        "jitter_local": [0.010, 0.012, 0.014],
        "shimmer_local": [0.04, 0.05, 0.06],
    })

    res = extract_feature_layer_v2({"voice_interview": voice_df}, session_id=sid, l0_series_dir=tmp_path)
    vd = res["tracks"]["voice_dynamics"]

    assert res["available"] is True
    assert vd["voice_log_available"] is True
    assert vd["praat_features_available"] is True
    assert vd["npz_cache_available"] is True
    assert abs(vd["hnr_db_median"] - 14.0) < 1e-4
    assert abs(vd["jitter_local_median"] - 0.012) < 1e-5
    assert abs(vd["shimmer_local_median"] - 0.05) < 1e-5
    assert abs(vd["f0_slope_semitone_per_sec"] - 6.0) < 0.05
    assert abs(vd["voiced_burst_regularity"] - 1.0) < 1e-4
    assert vd["energy_modulation_cv"] > 0.2
    assert "Praat-Parselmouth" in res["framework_tags"]
    assert "Silero-VAD-ONNX" in res["framework_tags"]

    # Corrupted npz must not crash
    (tmp_path / "broken_sid.npz").write_bytes(b"not_a_valid_npz")
    res_broken = extract_feature_layer_v2({"voice_interview": voice_df}, session_id="broken_sid", l0_series_dir=tmp_path)
    assert res_broken["tracks"]["voice_dynamics"]["npz_cache_available"] is False
    assert abs(res_broken["tracks"]["voice_dynamics"]["hnr_db_median"] - 14.0) < 1e-4


def test_gesture_sparc_smoothness_and_dual_hand_continuity(tmp_path: Path):
    n = 120
    t = np.linspace(0.0, 10.0, n)
    rng = np.random.default_rng(7)

    smooth_wave = 0.05 + 0.04 * np.sin(2.0 * np.pi * 0.25 * t)
    jagged_wave = smooth_wave + 0.04 * np.sin(2.0 * np.pi * 4.5 * t) + 0.025 * rng.normal(size=n)

    g_smooth = pd.DataFrame({
        "left_wrist_jitter": smooth_wave,
        "right_wrist_jitter": smooth_wave,
        "torso_jitter": 0.5 * smooth_wave,
        "hand_visible_left": [1.0, 1.0, np.nan, 1.0] * 30,
        "hand_visible_right": [1.0, np.nan, 1.0, 1.0] * 30,
    })
    g_jagged = pd.DataFrame({
        "left_wrist_jitter": jagged_wave,
        "right_wrist_jitter": jagged_wave,
        "torso_jitter": 0.5 * jagged_wave,
        "hand_visible_left": [1.0] * n,
        "hand_visible_right": [1.0] * n,
    })

    r_smooth = extract_feature_layer_v2({"gesture": g_smooth}, l0_series_dir=tmp_path)
    r_jagged = extract_feature_layer_v2({"gesture": g_jagged}, l0_series_dir=tmp_path)

    gs_s = r_smooth["tracks"]["gesture_smoothness"]
    gs_j = r_jagged["tracks"]["gesture_smoothness"]

    assert gs_s["sparc"] <= 0.0
    assert gs_j["sparc"] <= 0.0
    assert gs_s["sparc"] > gs_j["sparc"]
    assert gs_s["log_dimensionless_jerk"] > gs_j["log_dimensionless_jerk"]
    assert gs_s["motion_smoothness_index"] > gs_j["motion_smoothness_index"]
    assert abs(gs_s["dual_hand_visible_ratio"] - 0.5) < 1e-4
    assert abs(gs_s["pose_continuity_ratio"] - 1.0) < 1e-4
    assert "SPARC运动平滑度" in r_smooth["framework_tags"]

    # Constant or short gesture input should leave sparc as None
    g_const = pd.DataFrame({"torso_jitter": [0.02] * 20})
    r_const = extract_feature_layer_v2({"gesture": g_const}, l0_series_dir=tmp_path)
    assert r_const["tracks"]["gesture_smoothness"]["sparc"] is None


def test_pycatch22_nonconstant_vs_short_and_constant(tmp_path: Path):
    sid = "sess_c22"
    n = 128
    t = np.linspace(0.0, 8.0, n)
    rng = np.random.default_rng(123)

    face_df = pd.DataFrame({
        "timestamp": t,
        "au12_smile": 0.2 + 0.15 * np.sin(2.0 * np.pi * 0.4 * t),
        "au7_eye_squeeze": 0.5 + 0.1 * np.cos(2.0 * np.pi * 0.7 * t) + 0.02 * rng.normal(size=n),
        "gaze_direction_x": 0.05 * np.sin(2.0 * np.pi * 0.3 * t) + 0.01 * rng.normal(size=n),
    })
    gesture_df = pd.DataFrame({
        "torso_jitter": 0.01 + 0.008 * np.abs(np.sin(2.0 * np.pi * 0.5 * t)),
    })

    # Also test Schema A (__meta__) npz loading
    f0 = 150.0 + 25.0 * np.sin(2.0 * np.pi * 0.5 * t)
    vp = np.full(n, 0.9, dtype=float)
    rms = 0.04 + 0.02 * np.cos(2.0 * np.pi * 0.6 * t)
    np.savez_compressed(
        tmp_path / f"{sid}.npz",
        whole__f0=f0,
        whole__vp=vp,
        whole__rms=rms,
        __meta__=np.array([f"whole|{t[1] - t[0]}|synthetic"]),
    )

    res = extract_feature_layer_v2({"face": face_df, "gesture": gesture_df}, session_id=sid, l0_series_dir=tmp_path)
    c22 = res["tracks"]["catch22_dynamics"]

    assert c22["n_streams"] == 6
    assert set(c22["streams"].keys()) == {
        "au12_smile", "au7_eye_squeeze", "gaze_direction_x", "torso_jitter", "voiced_f0", "rms"
    }
    for stream_metrics in c22["streams"].values():
        assert len(stream_metrics) == 4
        assert all(isinstance(v, float) and np.isfinite(v) for v in stream_metrics.values())
    assert c22["summary"]["autocorr_timescale_median"] is not None
    assert c22["summary"]["low_freq_power_ratio_median"] is not None
    assert c22["summary"]["persistence_stretch_median"] is not None
    assert len(c22["summary"]["catch22_streams_computed"]) == 6
    assert "pycatch22时间序列动力学" in res["framework_tags"]

    # Short & constant streams must be skipped cleanly (even when DataFrame has 30 rows with NaNs)
    short_const_face = pd.DataFrame({
        "au12_smile": [0.1] * 30,
        "au7_eye_squeeze": np.concatenate([np.linspace(0.1, 0.5, 10), np.full(20, np.nan)]),
    })
    res_skip = extract_feature_layer_v2({"face": short_const_face}, l0_series_dir=tmp_path)
    assert res_skip["tracks"]["catch22_dynamics"]["n_streams"] == 0


def test_latency_no_torch_highlights_and_banned_words(tmp_path: Path):
    sid = "sess_perf_300"
    n = 300
    t = np.arange(n, dtype=float) * 0.1
    rng = np.random.default_rng(99)

    face = pd.DataFrame({
        "timestamp": t,
        "head_yaw_3d": 8.0 * np.sin(0.3 * t),
        "head_pitch_3d": 4.0 * np.cos(0.2 * t),
        "head_roll_3d": 2.0 * np.sin(0.15 * t),
        "frame_quality_ok": np.ones(n, dtype=bool),
        "au12_smile": 0.25 + 0.15 * np.sin(0.5 * t),
        "au7_eye_squeeze": 0.6 + 0.08 * np.cos(0.4 * t),
        "gaze_direction_x": 0.04 * np.sin(0.7 * t) + 0.01 * rng.normal(size=n),
    })
    gesture = pd.DataFrame({
        "left_wrist_jitter": 0.02 + 0.01 * np.sin(0.8 * t),
        "right_wrist_jitter": 0.02 + 0.01 * np.cos(0.6 * t),
        "torso_jitter": 0.005 + 0.003 * np.abs(np.sin(0.4 * t)),
        "hand_visible_left": np.ones(n),
        "hand_visible_right": np.ones(n),
    })
    voice = pd.DataFrame({"hnr_db": [15.2, 16.0], "jitter_local": [0.011, 0.009], "shimmer_local": [0.045, 0.041]})

    n_audio = 600
    t_a = np.arange(n_audio, dtype=float) * 0.032
    vf = (np.arange(n_audio) % 60) < 40
    f0 = np.where(vf, 140.0 + 15.0 * np.sin(0.6 * t_a), np.nan)
    rms = np.where(vf, 0.05 + 0.02 * np.cos(0.9 * t_a), 0.002)
    np.savez_compressed(tmp_path / f"{sid}.npz", t=t_a, f0=f0, rms=rms, voiced_flag=vf, sr=16000)

    logs = {"face": face, "gesture": gesture, "voice_interview": voice}
    # Warm call + measured call
    _ = extract_feature_layer_v2(logs, session_id=sid, l0_series_dir=tmp_path)
    wall0 = time.perf_counter()
    res = extract_feature_layer_v2(logs, session_id=sid, l0_series_dir=tmp_path)
    wall_ms = (time.perf_counter() - wall0) * 1000.0

    assert "torch" not in sys.modules
    assert res["available"] is True
    assert res["inference_ms"] > 0.0
    assert res["inference_ms"] < 100.0
    assert wall_ms < 100.0
    assert 1 <= len(res["highlights"]) <= 8
    assert all(h["value"] is not None and np.isfinite(h["value"]) for h in res["highlights"])
    assert "（" not in res["summary_plain"] and "(" not in res["summary_plain"]
    json.dumps(res, allow_nan=False)

    # AST scan on feature_layer_v2.py string literals for all banned words
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    all_banned = tuple(BANNED) + EXTRA_BANNED
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for word in all_banned:
                if word in node.value:
                    offenders.append(f"line {node.lineno}: {word}")
    assert not offenders, f"Banned words found in feature_layer_v2.py: {offenders}"
