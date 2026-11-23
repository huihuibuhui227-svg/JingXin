# tests/test_behavior_regime.py
"""P1 无监督行为节奏分段模型与时间轴测试。"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

import redline_guard
from report_frontend import behavior_regime_model as brm
from report_frontend.data_loader import LogDataLoader

BANNED_WORDS = (
    "焦虑", "紧张", "压力", "抗压", "情绪稳定", "说谎", "诚信",
    "录用", "人格", "心理画像", "常模", "心理", "情绪", "得分", "评级", "评分",
)
FORBIDDEN_KEYS = {
    "total_score", "total_level", "score", "level", "dimension_weights",
    "normalized_score", "contribution", "status", "direction", "weight",
    "positive_factors", "negative_factors", "narrative", "simple_narrative", "percentile",
}


def _assert_clean_text(text: str) -> None:
    assert redline_guard.find_assistant_violations(text) == []
    for w in BANNED_WORDS:
        assert w not in text, f"输出文案包含禁止词 {w!r}: {text}"
    for ch in ("（", "）", "(", ")"):
        assert ch not in text, f"输出文案包含冗余括号 {ch!r}: {text}"


def _make_multi_regime_session(n_per_phase: int = 12):
    rng = np.random.default_rng(2026)
    n = n_per_phase * 4
    ts = 1700000000.0 + np.arange(n, dtype=float)

    # Phase 0: pause_thinking; Phase 1: steady_speaking; Phase 2: active_expressive; Phase 3: low_signal
    au1 = rng.uniform(0.05, 0.15, size=n)
    au25 = np.empty(n, dtype=float)
    gaze_dev = np.empty(n, dtype=float)
    head_yaw = np.empty(n, dtype=float)
    head_pitch = np.empty(n, dtype=float)
    wrist_jit = np.empty(n, dtype=float)
    torso_jit = np.empty(n, dtype=float)
    spread = np.empty(n, dtype=float)

    # Phase 0: pause_thinking
    s0, e0 = 0, n_per_phase
    au25[s0:e0] = rng.uniform(0.001, 0.003, size=n_per_phase)
    gaze_dev[s0:e0] = rng.uniform(0.18, 0.22, size=n_per_phase)
    head_yaw[s0:e0] = rng.normal(0.0, 0.003, size=n_per_phase)
    head_pitch[s0:e0] = rng.normal(0.0, 0.003, size=n_per_phase)
    wrist_jit[s0:e0] = rng.uniform(0.0002, 0.0006, size=n_per_phase)
    torso_jit[s0:e0] = rng.uniform(0.0002, 0.0005, size=n_per_phase)
    spread[s0:e0] = np.nan

    # Phase 1: steady_speaking
    s1, e1 = e0, e0 + n_per_phase
    au25[s1:e1] = 0.032 + 0.022 * np.sin(np.arange(n_per_phase) * 1.4)
    gaze_dev[s1:e1] = rng.uniform(0.16, 0.20, size=n_per_phase)
    head_yaw[s1:e1] = rng.normal(0.0, 0.008, size=n_per_phase)
    head_pitch[s1:e1] = rng.normal(0.0, 0.006, size=n_per_phase)
    wrist_jit[s1:e1] = rng.uniform(0.0006, 0.0012, size=n_per_phase)
    torso_jit[s1:e1] = rng.uniform(0.0004, 0.0009, size=n_per_phase)
    spread[s1:e1] = rng.uniform(1.5, 2.2, size=n_per_phase)

    # Phase 2: active_expressive
    s2, e2 = e1, e1 + n_per_phase
    au25[s2:e2] = 0.035 + 0.025 * np.cos(np.arange(n_per_phase) * 1.6)
    gaze_dev[s2:e2] = rng.uniform(0.32, 0.45, size=n_per_phase)
    head_yaw[s2:e2] = np.cumsum(rng.choice([-0.12, 0.12], size=n_per_phase))
    head_pitch[s2:e2] = np.cumsum(rng.choice([-0.08, 0.08], size=n_per_phase))
    wrist_jit[s2:e2] = rng.uniform(0.008, 0.015, size=n_per_phase)
    torso_jit[s2:e2] = rng.uniform(0.005, 0.010, size=n_per_phase)
    spread[s2:e2] = rng.uniform(2.5, 3.8, size=n_per_phase)

    # Phase 3: low_signal
    s3, e3 = e2, n
    au1[s3:e3] = np.nan
    au25[s3:e3] = np.nan
    gaze_dev[s3:e3] = np.nan
    head_yaw[s3:e3] = np.nan
    head_pitch[s3:e3] = np.nan
    wrist_jit[s3:e3] = rng.uniform(0.001, 0.002, size=n_per_phase)
    torso_jit[s3:e3] = rng.uniform(0.0008, 0.0015, size=n_per_phase)
    spread[s3:e3] = np.nan

    face_df = pd.DataFrame({
        "timestamp": ts,
        "au1_inner_brow_raise": au1,
        "au25_mouth_open": au25,
        "gaze_deviation": gaze_dev,
        "head_yaw": head_yaw,
        "head_pitch": head_pitch,
    })
    gesture_df = pd.DataFrame({
        "timestamp": ts + 100.0,
        "left_wrist_jitter_world": wrist_jit,
        "right_wrist_jitter_world": wrist_jit * 1.05,
        "torso_jitter": torso_jit,
        "left_hand_spread": spread,
        "right_hand_spread": spread,
    })

    # Whole-session VoiceSeries at 25 Hz (0.04 s/frame): silent in phase 0, voiced in phases 1 & 2
    frame_sec = 0.04
    n_v = int(n / frame_sec)
    f0 = np.full(n_v, np.nan, dtype=float)
    f0[int(s1 / frame_sec) : int(e2 / frame_sec)] = 185.0
    vs = [SimpleNamespace(index=None, frame_sec=frame_sec, f0=f0)]
    return {"face": face_df, "gesture": gesture_df}, vs


def test_extract_regime_windows_and_voice_alignment():
    logs, vs = _make_multi_regime_session(n_per_phase=8)  # 32 rows -> (32 - 8)//4 + 1 = 7 windows
    X, meta = brm.extract_regime_windows(logs["face"], logs["gesture"], voice_series=vs, window_size=8, stride=4)
    assert X.shape == (7, 8)
    assert len(meta) == 7
    assert meta[0]["start_sec"] == 0.0
    assert meta[0]["end_sec"] == 7.0
    # Window 0 [0,7] falls entirely in silent phase 0; window 2 [8,15] falls in voiced phase 1
    assert X[0, 7] == 0.0
    assert X[2, 7] == 1.0
    # Phase 3 (rows 24..31) has NaN face rows -> window 6 [24,31] has 0.0 face_valid_ratio
    assert X[6, 0] == 0.0

    # Per-answer or short voice_series must NOT be misaligned -> voiced_ratio stays NaN
    short_vs = [SimpleNamespace(index=None, frame_sec=0.04, f0=np.ones(50))]
    X_short, _ = brm.extract_regime_windows(logs["face"], logs["gesture"], voice_series=short_vs)
    assert np.isnan(X_short[:, 7]).all()

    ans_vs = [SimpleNamespace(index=0, frame_sec=0.04, f0=np.ones(500))]
    X_ans, _ = brm.extract_regime_windows(logs["face"], logs["gesture"], voice_series=ans_vs)
    assert np.isnan(X_ans[:, 7]).all()

    # Fallback columns (bs_jawOpen, gaze_direction_x/y, left_wrist_jitter, hand_visible_left)
    f_alt = pd.DataFrame({
        "timestamp": np.arange(12, dtype=float),
        "bs_jawOpen": np.linspace(0.01, 0.05, 12),
        "gaze_direction_x": np.full(12, 0.1),
        "gaze_direction_y": np.full(12, 0.1),
        "head_yaw": np.zeros(12),
    })
    g_alt = pd.DataFrame({
        "timestamp": np.arange(12, dtype=float),
        "left_wrist_jitter": np.full(12, 0.005),
        "hand_visible_left": np.ones(12),
    })
    X_alt, _ = brm.extract_regime_windows(f_alt, g_alt)
    assert X_alt.shape == (2, 8)
    assert np.isfinite(X_alt[:, :7]).all()


def test_insufficient_frames_returns_unavailable():
    for empty_logs in (None, {}, {"face": pd.DataFrame(), "gesture": pd.DataFrame()}, {"face": pd.DataFrame({"au25_mouth_open": [0.1] * 7})}):
        res = brm.segment_session_regimes(empty_logs, session_id=None)
        assert res["available"] is False
        assert res["advisory_only"] is True
        assert res["segments"] == []
        assert res["state_counts"] == {}
        assert res["state_ratios"] == {}
        assert res["transition_count"] == 0
        assert res["total_duration_sec"] == 0.0
        _assert_clean_text(res["summary_plain"])
        _assert_clean_text(res["disclaimer"])
        assert not (FORBIDDEN_KEYS & set(res.keys()))


def test_temporal_smoothing_and_segment_merging():
    raw = ["steady_speaking", "steady_speaking", "active_expressive", "steady_speaking", "steady_speaking"]
    smoothed = brm.smooth_state_sequence(raw, window=3)
    assert smoothed == ["steady_speaking"] * 5

    seq = ["pause_thinking", "pause_thinking", "pause_thinking", "steady_speaking", "steady_speaking"]
    meta = [
        {"start_sec": 0.0, "end_sec": 7.0},
        {"start_sec": 4.0, "end_sec": 11.0},
        {"start_sec": 8.0, "end_sec": 15.0},
        {"start_sec": 12.0, "end_sec": 19.0},
        {"start_sec": 16.0, "end_sec": 23.0},
    ]
    segs = brm.merge_state_segments(seq, meta)
    assert len(segs) == 2
    assert segs[0]["state"] == "pause_thinking"
    assert segs[1]["state"] == "steady_speaking"
    assert segs[0]["end_sec"] == segs[1]["start_sec"] == 13.5
    assert round(sum(s["duration_sec"] for s in segs), 2) == 23.0


def test_deterministic_cluster_mapping_permutation_invariant():
    # Rows represent distinct physical centroids in arbitrary order
    base = np.array([
        [0.65, 0.50, 0.50, -0.35, 0.05, -0.25, 0.0, 0.20],   # steady_speaking
        [-1.70, 0.05, 0.05, 0.05, -0.10, 0.20, 0.0, 0.00],   # low_signal
        [0.60, -1.50, -1.50, 0.30, -0.90, -0.45, 0.2, -0.50], # pause_thinking
        [-0.30, -0.15, -0.10, 1.15, 1.30, 1.35, -0.1, -0.05], # active_expressive
    ])
    perm = [2, 0, 3, 1]
    mp_a = brm.assign_cluster_states(base)
    mp_b = brm.assign_cluster_states(base[perm])
    assert set(mp_a.values()) == set(brm.STATE_ORDER)
    for orig_idx, new_idx in enumerate(perm):
        assert mp_b[orig_idx] == mp_a[new_idx]


def test_segment_session_regimes_synthetic_and_fallback(tmp_path):
    logs, _ = _make_multi_regime_session(n_per_phase=12)

    # 1. 缺模型时纯规则降级
    missing_mdl = tmp_path / "absent.joblib"
    res_rule = brm.segment_session_regimes(logs, session_id=None, model_path=missing_mdl)
    assert res_rule["available"] is True
    assert res_rule["advisory_only"] is True
    assert res_rule["model_used"] is False
    assert res_rule["model_type"] == "rule_fallback"
    assert res_rule["transition_count"] >= 2
    assert len(res_rule["segments"]) >= 3
    assert abs(sum(res_rule["state_ratios"].values()) - 1.0) <= 0.02
    _assert_clean_text(res_rule["summary_plain"])
    _assert_clean_text(res_rule["disclaimer"])

    # 2. 真实训练产出模型纯 CPU 推理
    if brm.DEFAULT_MODEL_PATH.is_file():
        # Warm cache first
        _ = brm.segment_session_regimes(logs, session_id=None)
        res_mdl = brm.segment_session_regimes(logs, session_id=None)
        assert res_mdl["available"] is True
        assert res_mdl["advisory_only"] is True
        assert res_mdl["model_used"] is True
        assert res_mdl["model_type"] == "GaussianMixture"
        assert res_mdl["inference_ms"] < 15.0
        assert res_mdl["transition_count"] >= 2
        assert abs(sum(res_mdl["state_ratios"].values()) - 1.0) <= 0.02
        # Multi-regime session should uncover multiple states including low_signal & active/pause
        assert res_mdl["state_counts"]["low_signal"] >= 1
        assert res_mdl["state_counts"]["active_expressive"] >= 1
        _assert_clean_text(res_mdl["summary_plain"])
        _assert_clean_text(res_mdl["disclaimer"])
        assert not (FORBIDDEN_KEYS & set(res_mdl.keys()))


def test_metrics_json_and_real_session_integration():
    if brm.DEFAULT_METRICS_PATH.is_file():
        metrics = json.loads(brm.DEFAULT_METRICS_PATH.read_text(encoding="utf-8"))
        assert metrics["scoring_participation"] is False
        assert metrics["val_silhouette_mean"] >= 0.15
        assert metrics["cluster_stability_ari_mean"] >= 0.75
        assert metrics["n_sessions"] >= 40
        assert metrics["n_windows"] >= 5000
        assert "session_ids" not in metrics
        _assert_clean_text(metrics["boundary_note"])

    sid = "20260929_180047_6fee"
    loader = LogDataLoader()
    groups = loader._scan_and_group_files()
    if any(f["session_id"] == sid for f in groups.get("face", [])) and brm.DEFAULT_MODEL_PATH.is_file():
        logs = loader.get_fused_latest_data(sid)
        _ = brm.segment_session_regimes(logs, session_id=sid)
        res = brm.segment_session_regimes(logs, session_id=sid)
        assert res["available"] is True
        assert res["advisory_only"] is True
        assert res["model_used"] is True
        assert len(res["segments"]) >= 3
        for i in range(len(res["segments"]) - 1):
            assert res["segments"][i + 1]["start_sec"] >= res["segments"][i]["end_sec"]
        _assert_clean_text(res["summary_plain"])
        _assert_clean_text(res["disclaimer"])
