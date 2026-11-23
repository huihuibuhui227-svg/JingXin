"""Tests for multi-corpus research feature extraction, training benchmarks, and offline inference."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from research_lab import RESEARCH_CACHE_DIR, RESEARCH_MODELS_DIR, RESEARCH_RESULTS_DIR
from research_lab.extract_multicorpus_features import (
    DEFAULT_DATASETS_ROOT,
    build_chalearn_jobs,
    build_mit_jobs,
    build_recruitview_jobs,
    extract_acoustic_features_from_array,
    extract_visual_features_from_frames,
    rotation_matrix_to_euler_deg,
)
from research_lab.research_inference import predict_research_profile, summarize_l0_inputs
from research_lab.train_multicorpus_models import (
    CHALEARN_TARGETS,
    MIT_ALL_TARGETS,
    RECRUITVIEW_ALL_TARGETS,
    REGIME_STATE_ORDER,
    SHARED_REGIME_FEATURES,
    assign_audiovisual_cluster_states,
    clean_transcript_text,
    eval_regression,
)


def _has_finite_only(obj) -> bool:
    if isinstance(obj, float):
        return math.isfinite(obj)
    if isinstance(obj, dict):
        return all(_has_finite_only(v) for v in obj.values())
    if isinstance(obj, list):
        return all(_has_finite_only(v) for v in obj)
    return True


def test_rotation_matrix_to_euler_deg_recovers_axes():
    r_mat = Rotation.from_euler("xyz", [15.0, -22.0, 8.5], degrees=True).as_matrix()
    m4 = np.eye(4, dtype=float)
    m4[:3, :3] = r_mat
    yaw, pitch, roll = rotation_matrix_to_euler_deg(m4)
    assert math.isclose(pitch, 15.0, abs_tol=1e-4)
    assert math.isclose(yaw, -22.0, abs_tol=1e-4)
    assert math.isclose(roll, 8.5, abs_tol=1e-4)


def test_extract_acoustic_features_synthetic_and_silent():
    sr = 16000
    t = np.arange(int(1.2 * sr), dtype=np.float64) / sr
    mod = 0.65 + 0.35 * np.sin(2.0 * np.pi * 4.0 * t)
    f_inst = 150.0 + 90.0 * np.sin(2.0 * np.pi * 2.5 * t)
    phase = 2.0 * np.pi * np.cumsum(f_inst) / sr
    burst = np.zeros_like(t)
    for k in range(1, 8):
        burst += (0.5 / k) * np.sin(k * phase)
    burst = (0.4 * mod * burst).astype(np.float32)

    lead = np.zeros(int(0.4 * sr), dtype=np.float32)
    gap = np.zeros(int(0.6 * sr), dtype=np.float32)
    audio = np.concatenate([lead, burst, gap, burst]).astype(np.float32)

    feats = extract_acoustic_features_from_array(audio, sr=sr)
    assert feats["vad_backend"] in ("silero_vad_onnx", "rms_fallback")
    assert 0.2 <= feats["speech_ratio"] <= 0.9
    assert feats["pause_count"] >= 1.0
    assert 100.0 <= feats["pitch_mean"] <= 260.0
    assert np.isfinite(feats["hnr_db"])
    assert feats["rms_mean"] > 0.05

    silent_feats = extract_acoustic_features_from_array(np.zeros(sr, dtype=np.float32), sr=sr)
    assert silent_feats["vad_backend"] == "none"
    assert math.isnan(silent_feats["pitch_mean"])


def test_extract_visual_features_no_landmarker_handles_gracefully():
    dummy_frames = [np.full((64, 64, 3), 120, dtype=np.uint8) for _ in range(4)]
    feats = extract_visual_features_from_frames(dummy_frames, landmarker=None, frame_dt_sec=1.0)
    assert feats["n_sampled_frames"] == 4.0
    assert feats["n_valid_face_frames"] == 0.0
    assert feats["face_valid_ratio"] == 0.0
    assert math.isclose(feats["brightness_mean"], 120.0, abs_tol=1e-4)


def test_clean_transcript_text_and_eval_regression():
    raw = "[00:01 - 00:11] Hello world. [00:12 - 00:25]   Testing speech! "
    assert clean_transcript_text(raw) == "Hello world. Testing speech!"

    yt = np.array([[0.1, 0.8], [0.4, 0.5], [0.7, 0.2], [0.9, 0.1]], dtype=float)
    yp = yt.copy()
    res = eval_regression(yt, yp, ["t1", "t2"])
    assert math.isclose(res["macro"]["pearson_r"], 1.0, abs_tol=1e-6)
    assert math.isclose(res["macro"]["acc_1_minus_mae"], 1.0, abs_tol=1e-6)
    assert math.isclose(res["macro"]["rmse"], 0.0, abs_tol=1e-6)


def test_assign_audiovisual_cluster_states_permutation_invariant():
    # Rows correspond to: low_signal, active_expressive, pause_thinking, steady_speaking
    base = np.array([
        [-2.0, -0.5, -0.5, 0.0, -0.5, -0.5],
        [0.5, 0.3, 1.8, 1.2, 1.8, 0.4],
        [0.4, -1.2, -1.1, -0.2, -0.6, -1.4],
        [0.6, 0.7, 0.2, -0.3, 0.0, 1.0],
    ])
    perm = [2, 0, 3, 1]
    mapping = assign_audiovisual_cluster_states(base[perm], SHARED_REGIME_FEATURES)
    assert mapping[0] == "pause_thinking"
    assert mapping[1] == "low_signal"
    assert mapping[2] == "steady_speaking"
    assert mapping[3] == "active_expressive"
    assert set(mapping.values()) == set(REGIME_STATE_ORDER)


@pytest.mark.skipif(not DEFAULT_DATASETS_ROOT.exists(), reason="DATASETS directory not present")
def test_corpus_job_builders_zero_leakage():
    mit_jobs = build_mit_jobs(DEFAULT_DATASETS_ROOT, limit=20)
    assert len(mit_jobs) == 20
    assert len(mit_jobs[0].targets) == 18

    rv_jobs = build_recruitview_jobs(DEFAULT_DATASETS_ROOT, limit=60)
    assert len(rv_jobs) == 60
    assert len({j.group_id for j in rv_jobs}) >= 30

    ch_jobs = build_chalearn_jobs(DEFAULT_DATASETS_ROOT, {"train": 20, "val": 10, "test": 10})
    assert len(ch_jobs) == 40
    by_split = {sp: {j.group_id for j in ch_jobs if j.split == sp} for sp in ("train", "val", "test")}
    assert by_split["train"].isdisjoint(by_split["val"])
    assert by_split["train"].isdisjoint(by_split["test"])
    assert by_split["val"].isdisjoint(by_split["test"])


def test_benchmark_artifacts_and_models_exist_and_valid():
    json_path = RESEARCH_RESULTS_DIR / "multicorpus_benchmark_20261124.json"
    md_path = RESEARCH_RESULTS_DIR / "multicorpus_benchmark_20261124.md"
    if not json_path.is_file():
        pytest.skip("Benchmark outputs not yet generated")

    raw_text = json_path.read_text(encoding="utf-8")
    assert not re.search(r"face_au_log_\d{8}_\d{6}", raw_text), "PII/session log filename leaked in JSON"
    bench = json.loads(raw_text)
    assert bench["research_only"] is True
    assert _has_finite_only(bench), "Non-finite float found in benchmark JSON"

    for exp_key in (
        "experiment_1_mit",
        "experiment_2_recruitview",
        "experiment_3_chalearn",
        "experiment_4_regime_transfer",
    ):
        assert exp_key in bench

    assert bench["experiment_1_mit"]["n_interviews"] == 138
    assert bench["experiment_1_mit"]["n_participants"] == 69
    assert bench["experiment_1_mit"]["gpu_mlp_hardware"].get("dtype") == "torch.float32"

    assert md_path.is_file() and len(md_path.read_text(encoding="utf-8")) > 500

    for fname in (
        "mit_interview_multimodal.joblib",
        "recruitview_bigfive_multimodal.joblib",
        "chalearn_bigfive_behavioral.joblib",
        "behavior_regime_multicorpus.joblib",
    ):
        assert (RESEARCH_MODELS_DIR / fname).is_file(), f"Missing trained model {fname}"


def test_predict_research_profile_synthetic_and_graceful_empty(tmp_path):
    # Empty input with missing models directory -> graceful non-crash
    empty_res = predict_research_profile(None, None, None, "", models_dir=tmp_path)
    assert empty_res["research_only"] is True
    assert empty_res["chalearn_behavioral_projections"]["available"] is False
    assert empty_res["recruitview_multimodal_projections"]["available"] is False
    assert empty_res["mit_observable_correlates"]["available"] is False
    assert empty_res["cross_corpus_regime"]["available"] is False

    # If trained models exist, verify full end-to-end multimodal inference on synthetic session
    if not (RESEARCH_MODELS_DIR / "chalearn_bigfive_behavioral.joblib").is_file():
        pytest.skip("Trained research models not yet generated")

    rng = np.random.RandomState(7)
    n_rows = 24
    face_df = pd.DataFrame({
        "timestamp": np.arange(n_rows, dtype=float),
        "au4_frown": rng.uniform(0.3, 0.5, n_rows),
        "au6_cheek_raise": rng.uniform(0.2, 0.4, n_rows),
        "au7_eye_squeeze": rng.uniform(0.6, 0.7, n_rows),
        "au12_smile": rng.uniform(0.05, 0.2, n_rows),
        "au23_lip_compression": rng.uniform(0.05, 0.1, n_rows),
        "au25_mouth_open": rng.uniform(0.01, 0.08, n_rows),
        "avg_ear": rng.uniform(0.24, 0.33, n_rows),
        "gaze_deviation": rng.uniform(0.15, 0.25, n_rows),
        "symmetry_score": rng.uniform(0.92, 0.99, n_rows),
        "head_yaw": rng.normal(0.0, 0.03, n_rows),
        "head_pitch": rng.normal(-0.7, 0.02, n_rows),
        "bs_mouthSmileLeft": rng.uniform(0.1, 0.4, n_rows),
        "bs_mouthSmileRight": rng.uniform(0.1, 0.4, n_rows),
        "bs_jawOpen": rng.uniform(0.05, 0.2, n_rows),
        "bs_eyeBlinkLeft": rng.uniform(0.05, 0.25, n_rows),
        "bs_eyeBlinkRight": rng.uniform(0.05, 0.25, n_rows),
    })
    voice_series = {
        "frame_sec": 0.032,
        "f0": np.where(rng.rand(700) > 0.4, rng.uniform(130.0, 210.0, 700), np.nan),
        "rms": rng.uniform(0.01, 0.1, 700),
    }
    res = predict_research_profile(
        face_df,
        None,
        voice_series,
        "I led a cross-functional machine learning project and delivered clean results on schedule.",
    )
    assert res["research_only"] is True
    assert res["input_summary"]["has_face"] is True
    assert res["input_summary"]["has_voice"] is True
    assert res["input_summary"]["has_transcript"] is True

    ch = res["chalearn_behavioral_projections"]
    assert ch["available"] is True
    assert set(ch["projections"].keys()) == set(CHALEARN_TARGETS)
    assert all(0.0 <= v <= 1.0 for v in ch["projections"].values())

    rv = res["recruitview_multimodal_projections"]
    assert rv["available"] is True
    assert rv["mode"] == "multimodal_late_fusion"
    assert set(rv["projections"].keys()) == set(RECRUITVIEW_ALL_TARGETS)
    assert _has_finite_only(rv["projections"])

    mit = res["mit_observable_correlates"]
    assert mit["available"] is True
    assert mit["mode"] == "multimodal_late_fusion"
    assert set(mit["all_correlates"].keys()) == set(MIT_ALL_TARGETS)
    assert _has_finite_only(mit["all_correlates"])

    reg = res["cross_corpus_regime"]
    assert reg["available"] is True
    assert reg["n_windows"] == 5
    assert math.isclose(sum(reg["state_ratios"].values()), 1.0, abs_tol=1e-2)
