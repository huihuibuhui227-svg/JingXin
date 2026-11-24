# tests/test_referee_ml.py
"""Layer 3 ML 行为阶段切分与证据核验门控回归测试。"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

import redline_guard
from report_frontend import referee_ml_engine as rme
from report_frontend.data_loader import LogDataLoader

BANNED_WORDS = (
    "焦虑", "紧张", "压力", "抗压", "情绪稳定", "说谎", "诚信",
    "录用", "人格", "心理画像", "常模", "心理", "情绪", "得分", "评级", "评分",
)

FORBIDDEN_TOP_OR_NESTED_KEYS = {
    "score", "total_score", "total_level", "rating", "grade",
    "percentile", "rank", "dimension_weights", "normalized_score",
}


def _assert_clean_copy(text: str, *, allow_method_parens: bool = False) -> None:
    assert isinstance(text, str) and text.strip()
    assert redline_guard.find_assistant_violations(text) == []
    for word in BANNED_WORDS:
        assert word not in text, f"输出文案包含禁止词 {word!r}: {text}"
    if not allow_method_parens:
        for ch in ("（", "）", "(", ")"):
            assert ch not in text, f"输出文案包含冗余括号 {ch!r}: {text}"


def _assert_no_forbidden_keys(obj) -> None:
    if isinstance(obj, dict):
        hit = FORBIDDEN_TOP_OR_NESTED_KEYS & set(obj.keys())
        assert not hit, f"发现禁止评分键: {hit}"
        for v in obj.values():
            _assert_no_forbidden_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            _assert_no_forbidden_keys(item)


def _make_multi_stage_logs() -> tuple[dict[str, pd.DataFrame], list]:
    """构造含早期停顿建立期 -> 平稳展开期 -> 高互动活跃期 -> 采集受限期的 80 帧合成会话。"""
    rng = np.random.default_rng(20261124)
    n1, n2, n3, n4 = 12, 28, 24, 16
    n = n1 + n2 + n3 + n4
    ts = 1700000000.0 + np.arange(n, dtype=float)

    au1 = rng.uniform(0.05, 0.15, size=n)
    au12 = rng.uniform(0.05, 0.25, size=n)
    au4 = rng.uniform(0.03, 0.18, size=n)
    avg_ear = rng.uniform(0.24, 0.30, size=n)
    au25 = np.empty(n, dtype=float)
    gaze_dev = np.empty(n, dtype=float)
    head_yaw = np.empty(n, dtype=float)
    head_pitch = np.empty(n, dtype=float)
    wrist_jit = np.empty(n, dtype=float)
    torso_jit = np.empty(n, dtype=float)
    shoulder_w = np.full(n, 0.35, dtype=float)
    spread = np.empty(n, dtype=float)

    # Phase 1: 早期停顿组织（适应建立）
    s0, e0 = 0, n1
    au25[s0:e0] = rng.uniform(0.001, 0.003, size=n1)
    gaze_dev[s0:e0] = rng.uniform(0.18, 0.22, size=n1)
    head_yaw[s0:e0] = rng.normal(0.0, 0.003, size=n1)
    head_pitch[s0:e0] = rng.normal(0.0, 0.003, size=n1)
    wrist_jit[s0:e0] = rng.uniform(0.0002, 0.0006, size=n1)
    torso_jit[s0:e0] = rng.uniform(0.0002, 0.0005, size=n1)
    spread[s0:e0] = 0.8

    # Phase 2: 平稳展开期
    s1, e1 = e0, e0 + n2
    au25[s1:e1] = 0.032 + 0.020 * np.sin(np.arange(n2) * 1.4)
    gaze_dev[s1:e1] = rng.uniform(0.14, 0.19, size=n2)
    head_yaw[s1:e1] = rng.normal(0.0, 0.007, size=n2)
    head_pitch[s1:e1] = rng.normal(0.0, 0.005, size=n2)
    wrist_jit[s1:e1] = rng.uniform(0.0006, 0.0012, size=n2)
    torso_jit[s1:e1] = rng.uniform(0.0004, 0.0009, size=n2)
    spread[s1:e1] = rng.uniform(1.5, 2.2, size=n2)

    # Phase 3: 高互动活跃期
    s2, e2 = e1, e1 + n3
    au25[s2:e2] = 0.040 + 0.025 * np.cos(np.arange(n3) * 1.6)
    gaze_dev[s2:e2] = rng.uniform(0.32, 0.45, size=n3)
    head_yaw[s2:e2] = np.cumsum(rng.choice([-0.11, 0.11], size=n3))
    head_pitch[s2:e2] = np.cumsum(rng.choice([-0.08, 0.08], size=n3))
    wrist_jit[s2:e2] = rng.uniform(0.008, 0.015, size=n3)
    torso_jit[s2:e2] = rng.uniform(0.005, 0.010, size=n3)
    spread[s2:e2] = rng.uniform(2.6, 3.8, size=n3)

    # Phase 4: 采集受限期
    s3, e3 = e2, n
    au1[s3:e3] = np.nan
    au12[s3:e3] = np.nan
    au4[s3:e3] = np.nan
    avg_ear[s3:e3] = np.nan
    au25[s3:e3] = np.nan
    gaze_dev[s3:e3] = np.nan
    head_yaw[s3:e3] = np.nan
    head_pitch[s3:e3] = np.nan
    wrist_jit[s3:e3] = rng.uniform(0.001, 0.002, size=n4)
    torso_jit[s3:e3] = rng.uniform(0.0008, 0.0015, size=n4)
    spread[s3:e3] = np.nan

    face_df = pd.DataFrame({
        "timestamp": ts,
        "au1_inner_brow_raise": au1,
        "au4_frown": au4,
        "au12_smile": au12,
        "avg_ear": avg_ear,
        "au25_mouth_open": au25,
        "gaze_deviation": gaze_dev,
        "head_yaw": head_yaw,
        "head_pitch": head_pitch,
    })
    gesture_df = pd.DataFrame({
        "timestamp": ts,
        "left_wrist_jitter_world": wrist_jit,
        "right_wrist_jitter_world": wrist_jit * 1.05,
        "torso_jitter": torso_jit,
        "shoulder_width": shoulder_w,
        "left_hand_spread": spread,
        "right_hand_spread": spread,
    })
    voice_df = pd.DataFrame({
        "timestamp": [ts[15], ts[40]],
        "duration_sec": [32.0, 28.0],
        "speech_duration_sec": [20.0, 18.0],
        "n_chars": [110, 95],
        "transcript": ["我们首先分析实验变量和控制组之间的差异。", "随后对比模型在验证集上的收敛表现。"],
    })
    logs = {"face": face_df, "gesture": gesture_df, "voice_interview": voice_df}
    frame_sec = 0.04
    f0 = np.full(int(n / frame_sec), np.nan, dtype=float)
    f0[int(s1 / frame_sec) : int(e2 / frame_sec)] = 175.0
    vs = [SimpleNamespace(index=None, frame_sec=frame_sec, f0=f0)]
    return logs, vs


def test_empty_logs_strictly_deterministic_and_zero_latency():
    r1 = rme.evaluate_referee_ml({}, session_id=None)
    r2 = rme.evaluate_referee_ml({}, session_id=None)
    r3 = rme.evaluate_referee_ml(None, session_id=None)
    r4 = rme.evaluate_referee_ml({"face": pd.DataFrame(), "gesture": pd.DataFrame()}, session_id=None)

    assert r1 == r2 == r3 == r4
    assert r1["available"] is False
    assert r1["advisory_only"] is True
    assert r1["inference_ms"] == 0.0
    assert r1["evidence_gate"]["stage_code"] == "unavailable"
    assert r1["evidence_gate"]["conformal_set"] == []
    assert r1["evidence_gate"]["uncertainty_escalated"] is False
    assert r1["stage_segmentation"]["available"] is False
    assert r1["stage_segmentation"]["macro_stages"] == []
    assert r1["stage_segmentation"]["change_points"] == []
    _assert_no_forbidden_keys(r1)
    _assert_clean_copy(r1["non_scoring_guarantee"])
    _assert_clean_copy(r1["disclaimer"])
    _assert_clean_copy(r1["evidence_gate"]["summary_plain"])
    _assert_clean_copy(r1["stage_segmentation"]["summary_plain"])
    json.dumps(r1, allow_nan=False)


def test_rule_fallback_when_models_absent(tmp_path):
    logs, _ = _make_multi_stage_logs()
    res = rme.evaluate_referee_ml(logs, session_id=None, model_dir=tmp_path)
    assert res["available"] is True
    assert res["advisory_only"] is True
    assert res["evidence_gate"]["stage_code"] in rme.STAGE_ORDER
    assert len(res["evidence_gate"]["attributions"]) == len(rme.EVIDENCE_FEATURE_NAMES)
    assert res["stage_segmentation"]["available"] is True
    assert len(res["stage_segmentation"]["macro_stages"]) >= 2
    _assert_no_forbidden_keys(res)
    _assert_clean_copy(res["evidence_gate"]["summary_plain"])
    _assert_clean_copy(res["stage_segmentation"]["summary_plain"])
    json.dumps(res, allow_nan=False)


def test_trained_models_artifacts_and_metrics_integrity():
    assert rme.DEFAULT_HMM_PATH.is_file()
    assert rme.DEFAULT_CONFORMAL_PATH.is_file()
    assert rme.DEFAULT_METRICS_PATH.is_file()

    hmm_kb = rme.DEFAULT_HMM_PATH.stat().st_size / 1024.0
    gate_kb = rme.DEFAULT_CONFORMAL_PATH.stat().st_size / 1024.0
    assert 2.0 <= hmm_kb <= 25.0, f"HMM 模型体积异常: {hmm_kb:.2f} KB"
    assert 10.0 <= gate_kb <= 60.0, f"Conformal Gate 模型体积异常: {gate_kb:.2f} KB"

    metrics = json.loads(rme.DEFAULT_METRICS_PATH.read_text(encoding="utf-8"))
    assert metrics["scoring_participation"] is False
    assert "session_ids" not in metrics
    assert metrics["dataset"]["n_real_sessions"] == 47
    assert metrics["dataset"]["n_real_windows"] == 5666

    seg_m = metrics["stage_segmentation"]
    assert seg_m["flicker_reduction_pct"] >= 20.0
    assert seg_m["hmm_mean_dwell_sec"] > seg_m["gmm_mean_dwell_sec"]
    assert seg_m["cross_fold_viterbi_ari_mean"] >= 0.85

    gate_m = metrics["evidence_gate"]
    assert gate_m["alpha"] == 0.10
    assert gate_m["target_coverage"] == 0.90
    assert gate_m["cv_accuracy"] >= 0.85
    assert gate_m["cv_macro_f1"] >= 0.85
    assert gate_m["empirical_conformal_coverage"] >= 0.90
    assert gate_m["ood_auroc_mean"] >= 0.95
    _assert_clean_copy(metrics["boundary_note"])
    _assert_clean_copy(metrics["non_scoring_guarantee"])


def test_evidence_gate_conformal_ood_and_attributions():
    logs, _ = _make_multi_stage_logs()
    # 1. 干净充分的多模态会话
    clean_fv2 = {
        "measured_ratio": 0.98,
        "valid_modality_count_norm": 1.0,
        "signal_usable_ratio": 0.96,
        "cross_modal_overlap_ratio": 0.90,
        "face_valid_frame_ratio": 0.97,
        "audio_speech_coverage": 0.56,
        "gesture_continuity_ratio": 0.98,
        "dynamics_stability_score": 0.88,
    }
    res_clean = rme.evaluate_referee_ml(logs, session_id=None, feature_v2=clean_fv2)
    eg_c = res_clean["evidence_gate"]
    assert eg_c["stage_code"] == "stage_1_verified"
    assert eg_c["stage_label"] == "阶段 I · 证据充分可核验"
    assert "stage_1_verified" in eg_c["conformal_set"]
    assert eg_c["ood_anomaly_flag"] is False
    assert abs(sum(eg_c["stage_probs"].values()) - 1.0) <= 0.01

    # 归因严格按绝对贡献降序，方向标签与正负号严格一致
    attrs = eg_c["attributions"]
    assert len(attrs) == 8
    mags = [abs(a["contribution"]) for a in attrs]
    assert mags == sorted(mags, reverse=True)
    for a in attrs:
        _assert_clean_copy(a["label"])
        assert a["direction"] == ("正向支撑" if a["contribution"] >= 0.0 else "待补强")

    # 2. 物理矛盾的 OOD 异常会话 -> 必须触发 ood_anomaly_flag 与 uncertainty_escalated
    ood_fv2 = {
        "measured_ratio": 0.98,
        "valid_modality_count_norm": 1.0,
        "signal_usable_ratio": 0.02,
        "cross_modal_overlap_ratio": 0.01,
        "face_valid_frame_ratio": 0.01,
        "audio_speech_coverage": 0.01,
        "gesture_continuity_ratio": 0.01,
        "dynamics_stability_score": 0.05,
    }
    res_ood = rme.evaluate_referee_ml(logs, session_id=None, feature_v2=ood_fv2)
    eg_o = res_ood["evidence_gate"]
    assert eg_o["ood_anomaly_flag"] is True
    assert eg_o["uncertainty_escalated"] is True
    assert isinstance(eg_o["ood_score"], float)
    _assert_clean_copy(eg_o["summary_plain"])
    json.dumps(res_ood, allow_nan=False)

    # 3. 稀疏不足 8 帧保护：阶段切分不崩溃
    tiny_logs = {"face": logs["face"].iloc[:5].copy()}
    res_tiny = rme.evaluate_referee_ml(tiny_logs, session_id=None)
    assert res_tiny["available"] is True
    assert res_tiny["stage_segmentation"]["available"] is False
    assert res_tiny["stage_segmentation"]["change_points"] == []
    json.dumps(res_tiny, allow_nan=False)


def test_hmm_stage_segmentation_and_pelt_changepoints():
    logs, _ = _make_multi_stage_logs()
    res = rme.evaluate_referee_ml(logs, session_id=None)
    ss = res["stage_segmentation"]
    assert ss["available"] is True
    assert ss["method"] == "GaussianHMM (4状态Markov) + PELT变点检测"
    assert ss["n_windows"] == 19
    assert ss["transition_count"] == max(0, len(ss["macro_stages"]) - 1)
    assert ss["mean_dwell_sec"] > 0.0
    assert ss["transition_entropy"] >= 0.0

    stages = ss["macro_stages"]
    assert len(stages) >= 3
    assert abs(sum(s["ratio"] for s in stages) - 1.0) <= 0.02
    for i in range(len(stages) - 1):
        assert stages[i + 1]["start_sec"] >= stages[i]["end_sec"]
    # 验证早期适应建立期标注
    if stages[0]["stage"] != "steady_speaking" and len(stages) >= 2 and stages[1]["stage"] == "steady_speaking":
        assert stages[0].get("annotation") == "适应建立期"

    # PELT 应能定位至少 1 个行为转折点
    assert len(ss["change_points"]) >= 1
    for cp in ss["change_points"]:
        assert 0.0 <= cp["time_sec"] <= ss["total_duration_sec"] + 1.0
        assert cp["magnitude"] > 0.0
        _assert_clean_copy(cp["cue_summary"])

    _assert_clean_copy(ss["summary_plain"])
    _assert_no_forbidden_keys(res)


def test_real_sessions_integration(monkeypatch):
    series_dir = rme.REPO_ROOT / "data" / "l0_series"
    if series_dir.is_dir():
        monkeypatch.setenv("JINGXIN_SERIES_DIR", str(series_dir))

    loader = LogDataLoader()
    groups = loader._scan_and_group_files()
    face_sids = {f["session_id"] for f in groups.get("face", [])}

    if "20260929_180047_6fee" in face_sids:
        logs = loader.get_fused_latest_data("20260929_180047_6fee")
        _ = rme.evaluate_referee_ml(logs, session_id="20260929_180047_6fee")
        res = rme.evaluate_referee_ml(logs, session_id="20260929_180047_6fee")
        assert res["available"] is True
        assert res["evidence_gate"]["stage_code"] == "stage_1_verified"
        assert res["stage_segmentation"]["available"] is True
        assert len(res["stage_segmentation"]["macro_stages"]) >= 2
        _assert_no_forbidden_keys(res)
        _assert_clean_copy(res["evidence_gate"]["summary_plain"])
        _assert_clean_copy(res["stage_segmentation"]["summary_plain"])
        json.dumps(res, allow_nan=False)

    if "20261026_002930_cb79" in face_sids:
        logs = loader.get_fused_latest_data("20261026_002930_cb79")
        # 传入完整 L1 摘要验证部分覆盖场次识别为 stage_2_review（且隔离环境无账本时降为受限/复核）
        res_iso = rme.evaluate_referee_ml(logs, session_id="20261026_002930_cb79")
        assert res_iso["available"] is True
        assert res_iso["evidence_gate"]["stage_code"] in ("stage_2_review", "stage_3_limited")

        l1_mock_cb79 = {
            "n_measured": 40,
            "n_total": 41,
            "modality_summary": {
                "face": {"n_measured": 15, "n_total": 16},
                "gesture": {"n_measured": 9, "n_total": 9},
                "voice": {"n_measured": 13, "n_total": 13},
                "text": {"n_measured": 3, "n_total": 3},
            },
        }
        res_cb79 = rme.evaluate_referee_ml(logs, session_id="20261026_002930_cb79", l1_summary=l1_mock_cb79)
        assert res_cb79["evidence_gate"]["stage_code"] == "stage_2_review"
        assert res_cb79["stage_segmentation"]["available"] is True
        _assert_no_forbidden_keys(res_cb79)
        _assert_clean_copy(res_cb79["evidence_gate"]["summary_plain"])
        _assert_clean_copy(res_cb79["stage_segmentation"]["summary_plain"])
        json.dumps(res_cb79, allow_nan=False)
