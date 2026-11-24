# tests/test_evidence_chain_enhancements.py
"""Phase 2 测量学证据链补强测试：
- instrument_provenance 不再把 audio_* 误判为视觉 geometric
- l1_face._run_lengths 不产出 0.0 秒右删失假时长
- head_motion_per_sec 不再谎报「弧度/秒」
- parameter_registry.json 与代码常量一致且激活 provisional_flags
- cross_modal_coverage 按秒门控与无锚点阻断
- asr_confidence_disclosure 严守无 0.75 硬切与同源低尾降档
- 5 类 zero_kind 精准 0 值解释与空离散度三分流
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from report_frontend import (
    cross_modal_coverage,
    instrument_provenance,
    l1_face,
    l1_quality,
    l1_report,
    l1_thresholds,
    parameter_provenance,
    session_timeline,
    transcript_chars,
)


def test_instrument_provenance_does_not_misclassify_audio_prefix():
    assert instrument_provenance.instrument_of_column("audio_rms") == instrument_provenance.INSTRUMENT_AUDIO
    assert instrument_provenance.instrument_of_column("audio_duration") == instrument_provenance.INSTRUMENT_AUDIO
    assert instrument_provenance.instrument_of_column("au4_frown") == instrument_provenance.INSTRUMENT_GEOMETRIC
    assert instrument_provenance.instrument_of_column("au12_smile") == instrument_provenance.INSTRUMENT_GEOMETRIC
    assert instrument_provenance.sub_instrument_of_column("au12_smile") == "mediapipe_face_478_geo"
    assert instrument_provenance.sub_instrument_of_column("hand_visible_left") == "mediapipe_hand_21_geo"
    assert instrument_provenance.sub_instrument_of_column("shrug_level") == "mediapipe_pose_33_geo"


def test_run_lengths_never_emits_zero_second_trailing_frame():
    dt = np.array([np.nan, 1.0, 1.0, 1.0])
    # 末尾单帧右删失：不应产出 0.0 秒
    assert l1_face._run_lengths(np.array([0, 0, 0, 1]), dt) == []
    # 前段完整 + 末尾单帧右删失：只保留前段 1.0 秒
    runs = l1_face._run_lengths(np.array([0, 1, 0, 1]), dt)
    assert runs == [1.0]


def test_head_motion_unit_and_f5_brow_lip_rename():
    assert l1_face._ALL_SPECS["brow_furrow_ratio_vs_baseline"][0] == "F5_brow_lip"
    assert l1_face._ALL_SPECS["lip_compression_ratio_vs_baseline"][0] == "F5_brow_lip"
    assert "F5_tension" not in l1_quality.FAMILY_MODALITY
    assert l1_quality.FAMILY_MODALITY["T1_lexical"] == "text"
    assert l1_quality.FAMILY_MODALITY["T2_discourse"] == "text"
    assert l1_quality.FAMILY_MODALITY["T3_repair"] == "text"

    df = pd.DataFrame({"head_yaw": [0.0, 0.1, 0.2], "head_pitch": [0.0, 0.05, 0.1]})
    t = pd.Series([0.0, 1.0, 2.0])
    from report_frontend.event_windows import WindowSet
    ws = WindowSet("s1", "session", [])
    inds = {i.name: i for i in l1_face.compute_face_indicators(df, ws, t)}
    assert inds["head_motion_per_sec"].unit == "归一化位移/秒"


def test_parameter_registry_matches_live_code_constants():
    pp = parameter_provenance.build_parameter_provenance(list(l1_quality.INDICATOR_LABEL))
    by_id = {p["param_id"]: p for p in pp["registry"]}
    assert by_id["face.smile.au12_tau"]["value"] == l1_thresholds.AU12_SMILE_TAU
    assert by_id["common.self_baseline.mad_k"]["value"] == l1_thresholds.SELF_K
    assert by_id["cross_modal.jaccard_ok"]["value"] == cross_modal_coverage.JOINT_COV_OK
    assert by_id["cross_modal.jaccard_block"]["value"] == cross_modal_coverage.JOINT_COV_BLOCK
    assert pp["counts_by_tier"]["T2b"] == 0
    assert pp["n_provisional"] > 0


def test_zero_explanations_distinguish_five_kinds():
    assert "超过阈值" in l1_quality.explain_zero("smile_ratio", 0.0, 64)
    assert "低于触发阈值" in l1_quality.explain_zero("brow_furrow_ratio_vs_baseline", 0.0, 64)
    assert "未出现该事件" in l1_quality.explain_zero("discourse_marker_rate", 0.0, 200)
    assert "变化幅度为 0" in l1_quality.explain_zero("shrug_elevation_p90", 0.0, 50)
    assert "实测为 0" in l1_quality.explain_zero("response_latency_sec", 0.0, 8)
    # 0 事件手势爆发使用总帧数而非「看过 1 个有效观测」
    g_msg = l1_quality.explain_zero("gesture_burst_rate_per_min_vs_baseline", 0.0, 1, 240)
    assert "240 帧" in g_msg and "1 个有效观测" not in g_msg


def test_cross_modal_coverage_gate_aligned_vs_blocked_vs_unverifiable():
    from report_frontend.event_windows import WindowSet
    from report_frontend.l1_session import L1Result

    t_face = pd.Series(np.arange(0, 60, dtype=float))
    t_gest_ok = pd.Series(np.arange(0, 60, dtype=float))
    t_gest_disjoint = pd.Series(np.arange(100, 160, dtype=float))

    tb_face = session_timeline.Timebase("face", session_timeline.TimebaseStatus.OK, t_face, n_frames=60, anchored=True)
    tb_gest_ok = session_timeline.Timebase("gesture", session_timeline.TimebaseStatus.OK, t_gest_ok, n_frames=60, anchored=True)
    tb_voice_unanc = session_timeline.Timebase("voice", session_timeline.TimebaseStatus.OK, t_face, n_frames=60, anchored=False)

    tl_ok = session_timeline.SessionTimeline("s1", session_timeline.TimebaseStatus.OK, 0.0,
                                             {"face": tb_face, "gesture": tb_gest_ok, "voice": tb_voice_unanc})
    res_ok = L1Result("s1", tl_ok, WindowSet("s1", "session", []), [])
    cmc_ok = cross_modal_coverage.evaluate_cross_modal_coverage(res_ok)
    assert cmc_ok["session"]["pairs"]["face_gesture"]["status"] == "aligned"
    assert cmc_ok["session"]["pairs"]["face_voice"]["status"] == "unverifiable"
    assert cmc_ok["by_dimension"]["stress_resilience"]["allow_joint_claim"] is True

    tb_gest_bad = session_timeline.Timebase("gesture", session_timeline.TimebaseStatus.OK, t_gest_disjoint, n_frames=60, anchored=True)
    tl_bad = session_timeline.SessionTimeline("s2", session_timeline.TimebaseStatus.OK, 0.0,
                                              {"face": tb_face, "gesture": tb_gest_bad})
    res_bad = L1Result("s2", tl_bad, WindowSet("s2", "session", []), [])
    cmc_bad = cross_modal_coverage.evaluate_cross_modal_coverage(res_bad)
    assert cmc_bad["session"]["pairs"]["face_gesture"]["status"] == "blocked"
    assert cmc_bad["by_dimension"]["stress_resilience"]["allow_joint_claim"] is False


def test_asr_confidence_disclosure_degrades_on_high_low_tail(monkeypatch):
    monkeypatch.setattr(
        transcript_chars,
        "load_transcript_with_asr",
        lambda sid: {
            "text": "这是一段用于验证词级低尾降档机制的测试文本。",
            "n_chars": 22,
            "has_segments": True,
            "asr": {
                "engine": "faster_whisper_base",
                "audio_scope": "qa_segment",
                "asr_confidence_source": "word_prob_aligned_same_engine",
                "word_conf_summary": {
                    "n_words": 20, "median": 0.62, "p20": 0.28, "low_word_ratio_lt_0_5": 0.42
                },
            },
        },
    )
    inds = [{"name": "discourse_marker_rate", "value": 1.2}, {"name": "speech_rate", "value": 3.5}]
    disc = l1_report._build_asr_confidence_disclosure("sid_test", inds)
    assert disc["status"] == "available"
    assert set(disc["degraded_indicators"]) == {"discourse_marker_rate", "speech_rate"}


def test_build_l1_report_exposes_all_additive_evidence_blocks():
    payload = l1_report.build_l1_report("s_empty", logs={})
    l1 = payload["l1"]
    for k in (
        "lead_summary", "indicator_meta", "zero_explanations", "dimension_registry",
        "parameter_provenance", "cross_modal_coverage", "asr_confidence_disclosure",
        "governance_ledger", "blocked_dependency_warnings", "second_instrument_appendix",
        "face_dual_instrument", "blendshape_calibration", "feature_layer_v2",
        "behavior_regimes", "referee_ml", "signal_quality_audit", "speaker_turn_audit",
    ):
        assert k in l1, f"l1 载荷缺少增量字段 {k}"
    assert len(l1["indicator_meta"]) == 41
    assert len(l1["governance_ledger"]["l0_blocked"]) == 5
    for dim_info in l1["dimension_registry"].values():
        assert dim_info.get("user_summary")
        assert dim_info.get("literature_anchor")
    assert l1["speaker_turn_audit"]["isolation_mode"] == "mixed_unsegmented"
    assert l1["behavior_regimes"]["available"] is False
    assert l1["behavior_regimes"]["advisory_only"] is True
    assert l1["face_dual_instrument"]["available"] is False
    assert l1["blendshape_calibration"]["available"] is False
    assert l1["blendshape_calibration"]["inference_ms"] == 0.0
    assert l1["feature_layer_v2"]["available"] is False
    assert l1["feature_layer_v2"]["inference_ms"] == 0.0
    assert l1["referee_ml"]["available"] is False
    assert l1["referee_ml"]["advisory_only"] is True
    assert l1["referee_ml"]["inference_ms"] == 0.0


def test_transcript_chars_merges_whisper_conf_sidecar(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("JINGXIN_TRANSCRIPTS_DIR", str(tmp_path))
    sid = "20261124_000001_ab12"
    (tmp_path / f"{sid}.json").write_text(
        json.dumps({
            "text": "这是一段用于测试旁路合并的中文回答内容。",
            "n_chars": 21,
            "asr": {
                "engine": "funasr",
                "asr_confidence": None,
                "asr_confidence_source": "unavailable",
            },
        }),
        encoding="utf-8",
    )
    (tmp_path / f"{sid}.whisper_conf.json").write_text(
        json.dumps({
            "session_id": sid,
            "asr_confidence_source": "word_prob_aligned_same_engine",
            "engine": "faster-whisper",
            "model_size": "tiny",
            "audio_scope": "qa_segment",
            "word_conf_summary": {
                "n_words": 15,
                "median": 0.88,
                "p20": 0.65,
                "low_word_ratio_lt_0_5": 0.05,
            },
        }),
        encoding="utf-8",
    )
    tr = transcript_chars.load_transcript_with_asr(sid)
    assert tr is not None
    assert tr["asr"]["confidence_engine"] == "faster-whisper"
    assert tr["asr"]["word_conf_summary"]["median"] == 0.88
    disc = l1_report._build_asr_confidence_disclosure(
        sid, [{"name": "discourse_marker_rate", "value": 1.2}]
    )
    assert disc["status"] == "available"
    assert disc["degraded_indicators"] == []
    assert "旁路同音频识别模型" in disc["note_plain"]
