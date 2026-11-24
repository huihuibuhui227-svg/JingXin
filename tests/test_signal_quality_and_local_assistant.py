# tests/test_signal_quality_and_local_assistant.py
"""Step 3 回归测试：自监督采集信号质量分类器 + 严格隔离的本地按需辅助模块。"""
from __future__ import annotations

import numpy as np
import pandas as pd

import app as panel
import redline_guard
from report_frontend import local_llm_assistant, signal_quality_clf


def _synthetic_face_gesture(n: int = 30, *, occluded: bool = False) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(7)
    ts = 1700000000.0 + np.arange(n, dtype=float)
    au1 = rng.uniform(0.05, 0.35, size=n)
    au4 = rng.uniform(0.02, 0.25, size=n)
    au12 = rng.uniform(0.05, 0.40, size=n)
    gx = rng.normal(0.0, 0.04, size=n)
    gy = rng.normal(0.0, 0.04, size=n)
    yaw = rng.normal(0.0, 0.08, size=n)
    if occluded:
        au1[: int(n * 0.85)] = np.nan
        au4[: int(n * 0.85)] = np.nan
        au12[: int(n * 0.85)] = np.nan
        gx[: int(n * 0.85)] = np.nan
    face_df = pd.DataFrame({
        "timestamp": ts,
        "au1_inner_brow_raise": au1,
        "au4_frown": au4,
        "au12_smile": au12,
        "gaze_direction_x": gx,
        "gaze_direction_y": gy,
        "head_yaw": yaw,
    })
    gest_df = pd.DataFrame({
        "timestamp": ts,
        "left_wrist_jitter_world": rng.uniform(0.001, 0.015, size=n),
        "right_wrist_jitter_world": rng.uniform(0.001, 0.015, size=n),
        "left_shoulder_angle": rng.uniform(80.0, 100.0, size=n),
    })
    return {"face": face_df, "gesture": gest_df}


def test_signal_quality_audit_distinguishes_clean_and_occluded(tmp_path):
    clean = _synthetic_face_gesture(30, occluded=False)
    bad = _synthetic_face_gesture(30, occluded=True)

    # 1. 无模型时纯规则降级
    missing_mdl = tmp_path / "absent.joblib"
    r_clean_rule = signal_quality_clf.audit_session_signal_quality(clean, "s1", model_path=missing_mdl)
    r_bad_rule = signal_quality_clf.audit_session_signal_quality(bad, "s2", model_path=missing_mdl)
    assert r_clean_rule["status"] == "fallback_rule"
    assert r_clean_rule["advisory_only"] is True
    assert r_clean_rule["usable_window_ratio"] >= 0.8
    assert r_bad_rule["usable_window_ratio"] < r_clean_rule["usable_window_ratio"]

    # 2. 帧数不足 10 帧时显式说明
    tiny = {"face": clean["face"].iloc[:5]}
    r_tiny = signal_quality_clf.audit_session_signal_quality(tiny, "s3", model_path=missing_mdl)
    assert r_tiny["status"] == "insufficient_windows"
    assert r_tiny["n_windows"] == 0

    # 3. 真实训练产出模型存在时使用纯 CPU 毫秒级推理（验证模型加载 + 缓存后热推理）
    if signal_quality_clf.DEFAULT_MODEL_PATH.is_file():
        r_mdl_cold = signal_quality_clf.audit_session_signal_quality(clean, "s4")
        assert r_mdl_cold["status"] == "ready"
        assert r_mdl_cold["model_used"] is True
        r_mdl_warm = signal_quality_clf.audit_session_signal_quality(clean, "s4")
        assert r_mdl_warm["inference_ms"] < 50.0


def test_local_llm_assistant_enforces_redline_guard_and_fallback(monkeypatch):
    sample_payload = {
        "coverage": {"n_passed": 10, "n_slots": 14},
        "dimensions": {
            "logical_thinking": {
                "name": "话语结构特征",
                "coverage": {"n_passed": 2, "n_slots": 3},
                "confidence": "中",
                "evidence_gaps": ["连接词密度：语音转写尚未采集"],
            }
        },
        "l1": {
            "lead_summary": {"headline": "本场 10/14 项指标通过白盒审查。"},
            "cross_modal_coverage": {"status": "ok", "summary_plain": "跨模态按秒重叠率 88%。"},
            "signal_quality_audit": {"summary_plain": "共巡检 12 个窗口，可用率 100%。"},
        },
    }

    # 情况 A：本地模型违规输出分数与临床词 → 必须拦截并回退确定性摘要
    monkeypatch.setattr(
        local_llm_assistant,
        "_call_local_openai_chat",
        lambda *a, **k: ("该受试者综合得分 88 分，明显处于焦虑状态。", None, "mock-llm"),
    )
    res_blocked = local_llm_assistant.assist_report(sample_payload, mode="summary", use_llm=True)
    assert res_blocked["source"] == "deterministic_fallback"
    assert res_blocked["redline_verified"] is True
    assert "红线词" in (res_blocked["fallback_reason"] or "")
    assert redline_guard.find_assistant_violations(res_blocked["content"]) == []

    # 情况 B：本地模型输出合规文本 → 直接透出并带强制水印
    monkeypatch.setattr(
        local_llm_assistant,
        "_call_local_openai_chat",
        lambda *a, **k: ("本场共有 10/14 项指标通过审查，话语结构特征因语音转写缺失而证据偏薄。", None, "mock-llm"),
    )
    res_ok = local_llm_assistant.assist_report(sample_payload, mode="summary", use_llm=True)
    assert res_ok["source"] == "local_llm"
    assert res_ok["redline_verified"] is True
    assert "不参与任何指标计算与评定" in res_ok["watermark"]

    # 情况 C：追问提纲模式（离线/确定性）
    res_guide = local_llm_assistant.assist_report(sample_payload, mode="interview_guide", use_llm=False)
    assert res_guide["mode"] == "interview_guide"
    assert res_guide["source"] == "deterministic_fallback"
    assert "话语结构特征" in res_guide["content"]
    assert redline_guard.find_assistant_violations(res_guide["content"]) == []


def test_question_bank_tfidf_search_and_flask_routes(tmp_path, monkeypatch):
    import question_bank
    monkeypatch.setattr(question_bank, "BANK_PATH", tmp_path / "qb.json")
    question_bank.create_bank(
        "research",
        "实验复现专项题库",
        [
            {"question": "当实验结果与预期假设不符时，你如何设计对照实验排查数据误差？", "reference_answer": "控制变量与日志溯源"},
            {"question": "请介绍一次你阅读英文顶会论文并复现关键算法的经历。", "reference_answer": "代码复现与超参核对"},
        ],
    )

    out = local_llm_assistant.search_question_bank("research", "对照实验 排查误差", top_k=3, use_llm=False)
    assert out["source"] == "local_tfidf"
    assert out["hits"], out
    assert "对照实验" in out["hits"][0]["question"]

    panel.app.config.update(TESTING=True)
    c = panel.app.test_client()
    resp = c.post("/api/question-bank/llm-search", json={"module": "research", "query": "复现算法", "top_k": 3})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "success"
    assert body["hits"]

    # `/api/report/llm-assist` 使用隔离空日志目录验证毫秒级返回
    import report_frontend.data_loader as dl
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    orig_cls = dl.LogDataLoader
    monkeypatch.setattr(dl, "LogDataLoader", lambda *a, **k: orig_cls(str(log_dir)))
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path / "no_rec"))
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path / "no_ser"))

    r_assist = c.post("/api/report/llm-assist", json={"mode": "summary", "use_llm": False})
    assert r_assist.status_code == 200
    assist_body = r_assist.get_json()
    assert assist_body["status"] == "success"
    assert assist_body["redline_verified"] is True
    assert "不参与任何指标计算与评定" in assist_body["watermark"]


def test_narrative_review_and_participant_feedback_modes():
    from report_frontend.l1_report import build_l1_report

    real_shape = build_l1_report("nonexistent_synth_narrative", logs={}, scene_profile="ai_metacognition")
    res_rev = local_llm_assistant.assist_report(
        real_shape,
        mode="narrative_review",
        use_llm=False,
        scene_profile="ai_metacognition",
    )
    assert res_rev["mode"] == "narrative_review"
    assert res_rev["redline_verified"] is True
    assert len(res_rev["sections"]) == 5
    assert "元智审辨" in res_rev["content"]
    assert redline_guard.find_assistant_violations(res_rev["content"]) == []

    res_stu = local_llm_assistant.assist_report(
        real_shape,
        mode="participant_feedback",
        use_llm=False,
        scene_profile="ai_metacognition",
    )
    assert res_stu["mode"] == "participant_feedback"
    assert res_stu["redline_verified"] is True
    assert len(res_stu["sections"]) == 5
    assert "你" in res_stu["content"]
    assert redline_guard.find_assistant_violations(res_stu["content"]) == []

