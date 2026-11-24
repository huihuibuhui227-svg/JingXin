# tests/test_longitudinal_tracker.py
"""基地三个人纵向随访引擎与基地一多场对比测试。"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import app as panel_app
import redline_guard
from report_frontend import longitudinal_tracker as lt
from tests.test_report_layer import BANNED

_EXTRA_BANNED = ("心理", "情绪", "得分", "评级", "评分", "抑郁", "临床")


def test_longitudinal_tracker_module_has_zero_banned_words():
    src = Path(lt.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for w in (*BANNED, *_EXTRA_BANNED):
                assert w not in node.value, f"longitudinal_tracker 包含禁用词 {w!r}: {node.value}"
            assert redline_guard.find_assistant_violations(node.value) == []


def test_def_version_comparability_guard_and_gap_not_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_LONGITUDINAL_FILE", str(tmp_path / "subjects.json"))

    lt.upsert_subject("SUBJ-001", alias_note="基线随访试跑")
    lt.attach_visit("SUBJ-001", "20260924_100000_aaaa", visit_label="初代访次")
    lt.attach_visit("SUBJ-001", "20261026_100000_bbbb", visit_label="二代访次")

    class _FakeLoader:
        def get_fused_latest_data(self, sid):
            return {"mock": True}

    monkeypatch.setattr(
        lt,
        "build_l1_report",
        lambda sid, logs=None, scene_profile=None: {
            "coverage": {"n_passed": 8, "n_slots": 14, "confidence_cap": "中"},
            "l1": {
                "indicators": [
                    {"name": "voiced_ratio", "value": None, "confidence": "无", "gap": "no_samples"},
                    {"name": "pause_structure_entropy", "value": 0.85, "confidence": "高", "gap": None},
                ],
                "feature_layer_v2": {
                    "tracks": {
                        "face_kinematics": {
                            "pose_mode": "3d_matrix",
                            "head_angular_velocity_3d_p50": 12.4,
                        }
                    }
                },
                "blendshape_calibration": {"pose_mode": "3d_matrix"},
                "referee_ml": {"evidence_gate": {"stage_code": "stage_2_review", "stage_label": "复核阶段", "uncertainty_escalated": True, "ood_anomaly_flag": False}},
            },
        },
    )

    cross_gen = lt.get_subject_trajectory("SUBJ-001", loader=_FakeLoader())
    assert cross_gen["comparable"] is False
    assert cross_gen["def_versions"] == ["capture:gen1", "capture:gen2"]
    voiced_pts = next(s for s in cross_gen["series"] if s["key"] == "voiced_ratio")["points"]
    # 缺口绝不补零
    assert all(p["value"] is None and p["gap"] == "no_samples" for p in voiced_pts)
    ang_pts = next(s for s in cross_gen["series"] if s["key"] == "head_angular_vel_3d")["points"]
    assert all(p["value"] == 12.4 and p["confidence"] == "高" for p in ang_pts)

    # 同代次场次可比并输出相对 t0 偏移
    same_gen = lt.compare_sessions(["20261010_100000_cccc", "20261020_100000_dddd"], loader=_FakeLoader())
    assert same_gen["comparable"] is True
    assert same_gen["def_versions"] == ["capture:gen2"]
    pse = next(s for s in same_gen["series"] if s["key"] == "pause_structure_entropy")
    assert pse["summary"]["drawable"] is True
    assert pse["points"][0]["delta_from_t0_pct"] == 0.0

    # 二维代换口径不得混入 3D 角速度（度/秒）
    monkeypatch.setattr(
        lt,
        "build_l1_report",
        lambda sid, logs=None, scene_profile=None: {
            "coverage": {"n_passed": 4, "n_slots": 14, "confidence_cap": "低"},
            "l1": {
                "indicators": [],
                "feature_layer_v2": {
                    "tracks": {
                        "face_kinematics": {
                            "pose_mode": "2d_legacy",
                            "head_angular_velocity_3d_p50": 0.45,
                        }
                    }
                },
                "blendshape_calibration": {"pose_mode": "2d_legacy"},
            },
        },
    )
    legacy_only = lt.compare_sessions(["20261010_100000_cccc"], loader=_FakeLoader())
    legacy_ang = next(s for s in legacy_only["series"] if s["key"] == "head_angular_vel_3d")["points"][0]
    assert legacy_ang["value"] is None
    assert legacy_ang["gap"] == "requires_3d_pose"


def test_longitudinal_and_compare_flask_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_LONGITUDINAL_FILE", str(tmp_path / "subj_api.json"))
    from report_frontend.data_loader import LogDataLoader

    monkeypatch.setattr(LogDataLoader, "get_fused_latest_data", lambda self, sid=None: {})

    client = panel_app.app.test_client()
    r1 = client.post("/api/longitudinal/subjects", json={"subject_code": "SUBJ-099", "alias_note": "两组照看"})
    assert r1.status_code == 200
    r2 = client.post("/api/longitudinal/subjects/SUBJ-099/visits", json={"session_id": "20261015_120000_ab12"})
    assert r2.status_code == 200
    r3 = client.get("/api/longitudinal/subjects/SUBJ-099/trajectory")
    assert r3.status_code == 200
    assert r3.get_json()["n_visits"] == 1

    r_cmp = client.get("/api/reports/compare?session_ids=20261015_120000_ab12,20261016_120000_cd34")
    assert r_cmp.status_code == 200
    assert r_cmp.get_json()["n_visits"] == 2
