# tests/test_scene_profiles.py
"""声明式多场景档案、`scene_extensions` 与 `app.py` 路由回归测试。"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

import app as panel_app
import redline_guard
from report_frontend import l1_report, scene_extensions, scene_profiles
from report_frontend.report_generator import ReportGenerator
from tests.test_report_layer import BANNED

_REPO = Path(__file__).resolve().parents[1]
_EXTRA_BANNED = ("心理", "情绪", "得分", "评级", "评分", "科研潜质评估", "抑郁", "临床级")


def test_scene_profiles_registry_has_three_bases_and_zero_banned_words():
    profiles = scene_profiles.list_scene_profiles()
    ids = [p["id"] for p in profiles]
    assert ids == ["research_interview", "ai_metacognition", "longitudinal_followup"]

    raw_json = (_REPO / "report_frontend" / "scene_profiles.json").read_text(encoding="utf-8")
    for w in (*BANNED, *_EXTRA_BANNED):
        assert w not in raw_json, f"scene_profiles.json 含禁用词 {w!r}"

    for py_name in ("scene_profiles.py", "scene_extensions.py"):
        src = (_REPO / "report_frontend" / py_name).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for w in (*BANNED, *_EXTRA_BANNED):
                    assert w not in node.value, f"{py_name} 字符串字面量含禁用词 {w!r}"


def test_build_l1_report_preserves_41_core_indicators_across_all_scenes():
    sid = "20260927_212143_95a8"
    for pid in ("research_interview", "ai_metacognition", "longitudinal_followup"):
        payload = l1_report.build_l1_report(sid, logs={}, scene_profile=pid)
        assert payload["l1"]["n_total"] == 41
        assert len(payload["l1"]["indicators"]) == 41
        assert payload["brand"]["subtitle"] == "多模态行为证据分析系统"
        sp = payload["scene_profile"]
        assert sp["id"] == pid
        assert redline_guard.find_assistant_violations(sp["compliance_banner"]) == []
        html = ReportGenerator(output_dir="/tmp")._build_html_report(
            payload, {}, {}, {}, [], target=sid
        )
        assert "当前场景：" in html
        assert sp["name"] in html


def test_metacognition_text_extensions_pure_rules_and_gaps():
    res_missing = scene_extensions.compute_metacognition_text_extensions("nonexistent_sid_xyz", logs={})
    assert res_missing["available"] is False
    assert len(res_missing["indicators"]) == 2
    assert all(ind["value"] is None and ind["gap"] is not None for ind in res_missing["indicators"])

    sample_text = (
        "原回答认为样本均值就能代表全体变化，但这一前提不成立，且忽视了不同访次口音差异这一边界条件。"
        "因此问题在于缺少个人历史对照基线，应当改为采用受试者自身中位数与离散带进行参照；"
        "同时上一版把停顿比例与发声占比同时计分混淆了代数关系，建议修改为显式淘汰冗余变量。"
    )
    res_ok = scene_extensions.compute_metacognition_text_extensions(
        "synthetic_meta_sid",
        logs={"transcript_text": sample_text},
    )
    assert res_ok["available"] is True
    by_name = {x["name"]: x for x in res_ok["indicators"]}
    assert by_name["counter_premise_marker_rate"]["value"] > 0
    assert by_name["revision_anchor_density"]["value"] > 0


def test_app_scene_profiles_endpoint_and_structured_switch(monkeypatch):
    from report_frontend.data_loader import LogDataLoader

    monkeypatch.setattr(LogDataLoader, "get_fused_latest_data", lambda self, sid=None: {})

    client = panel_app.app.test_client()
    r = client.get("/api/scene-profiles")
    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "success"
    assert len(data["profiles"]) == 3

    r_single = client.get("/api/scene-profiles?id=metacognition")
    assert r_single.status_code == 200
    assert r_single.get_json()["profile"]["id"] == "ai_metacognition"

    r_bad = client.get("/api/scene-profiles?id=invalid_scene")
    assert r_bad.status_code == 400

    r_rep = client.get("/api/report/structured?session_id=20260927_212143_95a8&scene_profile=followup")
    assert r_rep.status_code == 200
    rep_json = r_rep.get_json()
    assert rep_json["result"]["scene_profile"]["id"] == "longitudinal_followup"
    assert rep_json["result"]["l1"]["n_total"] == 41
