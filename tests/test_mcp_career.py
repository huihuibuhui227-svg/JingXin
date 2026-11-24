# tests/test_mcp_career.py
"""标准 JSON-RPC 2.0 Career MCP Server 与求职证据缺口对齐测试。"""
from __future__ import annotations

import json

import app as panel_app
import redline_guard
from mcp_career import server as mcp_srv


def test_mcp_jsonrpc_handshake_and_tools_list():
    init_resp = mcp_srv.handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init_resp["result"]["serverInfo"]["name"] == "hengcai-career-mcp"
    assert mcp_srv.handle_jsonrpc({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    list_resp = mcp_srv.handle_jsonrpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = list_resp["result"]["tools"]
    names = [t["name"] for t in tools]
    assert names == [
        "search_career_positions",
        "decompose_jd_competencies",
        "generate_ai_critique_task",
        "align_session_evidence_to_jd",
    ]


def test_mcp_search_decompose_and_redline_safety():
    call_search = mcp_srv.handle_jsonrpc({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "search_career_positions", "arguments": {"keyword": "算法"}},
    })
    assert call_search["result"]["isError"] is False
    positions = call_search["result"]["structuredContent"]["positions"]
    assert len(positions) >= 1
    assert positions[0]["position_id"] == "pos_ai_algo"

    jd_res = mcp_srv.tool_decompose_jd_competencies(
        "负责大模型应用研发、Python/SQL 脚本开发、假设检验验证与清晰技术汇报。"
    )
    assert len(jd_res["observable_competencies"]) >= 2
    assert len(jd_res["blocked_hard_skills"]) >= 1

    # 校验岗位库与任务输出零红线违禁词
    for pos in mcp_srv.load_catalog():
        blob = json.dumps(pos, ensure_ascii=False)
        assert redline_guard.find_assistant_violations(blob) == [], pos["position_id"]
        task = mcp_srv.tool_generate_ai_critique_task(position_id=pos["position_id"])
        assert len(task["stages"]) == 3
        assert redline_guard.find_assistant_violations(json.dumps(task, ensure_ascii=False)) == []


def test_mcp_align_session_never_scores_and_exposes_flask_bridge(monkeypatch):
    from report_frontend.data_loader import LogDataLoader

    monkeypatch.setattr(LogDataLoader, "get_fused_latest_data", lambda self, sid=None: {})

    aligned = mcp_srv.tool_align_session_evidence_to_jd(
        "20260927_212143_95a8",
        position_id="pos_data_analyst",
        logs={
            "transcript_text": (
                "原回答仅统计整体均值，但该前提不成立且忽视了分母口径差异这一边界条件。"
                "问题在于没有分组对照，应当改为按分层基线检验后再阐述结论。"
            )
        },
    )
    assert aligned["scoring_participation"] is False
    assert "score" not in aligned and "match_score" not in aligned
    assert aligned["summary_counts"]["competencies_total"] == 3
    assert redline_guard.find_assistant_violations(json.dumps(aligned, ensure_ascii=False)) == []

    client = panel_app.app.test_client()
    r_rpc = client.post("/api/mcp/career", json={"jsonrpc": "2.0", "id": 9, "method": "tools/list"})
    assert r_rpc.status_code == 200
    assert len(r_rpc.get_json()["result"]["tools"]) == 4

    r_pos = client.get("/api/career/positions?track=科研与学术训练")
    assert r_pos.status_code == 200
    assert r_pos.get_json()["total"] >= 1

    r_align = client.post("/api/career/align-evidence", json={
        "session_id": "20260927_212143_95a8",
        "position_id": "pos_ai_algo",
    })
    assert r_align.status_code == 200
    assert r_align.get_json()["scoring_participation"] is False
