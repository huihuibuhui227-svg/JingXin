# tests/test_reviewer_notes.py
"""管理员人工复核注记持久化与红线守卫测试。"""
from __future__ import annotations

import app as panel_app
import recordings_browser
from report_frontend import reviewer_notes


def test_reviewer_notes_roundtrip_redline_and_admin_guard(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_REVIEW_NOTES_DIR", str(tmp_path))

    default_note = reviewer_notes.read_reviewer_note("20260927_212143_95a8")
    assert default_note["exists"] is False
    assert default_note["verification_status"] == "pending_review"

    saved = reviewer_notes.write_reviewer_note(
        "20260927_212143_95a8",
        verification_status="verified_consistent",
        focus_tags=["核验手势入画率", "追问推导假设"],
        note_text="已对照实采视频核实第 3 题停顿延长源自板书推导，非设备故障。",
    )
    assert saved["exists"] is True
    assert reviewer_notes.read_reviewer_note("20260927_212143_95a8")["note_text"] == saved["note_text"]

    # 含打分或受限词必须被拒收
    for bad_txt in ("综合得分 88 分建议通过", "受试者存在高度焦虑情绪"):
        try:
            reviewer_notes.write_reviewer_note(
                "20260927_212143_95a8",
                verification_status="verified_consistent",
                note_text=bad_txt,
            )
            raise AssertionError("应当拒绝写入违规文本")
        except ValueError:
            pass

    client = panel_app.app.test_client()
    r_get = client.get("/api/report/review-notes/20260927_212143_95a8")
    assert r_get.status_code == 200
    assert r_get.get_json()["note"]["exists"] is True

    # 未授权管理员禁止写入（无论未设密码 503 或无 Bearer token 401）
    r_unauth = client.post(
        "/api/report/review-notes/20260927_212143_95a8",
        json={"verification_status": "needs_recollect", "note_text": "建议重录"},
    )
    assert r_unauth.status_code in (401, 403, 503)

    monkeypatch.setenv("JX_ADMIN_PASSWORD", "test-admin-pass")
    r_denied = client.post(
        "/api/report/review-notes/20260927_212143_95a8",
        json={"verification_status": "needs_recollect", "note_text": "建议重录"},
    )
    assert r_denied.status_code == 401

    r_login = client.post("/api/admin/login", json={"password": "test-admin-pass"})
    assert r_login.status_code == 200
    token = r_login.get_json()["token"]

    r_ok = client.post(
        "/api/report/review-notes/20260927_212143_95a8",
        headers={"Authorization": f"Bearer {token}"},
        json={"verification_status": "needs_recollect", "note_text": "面部角度偏大建议补充正面采录。"},
    )
    assert r_ok.status_code == 200
    assert r_ok.get_json()["note"]["verification_status"] == "needs_recollect"
