# tests/test_auth_and_redaction.py
"""鉴权加固与演示模式 PII 脱敏守卫。

覆盖：
1. 未认证请求 `/api/report/structured` 自动脱敏受试者身份，管理员 token 解锁完整身份；
2. 未认证请求 `/api/files/reports` 隐藏含真实姓名的后缀文件名，返回 `/output/redacted/<token>`，
   且 `/output/redacted/<token>` 与直访 `/output/<文件名>` 均自动剥除 `<div class="subject">` 身份块；
3. `/api/question-bank/*` 写/删/激活接口必须校验管理员身份；
4. `/api/run/face` 等废旧模块返回 HTTP 400；
5. `/api/system/health` 返回四路服务结构；
6. `POST /session/{sid}/discard` 在配置了 `JX_ADMIN_PASSWORD` 时阻止匿名 HTTP 删除历史场次。
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

import app as panel
import media_retention
import recordings_browser as rb
import session_purge


def _admin_headers(client, password: str = "test-admin-secret") -> dict[str, str]:
    resp = client.post("/api/admin/login", json={"password": password})
    assert resp.status_code == 200, resp.get_json()
    tok = resp.get_json()["token"]
    return {"Authorization": f"Bearer {tok}"}


def test_structured_report_redacts_subject_for_anonymous_and_reveals_for_admin(tmp_path, monkeypatch):
    sid = "20261123_110000_ab12"
    rec_root = tmp_path / "recordings"
    sdir = rec_root / f"01-演示受试者甲-20260001-信息学院__{sid}"
    sdir.mkdir(parents=True)
    (sdir / "label.json").write_text(
        json.dumps({"name": "演示受试者甲", "student_id": "20260001", "department": "信息学院", "serial": "01"}),
        encoding="utf-8",
    )
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / f"face_au_log_{sid}.csv").write_text(
        "session_id,timestamp,au12_smile\n" + f"{sid},1700000000.0,0.2\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(rec_root))
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path / "series"))
    monkeypatch.setenv("JX_ADMIN_PASSWORD", "test-admin-secret")
    media_retention._reset_for_tests()

    import report_frontend.data_loader as dl
    orig_cls = dl.LogDataLoader
    monkeypatch.setattr(dl, "LogDataLoader", lambda *a, **k: orig_cls(str(log_dir)))

    panel.app.config.update(TESTING=True)
    c = panel.app.test_client()

    anon = c.get(f"/api/report/structured?session_id={sid}").get_json()
    assert anon["status"] == "success"
    assert anon["subject"].get("status") == "redacted"
    assert "演示受试者甲" not in json.dumps(anon, ensure_ascii=False)

    adm = c.get(
        f"/api/report/structured?session_id={sid}",
        headers=_admin_headers(c),
    ).get_json()
    assert adm["status"] == "success"
    assert adm["subject"]["name"] == "演示受试者甲"
    assert adm["subject"]["student_id"] == "20260001"


def test_report_gallery_and_output_redact_named_files_for_anonymous(tmp_path, monkeypatch):
    monkeypatch.setattr(panel, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("JX_ADMIN_PASSWORD", "test-admin-secret")
    panel.app.config.update(TESTING=True)

    sid = "20261123_110500_cd34"
    fname = "HengCai_Evidence_Report_20261123_110500-演示受试者乙.html"
    html = (
        f'<!DOCTYPE html><html><body>'
        f'<div class="subject"><span class="subject-k">被测者</span><span class="subject-v">演示受试者乙</span></div>'
        f'<p>会话 {sid}</p></body></html>'
    )
    (tmp_path / fname).write_text(html, encoding="utf-8")

    c = panel.app.test_client()
    items = c.get("/api/files/reports").get_json()
    assert len(items) == 1
    assert items[0]["redacted"] is True
    assert "演示受试者乙" not in items[0]["name"]
    assert items[0]["session_id"] == sid
    assert items[0]["url"].startswith("/output/redacted/")

    # 匿名经 token 访问：剥除身份横幅
    r_tok = c.get(items[0]["url"])
    assert r_tok.status_code == 200
    body_tok = r_tok.get_data(as_text=True)
    assert "演示受试者乙" not in body_tok
    assert "已脱敏" in body_tok

    # 匿名直访原路径：同样剥除身份横幅
    r_direct = c.get(f"/output/{fname}")
    assert r_direct.status_code == 200
    assert "演示受试者乙" not in r_direct.get_data(as_text=True)

    # 管理员身份：恢复完整文件名与正文身份
    hdrs = _admin_headers(c)
    adm_items = c.get("/api/files/reports", headers=hdrs).get_json()
    assert adm_items[0]["name"] == fname
    assert adm_items[0]["redacted"] is False
    r_adm = c.get(adm_items[0]["url"], headers=hdrs)
    assert "演示受试者乙" in r_adm.get_data(as_text=True)


def test_question_bank_mutations_require_admin_and_run_rejects_obsolete(tmp_path, monkeypatch):
    monkeypatch.setenv("JX_ADMIN_PASSWORD", "test-admin-secret")
    import question_bank
    monkeypatch.setattr(question_bank, "BANK_PATH", tmp_path / "qb.json")

    panel.app.config.update(TESTING=True)
    c = panel.app.test_client()

    # 只读 GET 公开
    assert c.get("/api/question-bank/interview").status_code == 200

    # 写/激活未认证 -> 401
    assert c.post("/api/question-bank/interview/banks", json={"name": "新库", "questions": [{"question": "题1"}]}).status_code == 401
    assert c.post("/api/question-bank/interview/active", json={"id": None}).status_code == 401

    # 管理员认证 -> 200
    hdrs = _admin_headers(c)
    created = c.post(
        "/api/question-bank/interview/banks",
        json={"name": "测试题库", "questions": [{"question": "请说明你的实验复现流程"}]},
        headers=hdrs,
    )
    assert created.status_code == 200

    # 废旧模块 -> 400
    assert c.post("/api/run/face").status_code == 400
    assert c.post("/api/run/voice").status_code == 400


def test_system_health_endpoint_returns_four_services(monkeypatch):
    monkeypatch.setattr(
        panel,
        "_probe_http_health",
        lambda name, url, timeout=1.2: (name, {"ok": True, "status": "healthy", "latency_ms": 1.2, "url": url}),
    )
    panel.app.config.update(TESTING=True)
    c = panel.app.test_client()
    data = c.get("/api/system/health").get_json()
    assert data["status"] == "healthy"
    assert data["brand"] == "衡才智析"
    assert set(data["services"].keys()) == {"panel", "face", "voice", "gesture"}


def test_voice_discard_blocks_anonymous_http_delete_of_historical_session(tmp_path, monkeypatch):
    voice_app = importlib.import_module("voice_interaction.api.app")
    rec_root = tmp_path / "recordings"
    sid = "20260928_212130_5b10"
    sdir = rec_root / f"18-测试丙-20260099-物理学院__{sid}"
    sdir.mkdir(parents=True)
    (sdir / "label.json").write_text("{}", encoding="utf-8")
    logs = tmp_path / "logs"
    logs.mkdir()

    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(rec_root))
    monkeypatch.setenv("JX_ADMIN_PASSWORD", "hist-protect-pw")
    monkeypatch.setattr(session_purge, "default_logs_dir", lambda: logs)
    voice_app._MINTED_DISCARD_TOKENS.pop(sid, None)
    media_retention._reset_for_tests()

    tc = TestClient(voice_app.app)
    r_deny = tc.post(f"/session/{sid}/discard")
    assert r_deny.status_code == 403
    assert sdir.is_dir()

    # 即使未设置 JX_ADMIN_PASSWORD，匿名 HTTP 也必须 fail-closed 拒绝历史场次删除
    monkeypatch.delenv("JX_ADMIN_PASSWORD", raising=False)
    r_deny_no_env = tc.post(f"/session/{sid}/discard")
    assert r_deny_no_env.status_code == 403
    assert sdir.is_dir()

    # 携带正确管理员密码可放行
    monkeypatch.setenv("JX_ADMIN_PASSWORD", "hist-protect-pw")
    r_ok = tc.post(f"/session/{sid}/discard", headers={"X-Admin-Password": "hist-protect-pw"})
    assert r_ok.status_code == 200
    assert not sdir.exists()
