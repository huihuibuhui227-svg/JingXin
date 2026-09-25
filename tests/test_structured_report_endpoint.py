# tests/test_structured_report_endpoint.py
"""第 18 条(复审 I3):`/api/report/structured` 必须说得出"这是哪一场",空数据也要照常回答。

失效现场(2026-09-24 复审实测):
    GET /api/report/structured                          → {"status":"error","message":"未找到评估日志数据"}
    GET /api/report/structured?session_id=<有数据那场>  → {"status":"success", ...}
第一条会打中的是**最新的一场**(可能是一场刚 /interview/start、还没有任何数据的会话),
于是前端报告页显示「暂无报告数据」,而盘上明明有一场三模态齐全的会话。
并且成功那一支的 JSON 里**没有** session_id、没有任何"这是哪一场"的字样
—— spec D2 的"写明是哪一场"此前只落在 HTML 报告里。
"""

import csv
from pathlib import Path

import pytest

from report_frontend.data_loader import LogDataLoader
import app as panel

_SID = "20260924_230914_262f"


def _write(log_dir: Path, filename: str, rows: list) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    fields = ["session_id", "timestamp", "connective_density", "connective_density_std", "n_rows"]
    p = log_dir / filename
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


def _row(sid: str, density: float | None = 2.5) -> dict:
    return {"session_id": sid, "timestamp": "2026-09-24T23:09:14",
            "connective_density": density, "connective_density_std": 0.1, "n_rows": 1}


@pytest.fixture
def client(tmp_path, monkeypatch):
    """把端点里 `LogDataLoader()` 的默认目录换到 tmp —— 端点用的是无参构造。"""
    def _make(files=()):
        log_dir = tmp_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in files:
            _write(log_dir, name, rows)
        # 端点是在函数体里 `from report_frontend.data_loader import LogDataLoader`,
        # 每次调用都重查模块属性 —— 所以补丁要打在**那个模块**上,而不是总控 `app` 上。
        import report_frontend.data_loader as dl
        monkeypatch.setattr(dl, "LogDataLoader", lambda *a, **k: LogDataLoader(str(log_dir)))
        return panel.app.test_client()

    return _make


def test_structured_report_names_the_session_it_describes(client):
    """★ 成功那一支必须带上 session_id 与三态来源。"""
    body = client([(f"face_au_log_{_SID}.csv", [_row(_SID)])]) \
        .get("/api/report/structured").get_json()

    assert body["status"] == "success", body
    assert body["session_id"] == _SID, f"JSON 里看不出这是哪一场:{body.keys()}"
    assert body["sources"]["face"]["status"] == "loaded", body["sources"]
    assert body["sources"]["gesture"]["status"] == "missing", body["sources"]


def test_structured_report_still_answers_when_the_target_has_no_logs(client):
    """★ 空数据不许变成 error —— spec §6 行 2:生成报告并显式说明本场没有任何日志。

    红法:退回 `if not data: return jsonify({"status":"error",...})`。
    """
    c = client([("face_au_log_NONE.csv", [_row("NONE", None)])])
    body = c.get(f"/api/report/structured?session_id={_SID}").get_json()

    assert body["status"] == "success", body
    assert body["session_id"] == _SID, "空数据时反而说不出描述的是哪一场"
    assert body["result"]["coverage"]["n_passed"] == 0, body["result"]["coverage"]
    # 分母也要钉住(复审 Minor 4):"0 / 20" 里的 20 是可回归的
    assert body["result"]["coverage"]["n_slots"] == 20, body["result"]["coverage"]
    assert all(v["status"] == "missing"
               for k, v in body["sources"].items() if k != "none_bucket"), body["sources"]


def test_structured_report_with_only_the_none_bucket_says_so(client):
    """只有 NONE 桶 → session_id 为 None,但 NONE 桶行数要在 sources 里报出来。"""
    c = client([("face_au_log_NONE.csv", [_row("NONE", None)])])
    body = c.get("/api/report/structured").get_json()

    assert body["status"] == "success", body
    assert body["session_id"] is None, body["session_id"]
    assert body["sources"]["none_bucket"]["rows"] == 1, body["sources"]
