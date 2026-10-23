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
    # ⚠️ 2026-10-18:本条钉的是**旧引擎**的契约(`n_slots == 14` 个声明槽)。
    #    L1 引擎的分母也是 14,但它还会读两个本条没隔离的旁路 —— 录制目录的
    #    帧账本与语音序列缓存 —— 于是"日志目录里什么都没有"不等于"没有证据"。
    #    L1 那条契约由下面 `test_l1_engine_*` 在**完全隔离**下单独钉。
    c = client([("face_au_log_NONE.csv", [_row("NONE", None)])])
    body = c.get(f"/api/report/structured?session_id={_SID}&engine=legacy").get_json()

    assert body["status"] == "success", body
    assert body["session_id"] == _SID, "空数据时反而说不出描述的是哪一场"
    assert body["result"]["coverage"]["n_passed"] == 0, body["result"]["coverage"]
    # 分母要钉住(复审 Minor 4)—— 它是可回归的。
    # ⚠️ 这个数**抄过三次、错了两次**(20 → 17 → 15),每次都是"人工记得改";
    #    2026-10-03 丙-17 第二批删槽(`au7_freq` / `blink_rate`)时**又漏了它**,
    #    是 `tools/jx_selfcheck.py --all` 跑全量才抓出来的。所以现在改成**两条**:
    #    ① 端点报的数必须等于槽位表里**声明槽的总数**(推导,不会过期);
    #    ② 推导出来的总数钉在一个字面量上(那一条才是"槽表被改过"的警报)。
    from report_frontend.research_mapper import ResearchCapabilityMapper

    declared = sum(len(s["indicators"])
                   for s in ResearchCapabilityMapper().mapping_rules.values())
    got = body["result"]["coverage"]["n_slots"]
    assert got == declared, (
        f"端点报的槽数 {got} 与槽位表声明的 {declared} 不符 —— "
        f"覆盖率的分母必须数全**声明**的槽(包括永不可能过门的那些,那是刻意的)")
    assert declared == 14, (
        f"槽位表的声明槽总数从 15 变成了 {declared}。若是有意改动(加/删槽),"
        f"请一并同步:`docs/下一步.md`「丙、17 / 丙、18」的两张表、"
        f"`tests/test_dimension_live_weight.py` 的 `_DEAD_WEIGHT`、"
        f"以及本文件这个数。若并非有意 —— 那是槽表被误动,先查再改。")
    assert all(v["status"] == "missing"
               for k, v in body["sources"].items() if k != "none_bucket"), body["sources"]


def test_structured_report_with_only_the_none_bucket_says_so(client):
    """只有 NONE 桶 → session_id 为 None,但 NONE 桶行数要在 sources 里报出来。"""
    c = client([("face_au_log_NONE.csv", [_row("NONE", None)])])
    body = c.get("/api/report/structured").get_json()

    assert body["status"] == "success", body
    assert body["session_id"] is None, body["session_id"]
    assert body["sources"]["none_bucket"]["rows"] == 1, body["sources"]


# ── L1 引擎(M4.5,2026-10-18 起是默认引擎)────────────────────────────────

@pytest.fixture
def isolated(client, tmp_path, monkeypatch):
    """把 L1 会读的**三条旁路**全指向空目录。

    L1 不只读 `data/logs`:`session_timeline` 要读录制目录里的帧账本(拿 `declared_ts`),
    `voice_series` 要读序列缓存。不隔离的话,"日志目录里没有"仍可能量出值来 ——
    那是**测试环境泄漏**,不是被测行为。
    """
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path / "no_recordings"))
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path / "no_series"))
    return client


def test_l1_engine_answers_with_no_logs_and_says_why(isolated):
    """★ 空数据下 L1 引擎:照常回答、说出是哪一场、**并且说出为什么薄**。

    红法:把 `build_l1_report` 的 `evidence_gaps` 去掉 ⟹ 本条红 ——
    那样报告只会说"0/3 过门",而读者不知道是"没采到"还是"没这个量"。
    """
    body = isolated([("face_au_log_NONE.csv", [_row("NONE", None)])]) \
        .get(f"/api/report/structured?session_id={_SID}").get_json()

    assert body["status"] == "success", body
    assert body["engine"] == "l1", body.get("engine")
    assert body["session_id"] == _SID

    res = body["result"]
    assert res["coverage"]["n_passed"] == 0, res["coverage"]
    # 分母仍是**声明的全部旧槽数** —— 消失的维度那 3 个槽也算在内。
    assert res["coverage"]["n_slots"] == 14, res["coverage"]
    # 2026-10-18 T 族落地后,`connective_density` 不再是"删"而是"换"
    # (`discourse_marker_rate`)⟹ 删掉的从 7 个变 6 个。
    assert res["coverage"]["n_dropped"] == 6, res["coverage"]

    # 消失的维度要能看见,且带理由
    vanished = {v["key"]: v for v in res["vanished_dimensions"]}
    assert "confidence_level" in vanished
    assert vanished["confidence_level"]["n_slots"] == 3

    # 每个维度都要说清"为什么薄"
    for key, dim in res["dimensions"].items():
        assert dim["evidence_gaps"], f"{key} 没有缺口说明"


def test_l1_engine_replaces_the_false_algorithm_statement(isolated):
    """★ 旧载荷每个维度都写 `algorithm: "加权线性组合 + 认知负荷推断"` —— L1 下这句**不成立**。

    这一层没有加权组合(权重只用于删槽后的记账,不参与求和),也没有认知负荷推断。

    红法:把 `DIM_ALGORITHM` 改回那句旧话 ⟹ 本条红。
    """
    body = isolated([("face_au_log_NONE.csv", [_row("NONE", None)])]) \
        .get(f"/api/report/structured?session_id={_SID}").get_json()

    for key, dim in body["result"]["dimensions"].items():
        assert "加权线性组合" not in dim["algorithm"], (key, dim["algorithm"])
        assert "认知负荷" not in dim["algorithm"], (key, dim["algorithm"])
        assert "推断" not in dim["algorithm"], (key, dim["algorithm"])


def test_l1_engine_payload_carries_no_scoring_keys(isolated):
    """★ 没有标定样本 ⟹ 载荷里不许出现 `score` / `level` / `percentile`。

    红法:在 `build_l1_report` 的维度字典里加一行 `"score": ...` ⟹ 本条红。
    类型层是"悄悄加回来"的最后一道闸,但类型只在前端;后端也要有一条。
    """
    body = isolated([("face_au_log_NONE.csv", [_row("NONE", None)])]) \
        .get(f"/api/report/structured?session_id={_SID}").get_json()

    res = body["result"]
    assert "total_score" not in res and "total_level" not in res
    for key, dim in res["dimensions"].items():
        assert "score" not in dim and "level" not in dim, key
        for stat in dim["stats"].values():
            assert "percentile" not in stat, (key, stat)


def test_unknown_engine_is_a_400_not_a_silent_fallback(client):
    """★ 拼错引擎名必须报错,不许**静默**退回某个默认值。

    红法:把 `else: return 400` 改成 `which = "l1"` ⟹ 本条红。
    静默兜底会让"我请求的是旧引擎"和"实际拿到的是新引擎"分不清。
    """
    r = client().get(f"/api/report/structured?session_id={_SID}&engine=L2")
    assert r.status_code == 400
    assert "engine" in r.get_json()["message"]
