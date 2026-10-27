"""数据导出:CSV 清单 / 转写导出 / 图表 PNG。

这些接口不参与报告口径，但**导出的内容会被拿去做分析** —— 所以至少压住：
* 清单只说盘上事实（存在/行数/大小）；
* 转写里带得上题干与标注（否则一堆 sid 没法对号）；
* 图能在缺列/空场时给出"没数据"，而不是画一张空图。
"""
import json

import pytest

import exports

SID = "20261026_002930_cb79"

FACE = (
    "session_id,timestamp,au12_smile,au4_frown,gaze_direction_x,gaze_direction_y\n"
    + "".join(f"{SID},{i},0.1,0.2,0.01,-0.02\n" for i in range(4))
)
GESTURE = (
    "session_id,timestamp,left_hand_jitter,right_hand_jitter,left_hand_spread,right_hand_spread\n"
    + "".join(f"{SID},{1792945700 + i},0.01,0.02,2.0,1.5\n" for i in range(3))
)
VOICE = (
    "session_id,timestamp,pitch_mean,energy_mean,duration_sec,connective_density\n"
    + "".join(f"{SID},2026-10-26T00:31:0{i},100,0.05,8.0,0.0\n" for i in range(3))
)


@pytest.fixture
def session(tmp_path, monkeypatch):
    """一场有面板/语音日志、有转写与标注的假会话。"""
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / f"face_au_log_{SID}.csv").write_text(FACE, encoding="utf-8")
    (logs / f"gesture_emotion_log_{SID}.csv").write_text(GESTURE, encoding="utf-8")
    (logs / f"research_emotion_log_{SID}.csv").write_text(VOICE, encoding="utf-8")
    monkeypatch.setattr(exports, "_logs_dir", lambda: logs)

    sess = tmp_path / f"1026-候选Y__{SID}"
    sess.mkdir()
    (sess / "label.json").write_text(json.dumps({
        "session_id": SID, "name": "候选Y", "student_id": "900000001",
        "department": "计算机与人工智能学院", "serial": "1026",
    }, ensure_ascii=False), encoding="utf-8")
    (sess / "questions.jsonl").write_text(
        json.dumps({"index": 0, "qid": "请描述一个你研究过的问题？"}, ensure_ascii=False) + "\n",
        encoding="utf-8")
    (sess / "transcript.json").write_text(json.dumps({
        "session_id": SID,
        "recorded_at": "2026-10-26T00:31:02+08:00",
        "merged": {"text": "回答一", "n_chars": 3},
        "segments": [{"index": 0, "text": "回答一"}],
    }, ensure_ascii=False), encoding="utf-8")
    return SID


def test_files_lists_what_is_on_disk(session):
    d = exports.list_payload(session)
    by_key = {x["key"]: x for x in d["logs"]}
    assert by_key["face"]["exists"] and by_key["face"]["rows"] == 4
    assert by_key["gesture"]["exists"]
    assert by_key["voice_research"]["exists"]
    assert not by_key["voice_interview"]["exists"]      # 面试那份本来就没有
    assert d["transcript"]["exists"]
    assert d["transcript"]["n_segments"] == 1
    assert d["questions"] == 1


def test_transcript_txt_carries_subject_and_question(session):
    txt = exports.transcript_txt(session)
    assert "候选Y" in txt and "900000001" in txt
    assert "请描述一个你研究过的问题？" in txt
    assert "回答一" in txt


def test_transcript_csv_has_subject_column(session):
    text = exports.transcript_csv(session)
    first = text.splitlines()[0]
    assert first.startswith("index,subject,question,answer,n_chars")
    assert "候选Y" in text.splitlines()[1]


def test_chart_png_is_a_real_png(session):
    png = exports.chart_png(session, "face")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 5000


def test_chart_without_data_says_so(tmp_path, monkeypatch):
    """没有这一场的日志 ⟹ 抛 NoData（路由转 404），不画空图。"""
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(exports, "_logs_dir", lambda: logs)
    with pytest.raises(exports.NoData):
        exports.chart_png(SID, "face")


# ── 守卫:导出的是**受试者的原始数据**,未登录一条都不许给 ────────────────────
#
# 2026-10-27 实测补的洞:这五个路由此前**一条守卫都没有**。同一场会话在素材页
# `/api/recordings/<sid>` 未登录是 401,而 `/api/export/<sid>/transcript` 未登录
# 回了 200 + 10 KB 的转写原文(含姓名/学号/院系)。两条路对同一份数据给出不同
# 保护 ⟹ 实际保护等于最松的那条。
#
# 断言用"允许清单"而不是"禁止清单":禁止清单会随着新字段静默失效。

import app as panel          # noqa: E402  (放后面:上面那批是纯函数,不需要 Flask)


@pytest.fixture
def client(session, tmp_path, monkeypatch):
    """一场假会话 + 一个没登录的面板客户端。密码配上,否则 503 会盖住 401。"""
    monkeypatch.setenv("JX_ADMIN_PASSWORD", "test-password")
    monkeypatch.setitem(panel.app.config, "LOGS_DIR", str(tmp_path / "logs"))
    panel.app.config["TESTING"] = True
    return panel.app.test_client()


_UNAUTH_URLS = (
    "/api/export/sessions",
    f"/api/export/{SID}/files",
    f"/api/export/{SID}/csv/face",
    f"/api/export/{SID}/transcript?format=json",
    f"/api/export/{SID}/chart?module=face",
)


@pytest.mark.parametrize("url", _UNAUTH_URLS)
def test_export_endpoints_refuse_anonymous(client, url):
    """未登录一律 401。

    红法:去掉 `exports.py` 里任意一条 `@_admin_required(...)` ⟹ 对应那条红。
    """
    r = client.get(url)
    assert r.status_code == 401, f"{url} 未登录回了 {r.status_code}"


@pytest.mark.parametrize("url", _UNAUTH_URLS)
def test_anonymous_response_leaks_no_identity(client, url):
    """401 的正文里也不许出现身份(错误信息最容易顺手带上)。"""
    body = client.get(url).get_data(as_text=True)
    for probe in ("候选Y", "900000001", "示例学院"):
        assert probe not in body, f"{url} 的 401 正文里带了身份:{probe}"


def test_admin_token_unlocks_the_same_endpoints(client):
    """拿到 token 之后同一批地址要能通 —— 否则守卫就变成"锁死",不是"加锁"。"""
    r = client.post("/api/admin/login", json={"password": "test-password"})
    assert r.status_code == 200, r.get_data(as_text=True)
    token = r.get_json()["token"]
    h = {"Authorization": f"Bearer {token}"}

    r = client.get("/api/export/sessions", headers=h)
    assert r.status_code == 200 and "sessions" in r.get_json()

    r = client.get(f"/api/export/{SID}/files", headers=h)
    assert r.status_code == 200 and r.get_json()["session_id"] == SID

    r = client.get(f"/api/export/{SID}/transcript?format=json", headers=h)
    assert r.status_code == 200 and r.get_json()["session_id"] == SID


def test_download_links_may_carry_the_token_in_the_query(client):
    """`<a download>` / `<img src>` 发不了自定义头 ⟹ 这三条**必须**认 query 里的 token。

    红法:把 csv / transcript / chart 的 `allow_query=True` 去掉 ⟹ 本条红,
    而前端那两个"下载"按钮会静默变成 401(浏览器只会存下一张错误页)。
    """
    r = client.post("/api/admin/login", json={"password": "test-password"})
    token = r.get_json()["token"]

    for url in (f"/api/export/{SID}/csv/face",
                f"/api/export/{SID}/transcript?format=txt",
                f"/api/export/{SID}/chart?module=face"):
        sep = "&" if "?" in url else "?"
        got = client.get(f"{url}{sep}token={token}")
        assert got.status_code == 200, f"{url} 带 query token 仍被拒({got.status_code})"


def test_json_endpoints_stay_header_only(client):
    """JSON 接口**不认** query token —— 凭证暴露面越小越好(与素材页同一口径)。"""
    r = client.post("/api/admin/login", json={"password": "test-password"})
    token = r.get_json()["token"]
    assert client.get(f"/api/export/sessions?token={token}").status_code == 401
    assert client.get(f"/api/export/{SID}/files?token={token}").status_code == 401


def test_guard_is_the_same_one_the_recordings_page_uses():
    """鉴权**只有一处定义**。导出页自己写一份 ⟹ 两页的"谁能看"迟早分叉。"""
    import recordings_browser as rb
    assert exports._admin_required is rb._admin_required
