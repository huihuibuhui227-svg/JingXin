# tests/test_session_purge_leaves_no_trace.py
"""「不留存」的语义是**当它没发生过** —— 派生品也算这一场的数据。

## 为什么这个文件存在

`session_purge.purge_session` 的 docstring 写着「把一场素材从盘上**彻底**抹掉」、
语义是「当它没发生过」,但实现**只删两处**:场次目录 + `data/logs/` 里四份按 sid
落盘的 CSV。2026-10-25 凌晨的审计实测,盘上还有这些**purge 碰不到**的痕迹:

| 位置 | 实测 | 带 sid? |
|---|---|---|
| `data/output/Research_Assessment_Report_*.html` | 48 份,其中 **42 份正文含 sid**(17 个不同 sid) | 是 |
| `data/output/` 图表 HTML(`radar_chart_*` / `evidence_*` / `gaze_trajectory_*`) | 644 份,2.9 GB | 文件名是 epoch,正文无 sid |
| `data/l0_series/<sid>.npz` | 本机暂空,但 `voice_series.py` 会写 | 是(sid 命名) |

也就是说:用户点「不留存」、界面上报成功、场次目录没了,**但一份带 sid 的报告
连同它引用的图表还躺在 `data/output/` 里**。这与本仓反复栽的形态同族 ——
「弹窗说成功、盘上没东西」的镜像:**弹窗说删掉了、盘上还在**。

## 这个文件守什么

1. 正文含本场 sid 的报告 ⟹ 删;
2. 那些报告引用的图表 ⟹ 删,**但被别的幸存报告引用的不许删**(实测 288 份图表
   被 48 份报告引用,**没有任何一份被两个 sid 共用** —— 这个判据现在恒真,
   但代码路径必须对,否则以后报告改成共用图表就会误删别人的图);
3. 正文**不含**本场 sid 的报告 ⟹ 不动(认不出归属的东西不许猜);
4. `data/l0_series/<sid>.npz` ⟹ 删;
5. 返回的 `removed` 里要**列出来删了什么** —— 与现有 `logs` 一个口径,
   让调用方能核对,而不是"相信它删了"。
"""
import asyncio
import importlib

import pytest

import media_retention as mr
import session_purge

SID = "20260928_212130_5b10"
OTHER = "20260928_180131_8102"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    root = tmp_path / "recordings"
    root.mkdir()
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(root))
    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(session_purge, "default_logs_dir", lambda: logs)
    out = tmp_path / "output"
    out.mkdir()
    series = tmp_path / "l0_series"
    series.mkdir()
    monkeypatch.setattr(session_purge, "default_output_dir", lambda: out, raising=False)
    monkeypatch.setattr(session_purge, "default_series_dir", lambda: series, raising=False)
    mr._reset_for_tests()
    return root, logs, out, series


def _session(root, sid):
    d = root / f"18-赵六-2021004-物理学院__{sid}"
    (d / "media" / "face").mkdir(parents=True)
    (d / "media" / "face" / "000001.jpg").write_bytes(b"\xff\xd8\xff")
    return d


def _report(out, name, sid, charts=()):
    """造一份报告:正文里出现 sid 与它引用的图表文件名。"""
    links = "\n".join(f'<img src="{c}">' for c in charts)
    body = f"<html><body><h1>报告</h1><p>场次 {sid}</p>{links}</body></html>"
    p = out / name
    p.write_text(body, encoding="utf-8")
    return p


def _chart(out, name):
    p = out / name
    p.write_text("<html>chart</html>", encoding="utf-8")
    return p


def test_report_containing_the_sid_is_deleted(_isolated):
    """★ 红法:把 `purge_session` 里扫 `output_dir` 那一段去掉 ⟹ 报告还在。

    「不留存」之后盘上留着一份写着这个 sid 的 HTML,等于**没有作废**。
    """
    root, logs, out, series = _isolated
    _session(root, SID)
    mine = _report(out, "Research_Assessment_Report_20260928_212200.html", SID)
    other = _report(out, "Research_Assessment_Report_20260928_180200.html", OTHER)

    removed = session_purge.purge_session(SID, logs_dir=logs)

    assert not mine.exists(), "正文含本场 sid 的报告没被删"
    assert other.exists(), "别的场次的报告被误删了"
    assert "Research_Assessment_Report_20260928_212200.html" in removed["reports"], (
        f"删了却没在返回值里说:{removed}")


def test_charts_of_a_deleted_report_go_with_it(_isolated):
    """★ 红法:只删报告、不删它引用的图表 ⟹ 图表还在。

    图表文件名是 epoch、正文里没有 sid,所以**只能顺着报告找**。
    留下的图表是一张面部 AU / 视线曲线图 —— 值层面仍然是这一场的数据。
    """
    root, logs, out, series = _isolated
    _session(root, SID)
    _chart(out, "radar_chart_1773475158.html")
    _chart(out, "evidence_cognitive_efficiency_1773475158.html")
    _report(out, "Research_Assessment_Report_20260928_212200.html", SID,
            charts=("radar_chart_1773475158.html",
                    "evidence_cognitive_efficiency_1773475158.html"))

    removed = session_purge.purge_session(SID, logs_dir=logs)

    assert not (out / "radar_chart_1773475158.html").exists(), "报告删了、图还在"
    assert not (out / "evidence_cognitive_efficiency_1773475158.html").exists()
    assert sorted(removed["charts"]) == [
        "evidence_cognitive_efficiency_1773475158.html",
        "radar_chart_1773475158.html",
    ], f"返回值没列出删掉的图表:{removed}"


def test_a_chart_shared_with_a_surviving_report_is_kept(_isolated):
    """★ 红法:把「被幸存报告引用就不删」那个判断去掉 ⟹ 本场红。

    实测当前 288 份被引用的图表**没有一份被两个 sid 共用**,所以这条判据现在
    恒真 —— 但**恒真的判据更要写**:报告层哪天改成共用图表(比如同一场多份报告),
    少这个判断就会删掉别人报告里的图,而那种坏法很难在界面上看出来。
    """
    root, logs, out, series = _isolated
    _session(root, SID)
    _chart(out, "radar_chart_1773475158.html")
    _report(out, "Research_Assessment_Report_20260928_212200.html", SID,
            charts=("radar_chart_1773475158.html",))
    # 另一场次的报告也引用同一张图
    _report(out, "Research_Assessment_Report_20260928_180200.html", OTHER,
            charts=("radar_chart_1773475158.html",))

    removed = session_purge.purge_session(SID, logs_dir=logs)

    assert (out / "radar_chart_1773475158.html").exists(), (
        "删掉了一张还被别的场次报告引用的图 —— 那份报告会显示破图")
    assert removed["charts"] == [], f"没删却报了图表:{removed['charts']}"


def test_report_without_the_sid_is_not_guessed_at(_isolated):
    """★ 红法:把「正文含 sid」这个条件改成「文件名/全部报告」⟹ 本条红。

    盘上 48 份报告里有 **6 份正文不含任何 sid**(旧格式 / 生成时没拿到场次)。
    认不出归属的东西**不许猜** —— 猜错就是删掉别人的报告。
    """
    root, logs, out, series = _isolated
    _session(root, SID)
    anonymous = _report(out, "Research_Assessment_Report_20260901_101010.html", "")

    removed = session_purge.purge_session(SID, logs_dir=logs)

    assert anonymous.exists(), "正文里没有 sid 的报告被猜着删了"
    assert removed["reports"] == [], f"没删却报了报告:{removed['reports']}"


def test_l0_series_npz_is_deleted(_isolated):
    """★ 红法:把 `series_dir` 那一段去掉 ⟹ npz 还在。

    `report_frontend/voice_series.py` 按 `<sid>.npz` 写 f0/rms 序列 ——
    那是这一场嗓音频谱的逐帧序列,与 `data/logs` 里那几份同性质。
    """
    root, logs, out, series = _isolated
    _session(root, SID)
    npz = series / f"{SID}.npz"
    npz.write_bytes(b"PK\x03\x04")
    other_npz = series / f"{OTHER}.npz"
    other_npz.write_bytes(b"PK\x03\x04")

    removed = session_purge.purge_session(SID, logs_dir=logs)

    assert not npz.exists(), "本场的 l0 序列没删"
    assert other_npz.exists(), "别的场次的序列被误删了"
    assert f"{SID}.npz" in removed["series"], f"返回值没列出序列:{removed}"


def test_purging_a_session_with_only_derived_artifacts_is_not_a_filenotfound(_isolated):
    """★ 红法:让 `FileNotFoundError` 的判据只看目录与日志 ⟹ 本条红。

    盘上实测有 **7 个报告里的 sid 有日志、但没有录制目录**。若判据只看前两处,
    这种场次 purge 会抛 `FileNotFoundError` —— 而它明明删掉了报告,调用方却
    以为"没有这一场",于是报错退出、报告删了一半。
    """
    root, logs, out, series = _isolated
    # 不建场次目录、不建日志 CSV,只留一份报告
    _report(out, "Research_Assessment_Report_20260928_212200.html", SID)

    removed = session_purge.purge_session(SID, logs_dir=logs)

    assert removed["reports"] == ["Research_Assessment_Report_20260928_212200.html"]


def test_panel_purge_uses_the_panel_output_dir_not_the_module_default(_isolated, tmp_path,
                                                                    monkeypatch):
    """面板那条 purge 必须删**面板自己那个落点**的报告,不是模块默认的 `data/output`。

    ⚠️ 两个目录今天是同一个,所以这条判据在真盘上**看不出来** —— 正是这个原因
    才要钉:哪天面板的落点挪了(或测试隔离要求分开),`purge_session(sid)` 会
    静默退回 `default_output_dir()`,于是"面板报成功、报告还在"。
    红法(实测):把 `output_dir=current_app.config.get("OUTPUT_DIR")` 从
    `recordings_browser.recording_purge` 里去掉 ⟹ 本条红。
    """
    import recordings_browser as rb

    root, logs, _module_default_out, _series = _isolated
    monkeypatch.setenv(rb.ADMIN_PASSWORD_ENV, "s3cret")
    panel_out = tmp_path / "panel_output"
    panel_out.mkdir()

    _session(root, SID)
    report = _report(panel_out, "Research_Assessment_Report_20260928_212200.html", SID)

    # 用**真的**面板 app(蓝图 + 管理员门 + token)而不是另起一个 Flask ——
    # 另起的话注册的蓝图与生产那份不是同一个 `current_app.config` 读法。
    panel = importlib.import_module("app")
    monkeypatch.setitem(panel.app.config, "OUTPUT_DIR", str(panel_out))
    monkeypatch.setitem(panel.app.config, "LOGS_DIR", str(logs))
    c = panel.app.test_client()
    tok = c.post("/api/admin/login", json={"password": "s3cret"}).get_json()["token"]
    c.environ_base["HTTP_AUTHORIZATION"] = f"Bearer {tok}"

    r = c.post(f"/api/recordings/{SID}/purge", json={"confirm": SID})
    assert r.status_code == 200, r.get_json()

    assert not report.exists(), (
        "面板的 purge 没删面板落点里的报告 —— 它退回了模块默认的 data/output")


def test_discard_endpoint_reaches_the_derived_artifacts(_isolated):
    """端到端:录制页点「不留存」走的是这个端点,不是直接调 `purge_session`。

    红法:把端点里传给 `purge_session` 的参数改回只传 `logs_dir` ⟹ 本条红。
    与 `test_answer_endpoint_writes_no_artifact_per_question` 那两条一样:
    单独测函数**证明不了**接线接上了。
    """
    root, logs, out, series = _isolated
    _session(root, SID)
    report = _report(out, "Research_Assessment_Report_20260928_212200.html", SID)

    voice_app = importlib.import_module("voice_interaction.api.app")
    asyncio.run(voice_app.discard_session(session_id=SID))

    assert not report.exists(), "端点没把派生品一起删 —— 用户看到的是「已作废」"
