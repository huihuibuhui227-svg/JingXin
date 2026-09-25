# tests/test_report_empty_session.py
"""第 17 条(复审 I2):**本场没有任何日志时,报告照样要出**。

spec §6 的三行契约:
    目标 id 一个模态都没找到 → 报告仍生成,头里逐条列出缺什么;不抛
    目标 id 全部模态都缺     → 生成报告并显式说明本场没有任何日志;不回退去拼别的场次
    前端在拿到铸号前发帧     → 报告头要点出 NONE 桶存在

失效现场(2026-09-24 复审实测):`report_frontend/report_generator.py:121` 的
`raise ValueError("无面部数据")` 被同一个 try 的 `except` 吞成 `return ""` ——
**一份报告都没有**;而 `app.py:53` 只看子进程 returncode,于是面板报「任务完成！」。
"""

import csv
from pathlib import Path

import pytest

import report_frontend.report_generator as report_generator
from report_frontend.data_loader import LogDataLoader
from report_frontend.report_generator import ReportGenerator, main, sources_disclosure

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


def _none_row() -> dict:
    return {"session_id": "NONE", "timestamp": "2026-09-24T23:00:35",
            "connective_density": None, "connective_density_std": None, "n_rows": 1}


def _session_row(sid: str) -> dict:
    return {"session_id": sid, "timestamp": "2026-09-24T23:09:14",
            "connective_density": 2.5, "connective_density_std": 0.1, "n_rows": 1}


@pytest.fixture
def render(tmp_path, monkeypatch):
    """真加载器 + 真渲染,只把日志目录与"打开浏览器"换掉。

    **不** import 总控 `app.py`(端点的契约由 Task 2 的测试压),这里只压报告这条链。
    """
    def _run(session_id=None, files=()) -> str:
        log_dir = tmp_path / "logs"
        # `data/logs` 一定存在(加载器按设计要求它存在,不存在就直接报错)—— 所以
        # `files=[]` 这个"一份日志都没有"的场景也要先把空目录建出来,否则测的是另一件事。
        log_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in files:
            _write(log_dir, name, rows)
        monkeypatch.setattr(report_generator, "LogDataLoader",
                            lambda *a, **k: LogDataLoader(str(log_dir)))
        monkeypatch.setattr(report_generator.webbrowser, "open", lambda *a, **k: None)

        path = ReportGenerator(output_dir=str(tmp_path / "out")).generate_report(session_id)
        assert path, "报告没生成 —— 本测试的前提不成立"
        return Path(path).read_text(encoding="utf-8")

    return _run


def test_report_is_still_generated_when_only_the_none_bucket_exists(render):
    """★ 只有 NONE 桶(前端还没拿到铸号就发帧)→ 报告仍出,且点出 NONE 桶行数。

    红法:恢复 `if not data or 'face' not in data: raise ValueError("无面部数据")`
    → `generate_report` 返回 `''` → 上面的 `assert path` 先红。
    """
    html = render(files=[("face_au_log_NONE.csv", [_none_row()])])

    assert "本场没有任何日志" in html, "只有 NONE 桶时没说清「本场根本没有日志」"
    assert "NONE 桶 1 行" in html, "存在 NONE 桶却没在报告头点出来(spec §6 行 3)"


def test_report_is_still_generated_when_there_are_no_logs_at_all(render):
    """★ 连 NONE 桶都没有 → 仍要出一份报告,并写明本场没有任何日志(Review Focus 第 1 条)。

    红法:同上的 `raise`;另一条红法是让 `sources_disclosure` 在 items 为空时走早退
    (旧的两处 `return "本报告没有装配任何模态日志。"`)→ 句子里没有「本场没有任何日志」。
    """
    html = render(files=[])

    assert "本场没有任何日志" in html


def test_report_lists_every_missing_modality_when_the_named_session_has_no_logs(render):
    """★ 显式指了一个没有日志的 id → 报告仍生成,四个模态逐条列「缺失」。

    红法:同上的 `raise`。
    """
    html = render(session_id=_SID, files=[("face_au_log_NONE.csv", [_none_row()])])

    assert _SID in html, "报告没点名它描述的是哪个 session_id"
    missing = "缺失（本场没有这个模态的日志）"
    assert html.count(missing) == 4, (
        f"没有逐条列出四个模态缺什么,实际命中 {html.count(missing)} 次")


def test_a_loaded_session_still_renders_normally(render):
    """反向的那一侧:有数据的会话照旧出报告(改动不许把正常路径弄坏)。"""
    html = render(session_id=_SID,
                  files=[(f"face_au_log_{_SID}.csv", [_session_row(_SID)])])

    assert _SID in html and "已读入" in html
    assert "本场没有任何日志" not in html


def test_disclosure_without_a_target_says_there_is_no_session():
    """`target` 不给且没有任何模态条目 → 必须说「本场没有任何日志」,不许编一个 id 出来。

    红法:退回旧的两处早退(`if not items: return "本报告没有装配任何模态日志。"`)
    → 句子里没有「本场没有任何日志」→ 红。
    """
    html = sources_disclosure(
        {"none_bucket": {"session_id": "NONE", "status": "present", "rows": 3}})

    assert "本场没有任何日志" in html
    assert "NONE 桶 3 行" in html


def test_cli_exits_nonzero_when_no_report_was_written(monkeypatch):
    """★ 复审 I2 的额外症状:面板只看子进程 returncode ——
    "什么都没生成但 exit 0" 会被显示成「任务完成！」(`app.py:53`)。

    红法:退回旧的 `__main__`(不设退出码)→ `main` 不存在 → ImportError 红;
    若只把 `main` 写成恒返回 0,则本条断言红。
    """
    monkeypatch.setattr(ReportGenerator, "generate_report", lambda self, sid=None: "")

    assert main([]) == 1


def test_cli_exits_zero_and_passes_the_session_id_through(monkeypatch):
    """`--session-id` 要真的传下去,产出了报告才返回 0。"""
    seen = []
    monkeypatch.setattr(ReportGenerator, "generate_report",
                        lambda self, sid=None: seen.append(sid) or "/tmp/x.html")

    assert main(["--session-id", _SID]) == 0
    assert seen == [_SID]


# ── M2.1 独立复审(2026-09-25)补的两条 ──────────────────────────────────

_OTHER = "20260924_221811_9212"     # 旁边那场**有数据**的会话


def test_report_never_falls_back_to_another_session_with_data(render):
    """★ spec §6 行 2 的后半句:**不许**回退去拼别的场次(复审 Minor 3)。

    上面那条「显式不存在 id」的测试,目录里**只有 NONE 文件**,没有第二场可回退 ——
    所以它压不住这条契约。这里另造一场有数据的会话,目标仍是那个空 id。

    红法:把 `resolve_target_session` 改成"显式 id 找不到就取最新一场" → 报告头会点名
    `_OTHER` 并显示「已读入」→ 两条断言都红。
    """
    html = render(session_id=_SID,
                  files=[(f"face_au_log_{_OTHER}.csv", [_session_row(_OTHER)])])

    assert _SID in html
    assert _OTHER not in html, "回退去描述了旁边那场有数据的会话 —— spec §6 行 2 明令不许"
    assert "已读入" not in html


def test_a_failing_browser_open_does_not_turn_a_written_report_into_a_failure(
        tmp_path, monkeypatch):
    """★ 报告**已经落盘**了 —— 打开浏览器失败不许把返回值变成 `''`(复审 Minor 5)。

    为什么这条要紧:退出码是本轮新引入的(第 17 条),`return ""` 会被面板报成
    「任务失败」,而盘上明明有一份报告 —— 一个假警报。

    红法:把 `webbrowser.open` 挪回主 `try` 内(它就在 `write` 之后)。
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    _write(log_dir, f"face_au_log_{_SID}.csv", [_session_row(_SID)])
    monkeypatch.setattr(report_generator, "LogDataLoader",
                        lambda *a, **k: LogDataLoader(str(log_dir)))

    def _boom(url):
        raise RuntimeError("这台机器没有可用浏览器")

    monkeypatch.setattr(report_generator.webbrowser, "open", _boom)

    path = ReportGenerator(output_dir=str(tmp_path / "out")).generate_report(_SID)

    assert path, "报告已经落盘,却因为打不开浏览器被判成了失败"
    assert Path(path).exists()
