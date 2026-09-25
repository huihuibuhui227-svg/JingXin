# tests/test_panel_report_task_status.py
"""第 17 条同族:面板的**实时**报告路径也不许「没产出却报成功」。

失效现场(2026-09-24 复审 I2 的邻居;2026-09-25 M2.1 独立复审点名):
`app.py` 的 `_run_live_report` 拿到 `generate_report_live()` 的 `""`(失败时它正是
`return ""`)之后**无条件**写 `"status": "success"`,只在 `logs` 里塞一句「生成失败」——
`/api/task/<id>` 于是把"什么都没生成"报成「实时报告生成完成！」。
与 Task 1 修掉的 CLI 那条路**逐字同形**,只是走在隔壁。

红法:把 `"status": "success" if path else "error"` 改回恒 `"success"`。
"""

import app as panel


def test_live_report_task_reports_error_when_nothing_was_generated(monkeypatch):
    monkeypatch.setattr(
        "report_frontend.report_generator.ReportGenerator.generate_report_live",
        lambda self, sid: "")
    panel.task_status["probe-live"] = {"module": "report_live", "status": "pending"}

    panel._run_live_report("probe-live", "20260924_230914_262f")

    assert panel.task_status["probe-live"]["status"] == "error", (
        f"没产出却报成功:{panel.task_status['probe-live']}")


def test_live_report_task_still_reports_success_when_it_worked(monkeypatch):
    """另一侧:真的产出了就照旧 `success`,并把路径写进 logs(别把成功路径一起弄坏)。"""
    monkeypatch.setattr(
        "report_frontend.report_generator.ReportGenerator.generate_report_live",
        lambda self, sid: "/tmp/live_report.html")
    panel.task_status["probe-live-ok"] = {"module": "report_live", "status": "pending"}

    panel._run_live_report("probe-live-ok", "20260924_230914_262f")

    st = panel.task_status["probe-live-ok"]
    assert st["status"] == "success" and "/tmp/live_report.html" in st["logs"], st


def test_live_report_task_reports_error_when_the_generator_raises(monkeypatch):
    """抛异常那条路原本就是 error(既有行为),一起钉住,免得上面的改动把它带坏。"""
    def _boom(self, sid):
        raise RuntimeError("API 取不到实时数据")

    monkeypatch.setattr(
        "report_frontend.report_generator.ReportGenerator.generate_report_live", _boom)
    panel.task_status["probe-live-boom"] = {"module": "report_live", "status": "pending"}

    panel._run_live_report("probe-live-boom", "20260924_230914_262f")

    assert panel.task_status["probe-live-boom"]["status"] == "error"
