# tests/test_report_sources.py
"""I4 的原意 + M2 的新契约:报告必须说清**自己是哪一场的、每个模态进来了没有**。

I4(2026-09-24 最终审查)当时要解决的是"跨场拼接在报告里是隐形的"。那时选择策略还是
"每模态按文件名时间戳取最新",所以披露写成了**同场 / 不同场**两态。

**M2 把选择策略修好了**(按 `session_id` 选,spec `2026-09-24-m2-read-side-session-alignment-design.md`),
于是"不同场"这个状态**不再存在** —— 披露改为报**哪几个模态没进来、为什么**:

    loaded / unreadable(文件在但读不出) / missing(本场没有)

⚠️ 选取与披露的**单元级**覆盖已移到 `tests/test_read_side_session_selection.py`。
本文件保留它独有的价值:**走完整 `generate_report` 路径**的集成测(真加载器 → 真渲染)。
"""

import csv
from pathlib import Path

import pytest

import report_frontend.report_generator as report_generator
from report_frontend.data_loader import LogDataLoader
from report_frontend.report_generator import ReportGenerator

_SID = "20260924_221811_9212"      # 一场会话只有一个 id —— 三个模态都用它


def _write_log(log_dir: Path, filename: str, rows: list) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    fields = ["session_id", "timestamp", "connective_density", "connective_density_std",
              "n_rows", "focus_score", "symmetry_score"]
    p = log_dir / filename
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return p


def _row(sid: str, density=None) -> dict:
    return {"session_id": sid, "timestamp": "2026-09-24T22:18:11",
            "connective_density": density, "connective_density_std": 0.5, "n_rows": 1,
            "focus_score": 0.5, "symmetry_score": 0.9}


def _scene(log_dir: Path, voice_rows: int) -> None:
    """**同一场**的三个模态(这才是真实会话的形态)。`voice_rows=0` → 语音只有表头。"""
    _write_log(log_dir, f"face_au_log_{_SID}.csv", [_row(_SID)])
    _write_log(log_dir, f"gesture_emotion_log_{_SID}.csv", [_row(_SID)])
    _write_log(log_dir, f"interview_emotion_log_{_SID}.csv",
               [_row(_SID, 3.0 + i) for i in range(voice_rows)])


@pytest.fixture
def batch_report(tmp_path, monkeypatch):
    """把 `generate_report` 的加载目录与浏览器打开都换掉,只保留装配与渲染。"""
    def _run(voice_rows: int) -> str:
        log_dir = tmp_path / "logs"
        _scene(log_dir, voice_rows)
        monkeypatch.setattr(report_generator, "LogDataLoader",
                            lambda *a, **k: LogDataLoader(str(log_dir)))
        monkeypatch.setattr(report_generator.webbrowser, "open", lambda *a, **k: None)

        gen = ReportGenerator(output_dir=str(tmp_path / "out"))
        path = gen.generate_report()
        assert path, "报告没生成 —— 本测试的前提不成立"
        return Path(path).read_text(encoding="utf-8")

    return _run


def test_report_header_names_the_session_it_describes(batch_report):
    """报告头必须点名**这是哪一场** —— 否则读者无从判断自己在看谁的观测。

    红法:把 `generate_report` 不把 `loader.selected_sessions` 传给渲染层(只在单元测试里
    直接调 `sources_disclosure` 时绿、走完整路径时什么都不显示)。
    """
    html = batch_report(voice_rows=3)

    assert _SID in html, "报告没说自己描述的是哪一场"
    assert "本场会话" in html, "缺少「本场是哪一场」的显式标题"
    for label in ("面部", "手势", "语音（面试）"):
        assert label in html, f"披露里少了 {label} 这个模态"


def test_unreadable_modality_is_named_instead_of_vanishing(batch_report):
    """★ I4 的缺口(spec §3.3):语音文件在但读不出时,**必须点名**,不许静默消失。

    真实故障里报告写着「以上模态来自同一场会话」而语音整个没了 —— 读者以为一切正常。
    红法:退回 I4 的"只列真正读出来的模态" → 披露里没有语音那一项 → 红。
    """
    html = batch_report(voice_rows=0)

    assert "未读到数据" in html, (
        "语音读不出来,报告里却没有任何「未读到数据」的痕迹 —— 这正是 I4 的缺口")
    assert "语音（面试）" in html, "连模态名都没有,读者不知道少了什么"
