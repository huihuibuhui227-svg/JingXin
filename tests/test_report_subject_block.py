"""报告头的「这一场是谁的」—— 姓名 / 学号 / 院系 / 场次序号。

为什么值得一条钉子(2026-10-26,使用者原话):
「要求显示出被测人的个人信息,不然看报告分辨不出」。报告此前只印会话号,
同一批十几个人录下来,十几份 HTML 在盘上长得一模一样。

三种状态都必须有话说,而且**不许混**:
* 有标注 → 逐字段印
* 没标注 → 明说「未标注」(不是空白 —— 空白读起来像读取失败)
* 读坏了 → 明说「读取失败」并指到 `label.json`
"""
import json
from pathlib import Path

import pytest

import report_frontend.report_generator as report_generator
from report_frontend.data_loader import LogDataLoader
from report_frontend.report_generator import (
    ReportGenerator,
    _meta_line,
    _subject_block,
    _subject_slug,
    _unique_report_path,
    session_started_at,
)

SID = "20261026_002930_cb79"


def _write_label(root, **fields) -> None:
    """造一场已标注的会话目录(`<标签>__<sid>`/label.json)。"""
    d = root / f"1026-张三__{SID}"
    d.mkdir()
    (d / "label.json").write_text(
        json.dumps({"session_id": SID, **fields}, ensure_ascii=False),
        encoding="utf-8")


def test_prints_all_four_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    _write_label(tmp_path, serial="1026", name="张三", student_id="241650137",
                 department="计算机与人工智能学院")
    html = _subject_block(SID)
    for want in ("张三", "241650137", "计算机与人工智能学院", "1026"):
        assert want in html, f"报告头少了 {want}:{html}"


def test_missing_label_says_untagged_not_blank(tmp_path, monkeypatch):
    """没标注 ≠ 读取失败:必须字面写出来,不能留白。"""
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    html = _subject_block(SID)
    assert "未标注" in html
    assert "读取失败" not in html


def test_corrupt_label_says_failed(tmp_path, monkeypatch):
    """读坏了要说「读取失败」—— 猜成「未标注」会让人以为没人填过。"""
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    d = tmp_path / f"1026-张三__{SID}"
    d.mkdir()
    (d / "label.json").write_text("{ 这不是 JSON", encoding="utf-8")
    html = _subject_block(SID)
    assert "读取失败" in html
    assert "label.json" in html


def test_label_text_is_escaped(tmp_path, monkeypatch):
    """标注是客户端可控文本,拼进 HTML 前必须转义。"""
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    _write_label(tmp_path, name="<script>alert(1)</script>", department="A&B")
    html = _subject_block(SID)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "A&amp;B" in html


def test_no_session_id_prints_nothing():
    """报告没指定会话时不硬编一个空块。"""
    assert _subject_block(None) == ""


# ── 文件名 / 两个时间 ────────────────────────────────────────────────────────

def test_subject_slug_puts_the_name_in_the_filename(tmp_path, monkeypatch):
    """文件名要带姓名 —— 报告列表里 29 份 `_<时刻>.html` 分不出谁是谁(2026-10-26)。"""
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    _write_label(tmp_path, name="张三")
    assert _subject_slug(SID) == "-张三"


def test_subject_slug_strips_path_characters(tmp_path, monkeypatch):
    """姓名会被拼进**文件名**,非法字符必须去掉(否则建文件直接抛)。"""
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    _write_label(tmp_path, name='a/b\\c:d*e?f"g<h>i|j')
    assert _subject_slug(SID) == "-abcdefghij"


def test_subject_slug_empty_without_label(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    assert _subject_slug(SID) == ""


def test_same_second_does_not_overwrite_another_sessions_report(tmp_path):
    """同一秒里连跑两场:第二个名字必须换,不许把第一场的报告盖掉。

    2026-10-26「全部重跑」实测:三场在一秒内跑完,文件名相同 ⟹ 后一场静默覆盖前一场,
    最后只剩一份。文件名的时间戳只到秒,这个碰撞是必然而不是偶然。
    """
    (tmp_path / "R_20261026_120000.html").write_text("first", encoding="utf-8")
    p = _unique_report_path(str(tmp_path), "R_20261026_120000", SID)
    assert Path(p).name == "R_20261026_120000-cb79.html"
    assert (tmp_path / "R_20261026_120000.html").read_text(encoding="utf-8") == "first"


def test_meta_line_prints_test_time_and_report_time():
    """两个时间都要印:测试时间(铸号时刻)不随重跑变,报告生成时间会变。"""
    line = _meta_line(SID, "l1-1")
    assert "测试时间 2026-10-26 00:29:30" in line
    assert "报告生成时间" in line
    assert "报告口径版本 l1-1" in line
    # 测试时间不是"今天"—— 它是会话 id 自带的那个时刻。
    assert session_started_at(SID).strftime("%Y-%m-%d %H:%M:%S") == "2026-10-26 00:29:30"


def test_unparseable_session_id_has_no_fake_time():
    assert session_started_at("not-a-session") is None


def test_generated_report_filename_carries_name_and_times(tmp_path, monkeypatch):
    """整条链:有标注的实验场 → 文件名带姓名,报告头印两个时间。"""
    rec = tmp_path / "rec"
    rec.mkdir()
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(rec))
    _write_label(rec, name="张三", student_id="241650137",
                 department="计算机与人工智能学院", serial="1026")

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    monkeypatch.setattr(report_generator, "LogDataLoader",
                        lambda *a, **k: LogDataLoader(str(log_dir)))
    monkeypatch.setattr(report_generator.webbrowser, "open", lambda *a, **k: None)

    p = ReportGenerator(output_dir=str(tmp_path / "out")).generate_report(SID)
    assert p, "空日志也该照样出报告(既有契约)"
    assert Path(p).name.endswith("-张三.html"), Path(p).name
    html = Path(p).read_text(encoding="utf-8")
    assert "测试时间 2026-10-26 00:29:30" in html
    assert "报告生成时间" in html
    assert "张三" in html and "计算机与人工智能学院" in html
