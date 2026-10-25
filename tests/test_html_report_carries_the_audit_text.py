"""HTML 报告必须带上**载荷里那些"这份报告能说什么"的前提**。

## 为什么

`/api/report/structured` 的 `l1` 段里有四组文字,前端 `L1IndicatorTable` 全部渲染了,
而 HTML 报告(`report_generator`)一个都没印:

| 载荷字段 | 它回答的问题 | 不印的后果 |
|---|---|---|
| `l1.gaps` | 本场**素材层面**的限制(9 段拼接、没有提问台账、没有 ASR) | 读者把"只覆盖最后 7 分钟"的数当成整场 |
| `l1.comparability_notes` | 哪些槽名与旧报告相同但**统计量换了** | 读者把两份报告的数相减,得到一个没有含义的差 |
| `l1.low_confidence_notes` | 置信度为什么只有"低" | 读者无从判断这个数能不能用 |
| `l1.windows` | 整场一个窗口,还是逐回答一个窗口 | 两个不同口径的 F/G 值被当成同一件事 |

前端对这批字段的判断是「放在表格上方而不是折叠里:这些是『这份报告能说什么』的前提」。
交付物是同一份报告的另一个出口,不该比界面少说这四句。

## 这个文件守什么

1. 四组文字都在 HTML 里,且**在正文**(不是注释、不是折叠)。
2. 它们来自**载荷**,不是渲染层另编的话 —— 改载荷,报告跟着变。
3. 空载荷(没限制可报)时不出现空标题。
"""
from __future__ import annotations

import re

import pytest

import report_frontend.report_generator as report_generator
from report_frontend.data_loader import LogDataLoader
from report_frontend.l1_report import build_l1_report
from report_frontend.report_generator import ReportGenerator

_SID = "20260101_000000_0000"


def _text_of(html: str) -> str:
    body = re.sub(r"(?s)<(script|style).*?</\1>", "", html)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body))


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path / "no_rec"))
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path / "no_series"))
    return tmp_path


def _render(tmp_path, monkeypatch, payload=None):
    log_dir = tmp_path / "logs"
    log_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(report_generator, "LogDataLoader",
                        lambda *a, **k: LogDataLoader(str(log_dir)))
    monkeypatch.setattr(report_generator.webbrowser, "open", lambda *a, **k: None)
    p = ReportGenerator(output_dir=str(tmp_path / "out")).generate_report(_SID)
    assert p, "报告没生成 —— 本测试的前提不成立"
    return open(p, encoding="utf-8").read()


@pytest.fixture
def html(isolated, monkeypatch):
    return _render(isolated, monkeypatch)


@pytest.fixture
def payload(isolated):
    return build_l1_report(_SID, logs={})


def test_session_limits_from_the_payload_reach_the_report(html, payload):
    """★ 红法:删掉 `_render_l1_audit_text` 的调用 ⟹ 这里红。

    地板:空数据下 `l1.gaps` 至少有 3 条(没有台账 / 没有 ASR / 没有 voice 日志)。
    没有这句,载荷结构一变下面的循环一次都不进,本测试静默全绿。
    """
    gaps = payload["l1"]["gaps"]
    assert len(gaps) >= 3, f"载荷里的本场限制条数变了:{gaps}"
    body = _text_of(html)
    for g in gaps:
        # 载荷原句里可能带 markdown 记号,报告层会剥掉 —— 按前 12 字比对。
        head = re.sub(r"[*`]", "", g)[:12]
        assert head in body, f"本场限制没进报告:「{g}」"


def test_the_comparability_warning_reaches_the_report(html, payload):
    """★ 红法:同上。这条不印,读者会把两份报告的数相减。"""
    notes = payload["l1"]["comparability_notes"]
    assert notes, "载荷里没有比较陷阱说明 —— 本测试退化为空断言"
    body = _text_of(html)
    assert "不可相减" in body or "不可直接比" in body, (
        "报告里没有那句「与旧引擎报告的数不可相减」的提醒")


def test_the_window_granularity_reaches_the_report(html, payload):
    """★ 红法:同上。整场口径与逐题口径的 F/G 值不能直接比。"""
    w = payload["l1"]["windows"]
    assert w.get("granularity"), "载荷里没有窗口口径 —— 本测试退化为空断言"
    body = _text_of(html)
    assert ("整场一个窗口" in body) or ("逐回答窗口" in body), (
        f"报告里没说这一场是哪种统计口径:{w}")


def test_the_two_window_granularities_print_different_text():
    """★ 红法:把 `granularity == "answer"` 写成 `False`(或反过来)⟹ 这里红。

    ⚠️ 上一条测试**杀不掉这个突变**:空数据载荷的 `granularity` 恰好是 `session`,
    写死成"整场"与真实值一致,断言照样过(2026-10-25 实测,M4 存活)。
    真正要守的是**两种口径印出两句不同的话** —— 同一句印两遍等于没说。
    """
    def _render(granularity, **extra):
        return ReportGenerator(output_dir="/tmp")._render_l1_audit_text(
            {"l1": {"gaps": [], "comparability_notes": [],
                    "low_confidence_notes": [],
                    "windows": {"granularity": granularity, **extra}}})

    session_text = _text_of(_render("session"))
    answer_text = _text_of(_render("answer", n_windows=7))

    assert "整场一个窗口" in session_text
    assert "逐回答窗口" not in session_text

    assert "逐回答窗口" in answer_text and "7" in answer_text
    assert "整场一个窗口" not in answer_text


def test_frames_outside_a_window_are_disclosed_when_there_are_any():
    """★ 红法:把 `frames_outside` 那段删掉 ⟹ 这里红。

    有提问台账时,首个 `ask_end` 之前的帧不属于任何窗口 —— 这是读者唯一能看出
    "分母被排除了多少"的数。零帧的不印(印"0 帧"是噪音)。
    """
    out = _text_of(ReportGenerator(output_dir="/tmp")._render_l1_audit_text(
        {"l1": {"windows": {"granularity": "answer", "n_windows": 3,
                            "frames_outside": {"face": 412, "gesture": 0}}}}))
    assert "412" in out and "face" in out
    assert "gesture" not in out, "零帧的模态不该印出来"


def test_the_audit_text_comes_from_the_payload_not_a_second_copy(
        isolated, monkeypatch):
    """★ 红法:在渲染层另写一份固定的限制清单 ⟹ 这里红。

    判据:把载荷里的 `gaps` 换成可识别标记,报告必须跟着变。
    """
    marker = "只可能来自载荷的l1gaps"
    real = report_generator.build_l1_report

    def fake(session_id, logs=None):
        p = real(session_id, logs=logs)
        p["l1"] = {**p["l1"], "gaps": [marker]}
        return p

    monkeypatch.setattr(report_generator, "build_l1_report", fake)
    html = _render(isolated, monkeypatch)
    assert marker in _text_of(html), (
        "报告里的本场限制不是从载荷读的 —— 渲染层自己编了一份")


def test_no_empty_headings_when_there_is_nothing_to_report():
    """★ 红法:把条件渲染改成无条件 ⟹ 这里红。

    载荷干净(没有限制 / 没有比较陷阱 / 没有低置信理由)时,不该印三个空标题 ——
    空标题会被读成"这里本来该有内容但丢了"。
    """
    out = ReportGenerator(output_dir="/tmp")._render_l1_audit_text(
        {"l1": {"gaps": [], "comparability_notes": [],
                "low_confidence_notes": [], "windows": {}}})
    assert out.strip() == "", f"空载荷却渲染出了内容:{out!r}"
