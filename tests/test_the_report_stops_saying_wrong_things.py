"""交付物里剩下的三处**说错话 / 说空话**。

这三条都是 2026-10-25 深夜端到端看真实报告时肉眼发现的,
前一轮的「内部记号」测试没覆盖到,因为形态各不相同:

## ① 缺口码 `not_stored` 直接印进了报告

    「语音日志在,但没有帧级序列缓存 —— V 族 8 条序列指标报 not_stored(可跑 …)」

前半句是人话,后半句漏了一个**内部缺口码**。同类码 `time_unusable` /
`window_none` 在别处都走了 `l1_quality.GAP_EXPLANATION` 那张表,这一条是手写的
字符串,所以逃过了那张表。**这条测试守的是"新写的 gap 句也必须走那张表"** ——
不是"这一条改掉就好":手写字符串会不断新增,得让判据扫全部 gap。

## ② 标题写死「五维」,而 L1 只有四维

`📊 五维证据覆盖` 是旧引擎时代的字面量。L1 引擎下 `行为表现活跃度` 整维消失,
报告正文里只有四个维度 ⟹ 标题与内容**自相矛盾**,而读者会以为少了一维。
维度数是**从载荷算的**,不该是字面量。

## ③ 空的「🔍 分维度证据链」标题

逐维证据链图 2026-10-04 随评分层摘除后,那行 `<h2>` 还留着 ——
报告末尾因此挂着一个**下面什么都没有**的二级标题。读者会以为内容丢了。
"""
from __future__ import annotations

import re

import pandas as pd
import pytest

import report_frontend.report_generator as report_generator
from report_frontend.data_loader import LogDataLoader
from report_frontend.l1_quality import GAP_EXPLANATION
from report_frontend.l1_report import build_l1_report
from report_frontend.report_generator import ReportGenerator

_SID = "20260101_000000_0000"


def _text_of(html: str) -> str:
    """报告正文 —— **先剥 HTML 注释**,再剥标签。

    ⚠️ 只剥标签会漏:`<!-- 这里原来有 <h2>🔍 分维度证据链</h2> -->` 里第一个
    `<[^>]+>` 只吃掉注释前半截,标题字面量留在正文里 —— 于是断言「正文里没有
    这个标题」会红,而报告其实是对的。注释不渲染,不算正文。
    """
    body = re.sub(r"(?s)<!--.*?-->", "", html)
    body = re.sub(r"(?s)<(script|style).*?</\1>", "", body)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body))


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path / "no_rec"))
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path / "no_series"))
    return tmp_path


def _render(tmp_path, monkeypatch):
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


# ---------------------------------------------------------------------------
# ① 缺口码不许进交付物
# ---------------------------------------------------------------------------

def test_no_gap_code_reaches_the_report(html):
    """★ 红法:把某条 gap 句里的解释换回缺口码常量 ⟹ 这里红。

    判据是**扫全部缺口码**,不是只查 `not_stored` —— 手写 gap 字符串会不断新增,
    逐个点名守不住下一条。
    """
    body = _text_of(html)
    leaked = sorted(code for code in GAP_EXPLANATION if code in body)
    assert not leaked, (
        "报告正文里露出了内部缺口码(读者不认识它们,该走 GAP_EXPLANATION):\n  "
        + "\n  ".join(leaked))


def test_the_series_gap_still_says_what_is_missing_and_how_to_get_it(isolated):
    """★ 红法:把那条 gap 整句删掉 ⟹ 这里红。

    改成人话 ≠ 少说。读者仍要知道:V 族 8 条序列指标没有值、为什么、怎么补。

    ⚠️ 必须喂一条**非空 voice 日志**才走得到这条分支 —— 空数据下报的是
    「本场没有 voice 日志」,那是另一条缺口。第一次写这条时用空 logs,
    断言直接扑空(载荷里根本没有序列那条),是 `assert line` 那句地板抓住的。
    """
    df = pd.DataFrame({"timestamp": [1.0, 2.0], "energy": [0.1, 0.2]})
    rec = build_l1_report(_SID, logs={"voice_interview": df})
    line = next((g for g in rec["l1"]["gaps"] if "序列" in g), None)
    assert line, f"载荷里没有序列缓存那条限制:{rec['l1']['gaps']}"
    assert "8" in line, f"没说是 8 条:{line}"
    assert "extract_session" in line, f"没说怎么补:{line}"
    assert "not_stored" not in line, f"还带着缺口码:{line}"


def test_the_series_gap_appears_on_both_code_paths(isolated):
    """★ 红法:把早退分支里那次 `series_gap_note` 调用删掉 ⟹ 这里红。

    `compute_session_l1` 有两条路:有会话原点走主路,没有原点(正式场的常态)
    走早退分支。这条限制原先**只写在主路上** —— 同一场会话,有没有留存帧
    会决定这句提示出不出现,而缺的东西是同一个(序列缓存)。

    本测试走**早退分支**(没有 questions.jsonl ⟹ 没有原点)。
    """
    df = pd.DataFrame({"timestamp": [1.0, 2.0], "energy": [0.1, 0.2]})
    rec = build_l1_report(_SID, logs={"voice_interview": df})
    assert any("提问台账" in g for g in rec["l1"]["gaps"]), (
        "本测试的前提不成立:没有走早退分支")
    assert any("序列" in g for g in rec["l1"]["gaps"]), (
        "早退分支漏了序列缓存那条限制 —— 与主路不一致")


# ---------------------------------------------------------------------------
# ② 维度数从载荷算,不是字面量
# ---------------------------------------------------------------------------

def test_the_coverage_heading_counts_the_dimensions_in_the_payload(html, isolated):
    """★ 红法:把标题改回字面量「五维」⟹ 这里红。

    地板:空数据下 L1 只出四个维度(第五维整维消失)。没有这句,
    维度数变化时下面的比对会拿一个凑巧相等的数糊过去。
    """
    rec = build_l1_report(_SID, logs={})
    n = len(rec["dimensions"])
    assert n == 4, f"空数据下的维度数变了:{sorted(rec['dimensions'])}"
    body = _text_of(html)
    assert "四维" in body or "4 维" in body, (
        f"报告里没有「{n} 个维度」的说法 —— 标题很可能还是写死的「五维」")
    assert "五维" not in body, "报告里还写着「五维」,而这一场只有四个维度"


def test_the_vanished_dimension_is_not_counted_twice():
    """★ 红法:让标题数 `dimensions + vanished_dimensions` ⟹ 这里红。

    消失的那一维**声明了槽但没进 `dimensions`** —— 它在正文里另有一块说明,
    不该再被算进「本次观测的维度」里,否则读者会以为它出了值。
    """
    rec = build_l1_report(_SID, logs={})
    assert rec.get("vanished_dimensions"), "本测试的前提不成立:没有消失的维度"
    assert len(rec["dimensions"]) + len(rec["vanished_dimensions"]) == 5
    out = ReportGenerator(output_dir="/tmp")._coverage_heading(rec)
    assert "四维" in out or "4 维" in out, out


# ---------------------------------------------------------------------------
# ③ 空的标题不许留着
# ---------------------------------------------------------------------------

def test_no_empty_evidence_chain_heading(html):
    """★ 红法:把那个 `<h2>` 加回去 ⟹ 这里红。

    逐维证据链图摘除后,标题还挂着,下面什么都没有。
    """
    body = _text_of(html)
    assert "分维度证据链" not in body, (
        "报告末尾挂着一个空的「分维度证据链」标题 —— 读者会以为内容丢了")
