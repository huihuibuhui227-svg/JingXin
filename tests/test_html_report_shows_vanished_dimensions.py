"""整维消失必须在 **HTML 报告** 里也说出来 —— 前端说了,交付物也得说。

## 为什么

L1 的构念对账(2026-10-18)判定 `行为表现活跃度` 的三个槽
(手掌张开度 / 肩部放松度 / 眼神接触)**没有同构念的 L1 指标** ——
不是"这一场没测到",是那个维度量的东西当前管线都不测。它因此整个不进
`dimensions`,而进 `vanished_dimensions` 并带理由。

前端 `ReportViewer` 为此专门加了一块「整维未出(当前管线没有对应的可测量)」,
注释原话是「不渲染的话报告会**悄悄少一块**,而少的那块恰好是『测不到的东西』
—— 那是最该说出来的部分」。

但 `report_generator._generate_deep_text_analysis` 只遍历 `result["dimensions"]`
⟹ **HTML 报告里那一维凭空消失**,一个字都不提。同一场会话,界面上(前端报告页)
说"这一维整维未出",下载下来的 HTML 说"只有四个维度"。读者只会以为
HTML 更权威、或者以为那一维本来就只算四个。

⟹ 同一个载荷,两个出口,一个说、一个不说。这不是"少了个锦上添花的段落",
是**交付物比界面少了一句真话**。

## 连带的一处

雷达图 `_build_radar_figure` 对缺失的维度补轴名 `"未知"`、值 0。
`未知` 读起来是"这一场没测到",而真相是"这一维量的东西当前管线都不测" ——
与上面那条**恰好相反**的含义(前者让人等下一场,后者让人别再等)。
轴名该用它自己的名字。

## 这个文件守什么

1. HTML 报告里出现消失维度的**名字**,并说明是"本层不测"而不是"本场没测到"。
2. 那一块里不出现内部槽键(读者不认识 `left_hand_spread`)。
3. 雷达图上那一维的轴名不是 `未知`。
"""
from __future__ import annotations

import copy
import re

import pytest

import report_frontend.l1_mapper as l1_mapper
import report_frontend.report_generator as report_generator
from report_frontend.data_loader import LogDataLoader
from report_frontend.l1_mapper import SLOT_MAP
from report_frontend.l1_report import build_l1_report
from report_frontend.report_generator import ReportGenerator
from report_frontend.visualizer import ReportVisualizer

_SID = "20260101_000000_0000"


def _text_of(html: str) -> str:
    body = re.sub(r"(?s)<(script|style).*?</\1>", "", html)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body))


#: 构造用的"整维没有对应指标"维度键(用完即撤)。
_SYNTHETIC_KEY = "synthetic_dead_dimension"


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """三条旁路都指到空目录 —— "空数据"这件事本身得先成立。"""
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path / "no_rec"))
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path / "no_series"))
    return tmp_path


@pytest.fixture
def payload(isolated, monkeypatch):
    """空数据载荷 + **一个构造出来的**整维消失维度。

    ## ⚠️ 2026-11-22:为什么改成构造

    补进三条新指标之后,生产数据里**再没有任何维度整体消失**
    (`行为表现活跃度` 从"3 槽全删"变成"2 槽有指标")。而"整维消失要在 HTML 里
    说出来"这条判据仍然必须有人守。构造借的三个槽(`fluency_score` /
    `eye_contact` / `n_chars`)是今天真的没有对应指标的那三个 ——
    所以验的仍是真表(`SLOT_LABEL` / `DROP_REASON`),造出来的只是
    "它们同属一个维度"这一件事。生产侧"一个都不消失"由
    `tests/test_vanished_dimensions_are_printable.py::test_the_real_payload_no_longer_vanishes_any_dimension` 钉着。

    ⚠️ 三条旁路都得指到空目录,否则本机盘上的真实日志会让维度活过来,
    本测试的前提就不成立了(见 `tests/test_report_gaps_are_readable.py` 的同类说明)。
    """
    # `map_l1_to_dimensions` 在**调用时**读模块级 `DIMENSIONS`
    # ⟹ 这里改的那张表就是装配时用到的同一张(HTML 那条路也一样,它内部也调
    # `build_l1_report`)。用完由 monkeypatch 自动撤回。
    monkeypatch.setitem(l1_mapper.DIMENSIONS, _SYNTHETIC_KEY, {
        "name": "构造用整维",   # ≤6 字:雷达图轴名会截断(见 report_generator)
        "slots": {"fluency_score": 0.4, "eye_contact": 0.35, "n_chars": 0.25},
    })
    got = build_l1_report(_SID, logs={})
    assert [v["key"] for v in got["vanished_dimensions"]] == [_SYNTHETIC_KEY], (
        f"消失的维度不是构造进来的那一个:{[v['key'] for v in got['vanished_dimensions']]}")
    return got


@pytest.fixture
def html(tmp_path, payload, monkeypatch):
    """真渲染,只换日志目录与"打开浏览器"。"""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    monkeypatch.setattr(report_generator, "LogDataLoader",
                        lambda *a, **k: LogDataLoader(str(log_dir)))
    monkeypatch.setattr(report_generator.webbrowser, "open", lambda *a, **k: None)
    p = ReportGenerator(output_dir=str(tmp_path / "out")).generate_report(_SID)
    assert p, "报告没生成 —— 本测试的前提不成立"
    return open(p, encoding="utf-8").read()


def _vanished_block(html: str) -> str:
    """只取「整维未出」那一块。

    ⚠️ 必须**只在这一块里**断言。第一版断言的是整份报告的正文:
    `test_the_vanished_block_says_the_layer_does_not_measure_it` 当时**一写就绿**
    —— 因为别的维度的缺口行里本来就有「本层不测这一条」那句话。它声称守的
    「消失维度要说明原因」一次都没生效过,典型的空断言(本仓记过多次)。

    ⚠️ 2026-10-26 换过提取方式:原来是 `</div>\\s*</div>`(靠"块后面紧跟外层卡片
    的收口"),而报告改成无卡片布局之后那个模式**匹配不到任何东西** ——
    测试报"没有这一块",而块其实好好地在正文里。判据不该依赖外层布局。
    这一块内部没有任何嵌套 `<div>`,所以收到**自己的**第一个 `</div>` 就正好。
    """
    m = re.search(r'(?s)<div id="vanished-dimensions".*?</div>', html)
    assert m, "HTML 报告里没有「整维未出」那一块 —— 消失的维度仍然只字未提"
    return m.group(0)


def test_html_report_names_the_dimension_that_vanished(html, payload):
    """★ 红法:删掉 `_render_vanished_dimensions` 的调用 ⟹ 这里红。

    前端那块卡片的标题是「整维未出(当前管线没有对应的可测量)」;
    HTML 至少要同样说得出**是哪一个维度**、以及**为什么**。
    """
    block = _text_of(_vanished_block(html))
    for v in payload["vanished_dimensions"]:
        assert v["display_name"] in block, (
            f"「整维未出」那一块里没有「{v['display_name']}」—— "
            "这一维整维消失了,交付物却一个字都不提,读者只会以为本来就只有四个维度")


def test_the_vanished_block_says_the_layer_does_not_measure_it(html):
    """★ 红法:把那句改成「本场未测到」⟹ 这里红。

    「本层不测」与「本场没测到」是两件事:后者补采/补缓存就有,前者再采
    一百次也不会有。混成一句,读者会去追一个不存在的数据。

    ⚠️ 只看**块首那句说明**(`<p>`),不看逐槽的 `<li>`。第一版断言的是整块:
    逐槽行本来就有「本层不测这一条」,于是把块首那句改成「本场未测到」
    **照样绿** —— 变异实测确认过,那条断言守的不是它声称的东西。
    """
    block = _vanished_block(html)
    m = re.search(r"(?s)<p>(.*?)</p>", block)
    assert m, "「整维未出」那一块没有说明句"
    prose = _text_of(m.group(1))
    # ⚠️ 2026-10-26 换措辞:原来守的是「本层不测」/「当前管线都不测」——
    #    「本层」「管线」都是内部叫法。意思没变:这一维要量的东西
    #    **当前这套采集根本没有**,不是这一场没测到。
    assert ("这套采集" in prose and "根本没有" in prose) or ("当前管线都不测" in prose), (
        f"块首那句没说清是「整套采集都不测」还是「本场没测到」:{prose}")
    assert "本场没测到" not in prose.replace("不是本场没测到", ""), (
        f"块首那句把它说成了「本场没测到」—— 读者会去等下一场:{prose}")


def test_the_vanished_block_names_no_internal_slot_key(html):
    """★ 红法:直接把 `d["slot"]` 拼进去 ⟹ 这里红。

    与 `test_report_gaps_are_readable.py` 同一条规矩:交付物印中文名,
    内部键留在载荷里。
    """
    block = _text_of(_vanished_block(html))
    leaked = sorted(s for s in SLOT_MAP if s in block)
    assert not leaked, f"「整维未出」那一块里露出了内部槽键:{leaked}"


def test_radar_axis_uses_the_dimension_name_not_unknown(isolated, monkeypatch):
    """★ 红法:把 `_build_radar_figure` 里的 `"未知"` 改回来 ⟹ 这里红。

    `未知` 读起来是"这一场没测到";真相是"这一维量的东西当前管线都不测"。
    两种话在图上长得一样,含义相反 —— 轴名该用它自己的名字。

    ## ⚠️ 2026-11-22:为什么用"把真维度挪进 vanished"而不是那个构造维度

    `_build_radar_figure` 只认 `DIMENSIONS` 里那**五个固定键**(它按
    `dim_order = ['logical_thinking', 'stress_resilience', …]` 逐根画轴),
    所以"构造一个第六维"在雷达图上**根本不会出现轴** —— 用它测这条会变成
    空断言(实测:轴里确实没有构造那根,而断言失败的原因与"未知"无关)。
    ⟹ 这里把**真载荷**里的 `confidence_level` 整块挪进 `vanished_dimensions`:
    形状与"当年这一维真的整维消失"逐键相同,而且落在 `dim_order` 认的键上。
    """
    real = build_l1_report(_SID, logs={})
    assert "confidence_level" in real["dimensions"], "构造前提:这一维现在是活的"
    fake = copy.deepcopy(real)
    dim = fake["dimensions"].pop("confidence_level")
    fake["vanished_dimensions"] = [{
        "key": "confidence_level", "display_name": dim["display_name"],
        "n_slots": 3, "dropped": [], "reason": "构造:整维没有对应指标",
    }]

    fig = ReportVisualizer(output_dir="/tmp")._build_radar_figure(fake)
    theta = list(fig.data[0].theta)
    assert "未知" not in theta, f"雷达图上出现了「未知」轴:{theta}"
    assert dim["display_name"] in theta, (
        f"雷达图上没有「{dim['display_name']}」这根轴:{theta}")
