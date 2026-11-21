# tests/test_report_instrument_rendering.py
"""**A5 仪器来源披露真的接进了报告层** —— 不是只躺在载荷里。

## 为什么这个文件必须存在

`instrument_provenance.py` 自己的 35 条测试全绿,`blendshape_second_instrument.py`
42 条全绿 —— 但它们证明的只是"这个模块算得对"。接入之前,下面这条命令在本仓是
**零命中**:

    grep -niE "instrument|仪器" report_frontend/l1_report.py \\
         report_frontend/l1_quality.py report_frontend/report_generator.py

⟹ 字段在载荷里,而逐条指标的表格里没有出处,页脚没有汇总,前端不渲染。
本仓记过多次的形态:**"知道了却没说"与"不知道却说知道"一样是错的**;
只测模块、不测出口,等于没接。

## 这个文件守什么(六条,每条都对应一次真实的失败形态)

1. **每条指标都带可读的仪器** —— 少一条就是披露漏了一条,而且不会有任何报错;
2. **既有键值零变化** —— 关掉标注再装配一次,除新键外两份载荷**逐键相等**。
   这是"旧消费方行为不变"的**证据**,不是声明;
3. **多仪器指标要渲染成两路** —— `speech_rate = 转写字数 ÷ 有声时长`,一头文本
   一头音频。挑一个写上去(实测最容易发生的错)这里当场红;
4. **真实生成路径的 HTML 里有仪器字样** —— 判据落在 `generate_report()` 落盘的那份
   文件上,而不是只落在 `_build_html_report()` 的返回值上;
5. **「未登记」是披露缺口,不是一种仪器** —— 它必须有自己的样式(`prov-unknown`),
   并在汇总里点出**缺哪几列**(只说"8 条未登记"没人能动手);
6. **页脚那句话与逐条标签同一套措辞** —— 两处各拼一份,读者会以为是两样东西。

## 为什么用空日志(`logs={}`)装配

"这条指标算不出来"与"这条指标出自哪套仪器"是两件事:依赖列是**声明过**的,
值可以是 None。用空日志能把披露从"有没有数据"里剥出来 —— 一场什么都没采到的
会话,披露也必须完整(那是它的边界条件)。
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

import pandas as pd
import pytest

import report_frontend.instrument_provenance as instrument_provenance
import report_frontend.report_generator as report_generator
from report_frontend import l1_report
from report_frontend.data_loader import LogDataLoader
from report_frontend.l1_report import build_l1_report
from report_frontend.report_generator import ReportGenerator, _instrument_cell

#: 一个**不存在的**会话号:装配照样产出 38 条指标(带 gap),但不碰任何真素材。
#: 用真会话号有风险 —— `data/` 里有没有那一场取决于跑测试的机器。
_SID = "20260101_000000_0000"

#: 新键。除它们之外,装配前后两份载荷必须**逐键相等**(见 `test_annotation_changes_nothing_but_its_own_keys`)。
_NEW_KEYS = {"instrument", "instruments"}

#: `l1.instruments` 那一块(汇总)也是新增的:它的键名就叫 `instruments`,
#: 所以 `_strip()` 一并摘掉 —— 它同样是"新增,不是改动"。


@pytest.fixture(scope="module")
def payload() -> dict:
    """真的走一遍装配(`build_l1_report`),不是手工拼的假载荷。"""
    return build_l1_report(_SID, logs={})


def _strip(obj):
    """递归摘掉新键 —— 用来比较"装配前后除新键外的部分"。"""
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in _NEW_KEYS}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def _normalized(p: dict) -> dict:
    """去掉每次都变的 `model_metadata`(时间戳),其余原样。"""
    out = _strip(p)
    out.pop("model_metadata", None)
    return out


# ── 1. 每条指标都有可读仪器 ────────────────────────────────────────────────

def test_every_indicator_carries_a_readable_instrument(payload):
    """38 条指标,一条都不许漏;而且每条都要能印成人话。

    判据两层:`instruments_of()` 能读出非空列表(**结构上没漏**),且每个仪器名
    在 `l1_report.INSTRUMENT_LABEL` 里有中文标签(**印得出来**)。
    只判第一层不够 —— 名字在没有标签的表里,报告上就是一串英文键。
    """
    rows = payload["l1"]["indicators"]
    assert len(rows) == 38, f"指标条数变了(是 38 吗?):{len(rows)}"

    for row in rows:
        got = instrument_provenance.instruments_of(row)
        assert got, f"{row['name']} 装配后没有任何仪器来源 —— 披露漏了一条"
        unknown = [n for n in got if n not in instrument_provenance.INSTRUMENTS]
        assert not unknown, f"{row['name']} 报出了没登记过的仪器名:{unknown}"
        missing = [n for n in got if n not in l1_report.INSTRUMENT_LABEL]
        assert not missing, f"{row['name']}:{missing} 没有中文标签 —— 报告里会露英文键"
        # 单值 / 多值二选一(由 `annotate_indicators` 决定),两种都能被 `instruments_of` 读出来
        assert ("instrument" in row) != ("instruments" in row), (
            f"{row['name']} 同时带了 instrument 与 instruments,读法就不止一种了")

    # 汇总块与逐条一致:总数对得上,5 个键一个不缺(含计数为 0 的)
    block = payload["l1"]["instruments"]
    assert block["n_indicators"] == 38
    assert set(block["counts"]) == set(instrument_provenance.INSTRUMENTS)
    assert sum(block["counts"].values()) >= block["n_indicators"], (
        "多仪器指标在每个仪器下各计 1 ⟹ 各项之和只会 ≥ 条数;小于就说明有指标没计数")


def test_evidence_rows_carry_the_same_instrument_as_the_audit_chain():
    """逐维度表里那一行(`evidence_chain`)与审计链是同一条指标、同一个出处。

    两处各标一次就会分叉 —— 而这份报告的两个出口(HTML 表 / 前端表)读的正是这两处。
    判据:同一个 `feature`,两处拿到的仪器列表必须相等。

    ⚠️ **不能用"真素材"来测这条**:`tests/conftest.py` 在导入时就把语音日志目录指到
    临时目录(`_voice_logger_module.LOGS_DIR = tmp`),pytest 里读不到仓库 `data/logs`
    ⟹ `build_l1_report(真会话号)` 的 `evidence_chain` 全是空的,循环体一次都不进。
    这正是原来那版用例的毛病(被 `test_assert_coverage` 抓出来):它看起来在测真数据,
    实际是空转,还顺带让"换条来路披露会变"这句话失去了被验证的机会。
    真素材的证据走 `verify/` 里的 CLI 脚本(demo_run.py + SSR 冒烟),那些在 pytest 之外跑。
    """
    rows = [
        {"name": "speech_rate", "required_l0": ["n_chars", "speech_duration_sec"]},
        {"name": "pitch_variability_per_sec", "required_l0": ["f0", "frame_sec"]},
        {"name": "smile_ratio", "required_l0": ["bs_mouthSmileLeft"]},
        {"name": "skin_conductance_mean", "required_l0": ["skin_conductance"]},
    ]
    instrument_provenance.annotate_indicators(rows)
    slot_of = {"speech_rate": "V1_amount", "pitch_variability_per_sec": "V3_range",
               "smile_ratio": "V4_ratio", "skin_conductance_mean": "V9_gsr"}
    checked = 0
    for row in rows:
        want = instrument_provenance.instruments_of(row)
        item = l1_report._evidence_item(
            {"l1": row["name"], "slot": slot_of[row["name"]],
             "value": 1.0, "n_valid": 12}, want)
        assert item["instruments"] == want, (
            f"{row['name']}:证据行与审计链的仪器来源不一致")
        # 形状契约:永远是 list(单仪器也是单元素),渲染层不该再判断单值/多值
        assert isinstance(item["instruments"], list) and item["instruments"]
        assert "instrument" not in item, (
            "证据行出现了单值键 —— 渲染层就得判断两套读法,那正是这条用例要防的分叉")
        checked += 1
    assert checked == len(rows), "有指标没被核实 —— 用例退化成空转"
    # 同一条指标换条数据来路:列名从 `pitch` 换成 `f0` 之后,仪器必须还是"音频"
    assert instrument_provenance.instruments_of(rows[1]) == ["audio"]


# ── 2. 既有键值零变化 ──────────────────────────────────────────────────────

def test_annotation_changes_nothing_but_its_own_keys(payload, monkeypatch):
    """**把标注关掉再装配一次**,除新键外两份载荷必须逐键相等。

    ## 为什么这条是"证据"而不是"声明"

    "只加键、不改值"很容易在实现里破掉(`annotate_indicators` 一旦改成就地修改,
    或者有人在装配处顺手重排了 rows)。所以这里拿**同一个装配函数**跑两遍:
    一遍正常,一遍把 `annotate_indicators` 换成"只复制、不标注"的替身 ⟹
    两份载荷除了新键必须一模一样。

    顺带钉住另一件事:**新键真的加上了**。只比"相等"是不够的 —— 如果标注整个
    没接上,两份都会不带新键,那条比较照样绿(本仓记过多次的"空断言")。
    """
    with_annotation = payload
    real = instrument_provenance.annotate_indicators

    def _no_annotation(rows):
        # 替身:形状与真的相同(返回副本列表),只是不加仪器键。
        return [dict(r) for r in rows]

    monkeypatch.setattr(instrument_provenance, "annotate_indicators", _no_annotation)
    try:
        without = build_l1_report(_SID, logs={})
    finally:
        monkeypatch.setattr(instrument_provenance, "annotate_indicators", real)

    assert _normalized(without) == _normalized(with_annotation), (
        "除了 instrument / instruments 两个新键,装配结果变了 —— 旧消费方会受影响")

    # 非空断言:标注必须真的接上了
    assert any("instrument" in r or "instruments" in r
               for r in with_annotation["l1"]["indicators"]), "标注没接上"
    assert "instruments" in with_annotation["l1"], "汇总块没进载荷"
    # 替身确实生效了:逐条指标那层没有新键(汇总块是 `describe()` 现算的,与标注无关)
    assert not any("instrument" in r or "instruments" in r
                   for r in without["l1"]["indicators"]), (
        "替身没生效 —— 这条比较测不到东西")


# ── 3. 多仪器指标 ──────────────────────────────────────────────────────────

def test_multi_instrument_indicator_renders_both_paths(payload):
    """`speech_rate = n_chars ÷ speech_duration_sec` ⟹ 文本 + 音频,两路都要印。

    挑一个写上去是这个仓最容易发生的错:两边都"不算错",而报告会说"这条来自音频"
    —— 于是"两套仪器"这件事在报告里被抹平成了"一条数一路来源"。
    """
    row = next(r for r in payload["l1"]["indicators"] if r["name"] == "speech_rate")
    assert instrument_provenance.instruments_of(row) == ["audio", "text"], (
        f"语音字数率被记成了单仪器:{row}")

    labels = payload["l1"]["instruments"]["labels"]
    assert _instrument_cell(row, labels) == "音频、文本"

    # 走渲染层:一条 speech_rate 的表格行
    dim = {
        "display_name": "沟通表达流畅度",
        "description": l1_report.DIM_DESCRIPTION,
        "algorithm": l1_report.DIM_ALGORITHM,
        "matched_indicators": "1/3",
        "confidence": "低",
        "evidence_gaps": [],
        "evidence_gaps_not_measured": [],
        "evidence_chain": [{
            "feature": "speech_rate",
            "human_name": "语音字数率",
            "raw_value": 5.03,
            "n_valid": 214,
            "std": None,
            "observed_interval": [],
            "instruments": instrument_provenance.instruments_of(row),
        }],
    }
    html = ReportGenerator(output_dir="/tmp")._render_dimension_block("x", dim, labels)
    assert "<th>仪器来源</th>" in html, "表格里没有「仪器来源」这一列"
    assert "音频、文本" in html, f"多仪器没印全:{html}"


def test_a_single_instrument_row_is_not_mistaken_for_two(payload):
    """反向判据:单仪器那条不许印成两路 —— 否则"两路"这个说法就不值钱了。"""
    row = next(r for r in payload["l1"]["indicators"] if r["name"] == "lexical_diversity")
    labels = payload["l1"]["instruments"]["labels"]
    assert instrument_provenance.instruments_of(row) == ["text"]
    assert _instrument_cell(row, labels) == "文本"


# ── 4. 真实生成路径的 HTML ────────────────────────────────────────────────

#: 一份最小日志(形状取自真实列名)。与 `test_html_report_uses_the_l1_engine` 同源 ——
#: 本测试关心的是**出口有没有这句话**,不是这一场的数是多少。
_FACE = pd.DataFrame([{
    "session_id": _SID, "timestamp": "2026-01-01T00:00:00",
    "au12_smile": 0.31, "au4_frown": 0.12, "au1_inner_brow": 0.08,
    "gaze_direction_x": 0.02, "gaze_direction_x_std": 0.09,
    "blink_rate_per_min": 17.0, "frame_count": 1,
}])


@pytest.fixture
def generated_html(tmp_path, monkeypatch) -> str:
    """走**生产路径**:`generate_report()` → 落盘的 HTML 文件 → 读回来。"""
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    _FACE.to_csv(log_dir / f"face_au_log_{_SID}.csv", index=False)
    monkeypatch.setattr(report_generator, "LogDataLoader",
                        lambda *a, **k: LogDataLoader(str(log_dir)))
    monkeypatch.setattr(report_generator.webbrowser, "open", lambda *a, **k: None)
    path = ReportGenerator(output_dir=str(tmp_path / "out")).generate_report(_SID)
    assert path, "报告没生成 —— 本测试的前提不成立"
    return Path(path).read_text(encoding="utf-8")


def test_the_generated_html_discloses_instrument_source(generated_html):
    """判据落在**落盘的那份文件**上:里面有仪器来源,而且有汇总那一句。"""
    html = generated_html
    assert "仪器来源" in html, "交付物里没有「仪器来源」四个字"
    assert 'id="instrument-disclosure"' in html, "披露那一段整块不见了"
    for word in ("几何代理", "模型系数", "未登记"):
        assert word in html, f"汇总标签里少了「{word}」"

    # 页脚那句(由 `instrument_provenance.format_footer()` 出):
    # 结构 + 五个标签都要在,数字不限 —— 这一场的数与 fixture 无关。
    m = re.search(r'class="prov-footer">([^<]+)<', html)
    assert m, "页脚没有那句汇总"
    line = m.group(1)
    for label in l1_report.INSTRUMENT_LABEL.values():
        assert f"{label} " in line, f"页脚那句里没有「{label}」:{line}"
    assert line.startswith("共 "), f"页脚那句话不以「共 N 条指标」开头:{line}"


def test_the_disclosure_says_the_two_facial_instruments_are_not_interchangeable(
        generated_html):
    """"两套仪器不可混用"必须出现在交付物里 —— 否则读者会把两套数当两次测量。

    出处:`instrument_provenance` 模块 docstring(离线实测 32 场 / 389 条,
    12 个构念里只有 1 个达 ρ ≥ 0.7)。
    """
    assert "blendshape" in generated_html
    assert "geometric" in generated_html
    assert "不可混用" in generated_html


# ── 5. 「未登记」是缺口,不是仪器 ────────────────────────────────────────────

def test_unknown_is_rendered_as_a_gap_not_as_an_instrument():
    """未登记的依赖列 ⟹ 「未登记」,而且它有自己的样式。

    这个用例原来是拿 `f0` 当例子的(真实数据走**整场录音重算**那条路时,8 条音频
    指标的 `required_l0` 就是 `f0` / `rms` / `frame_sec`)。2026-11-21 已把这两个
    列名登记为「音频」、把 `frame_sec` 登记为"不是仪器" ⟹ 那个真实缺口关掉了,
    所以这里改用**一个真的还没登记的列名**(皮肤电导,语料里没有、登记表里也没有)
    来守住同一条判据:查不到就必须说查不到,而且要有缺口样式。
    """
    row = {"name": "pitch_median", "required_l0": ["skin_conductance"]}
    assert instrument_provenance.instruments_of(row) == [
        instrument_provenance.INSTRUMENT_UNKNOWN]

    cell = _instrument_cell({"instruments": ["unknown"]}, l1_report.INSTRUMENT_LABEL)
    assert "未登记" in cell
    assert 'class="prov-unknown"' in cell, "「未登记」没用缺口样式 —— 会被当成一种仪器"

    # 汇总区:点出缺的是哪几列,并说明它是缺口
    # ⚠️ 深拷贝:`payload` 是 module 级 fixture,就地改会把后面的用例一起改坏。
    payload = copy.deepcopy(build_l1_report(_SID, logs={}))
    payload["l1"]["instruments"] = {
        **payload["l1"]["instruments"],
        "counts": {**payload["l1"]["instruments"]["counts"], "unknown": 8},
        "unmapped_columns": ["skin_conductance"],
        "footer": "共 38 条指标 · 仪器来源:模型系数 0 / 几何代理 15 / 音频 4 / 文本 4 / 未登记 8",
    }
    html = ReportGenerator(output_dir="/tmp")._build_html_report(payload, {}, {}, {}, [])
    assert "披露缺口" in html
    assert "<code>skin_conductance</code>" in html, (
        "只说了「8 条未登记」而没点出缺哪几列 —— 读的人没法动手")
    assert 'class="prov-warn"' in html


def test_a_payload_without_the_block_does_not_claim_unknown(tmp_path):
    """载荷没有这一块(旧引擎)⟹ 印「未提供」,不许印成「未登记」。

    两件事完全不同:前者是**这一份载荷没披露**,后者是**披露了、但登记表里查不到**。
    印成同一句,就是把"我们还没标"说成了"标了但查不到"。
    """
    legacy = _strip(build_l1_report(_SID, logs={}))
    html = ReportGenerator(output_dir="/tmp")._build_html_report(legacy, {}, {}, {}, [])
    # ⚠️ 只看正文:`_CSS` 里有一句注释提到「未登记」(解释它为什么用警示色),
    #    那不是这句话,扫进来会把这条判据变成假红。
    body = re.sub(r"(?s)<style>.*?</style>", "", html)
    assert "未提供" in body
    assert "未登记" not in body, "旧载荷被印成了「未登记」——把两种缺口混成了一句话"


# ── 6. 页脚与逐条标签同一套措辞 ────────────────────────────────────────────

def test_labels_agree_with_the_footer():
    """`format_footer()` 里那张局部标签表与本模块的 `INSTRUMENT_LABEL` 必须逐字一致。

    为什么这里要"抄"一份:`format_footer()` 的标签是**函数内局部变量**,
    `l1_report` 读不到;而 `instrument_provenance.py` 不归 A5 接入这次改动。
    两份措辞分叉的后果很具体:页脚写「几何代理」、指标旁边写「几何」,
    读者会以为是两样东西。⟹ 用 `format_footer()` 的**真实输出**逐项对账。
    """
    rows = [
        {"name": "a", "required_l0": ["bs_mouthSmileLeft"]},      # 模型系数
        {"name": "b", "required_l0": ["au12_smile"]},             # 几何代理
        {"name": "c", "required_l0": ["skin_conductance"]},       # 未登记
        {"name": "d", "required_l0": ["transcript_text"]},        # 文本
        {"name": "e", "required_l0": ["n_chars", "speech_duration_sec"]},  # 音频 + 文本
    ]
    counts = instrument_provenance.describe(rows)
    footer = instrument_provenance.format_footer(rows)
    assert set(counts) == set(l1_report.INSTRUMENT_LABEL), (
        "仪器集合与标签表对不上 —— 新增仪器时这里会红,提醒同改两处")
    for name, label in l1_report.INSTRUMENT_LABEL.items():
        assert f"{label} {counts[name]}" in footer, (
            f"页脚里的「{label}」措辞与标签表不一致:{footer}")
    assert counts["text"] == 2, "多仪器指标没在两个仪器下各计一次"


# ── 7. 空日志那条路:38 条指标照样逐条有披露 ────────────────────────────────

def test_every_indicator_is_disclosed_even_without_any_logs():
    """空日志(`logs={}`)装配出来的 38 条指标,也必须**逐条**带得出仪器来源。

    这条护的是"披露不依赖有没有采到数":指标不出值时更要说清是哪一路仪器没给数,
    否则读者会把"没采到"读成"没这项能力"。

    ⚠️ 注意本仓的测试环境:`tests/conftest.py` 把语音日志目录重定向到临时目录,
    所以"用真会话号跑一遍"在 pytest 里并不等于读到真素材(见上一条用例的说明)。
    真素材证据在 `verify/` 的 CLI 脚本里,不在 pytest 里假装。
    """
    p = build_l1_report(_SID, logs={})
    rows = p["l1"]["indicators"]
    assert len(rows) == 38
    for row in rows:
        assert instrument_provenance.instruments_of(row), f"{row['name']} 没带仪器来源"

    block = p["l1"]["instruments"]
    assert "仪器来源" in block["footer"]
    # 「未登记」与「缺哪几列」必须同真同假 —— 只报个数读者无从下手,只报列名又说不清量级。
    # ⚠️ 写成 if 分支里的断言会退化成"只在有缺口时才执行"(2026-11-21 把 `f0`/`rms`
    #    登记进音频之后,未登记变 0,那条断言就再也没跑过 —— 被 meta 检查抓了出来)。
    unknown_n = block["counts"][instrument_provenance.INSTRUMENT_UNKNOWN]
    unmapped = block["unmapped_columns"]
    assert bool(unmapped) == bool(unknown_n), (
        f"未登记 {unknown_n} 条,却报了 {len(unmapped)} 个缺失列名 {unmapped} —— "
        "两者必须同时出现或同时为空")


def test_a_payload_without_the_block_does_not_claim_unknown(tmp_path):
    """载荷没有这一块(旧引擎)⟹ 印「未提供」,不许印成「未登记」。

    两件事完全不同:前者是**这一份载荷没披露**,后者是**披露了、但登记表里查不到**。
    印成同一句,就是把"我们还没标"说成了"标了但查不到"。
    """
    legacy = _strip(build_l1_report(_SID, logs={}))
    html = ReportGenerator(output_dir="/tmp")._build_html_report(legacy, {}, {}, {}, [])
    # ⚠️ 只看正文:`_CSS` 里有一句注释提到「未登记」(解释它为什么用警示色),
    #    那不是这句话,扫进来会把这条判据变成假红。
    body = re.sub(r"(?s)<style>.*?</style>", "", html)
    assert "未提供" in body
    assert "未登记" not in body, "旧载荷被印成了「未登记」——把两种缺口混成了一句话"


# ── 6. 页脚与逐条标签同一套措辞 ────────────────────────────────────────────

def test_labels_agree_with_the_footer():
    """`format_footer()` 里那张局部标签表与本模块的 `INSTRUMENT_LABEL` 必须逐字一致。

    为什么这里要"抄"一份:`format_footer()` 的标签是**函数内局部变量**,
    `l1_report` 读不到;而 `instrument_provenance.py` 不归 A5 接入这次改动。
    两份措辞分叉的后果很具体:页脚写「几何代理」、指标旁边写「几何」,
    读者会以为是两样东西。⟹ 用 `format_footer()` 的**真实输出**逐项对账。
    """
    rows = [
        {"name": "a", "required_l0": ["bs_mouthSmileLeft"]},      # 模型系数
        {"name": "b", "required_l0": ["au12_smile"]},             # 几何代理
        {"name": "c", "required_l0": ["skin_conductance"]},       # 未登记
        {"name": "d", "required_l0": ["transcript_text"]},        # 文本
        {"name": "e", "required_l0": ["n_chars", "speech_duration_sec"]},  # 音频 + 文本
    ]
    counts = instrument_provenance.describe(rows)
    footer = instrument_provenance.format_footer(rows)
    assert set(counts) == set(l1_report.INSTRUMENT_LABEL), (
        "仪器集合与标签表对不上 —— 新增仪器时这里会红,提醒同改两处")
    for name, label in l1_report.INSTRUMENT_LABEL.items():
        assert f"{label} {counts[name]}" in footer, (
            f"页脚里的「{label}」措辞与标签表不一致:{footer}")
    assert counts["text"] == 2, "多仪器指标没在两个仪器下各计一次"
