# tests/test_report_says_something_to_a_person.py
r"""报告是**给人看的**,不是把数据库表格倒进 HTML。

## 失效现场(2026-10-26,使用者第一次拿到新版报告之后的原话)

> 「这份报告是给人看的，图文并茂，简洁易懂一针见血，你写的是什么什么东西」
> 「就一个雷达图，完全没有分析」
> 「语音晦涩难懂，这怎么看」

逐条核对过,全部成立:

| 使用者的说法 | 报告里实际是什么 |
|---|---|
| 没有分析 | 全篇 127 行只有「原始值 / 未采集到会话内变异信息 / 有效样本量」三列 |
| 就一个雷达图 | `visualizer` 只产一张图,半径是**每个维度过了几个槽**(3/2/0/0/1) —— 它不是任何行为量 |
| 语音晦涩 | 「证据单位是**段数**(观察到几次运动段)」「n_valid=1」直接印在正文里 |
| 全是跟上一次比 | 「与旧引擎报告对比时注意」那 6 条**每份报告都有**,与使用者这一场无关 |

最后一条尤其要命:那一段是**引擎迁移的说明**,写给改过引擎的人。它印在交付物
正文里,读者只会读成"跟上次比怎么样"。

## 根因不是"写得不够花哨"

前几轮把「不许编造分数和评级」做对了,然后**矫枉过正**:从"编造评级"直接跳到
"只印表格和免责声明",中间那层 —— 说人话、给参照、指出这一场哪里值得看 ——
被一起删掉了。诚实不等于不可读。

## 这个文件守什么

1. 报告开头有一段**从载荷算出来的**人话结论(不是渲染层编的固定句);
2. 一个 0 值必须说清"看了多少帧、一次也没发生",不许只印 `0`;
3. 「与旧引擎报告对比」那段**移到附录**(折叠),正文里不许再出现;
4. 逐题离散度(`l1.dispersion`,载荷里一直有、报告里从没渲染过)必须画出来;
5. 图**不止一张**,而且画的是行为量,不是"过了几个槽"的计数;
6. 报告是**单文件**:不再 iframe 引 4.8 MB 的 plotly 图。
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
    """正文 —— 先剥注释与 script/style,再剥标签。"""
    body = re.sub(r"(?s)<!--.*?-->", "", html)
    body = re.sub(r"(?s)<(script|style).*?</\1>", "", body)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body))


def _body_before_appendix(html: str) -> str:
    """附录(`<details>`)之前的那部分 —— 也就是"正常读完"能看到的内容。"""
    m = re.search(r"(?s)<details", html)
    return _text_of(html if m is None else html[:m.start()])


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path / "no_rec"))
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path / "no_series"))
    return tmp_path


def _render(tmp_path, monkeypatch, payload=None):
    log_dir = tmp_path / "logs"
    # `parents=True`:本测试会传 `tmp_path / "a"` 这样的子路径,子目录本身还不存在。
    log_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(report_generator, "LogDataLoader",
                        lambda *a, **k: LogDataLoader(str(log_dir)))
    monkeypatch.setattr(report_generator.webbrowser, "open", lambda *a, **k: None)
    if payload is not None:
        monkeypatch.setattr(report_generator, "build_l1_report",
                            lambda sid, logs=None: payload)
    p = ReportGenerator(output_dir=str(tmp_path / "out")).generate_report(_SID)
    assert p, "报告没生成 —— 本测试的前提不成立"
    return open(p, encoding="utf-8").read()


@pytest.fixture
def html(isolated, monkeypatch):
    return _render(isolated, monkeypatch)


# ---------------------------------------------------------------------------
# ① 开头得有一句人话结论,而且是**从载荷算的**
# ---------------------------------------------------------------------------

def _payload(n_passed: int) -> dict:
    """一个最小载荷。`n_passed` 由调用方定 —— 结论句必须跟着它变。"""
    return {
        "engine": "l1",
        "dimensions": {
            "logical_thinking": {
                "display_name": "话语结构特征", "description": "", "algorithm": "",
                "evidence_chain": [
                    {"feature": "discourse_marker_rate", "human_name": "连接词密度",
                     "raw_value": 3.4, "n_valid": 301, "n_total": 301, "std": None,
                     "observed_interval": []},
                ],
                "evidence_gaps": ["面部对称性:本层不测这一条 —— 当前管线没有测面部对称性"],
                "confidence": "中", "matched_indicators": f"{n_passed}/2",
                "stats": {},
            },
        },
        "vanished_dimensions": [{
            "key": "confidence_level", "display_name": "行为表现活跃度", "n_slots": 1,
            "dropped": [{"slot": "left_hand_spread", "human_name": "手掌张开度",
                         "plain_reason": "当前管线只测手有没有被看到,不测手掌张开的幅度",
                         "reason": "x"}],
            "reason": "x",
        }],
        "summary_narrative": "未产出综合评分。",
        "coverage": {"n_slots": 3, "n_passed": n_passed, "n_dropped": 2,
                     "confidence_cap": "中",
                     "passed_slots": []},
        "model_metadata": {"version": "l1-1", "timestamp": "2026-10-26T00:00:00"},
        "l1": {"n_total": 38, "n_measured": 35, "by_confidence": {}, "by_gap": {},
               "modality_coverage": {}, "modality_summary": {},
               "timebase": {}, "windows": {"granularity": "answer", "n_windows": 8,
                                           "n_questions": 8},
               "indicators": [], "gaps": [], "comparability_notes": [],
               "comparability_notes_plain": [], "low_confidence_notes": [],
               "low_confidence_notes_plain": [],
               "dispersion": {"smile_ratio": {"iqr": 0.13, "max": 0.38, "median": 0.25,
                                              "min": 0.11, "n_questions": 4}},
               "dispersion_note": "逐题值的题间离散度"},
    }


def _lead_paragraph(html: str) -> str:
    """只取开头那段结论句(`<p class="lead">`)。

    ⚠️ 2026-10-26 补:`test_the_lead_sentence_is_derived_from_the_payload`
    第一版比的是**整份正文**,于是它是一句空断言 —— 覆盖数 `0 / 3` 与 `1 / 3`
    本来就会让正文不同,结论句写成固定文案**照样绿**(变异实测确认过)。
    要守的是"这一段跟着载荷变",就得只取这一段。
    """
    m = re.search(r'(?s)<p class="lead">(.*?)</p>', html)
    assert m, "报告里没有结论句(`<p class=\"lead\">`)"
    return _text_of(m.group(1))


def test_the_lead_sentence_is_derived_from_the_payload(tmp_path, monkeypatch, isolated):
    """★ 红法:把结论句写成固定文案(不读 `coverage`)⟹ 这里红。

    这是本仓记过多次的形态:渲染层自己编一句话,载荷一变它就变成假话。
    """
    a = _lead_paragraph(_render(tmp_path / "a", monkeypatch, _payload(0)))
    b = _lead_paragraph(_render(tmp_path / "b", monkeypatch, _payload(1)))
    assert a != b, "两个覆盖数不同的载荷渲染出了逐字相同的结论句 —— 它没有读载荷"
    assert "0 / 3" in a or "0/3" in a or "一个指标槽都没过" in a, (
        f"覆盖数为 0 的那份,结论句没说出 0:{a}")
    assert "1 / 3" in b or "1/3" in b, f"覆盖数为 1 的那份,结论句没说出 1:{b}"


def test_the_lead_sentence_says_the_coverage_in_words(html):
    """结论句要把「量到几条 / 一共几条」说出来,读者不用去数格子。"""
    body = _body_before_appendix(html)
    assert re.search(r"\d+\s*/\s*14", body), "正文里没有「N / 14」这个覆盖数"
    assert "个指标槽" in body or "条指标" in body, (
        f"没说清分母是什么,读者不知道 14 指的是什么:{body[:400]}")


# ---------------------------------------------------------------------------
# ② 0 值必须解释,不许裸印
# ---------------------------------------------------------------------------

def _payload_with_zero() -> dict:
    p = _payload(1)
    p["dimensions"]["logical_thinking"]["evidence_chain"] = [
        {"feature": "smile_ratio", "human_name": "微笑占比",
         "raw_value": 0.0, "n_valid": 64, "n_total": 542, "std": None,
         "observed_interval": []},
    ]
    p["dimensions"]["logical_thinking"]["matched_indicators"] = "1/2"
    return p


def test_a_zero_is_explained_as_measured_but_never_above_threshold(
        tmp_path, monkeypatch, isolated):
    """★ 红法:把 `_zero_note` 的调用删掉 ⟹ 这里红。

    实测:`smile_ratio` = 0.0、`n_valid` = 64。读者看到 `0` 会读成
    「没测到」或「测出来是零」,而真相是**看了 64 帧,一次也没超过阈值**。
    两者对"这一场能说明什么"的含义完全不同。
    """
    html = _render(tmp_path / "z", monkeypatch, _payload_with_zero())
    body = _text_of(html)
    assert "64" in body, "报告里没有那 64 个有效帧 —— 0 值没有交代"
    assert re.search(r"(一次也没|没有一次|都没超过|未出现)", body), (
        f"0 值旁边没有一句「看了多少帧、一次也没发生」的解释:{body}")


# ---------------------------------------------------------------------------
# ③ 引擎迁移说明挪进附录
# ---------------------------------------------------------------------------

def test_the_engine_migration_notes_are_in_an_appendix_not_the_body(html):
    """★ 红法:把它挪回正文 ⟹ 这里红。

    使用者原话:「科研报告上面全是更上一次比怎么这么样」。那 6 条是
    **引擎迁移说明**,每份报告都有、与这一场无关,印在正文里必然被读成
    "跟上次比"。内容不删(复核要查),但必须收进折叠附录。
    """
    body = _body_before_appendix(html)
    assert "与旧引擎报告对比" not in body, (
        "引擎迁移说明还在正文里 —— 读者会把它读成「跟上一次比怎么样」")
    assert re.search(r"(?s)<details", html), "报告里没有附录(折叠区)"


def test_the_migration_notes_are_still_there_in_the_appendix(html):
    """另一侧:收进附录 ≠ 删掉。复核口径的人还要查得到。"""
    assert "与旧引擎报告对比" in _text_of(html), (
        "迁移说明被整段删掉了 —— 复核的人查不到两边口径的差别")


# ---------------------------------------------------------------------------
# ④ 逐题离散度:载荷里有,报告里从来没有
# ---------------------------------------------------------------------------

def test_the_per_question_dispersion_reaches_the_report(html):
    """★ 红法:把离散度那一段删掉 ⟹ 这里红。

    `l1.dispersion` 是**20 条指标的逐题中位/IQR/极值**,`l1_report` 一直在发,
    前端 `L1IndicatorTable` 也一直在渲染 —— 只有 HTML 交付物从没印过。
    它回答的正是使用者问的"这场稳不稳":同一个指标,八道题之间差多少。
    """
    body = _text_of(html)
    assert "逐题" in body or "题间" in body, "报告里没有逐题离散度这一块"
    assert "中位" in body, "离散度只印了标签,没印统计量"


# ---------------------------------------------------------------------------
# ⑤ 图不止一张,而且画的是行为量
# ---------------------------------------------------------------------------

def test_the_report_draws_more_than_one_figure(html):
    """★ 红法:退回"只有一张雷达图" ⟹ 这里红。

    使用者原话:「就一个雷达图，完全没有分析」。一张图画的是"每个维度过了
    几个槽",那不是任何行为量 —— 它是覆盖率计数,该用条形,不该占满一屏。
    """
    figures = re.findall(r'<(?:figure|svg)\b', html)
    assert len(figures) >= 2, (
        f"报告里只有 {len(figures)} 个图元素 —— 一张覆盖率图讲不完这一场")


def test_the_dispersion_figure_draws_the_actual_values(tmp_path, monkeypatch, isolated):
    """★ 红法:让离散度图只画个空框 ⟹ 这里红。

    图里必须出现载荷里的真数,否则它只是装饰。
    """
    html = _render(tmp_path / "d", monkeypatch, _payload(1))
    # 载荷里 smile_ratio 的 median=0.25 / min=0.11 / max=0.38
    for v in ("0.25", "0.11", "0.38"):
        assert v in html, f"离散度图里没有载荷中的真值 {v}"


def test_a_row_where_every_question_agrees_is_drawn_once_not_twice(
        tmp_path, monkeypatch, isolated):
    """★ 红法:把 `flat` 那条分支删掉 ⟹ 这里红。

    ## 实测怎么发现的

    真实那一场(20261026_002930_cb79)20 行里有 6 行 `min == max`:
    四条是"每道题都是 0"(一次也没超过阈值),`pose_valid_ratio` 是"每道题都是 1"。
    常规画法把两个端点标签放在**同一个 x 上**,渲染出来是糊掉的「00」/「11」
    —— 而这一行恰恰是全图**最稳**的那一行,读者看到的却是最乱的那个。

    ⟹ 各题相同就把点放正中、只标一个数、旁边写「各题相同」。
    """
    p = _payload(1)
    p["l1"]["dispersion"] = {
        # 各题相同(且是 0):最容易被误读成"没测到"
        "brow_furrow_ratio_vs_baseline": {"iqr": 0.0, "max": 0.0, "median": 0.0,
                                          "min": 0.0, "n_questions": 2},
        # 正常有跨度的一行,用来确认这条分支没有把别的行也压成平点
        "smile_ratio": {"iqr": 0.13, "max": 0.38, "median": 0.25,
                        "min": 0.11, "n_questions": 4},
    }
    fig = re.search(r'(?s)<svg[^>]*aria-label="每条指标在题与题之间的分布".*?</svg>',
                    _render(tmp_path / "f", monkeypatch, p))
    assert fig, "报告里没有离散度图"
    svg = fig.group(0)
    assert "各题相同" in svg, "各题取到同一个值的那一行没有标出来"
    # 平的那一行只该出现**一个** 0 的刻度;两个就是重叠的端点标签。
    assert svg.count(">0<") == 1, (
        f"平的那一行把同一个数标了两次(端点重叠):{svg}")
    # 有跨度那一行照常三个数都在
    for v in ("0.11", "0.25", "0.38"):
        assert f">{v}<" in svg, f"有跨度的那一行少了真值 {v}"


def test_the_report_is_one_self_contained_file(html):
    """★ 红法:把 iframe + plotly 那张 4.8 MB 的图加回来 ⟹ 这里红。

    原先每份报告旁边躺一个 `radar_chart_*.html`(单份 4.8 MB,15 份就是 72 MB),
    报告用 `<iframe src="radar_chart_….html">` 引它 —— 报告一拷走,图就没了。
    现在图是内联 SVG:一份报告就是一个文件。
    """
    assert "<iframe" not in html, "报告仍在 iframe 里引外部图 —— 拷走就没有图了"
    assert "plotly" not in html.lower(), "报告仍在加载 plotly"


# ---------------------------------------------------------------------------
# ⑥ 语音那一段的人话
# ---------------------------------------------------------------------------

def test_the_low_confidence_notes_carry_no_internal_notation(html):
    """★ 红法:把 `n_valid=1` 之类的记号写回理由 ⟹ 这里红。

    使用者原话:「语音晦涩难懂，这怎么看」。实测正文里印着
    「证据单位是**段数**」「n_valid=1」—— 前者带 markdown 星号,
    后者是代码里的字段名。
    """
    body = _text_of(html)
    for token in ("n_valid", "n_total", "**", "`"):
        assert token not in body, f"交付物正文里还印着内部记号 {token!r}"


def test_no_internal_indicator_key_survives_in_the_report_body(html):
    """★ 红法:把某个理由里的指标键改回来 ⟹ 这里红。

    与 `test_report_numbers_and_reasons_are_readable` 同一条规矩,
    这里再钉一遍:低置信理由曾把 `gesture_burst_rate_per_min_vs_baseline`
    原样印进正文。
    """
    from report_frontend import l1_face, l1_gesture, l1_text, l1_voice

    names = (set(l1_face.indicator_names()) | set(l1_gesture.ORDER)
             | set(l1_voice.ORDER) | set(l1_text.ORDER))
    body = _text_of(html)
    leaked = sorted(n for n in names if n in body)
    assert not leaked, f"正文里露出了内部指标键:{leaked}"


# ---------------------------------------------------------------------------
# ⑦ 同一批数字不许印两遍
# ---------------------------------------------------------------------------

def test_the_same_numbers_are_not_printed_twice(tmp_path, monkeypatch, isolated):
    """★ 红法:把「依据:通过证据门的指标」那段平铺列表加回来 ⟹ 这里红。

    ## 实测怎么发现的

    那一版报告里,每个过门槽都印了两遍:一遍在「每一维量到了什么」的表里
    (维度 · 指标 · 原始值 · 样本量),一遍在「依据」那段里
    (维度 · 指标 · 原始值 · 区间 · 样本量)。同一批数字,一份报告。
    使用者原话:「简洁易懂」。

    更糟的是「依据」那段的每一行都带「本场会话内观测区间 未采集到会话内变异信息
    (与原始值同量纲)」—— 读者会以为每个指标都缺了区间,而真相是 L1 每个指标
    整场只有一个值,本来就没有"场内变异"。
    """
    # ⚠️ 必须用带证据的载荷:空载荷下一条指标都不过门,逐维表压根不渲染,
    #    「只印一次」就退化成"一次都没印"的空断言。
    body = _text_of(_render(tmp_path / "n", monkeypatch, _payload(1)))
    # 两次是对的:一次在「量到了这几条：…」那句人话里,一次在维度表那一行。
    # 第三次就说明又出现了一份重复的表。
    assert body.count("连接词密度") == 2, (
        f"「连接词密度」在正文里出现了 {body.count('连接词密度')} 次 —— "
        "除了「量到了这几条」与维度表各一次,多出来的就是重复")
    assert "依据：通过证据门的指标" not in body, (
        "「依据」那段平铺列表还在 —— 它与逐维表逐字重复")
    # 逐维表只有一份:表头「原始值」只该出现一次。
    assert body.count("原始值") == 1, (
        f"正文里有 {body.count('原始值')} 张指标表 —— 同一批数字印了不止一遍")


def test_an_interval_column_that_is_always_empty_is_not_printed(
        tmp_path, monkeypatch, isolated):
    """★ 红法:把 `has_interval` 判据删掉(永远印这一列)⟹ 这里红。

    L1 的每个指标都是**整场一个值**,`observed_interval` 恒为空数组。
    整列印出来就是四行同一句「未采集到会话内变异信息」—— 读者会读成
    "这一维什么都没测出变异",而真相是这一层不产生区间。
    """
    p = _payload(1)
    for e in p["dimensions"]["logical_thinking"]["evidence_chain"]:
        e["observed_interval"] = []
    html = _render(tmp_path / "i", monkeypatch, p)
    body = _text_of(html)
    assert "未采集到会话内变异信息" not in body, (
        "整列为空的区间列还在印 —— 四行同一句废话")
    assert "没有「场内的变异」可报" in body, (
        "撤掉这一列之后没给理由 —— 读者会以为这一段丢了")

    # 另一侧:真有区间时必须照印,否则这条判据会退化成"永远不印"。
    q = _payload(1)
    q["dimensions"]["logical_thinking"]["evidence_chain"][0]["observed_interval"] = [3.0, 3.8]
    body2 = _text_of(_render(tmp_path / "j", monkeypatch, q))
    assert "3.0 – 3.8" in body2, "有区间时反而不印了"
