# tests/test_gap_explanations.py
"""缺口码要说成**一句人话**,而且特例不许套通用句。

## 为什么有这个文件

2026-11 的实测:一份交付物里读者能看到的关于「这一条为什么没有值」的全部信息是

    connective_density → discourse_marker_rate:not_stored

`l1_quality.GAP_EXPLANATION` 后来把码翻成了中文(「上游只存了聚合量…」),
但**按码**的翻译答不了一件事:同一个码、不同指标,病因可以完全不同。

| 指标 | 码 | 通用句指向的补救动作 | 真实情况 |
|---|---|---|---|
| `pause_ratio` | `not_produced` | 无(好) | 与发声占比秩相关 0.999,不产出是对的 |
| `micro_exp_*` | `column_empty` | 去改采集 | 列在、公式在,**没有计算代码** ⟹ 补采也没用 |
| `response_latency_sec` | `window_none` | 无窗口 | 26/38 场没有题目时刻台账 ⟹ 补台账就有 |

后两行会让使用者去做**一件白做的事**。本仓为此栽过两次(见
`l1_indicators.GAP_NOT_PRODUCED` 的注释)。这个文件钉的就是"别再做白事"。

## 这个文件守什么

1. **9 类缺口码各有一条通用解释**,每句都同时答"为什么"和"能做什么";
2. **4 个已知特例有专属解释**(`pause_ratio` / 两条 `micro_exp_*` /
   `response_latency_sec`),且**与通用句不同** —— 相同就等于没做这个补丁;
3. **未知码返回 `None`** —— 不兜一句"原因未登记",那会让新缺口看起来已解释;
4. **每句 ≤60 字**;
5. 措辞规矩:不含缺口码常量、不含指标键名(英文)、不含 `**`/反引号/`⟹`/`→`、
   不含行话(`本层`/`当前管线`/`同构念`/`证据门`/`过门`/`效标`)——
   这些字直接进交付物正文,`tests/test_report_says_something_to_a_person.py`
   对正文有同样的规矩;
6. `provisional_flags.mark_provisional` 的三条硬规矩(无名/无数/无依据都拒)。
"""
from __future__ import annotations

import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

#: 被测模块可能有两处落点 —— 装进包之后在 `report_frontend/`,未安装时在 `patch/`。
#: 两种都要能跑:本补丁包的使用方式是"先跑测试再 cp",测试不该因为"还没 cp"而红。
_WHERE = ("report_frontend", "patch")


def _load(modname: str):
    try:  # ① 已安装进包(推荐落点)
        return importlib.import_module(f"report_frontend.{modname}")
    except Exception:  # noqa: BLE001 - 未安装时走 ②
        pass
    for sub in _WHERE:  # ② 按路径加载
        p = REPO / sub / f"{modname}.py"
        if p.is_file():
            spec = importlib.util.spec_from_file_location(f"_jx_{modname}", p)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[f"_jx_{modname}"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise RuntimeError(
        f"找不到 {modname}.py —— 它该在 report_frontend/ 或 patch/ 下")


gapx = _load("gap_explanations")
prov = _load("provisional_flags")
#: A5(2026-11-21)之后,明细行会多带 `instrument`/`instruments` —— 仪器名要从
#: 登记处取,不在这里抄一份字面量(否则登记表改了而测试还绿)。
instr = _load("instrument_provenance")

#: 9 个缺口码。**不从被测模块抄** —— 从包里的登记处取(唯一真源),
#: 拿不到才退回字面量;两种情况下面的 9 条用例都必须一一对上。
try:
    from report_frontend.l1_indicators import KINDS as CODES
except Exception:  # noqa: BLE001
    CODES = ("uncalibrated", "column_empty", "modality_missing", "time_unusable",
             "window_none", "no_samples", "not_stored", "not_extracted",
             "not_produced")

#: 4 个特例(名字, 实际会出现的码)。**码是从产出方读出来的**:
#: * `pause_ratio`  ⟵ `l1_voice._NOT_EXTRACTABLE_GAP`(`not_produced`)
#: * `micro_exp_*`  ⟵ `l1_face` 的 `GAP_NO_SAMPLES if present else GAP_COLUMN_EMPTY`
#: * `response_latency_sec` ⟵ `l1_voice._NOT_EXTRACTABLE_GAP`(`window_none`)
SPECIALS = (
    ("pause_ratio", "not_produced"),
    ("micro_exp_rate_per_min", "column_empty"),
    ("micro_exp_duration_mean_sec", "column_empty"),
    ("response_latency_sec", "window_none"),
)

#: 交付物正文里不许出现的行话 —— 与 `test_report_says_something_to_a_person`
#: 的 `_JARGON` 同源。这里提前在**产出处**拦一道,不指望渲染层擦。
_JARGON = ("指标槽", "证据门", "过门", "同构念", "效标", "本层", "当前管线")
_SYMBOLS = ("⟹", "→", "⇒")


def _explain(name: str, gap: str | None) -> str | None:
    return gapx.explain({"name": name, "gap": gap, "value": None,
                         "window": "session", "n_valid": 0, "n_total": 0})


# ---------------------------------------------------------------------------
# ① 9 类缺口码各一条
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code", CODES)
def test_every_gap_kind_has_one_generic_sentence(code):
    """★ 红法:从 `GENERIC` 里删掉任意一条 ⟹ 对应那条参数红。

    地板:`CODES` 必须是 9 个 —— 哪天登记处少了一个码,本测试会跟着少一条,
    而不是静默全绿。所以这里先钉住条数。
    """
    assert len(CODES) == 9, f"缺口码不再是 9 类:{CODES}"
    text = _explain("some_indicator", code)
    assert isinstance(text, str) and text.strip(), f"{code} 没有通用解释"
    # 每句都要答"为什么"与"能做什么":后者用「要/补/跑/换/等/别」这类动作词落地。
    assert re.search(r"[要补跑换带改等别重直]", text), (
        f"{code} 的解释没有给读者任何动作:{text}")
    assert text == gapx.GENERIC[code], "通用句必须来自 GENERIC 这张表,不许另拼"


def test_the_nine_sentences_are_all_different():
    """★ 红法:把两句写成一模一样 ⟹ 红。

    9 个码的**病因不同**,解释必须不同 —— 两句相同就说明其中一个没被真正想过。
    """
    values = list(gapx.GENERIC.values())
    assert len(set(values)) == 9, f"有重复的解释:{values}"


def test_generic_sentences_do_not_leak_the_code():
    """★ 红法:在解释里写一句「(not_stored)」⟹ 红。读者不认识这些常量。"""
    for code, text in gapx.GENERIC.items():
        assert code not in text, f"{code} 的解释里印出了缺口码本身:{text}"


# ---------------------------------------------------------------------------
# ② 4 个特例各一条,且**必须与通用句不同**
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,code", SPECIALS)
def test_the_four_known_cases_get_their_own_sentence(name, code):
    """★ 红法:把 `SPECIAL` 清空 ⟹ 四条全红。

    这是本补丁的**全部价值**:同名同码返回的必须是专属句,不是通用句。
    """
    text = _explain(name, code)
    assert isinstance(text, str) and text.strip(), f"{name} 没有专属解释"
    assert text != gapx.GENERIC[code], (
        f"{name} 拿到的还是 {code} 的通用句 —— 那么「专属解释」等于没写")


def test_pause_ratio_says_why_it_is_deliberately_not_produced():
    """`pause_ratio` 的理由是**恒等变换**,不是一个笼统的"故意不算"。

    实测(39 场序列复核):它与 `voiced_ratio` 的秩相关 −0.998;按
    "全部非语音"算时是解析上的恒等式(`1/voiced_ratio − 1`)。⟹ 解释里必须
    点到"同一个数",否则读者会以为系统偷懒。
    """
    text = _explain("pause_ratio", "not_produced")
    assert "发声占比" in text, f"没有点明它与谁重复:{text}"
    assert "0.99" in text, f"没有给出相关系数(证据):{text}"


def test_micro_exp_says_it_is_not_a_data_problem():
    """两条 `micro_exp_*`:实测列在、公式在、**代码里没有计算分支**。

    ⟹ 解释必须让读者知道"补采没用" —— 否则他真会去重录一场。
    """
    for name in ("micro_exp_rate_per_min", "micro_exp_duration_mean_sec"):
        text = _explain(name, "column_empty")
        assert "代码" in text, f"{name} 没说是代码的问题:{text}"
        assert "不用补采" in text or "不是数据问题" in text, (
            f"{name} 没拦住「去补采」这个白做的动作:{text}")


def test_response_latency_blames_the_missing_question_log():
    """`response_latency_sec` 报 `window_none` 的真相:多数场没有题目时刻台账。

    26/38 场如此。⟹ 解释里要有"题目时刻"这条补救路径,且不能说成算力问题。
    """
    text = _explain("response_latency_sec", "window_none")
    assert "题目时刻" in text or "起止时刻" in text, (
        f"没说清缺的是题目时刻台账:{text}")
    assert "补" in text, f"没给出补救动作:{text}"


def test_the_special_cases_do_not_bleed_into_other_codes():
    """★ 红法:把 `SPECIAL` 改成按名字命中(忽略码)⟹ 红。

    同一个指标换一个码,病因就换了。实测 `response_latency_sec` 在**有**题目台账
    的场次是有值的;若哪天它报 `no_samples`,那是"这一场没答",不是"没存台账"。
    """
    text = _explain("response_latency_sec", "no_samples")
    assert text == gapx.GENERIC["no_samples"], (
        f"`window_none` 的专属句被套到了 `no_samples` 上(判词比事实更狠):{text}")
    text = _explain("pause_ratio", "not_extracted")
    assert text == gapx.GENERIC["not_extracted"], (
        f"`not_produced` 的专属句被套到了 `not_extracted` 上:{text}")


# ---------------------------------------------------------------------------
# ③ 未知/缺失的码返回 None
# ---------------------------------------------------------------------------

def test_unknown_gap_code_returns_none():
    """★ 红法:让 `explain` 对未知码返回一句"原因未登记"⟹ 红。

    兜底句会让**没登记过的新缺口**看起来像已经解释过了。宁可露出英文。
    """
    assert _explain("some_indicator", "some_new_code") is None
    assert _explain("some_indicator", "") is None
    assert _explain("some_indicator", None) is None


def test_a_measured_indicator_explains_nothing():
    """有值的指标没有"为什么没有值"可说 —— 返回 `None`,不许编一句。"""
    assert gapx.explain({"name": "blink_rate_per_min", "gap": None,
                         "value": 14.0, "confidence": "中"}) is None


def test_explain_tolerates_a_dict_without_a_name():
    """载荷里只给码、不给名字时仍能用通用句(调用方不必先判键存在)。"""
    assert gapx.explain({"gap": "no_samples"}) == gapx.GENERIC["no_samples"]


def test_explain_rejects_non_dict_input():
    """传进来不是字典 ⟹ `None`,不抛错 —— 报告是读侧,不许因为一行坏数据炸掉。"""
    assert gapx.explain(None) is None
    assert gapx.explain("no_samples") is None


# ---------------------------------------------------------------------------
# ④ 长度 ≤60 字,且不带内部记号
# ---------------------------------------------------------------------------

def _all_sentences():
    out = [(f"GENERIC[{c}]", t) for c, t in gapx.GENERIC.items()]
    for name, by_gap in gapx.SPECIAL.items():
        for code, text in by_gap.items():
            out.append((f"SPECIAL[{name}][{code}]", text))
    return out


def test_every_sentence_is_one_sentence():
    """★ 红法:把某句扩成两句、或加一段解释 ⟹ 红。

    上限 60 字是"一句话"的物理约束:超了它就不是一句话,而是又一篇文章 ——
    而读者要的正是"一句话"。
    """
    assert gapx.LIMIT == 60, "长度上限被改了(材料的说法会跟着错)"
    too_long = [(k, len(t), t) for k, t in _all_sentences() if len(t) > gapx.LIMIT]
    assert not too_long, "这些解释超过 60 字:\n  " + "\n  ".join(
        f"{k} ({n} 字):{t}" for k, n, t in too_long)


def test_every_sentence_has_no_internal_notation():
    """★ 红法:在任意一句里塞一个指标键名或 `⟹` ⟹ 红。

    这些字**直接进交付物正文**,而正文的规矩是「不许有内部记号」
    (`tests/test_report_says_something_to_a_person.py` 的 `_JARGON` / `_SYMBOLS`)。
    在产出处拦一道,比指望渲染层擦干净更可靠 —— 交付物不止 HTML 一个出口。
    """
    bad = []
    for key, text in _all_sentences():
        if re.search(r"[A-Za-z]", text):
            bad.append((key, "英文/指标键", text))
        if "**" in text or "`" in text:
            bad.append((key, "markdown 记号", text))
        if any(s in text for s in _SYMBOLS):
            bad.append((key, "推理符号", text))
        if any(w in text for w in _JARGON):
            bad.append((key, "行话", text))
        if re.search(r"n_valid|n_total", text):
            bad.append((key, "字段名", text))
    assert not bad, "解释里带着内部记号:\n  " + "\n  ".join(
        f"{k}:{why} —— {t}" for k, why, t in bad)


def test_the_fallback_codes_are_the_same_nine():
    """★ 红法:给 `_FALLBACK_KINDS` 少写一个码 ⟹ 红。

    模块为了能脱离包单独运行,自带了一份 9 码退路。第二份表就是迟早分叉的地方,
    所以在这里逐字对账 —— 包里那份才是真源。
    """
    assert tuple(gapx._FALLBACK_KINDS) == tuple(CODES), (
        f"退路码表与包里登记处不一致:{gapx._FALLBACK_KINDS} vs {CODES}")


# ---------------------------------------------------------------------------
# ⑤ 两个入口:整句(`explain`)/ 只给特例(`explain_specific`)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,code", SPECIALS)
def test_the_two_entries_agree_on_the_four_known_cases(name, code):
    """★ 红法:让 `explain_specific` 也返回通用句 ⟹ 红。

    两个入口对特例必须**给同一句**:`explain()` = 整句,`explain_specific()` =
    同一句(因为特例句本来就是完整的一句)。接入处一个用它顶替通用句、
    一个用它单独成行,两边分叉会让同一份报告的两处说法不同。
    """
    got = gapx.explain_specific({"name": name, "gap": code})
    assert got == gapx.explain({"name": name, "gap": code}), (
        f"{name} 在两个入口拿到了不同的解释:{got!r}")


def test_explain_specific_stays_silent_for_generic_cases():
    """★ 红法:让 `explain_specific` 兜底返回通用句 ⟹ 红。

    这一条守的是**不重复**:`_gap_lines` 会用它**顶替**通用句,若它也为普通槽
    返回句子,那么每个缺口行都变成了两张表拼起来 —— 实测第一版就是这样,
    产出的是「本场没有这个模态的日志;这一场没有这类设备的数据」。
    """
    assert gapx.explain_specific({"name": "blink_rate_per_min",
                                  "gap": "modality_missing"}) is None
    assert gapx.explain_specific({"name": "pause_ratio",
                                  "gap": "not_extracted"}) is None
    assert gapx.explain_specific({"name": "some_indicator"}) is None
    assert gapx.explain_specific(None) is None


# ---------------------------------------------------------------------------
# ⑥ `provisional_flags`:未经标定的阈值必须自带理由
# ---------------------------------------------------------------------------

def test_mark_provisional_returns_value_with_status_and_basis():
    """0.75 这个转写置信度阈值的标准登记形状。"""
    marked = prov.mark_provisional(
        "asr_low_conf", 0.75, "无标定集;135 个逐题转写里 32.6% 低于它,不具区分力")
    assert marked["value"] == 0.75, "标记不许改值 —— 它不是折算器"
    assert marked["status"] == "provisional"
    assert "32.6%" in marked["basis"]
    assert marked["name"] == "asr_low_conf"
    assert prov.is_provisional(marked)
    assert prov.line(marked) == "asr_low_conf:0.75(未标定)"


@pytest.mark.parametrize("basis", ["", "   ", "TODO", "?", "无", None, {}, []])
def test_a_provisional_mark_without_a_basis_is_refused(basis):
    """★ 红法:让空依据通过 ⟹ 红。

    不带理由的 provisional 等于没标:读者照样不知道这个数有没有依据。
    """
    with pytest.raises(ValueError):
        prov.mark_provisional("asr_low_conf", 0.75, basis)


@pytest.mark.parametrize("value", [None, "0.75", True, [0.75]])
def test_a_provisional_mark_needs_a_real_number(value):
    """★ 红法:放过 `None` ⟹ 红。"没有数"该报缺口,不该报"这个数不可信"。"""
    with pytest.raises(ValueError):
        prov.mark_provisional("asr_low_conf", value, "无标定集")


def test_a_provisional_mark_needs_a_name():
    """★ 红法:放过空名字 ⟹ 红。清单里一排没名字的值分不清是哪一条。"""
    with pytest.raises(ValueError):
        prov.mark_provisional("", 0.75, "无标定集")


def test_a_plain_number_is_not_a_provisional_mark():
    """★ 红法:让 `is_provisional(0.75)` 为真 ⟹ 红。

    这一条守的是**标记与数的区分**:0.75 本身不带标签,只有登记过才带。
    报告层据此决定要不要印"未标定"。
    """
    assert not prov.is_provisional(0.75)
    assert not prov.is_provisional(None)
    assert not prov.is_provisional({"value": 0.75, "status": "calibrated"})
    assert prov.line(0.75) == ""


# ---------------------------------------------------------------------------
# ⑦ 接入点的钉子(装上 `patch/B2B4_应用.md` 那两段之后必须绿)
# ---------------------------------------------------------------------------

_L1_REPORT_SRC = REPO / "report_frontend" / "l1_report.py"
_L1_QUALITY_SRC = REPO / "report_frontend" / "l1_quality.py"


def _wired(path: Path) -> bool:
    """这个文件有没有接上本模块 —— 没接就 **skip**,不是"绿"。

    ## 为什么用 skip 而不是"红"

    本补丁包的分发方式是"先跑测试、再 cp、再贴片段"。安装**之前**这条断言
    必然不成立,若让它红,使用者看到的第一屏就是红的 —— 而那时候什么都没坏。
    ⟹ 未接入时 skip 并把"装好之后必须绿"写进理由;

    ## 为什么不能只有 skip

    skip 是**看不见的绿**。所以另有一条 `test_the_two_snippets_are_both_or_neither`:
    只贴了一段、漏了另一段 ⟹ 红。这样"忘了"不会躲进 skip 里。
    """
    return path.is_file() and "gap_explanations" in path.read_text(encoding="utf-8")


needs_snippets = pytest.mark.skipif(
    not (_wired(_L1_REPORT_SRC) and _wired(_L1_QUALITY_SRC)),
    reason="接入片段还没粘贴(见 patch/B2B4_应用.md);装好之后本测试必须绿")


def test_the_two_snippets_are_both_or_neither():
    """★ 红法:只贴 ①(缺口行)、漏了 ②(明细行)⟹ 红。

    两处服务两个出口:①进报告正文(读者读的那份),②进逐条明细(复核查的那份)。
    只贴一处会造出"正文说了、明细没说"的不一致 —— 本仓对同一件事两个出口说两句
    话的病,一向是当缺陷记的(见 `l1_quality.modality_summary` 的注释)。
    """
    a, b = _wired(_L1_REPORT_SRC), _wired(_L1_QUALITY_SRC)
    assert a == b, (
        f"接入片段只贴了一半:l1_report={'贴了' if a else '没贴'} / "
        f"l1_quality={'贴了' if b else '没贴'} —— 两处要一起装,见 B2B4_应用.md")


@needs_snippets
def test_the_report_gap_lines_use_the_special_sentence():
    """★ 红法:把 `_gap_lines` 里的 `explain_specific(...)` 删掉 ⟹ 红。

    这是**端到端**那一条:喂一个真会出现的槽(反应延迟 / 窗口切不出来),
    报告正文那一行必须是专属句,而不是通用句。
    """
    from report_frontend import l1_mapper, l1_report

    slot = {"slot": "reaction_time", "l1": "response_latency_sec", "value": None,
            "gap": "window_none", "n_valid": 0, "confidence": "无"}
    dim = types.SimpleNamespace(slots=[slot], dropped=[])
    lines = l1_report._gap_lines(dim)
    want = l1_mapper.slot_label("reaction_time") + ":" + \
        gapx.SPECIAL["response_latency_sec"]["window_none"]
    assert lines == [want], f"正文那一行不是专属句:{lines}"


@needs_snippets
def test_the_audit_rows_carry_their_own_explanation():
    """★ 红法:把 `row["gap_explanation"] = ...` 删掉 ⟹ 红。

    逐条明细(`l1.indicators`)是复核时一条一条对的那张表。它现在多了**一个键**,
    既有键一个都没动 —— 这正是本补丁"只增不改"的兑现处。
    """
    from report_frontend.l1_indicators import unavailable, GAP_NOT_PRODUCED
    from report_frontend.l1_quality import build_quality_report
    from report_frontend.event_windows import WindowSet
    from report_frontend.l1_session import L1Result
    from report_frontend.session_timeline import Timebase, TimebaseStatus
    import pandas as pd

    face = Timebase(modality="face", status=TimebaseStatus.OK,
                    t_sec=pd.Series([0.0, 1.0]), n_frames=2,
                    wall_back_steps=0, drift_sec=0.0)
    result = L1Result(
        session_id="20260101_000000_0000",
        timebase=types.SimpleNamespace(modalities={"face": face}),
        windows=WindowSet(session_id="20260101_000000_0000",
                          granularity="session", windows=[], n_questions=0),
        indicators=[unavailable("pause_ratio", "V2_pause", "停顿总时长 ÷ 语音时长",
                                ("duration_sec", "speech_duration_sec"),
                                "session", "无量纲", GAP_NOT_PRODUCED, 0)],
    )
    row = build_quality_report(result).indicators[0]
    assert row["gap_explanation"] == gapx.SPECIAL["pause_ratio"]["not_produced"], (
        f"明细行没带专属解释:{row.get('gap_explanation')}")
    # 既有键一个不少;多出来的只允许是"仪器来源"那两键(A5,2026-11-21 接入)。
    # 再多一个就必须在这里显式登记 —— 这份断言的作用就是拦住"悄悄长出新字段"。
    base_keys = {"name", "family", "formula", "required_l0", "window", "value",
                 "unit", "n_valid", "n_total", "confidence", "gap",
                 "gap_explanation"}
    provenance_keys = {"instrument", "instruments"}
    assert base_keys <= set(row), f"明细行丢了既有键:{sorted(base_keys - set(row))}"
    extra = set(row) - base_keys
    assert extra and extra <= provenance_keys, (
        f"明细行的新增键只允许「仪器来源」两键之一,实际:{sorted(extra)}")
    assert row.get("instrument") in instr.INSTRUMENTS, (
        f"仪器名不在登记表里:{row.get('instrument')!r}")
    # 披露不许改值:加了 instrument 之后,gap/解释/数值都必须还是原来那份。
    assert row["gap"] == GAP_NOT_PRODUCED
    assert row["value"] is None and row["n_valid"] == 0
