# report_frontend/l1_text.py
"""**T 族(文本)3 条** —— `lexical_diversity` / `discourse_marker_rate` / `self_repair_rate`。

## 为什么现在才做

规格 §5.5 原本只有一句:「三个名字**先占位,标注到位前不实现**」。
卡的是**定义**,不是数据。数据一直在:`data/transcripts/<sid>.json`
(`tools/jx_transcribe_recordings.py` 的批量 ASR 产物),13/13 场 A 档全有,
320~3899 字。2026-10-18 使用者裁定按本模块的定义落地。

## 三条口径的共同约束:分母是**字**,不是时长

`voice_interaction/asr/connective_density.py` 里写着本仓的一条既有纪律:
文本层指标**不掺时长**("1410 维被时长污染"那条)。所以三条全部按**每百字**归一,
即使其中两条的名字里带 "rate"。名字是规格里的占位名,口径以公式串为准。

## ⚠️ 三条都是"全场混音"量

转写没有说话人分离(面试官 + 考生混在一起),所以三条量的都是**这一场的话语**,
不是"考生的话语"。这条限制写进每条的公式串里 —— 不写清楚,读者会把
`self_repair_rate` 当成考生的口头禅频率,而里面混着面试官的。

## 三条各自的定义(以及为什么这么定)

### ⚠️ 转写是**中英混说**的 —— 三条的分母因此按脚本分开

实测 13 场 A 档:拉丁字母占正文字数的**中位 8.3%**,但有一场高达 **49.2%**
(那场基本是英文面试)。这条事实决定了两个口径:

* `lexical_diversity` 的**词元流**按脚本分别切:汉字切**二元组**,英文切**词**
  (`[A-Za-z']+`)。否则英文会被切成 `he`/`el`/`ll`/`lo` 这类字母二元组,
  量到的是英文字母的冗余度,不是用词丰富度 —— 那一条与其他场**不可比**。
  合并后按原文顺序走同一个窗口,窗口仍然是"一段连续话语"。
* `discourse_marker_rate` / `self_repair_rate` 的**分母只用汉字数**。
  标记表是中文的,它只能出现在汉字里;拿"汉字+英文字母"当分母,
  英文多的场次会被无端稀释(49.2% 那场会直接腰斩一半)。

### `lexical_diversity` = 词元 MATTR(窗口 100)

原始 TTR(`独特字数 ÷ 总字数`)**随文本变长单调下降**,实测 13 场:
320 字那场 0.4125,3899 字那场 0.1411 —— 那是长度伪影,不是"用词丰富度"。
所以用 **MATTR**(滑动窗口 TTR 的平均):窗口固定,长度就消掉了。
实测 320 字 0.796 / 3899 字 0.857,无长度趋势。

用**二元组**而不是单字:单字 MATTR 实测挤在 0.555~0.608(动态范围 0.053),
二元组是 0.794~0.889(0.095),区分度接近两倍。中文没有分词器可用
(`jieba`/`pkuseg`/`hanlp` 均未安装),二元组是**不需要词表**的近似。

### `discourse_marker_rate` = 标记**出现次数** ÷ 字数 × 100

⚠️ 与既有的 `connective_density` **不是同一个量**:那条数的是"**几个不同标记
出现过**"(presence-once,判"这个标记在不在"),本条数的是"**总共说了多少次**"。
重复说同一个连接词是冗词问题,不是结构丰富度 —— 两条各管一头,不要互相替代。

### `self_repair_rate` = (紧邻重复的二字组 + 修正标记命中) ÷ 字数 × 100

紧邻重复的**二字组**(`就是就是`)假阳性低:中文的合法叠词(`看看`/`慢慢`)
是单字重叠,不会被二字组规则抓到。单字重复**故意不做** —— 假阳性太高。

修正标记来自 `self_repair_markers.json`(版本化、`_provisional`),
与 `connective_markers.json` 同一纪律。
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from report_frontend.l1_indicators import (
    GAP_NO_SAMPLES, GAP_NOT_STORED, L1Indicator, measured, unavailable,
)

#: MATTR 窗口(以**二元组**计)。改它必须同时改 `TEXT_MIN_CHARS` 的推导。
MATTR_WINDOW = 100

#: 算得出这三条所需的最少**字数** = 3 个不重叠窗口。
#: 这是**从窗口推出来的**,不是标定值:窗口 100 个词元 ≈ 101 字,
#: 三个窗口 ≈ 303 字,取整 300。
#:
#: ⚠️ 两条中文标记类指标的分母是**汉字数**,所以它们直接用这个常量;
#: `lexical_diversity` 的 `n_valid` 是**词元数**,必须用下面那个同单位的常量 ——
#: 拿词元数去比字数阈值,英文占比高的场次会被系统性压到「中」(900 字 ≈ 250 词元)。
TEXT_MIN_CHARS = 300

#: `lexical_diversity` 的置信门槛,**单位是词元**(与它的 `n_valid` 一致)。
TEXT_MIN_TOKENS = 3 * MATTR_WINDOW

_MARKERS_PATH = Path(__file__).with_name("self_repair_markers.json")


@lru_cache(maxsize=1)
def _load_repair_table_cached(path: str | None = None) -> dict:
    return json.loads(Path(path or _MARKERS_PATH).read_text(encoding="utf-8"))


def load_repair_table(path: str | None = None) -> dict:
    """读版本化修正标记表。**交出副本**(与 `connective_density.load_markers` 同理:
    缓存是进程级的,调用方一次就地修改会污染此后所有结果)。"""
    import copy
    return copy.deepcopy(_load_repair_table_cached(path))


def _connective_markers() -> list[str]:
    from voice_interaction.asr.connective_density import load_markers
    return load_markers()["markers"]


#: 名字 → (族, 公式, 依赖)。顺序即输出顺序。
_SPECS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "lexical_diversity": (
        "T1_lexical",
        "字二元组 MATTR(滑动窗口 100 个二元组的 TTR 均值);"
        "⚠️ 转写为全场混音(含面试官),量的是全场话语用词丰富度",
        ("transcript_text",)),
    "discourse_marker_rate": (
        "T2_discourse",
        "话语标记**出现次数** ÷ 汉字数 × 100(与 connective_density 的"
        "「不同标记出现过几次」不同口径;分母只数汉字 —— 标记表是中文的,"
        "英文词不构成可命中的材料);⚠️ 全场混音,含面试官",
        ("transcript_text",)),
    "self_repair_rate": (
        "T3_repair",
        "(紧邻重复的二字组次数 + 修正标记命中次数) ÷ 汉字数 × 100"
        "(分母只数汉字);⚠️ 全场混音,含面试官",
        ("transcript_text",)),
}

ORDER = tuple(_SPECS)


def _cjk(text: str) -> str:
    """只留汉字 —— 中文标记类指标的分母。

    口径与 `funasr_engine.count_cjk_chars` 一致(只数汉字),
    所以与转写元数据里的 `n_chars` 可以对账。
    """
    return "".join(c for c in text if "\u4e00" <= c <= "\u9fff")


#: 英文词:`[A-Za-z']+`。转写里的英文没有词形还原需求,原样小写即可。
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z']*")


def tokens(text: str) -> list[str]:
    """**按脚本切词元** —— 汉字切二元组,英文切整词,顺序保持原文。

    这是 `lexical_diversity` 的输入。为什么不能一律按字符切:
    英文按字符切会得到 `he`/`el`/`ll`/`lo` 这类**字母**二元组,
    它们量的是拼写冗余度,与中文的字二元组不是同一个构念;
    混在一张表里比较,英文多的场次会被系统性压低。

    汉字**单字**独立出现时(前后都不是汉字)也自成词元 ——
    否则那一个字会被静默丢掉。
    """
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if "\u4e00" <= c <= "\u9fff":
            j = i
            while j < n and "\u4e00" <= text[j] <= "\u9fff":
                j += 1
            run = text[i:j]
            if len(run) == 1:
                out.append(run)
            else:
                out.extend(run[k:k + 2] for k in range(len(run) - 1))
            i = j
            continue
        m = _LATIN_WORD.match(text, i)
        if m:
            out.append(m.group(0).lower())
            i = m.end()
            continue
        i += 1
    return out


def mattr(tokens: list[str], window: int = MATTR_WINDOW) -> float | None:
    """滑动窗口 TTR 的均值。`len(tokens) < window` ⟹ `None`(算不出来,不猜)。"""
    if window <= 0 or len(tokens) < window:
        return None
    total = 0.0
    n_win = len(tokens) - window + 1
    for i in range(n_win):
        total += len(set(tokens[i:i + window])) / window
    return total / n_win


def immediate_bigram_repeats(text: str) -> int:
    """紧邻重复的二字组次数:`就是就是` 计 1 次、`就是`×3 计 2 次(数的是**接缝**)。

    计数是**不重叠**的:命中一处后跳过被重复的那个单元,所以 k 次连说计 k−1 次。

    只看**二字组**,不看单字 —— 单字重叠在中文里大量合法(`看看`/`慢慢`/`试试`),
    数进来会把正常口语算成口吃。
    """
    n = 0
    i = 0
    while i + 3 < len(text):
        if text[i:i + 2] == text[i + 2:i + 4]:
            n += 1
            # ★ 命中后**跳过被重复的那个单元**,否则同一处重复会被数多次。
            #   踩过的坑:`就是`×k 在偏移 2 的周期串上**每个 i 都命中**
            #   (连奇数位也命中,因为 `是就` 也等于 `是就`),于是 k=10 数成 17
            #   而不是 9 —— 真实语料因此虚高 1.13 倍(214 vs 190)。
            i += 2
        else:
            i += 1
    return n


def compute_text_indicators(text: str | None,
                            thresholds: dict[str, int] | None = None) -> list[L1Indicator]:
    """T 族 3 条。`text is None` ⟹ 三条全 `not_stored`(不是 0)。"""
    thr = thresholds or {}
    if text is None:
        return [unavailable(n, _SPECS[n][0], _SPECS[n][1], _SPECS[n][2],
                            "session", "", GAP_NOT_STORED)
                for n in ORDER]

    body = _cjk(text)
    chars = len(body)
    toks = tokens(text)
    if chars == 0 and not toks:
        return [unavailable(n, _SPECS[n][0], _SPECS[n][1], _SPECS[n][2],
                            "session", "", GAP_NO_SAMPLES)
                for n in ORDER]

    got: dict[str, L1Indicator] = {}

    # ── T1 词汇多样性 ────────────────────────────────────────────────────
    value = mattr(toks, MATTR_WINDOW)
    if value is None:
        got["lexical_diversity"] = unavailable(
            "lexical_diversity", *_SPECS["lexical_diversity"][:2],
            _SPECS["lexical_diversity"][2], "session", "", GAP_NO_SAMPLES)
    else:
        got["lexical_diversity"] = measured(
            "lexical_diversity", *_SPECS["lexical_diversity"][:2],
            _SPECS["lexical_diversity"][2], "session", "无量纲",
            value=value, n_valid=len(toks), n_total=len(toks),
            threshold=TEXT_MIN_TOKENS)

    # ── T2 / T3 的分母是**汉字数**。整场没有汉字(纯英文面试)⟹ 分母为 0,
    #   这两条**不可出**:标记表是中文的,没有材料就没有测量,报 0 会变成
    #   "这场一次都没用连接词" —— 那是在替一场英文面试捏一个中文结论。
    if chars == 0:
        for name in ("discourse_marker_rate", "self_repair_rate"):
            got[name] = unavailable(
                name, *_SPECS[name][:2], _SPECS[name][2], "session", "", GAP_NO_SAMPLES)
        return [got[n] for n in ORDER]

    # ── T2 话语标记出现次数 ──────────────────────────────────────────────
    occ = sum(body.count(m) for m in _connective_markers())
    got["discourse_marker_rate"] = measured(
        "discourse_marker_rate", *_SPECS["discourse_marker_rate"][:2],
        _SPECS["discourse_marker_rate"][2], "session", "次/百字",
        value=occ / chars * 100.0, n_valid=chars, n_total=chars,
        threshold=TEXT_MIN_CHARS)

    # ── T3 自我修正 ──────────────────────────────────────────────────────
    repair = sum(body.count(m) for m in load_repair_table()["markers"])
    hits = repair + immediate_bigram_repeats(body)
    got["self_repair_rate"] = measured(
        "self_repair_rate", *_SPECS["self_repair_rate"][:2],
        _SPECS["self_repair_rate"][2], "session", "次/百字",
        # 0 次是**测出来的 0**(这场确实没改口),不是"没测到" ⟹ 照常给值。
        value=hits / chars * 100.0, n_valid=chars, n_total=chars,
        threshold=TEXT_MIN_CHARS)

    return [got[n] for n in ORDER]


def indicator_names() -> tuple[str, ...]:
    return ORDER
