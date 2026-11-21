# patch/gap_explanations.py  →  安装到 report_frontend/gap_explanations.py
"""**缺口码 → 一句人话(按指标)** —— 报告里"为什么不产出 / 为什么不可信"的那句话。

## 这一层与 `l1_quality.GAP_EXPLANATION` 的分工

`l1_quality.GAP_EXPLANATION` 是**按码**的通用表:9 个码各一句,回答
「这一类缺口是什么」。它已经在 `l1_report._gap_lines` 里用着,不许另抄第二份。

但"按码"回答不了一件事:**同一个码,不同指标的病因不一样**。实测:

| 指标 | 缺口码 | 通用句说什么 | 真实情况 |
|---|---|---|---|
| `pause_ratio` | `not_produced` | 故意不算 | 与发声占比是**同一个数**(秩相关 0.999)⟹ 不产出是对的 |
| `micro_exp_rate_per_min` | `column_empty` | 上游没采集这一列 | 列**在**,公式**在**,**没有计算代码** ⟹ 补采也没用 |
| `response_latency_sec` | `window_none` | 没有可用回答窗口 | 26/38 场没有题目时刻台账 ⟹ 补台账就有,与算力无关 |

三句通用话都"不算错",却都指向一个**错的补救动作**:去改采集、去等标定、
去重录。本仓栽过两次的正是这个形态(见 `l1_indicators.GAP_NOT_PRODUCED` 的
注释:上一版喊"查采集",这一版喊"再跑一次提取")。

⟹ 本模块**只增不改**:不动 `L1Indicator` 的字段、不动 `GAP_EXPLANATION`、
不改任何既有载荷键;只提供一个函数,由调用方决定接在哪一处。

## 一句解释的三段

每句都要同时回答两件事(使用者原话:「报告里只有缺口码,评委看不懂」):

1. **为什么没有值 / 为什么不可信** —— 病因,不是"证据不足"这种同义反复;
2. **能做什么** —— 补采、补台账、跑提取、改采集,还是"什么也做不了,别等"。

## 五条措辞规矩(全部由 `tests/test_gap_explanations.py` 扫)

这些字**直接进交付物正文**,所以规矩与 `l1_quality.LOW_CONFIDENCE_REASON` 同源:

* 不出现 9 个缺口码常量(读者不认识 `not_produced`);
* 不出现任何指标键名(英文)—— 正文里只许有中文名(`l1_quality.INDICATOR_LABEL`);
* 不出现 `**` / 反引号 / `⟹` / `→` / `⇒`(交付物不渲染 markdown,符号不会念);
* 不出现行话(`本层` / `当前管线` / `同构念` / `证据门` / `过门` / `效标`);
* **≤ 60 字** —— 这是"一句话"的上限。超了就不是一句话,而是又一篇文章。

## 为什么未知码返回 `None` 而不是兜底句子

兜一句"原因未登记"会让**没登记过的新缺口**看起来像已经解释过了。
返回 `None` ⟹ 调用方保留原来的通用行或露出码;宁可露出英文,也不许假装说清了。
"""

from __future__ import annotations

#: 一句话的长度上限(字符)。`tests/test_gap_explanations.py` 逐句卡。
LIMIT = 60

#: 9 个缺口码。**这是第二份**,只作 `report_frontend` 不可导入时的退路 ——
#: 单独拿这个文件去跑脚本时不该因为"包不在"就崩。
#: 与 `report_frontend.l1_indicators.KINDS` 的一致性由测试
#: `test_the_fallback_codes_are_the_same_nine` 对账(改一处漏一处会红)。
_FALLBACK_KINDS = (
    "uncalibrated", "column_empty", "modality_missing", "time_unusable",
    "window_none", "no_samples", "not_stored", "not_extracted", "not_produced",
)

try:  # pragma: no cover - 有包时走这一支(唯一的登记处)
    from report_frontend.l1_indicators import KINDS as _KINDS
    _CODES_SOURCE = "report_frontend.l1_indicators.KINDS"
except Exception:  # noqa: BLE001 - 独立运行(未装进包)时的退路
    _KINDS = _FALLBACK_KINDS
    _CODES_SOURCE = "gap_explanations._FALLBACK_KINDS"

#: **按码**的通用解释 —— 9 类各一句,每句都带"能做什么"。
#:
#: 与 `l1_quality.GAP_EXPLANATION` **逐句不同**,但**病因是重合的** —— 这没关系:
#: 那张表回答"这一类是什么",这张表在同样的病因后面**多给一个动作**
#: ("改采集 / 补台账 / 跑一次提取 / 别等")。两者都留着是有意的:
#: 前者已经进了 `evidence_gaps`,动它要改既有载荷与三条钉子;
#: 后者是新增的一句话,接不接由调用方决定 —— 真接进同一处时用
#: `explain_specific()`(用特例句**顶替**通用句),别把两句拼起来。
#: 拼起来的实测后果见 `explain_specific()` 的 docstring。
GENERIC: dict[str, str] = {
    "uncalibrated": "缺的是一条标定出来的判定门限,数据本身在;标定完成前按缺值读,别当零。",
    "column_empty": "上游采集没有这一列,再等也不会自己出现;要它得改采集,补录无用。",
    "modality_missing": "这一场没有这类设备的数据;下次带上这类设备重录才有。",
    "time_unusable": "时间轴对不上,切不出时段;只能重录,或在采集端把时钟修好。",
    "window_none": "这一场没有答题起止时刻,切不出答题片段;补录题目时刻即可。",
    "no_samples": "这个片段里没有一条有效观测(不等于零);换一段或补录。",
    "not_stored": "上游只存了汇总数,粒度太粗,算不出这一条;要补只能改采集。",
    "not_extracted": "素材就在盘上,只是还没算过;跑一次提取就有值,不必重录。",
    "not_produced": "这一条是故意不算的:与另一条是同一个数的换算,再等也不会有。",
}

#: 四个已知特例的专属解释。结构是 `{指标名: {缺口码: 一句话}}` ——
#: **码也要对上**才用,不是"按指标名命中"。同一个指标换了码,病因就换了,不许硬套。
#:
#: 为什么必须按键对而不是按指标名:实测 `response_latency_sec` 在**有**题目台账的
#: 场次是有值的(`reaction_time` 列),报 `no_samples` 时的病因与 `window_none`
#: 完全不同。按名字一刀切会把"这一场没答"说成"没存题目时刻"。
_MICRO = "微表情这两条只有公式、没有计算代码,任何一场都不会出值;不是数据问题,不用补采。"

_PAUSE_RATIO = ("停顿占比与发声占比是同一个数换了个算法(秩相关 0.999),"
                "重复报等于把同一份证据数两遍,直接读发声占比。")

_RESPONSE_LATENCY = ("要的是每题提问结束那一刻,而多数场次没存题目时刻表;"
                     "补上题目时刻就能算,与算力无关。")

SPECIAL: dict[str, dict[str, str]] = {
    "pause_ratio": {"not_produced": _PAUSE_RATIO},
    "micro_exp_rate_per_min": {"column_empty": _MICRO, "no_samples": _MICRO},
    "micro_exp_duration_mean_sec": {"column_empty": _MICRO, "no_samples": _MICRO},
    "response_latency_sec": {"window_none": _RESPONSE_LATENCY},
}

#: 四个特例指标的名字(供调用方/测试清点,不用自己再列一遍)。
SPECIAL_NAMES: tuple[str, ...] = tuple(SPECIAL)


def explain(indicator: dict) -> str | None:
    """一条指标 → 一句人话解释。**没有缺口就返回 `None`**。

    ## 入参只读这几个键(取不到就当没有,不抛错)

    * `name` —— 指标名(`L1Indicator.name`,如 `pause_ratio`);
    * `gap`  —— 缺口码(`L1Indicator.gap` 的 9 个值之一);

    其它键(`value` / `n_valid` / `window` …)**一概不看**:这一层只解释缺口,
    不做判断、不改数。传 `L1Indicator.to_dict()` 出来的那一行正好够用。

    ## 三种返回

    | 情形 | 返回 |
    |---|---|
    | `gap` 是 9 个已登记的码之一 | 一句 ≤60 字的中文解释(特例优先) |
    | `gap is None` / 空 / 不在 9 类里 | `None` —— **不兜底**,见模块 docstring |

    ## 为什么特例只认 `(名字, 码)` 这一对

    见 `SPECIAL` 的注释:同一个指标换一个码,病因就换了。硬套会把
    "这一场没答"说成"没存题目时刻",那正是本仓要消灭的"判词比事实更狠"。
    """
    if not isinstance(indicator, dict):
        return None
    gap = indicator.get("gap")
    if not isinstance(gap, str) or gap not in _KINDS:
        return None
    name = indicator.get("name")
    if isinstance(name, str):
        hit = SPECIAL.get(name, {}).get(gap)
        if hit:
            return hit
    return GENERIC[gap]


def explain_specific(indicator: dict) -> str | None:
    """**只**返回特例专属句;不是 4 个特例(或码对不上)⟹ `None`。

    ## 为什么需要第二个入口(2026-11 实测踩到)

    第一版接入是把 `explain()` 接在通用句**后面**(`f"{why};{extra}"`)。
    实测产出的正文:

        连接词密度:上游只存了聚合量,这一层需要的粒度没落盘;上游只存了汇总数,
        粒度太粗,算不出这一条;要补只能改采集。

    同一件事说了两遍 —— 通用表(`l1_quality.GAP_EXPLANATION`)与这里的
    `GENERIC` 病因相同,只有"能做什么"是新信息。判词加倍,读者只会更烦。

    ⟹ 两个入口分工:

    * `explain()` —— **整句**(病因 + 动作)。给"一行一条指标、旁边没有通用句"的
      表格用(`l1.indicators` 审计链、38 条台账);
    * `explain_specific()` —— **只给特例句,用来**顶替**通用句(不是追加)**。
      给已经印了通用句的地方用(报告正文的缺口行):那里不缺病因,缺的是
      "这一条**特殊**在哪、该做什么"。

    所以特例句本身必须是**完整的一句**(病因 + 动作),不能是半句 ——
    它会把通用句整句替换掉,不是拼在后面。
    """
    if not isinstance(indicator, dict):
        return None
    gap = indicator.get("gap")
    name = indicator.get("name")
    if not isinstance(gap, str) or not isinstance(name, str):
        return None
    return SPECIAL.get(name, {}).get(gap)


def explain_all(indicators) -> dict[tuple[str, str], str]:
    """一批指标 → `{(指标名, 窗口): 解释}`,只收有解释的那几条。

    给"逐条指标"那张表用(`report_frontend.l1_quality.build_quality_report` 的
    `rows`):那里一行一条指标,逐行取解释即可,不必自己判空。

    ⚠️ 键里带 `window` 是因为同一指标在不同窗口各有一行(逐题窗口那批),
    只按名字做键会让后一行盖掉前一行。
    """
    out: dict[tuple[str, str], str] = {}
    for ind in indicators or ():
        row = ind if isinstance(ind, dict) else (
            ind.to_dict() if hasattr(ind, "to_dict") else None)
        if row is None:
            continue
        text = explain(row)
        if text:
            out[(str(row.get("name", "")), str(row.get("window", "")))] = text
    return out
