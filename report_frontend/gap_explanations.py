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
| `response_latency_sec` | `window_none` | 没有可用回答窗口 | **两种病因**:25 场没走题库作答流程、6 场台账在但语音日志空 ⟹ 见 `response_latency_text()` |

第三行是 2026-11-21 改的(上一版对 31 场一律说"多数场次没存题目时刻表")。
实测 45 场之后:没台账的 27 场语音日志 27/27 全 0 行,而有台账的 18 场里
**8 场语音日志也是 0 行** —— 那 6 场报 `window_none` 的场次台账明明在盘上,
"没存题目时刻表"是当众说错话(评审一查 `questions.jsonl` 就露)。

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

#: 反应延迟报缺口时的**兜底句**:只在"这一层没拿到诊断事实"时用。
#:
#: 2026-11-21 改口的原因:原文案是「要的是每题提问结束那一刻,而多数场次没存题目时刻表;
#: 补上题目时刻就能算,与算力无关」。实测 45 场之后,这句话里**有两处不成立**:
#:
#: * 「没存题目时刻表」不是这条指标的病因 —— 读侧**只读语音日志里的应答时刻**
#:   (`l1_voice` 那段只取 `reaction_time` 一列),全仓没有任何报告路径按
#:   `questions.jsonl` 重算;补台账补不出一个"当时没落盘的应答时刻";
#: * 31 场里有 6 场**台账就在盘上**(`a971`/`0dc7`/`253a`/`4579`/`90e7`/`87c9`),
#:   对它们说"没存题目时刻表"是当众说错话。
#:
#: ⟹ 现在的分工:有诊断事实就按事实选句(`response_latency_text`),没有才用这句。
#: 这句本身必须**在任何一场都为真**,所以只说"只取这一列 + 没落盘就补不回来",
#: 不猜是哪一种没落盘。
_RESPONSE_LATENCY = ("反应延迟只取语音日志里每题的应答时刻,不按题目时刻重算;"
                     "这一场这个数没落盘,事后补不回来。")

#: 反应延迟的三种病因 —— 按**实测到的诊断事实**选,不按"多数场次"猜。
#:
#: 事实由产出方 `l1_voice._rt_gap_detail()` 带出(`gap_detail` 三个键):
#: `has_ledger`(录像目录里 `questions.jsonl` 在不在)、`voice_rows`(实际加载到的
#: 语音日志行数)、`rt_col_present`(日志里有没有应答时刻那一列)。
#:
#: | 情形 | 判据 | 说法 |
#: |---|---|---|
#: | 没走题库作答流程 | 无台账 且 语音 0 行 | `_RT_NO_FLOW` |
#: | 台账在、日志空 | 有台账 且 语音 0 行 | `_RT_LEDGER_BUT_EMPTY_LOG` |
#: | 有日志、没那一列 | 语音 > 0 行 | `_RT_VOICE_BUT_NO_COLUMN` |
_RT_NO_FLOW = ("这场既没有题目台账,也没有逐题语音作答记录:没走题库作答流程,"
               "反应延迟无从计算,要这个数得下次按流程录。")

_RT_LEDGER_BUT_EMPTY_LOG = ("台账在,但这一场的语音作答日志是空的:"
                            "应答时刻要采集端当场落盘,这一场补不回来,与算力无关。")

_RT_VOICE_BUT_NO_COLUMN = ("语音作答日志有记录,但没有存应答时刻那一列(或整列空着):"
                           "反应延迟无从计算,要它只能改采集端。")

SPECIAL: dict[str, dict[str, str]] = {
    "pause_ratio": {"not_produced": _PAUSE_RATIO},
    "micro_exp_rate_per_min": {"column_empty": _MICRO, "no_samples": _MICRO},
    "micro_exp_duration_mean_sec": {"column_empty": _MICRO, "no_samples": _MICRO},
    # `window_none`:没有应答时刻 ⟹ 句子按诊断事实选,这里只是**没事实时的兜底**。
    # `not_stored`:日志有行、但没有应答时刻那一列(见 `l1_voice` 的收尾那段)——
    #   这一条无论有没有事实都只有一种病因,所以直接写死在这一格。
    "response_latency_sec": {"window_none": _RESPONSE_LATENCY,
                             "not_stored": _RT_VOICE_BUT_NO_COLUMN},
}

#: 反应延迟这一条会产生缺口的码(其余码的病因与它无关,不许套用这组句子)。
_RT_GAPS = ("window_none", "not_stored")


def response_latency_text(indicator: dict) -> str | None:
    """反应延迟 → **按实测诊断事实**选一句;不是这一条、或事实不全 ⟹ `None`。

    ## 为什么这一条要单独一个入口

    其余特例按 `(指标名, 缺口码)` 就能定句:一个码只有一种病因。反应延迟不是 ——
    同一个 `window_none`,在 45 场里有两种完全不同的病因(见 `_RT_*` 三个常量),
    而**补救动作相反**:没走流程的那批要重录、日志空的那批要查采集端落盘。
    用一句"多数场次没存题目时刻表"盖住两批,对后一批就是错的。

    ## 判据只有三个,全部来自实测(不许猜)

    见 `l1_voice._rt_gap_detail()`:`has_ledger` / `voice_rows` / `rt_col_present`。

    ## 拿不到事实时返回 `None`

    调用方会退回 `SPECIAL[...][gap]` 那句(它必须在任何一场都为真)。宁可少说一句
    "为什么",也不许把 `None`(没证据)读成 `False`(实测没有)——
    `has_ledger is None` 与 `False` 在这里是两条不同的路。
    """
    if not isinstance(indicator, dict):
        return None
    if indicator.get("name") != "response_latency_sec":
        return None
    if indicator.get("gap") not in _RT_GAPS:
        return None
    detail = indicator.get("gap_detail")
    if not isinstance(detail, dict):
        return None
    try:
        rows = int(detail.get("voice_rows"))          # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if rows > 0:
        # 日志有行 ⟹ 缺的是"应答时刻那一列",与台账在不在无关。
        return _RT_VOICE_BUT_NO_COLUMN
    has_ledger = detail.get("has_ledger")
    if has_ledger is True:
        return _RT_LEDGER_BUT_EMPTY_LOG
    if has_ledger is False:
        return _RT_NO_FLOW
    return None


#: 四个特例指标的名字(供调用方/测试清点,不用自己再列一遍)。
SPECIAL_NAMES: tuple[str, ...] = tuple(SPECIAL)


def explain(indicator: dict) -> str | None:
    """一条指标 → 一句人话解释。**没有缺口就返回 `None`**。

    ## 入参只读这几个键(取不到就当没有,不抛错)

    * `name` —— 指标名(`L1Indicator.name`,如 `pause_ratio`);
    * `gap`  —— 缺口码(`L1Indicator.gap` 的 9 个值之一);
    * `gap_detail` —— (可选)报缺口时实测到的诊断事实,只有反应延迟用得上,
      见 `response_latency_text()`。不给就退回 `SPECIAL` 那一格。

    其它键(`value` / `n_valid` / `window` …)**一概不看**:这一层只解释缺口,
    不做判断、不改数。传 `L1Indicator.to_dict()` 出来的那一行正好够用
    (事实那一个键要调用方另塞,`to_dict()` 不带它 —— 它不进载荷)。

    ## 三种返回

    | 情形 | 返回 |
    |---|---|
    | `gap` 是 9 个已登记的码之一 | 一句 ≤60 字的中文解释(特例优先;反应延迟先看事实) |
    | `gap is None` / 空 / 不在 9 类里 | `None` —— **不兜底**,见模块 docstring |

    ## 为什么特例只认 `(名字, 码)` 这一对

    见 `SPECIAL` 的注释:同一个指标换一个码,病因就换了。硬套会把
    "这一场没答"说成"没存题目时刻",那正是本仓要消灭的"判词比事实更狠"。

    ## 反应延迟为什么还要看事实

    它是唯一一个**同一个码有两种病因**的特例:31 场 `window_none` 里 25 场是
    "没走题库作答流程"、6 场是"台账在但语音日志空"。前者重录有用,后者是采集端
    没落盘。用一句话盖住两批,对那 6 场就是错的(见 `response_latency_text`)。
    """
    if not isinstance(indicator, dict):
        return None
    gap = indicator.get("gap")
    if not isinstance(gap, str) or gap not in _KINDS:
        return None
    name = indicator.get("name")
    # 反应延迟要**先看事实再选句**(同一个码两种病因),其余特例按 (名字, 码) 就够。
    detail_text = response_latency_text(indicator)
    if detail_text:
        return detail_text
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
    # 同上:反应延迟先看诊断事实(`gap_detail`),没有才退回 `SPECIAL` 那一格。
    detail_text = response_latency_text(indicator)
    if detail_text:
        return detail_text
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
