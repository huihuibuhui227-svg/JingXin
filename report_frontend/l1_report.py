# report_frontend/l1_report.py
"""**把 L1 指标装成报告载荷** —— `/api/report/structured` 的新引擎。

## 为什么需要这一层

`l1_mapper` 产出的是 `DimensionResult`(带槽、权重、gap),而前端
`src/types/assessment.ts` 的 `EvaluationResult` 要求的是另一套字段名
(`evidence_chain[].feature` / `.raw_value` / `.observed_interval`…)。
本模块只做**形状转换**,不新增判断 —— 判断都在 `l1_mapper` 与 `l1_quality` 里。

## 两处必须换掉的**假陈述**(M4.5 的验收判据就是这句)

旧载荷每个维度都带 `algorithm: "加权线性组合 + 认知负荷推断"`。在 L1 口径下
这句话**不成立**:这一层没有加权线性组合(权重只用于"删槽后重新归一"的记账,
不参与任何求和),也没有认知负荷推断(本仓不做推断,spec §5.4)。

⟹ 本模块一律写:
* `description` = "按 L1 指标直读;不做加权组合、不做推断。"
* `algorithm`   = "逐指标实测,无复合分"

## 消失的维度要**说出来**

`行为表现活跃度` 的三个槽(手掌张开度 / 肩部放松度 / 眼神接触)没有同构念的 L1 指标
⟹ 整个维度不进 `dimensions`。但"不进"不等于"读者不该知道" —— 它进
`vanished_dimensions` 并带理由。前端 `ReportViewer` 对缺失维度是
`if (!dim) return null`,所以少一个键是安全的。

## 无复合分

`summary_narrative` 沿用旧口径那句话;`stats` 只放**实测值**,不放百分位
(没有人群参照就没有位置,spec §5.5)。

## A5 仪器来源披露(2026-11-21)

本模块**新增**三处字段,原有键一个不动(见 `_instrument_block` 与
`_evidence_item` 的注释):

| 位置 | 新字段 | 谁读 |
|---|---|---|
| `l1.indicators[]` | `instrument` **或** `instruments`(二选一,来自 `annotate_indicators`) | 复核 / 前端逐条标签 |
| `dimensions.*.evidence_chain[]`、`coverage.passed_slots[]` | `instruments`(**永远是 list**) | 报告表格那一列 |
| `l1.instruments` | 汇总块(`counts` / `footer` / `labels` / `order` / 未登记缺口) | HTML 页脚 + 前端汇总区 |

标注本身发生在**上游装配点** `l1_quality.build_quality_report`(38 条指标刚定型处),
本模块只查表 —— 不在两处各算一份口径。逐条有出处这件事从此可以核对:
`tests/test_report_instrument_rendering.py`。

## 覆盖数的三段拆分(2026-11-21)

`coverage` 原先只发 `n_passed / n_slots`。实测那一场是 `8 / 14` —— 而 14 里有
**6 个槽永远填不上**:3 个「这套采集不测」(`dimensions.*.evidence_gaps_not_measured`)、
3 个「这一维没有同构念 L1 指标」(整维进了 `vanished_dimensions`)。于是 `8/14`
被读者读成"只覆盖了 57%",而可测量的那 8 个槽是 8/8 全过。

⟹ 新增分段计数,原有键(`n_slots` / `n_passed` / `n_dropped` / `passed_slots` /
`confidence_cap`)**一个语义都没动**:

| 新字段 | 含义 | 谁读 |
|---|---|---|
| `n_measurable` | 有同构念指标、且这套采集确实会测的槽数 | HTML / 前端 / demo_run |
| `n_measurable_no_data` | 上面那批里本场没量到可用数的槽数 | 同上(说明"可测 ≠ 已通过") |
| `n_not_measured` | 维度表内被删的槽(这套采集不测) | 同上 |
| `n_no_indicator` | 整维被删的槽(无同构念指标) | 同上 |
| `pass_rate_of_measurable` | `n_passed / n_measurable`;分母为 0 时是 `None` | 同上 |
| `summary_line` / `explanation` / `segments` | 三段式那句话 + 口径解释 + 三段明细 | 三个出口都读载荷,不各自拼措辞 |

推导方式:三段计数是把 `n_slots` 原本那两个 `sum` **提出来复用**,不是另数一遍 ——
`n_measurable + n_not_measured + n_no_indicator == n_slots` 恒成立
(因为 `DimensionResult.n_slots` 就是 `len(slots) + len(dropped)`)。
钉子:`tests/test_coverage_three_segments.py`。
"""

from __future__ import annotations

import datetime as _dt

from . import gap_explanations, instrument_provenance, l1_mapper, l1_quality
from .l1_quality import GAP_EXPLANATION, build_quality_report, explain_confidence
from .l1_session import compute_session_l1

#: 版本串。改了载荷结构就动它 —— 报告里要能看出是哪一代口径。
#: `l1-2`(2026-11-21):`coverage` 增三段拆分 + `summary_line` / `explanation` / `segments`。
#: 前端 store 里可能还躺着 `l1-1` 的旧结果(没有这些键)⟹ 三个出口都必须容忍缺席。
PAYLOAD_VERSION = "l1-2"

#: 逐维度固定话术。**不是**模板填空,是把"这一层不做什么"写死。
DIM_DESCRIPTION = "按 L1 指标直读;不做加权组合、不做推断。"
DIM_ALGORITHM = "逐指标实测,无复合分"

NO_COMPOSITE = ("未产出综合评分,也不给评级:本系统尚无标定样本,"
                "不同指标的量纲无法折算为同一尺度。")


# ── A5 仪器来源披露(2026-11-21)──────────────────────────────────────────────

#: 仪器名 → 中文标签。**给「逐条指标」那一列用**,页脚那句汇总仍由
#: `instrument_provenance.format_footer()` 拼(A5 的规矩:那句话只许有一处拼法)。
#:
#: ⚠️ 这张表是 `format_footer()` 里那张**函数内局部表**的第二份。为什么不直接用它:
#: 它是局部变量,本模块读不到;`instrument_provenance.py` 不归本次改动,不在这里动它。
#: 两份措辞不一致的后果可预见:页脚写「几何代理」、指标旁边写「几何」,读者以为是
#: 两样东西。⟹ 一致性由
#: `tests/test_report_instrument_rendering.py::test_labels_agree_with_the_footer`
#: 逐项对账(拿 `format_footer()` 的真实输出比),谁改了一处而漏了另一处当场红。
INSTRUMENT_LABEL: dict[str, str] = {
    instrument_provenance.INSTRUMENT_BLENDSHAPE: "模型系数",
    instrument_provenance.INSTRUMENT_GEOMETRIC: "几何代理",
    instrument_provenance.INSTRUMENT_AUDIO: "音频",
    instrument_provenance.INSTRUMENT_TEXT: "文本",
    # ⚠️ 「未登记」**不是一种仪器**,是一条披露缺口。措辞里不许出现"其他""未知"
    #   这类像分类的名字 —— 那会让读者把它当成第三种仪器读过去。
    instrument_provenance.INSTRUMENT_UNKNOWN: "未登记",
}

#: 出现「未登记」时那句话。由载荷给,HTML 页脚与前端汇总区读同一份
#: (渲染层各写一句,两处措辞早晚不一致)。
INSTRUMENT_UNKNOWN_NOTE = (
    "「未登记」不是一种仪器,是一条披露缺口:这几条指标的依赖列还没归到任何一路"
    "仪器上。数是算出来了,但它出自哪套仪器暂时说不清 —— 要补的是登记表,不是采集。")

#: 多仪器指标的说明。`describe()` 会在每一个涉及的仪器下各计 1 ⟹ 各项相加会大于条数。
INSTRUMENT_MULTI_NOTE = (
    "有的指标同时来自两路(例如语音字数率 = 转写字数 ÷ 有声时长,一头是文本、"
    "一头是音频),它在两个仪器下各计一次 ⟹ 各项相加会大于指标条数。")


def _instrument_block(indicators: list[dict]) -> dict:
    """38 条指标 → 仪器来源汇总块(HTML 页脚与前端汇总区共用的那一份)。

    返回值**全部是算好的结果**,不是让渲染层再算一遍的材料:`counts` / `footer`
    都来自 `instrument_provenance`(它是唯一登记处),本模块只负责把它们搬进载荷。
    渲染层自己数一遍的后果本仓记过多次:两处口径一分叉,同一份报告里两个数。

    `unmapped_columns` 是**能据以动手的东西** —— 只说"8 条未登记"读者没法改;
    说"缺的是哪几列"下一个人就能去登记表里补那几行。

    ⚠️ 实测记录(2026-11-21):`demo_run.py --session 20260927_212143_95a8` 一开始真有
    8 条落进 `unknown`,缺的列是 `f0` / `rms` / `frame_sec` —— 它们是**整场录音重算**
    那条路(`voice_series`)的列名;而日志那条路上同样这几条指标写的是 `pitch` /
    `energy`(登记表里登记的是后一种)。⟹ 同一条指标换条数据来路,披露就从「音频」
    变成「未登记」。**当天已修**:`instrument_provenance` 把 `f0`/`rms` 登记为音频、
    `frame_sec` 登记为"不是仪器"(它是帧长/栅格步长),`unmapped_columns` 随之为空,
    这一场报告的 `未登记 0`。留给后人的那半句仍然成立:登记表在
    `instrument_provenance.py`,新增列名要在**那张表**里登记,否则又会冒出假的"未登记"。
    """
    counts = instrument_provenance.describe(indicators)
    unmapped = sorted({
        c for row in indicators
        for c in instrument_provenance.unmapped_columns(row.get("required_l0"))
    })
    n_multi = sum(1 for row in indicators
                  if len(instrument_provenance.instruments_of(row)) > 1)
    return {
        "n_indicators": len(indicators),
        "counts": counts,
        "footer": instrument_provenance.format_footer(indicators),
        "labels": dict(INSTRUMENT_LABEL),
        #: 页面上的固定顺序(= `instrument_provenance.INSTRUMENTS`)。
        #: `counts` 是 dict,前端按键遍历会得 JS 的整数键优先序 —— 界面顺序会随输入
        #: 抖动。给一份列表,两边都按它排。
        "order": list(instrument_provenance.INSTRUMENTS),
        "n_multi": n_multi,
        "unmapped_columns": unmapped,
        "unknown_note": INSTRUMENT_UNKNOWN_NOTE,
        "multi_note": INSTRUMENT_MULTI_NOTE,
    }


def _evidence_item(slot: dict, instruments: list[str]) -> dict:
    """一个槽 → 前端 `EvidenceItem`。字段名逐一对齐 `assessment.ts`。

    `observed_interval` 给空数组:L1 是**会话级**测量,一个指标本场只有一个值,
    没有"本场观测区间"可言。空数组是前端口径里"取不到区间"的写法,
    不是 0,也不是"区间是 [0,0]"。

    `human_name` 用**槽的中文名**,不用 L1 的 `formula` ——
    后者是给开发者核对算法的(两百多字,带 `⚠️` 和 markdown 粗体,还会点出
    另一个内部键名),它该待在 `l1.indicators` 的审计链里,不该占着「指标」这一列。
    钉子:`tests/test_report_gaps_are_readable.py::test_evidence_chain_names_are_not_raw_formulas`。

    ★ A5(2026-11-21):新增 `instruments`。**永远是 list**(单仪器也是单元素列表)——
    渲染层因此不必判断"该读 `instrument` 还是 `instruments`"(那种判断在本仓出现过:
    两处各判断一次,一次读错键就把多仪器读成了单仪器)。
    """
    return {
        "feature": slot["l1"],
        "human_name": l1_mapper.slot_label(slot["slot"]),
        "raw_value": slot["value"],
        "n_valid": slot["n_valid"],
        "std": None,
        "observed_interval": [],
        # 这条数由哪一路仪器产出。取不到给 `["unknown"]` —— 那是披露缺口,
        # **不是**"没有这条指标"(见 `instrument_provenance` 模块 docstring 规矩 2)。
        "instruments": list(instruments),
    }


def _gap_lines(dim) -> list[str]:
    """维度 → 逐行缺口说明(字符串列表,前端按 `<li>` 渲染)。

    ## 印人话,不印内部记号

    2026-10-25 实测产出过这样的行:

        connective_density → discourse_marker_rate:not_stored
        symmetry_score:删槽 —— L1 没有面部对称性指标

    里面三样东西读者都不认识:内部槽键、缺口码常量、`→` 这个迁移记号。
    它们只在**核对口径**时有用,不该进交付物。

    * 槽名走 `l1_mapper.slot_label`(与旧引擎报告同名,免得读者以为是两批指标);
    * 缺口码走 `l1_quality.GAP_EXPLANATION` —— **本函数不另抄一份**。
      `l1_quality` 的模块注释原话就是「不许直接渲染 `gap` 的英文常量」,
      抄第二份的后果可预见:新增一个码时改一处漏一处,界面上冒出英文单词。
    * 删槽行必须点明「**本层不测**」:它与「测了这次没数据」是两件事,
      后者补采/补缓存就有,前者再采一百次也不会有。混成一句会让人去追
      一个不存在的数据。
      钉子:`tests/test_report_gaps_are_readable.py`。

    ## ⚠️ 2026-10-26:这两类**不能塞进同一个列表**

    实测那一版报告里,「情境行为稳定性」下面印着:

        这次没量到可用数的指标：
          · 面部对称性：这一条当前这套采集不测 —— 这套采集没有测面部左右对不对称

    小标题说「**这次**没量到」(⟹ 下一场可能就有),下面那条说「**采集**不测」
    (⟹ 再采一百次也没有)。同一段里两句相反的话,读者只能信一句 ——
    而他多半会信小标题,然后去追一个不存在的数据。

    载荷本来就分得清(`dim.dropped` 是删槽,`dim.slots` 里 `value is None`
    是本场没量到),混在一起是**渲染层丢掉了这个区分**。
    ⟹ 现在两类分开给:`evidence_gaps`(本场没量到)与
    `evidence_gaps_not_measured`(整套采集不测)。字段名保留 `evidence_gaps`
    以免动前端契约;新增的第二个字段前端可以选用,不选也仍然是真话。
    """
    def _why(code: str | None) -> str:
        # 未登记的码原样印出 —— 宁可露出英文,也不许把它当成"已测"或吞掉。
        return GAP_EXPLANATION.get(code or "", code or "未说明原因")

    out: list[str] = []
    for s in dim.slots:
        if s["value"] is None:
            why = gap_explanations.explain_specific(
                {"name": s["l1"], "gap": s["gap"],
                 "gap_detail": s.get("gap_detail")}) or _why(s["gap"])
            out.append(f"{l1_mapper.slot_label(s['slot'])}:{why}")
    return out


def _not_measured_lines(dim) -> list[str]:
    """维度 → 被删槽的说明(整套采集不测,**不是**这一场没量到)。

    与 `_gap_lines` 分开的理由见那边 docstring 的「这两类不能塞进同一个列表」。
    """
    return [f"{l1_mapper.slot_label(d['slot'])}:这一条当前这套采集不测 —— "
            f"{l1_mapper.drop_reason(d['slot'])}"
            for d in dim.dropped]


def _printable_vanished(v: dict) -> dict:
    """消失的维度 → 前端**能直接印**的形状。

    逐槽补 `human_name`(中文槽名)与 `plain_reason`(一句话结论),
    审计字段 `slot` / `reason` 原样保留 —— 分层,不是删掉。
    见 `build_l1_report` 里 `vanished_dimensions` 那段说明。
    """
    return {
        **v,
        "dropped": [
            {**d,
             "human_name": l1_mapper.slot_label(d["slot"]),
             "plain_reason": l1_mapper.drop_reason(d["slot"])}
            for d in v.get("dropped", [])
        ],
    }


#: 三段式的固定顺序。前端不按键遍历、不自己排 —— 顺序在这里定一次。
COVERAGE_SEGMENT_KEYS = ("measurable", "not_measured", "no_indicator")

#: 三段式那句解释,**由载荷发给三个出口**(HTML 报告 / 前端报告页 / 命令行)。
#:
#: 为什么写在载荷里而不是各出口各写一份:同一件事在三个地方各写一遍,迟早
#: 三处说法不同(本仓记过多次的形态)。出口只负责排版,不负责措辞。
#:
#: 措辞纪律:
#: ① **只陈述口径,不评价这一场好不好** —— 缺口照写,不加"仍然"、"难能可贵";
#: ② **不用内部叫法**。`tests/test_report_says_something_to_a_person.py` 的
#:    `test_the_body_carries_no_internal_jargon` 明令交付物正文不许出现
#:    「指标槽 / 证据门 / 过门 / 同构念 / 本层 / 当前管线」——这句话要进 HTML 正文,
#:    所以这里一律说「槽 / 这套采集不测 / 整维没有对应指标」。
#:    内部叫什么不丢:`n_no_indicator` 这个键名、以及 `l1_mapper` 里的映射依据都还在。
COVERAGE_EXPLANATION = (
    "「这套采集不测」说的是这套采集不含这个量 —— 不是这一场没量到,再采一次也一样;"
    "「整维没有对应指标」说的是那一整维要的量当前没有任何一条指标承接,"
    "不是这一场量得不好。三段合起来才是槽总数;通过率只用第一段做分母,不含后两段。"
)


def coverage_summary_line(n_measurable: int, n_passed: int,
                          n_not_measured: int, n_no_indicator: int) -> str:
    """三段式那一行(`可测量 8 槽全部通过(8/8) · 这套采集不测 3 槽 · 整维没有对应指标 3 槽`)。

    第一段只在**全过**时才说「全部通过」;可测但本场没量到数的槽会留在分母里,
    于是那句自动变成「可测量 11 槽中 8 槽通过(8/11)」—— 分母为 0 时既不说
    「0%」也不说「100%」,只说「通过率不适用」。

    ⚠️ 行首不加「证据门:」——那是内部叫法,交付物正文不许出现(见上面那条纪律)。
    命令行那一行有自己的前缀(「覆盖」),HTML 有它的版式,都不需要这个词。

    钉子:`tests/test_coverage_three_segments.py`。
    """
    if n_measurable <= 0:
        first = "可测量 0 槽(分母为 0,通过率不适用)"
    elif n_passed >= n_measurable:
        first = f"可测量 {n_measurable} 槽全部通过({n_passed}/{n_measurable})"
    else:
        first = f"可测量 {n_measurable} 槽中 {n_passed} 槽通过({n_passed}/{n_measurable})"
    return (f"{first} · 这套采集不测 {n_not_measured} 槽"
            f" · 整维没有对应指标 {n_no_indicator} 槽")


def coverage_segments(n_measurable: int, n_passed: int, n_measurable_no_data: int,
                      n_not_measured: int, n_no_indicator: int) -> list[dict]:
    """三段拆分(**数字全部由调用方从现有结构派生后传进来**,本函数不数任何东西)。

    为什么不在这里数:`build_l1_report` 里 `n_slots` / `n_dropped` 已经有权威表达式,
    再从这里数一遍就是"两处各算一遍" —— 本仓为这件事栽过一次(`n_passed` 虚报成 7)。
    本函数只负责**把同一批数排成人能读的三段**,键名固定、顺序固定。

    每段给 `n_passed` 的规则:只有第一段有"通过"这回事 —— 后两段压根没有可过的门,
    给 0 会被读成"0 个通过"(那听起来像"没做到")。所以给 `None`,前端据此不画分子。
    """
    measurable_note = "有对应指标,这套采集也确实会测"
    if n_measurable_no_data > 0:
        measurable_note += f";其中 {n_measurable_no_data} 槽本场没有可用数"
    by_key = {
        "measurable": {"key": "measurable", "label": "可测量", "n": n_measurable,
                       "n_passed": n_passed, "note": measurable_note},
        "not_measured": {"key": "not_measured", "label": "这套采集不测",
                         "n": n_not_measured, "n_passed": None,
                         "note": "不是这一场没量到 —— 这套采集根本不含这个量"},
        "no_indicator": {"key": "no_indicator", "label": "整维没有对应指标",
                         "n": n_no_indicator, "n_passed": None,
                         "note": "这一整维要的量,当前没有任何一条指标承接"},
    }
    # 顺序由 `COVERAGE_SEGMENT_KEYS` 定,不靠字典的插入序 —— 前端按数组渲染,
    # 顺序变了读者就得重新认一遍"第一段是什么"。
    return [by_key[k] for k in COVERAGE_SEGMENT_KEYS]


def coverage_counts(dims: dict, vanished: list, n_passed: int) -> dict:
    """`coverage` 里的全部计数 —— **只在这里算一次**,三个出口都读这份。

    ## 推导(不另算一份口径)

    `n_slots` / `n_dropped` 原来的两个 `sum` 就是这三个分段数的来源,
    这里只是把它们**取名**(而不是新增统计):

        n_in_dimensions   = Σ dim.n_slots        # `DimensionResult.n_slots`
                                                 #   ≡ len(slots) + len(dropped)
        n_not_measured    = Σ len(dim.dropped)   # 维度表内被删的槽(这套采集不测)
        n_no_indicator    = Σ v["n_slots"]       # 整维消失的槽(无同构念指标)

        n_slots           = n_in_dimensions + n_no_indicator          # 14,语义未变
        n_dropped         = n_not_measured  + n_no_indicator          # 6,语义未变
        n_measurable      = n_in_dimensions - n_not_measured          # 8

    ⟹ `n_measurable + n_not_measured + n_no_indicator == n_slots` **恒成立**,
    这不是一条需要维护的不变式,是同一个加法换了个顺序。

    ## 为什么单独成函数

    "全都不测"(可测量槽 = 0)在真实场次里不出现,而它的行为恰恰是这次要守的
    (`pass_rate_of_measurable` 必须给 `None`,不许给 0 或 1)。留在 `build_l1_report`
    里就只能先跑一整场 L1 才能测到 ⟹ 抽出来,`tests/test_coverage_three_segments.py`
    能直接喂一个人造的 `dims={} / vanished=[14 个槽]`。

    `n_measurable_no_data` 单列的理由:可测 ≠ 已通过。可测的槽也可能这一场
    没量到数(值为 `None`),并进 `n_passed` 就是虚报;藏起来就是假装没有这个缺口。
    """
    n_in_dimensions = sum(d.n_slots for d in dims.values())
    n_not_measured = sum(len(d.dropped) for d in dims.values())
    n_no_indicator = sum(v["n_slots"] for v in vanished)
    n_measurable = n_in_dimensions - n_not_measured
    # 分母为 0 给 `None`(前端类型是 `number | null`):0 会被读成"覆盖率 0%",
    # 1.0 会被读成"全部通过",两个都是假话。
    pass_rate = (n_passed / n_measurable) if n_measurable else None
    return {
        "n_slots": n_in_dimensions + n_no_indicator,
        "n_passed": n_passed,
        "n_dropped": n_not_measured + n_no_indicator,
        "n_measurable": n_measurable,
        "n_measurable_no_data": n_measurable - n_passed,
        "n_not_measured": n_not_measured,
        "n_no_indicator": n_no_indicator,
        "pass_rate_of_measurable": pass_rate,
    }


def build_l1_report(session_id: str, logs: dict | None = None) -> dict:
    """跑一场 → `/api/report/structured` 的 `result`。"""
    result = compute_session_l1(session_id, logs=logs)
    dims, vanished = l1_mapper.map_l1_to_dimensions(result.indicators)
    quality = build_quality_report(result)

    # ★ A5 仪器来源披露:指标名 → 仪器的索引。
    #
    # 标注本身发生在**上游装配点**(`l1_quality.build_quality_report`,那里 38 条指标
    # 刚定型),这里只是查表 —— **不重算一份口径**。重算的话,`speech_rate` 这一条
    # 可能在两处得到不同的仪器组合,而它正是本仓"多仪器"的样板。
    #
    # `quality.indicators` 是审计链那份(38 条,一条一行);同名指标在逐题窗口下
    # 会重复出现,但那是 `dispersion` 的事,本索引取会话级那一行。
    instruments_by_indicator: dict[str, list[str]] = {}
    for row in quality.indicators:
        instruments_by_indicator.setdefault(
            row["name"], instrument_provenance.instruments_of(row))

    def _of(indicator_name: str) -> list[str]:
        # 查不到就记 `unknown` —— 宁可把缺口露出来,也不猜一个"看起来像"的仪器。
        return instruments_by_indicator.get(
            indicator_name, [instrument_provenance.INSTRUMENT_UNKNOWN])

    dimensions: dict[str, dict] = {}
    passed_slots: list[dict] = []
    for key, dim in dims.items():
        dimensions[key] = {
            "display_name": dim.display_name,
            "description": DIM_DESCRIPTION,
            "algorithm": DIM_ALGORITHM,
            "evidence_chain": [_evidence_item(s, _of(s["l1"])) for s in dim.slots
                               if s["value"] is not None],
            "evidence_gaps": _gap_lines(dim),
            # 整套采集都不测的槽 —— **不是**这一场没量到。分开给的理由见
            # `_gap_lines` docstring 里「这两类不能塞进同一个列表」。
            "evidence_gaps_not_measured": _not_measured_lines(dim),
            "confidence": dim.confidence,
            "matched_indicators": dim.matched_indicators,
            # `stats` 只放实测值。**没有 percentile** —— 见模块 docstring。
            "stats": {s["slot"]: {"val": s["value"]} for s in dim.slots
                      if s["value"] is not None},
        }
        for s in dim.slots:
            if s["value"] is None:
                continue
            passed_slots.append({
                "dimension": key,
                "display_name": dim.display_name,
                # 「指标」列给人看的是**中文槽名**,不是内部键 ——
                # 缺口行已经说人话了,这里再露 `connective_density`,
                # 读者会把同一个指标当成两个东西。内部键在 `feature` 里留着。
                "indicator": l1_mapper.slot_label(s["slot"]),
                "feature": s["l1"],
                "raw_value": s["value"],
                "observed_interval": [],
                "n_valid": s["n_valid"],
                "confidence": s["confidence"],
                # A5:与 `evidence_chain` 同一份形状(永远是 list)。前端两处都渲染,
                # 让它们读同一个键比让前端各判断一次稳。
                "instruments": _of(s["l1"]),
            })

    # 分母用**声明的全部旧槽数**(14),不是"幸存维度里的槽数"(11)。
    # 理由:消失的维度那 3 个槽**存在过**,读者要能看出"14 个槽里 3 个没有对应量被删",
    # 而不是看到"11 个槽"以为这一维本来就只有这些。
    #
    # ⚠️ 2026-11-21:计数整段搬进 `coverage_counts()`,这里只取结果 ——
    #    **值一个没变**(`n_slots` 仍 14、`n_dropped` 仍 6、`n_passed` 仍过门槽数),
    #    只是顺手把"可测量 / 这套采集不测 / 整维没有对应指标"三段一起算出来。
    #    计数只有那一处,渲染层与前端都不许再算一遍(本仓为"两处各算一遍"栽过)。
    counts = coverage_counts(dims, vanished, len(passed_slots))
    # `confidence_cap` 按前端类型的注释是「各维度置信度的**上限**(最乐观的那一个)」
    # ⟹ `CONF_RANK` 里高=3,取 **max**。旧代码用 `max` 配的是反序表(高=3 的那张),
    # 含义一致;我先前误用了从 `CONFIDENCE` 元组派生的反序表,差点把它算成最悲观。
    cap = "无"
    if passed_slots:
        cap = max((p["confidence"] for p in passed_slots),
                  key=lambda c: l1_mapper.CONF_RANK.get(c, 0))

    return {
        "engine": "l1",
        "dimensions": dimensions,
        # 消失的维度必须能看见 —— 否则读者会以为"这一场没测到",而真相是
        # "这个维度量的东西当前管线都不测"。
        #
        # ⚠️ 逐槽补 `human_name` / `plain_reason`(2026-10-25):前端 `ReportViewer`
        # 直接印 `d.slot` 与 `d.reason`,而那是**审计细节** —— 内部键 + 内部指标名
        # + markdown 星号。前端没有、也不该有槽名对照表(两张表迟早分叉),
        # ⟹ 人话只能由载荷给。审计字段 `slot` / `reason` **原样保留**。
        # 钉子:`tests/test_vanished_dimensions_are_printable.py`。
        "vanished_dimensions": [_printable_vanished(v) for v in vanished],
        "summary_narrative": NO_COMPOSITE,
        "coverage": {
            # ── 原有键(语义一个没动)─────────────────────────────────────
            "n_slots": counts["n_slots"],
            "n_passed": counts["n_passed"],
            "n_dropped": counts["n_dropped"],
            # ── 三段拆分(2026-11-21 新增)────────────────────────────────
            # 为什么:14 里有 6 个槽**永远填不上**(3 个本套采集不测 + 3 个整维没有
            # 同构念指标)。只发 `n_passed / n_slots` 会让读者把 8/14 读成"只覆盖了
            # 57%",而能测的 8 个槽这一场是 8/8。计数只在 `coverage_counts()` 里算一次。
            # 钉子:`tests/test_coverage_three_segments.py`。
            "n_measurable": counts["n_measurable"],
            "n_measurable_no_data": counts["n_measurable_no_data"],
            "n_not_measured": counts["n_not_measured"],
            "n_no_indicator": counts["n_no_indicator"],
            "pass_rate_of_measurable": counts["pass_rate_of_measurable"],
            # 三个出口(HTML / 前端 / demo_run)都读这三项,不各自拼措辞。
            "summary_line": coverage_summary_line(
                counts["n_measurable"], counts["n_passed"],
                counts["n_not_measured"], counts["n_no_indicator"]),
            "explanation": COVERAGE_EXPLANATION,
            "segments": coverage_segments(
                counts["n_measurable"], counts["n_passed"],
                counts["n_measurable_no_data"],
                counts["n_not_measured"], counts["n_no_indicator"]),
            "confidence_cap": cap,
            "passed_slots": passed_slots,
        },
        "model_metadata": {
            "version": PAYLOAD_VERSION,
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        },
        # L1 的完整审计链:35 条逐条状态 + 模态覆盖率 + gap 计数。
        # 报告层不渲染它,但**必须发** —— 复核时要能一条一条对。
        "l1": {
            "n_total": quality.n_total,
            "n_measured": quality.n_measured,
            "by_confidence": dict(quality.by_confidence),
            "by_gap": dict(quality.by_gap),
            "modality_coverage": quality.coverage,
            # ★ 每个模态**到底算出了几条**(2026-10-25)。报告头看的是日志文件的三态,
            #   而语音有第二个数据源(整场录音重算),不经日志 —— 于是头说「未读到数据」、
            #   正文说「V 族 10 条有值」。这句话由载荷给,渲染层不许自己数。
            "modality_summary": l1_quality.modality_summary(quality),
            "timebase": quality.timebase,
            "windows": quality.windows,
            "indicators": quality.indicators,
            # ★ 2026-10-18:`explain_confidence()` 此前**没有任何调用点** ——
            #   `LOW_CONFIDENCE_REASON` 那张表写好了却从没进过报告。
            #   后果是报告上只写「置信 低」,不写"为什么只有这么信";
            #   而后者才是读者要的那句话(置信度本身不回答它)。
            "low_confidence_notes": explain_confidence(quality),
            # ★ 同一批理由的**给人读版**(2026-10-25):审计版把 `{指标名}:{理由}`
            #   原样拼出来,而指标名是内部键 —— 实测交付物正文里印出了
            #   `speech_rate:` / `pause_rate_per_min:` 等 5 个键,而维度表里
            #   同一个指标印的是中文槽名。**两个版本都发**,分层的决定权在载荷。
            "low_confidence_notes_plain": explain_confidence(quality, plain=True),
            # ★ 与旧引擎报告对比时的那句提醒。见 `l1_mapper.NOT_COMPARABLE`。
            #   **两个版本都发**:`comparability_notes` 是审计版(带槽键与 L1 指标名,
            #   核对口径的人要查);`comparability_notes_plain` 是给人读版(交付物与
            #   界面都印这一版)。只发审计版 ⟹ 两个出口都会漏内部指标名;
            #   只发人话版 ⟹ 映射查不到。分层的决定权在载荷,不在渲染层。
            "comparability_notes": l1_mapper.comparability_notes(),
            "comparability_notes_plain": l1_mapper.comparability_notes_plain(),
            "gaps": list(result.gaps),
            # ★ 裁定①(2026-10-24):会话级 38 条的口径是**回答帧并集**,
            #   逐题那份只作**离散度**层,不冒充会话级值。见 `l1_session.dispersion_of`。
            "dispersion": dict(getattr(result, "dispersion", None) or {}),
            "dispersion_note": (
                "逐题值的题间离散度(中位/IQR/极值),**只在本场内有意义** —— "
                "它不能与别的场次比,也不是会话级值;会话级那 38 条见 indicators"
                "(口径:所有回答窗内帧的并集)。单题时不出现在这里(没有离散度可言)。"
            ),
            # ★ A5 仪器来源披露(2026-11-21):**汇总由载荷给**,渲染层不许自己数。
            #
            # 为什么必须进载荷:HTML 页脚与前端汇总区是同一份载荷的两个出口。
            # 各写一遍"数一下有几条 geometric",两处迟早不同 —— 而这份报告对外要
            # 说的那句话正是「报告已披露仪器来源」,它经不起两个页脚各说一套。
            #
            # 逐条的仪器在 `indicators[].instrument` / `[].instruments`(单值/多值
            # 二选一,由 `annotate_indicators` 决定)与逐行的 `evidence_chain[].instruments`
            # / `passed_slots[].instruments`(永远 list —— 渲染层只读这一个键)。
            "instruments": _instrument_block(quality.indicators),
        },
    }
