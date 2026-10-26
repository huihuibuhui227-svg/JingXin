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
"""

from __future__ import annotations

import datetime as _dt

from . import l1_mapper, l1_quality
from .l1_quality import GAP_EXPLANATION, build_quality_report, explain_confidence
from .l1_session import compute_session_l1

#: 版本串。改了载荷结构就动它 —— 报告里要能看出是哪一代口径。
PAYLOAD_VERSION = "l1-1"

#: 逐维度固定话术。**不是**模板填空,是把"这一层不做什么"写死。
DIM_DESCRIPTION = "按 L1 指标直读;不做加权组合、不做推断。"
DIM_ALGORITHM = "逐指标实测,无复合分"

NO_COMPOSITE = ("未产出综合评分,也不给评级:本系统尚无标定样本,"
                "不同指标的量纲无法折算为同一尺度。")


def _evidence_item(slot: dict) -> dict:
    """一个槽 → 前端 `EvidenceItem`。字段名逐一对齐 `assessment.ts`。

    `observed_interval` 给空数组:L1 是**会话级**测量,一个指标本场只有一个值,
    没有"本场观测区间"可言。空数组是前端口径里"取不到区间"的写法,
    不是 0,也不是"区间是 [0,0]"。

    `human_name` 用**槽的中文名**,不用 L1 的 `formula` ——
    后者是给开发者核对算法的(两百多字,带 `⚠️` 和 markdown 粗体,还会点出
    另一个内部键名),它该待在 `l1.indicators` 的审计链里,不该占着「指标」这一列。
    钉子:`tests/test_report_gaps_are_readable.py::test_evidence_chain_names_are_not_raw_formulas`。
    """
    return {
        "feature": slot["l1"],
        "human_name": l1_mapper.slot_label(slot["slot"]),
        "raw_value": slot["value"],
        "n_valid": slot["n_valid"],
        "std": None,
        "observed_interval": [],
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
            out.append(f"{l1_mapper.slot_label(s['slot'])}:{_why(s['gap'])}")
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


def build_l1_report(session_id: str, logs: dict | None = None) -> dict:
    """跑一场 → `/api/report/structured` 的 `result`。"""
    result = compute_session_l1(session_id, logs=logs)
    dims, vanished = l1_mapper.map_l1_to_dimensions(result.indicators)
    quality = build_quality_report(result)

    dimensions: dict[str, dict] = {}
    passed_slots: list[dict] = []
    for key, dim in dims.items():
        dimensions[key] = {
            "display_name": dim.display_name,
            "description": DIM_DESCRIPTION,
            "algorithm": DIM_ALGORITHM,
            "evidence_chain": [_evidence_item(s) for s in dim.slots
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
            })

    # 分母用**声明的全部旧槽数**(14),不是"幸存维度里的槽数"(11)。
    # 理由:消失的维度那 3 个槽**存在过**,读者要能看出"14 个槽里 3 个没有对应量被删",
    # 而不是看到"11 个槽"以为这一维本来就只有这些。
    n_slots = sum(d.n_slots for d in dims.values()) + sum(v["n_slots"] for v in vanished)
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
            "n_slots": n_slots,
            "n_passed": len(passed_slots),
            # 没有同构念 L1 指标、已删掉的槽数(它们仍计入 n_slots)。
            "n_dropped": sum(len(d.dropped) for d in dims.values())
            + sum(v["n_slots"] for v in vanished),
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
        },
    }
