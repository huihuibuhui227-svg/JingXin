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

`行为表现活跃度` 的三个槽(手掌张开度 / 耸肩幅度 / 眼神接触)没有同构念的 L1 指标
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
import json
from pathlib import Path

from . import (
    behavior_regime_model,
    blendshape_calibration,
    brand,
    cross_modal_coverage,
    evidence_gate,
    face_dual_instrument_audit,
    feature_layer_v2,
    gap_explanations,
    instrument_provenance,
    l1_mapper,
    l1_quality,
    parameter_provenance,
    referee_ml_engine,
    signal_quality_clf,
    speaker_turn_audit,
    transcript_chars,
)
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

#: 仪器名 → 中文标签（直接引用 `instrument_provenance.INSTRUMENT_LABEL` 唯一真相源）
INSTRUMENT_LABEL: dict[str, str] = dict(instrument_provenance.INSTRUMENT_LABEL)

#: 出现「未登记」时那句话。由载荷给,HTML 页脚与前端汇总区读同一份
INSTRUMENT_UNKNOWN_NOTE = (
    "「未登记」不是一种仪器,是一条披露缺口:这几条指标的依赖列还没归到任何一路"
    "仪器上。数是算出来了,但它出自哪套仪器暂时说不清 —— 要补的是登记表,不是采集。")

#: 多仪器指标的说明（正文禁用 `⟹` 推理符号）。
INSTRUMENT_MULTI_NOTE = (
    "有的指标同时来自两路(例如语音字数率 = 转写字数 ÷ 有声时长,一头是文本、"
    "一头是音频),它在两个仪器下各计一次，因此各项相加会大于指标条数。")

NON_INTERCHANGEABILITY_DISCLOSURE: dict = {
    "study_sessions": 32,
    "study_rows": 389,
    "constructs_total": 12,
    "constructs_rho_ge_0_7": 1,
    "conclusion_plain": (
        "离线实测（32 场 / 389 条）表明，面部关键点几何代理与 52 维模型回归系数"
        "在 12 个构念中仅有 1 个达到秩一致性 0.7 以上，因此两套测量结果不可混用；"
        "本报告对每一条指标明确标明仪器来源，只做披露、不改数值。"
    ),
}

_ASR_AFFECTED_INDICATORS: tuple[str, ...] = (
    "lexical_diversity",
    "discourse_marker_rate",
    "self_repair_rate",
    "answer_detail_chars_p50",
    "speech_rate",
)

_L0_PATH = Path(__file__).resolve().parent.parent / "l0_columns.json"


def _strip_dev_markup(text: str) -> str:
    """清洗开发注释里的 markdown / 箭头记号，确保进入交付物附录时人话可读。"""
    import re as _re
    s = _re.sub(r"\*\*|`", "", str(text or ""))
    s = s.replace("⟹", "，因此").replace("⇒", "，因此").replace("→", "变为")
    return s.strip()


_L0_BLOCKED_PLAIN: dict[str, tuple[str, str, str]] = {
    "au15_mouth_down": (
        "嘴角下压动作单元（AU15）",
        "现有二维关键点几何代理信噪比不足，未通过跨场稳定性检验",
        "完成专用嘴角下压标定集复核后解封",
    ),
    "head_yaw": (
        "三维头部偏航角",
        "缺少相机内参矩阵，当前仅输出二维归一化代理值，拒绝冒充真三维欧拉角",
        "标定相机内参并切换三维姿态求解后解封",
    ),
    "head_pitch": (
        "三维头部俯仰角",
        "缺少相机内参矩阵，当前仅输出二维归一化代理值，拒绝冒充真三维欧拉角",
        "标定相机内参并切换三维姿态求解后解封",
    ),
    "head_roll": (
        "三维头部滚转角",
        "缺少相机内参矩阵，未实现三维滚转角求解，拒绝二维近似替代",
        "标定相机内参并切换三维姿态求解后解封",
    ),
    "asr_confidence": (
        "语音转写逐字置信度",
        "当前实时转写服务端接口不返回逐词置信概率字段",
        "升级转写服务端返回词级概率或切换支持词级概率的同源引擎后解封",
    ),
}


def _load_l0_blocked() -> list[dict]:
    try:
        raw = json.loads(_L0_PATH.read_text(encoding="utf-8"))
        cols = raw.get("columns") if isinstance(raw, dict) else raw
    except Exception:
        return []
    out: list[dict] = []
    if not isinstance(cols, list):
        return out
    for c in cols:
        if not isinstance(c, dict) or c.get("status") != "blocked":
            continue
        mod = c.get("modality", "")
        name = c.get("column") or c.get("name") or ""
        plain = _L0_BLOCKED_PLAIN.get(name)
        label = plain[0] if plain else _strip_dev_markup(c.get("label") or name)
        reason_plain = plain[1] if plain else _strip_dev_markup(c.get("notes") or "当前硬件或上游接口不支持该列真值测量")
        unblock_human = plain[2] if plain else _strip_dev_markup(c.get("acceptance") or "完成对应标定或上游引擎升级后解封")
        out.append({
            "id": f"{mod}.{name}" if mod else name,
            "column": name,
            "modality": mod,
            "label": label,
            "reason_plain": reason_plain,
            "unblock_human": unblock_human,
            "unblock_machine": f"l0_columns[{mod}.{name}].status == 'implemented'",
            "permanent": False,
        })
    return out


def _build_governance_ledger() -> dict:
    """构建主动拒绝与显式封停台账（5 列 L0 blocked + 活跃封停 + 主动拒绝冗余项）。"""
    l0_blocked = _load_l0_blocked()
    quarantine_active = [
        {
            "id": key,
            "label": key,
            "reason_plain": _strip_dev_markup(q.reason),
            "unblock_human": _strip_dev_markup(q.unblock),
            "unblock_machine": f"remove_from_quarantine('{key}')",
            "permanent": "无对应里程碑" in q.unblock or "永久" in q.unblock,
        }
        for key, q in sorted(evidence_gate.QUARANTINE.items())
    ]
    retired_redundant = [
        {
            "id": "pause_ratio",
            "label": "停顿占比（已主动拒绝）",
            "reason_plain": "与发声占比构成精确代数恒等变换（实测秩相关 0.9990），主动拒绝重复计数",
            "unblock_human": "永久不产出（避免将同一份声学证据重复计算两遍）",
            "unblock_machine": "never_redundant",
            "permanent": True,
        },
        {
            "id": "fluency_score",
            "label": "旧版流畅度合成分（已主动拒绝）",
            "reason_plain": "可完全拆解为发声占比、平均发声段长与语速，零额外信息；现由「停顿结构均匀度」承接分布形状",
            "unblock_human": "永久不产出旧合成分",
            "unblock_machine": "never_redundant",
            "permanent": True,
        },
        {
            "id": "micro_exp_rate_per_min",
            "label": "毫秒级面部短时动作频率（已退役）",
            "reason_plain": "普通摄像头约 1 帧每秒的采样率在物理上无法捕获 40 至 200 毫秒短时面部动作，全语料无出值",
            "unblock_human": "需接入百帧以上高速专用相机并完成独立标定集后另建新项",
            "unblock_machine": "capture_fps >= 100 and facs_calibrated",
            "permanent": False,
        },
        {
            "id": "micro_exp_duration_mean_sec",
            "label": "毫秒级面部短时动作时长（已退役）",
            "reason_plain": "同上：低帧率普通摄像头在采样定理上无法测得毫秒级面部动作持续时间，全语料无出值",
            "unblock_human": "需接入百帧以上高速专用相机并完成独立标定集后另建新项",
            "unblock_machine": "capture_fps >= 100 and facs_calibrated",
            "permanent": False,
        },
    ]
    summary_plain = (
        f"台账共登记 {len(l0_blocked)} 项底层显式封停、"
        f"{len(quarantine_active)} 项旧口径代理隔离，以及 "
        f"{len(retired_redundant)} 项主动拒绝或退役的冗余/不可测项目。"
    )
    return {
        "l0_blocked": l0_blocked,
        "quarantine_active": quarantine_active,
        "retired_redundant": retired_redundant,
        "summary_plain": summary_plain,
    }


def _build_blocked_dependency_warnings(indicators: list[dict], l0_blocked: list[dict]) -> list[dict]:
    blocked_map = {item["column"]: item for item in l0_blocked if item.get("column")}
    warnings: list[dict] = []
    for row in indicators:
        req = instrument_provenance._as_column_list(row.get("required_l0"))
        hits = [c for c in req if c in blocked_map]
        if not hits:
            continue
        name = row.get("name", "")
        label = l1_quality.indicator_label(name)
        warnings.append({
            "indicator": name,
            "indicator_label": label,
            "blocked_columns": hits,
            "note_plain": (
                f"「{label}」底层依赖的 {'、'.join(hits)} 在 86 列台账中标为显式封停"
                "（当前为 2D 像素比代理，非真 3D 空间角），读数仅作本场相对位移参考。"
            ),
        })
    return warnings


def _build_asr_confidence_disclosure(session_id: str, indicators: list[dict]) -> dict:
    """构建 ASR 词级置信度披露与同源低尾降档摘要（绝不使用已证伪的 0.75 硬切）。"""
    tr = transcript_chars.load_transcript_with_asr(session_id) if session_id else None
    if not tr or not tr.get("text"):
        return {
            "status": "no_transcript",
            "engine": "none",
            "audio_scope": "none",
            "affected_indicators": list(_ASR_AFFECTED_INDICATORS),
            "word_conf_summary": None,
            "degraded_indicators": [],
            "note_plain": "本场尚无可用语音转写文本，5 项依赖转写的指标均按缺口处理。",
        }
    asr = tr.get("asr") or {}
    engine = str(asr.get("engine") or "funasr_paraformer")
    audio_scope = str(asr.get("audio_scope") or ("qa_segment" if tr.get("has_segments") else "whole_camera_webm"))
    conf_source = str(asr.get("asr_confidence_source") or "unavailable")
    word_summary = asr.get("word_conf_summary") if isinstance(asr.get("word_conf_summary"), dict) else None

    if conf_source != "word_prob_aligned_same_engine" or not word_summary:
        return {
            "status": "unavailable",
            "engine": engine,
            "audio_scope": audio_scope,
            "affected_indicators": list(_ASR_AFFECTED_INDICATORS),
            "word_conf_summary": None,
            "degraded_indicators": [],
            "note_plain": (
                "语音转写文本由本地识别引擎自动生成，5 项文本指标已满足字数门槛；"
                "因引擎不输出逐字识别概率，复核具体字词时可对照原始转写文本。"
            ),
        }

    # 同源词级概率可用：按分域低尾规则评估（拒绝 0.75 硬切）
    low_ratio = float(word_summary.get("low_word_ratio_lt_0_5", 0.0))
    p20 = float(word_summary.get("p20", 1.0))
    median = float(word_summary.get("median", 1.0))
    extreme_floor = 0.50 if audio_scope == "qa_segment" else 0.45
    need_degrade = (low_ratio > 0.30) or (p20 < 0.35) or (median < extreme_floor)

    degraded: list[str] = []
    if need_degrade:
        for row in indicators:
            if row.get("name") in _ASR_AFFECTED_INDICATORS and row.get("value") is not None:
                degraded.append(row["name"])

    scope_cn = "逐题音频" if audio_scope == "qa_segment" else "整场混音"
    conf_engine = str(asr.get("confidence_engine") or "")
    engine_hint = "旁路同音频识别模型" if conf_engine == "faster-whisper" else "同源词级置信度"
    note_plain = (
        f"已完成词级识别概率核验，音频口径为{scope_cn}，由{engine_hint}给出，"
        f"中位概率 {median:.2f}、后 20% 分位 {p20:.2f}、低于 0.50 词占比 {low_ratio:.0%}："
        + ("转写噪音偏高，建议将 5 项文本相关指标降档解读。" if need_degrade
           else "词级概率分布处于常规区间，无需额外降档。")
    )
    return {
        "status": "available",
        "engine": engine,
        "audio_scope": audio_scope,
        "affected_indicators": list(_ASR_AFFECTED_INDICATORS),
        "word_conf_summary": word_summary,
        "degraded_indicators": degraded,
        "note_plain": note_plain,
    }


def _build_second_instrument_appendix(logs: dict | None) -> dict:
    """若面部日志含 52 维模型系数列（`bs_*`），生成严格隔离的第二仪器只读对照附录。"""
    face_df = (logs or {}).get("face") if isinstance(logs, dict) else None
    if face_df is None or not hasattr(face_df, "columns"):
        return {
            "available": False,
            "disclaimer": "本场未加载含 52 维面部模型系数的日志，无第二仪器独立对照表。",
            "rows": [],
        }
    bs_cols = [c for c in face_df.columns if isinstance(c, str) and c.startswith("bs_")]
    if not bs_cols:
        return {
            "available": False,
            "disclaimer": "本场面部日志不含 52 维模型系数列（仅含关键点几何代理），无第二仪器对照项。",
            "rows": [],
        }
    try:
        from . import blendshape_second_instrument as bsi
        computed = bsi.compute_blendshape_indicators(face_df)
        rows = []
        for item in computed:
            d = item.to_dict() if hasattr(item, "to_dict") else (dict(item) if isinstance(item, dict) else None)
            if d is not None:
                rows.append(d)
        return {
            "available": bool(rows),
            "disclaimer": (
                "以下读数由 FaceLandmarker 52 维模型回归系数独立算得，严格隔离于主表之外，"
                "不参与任何 L1 指标计算与维度映射，不可与关键点几何代理直接混算或相减。"
            ),
            "rows": rows,
        }
    except Exception:
        return {
            "available": False,
            "disclaimer": "第二仪器对照模块暂未产出可展示行（不影响主表几何代理结果）。",
            "rows": [],
        }


def build_lead_summary(dimensions: dict, coverage: dict) -> str:
    """生成与离线 HTML 完全同源的自然语言结论句（修复满覆盖把 2/2 误报为最薄维度的 Bug）。"""
    n_passed = coverage.get("n_passed", 0)
    n_slots = coverage.get("n_slots", 0)
    n_dropped = coverage.get("n_dropped", 0)
    n_measurable = coverage.get("n_measurable")
    n_not_measured = coverage.get("n_not_measured")
    n_no_indicator = coverage.get("n_no_indicator")
    denom = n_measurable if isinstance(n_measurable, int) else n_slots

    def _pair(dim: dict) -> tuple[int, int]:
        got, _, total = str(dim.get("matched_indicators", "0/0")).partition("/")
        try:
            return int(got), int(total)
        except (TypeError, ValueError):
            return 0, 0

    ranked = sorted(
        dimensions.values(),
        key=lambda d: (_pair(d)[0] / (_pair(d)[1] or 1), _pair(d)[0]),
        reverse=True,
    )
    sentences: list[str] = []

    if not dimensions:
        sentences.append("这一场没有任何维度产出可核对的行为量。")
    elif not n_passed:
        pair = f"0 / {denom}" if denom else "0"
        sentences.append(
            f"这一场在 {len(dimensions)} 个维度上量到了 {pair} 项可测量指标"
            "—— 不是「没测」，是测到的还不足以支撑读数，逐条原因见下。"
        )
    else:
        all_full = all(_pair(d)[1] > 0 and _pair(d)[0] >= _pair(d)[1] for d in ranked)
        if all_full:
            sentences.append(
                f"这一场在 {len(dimensions)} 个维度上量到了 {n_passed} / {denom} 项可测量指标，"
                f"全部 {len(dimensions)} 个维度的可测量指标均已完整覆盖。"
            )
        else:
            best = ranked[0]
            b_got, b_tot = _pair(best)
            b_ratio = b_got / (b_tot or 1)
            sentences.append(
                f"这一场在 {len(dimensions)} 个维度上量到了 {n_passed} / {denom} 项可测量指标，"
                f"其中覆盖最完整的是「{best.get('display_name', '')}」（{b_got} / {b_tot}）。"
            )
            if len(ranked) > 1:
                worst = ranked[-1]
                w_got, w_tot = _pair(worst)
                w_ratio = w_got / (w_tot or 1)
                if w_ratio < b_ratio - 1e-9:
                    sentences.append(
                        f"覆盖最薄的是「{worst.get('display_name', '')}」（{w_got} / {w_tot}）。"
                    )

    if isinstance(n_not_measured, int) and isinstance(n_no_indicator, int):
        if n_not_measured or n_no_indicator:
            sentences.append(
                f"分母之外还有 {n_not_measured} 项「这套采集不测」、{n_no_indicator} 项「整维没有对应指标」—— "
                f"这 {n_not_measured + n_no_indicator} 项不是这一场量得好不好的问题，合起来才是 {n_slots} 个槽。"
            )
    elif n_dropped:
        sentences.append(
            f"另有 {n_dropped} 项在当前这套采集里没有对应的测法，已整条剔除。"
        )
    cap = coverage.get("confidence_cap")
    if cap and cap != "无":
        sentences.append(f"所有读数的置信度上限是「{cap}」。")
    return "".join(sentences)


def _instrument_block(indicators: list[dict]) -> dict:
    counts = instrument_provenance.describe(indicators)
    unmapped = sorted({
        c for row in indicators
        for c in instrument_provenance.unmapped_columns(row.get("required_l0"))
    })
    n_multi = sum(1 for row in indicators
                  if len(instrument_provenance.instruments_of(row)) > 1)
    sub_by_ind = {
        row["name"]: instrument_provenance.sub_instruments_of(row)
        for row in indicators if isinstance(row, dict) and row.get("name")
    }
    return {
        "n_indicators": len(indicators),
        "counts": counts,
        "footer": instrument_provenance.format_footer(indicators),
        "labels": dict(INSTRUMENT_LABEL),
        "order": list(instrument_provenance.INSTRUMENTS),
        "n_multi": n_multi,
        "unmapped_columns": unmapped,
        "unknown_note": INSTRUMENT_UNKNOWN_NOTE,
        "multi_note": INSTRUMENT_MULTI_NOTE,
        "sub_instruments_by_indicator": sub_by_ind,
        "sub_instrument_labels": dict(instrument_provenance.SUB_INSTRUMENT_LABELS),
        "non_interchangeability_disclosure": dict(NON_INTERCHANGEABILITY_DISCLOSURE),
    }


def _name_of(slot: dict) -> str:
    """这条记录该印的**中文名** —— 有同构念 L1 指标就用**指标名**,否则退回槽名。

    ## 为什么必须分两种(2026-11-22 修)

    槽名是**旧引擎**的叫法,而值已经是新指标算的。两条槽上这件事直接说反了意思:

    * `eye_contact` 槽叫「眼神接触**比例**」,承接它的 `gaze_dwell_p50` 数的是
      「眼神接触**时长**」(秒)——按槽名印,读者会把 `15.335`（秒）当成比例;
    * `symmetry_score` 槽叫「面部**对称**性」,承接它的 `facial_asymmetry_p90`
      数的是 `p90(1 − symmetry_score)`（越大越**不**对称）——按槽名印,方向说反。

    真删槽（没有对应指标）没有指标名可用,只能退槽名 —— 那是对的,报告里那些行
    本来就在讲"这一条当前这套采集不测"。
    """
    l1 = (slot or {}).get("l1")
    if isinstance(l1, str) and l1:
        label = l1_quality.indicator_label(l1)
        if isinstance(label, str) and label and label != l1:   # 未登记时它会原样吐键名
            return label
    return l1_mapper.slot_label((slot or {}).get("slot", ""))

def _evidence_item(slot: dict, instruments: list[str]) -> dict:
    """一个槽 → 前端 `EvidenceItem`。字段名逐一对齐 `assessment.ts`。

    `observed_interval` 给空数组:L1 是**会话级**测量,一个指标本场只有一个值,
    没有"本场观测区间"可言。空数组是前端口径里"取不到区间"的写法,
    不是 0,也不是"区间是 [0,0]"。

    `human_name` 用**指标的中文名**(有指标时)或槽名(真删槽时),**不用** L1 的 `formula` ——
    后者是给开发者核对算法的(两百多字,带 `⚠️` 和 markdown 粗体,还会点出
    另一个内部键名),它该待在 `l1.indicators` 的审计链里,不该占着「指标」这一列。
    钉子:`tests/test_report_gaps_are_readable.py::test_evidence_chain_names_are_not_raw_formulas`。

    ★ A5(2026-11-21):新增 `instruments`。**永远是 list**(单仪器也是单元素列表)——
    渲染层因此不必判断"该读 `instrument` 还是 `instruments`"(那种判断在本仓出现过:
    两处各判断一次,一次读错键就把多仪器读成了单仪器)。
    """
    return {
        "feature": slot["l1"],
        "human_name": _name_of(slot),
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
            out.append(f"{_name_of(s)}:{why}")
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
                "indicator": _name_of(s),
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

    coverage_payload = {
        "n_slots": counts["n_slots"],
        "n_passed": counts["n_passed"],
        "n_dropped": counts["n_dropped"],
        "n_measurable": counts["n_measurable"],
        "n_measurable_no_data": counts["n_measurable_no_data"],
        "n_not_measured": counts["n_not_measured"],
        "n_no_indicator": counts["n_no_indicator"],
        "pass_rate_of_measurable": counts["pass_rate_of_measurable"],
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
    }

    indicator_names = [row["name"] for row in quality.indicators if isinstance(row, dict) and row.get("name")]
    governance_ledger = _build_governance_ledger()
    zero_explanations = {
        row["name"]: z
        for row in quality.indicators
        if isinstance(row, dict)
        and (z := l1_quality.explain_zero(
            row.get("name", ""), row.get("value"), row.get("n_valid"), row.get("n_total")
        ))
    }

    mod_summary = l1_quality.modality_summary(quality)
    cm_coverage = cross_modal_coverage.evaluate_cross_modal_coverage(result, logs=logs)
    sq_audit = signal_quality_clf.audit_session_signal_quality(logs, session_id)
    bs_cal = blendshape_calibration.calibrate_session_blendshapes(logs)
    fv2 = feature_layer_v2.extract_feature_layer_v2(logs, session_id)
    has_referee_input = quality.n_measured > 0 or bool(fv2.get("available"))
    referee_summary = (
        {
            "n_total": quality.n_total,
            "n_measured": quality.n_measured,
            "modality_coverage": quality.coverage,
            "modality_summary": mod_summary,
            "cross_modal_coverage": cm_coverage,
            "signal_quality_audit": sq_audit,
        }
        if has_referee_input
        else None
    )
    referee_ml = referee_ml_engine.evaluate_referee_ml(
        logs,
        session_id,
        l1_summary=referee_summary,
        feature_v2=fv2 if fv2.get("available") else None,
    )

    return {
        "engine": "l1",
        "brand": {
            "name": brand.BRAND_NAME,
            "subtitle": brand.BRAND_SUBTITLE,
            "slogan": brand.BRAND_SLOGAN,
        },
        "dimensions": dimensions,
        "vanished_dimensions": [_printable_vanished(v) for v in vanished],
        "summary_narrative": NO_COMPOSITE,
        "coverage": coverage_payload,
        "model_metadata": {
            "version": PAYLOAD_VERSION,
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        },
        "l1": {
            "n_total": quality.n_total,
            "n_measured": quality.n_measured,
            "by_confidence": dict(quality.by_confidence),
            "by_gap": dict(quality.by_gap),
            "modality_coverage": quality.coverage,
            "modality_summary": mod_summary,
            "timebase": quality.timebase,
            "windows": quality.windows,
            "indicators": quality.indicators,
            "low_confidence_notes": explain_confidence(quality),
            "low_confidence_notes_plain": explain_confidence(quality, plain=True),
            "comparability_notes": l1_mapper.comparability_notes(),
            "comparability_notes_plain": l1_mapper.comparability_notes_plain(),
            "gaps": list(result.gaps),
            "dispersion": dict(getattr(result, "dispersion", None) or {}),
            "dispersion_note": (
                "展示本场各道题之间的数值波动范围，包含最小值、中位数、最大值与中间半数区间；"
                "仅用于观察本场不同题目间的变化，不能跨场次直接相减。"
            ),
            "instruments": _instrument_block(quality.indicators),
            # ── 2026-11-23 增量顶层证据链披露字段（不动既有行键，向后兼容） ──
            "lead_summary": build_lead_summary(dimensions, coverage_payload),
            "indicator_meta": l1_quality.build_indicator_meta(),
            "zero_explanations": zero_explanations,
            "dimension_registry": dict(l1_mapper.DIMENSION_REGISTRY),
            "parameter_provenance": parameter_provenance.build_parameter_provenance(indicator_names),
            "cross_modal_coverage": cm_coverage,
            "asr_confidence_disclosure": _build_asr_confidence_disclosure(session_id, quality.indicators),
            "governance_ledger": governance_ledger,
            "blocked_dependency_warnings": _build_blocked_dependency_warnings(
                quality.indicators, governance_ledger["l0_blocked"]
            ),
            "second_instrument_appendix": _build_second_instrument_appendix(logs),
            "face_dual_instrument": face_dual_instrument_audit.audit_face_dual_instrument(logs),
            "blendshape_calibration": bs_cal,
            "feature_layer_v2": fv2,
            "behavior_regimes": behavior_regime_model.segment_session_regimes(logs, session_id),
            "referee_ml": referee_ml,
            "signal_quality_audit": sq_audit,
            "speaker_turn_audit": speaker_turn_audit.audit_speaker_turns(session_id, logs=logs),
        },
    }
