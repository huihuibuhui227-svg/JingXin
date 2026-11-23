# report_frontend/cross_modal_coverage.py
"""跨模态时间窗覆盖率门控（纯规则、零黑盒、O(N)）。

## 解决什么问题
单模态各自覆盖率即使达到 100%，若面部覆盖 `[0, 7427s]` 而语音仅覆盖末段 435s，
三路联合时间重合度极低，不能把不同时间段的信号拼在一起做跨模态联合解读。

## 四条硬规矩
1. **先查锚点再算重合**：无挂钟锚点的音频时间轴（`anchored=False`）严禁假算逐秒重合，
   直接报 `unverifiable` 并附上时长比 `span_ratio`。
2. **只数有效观测帧**：空日志行（核心观测列全 NaN）不计入有效观测秒集合 `S_m`。
3. **不删单模态实测值**：门控只决定 `allow_joint_claim` 与警示等级，不隐藏单模态原始数。
4. **门槛全部登记于 `parameter_registry.json`（T3a 经验临时值）**。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from . import l1_quality

#: 跨模态按秒网格门控参数（同步登记在 parameter_registry.json，T3a 经验临时值）
RESOLUTION_SEC = 1
JOINT_COV_OK = 0.70
JOINT_COV_BLOCK = 0.30
MIN_OVERLAP_SEC_SESSION = 10
MIN_OVERLAP_SEC_WINDOW = 5

#: 5 大维度涉及的模态（用于逐维判断是否允许跨模态联合结论）
DIMENSION_MODALITIES: dict[str, tuple[str, ...]] = {
    "logical_thinking": ("text", "face"),
    "stress_resilience": ("face", "gesture"),
    "communication_fluency": ("voice",),
    "confidence_level": ("gesture", "face"),
    "cognitive_efficiency": ("text", "voice"),
}


def _anchor_status(tb) -> str:
    if tb is None or not getattr(tb, "usable", False):
        return "missing"
    status_val = getattr(tb.status, "value", str(tb.status))
    if status_val == "wall_unreliable":
        return "wall_drift_unreliable"
    if not getattr(tb, "anchored", True):
        return "unanchored_relative"
    return "wall_anchored"


def _valid_seconds_set(df: pd.DataFrame | None, tb, modality: str,
                       t_start: float | None = None,
                       t_end: float | None = None) -> set[int]:
    """提取可靠观测帧对应的整数秒集合 `S_m`。"""
    if tb is None or not getattr(tb, "usable", False) or tb.t_sec is None or len(tb.t_sec) == 0:
        return set()
    t = pd.to_numeric(tb.t_sec, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(t)
    if t_start is not None and t_end is not None:
        mask &= (t >= t_start) & (t < t_end)

    # 若有对应原始日志 DataFrame，剔除核心列全为空的无效行
    cols = l1_quality.COVERAGE_COLUMNS.get(modality)
    if df is not None and isinstance(df, pd.DataFrame) and len(df) == len(t) and cols:
        present = [c for c in cols if c in df.columns]
        if present:
            has_data = df[present].notna().any(axis=1).to_numpy(dtype=bool)
            mask &= has_data

    valid_t = t[mask]
    if valid_t.size == 0:
        return set()
    return set(np.floor(valid_t / RESOLUTION_SEC).astype(int).tolist())


def _pair_metrics(
    mod_a: str,
    mod_b: str,
    sets: dict[str, set[int]],
    anchors: dict[str, str],
    spans: dict[str, float | None],
    min_overlap_sec: int,
) -> dict[str, Any]:
    anc_a, anc_b = anchors.get(mod_a, "missing"), anchors.get(mod_b, "missing")
    span_a, span_b = spans.get(mod_a), spans.get(mod_b)
    span_ratio: float | None = None
    if span_a and span_b and span_a > 0 and span_b > 0:
        span_ratio = round(min(span_a, span_b) / max(span_a, span_b), 4)

    if anc_a == "missing" or anc_b == "missing":
        missing_which = [m for m, a in ((mod_a, anc_a), (mod_b, anc_b)) if a == "missing"]
        return {
            "overlap_sec": None,
            "union_sec": None,
            "jaccard": None,
            "span_ratio": span_ratio,
            "status": "blocked",
            "reason": f"缺少模态时间轴：{'、'.join(missing_which)}",
        }

    if anc_a != "wall_anchored" or anc_b != "wall_anchored":
        unanchored = [m for m, a in ((mod_a, anc_a), (mod_b, anc_b)) if a != "wall_anchored"]
        return {
            "overlap_sec": None,
            "union_sec": None,
            "jaccard": None,
            "span_ratio": span_ratio,
            "status": "unverifiable",
            "reason": f"{'、'.join(unanchored)} 时间轴缺少可靠挂钟锚点，无法验证逐秒时间对齐",
        }

    sa, sb = sets.get(mod_a, set()), sets.get(mod_b, set())
    overlap = len(sa & sb)
    union = len(sa | sb)
    if union == 0:
        return {
            "overlap_sec": 0,
            "union_sec": 0,
            "jaccard": None,
            "span_ratio": span_ratio,
            "status": "blocked",
            "reason": "两路模态均无有效观测秒",
        }
    jaccard = round(overlap / union, 4)
    if overlap < min_overlap_sec or jaccard < JOINT_COV_BLOCK:
        status = "blocked"
        reason = f"共同有效覆盖仅 {overlap} 秒（交并比 {jaccard:.0%}），低于阻断线 {JOINT_COV_BLOCK:.0%}"
    elif jaccard < JOINT_COV_OK:
        status = "degraded"
        reason = f"共同有效覆盖 {overlap} 秒（交并比 {jaccard:.0%}），仅部分时段重合，跨模态联合解读需谨慎"
    else:
        status = "aligned"
        reason = f"共同有效覆盖 {overlap} 秒（交并比 {jaccard:.0%}），时间对齐充分"

    return {
        "overlap_sec": overlap,
        "union_sec": union,
        "jaccard": jaccard,
        "span_ratio": span_ratio,
        "status": status,
        "reason": reason,
    }


def evaluate_cross_modal_coverage(
    result,
    logs: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """评估整场与逐题窗口的跨模态时间窗共同覆盖率。"""
    logs = logs or {}
    tl = getattr(result, "timebase", None)
    modalities = ("face", "gesture", "voice")

    anchors: dict[str, str] = {}
    spans: dict[str, float | None] = {}
    sets_session: dict[str, set[int]] = {}

    tl_mods = getattr(tl, "modalities", None) or {}
    for m in modalities:
        tb = tl_mods.get(m)
        anchors[m] = _anchor_status(tb)
        spans[m] = getattr(tb, "span_sec", None) if tb is not None else None
        df = logs.get(m) if m != "voice" else (
            logs.get("voice") if logs.get("voice") is not None
            else (logs.get("voice_interview") if logs.get("voice_interview") is not None
                  else logs.get("voice_research"))
        )
        sets_session[m] = _valid_seconds_set(df, tb, m)

    # 文本模态由转写覆盖，时间轴绑定语音/整场（非逐帧锚定）
    has_text = any(
        ind.family.startswith("T") and ind.has_value for ind in (result.indicators or [])
    )
    anchors["text"] = "transcript_session_level" if has_text else "missing"

    pairs = {
        "face_gesture": _pair_metrics(
            "face", "gesture", sets_session, anchors, spans, MIN_OVERLAP_SEC_SESSION
        ),
        "face_voice": _pair_metrics(
            "face", "voice", sets_session, anchors, spans, MIN_OVERLAP_SEC_SESSION
        ),
        "gesture_voice": _pair_metrics(
            "gesture", "voice", sets_session, anchors, spans, MIN_OVERLAP_SEC_SESSION
        ),
    }

    # 三路联合
    if all(anchors[m] == "wall_anchored" for m in modalities):
        s_all_inter = sets_session["face"] & sets_session["gesture"] & sets_session["voice"]
        s_all_union = sets_session["face"] | sets_session["gesture"] | sets_session["voice"]
        ov = len(s_all_inter)
        un = len(s_all_union)
        jac = round(ov / un, 4) if un else None
        if not un or ov < MIN_OVERLAP_SEC_SESSION or (jac is not None and jac < JOINT_COV_BLOCK):
            st, rsn = "blocked", f"三路共同有效覆盖仅 {ov} 秒，阻断三模态联合声明"
        elif jac is not None and jac < JOINT_COV_OK:
            st, rsn = "degraded", f"三路共同有效覆盖 {ov} 秒（交并比 {jac:.0%}），需谨慎联合解读"
        else:
            st, rsn = "aligned", f"三路共同有效覆盖 {ov} 秒（交并比 {jac:.0%}）"
        all_three = {"overlap_sec": ov, "union_sec": un, "jaccard": jac, "status": st, "reason": rsn}
    elif any(anchors[m] == "missing" for m in modalities):
        miss = [m for m in modalities if anchors[m] == "missing"]
        all_three = {
            "overlap_sec": None, "union_sec": None, "jaccard": None,
            "status": "blocked", "reason": f"缺少模态时间轴（{'、'.join(miss)}），无三模态联合覆盖",
        }
    else:
        unanc = [m for m in modalities if anchors[m] != "wall_anchored"]
        all_three = {
            "overlap_sec": None, "union_sec": None, "jaccard": None,
            "status": "unverifiable",
            "reason": f"{'、'.join(unanc)} 无可靠挂钟锚点，无法核实三模态逐秒对齐",
        }

    # 逐维跨模态门控判定
    by_dimension: dict[str, dict[str, Any]] = {}
    for dim_key, mods in DIMENSION_MODALITIES.items():
        if len(mods) <= 1:
            by_dimension[dim_key] = {
                "modalities": list(mods),
                "status": "single_modality",
                "allow_joint_claim": True,
                "reason": "单模态维度，不涉及跨模态时间对齐",
            }
            continue
        if set(mods) == {"face", "gesture"}:
            p = pairs["face_gesture"]
            by_dimension[dim_key] = {
                "modalities": list(mods),
                "status": p["status"],
                "allow_joint_claim": p["status"] in ("aligned", "degraded"),
                "reason": p["reason"],
            }
        else:
            # 含 text 的维度：文本为整场/逐段转写，无逐秒挂钟锚点
            other = [m for m in mods if m != "text"][0]
            if not has_text or anchors.get(other) == "missing":
                by_dimension[dim_key] = {
                    "modalities": list(mods),
                    "status": "blocked",
                    "allow_joint_claim": False,
                    "reason": "该跨模态维度缺少部分模态数据，只能单模态分读",
                }
            else:
                by_dimension[dim_key] = {
                    "modalities": list(mods),
                    "status": "unverifiable",
                    "allow_joint_claim": False,
                    "reason": "文本转写为会话/段级粒度且未挂接逐秒挂钟，与视觉或声学指标仅作分项并列呈现，不做逐秒跨模态联合推断",
                }

    # 逐题窗口跨模态覆盖（当存在回答窗口时）
    by_window: list[dict[str, Any]] = []
    ws = getattr(result, "windows", None)
    if getattr(ws, "windows", None):
        for w in ws.windows:
            w_sets = {}
            for m in modalities:
                tb = tl_mods.get(m)
                df = logs.get(m) if m != "voice" else None
                w_sets[m] = _valid_seconds_set(df, tb, m, w.t_start, w.t_end)
            fg = _pair_metrics("face", "gesture", w_sets, anchors, spans, MIN_OVERLAP_SEC_WINDOW)
            by_window.append({
                "window": f"answer:{w.index}",
                "duration_sec": round(max(0.0, float(w.t_end - w.t_start)), 2),
                "face_gesture": fg,
            })

    # 一句话人话汇总
    fg_pair = pairs["face_gesture"]
    fv_pair = pairs["face_voice"]
    parts_plain: list[str] = []
    if fg_pair["status"] == "aligned":
        parts_plain.append(
            f"面部与手势按秒对齐充分（共同有效观测 {fg_pair['overlap_sec']} 秒，交并比 {fg_pair['jaccard']:.0%}）"
        )
    elif fg_pair["status"] == "degraded":
        parts_plain.append(
            f"面部与手势仅部分时段重合（交并比 {fg_pair['jaccard']:.0%}），跨模态对照需谨慎"
        )
    else:
        parts_plain.append(f"面部与手势未通过跨模态时间对齐门控（{fg_pair['reason']}）")

    if fv_pair["status"] == "aligned":
        parts_plain.append(f"视觉与语音对齐充分（交并比 {fv_pair['jaccard']:.0%}）")
    elif fv_pair["status"] == "unverifiable":
        ratio_txt = f"（时长比 {fv_pair['span_ratio']:.2f}）" if fv_pair.get("span_ratio") is not None else ""
        parts_plain.append(f"语音时间轴无挂钟锚点{ratio_txt}，与视觉指标分项呈现、不做逐秒联合推断")
    else:
        parts_plain.append(f"视觉与语音未对齐（{fv_pair['reason']}）")

    return {
        "resolution_sec": RESOLUTION_SEC,
        "thresholds": {
            "ok": JOINT_COV_OK,
            "block": JOINT_COV_BLOCK,
            "min_overlap_sec": MIN_OVERLAP_SEC_SESSION,
            "min_overlap_sec_window": MIN_OVERLAP_SEC_WINDOW,
            "tier": "T3a",
        },
        "modality_anchors": anchors,
        "session": {
            "active_sec": {m: len(sets_session[m]) for m in modalities},
            "pairs": pairs,
            "all_three": all_three,
        },
        "by_dimension": by_dimension,
        "by_window": by_window,
        "summary_plain": "；".join(parts_plain) + "。",
    }
