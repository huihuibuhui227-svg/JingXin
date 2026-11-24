# report_frontend/scene_extensions.py
"""多场景差异化扩展指标计算模块（严格纯规则、不给总分）。

当前服务于场景档案的场景扩展层（挂载于 `payload["scene_profile"]["extensions"]`），
严格保持核心 41 项 L1 指标计数与契约不变：
1. 基地二（`ai_metacognition`）扩展 2 项纯规则审辨结构指标：
   - `counter_premise_marker_rate`：前提质疑、反例引入、边界限定标记每百汉字密度；
   - `revision_anchor_density`：先定位原回答缺陷、再给出修正阐述的完整句段占比。
2. 所有指标均遵循证据门三条不变量（缺数据时 value=None 且附 gap 与解释），绝不输出分值。
"""
from __future__ import annotations

import re
from typing import Any

from report_frontend import l1_text, transcript_chars
from report_frontend.l1_indicators import (
    GAP_NO_SAMPLES,
    GAP_NOT_STORED,
    confidence_for,
)

CRITIQUE_EXTENSION_MIN_CHARS = 80

_COUNTER_PREMISE_PATTERNS: tuple[str, ...] = (
    "前提不成立",
    "前提假设",
    "反例",
    "边界条件",
    "局限在于",
    "忽视了",
    "忽略了",
    "未必成立",
    "不一定",
    "假设有误",
    "样本偏差",
    "口径不一致",
    "混淆了",
    "不能推出",
    "成立条件",
    "例外情况",
    "存在漏洞",
    "缺乏依据",
    "过度外推",
)

_DEFECT_LOCATORS: tuple[str, ...] = (
    "问题在于",
    "错误在于",
    "漏洞在于",
    "原回答",
    "上一版",
    "AI 回答",
    "ai 回答",
    "这段话",
    "忽略了",
    "漏掉了",
    "没有考虑",
    "未考虑",
    "偏误",
)

_REVISION_MARKERS: tuple[str, ...] = (
    "应当改为",
    "建议修改",
    "应该修正",
    "更正为",
    "正确的做法",
    "替代方案",
    "补充约束",
    "改写为",
    "修订为",
    "因此需要",
    "重新验证",
    "改进方式",
)


def _split_clauses(text: str) -> list[str]:
    return [seg.strip() for seg in re.split(r"[。！？；\n]+", text) if seg.strip()]


def _resolve_transcript_text(session_id: str, logs: dict | None = None) -> str | None:
    if isinstance(logs, dict):
        raw = logs.get("transcript_text")
        if isinstance(raw, str) and raw.strip():
            return raw
    return transcript_chars.text(session_id)


def compute_metacognition_text_extensions(
    session_id: str,
    *,
    logs: dict | None = None,
    min_chars: int = CRITIQUE_EXTENSION_MIN_CHARS,
) -> dict[str, Any]:
    """计算基地二「AI 审辨与修正阐述」2 项纯规则确定性文本结构指标。"""
    text = _resolve_transcript_text(session_id, logs=logs)
    if text is None:
        return {
            "available": False,
            "source_chars": 0,
            "gap": GAP_NOT_STORED,
            "indicators": [
                {
                    "name": "counter_premise_marker_rate",
                    "label": "前提质疑与反例限定标记密度",
                    "unit": "次/百字",
                    "value": None,
                    "n_valid": 0,
                    "n_total": 0,
                    "confidence": "无",
                    "gap": GAP_NOT_STORED,
                    "summary": "本场未获取到转写文本，暂无法核验审辨质疑标记。",
                },
                {
                    "name": "revision_anchor_density",
                    "label": "指错定位并给出修正方案句段占比",
                    "unit": "比例",
                    "value": None,
                    "n_valid": 0,
                    "n_total": 0,
                    "confidence": "无",
                    "gap": GAP_NOT_STORED,
                    "summary": "本场未获取到转写文本，暂无法核验修正闭环句段。",
                },
            ],
        }

    n_han = len(l1_text._cjk(text))
    clauses = _split_clauses(text)
    n_clauses = len(clauses)
    eff_min = max(30, int(min_chars))

    if n_han < eff_min or n_clauses == 0:
        return {
            "available": False,
            "source_chars": n_han,
            "gap": GAP_NO_SAMPLES,
            "indicators": [
                {
                    "name": "counter_premise_marker_rate",
                    "label": "前提质疑与反例限定标记密度",
                    "unit": "次/百字",
                    "value": None,
                    "n_valid": n_han,
                    "n_total": n_han,
                    "confidence": "无",
                    "gap": GAP_NO_SAMPLES,
                    "summary": f"有效汉字仅 {n_han} 字（低于最小文本观测长度 {eff_min} 字），暂不给出密度估计。",
                },
                {
                    "name": "revision_anchor_density",
                    "label": "指错定位并给出修正方案句段占比",
                    "unit": "比例",
                    "value": None,
                    "n_valid": n_clauses,
                    "n_total": n_clauses,
                    "confidence": "无",
                    "gap": GAP_NO_SAMPLES,
                    "summary": f"文本量不足（{n_han} 字 / {n_clauses} 句），暂不统计修正闭环占比。",
                },
            ],
        }

    premise_hits = 0
    matched_terms: list[str] = []
    for pat in _COUNTER_PREMISE_PATTERNS:
        cnt = text.count(pat)
        if cnt > 0:
            premise_hits += cnt
            matched_terms.append(f"{pat}×{cnt}")
    premise_rate = round(premise_hits / max(n_han, 1) * 100.0, 4)
    premise_conf = confidence_for(n_han, n_han, eff_min)

    paired_clauses = 0
    for idx, cl in enumerate(clauses):
        has_defect = any(tok in cl for tok in _DEFECT_LOCATORS)
        has_rev = any(tok in cl for tok in _REVISION_MARKERS)
        # 相邻两句形成“前句指错 → 本句/下句给修正”同样视为完整审辨修正闭环
        next_cl = clauses[idx + 1] if idx + 1 < n_clauses else ""
        if (has_defect and has_rev) or (has_defect and any(tok in next_cl for tok in _REVISION_MARKERS)):
            paired_clauses += 1
    revision_ratio = round(paired_clauses / max(n_clauses, 1), 4)
    rev_conf = confidence_for(n_clauses, n_clauses, max(4, eff_min // 20))

    return {
        "available": True,
        "source_chars": n_han,
        "clause_count": n_clauses,
        "matched_premise_terms": matched_terms[:10],
        "gap": None,
        "indicators": [
            {
                "name": "counter_premise_marker_rate",
                "label": "前提质疑与反例限定标记密度",
                "unit": "次/百字",
                "value": premise_rate,
                "n_valid": n_han,
                "n_total": n_han,
                "confidence": premise_conf,
                "gap": None,
                "summary": (
                    f"共检出 {premise_hits} 处前提质疑或边界限定标记，"
                    f"合 {premise_rate:.2f} 次/百字（样本 {n_han} 汉字）。"
                ),
            },
            {
                "name": "revision_anchor_density",
                "label": "指错定位并给出修正方案句段占比",
                "unit": "比例",
                "value": revision_ratio,
                "n_valid": n_clauses,
                "n_total": n_clauses,
                "confidence": rev_conf,
                "gap": None,
                "summary": (
                    f"全段 {n_clauses} 个分句中识别到 {paired_clauses} 处「指错并修正」配对句段，"
                    f"占比 {revision_ratio * 100:.1f}%。"
                ),
            },
        ],
    }


def build_scene_extensions(
    scene_profile_id: str,
    session_id: str,
    *,
    logs: dict | None = None,
    feature_v2: dict | None = None,
) -> dict[str, Any]:
    """按场景标识组装增量化结构扩展信息，不改变核心 41 项指标集合。"""
    if scene_profile_id == "ai_metacognition":
        meta_text = compute_metacognition_text_extensions(session_id, logs=logs)
        return {
            "metacognition_text": meta_text,
            "stage_focus": ["stage_read_ai", "stage_critique", "stage_revise"],
        }
    if scene_profile_id == "longitudinal_followup":
        kin_summary: dict[str, Any] = {
            "available": False,
            "note": "个体内多次访次纵向比较请结合匿名随访档案与首访基线查询。",
        }
        if isinstance(feature_v2, dict) and feature_v2.get("available"):
            kin_summary = {
                "available": True,
                "face_3d_angular_vel_median": (feature_v2.get("face_3d") or {}).get(
                    "head_angular_vel_deg_per_sec_median"
                ),
                "gesture_sparc_smoothness": (feature_v2.get("gesture_dynamics") or {}).get(
                    "sparc_smoothness"
                ),
                "note": "当前单场运动学摘要可作为匿名纵向随访档案的单次访次快照。",
            }
        return {"longitudinal_snapshot": kin_summary}
    return {}
