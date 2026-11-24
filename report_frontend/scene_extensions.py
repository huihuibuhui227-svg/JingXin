# report_frontend/scene_extensions.py
"""多场景差异化扩展指标计算模块（严格纯规则、不给总分、理论构念对齐）。

本模块在保持核心 41 项物理指标计数与契约完全不变的前提下，
向 `payload["scene_profile"]["extensions"]` 挂载分场景增量观测量：

1. 元智审辨（`ai_metacognition`）扩展 4 项理论锚定、零黑盒规则指标（Nelson & Narens 1990；
   Schraw & Dennison 1994；Tankelevitch et al. CHI 2024；Lee et al. CHI 2025；鄢章华 2026）：
   - `metacognition_text`（2 项，保持原有 2 项结构契约不变）：
     - `counter_premise_marker_rate`：前提质疑与反例限定标记密度（监测环节）
     - `revision_anchor_density`：指错定位并给出修正方案句段占比（排错调节闭环）
   - `metacognition_regulation`（2 项新增调控指标）：
     - `evidence_verification_marker_rate`：信息核验与来源口径求证标记密度（核验环节）
     - `metacognitive_regulation_balance`：监测—核验—排错三环覆盖度（无分值，仅计命中环数占比）
   - `stage_windows_observed`：调用 `event_windows.extract_stage_windows()` 提取可观测任务阶段边界。
2. 时序观微（`longitudinal_followup`）：
   - 修复并从 `feature_layer_v2["tracks"]["face_kinematics"]`（仅真 3D 矩阵口径）与
     `feature_layer_v2["tracks"]["gesture_smoothness"]` 提取单次运动学快照摘要。
"""
from __future__ import annotations

import re
from typing import Any

from report_frontend import event_windows, l1_text, transcript_chars
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

_VERIFICATION_PATTERNS: tuple[str, ...] = (
    "来源",
    "原始数据",
    "实测",
    "口径",
    "交叉验证",
    "对照组",
    "控制变量",
    "样本量",
    "依据",
    "查证",
    "出处",
    "复现",
    "统计显著",
    "数据支撑",
    "文献",
    "核实",
    "核验",
    "检验",
    "实证",
    "基准",
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
    """计算「AI 审辨与修正阐述」2 项纯规则确定性文本结构指标（监测与排错）。"""
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


def compute_metacognition_regulation_extensions(
    session_id: str,
    *,
    logs: dict | None = None,
    min_chars: int = CRITIQUE_EXTENSION_MIN_CHARS,
    base_text_ext: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """计算人机协同元认知三环调控补充指标（信息核验 + 三环覆盖度）。"""
    text = _resolve_transcript_text(session_id, logs=logs)
    if text is None:
        return {
            "available": False,
            "source_chars": 0,
            "gap": GAP_NOT_STORED,
            "covered_loops_labels": [],
            "missing_loops_labels": ["前提质疑监测", "信息来源与口径核验", "指错修正排错闭环"],
            "indicators": [
                {
                    "name": "evidence_verification_marker_rate",
                    "label": "信息核验与来源口径求证标记密度",
                    "unit": "次/百字",
                    "value": None,
                    "n_valid": 0,
                    "n_total": 0,
                    "confidence": "无",
                    "gap": GAP_NOT_STORED,
                    "summary": "本场未获取到转写文本，暂无法核验信息溯源与口径求证标记。",
                },
                {
                    "name": "metacognitive_regulation_balance",
                    "label": "监测—核验—排错三环行为覆盖度",
                    "unit": "比例",
                    "value": None,
                    "n_valid": 0,
                    "n_total": 3,
                    "confidence": "无",
                    "gap": GAP_NOT_STORED,
                    "summary": "本场未获取到转写文本，暂无法统计元认知三环行为出现情况。",
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
            "covered_loops_labels": [],
            "missing_loops_labels": ["前提质疑监测", "信息来源与口径核验", "指错修正排错闭环"],
            "indicators": [
                {
                    "name": "evidence_verification_marker_rate",
                    "label": "信息核验与来源口径求证标记密度",
                    "unit": "次/百字",
                    "value": None,
                    "n_valid": n_han,
                    "n_total": n_han,
                    "confidence": "无",
                    "gap": GAP_NO_SAMPLES,
                    "summary": f"有效汉字仅 {n_han} 字（低于最小文本观测长度 {eff_min} 字），暂不给出密度估计。",
                },
                {
                    "name": "metacognitive_regulation_balance",
                    "label": "监测—核验—排错三环行为覆盖度",
                    "unit": "比例",
                    "value": None,
                    "n_valid": 0,
                    "n_total": 3,
                    "confidence": "无",
                    "gap": GAP_NO_SAMPLES,
                    "summary": f"文本量不足（{n_han} 字 / {n_clauses} 句），暂不统计三环行为覆盖。",
                },
            ],
        }

    verify_hits = 0
    matched_verify: list[str] = []
    for pat in _VERIFICATION_PATTERNS:
        cnt = text.count(pat)
        if cnt > 0:
            verify_hits += cnt
            matched_verify.append(f"{pat}×{cnt}")
    verify_rate = round(verify_hits / max(n_han, 1) * 100.0, 4)
    verify_conf = confidence_for(n_han, n_han, eff_min)

    text_ext = base_text_ext or compute_metacognition_text_extensions(
        session_id, logs=logs, min_chars=eff_min
    )
    ind_map = {
        str(row.get("name")): row
        for row in (text_ext.get("indicators") or [])
        if isinstance(row, dict)
    }
    has_monitor = float((ind_map.get("counter_premise_marker_rate") or {}).get("value") or 0.0) > 0.0
    has_verify = verify_hits > 0
    has_debug = float((ind_map.get("revision_anchor_density") or {}).get("value") or 0.0) > 0.0

    loop_checks = [
        ("前提质疑监测", has_monitor),
        ("信息来源与口径核验", has_verify),
        ("指错修正排错闭环", has_debug),
    ]
    covered_labels = [lbl for lbl, ok in loop_checks if ok]
    missing_labels = [lbl for lbl, ok in loop_checks if not ok]
    n_loops = len(covered_labels)
    balance_ratio = round(n_loops / 3.0, 4)
    balance_conf = confidence_for(n_han, n_han, eff_min)

    return {
        "available": True,
        "source_chars": n_han,
        "matched_verification_terms": matched_verify[:10],
        "covered_loops_labels": covered_labels,
        "missing_loops_labels": missing_labels,
        "gap": None,
        "indicators": [
            {
                "name": "evidence_verification_marker_rate",
                "label": "信息核验与来源口径求证标记密度",
                "unit": "次/百字",
                "value": verify_rate,
                "n_valid": n_han,
                "n_total": n_han,
                "confidence": verify_conf,
                "gap": None,
                "summary": (
                    f"共检出 {verify_hits} 处事实溯源、样本对照组或口径核验表达，"
                    f"合 {verify_rate:.2f} 次/百字（样本 {n_han} 汉字）。"
                ),
            },
            {
                "name": "metacognitive_regulation_balance",
                "label": "监测—核验—排错三环行为覆盖度",
                "unit": "比例",
                "value": balance_ratio,
                "n_valid": n_loops,
                "n_total": 3,
                "confidence": balance_conf,
                "gap": None,
                "summary": (
                    f"在可观测「监测、核验、排错」三个调控环节中观察到 {n_loops} / 3 个环节"
                    f"（已呈现：{'、'.join(covered_labels) or '暂无'}；"
                    f"待观察：{'、'.join(missing_labels) or '全部已覆盖'}）。"
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
        meta_reg = compute_metacognition_regulation_extensions(
            session_id, logs=logs, base_text_ext=meta_text
        )
        stages = event_windows.extract_stage_windows(session_id)
        all_meta_inds = list(meta_text.get("indicators") or []) + list(
            meta_reg.get("indicators") or []
        )
        return {
            "metacognition_text": meta_text,
            "metacognition_regulation": meta_reg,
            "all_metacognition_indicators": all_meta_inds,
            "stage_windows_observed": {
                "available": len(stages) > 0,
                "count": len(stages),
                "stages": stages,
            },
            "stage_focus": ["stage_read_ai", "stage_critique", "stage_revise"],
        }
    if scene_profile_id == "longitudinal_followup":
        kin_summary: dict[str, Any] = {
            "available": False,
            "note": "个体内多次访次纵向比较请结合匿名随访档案与首访基线查询。",
        }
        if isinstance(feature_v2, dict) and feature_v2.get("available"):
            tracks = feature_v2.get("tracks") or {}
            face_kin = tracks.get("face_kinematics") or {}
            gest_sm = tracks.get("gesture_smoothness") or {}
            pose_mode = str(face_kin.get("pose_mode") or "unavailable")
            ang_vel = (
                face_kin.get("head_angular_velocity_3d_p50")
                if pose_mode == "3d_matrix"
                else None
            )
            sparc = gest_sm.get("sparc")
            kin_summary = {
                "available": True,
                "pose_mode": pose_mode,
                "face_3d_angular_vel_median": ang_vel,
                "gesture_sparc_smoothness": sparc,
                "note": "当前单场运动学摘要可作为匿名纵向随访档案的单次访次快照。",
            }
        return {"longitudinal_snapshot": kin_summary}
    return {}
