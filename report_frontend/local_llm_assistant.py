# report_frontend/local_llm_assistant.py
"""严格隔离的按需本地辅助模块（可读行为观察报告 / 分场景追问提纲 / 题库检索）。

不可逾越的边界约束：
1. 本模块**绝不参与**任何物理指标计算、模型判定或维度汇总；
2. 严禁生成任何分数、分档、排名、招录筛选或临床医学推断；所有本地大模型返回内容必须通过
   `redline_guard.find_assistant_violations()` 二次审查，命中即立刻回退至确定性规则报告；
3. 本地大模型（OpenAI 兼容接口，默认 `http://127.0.0.1:8081/v1`，默认模型 `Qwen2.5-14B`）
   仅在按需请求时介入文本润色；离线、超时或未启用时，可在毫秒级直接输出结构完整的
   确定性多段落可读文字报告，绝不暴露晦涩工程计数或机器行话。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any

import redline_guard

DEFAULT_LLM_BASE_URL = "http://127.0.0.1:8081/v1"
DEFAULT_LLM_MODEL = "Qwen2.5-14B"
DEFAULT_LLM_TIMEOUT_SEC = 12.0
NARRATIVE_LLM_TIMEOUT_SEC = 70.0

ASSISTANT_WATERMARK = (
    "由本地辅助模块生成，仅作文本可读化与追问/检索参考，不参与任何指标计算与评定。"
)

AUDIENCE_LABELS: dict[str, str] = {
    "narrative_review": "导师与复核者循证观察版",
    "participant_feedback": "参与者个人成长反馈版",
    "interview_guide": "人工追问与实训提纲",
    "summary": "过程观察摘要版",
}

_CACHE_LOCK = threading.Lock()
_NARRATIVE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_CACHE_TTL_SEC = 600.0
_CACHE_MAX_ITEMS = 64

# 纯中文友好指标描述映射，确保绝不向文字报告泄漏 snake_case 内部键名
_INDICATOR_SPEC: dict[str, tuple[str, str, str]] = {
    "speech_rate": ("言语组织", "陈述语速", "字/秒"),
    "answer_detail_chars_p50": ("言语组织", "单题平均阐述篇幅", "字"),
    "discourse_marker_rate": ("言语组织", "因果与转折连接词密度", "次/百字"),
    "lexical_diversity": ("言语组织", "用词丰富度", ""),
    "self_repair_rate": ("言语组织", "表达自我修正频率", "次/百字"),
    "response_latency_sec": ("语音韵律", "提问后首句组织反应时间", "秒"),
    "voiced_ratio": ("语音韵律", "有效发声时长占比", "%ratio"),
    "pause_structure_entropy": ("语音韵律", "停顿节律均匀度", ""),
    "pause_duration_p50": ("语音韵律", "典型语句停顿中位时长", "秒"),
    "pitch_variability_per_sec": ("语音韵律", "语调音高每秒起伏幅度", "Hz/秒"),
    "gaze_stability": ("视线体态", "视线水平保持稳定度", ""),
    "gaze_dwell_p50": ("视线体态", "前方注视连续停留中位时长", "秒"),
    "gesture_activity_per_sec": ("视线体态", "上肢动作每秒活跃幅度", "单位/秒"),
    "hand_openness_p50": ("视线体态", "手势平均开合幅度", ""),
    "smile_duration_p50": ("视线体态", "微笑表现持续中位时长", "秒"),
}

_FALLBACK_INDICATOR_LABELS: dict[str, str] = {
    "smile_ratio": "微笑表达帧占比",
    "smile_rate_per_min": "微笑出现频次",
    "gaze_aversion_ratio_vs_baseline": "视线偏离基线占比",
    "gaze_aversion_rate_per_min_vs_baseline": "视线偏离频次",
    "blink_rate_per_min": "每分钟眨眼频次",
    "blink_duration_mean": "单次眨眼平均时长",
    "head_motion_per_sec": "头部运动幅度",
    "head_away_ratio_vs_baseline": "头部偏转偏离占比",
    "brow_furrow_ratio_vs_baseline": "眉间微动变化占比",
    "lip_compression_ratio_vs_baseline": "唇部紧抿变化占比",
    "au_activation_any_ratio": "面部动作活跃帧占比",
    "au_variability_per_sec": "面部动作每秒变化幅度",
    "facial_asymmetry_p90": "面部左右协同差值",
    "gesture_burst_rate_per_min_vs_baseline": "明显手势爆发频次",
    "posture_stability": "躯干肩部稳定性",
    "postural_shift_rate_per_min_vs_baseline": "坐姿或重心调整频次",
    "shrug_elevation_p90": "肩部抬升变化幅度",
    "hand_visible_ratio_l": "左手进入画面占比",
    "hand_visible_ratio_r": "右手进入画面占比",
    "pose_valid_ratio": "姿态有效识别帧占比",
    "pause_rate_per_min": "每分钟停顿频次",
    "pause_duration_p90": "长停顿持续时长",
    "pitch_median": "基频音高中位数",
    "pitch_range_semitone": "语调音高跨度",
    "energy_median": "发声响度中位数",
    "energy_dynamics_per_sec": "音量每秒变化幅度",
}


def _llm_thinking_enabled() -> bool:
    val = str(os.getenv("JX_LLM_THINKING", "0")).strip().lower()
    return val in ("1", "true", "yes", "on")


def _llm_config() -> tuple[str, str, float]:
    base = (os.getenv("JX_LLM_BASE_URL") or DEFAULT_LLM_BASE_URL).rstrip("/")
    model = (os.getenv("JX_LLM_MODEL") or DEFAULT_LLM_MODEL).strip()
    try:
        timeout = max(0.5, min(60.0, float(os.getenv("JX_LLM_TIMEOUT_SEC", str(DEFAULT_LLM_TIMEOUT_SEC)))))
    except ValueError:
        timeout = DEFAULT_LLM_TIMEOUT_SEC
    return base, model, timeout


def _call_local_openai_chat(
    system_prompt: str,
    user_prompt: str,
    *,
    timeout: float | None = None,
    max_tokens: int = 1200,
) -> tuple[str | None, str | None, str]:
    """调用本地 OpenAI 兼容模型（绕过 http_proxy），返回 `(文本, 失败原因, 模型名)`。"""
    base, model, default_timeout = _llm_config()
    eff_timeout = default_timeout if timeout is None else float(timeout)
    url = f"{base}/chat/completions"
    body = json.dumps({
        "model": model,
        "temperature": 0.25,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": _llm_thinking_enabled()},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=eff_timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        choices = data.get("choices") or []
        if not choices:
            return None, "本地模型未返回有效文本内容", model
        msg = choices[0].get("message") or {}
        content = str(msg.get("content") or "").strip()
        if not content:
            return None, "本地模型返回空内容", model
        return content, None, model
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return None, f"本地模型暂未连接或响应超时（{type(exc).__name__}）", model


def _finite(val: Any) -> float | None:
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        f = float(val)
        return f if math.isfinite(f) else None
    return None


def _fmt_value(val: float, unit: str) -> str:
    if unit == "%ratio":
        return f"{val * 100:.1f}%"
    a = abs(val)
    if a >= 100:
        num = f"{val:.1f}"
    elif a >= 1:
        num = f"{val:.2f}"
    elif a >= 0.001 or val == 0:
        num = f"{val:.3f}".rstrip("0").rstrip(".") if val != 0 else "0"
    else:
        num = f"{val:.2e}"
    return f"{num} {unit}".strip()


def _normalize_mode(mode: str | None) -> str:
    m = str(mode or "").strip()
    if m in ("narrative_review", "participant_feedback", "interview_guide", "summary"):
        return m
    return "summary"


def _build_behavioral_dossier(
    payload: dict[str, Any],
    scene_profile_id: str | None = None,
) -> dict[str, Any]:
    """从完整的行为报告载荷中抽取脱敏、人话化、分主题的行为事实档案。

    同时兼容完整 `build_l1_report()` 载荷与单元测试构造的最小合成载荷。
    """
    cov = payload.get("coverage") or {}
    l1 = payload.get("l1") or {}
    dims = payload.get("dimensions") or {}
    sp = payload.get("scene_profile") or {}
    resolved_scene_id = str(scene_profile_id or sp.get("id") or "research_interview")
    scene_name = str(sp.get("name") or "研思循证 · 科研行为过程分析")
    scene_banner = str(sp.get("compliance_banner") or "")
    scene_theory = (sp.get("theoretical_foundation") or {}).get("framework") or ""

    n_passed = cov.get("n_passed", 0)
    n_measurable = cov.get("n_measurable")
    n_slots = cov.get("n_slots", 14)
    confidence_cap = str(cov.get("confidence_cap") or "待核验")

    lead_raw = l1.get("lead_summary") or {}
    lead_text = (
        lead_raw.strip()
        if isinstance(lead_raw, str)
        else str((lead_raw or {}).get("headline") or "").strip()
        if isinstance(lead_raw, dict)
        else ""
    )

    # 1. 维度覆盖与缺口（兼容真实 display_name/matched_indicators 与测试合成 name/coverage）
    dim_rows: list[dict[str, Any]] = []
    thin_dims: list[str] = []
    gap_list: list[str] = []
    for key, d in dims.items():
        if not isinstance(d, dict):
            continue
        name = str(d.get("display_name") or d.get("name") or key)
        matched = str(d.get("matched_indicators") or "").strip()
        incomplete = False
        if "/" in matched:
            try:
                p_s, t_s = matched.split("/", 1)
                incomplete = int(p_s.strip()) < int(t_s.strip())
            except ValueError:
                incomplete = False
        elif isinstance(d.get("coverage"), dict):
            c = d["coverage"]
            p_n = c.get("n_passed", 0)
            t_n = c.get("n_slots", 0)
            matched = f"{p_n}/{t_n}"
            incomplete = int(p_n or 0) < int(t_n or 0)
        conf = str(d.get("confidence") or "待核验")
        if incomplete or conf in ("不足", "低"):
            thin_dims.append(name)
        gaps = [str(g).strip() for g in (d.get("evidence_gaps") or []) if isinstance(g, str) and g.strip()]
        for g in gaps[:3]:
            gap_list.append(f"{name}：{g}")
        dim_rows.append({
            "name": name,
            "matched": matched,
            "confidence": conf,
            "gaps": gaps[:3],
        })

    # 2. 核心 41 项实测值映射
    ind_map: dict[str, dict[str, Any]] = {}
    for row in l1.get("indicators") or []:
        if isinstance(row, dict) and row.get("name"):
            ind_map[str(row["name"])] = row
    ind_meta = l1.get("indicator_meta") or {}

    grouped: dict[str, list[dict[str, Any]]] = {
        "言语组织": [],
        "语音韵律": [],
        "视线体态": [],
    }
    raw_metrics: dict[str, float] = {}
    for k, (group_name, cn_label, unit_fmt) in _INDICATOR_SPEC.items():
        item = ind_map.get(k)
        if not item:
            continue
        val = _finite(item.get("value"))
        if val is None:
            continue
        raw_metrics[k] = val
        grouped[group_name].append({
            "label": cn_label,
            "value": val,
            "display": _fmt_value(val, unit_fmt),
            "confidence": str(item.get("confidence") or "中"),
        })

    # 3. 二层精细动力学（真 3D 头姿角速度、声学谐噪比、动作平滑度）
    fv2 = l1.get("feature_layer_v2") or {}
    if isinstance(fv2, dict) and fv2.get("available"):
        tracks = fv2.get("tracks") or {}
        face_kin = tracks.get("face_kinematics") or {}
        pose_mode = str(face_kin.get("pose_mode") or "unavailable")
        if pose_mode == "3d_matrix":
            ang3d = _finite(face_kin.get("head_angular_velocity_3d_p50"))
            if ang3d is not None:
                raw_metrics["head_angular_velocity_3d_p50"] = ang3d
                grouped["视线体态"].append({
                    "label": "头部三维转动角速度中位数",
                    "value": ang3d,
                    "display": f"{ang3d:.1f} 度/秒",
                    "confidence": "高",
                })
        voice_dyn = tracks.get("voice_dynamics") or {}
        hnr = _finite(voice_dyn.get("hnr_db_median"))
        if hnr is not None:
            raw_metrics["hnr_db_median"] = hnr
            grouped["语音韵律"].append({
                "label": "语音谐噪比中位数（清晰度参照）",
                "value": hnr,
                "display": f"{hnr:.1f} dB",
                "confidence": "高",
            })
        gest_sm = tracks.get("gesture_smoothness") or {}
        sparc = _finite(gest_sm.get("sparc"))
        if sparc is not None:
            raw_metrics["gesture_smoothness"] = sparc
            grouped["视线体态"].append({
                "label": "上肢动作连续平滑度",
                "value": sparc,
                "display": f"{sparc:.2f}",
                "confidence": "高",
            })
        dual_hand = _finite(gest_sm.get("dual_hand_visible_ratio"))
        if dual_hand is not None:
            raw_metrics["dual_hand_visible_ratio"] = dual_hand
            grouped["视线体态"].append({
                "label": "双手同框自然入镜占比",
                "value": dual_hand,
                "display": f"{dual_hand * 100:.1f}%",
                "confidence": "高",
            })

    # 4. 逐题变化亮点
    dispersion = l1.get("dispersion") or {}
    disp_highlights: list[str] = []
    if isinstance(dispersion, dict):
        scored_disp: list[tuple[float, str, dict[str, Any]]] = []
        for k, d_info in dispersion.items():
            if not isinstance(d_info, dict):
                continue
            iqr = _finite(d_info.get("iqr"))
            if iqr is not None and iqr > 0:
                meta_lbl = (ind_meta.get(k) or {}).get("label") if isinstance(ind_meta.get(k), dict) else None
                cn_lbl = (
                    meta_lbl
                    or (_INDICATOR_SPEC.get(k) or (None, None, None))[1]
                    or _FALLBACK_INDICATOR_LABELS.get(k)
                )
                if cn_lbl:
                    scored_disp.append((iqr, str(cn_lbl), d_info))
        scored_disp.sort(key=lambda x: x[0], reverse=True)
        for _iqr, cn_lbl, d_info in scored_disp[:3]:
            mn = _finite(d_info.get("min"))
            md = _finite(d_info.get("median"))
            mx = _finite(d_info.get("max"))
            nq = d_info.get("n_questions")
            if mn is not None and md is not None and mx is not None:
                disp_highlights.append(
                    f"{cn_lbl}在 {nq or '多'} 道题目间中位数为 {md:.2f}，波动区间 [{mn:.2f}, {mx:.2f}]"
                )

    # 5. 采集质量、跨通道同步、问答隔离与转写置信度
    sq = l1.get("signal_quality_audit") or {}
    cm = l1.get("cross_modal_coverage") or {}
    st = l1.get("speaker_turn_audit") or {}
    asr_disc = l1.get("asr_confidence_disclosure") or {}
    win_info = l1.get("windows") or {}

    # 6. 行为阶段与模型复核转交
    br = l1.get("behavior_regimes") or {}
    rm = l1.get("referee_ml") or {}
    eg = rm.get("evidence_gate") or {}
    ss = rm.get("stage_segmentation") or {}
    macro_stages = [
        {
            "label": str(s.get("label") or ""),
            "annotation": str(s.get("annotation") or ""),
            "duration_sec": _finite(s.get("duration_sec")),
            "ratio": _finite(s.get("ratio")),
        }
        for s in (ss.get("macro_stages") or [])
        if isinstance(s, dict) and s.get("label")
    ]
    change_points = [
        {
            "t_sec": _finite(cp.get("t_sec")),
            "cues": [str(c) for c in (cp.get("cues") or [])[:2]],
        }
        for cp in (ss.get("change_points") or [])
        if isinstance(cp, dict)
    ]

    pos_cues: list[str] = []
    weak_cues: list[str] = []
    for attr in eg.get("attributions") or []:
        if not isinstance(attr, dict) or not attr.get("label"):
            continue
        desc = f"{attr.get('label')}（{attr.get('value_display', '')}）".strip()
        direction = str(attr.get("direction") or "")
        if "待" in direction or "弱" in direction or "缺" in direction:
            weak_cues.append(desc)
        else:
            pos_cues.append(desc)

    # 7. 场景专属扩展（如元智审辨 4 指标）
    ext = sp.get("extensions") or {}
    meta_inds: list[dict[str, Any]] = []
    covered_loops: list[str] = []
    missing_loops: list[str] = []
    if isinstance(ext, dict):
        for blk_key in ("metacognition_text", "metacognition_regulation"):
            blk = ext.get(blk_key) or {}
            if isinstance(blk, dict):
                for ind in blk.get("indicators") or []:
                    if isinstance(ind, dict) and ind.get("label"):
                        meta_inds.append({
                            "label": str(ind["label"]),
                            "value": ind.get("value"),
                            "unit": str(ind.get("unit") or ""),
                            "confidence": str(ind.get("confidence") or "无"),
                            "summary": str(ind.get("summary") or ""),
                        })
                if blk_key == "metacognition_regulation":
                    covered_loops = list(blk.get("covered_loops_labels") or [])
                    missing_loops = list(blk.get("missing_loops_labels") or [])

    return {
        "scene": {
            "id": resolved_scene_id,
            "name": scene_name,
            "theory_framework": scene_theory,
            "compliance_banner": scene_banner,
            "meta_indicators": meta_inds,
            "covered_loops": covered_loops,
            "missing_loops": missing_loops,
            "longitudinal_snapshot": ext.get("longitudinal_snapshot") if isinstance(ext, dict) else None,
        },
        "reliability": {
            "n_passed": n_passed,
            "n_measurable": n_measurable if isinstance(n_measurable, int) else n_slots,
            "n_slots": n_slots,
            "confidence_cap": confidence_cap,
            "n_questions": win_info.get("n_questions"),
            "lead_text": lead_text,
            "signal_usable_ratio": _finite(sq.get("usable_window_ratio")),
            "signal_summary": str(sq.get("summary_plain") or ""),
            "cross_modal_summary": str(cm.get("summary_plain") or ""),
            "cross_modal_status": cm.get("status"),
            "speaker_turn_summary": str(st.get("summary_plain") or ""),
            "asr_note": str(asr_disc.get("note_plain") or ""),
        },
        "dimensions": dim_rows,
        "thin_dims": thin_dims,
        "gaps": gap_list[:5],
        "grouped_metrics": grouped,
        "raw_metrics": raw_metrics,
        "dispersion_highlights": disp_highlights,
        "stage_flow": {
            "regimes_summary": str(br.get("summary_plain") or ""),
            "gate_label": str(eg.get("stage_label") or ""),
            "uncertainty_escalated": bool(eg.get("uncertainty_escalated")),
            "ood_anomaly_flag": bool(eg.get("ood_anomaly_flag")),
            "macro_stages": macro_stages[:4],
            "change_points": change_points[:3],
            "stage_seg_summary": str(ss.get("summary_plain") or ""),
            "pos_cues": pos_cues[:4],
            "weak_cues": weak_cues[:3],
        },
    }


def _describe_verbal_evidence(dossier: dict[str, Any], friendly: bool = False) -> str:
    rm = dossier["raw_metrics"]
    bits: list[str] = []
    sr = rm.get("speech_rate")
    detail = rm.get("answer_detail_chars_p50")
    dm = rm.get("discourse_marker_rate")
    lex = rm.get("lexical_diversity")
    rep = rm.get("self_repair_rate")

    if sr is not None:
        if sr >= 4.8:
            pace = "语速相对偏快"
        elif sr >= 3.2:
            pace = "语速稳健适中"
        else:
            pace = "语速较为从容"
        bits.append(f"在语言组织与表达篇幅上，陈述语速约 {sr:.1f} 字/秒（{pace}）")
    if detail is not None:
        detail_desc = "展开较为充分" if detail >= 120 else ("阐述偏简练" if detail >= 45 else "单题字数较短")
        bits.append(f"单题平均回答篇幅约 {detail:.0f} 字（{detail_desc}）")
    if dm is not None:
        dm_desc = "条理衔接清晰" if dm >= 2.0 else "因果与转折连接词偏少"
        bits.append(f"逻辑与因果连接词使用频率约为每百字 {dm:.1f} 次（{dm_desc}）")
    if lex is not None:
        lex_tag = "用词多样性较高" if lex >= 0.75 else ("用词较稳健" if lex >= 0.5 else "用词偏重复")
        bits.append(f"词汇多样性系数为 {lex:.2f}（{lex_tag}）")
    if rep is not None and rep > 0:
        bits.append(f"自我纠偏与重述修正频率约为每百字 {rep:.1f} 次")

    meta_inds = dossier["scene"]["meta_indicators"]
    measured_meta = [m for m in meta_inds if _finite(m.get("value")) is not None]
    if dossier["scene"]["id"] == "ai_metacognition" and measured_meta:
        m_parts = [f"{m['label']}为 {m['value']}{m['unit']}".strip() for m in measured_meta[:3]]
        bits.append("结合「元智审辨」场景文本规则观测，" + "，".join(m_parts))
        cov_l = dossier["scene"]["covered_loops"]
        mis_l = dossier["scene"]["missing_loops"]
        if cov_l or mis_l:
            bits.append(
                f"元认知三环中已呈现{'、'.join(cov_l) or '待积累'}；"
                f"{'尚待补充展示' + '、'.join(mis_l) if mis_l else '三个关键环节均有清晰体现'}"
            )

    if not bits:
        return (
            "本场尚未积累足够长度的语音转写内容，因此语言衔接词密度与单题阐述篇幅暂未生成数值。"
            if not friendly
            else "本次记录暂未采集到完整清晰的语音文字内容，所以语速和表达结构特征还无法展现；后续采集时靠近麦克风完整回答即可自动补全。"
        )
    return "，".join(bits) + "。"


def _describe_vocal_posture_evidence(dossier: dict[str, Any], friendly: bool = False) -> str:
    rm = dossier["raw_metrics"]
    voice_bits: list[str] = []
    rl = rm.get("response_latency_sec")
    vr = rm.get("voiced_ratio")
    pe = rm.get("pause_structure_entropy")
    pd_med = rm.get("pause_duration_p50")
    pv = rm.get("pitch_variability_per_sec")
    hnr = rm.get("hnr_db_median")

    if rl is not None:
        rl_desc = "思考切入迅速" if rl <= 2.5 else ("有一定构思缓冲" if rl <= 5.0 else "首句组织停顿略长")
        voice_bits.append(f"提问结束后首句起讲平均缓冲约 {rl:.2f} 秒（{rl_desc}）")
    if vr is not None:
        voice_bits.append(f"有效出声作答时间占答题窗口的 {vr * 100:.1f}%")
    if pe is not None:
        pe_desc = "句间停顿分布较均衡" if pe >= 0.72 else "停顿长短分布有所起伏"
        voice_bits.append(f"停顿节律均匀度为 {pe:.2f}（{pe_desc}）")
    if pd_med is not None:
        voice_bits.append(f"典型停顿时长中位数约为 {pd_med:.2f} 秒")
    if pv is not None:
        voice_bits.append(f"语调音高每秒变化幅度为 {pv:.1f} Hz/秒")
    if hnr is not None:
        voice_bits.append(f"人声谐噪比中位数达 {hnr:.1f} dB，收音清晰度较好")

    posture_bits: list[str] = []
    gs = rm.get("gaze_stability")
    gd = rm.get("gaze_dwell_p50")
    ang = rm.get("head_angular_velocity_3d_p50")
    ga = rm.get("gesture_activity_per_sec")
    dh = rm.get("dual_hand_visible_ratio")

    if gs is not None:
        gs_desc = "视角指向稳定" if gs >= 0.75 else "视线有左右扫视移动"
        posture_bits.append(f"视觉端观测到水平视线稳定度为 {gs:.2f}（{gs_desc}）")
    if gd is not None:
        posture_bits.append(f"正向专注停留中位时长约 {gd:.2f} 秒")
    if ang is not None:
        ang_desc = "头部姿态从容平稳" if ang <= 18.0 else "头部伴随较多辅助转向"
        posture_bits.append(f"三维头部转动角速度中位数为 {ang:.1f} 度/秒（{ang_desc}）")
    if ga is not None:
        posture_bits.append(f"上肢手势每秒活跃度为 {ga:.2f}")
    if dh is not None and dh > 0:
        posture_bits.append(f"双手自然进入画面视野的时间占比约 {dh * 100:.1f}%")

    parts: list[str] = []
    if voice_bits:
        parts.append("在声音韵律方面，" + "，".join(voice_bits) + "。")
    else:
        parts.append(
            "本场声学片段有效时长偏短，声学韵律指标暂留空缺。"
            if not friendly
            else "本次声音采样偏短，语速停顿与语调变化暂未完全显现。"
        )
    if posture_bits:
        parts.append("在视线与体态协同方面，" + "，".join(posture_bits) + "。")
    else:
        parts.append(
            "由于部分视觉帧缺失或采用无影像隐私挡位，面部注视与上肢动作特征未作推算并保持显式空缺。"
            if not friendly
            else "由于本次采用仅语音/无影像模式或画面遮挡较多，眼神与手势特征未计入观察。"
        )
    return "".join(parts)


def _describe_stages_and_limits(dossier: dict[str, Any], friendly: bool = False) -> str:
    sf = dossier["stage_flow"]
    rel = dossier["reliability"]
    parts: list[str] = []

    stages = sf["macro_stages"]
    cps = sf["change_points"]
    if stages:
        st_desc = "；".join(
            f"「{s['label']}」（约 {s['duration_sec']:.0f} 秒，{s['annotation']}）"
            if s.get("duration_sec") is not None and s.get("annotation")
            else f"「{s['label']}」"
            for s in stages[:3]
        )
        parts.append(f"从整场过程演进看，行为状态可自然梳理为 {st_desc}。")
    elif sf["regimes_summary"]:
        parts.append(sf["regimes_summary"])

    if cps:
        cp0 = cps[0]
        if cp0.get("t_sec") is not None:
            cue_str = f"（主要表现为{'、'.join(cp0['cues'])}）" if cp0.get("cues") else ""
            parts.append(f"在第 {cp0['t_sec']:.0f} 秒前后观察到明显的状态节奏切换{cue_str}。")

    if dossier["dispersion_highlights"]:
        parts.append("逐题变化显示，" + dossier["dispersion_highlights"][0] + "。")

    lim_bits: list[str] = []
    if rel["signal_usable_ratio"] is not None and rel["signal_usable_ratio"] < 0.75:
        lim_bits.append(
            f"采录画面与信号稳定窗口占比约 {rel['signal_usable_ratio'] * 100:.0f}%，复核时宜留意受遮挡或停顿空白片段"
        )
    if dossier["gaps"]:
        lim_bits.append(f"主要采集缺口包括：{'；'.join(dossier['gaps'][:2])}")
    if not friendly and sf["uncertainty_escalated"]:
        lim_bits.append("核验状态已提示需要结合人工复核重点查看关键作答片段")
    if lim_bits:
        parts.append("需要同步留意的边界条件：" + "；".join(lim_bits) + "。")
    elif not parts:
        parts.append("本场整体过程状态平稳，各通道信号无异常波动，未见必须额外排查的明显缺口。")

    return "".join(parts)


def _describe_recommendations(dossier: dict[str, Any], mode: str) -> str:
    scene_id = dossier["scene"]["id"]
    thin_dims = dossier["thin_dims"]
    rm = dossier["raw_metrics"]
    if mode == "participant_feedback":
        tips: list[str] = []
        if rm.get("discourse_marker_rate") is not None and rm["discourse_marker_rate"] < 2.0:
            tips.append("表达方案或观点时，可以有意识多用「前提在于」「相比之下」「因此需要」等结构连接词，让听者一眼抓住论证主干")
        if scene_id == "ai_metacognition":
            mis = dossier["scene"]["missing_loops"]
            if mis:
                tips.append(f"在后续人机协同练习中，建议重点补齐「{'、'.join(mis)}」环节，做到既敢质疑假设、又能给出证据出处和替代方案")
            else:
                tips.append("在后续求职与实训练习中，可进一步针对目标岗位的核心能力条款准备 1 至 2 个具体量化反例")
        elif scene_id == "longitudinal_followup":
            tips.append("多次记录重在观察自身节律随时间的变化趋势，每次采录保持相对固定的光照与麦距，有助于提高跨访次可比性")
        else:
            tips.append("面对追问时先给核心假设与对照组结论、再展开细节推导，并注意保持正对镜头的稳定视线")
        if thin_dims:
            tips.append(f"本场在「{'、'.join(thin_dims[:2])}」方面有效素材较少，下次练习可确保摄像头视野清晰与连续开口阐述")
        return "；".join(tips) + "。"

    if scene_id == "ai_metacognition":
        mis = dossier["scene"]["missing_loops"]
        mis_note = f"当前文本显示「{'、'.join(mis)}」呈现相对偏弱，" if mis else ""
        return (
            f"{mis_note}建议复核或实训导师结合目标岗位要求，重点追问对方对生成内容前提假设的核查路径、数据口径比对方式以及修正方案的可验证性。"
            "（岗位对照仅呈现行为证据与待补齐缺口，不出具筛选决定。）"
        )
    if scene_id == "longitudinal_followup":
        return (
            "本场已记录核心声学与三维姿态动力学读数（无原始影像落盘），建议纳入同编号后续随访序列，"
            "基于本人首访基线与稳健离散区间观察跨期自然波动。"
        )
    dim_focus = (
        f"建议导师优先围绕「{'、'.join(thin_dims[:2])}」安排简短补充追问，"
        if thin_dims
        else "建议导师结合原始转写文本，"
    )
    return f"{dim_focus}围绕关键实验设计对照、异常现象归因与失败排查细节开展面对面学术核验。"


def _deterministic_narrative_sections(
    dossier: dict[str, Any],
    mode: str,
) -> list[dict[str, str]]:
    """根据完整行为档案生成自然流畅、分段清晰、完全不含工程行话的可读文字报告。"""
    rel = dossier["reliability"]
    sf = dossier["stage_flow"]
    scene = dossier["scene"]
    dim_rows = dossier["dimensions"]

    n_p = rel["n_passed"]
    n_m = rel["n_measurable"] or rel["n_slots"]
    q_cnt = rel["n_questions"]
    q_phrase = f"围绕 {q_cnt} 道题目" if q_cnt else "围绕本场问答流程"

    if mode == "participant_feedback":
        s1_body = (
            f"在本次「{scene['name']}」体验中，{q_phrase}系统顺利采集了你的多模态过程表现。"
            f"在 {n_m} 项可观测关键指标里，有 {n_p} 项获取到了完整有效的数据，整体数据可信度为「{rel['confidence_cap']}」。"
        )
        if rel["lead_text"]:
            s1_body += f"总体来看，{rel['lead_text']}"

        solid_bits: list[str] = []
        for row in dim_rows:
            if row["confidence"] in ("高", "中") and row["name"] not in dossier["thin_dims"]:
                solid_bits.append(row["name"])
        s2_header = (
            f"你在「{'、'.join(solid_bits)}」方面呈现出较扎实的实测记录。"
            if solid_bits
            else "以下是本次采集中已清晰记录到的个人表达特征。"
        )
        s2_body = s2_header + _describe_verbal_evidence(dossier, friendly=True)

        s3_body = _describe_vocal_posture_evidence(dossier, friendly=True)
        if sf["pos_cues"]:
            s3_body += f"其中较为突出的正向特征包括{'、'.join(sf['pos_cues'][:2])}。"

        s4_body = _describe_stages_and_limits(dossier, friendly=True)
        s5_body = _describe_recommendations(dossier, "participant_feedback")
        return [
            {"title": "一、本次过程整体印象", "body": s1_body},
            {"title": "二、表达组织与思考亮点", "body": s2_body},
            {"title": "三、声音节奏与体态自然度", "body": s3_body},
            {"title": "四、过程起伏与环境限制说明", "body": s4_body},
            {"title": "五、下一步可操作的练习建议", "body": s5_body},
        ]

    if mode == "interview_guide":
        return _deterministic_guide_sections(dossier)

    # narrative_review (默认完整导师与复核者循证观察报告)
    dim_overview = "；".join(
        f"{r['name']}（{r['matched']} 项有实测，可信度 {r['confidence']}）"
        for r in dim_rows[:5]
    )
    s1_body = (
        f"本报告基于「{scene['name']}」场景档案生成，{q_phrase}共对 {n_m} 项可观测核心关键点开展核验，"
        f"其中 {n_p} 项取得有效实测数据，整体证据可信度上限为「{rel['confidence_cap']}」。"
    )
    if dim_overview:
        s1_body += f"分维度覆盖情况如下：{dim_overview}。"
    if rel["speaker_turn_summary"]:
        s1_body += f"主试提问与受试作答分离情况：{rel['speaker_turn_summary']}"

    s2_body = _describe_verbal_evidence(dossier, friendly=False)
    s3_body = _describe_vocal_posture_evidence(dossier, friendly=False)
    s4_body = _describe_stages_and_limits(dossier, friendly=False)
    if sf["gate_label"]:
        esc = "当前建议结合人工重点复核" if sf["uncertainty_escalated"] else "整体位于稳态证据核验集合内"
        s4_body += f"综合过程核验阶段为「{sf['gate_label']}」（{esc}）。"
    s5_body = _describe_recommendations(dossier, "narrative_review")
    return [
        {"title": "一、采录概况与证据可信度", "body": s1_body},
        {"title": "二、言语组织与思维展开线索", "body": s2_body},
        {"title": "三、声学韵律、注视与体态协同", "body": s3_body},
        {"title": "四、过程节律演进与采集边界", "body": s4_body},
        {"title": "五、分场景复核与后续引导建议", "body": s5_body},
    ]


def _deterministic_guide_sections(dossier: dict[str, Any]) -> list[dict[str, str]]:
    """依据真实薄弱维度与缺口生成可读追问/实训提纲（保留缺口维度命名）。"""
    thin = dossier["thin_dims"]
    gaps = dossier["gaps"]
    cm_status = dossier["reliability"]["cross_modal_status"]
    scene_id = dossier["scene"]["id"]

    if thin:
        q1 = (
            f"针对本场有效依据相对偏薄的维度（{'、'.join(thin[:2])}），建议围绕受试者亲自承担过的核心任务，"
            "追问当时为何选择该技术路线、对照基线如何设定以及方案修改时的具体取舍依据，以弥补单场采集的片段局限。"
        )
    else:
        q1 = (
            "本场各观测维度覆盖较为完整，建议跳过基础概念复述，直接切入研究工作中的真实失败排查、反常样本现象与实验边界追问。"
        )

    if gaps:
        q2 = (
            f"针对本次记录存在的采集局限（{gaps[0][:90]}），请复核人员明确区分「设备或光线导致的信号漏采」"
            "与「真实表达迟疑」，必要时补充一题短时现场演示或口头复盘。"
        )
    else:
        q2 = (
            "本场信号连续性良好，可请对方在 2 分钟内脱离屏幕文稿口头复述最核心的一项结论及反向验证思路，"
            "观察无提示条件下的条理性与反应衔接。"
        )

    if scene_id == "ai_metacognition":
        mis = dossier["scene"]["missing_loops"]
        mis_tip = f"（本场尚欠缺{'、'.join(mis)}）" if mis else ""
        q3 = (
            f"结合「元智审辨」场景特征{mis_tip}，可设计一道带有隐蔽统计学偏误或样本泄露的 AI 生成回答，"
            "请学习者依次执行「指错位置 → 列举反例依据 → 给出修正措施」三步口头阐述。"
        )
    elif cm_status in ("degraded", "blocked"):
        q3 = (
            f"本场声画同步提示存在跨通道错位或单路受限（{dossier['reliability']['cross_modal_summary']}），"
            "不建议基于单通道表现推论整体状态，请直接回看关键题号的原始转写文本与画面片段。"
        )
    else:
        disp_note = (
            f"逐题特征提示「{dossier['dispersion_highlights'][0]}」，"
            if dossier["dispersion_highlights"]
            else ""
        )
        q3 = (
            f"{disp_note}建议挑选作答长度或停顿变化最显著的一道题目展开溯源沟通，了解对方在该题上的背景熟悉度与思考卡点。"
        )

    return [
        {"title": "建议一 · 针对关键薄弱环节补充实证追问", "body": q1},
        {"title": "建议二 · 核验采集局限与无文稿复述稳定性", "body": q2},
        {"title": "建议三 · 聚焦题目波动与场景重点深化交流", "body": q3},
    ]


def _sections_to_markdown(sections: list[dict[str, str]]) -> str:
    return "\n\n".join(f"### {sec['title']}\n{sec['body']}" for sec in sections if sec.get("body"))


def _deterministic_report_summary(payload: dict[str, Any]) -> str:
    """简洁过程摘要（向后兼容 summary 模式）。"""
    dossier = _build_behavioral_dossier(payload)
    secs = _deterministic_narrative_sections(dossier, "narrative_review")
    compact = [
        f"1. {secs[0]['body']}",
        f"2. {secs[1]['body']}",
        f"3. {secs[2]['body']}",
        f"4. {secs[3]['body']}",
    ]
    return "\n".join(compact)


def _deterministic_interview_guide(payload: dict[str, Any]) -> str:
    """依据本场真实证据缺口与低置信指标，生成客观人工追问提纲。"""
    dossier = _build_behavioral_dossier(payload)
    secs = _deterministic_guide_sections(dossier)
    return "\n".join(f"{idx}. {s['body']}" for idx, s in enumerate(secs, 1))


def _metric_qual_hint(key: str, val: float) -> str:
    """为本地大模型提供严格客观的量纲方向参照，避免误把高比例读低或相反。"""
    if key == "lexical_diversity":
        return "用词多样性较高" if val >= 0.75 else ("用词较稳健" if val >= 0.5 else "重复用词偏多")
    if key == "speech_rate":
        return "语速偏快" if val >= 4.8 else ("语速稳健适中" if val >= 3.2 else "语速较为从容")
    if key == "answer_detail_chars_p50":
        return "展开较充分" if val >= 120 else ("阐述偏简练" if val >= 50 else "单题字数偏简短")
    if key == "discourse_marker_rate":
        return "条理衔接清晰" if val >= 2.0 else "因果与转折连接词偏少"
    if key == "voiced_ratio":
        return "发声占比充裕" if val >= 0.45 else "静默间隔偏多"
    if key == "pause_structure_entropy":
        return "停顿节律均匀" if val >= 0.72 else "停顿分布略显参差"
    if key == "pitch_variability_per_sec":
        return "语调自然抑扬" if 20.0 <= val <= 85.0 else ("语调趋平稳" if val < 20.0 else "语调起伏较强")
    if key == "gaze_stability":
        return "视线保持稳定" if val >= 0.75 else "视线偶有偏移扫视"
    if key == "head_angular_velocity_3d_p50":
        return "三维头姿从容平稳" if val <= 18.0 else "头部伴随较多辅助转向"
    return "实测有效"


def _compact_facts_for_prompt(payload: dict[str, Any], scene_profile_id: str | None = None) -> str:
    """将脱敏后的多维度实测值、韵律体态读数、阶段流转与缺口打包为清晰中文 JSON，严禁传入 PII。"""
    dossier = _build_behavioral_dossier(payload, scene_profile_id=scene_profile_id)
    rm = dossier["raw_metrics"]
    key_to_label = {v[1]: k for k, v in _INDICATOR_SPEC.items()}
    key_to_label["头部三维转动角速度中位数"] = "head_angular_velocity_3d_p50"

    sig_ratio = dossier["reliability"]["signal_usable_ratio"]
    if sig_ratio is None:
        sig_note = dossier["reliability"]["signal_summary"] or "画面采集条件正常"
    elif sig_ratio >= 0.75:
        sig_note = f"采录画面与传感器状态稳定（可用窗占 {sig_ratio * 100:.0f}%）"
    else:
        sig_note = f"采录期间面部或画面存在间歇性遮挡或波动（稳定画面占比约 {sig_ratio * 100:.0f}%），复核时应注意采集环境影响"

    cm_status = dossier["reliability"]["cross_modal_status"]
    if cm_status in ("degraded", "blocked"):
        cm_note = "音频回答全程清晰，但音视频重叠同步时段相对有限，不宜单独依赖画面做全局推断"
    else:
        cm_note = "多通道时间窗同步对齐良好"

    def _fmt_item(item: dict[str, Any]) -> str:
        k = key_to_label.get(item["label"], "")
        hint = _metric_qual_hint(k, float(item["value"])) if k and isinstance(item.get("value"), (int, float)) else item["confidence"]
        return f"{item['label']}={item['display']}（{hint}）"

    facts = {
        "场景档案": {
            "名称": dossier["scene"]["name"],
            "理论底座": dossier["scene"]["theory_framework"],
            "审辨补充实测": [
                {"指标": m["label"], "读数": m["value"], "单位": m["unit"]}
                for m in dossier["scene"]["meta_indicators"]
                if m.get("value") is not None
            ],
            "元认知三环已呈现": dossier["scene"]["covered_loops"],
            "元认知三环待补充": dossier["scene"]["missing_loops"],
        },
        "整体可信与规模": {
            "关键观测项覆盖": f"{dossier['reliability']['n_measurable']} 项可观测关键点中 {dossier['reliability']['n_passed']} 项取得实测数据，可信度上限为{dossier['reliability']['confidence_cap']}",
            "问答题目数": dossier["reliability"]["n_questions"],
            "画面稳定性说明": sig_note,
            "多通道同步说明": cm_note,
            "主试与作答分离": dossier["reliability"]["speaker_turn_summary"],
        },
        "薄弱或需关注的维度": [
            r["name"] for r in dossier["dimensions"] if r["name"] in dossier["thin_dims"]
        ],
        "实测行为线索精选": {
            grp: [_fmt_item(item) for item in items[:4]]
            for grp, items in dossier["grouped_metrics"].items()
            if items
        },
        "逐题波动重点": dossier["dispersion_highlights"][:2],
        "过程阶段与转折": {
            "总体阶段归类": dossier["stage_flow"]["gate_label"],
            "是否建议人工重点复核": dossier["stage_flow"]["uncertainty_escalated"],
            "行为宏观阶段": dossier["stage_flow"]["macro_stages"],
            "节奏转折点": dossier["stage_flow"]["change_points"],
            "突出正向支撑线索": dossier["stage_flow"]["pos_cues"][:3],
            "有待加强或补录线索": dossier["stage_flow"]["weak_cues"][:2],
        },
        "主要采集缺口": dossier["gaps"][:3],
    }
    return json.dumps(facts, ensure_ascii=False)


def _parse_narrative_sections(
    text: str,
    fallback_titles: list[str],
) -> list[dict[str, str]]:
    """将大模型输出的带标题分段正文规范切分为 `{title, body}` 列表。"""
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    heading_re = re.compile(r"^(?:#{1,4}\s*|\*\*)(.+?)(?:\*\*)?\s*$")
    cn_prefix_re = re.compile(r"^([一二三四五六]、|\d+[.、])\s*(.+)$")

    sections: list[dict[str, str]] = []
    cur_title: str | None = None
    cur_lines: list[str] = []

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m_head = heading_re.match(line)
        m_cn = cn_prefix_re.match(line)
        is_heading = False
        candidate_title = ""
        if line.startswith("#") and m_head:
            is_heading = True
            candidate_title = m_head.group(1).strip().strip("*")
        elif m_head and cn_prefix_re.match(m_head.group(1).strip()):
            is_heading = True
            candidate_title = m_head.group(1).strip().strip("*")
        elif m_cn and len(line) <= 28 and "：" not in line and "，" not in line:
            is_heading = True
            candidate_title = line

        if is_heading and candidate_title:
            if cur_title and cur_lines:
                sections.append({"title": cur_title, "body": "\n".join(cur_lines).strip()})
            cur_title = candidate_title
            cur_lines = []
        else:
            cur_lines.append(line.replace("**", "").strip())

    if cur_title and cur_lines:
        sections.append({"title": cur_title, "body": "\n".join(cur_lines).strip()})
    elif not sections and cur_lines:
        title = fallback_titles[0] if fallback_titles else "行为观察正文"
        sections.append({"title": title, "body": "\n".join(cur_lines).strip()})

    return [s for s in sections if s.get("body")]


def _build_llm_prompts(norm_mode: str, dossier: dict[str, Any], facts_json: str) -> tuple[str, str, int]:
    scene_name = dossier["scene"]["name"]
    # 严禁在提示词字符串内出现红线禁止词（哪怕在否定句中也绝对避免）
    compliance_clause = (
        "1) 所有数值、缺口和事件必须严格忠实于给定脱敏 JSON，高/低评价方向必须严格遵从数据括号内的提示，不可凭空编造或反转解读；"
        "2) 严禁出现总分、打分、综合等级、淘汰推荐、跨人对比或任何医学诊断/品格猜测词汇；"
        "3) 请用自然流畅、温和专业的书面中文连贯成段，每段 2 至 4 句抓住核心发现展开阐述，切忌逐条罗列全部数字与「置信度中/高/低」流水账；"
        "4) 未测到或存在采集遮挡的内容请客观说明限制条件，并务必完整收尾最后一句。"
    )
    if norm_mode == "narrative_review":
        sys_prompt = (
            f"你是《衡才智析》的资深科研行为报告撰写助手，当前服务场景为「{scene_name}」。"
            f"{compliance_clause}"
            "5) 全文控制在 450–620 字左右，精炼易读，严格采用五个 `###` 小标题段落：\n"
            "### 一、采录概况与证据可信度\n"
            "### 二、言语组织与思维展开线索\n"
            "### 三、声学韵律、注视与体态协同\n"
            "### 四、过程节律演进与采集边界\n"
            "### 五、分场景复核与后续引导建议"
        )
        user_prompt = (
            "请依据以下脱敏多模态行为证据档案，面向高校导师与复核人员撰写一份精炼流畅、"
            "重点突出、完整收尾的中文正文报告：\n"
            f"{facts_json}"
        )
        return sys_prompt, user_prompt, 1150

    if norm_mode == "participant_feedback":
        sys_prompt = (
            f"你是《衡才智析》的学习与成长反馈顾问，当前场景为「{scene_name}」。"
            f"{compliance_clause}"
            "5) 使用第二人称「你」与亲切鼓励但客观克制的语气，全文约 420–580 字，完整收尾，采用五个 `###` 小标题段落：\n"
            "### 一、本次过程整体印象\n"
            "### 二、表达组织与思考亮点\n"
            "### 三、声音节奏与体态自然度\n"
            "### 四、过程起伏与环境限制说明\n"
            "### 五、下一步可操作的练习建议"
        )
        user_prompt = (
            "请依据以下脱敏多模态实测事实，为参与者本人撰写一份易懂、真诚、可操作的成长反馈文字报告：\n"
            f"{facts_json}"
        )
        return sys_prompt, user_prompt, 1100

    if norm_mode == "interview_guide":
        sys_prompt = (
            f"你是《衡才智析》的科研交流与实训提纲助手，当前场景为「{scene_name}」。"
            f"{compliance_clause}"
            "5) 输出 3 条结构化、具体可落地的追问与练习建议，采用 `### 建议一 · ...` 等小标题（约 400–600 字）。"
        )
        user_prompt = (
            "请严格结合以下行为证据缺口、薄弱维度与场景特点，生成 3 条面向人工面对面沟通或后续实训的具体追问提纲：\n"
            f"{facts_json}"
        )
        return sys_prompt, user_prompt, 800

    # summary 向后兼容模式
    sys_prompt = (
        f"你是《衡才智析》的行为证据导读助手，当前场景为「{scene_name}」。"
        f"{compliance_clause}"
        "5) 输出约 240–360 字简洁连贯的中文导读，概括有效指标实况、核心表现线索与主要采集局限。"
    )
    user_prompt = f"请把以下脱敏行为证据档案改写为简洁客观的中文导读：\n{facts_json}"
    return sys_prompt, user_prompt, 480


def assist_report(
    payload: dict[str, Any],
    *,
    mode: str = "summary",
    use_llm: bool = True,
    timeout: float | None = None,
    scene_profile: str | None = None,
    session_id: str | None = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """按需生成可读化中文文字报告或追问提纲（含本地大模型润色 + 红线复检 + 确定性回退）。"""
    t0 = time.perf_counter()
    norm_mode = _normalize_mode(mode)
    dossier = _build_behavioral_dossier(payload, scene_profile_id=scene_profile)
    eff_scene_id = dossier["scene"]["id"]

    fallback_sections = _deterministic_narrative_sections(dossier, norm_mode)
    if norm_mode == "summary":
        fallback_text = _deterministic_report_summary(payload)
    elif norm_mode == "interview_guide":
        # 如果拥有丰富实测信息且同时包含薄弱维度名，保证旧测试 `话语结构特征` 在内并保持流畅
        guide_lines = _deterministic_interview_guide(payload)
        fallback_text = _sections_to_markdown(fallback_sections) if len(dossier["raw_metrics"]) >= 4 else guide_lines
    else:
        fallback_text = _sections_to_markdown(fallback_sections)

    facts_json = _compact_facts_for_prompt(payload, scene_profile_id=eff_scene_id)
    cache_key: str | None = None
    if session_id and use_llm and not refresh:
        digest = hashlib.sha1(facts_json.encode("utf-8")).hexdigest()[:12]
        cache_key = f"{session_id}|{eff_scene_id}|{norm_mode}|{digest}"
        now_ts = time.time()
        with _CACHE_LOCK:
            cached = _NARRATIVE_CACHE.get(cache_key)
            if cached and (now_ts - cached[0]) < _CACHE_TTL_SEC:
                res = dict(cached[1])
                res["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
                return res

    source = "deterministic_fallback"
    model_name = _llm_config()[1]
    fallback_reason: str | None = None
    content = fallback_text
    sections_out = fallback_sections

    if use_llm:
        sys_prompt, user_prompt, max_toks = _build_llm_prompts(norm_mode, dossier, facts_json)
        eff_timeout = timeout
        if eff_timeout is None and norm_mode in ("narrative_review", "participant_feedback", "interview_guide"):
            eff_timeout = NARRATIVE_LLM_TIMEOUT_SEC

        llm_text, err, model_name = _call_local_openai_chat(
            sys_prompt,
            user_prompt,
            timeout=eff_timeout,
            max_tokens=max_toks,
        )
        if llm_text:
            violations = redline_guard.find_assistant_violations(llm_text)
            if violations:
                fallback_reason = (
                    f"本地模型输出命中红线词（{'、'.join(violations[:3])}），已自动切换确定性合规文字报告"
                )
            else:
                parsed = _parse_narrative_sections(
                    llm_text, [s["title"] for s in fallback_sections]
                )
                content = llm_text
                if parsed:
                    sections_out = parsed
                source = "local_llm"
        else:
            fallback_reason = err or "本地模型未就绪，已即时生成结构化合规文字报告"
    else:
        fallback_reason = None

    char_count = len(re.sub(r"\s+", "", content))
    reading_min = max(1, round(char_count / 350.0))

    result: dict[str, Any] = {
        "mode": norm_mode,
        "audience_label": AUDIENCE_LABELS.get(norm_mode, "行为观察报告"),
        "scene_profile_id": eff_scene_id,
        "content": content,
        "sections": sections_out,
        "reading_time_min": reading_min,
        "source": source,
        "model": model_name if source == "local_llm" else "deterministic-narrative-v2",
        "fallback_reason": fallback_reason,
        "redline_verified": len(redline_guard.find_assistant_violations(content)) == 0,
        "watermark": ASSISTANT_WATERMARK,
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }

    if cache_key and source == "local_llm":
        with _CACHE_LOCK:
            if len(_NARRATIVE_CACHE) >= _CACHE_MAX_ITEMS:
                oldest = next(iter(_NARRATIVE_CACHE))
                _NARRATIVE_CACHE.pop(oldest, None)
            _NARRATIVE_CACHE[cache_key] = (time.time(), dict(result))

    return result


def _tokenize_zh_en(text: str) -> list[str]:
    """轻量中英双语分词（英文词 + 中文单字与双字切片），零外部分词库依赖。"""
    s = (text or "").lower().strip()
    if not s:
        return []
    tokens: list[str] = re.findall(r"[a-z0-9_]+", s)
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", s)
    for run in cjk_runs:
        for ch in run:
            tokens.append(ch)
        for i in range(len(run) - 1):
            tokens.append(run[i : i + 2])
    return tokens


def search_question_bank(
    module: str,
    query: str,
    *,
    top_k: int = 5,
    use_llm: bool = False,
    timeout: float | None = None,
) -> dict[str, Any]:
    """本地毫秒级题库 TF-IDF 检索（可选本地小模型扩展检索词）。"""
    import question_bank

    t0 = time.perf_counter()
    q_raw = (query or "").strip()
    if not q_raw:
        return {
            "module": module,
            "query": "",
            "expanded_query": "",
            "source": "local_tfidf",
            "hits": [],
            "watermark": ASSISTANT_WATERMARK,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    state = question_bank.load(module)
    docs: list[dict[str, Any]] = []
    for idx, q_text in enumerate(state.get("builtin") or []):
        text = q_text if isinstance(q_text, str) else str(q_text.get("question") or "")
        docs.append({
            "bank_id": "builtin",
            "bank_name": "系统内置默认题库",
            "index": idx,
            "question": text,
            "reference_answer": "",
            "is_active": state.get("active_id") is None,
        })
    for b in state.get("banks") or []:
        bid = b.get("id")
        bname = b.get("name") or str(bid)
        for idx, item in enumerate(b.get("questions") or []):
            if isinstance(item, str):
                q_t, r_t = item, ""
            else:
                q_t = str(item.get("question") or "")
                r_t = str(item.get("reference_answer") or "")
            docs.append({
                "bank_id": bid,
                "bank_name": bname,
                "index": idx,
                "question": q_t,
                "reference_answer": r_t,
                "is_active": state.get("active_id") == bid,
            })

    expanded_query = q_raw
    source = "local_tfidf"
    if use_llm:
        ext, _, _ = _call_local_openai_chat(
            "输出 3 个与查询词最相关的中文学术科研/面试同义关键词，用空格隔开，不要解释。",
            q_raw,
            timeout=timeout or 4.0,
            max_tokens=40,
        )
        if ext and not redline_guard.find_assistant_violations(ext):
            expanded_query = f"{q_raw} {ext}"
            source = "local_llm+tfidf"

    q_tokens = _tokenize_zh_en(expanded_query)
    if not q_tokens or not docs:
        return {
            "module": module,
            "query": q_raw,
            "expanded_query": expanded_query,
            "source": source,
            "hits": [],
            "watermark": ASSISTANT_WATERMARK,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    doc_tokens = [_tokenize_zh_en(d["question"] + " " + d["reference_answer"]) for d in docs]
    n_docs = len(docs)
    df: dict[str, int] = {}
    for toks in doc_tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1

    idf = {t: math.log((n_docs + 1.0) / (df.get(t, 0) + 1.0)) + 1.0 for t in set(q_tokens)}
    q_vec: dict[str, float] = {}
    for t in q_tokens:
        q_vec[t] = q_vec.get(t, 0.0) + idf.get(t, 1.0)
    q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

    scored: list[dict[str, Any]] = []
    for d, toks in zip(docs, doc_tokens):
        if not toks:
            continue
        tf: dict[str, float] = {}
        for t in toks:
            tf[t] = tf.get(t, 0.0) + 1.0
        matched = [t for t in set(q_tokens) if t in tf]
        if not matched:
            continue
        dot = sum(q_vec[t] * tf[t] * idf.get(t, 1.0) for t in matched)
        d_norm = math.sqrt(sum((cnt * idf.get(t, 1.0)) ** 2 for t, cnt in tf.items())) or 1.0
        sim = dot / (q_norm * d_norm)
        if q_raw.lower() in d["question"].lower():
            sim += 0.35
        scored.append({
            **d,
            "similarity": round(float(sim), 4),
            "matched_terms": sorted({m for m in matched if len(m) >= 2})[:6],
        })

    scored.sort(key=lambda r: r["similarity"], reverse=True)
    return {
        "module": module,
        "query": q_raw,
        "expanded_query": expanded_query,
        "source": source,
        "total_candidates": len(docs),
        "hits": scored[: max(1, min(20, int(top_k)))],
        "watermark": ASSISTANT_WATERMARK,
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }
