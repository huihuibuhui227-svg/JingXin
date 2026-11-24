# report_frontend/longitudinal_tracker.py
"""《衡才智析》个体内多访次纵向追踪与跨场对比引擎（严守非医疗、不给分与可比性原则）。

方法学借鉴并产品化 `subject_trend.py`：
1. 严格匿名建档：仅存 `subject_code`（如 `SUBJ-001`）、场景标识与绑定访次序列，
   不保存真实姓名、学号或任何个体推断结论；
2. 跨版本定义守卫：若多场次跨越 `capture:gen1 / gen1.5 / gen2` 或重算脚本哈希变化，
   显式标注为不可比并拒绝拟合跨越代差的连线；
3. 缺口绝不当零守卫：某次缺测或未达门限时保留空值与原始缺口代码，图表按断点断开；
4. 首访自身基线锚定：以首次有效可比访次作为个人参考原点，计算客观动力学指标的
   相对变化幅度与中间半数分布带，遇模型保形升级或分布偏离时提示人工复核。
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import media_retention
import redline_guard
from .data_loader import LogDataLoader
from .l1_report import build_l1_report
from .report_generator import session_started_at

_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_STORE_PATH = _REPO / "data" / "longitudinal_subjects.json"
_SUBJECT_CODE_RE = re.compile(r"^[A-Za-z0-9_-]{2,48}$")
_LOCK = threading.Lock()

VERIFICATION_COMPLIANCE_NOTE = (
    "非医疗器械 · 本模块仅计算匿名个体多次采录的客观声学与运动学指标变化轨迹，"
    "绝不跨人对照或输出任何健康、身心状态及群体划分结论。"
)

TRACKED_METRICS: tuple[dict[str, str], ...] = (
    {"key": "voiced_ratio", "label": "语音发声占比", "unit": "比例", "source": "l1"},
    {"key": "pause_structure_entropy", "label": "停顿节律均匀度", "unit": "无量纲", "source": "l1"},
    {"key": "pitch_variability_per_sec", "label": "基频每秒变动幅度", "unit": "Hz/秒", "source": "l1"},
    {"key": "gaze_stability", "label": "视线水平稳定度", "unit": "无量纲", "source": "l1"},
    {"key": "gesture_activity_per_sec", "label": "上肢每秒活动幅度", "unit": "单位/秒", "source": "l1"},
    {"key": "head_angular_vel_3d", "label": "头姿三维角速度中位", "unit": "度/秒", "source": "fv2"},
)


def store_path() -> Path:
    env = os.getenv("JINGXIN_LONGITUDINAL_FILE")
    return Path(env) if env else _DEFAULT_STORE_PATH


def _validate_subject_code(code: str) -> str:
    s = str(code or "").strip()
    if not _SUBJECT_CODE_RE.fullmatch(s):
        raise ValueError("匿名随访编号仅允许字母、数字、下划线或连字符（2 至 48 位）。")
    return s


def _check_clean_label(text: str, field_name: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    hits = redline_guard.find_assistant_violations(s)
    if hits:
        raise ValueError(f"{field_name} 包含合规受限表述：{', '.join(hits)}")
    return s


def _load_store() -> dict[str, Any]:
    p = store_path()
    if not p.is_file():
        return {"schema_version": "longitudinal_subjects_v1", "subjects": {}}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and isinstance(raw.get("subjects"), dict):
            return raw
    except Exception:
        pass
    return {"schema_version": "longitudinal_subjects_v1", "subjects": {}}


def _save_store(data: dict[str, Any]) -> None:
    p = store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def list_subjects() -> list[dict[str, Any]]:
    with _LOCK:
        db = _load_store()
        rows = list(db["subjects"].values())
    rows.sort(key=lambda r: str(r.get("updated_at") or r.get("created_at") or ""), reverse=True)
    return rows


def upsert_subject(
    subject_code: str,
    *,
    scene_profile: str = "longitudinal_followup",
    alias_note: str = "",
    consent_mode: str = "anonymous_kinematics_only",
) -> dict[str, Any]:
    code = _validate_subject_code(subject_code)
    clean_note = _check_clean_label(alias_note, "随访分组备注")
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with _LOCK:
        db = _load_store()
        existing = db["subjects"].get(code)
        if existing is None:
            rec = {
                "subject_code": code,
                "scene_profile": scene_profile or "longitudinal_followup",
                "alias_note": clean_note,
                "consent_mode": consent_mode or "anonymous_kinematics_only",
                "created_at": now,
                "updated_at": now,
                "visits": [],
            }
        else:
            rec = dict(existing)
            if scene_profile:
                rec["scene_profile"] = scene_profile
            if alias_note:
                rec["alias_note"] = clean_note
            if consent_mode:
                rec["consent_mode"] = consent_mode
            rec["updated_at"] = now
        db["subjects"][code] = rec
        _save_store(db)
        return rec


def attach_visit(
    subject_code: str,
    session_id: str,
    *,
    visit_label: str = "",
    privacy_mode: str = "anonymous_kinematics_only",
) -> dict[str, Any]:
    code = _validate_subject_code(subject_code)
    sid = media_retention.validate_session_id(str(session_id or "").strip())
    clean_vlabel = _check_clean_label(visit_label, "访次标记")
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with _LOCK:
        db = _load_store()
        rec = db["subjects"].get(code)
        if rec is None:
            rec = {
                "subject_code": code,
                "scene_profile": "longitudinal_followup",
                "alias_note": "",
                "consent_mode": privacy_mode or "anonymous_kinematics_only",
                "created_at": now,
                "updated_at": now,
                "visits": [],
            }
        visits = [v for v in (rec.get("visits") or []) if v.get("session_id") != sid]
        visits.append({
            "session_id": sid,
            "visit_order": len(visits) + 1,
            "visit_label": clean_vlabel or f"第 {len(visits) + 1} 次随访",
            "privacy_mode": privacy_mode or rec.get("consent_mode") or "anonymous_kinematics_only",
            "attached_at": now,
        })
        for idx, item in enumerate(visits, 1):
            item["visit_order"] = idx
        rec["visits"] = visits
        rec["updated_at"] = now
        db["subjects"][code] = rec
        _save_store(db)
        return rec


def _infer_def_version(session_id: str) -> tuple[str, str]:
    """判定采集定义版本（对齐 subject_trend.py 两路口径）。"""
    manifest_path = _REPO / "data" / "logs_recomputed" / "_manifest" / f"{session_id}.json"
    if manifest_path.is_file():
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            head = str(m.get("pipeline_git_head") or "")[:7]
            if head:
                return f"recomputed@{head}", "基于重算清单哈希判定口径版本"
        except Exception:
            pass

    dt = session_started_at(session_id)
    if dt is None:
        m_date = re.match(r"^(\d{8})_", session_id)
        if m_date:
            try:
                dt = datetime.strptime(m_date.group(1), "%Y%m%d")
            except ValueError:
                dt = None
    if dt is None:
        return "capture:unknown", "缺少会话启动日期，保守记为未知版本"

    day = dt.strftime("%Y-%m-%d")
    if day < "2026-09-26":
        return "capture:gen1", "2026-09-26 之前初代采集口径"
    if day == "2026-09-26":
        return "capture:gen1.5", "2026-09-26 过渡日采集口径"
    return "capture:gen2", "2026-09-27 及之后现行采集口径"


def _extract_session_snapshot(
    session_id: str,
    *,
    scene_profile: str = "longitudinal_followup",
    loader: LogDataLoader | None = None,
) -> dict[str, Any]:
    ld = loader or LogDataLoader()
    data = ld.get_fused_latest_data(session_id) or {}
    payload = build_l1_report(session_id, logs=data, scene_profile=scene_profile)

    ind_map = {
        str(row["name"]): row
        for row in (payload.get("l1") or {}).get("indicators") or []
        if isinstance(row, dict) and row.get("name")
    }
    fv2 = (payload.get("l1") or {}).get("feature_layer_v2") or {}
    face_kin = ((fv2.get("tracks") or {}).get("face_kinematics") or {}) if isinstance(fv2, dict) else {}
    bs_cal = (payload.get("l1") or {}).get("blendshape_calibration") or {}
    pose_mode = str(face_kin.get("pose_mode") or bs_cal.get("pose_mode") or "unavailable")
    fv2_ang = face_kin.get("head_angular_velocity_3d_p50") if pose_mode == "3d_matrix" else None

    ref_ml = (payload.get("l1") or {}).get("referee_ml") or {}
    gate_info = ref_ml.get("evidence_gate") or {}
    coverage = payload.get("coverage") or {}

    metrics: dict[str, dict[str, Any]] = {}
    for spec in TRACKED_METRICS:
        k = spec["key"]
        if spec["source"] == "l1":
            row = ind_map.get(k) or {}
            val = row.get("value")
            if isinstance(val, (int, float)) and math.isfinite(float(val)):
                metrics[k] = {
                    "value": round(float(val), 6),
                    "confidence": row.get("confidence") or "无",
                    "gap": row.get("gap"),
                }
            else:
                metrics[k] = {
                    "value": None,
                    "confidence": row.get("confidence") or "无",
                    "gap": row.get("gap") or "no_samples",
                }
        else:
            if isinstance(fv2_ang, (int, float)) and math.isfinite(float(fv2_ang)):
                metrics[k] = {
                    "value": round(float(fv2_ang), 6),
                    "confidence": "高",
                    "gap": None,
                }
            else:
                metrics[k] = {
                    "value": None,
                    "confidence": "无",
                    "gap": "requires_3d_pose" if pose_mode != "3d_matrix" else "unavailable",
                }

    def_ver, def_note = _infer_def_version(session_id)
    dt = session_started_at(session_id)
    return {
        "session_id": session_id,
        "tested_at": dt.strftime("%Y-%m-%d %H:%M:%S") if dt else None,
        "def_version": def_ver,
        "def_version_note": def_note,
        "pose_mode": pose_mode,
        "has_loaded_logs": bool(data),
        "coverage_passed": int(coverage.get("n_passed") or 0),
        "coverage_slots": int(coverage.get("n_slots") or 14),
        "confidence_cap": coverage.get("confidence_cap") or "无",
        "referee_stage_code": gate_info.get("stage_code") or "stage_3_limited",
        "referee_stage_label": gate_info.get("stage_label") or "基础核验",
        "uncertainty_escalated": bool(gate_info.get("uncertainty_escalated")),
        "ood_anomaly_flag": bool(gate_info.get("ood_anomaly_flag")),
        "metrics": metrics,
    }


def compute_trajectory(
    visit_records: list[dict[str, Any]],
    *,
    scene_profile: str = "longitudinal_followup",
    loader: LogDataLoader | None = None,
) -> dict[str, Any]:
    """给定有序访次列表，输出跨版本可比性守卫与个体内首访基线偏移轨迹。"""
    snapshots: list[dict[str, Any]] = []
    for idx, v in enumerate(visit_records, 1):
        sid = str(v.get("session_id") or "").strip()
        if not sid:
            continue
        snap = _extract_session_snapshot(sid, scene_profile=scene_profile, loader=loader)
        snap["visit_order"] = int(v.get("visit_order") or idx)
        snap["visit_label"] = str(v.get("visit_label") or f"第 {idx} 次")
        snap["privacy_mode"] = str(v.get("privacy_mode") or "full")
        snapshots.append(snap)

    versions = [s["def_version"] for s in snapshots if s["has_loaded_logs"]]
    has_unknown = any(v == "capture:unknown" for v in versions)
    unique_versions = sorted(set(versions))

    comparability_reasons: list[str] = []
    if len(snapshots) < 2:
        comparability_reasons.append("有效访次不足 2 次，仅展示单次基线快照，暂不构建纵向变化曲线。")
    if has_unknown:
        comparability_reasons.append("存在无法确定定义版本的场次，按规范禁止强制连线。")
    if len(unique_versions) > 1:
        comparability_reasons.append(
            f"所选场次横跨多个采集定义版本（{', '.join(unique_versions)}），"
            "不同代次分母/坐标系不可混算，仅支持分代并排核对。"
        )
    if any(not s["has_loaded_logs"] for s in snapshots):
        comparability_reasons.append("存在未录得有效融合日志的空场次，该场次位置断线标为空值。")

    comparable = (
        len(snapshots) >= 2
        and not has_unknown
        and len(unique_versions) == 1
        and all(s["has_loaded_logs"] for s in snapshots)
    )

    series_list: list[dict[str, Any]] = []
    for spec in TRACKED_METRICS:
        k = spec["key"]
        pts: list[dict[str, Any]] = []
        valid_vals: list[float] = []
        anchor_val: float | None = None
        for snap in snapshots:
            m = snap["metrics"].get(k) or {}
            v = m.get("value")
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                vf = float(v)
                valid_vals.append(vf)
                if anchor_val is None:
                    anchor_val = vf
                delta_pct = (
                    round((vf - anchor_val) / abs(anchor_val) * 100.0, 2)
                    if anchor_val is not None and abs(anchor_val) > 1e-6
                    else (0.0 if anchor_val is not None else None)
                )
            else:
                delta_pct = None
            pts.append({
                "visit_order": snap["visit_order"],
                "visit_label": snap["visit_label"],
                "session_id": snap["session_id"],
                "value": m.get("value"),
                "delta_from_t0_pct": delta_pct,
                "confidence": m.get("confidence") or "无",
                "gap": m.get("gap"),
            })

        arr = np.asarray(valid_vals, dtype=float) if valid_vals else np.array([], dtype=float)
        summary = {
            "n_valid_visits": len(valid_vals),
            "t0_baseline": round(float(anchor_val), 6) if anchor_val is not None else None,
            "p25": round(float(np.quantile(arr, 0.25)), 6) if len(arr) >= 1 else None,
            "median": round(float(np.median(arr)), 6) if len(arr) >= 1 else None,
            "p75": round(float(np.quantile(arr, 0.75)), 6) if len(arr) >= 1 else None,
            "drawable": comparable and len(valid_vals) >= 2,
        }
        series_list.append({
            "key": k,
            "label": spec["label"],
            "unit": spec["unit"],
            "summary": summary,
            "points": pts,
        })

    review_alerts: list[str] = []
    for snap in snapshots:
        if snap.get("ood_anomaly_flag"):
            review_alerts.append(
                f"{snap['visit_label']}（{snap['session_id']}）检出节律分布偏离，"
                "建议人工复核设备遮挡、坐姿变化或背景干扰。"
            )
        elif snap.get("uncertainty_escalated"):
            review_alerts.append(
                f"{snap['visit_label']}（{snap['session_id']}）处于保形不确定升级状态，"
                "建议人工确认低置信指标项。"
            )

    return {
        "comparable": comparable,
        "def_versions": unique_versions,
        "comparability_notes": comparability_reasons,
        "n_visits": len(snapshots),
        "visits": snapshots,
        "series": series_list,
        "review_alerts": review_alerts,
        "compliance_note": VERIFICATION_COMPLIANCE_NOTE,
    }


def get_subject_trajectory(subject_code: str, *, loader: LogDataLoader | None = None) -> dict[str, Any]:
    code = _validate_subject_code(subject_code)
    with _LOCK:
        db = _load_store()
        rec = db["subjects"].get(code)
    if rec is None:
        raise KeyError(f"未找到匿名随访档案：{code}")
    traj = compute_trajectory(
        list(rec.get("visits") or []),
        scene_profile=str(rec.get("scene_profile") or "longitudinal_followup"),
        loader=loader,
    )
    return {
        "subject": rec,
        **traj,
    }


def compare_sessions(
    session_ids: list[str],
    *,
    scene_profile: str = "research_interview",
    loader: LogDataLoader | None = None,
) -> dict[str, Any]:
    """面向基地一科研过程的多场会话并排与纵向对比接口。"""
    cleaned: list[dict[str, Any]] = []
    for idx, raw_sid in enumerate(session_ids, 1):
        sid = media_retention.validate_session_id(str(raw_sid or "").strip())
        cleaned.append({
            "session_id": sid,
            "visit_order": idx,
            "visit_label": f"场次 {idx}",
            "privacy_mode": "full",
        })
    return compute_trajectory(cleaned, scene_profile=scene_profile, loader=loader)
