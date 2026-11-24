# report_frontend/referee_ml_engine.py
"""多模态行为阶段切分与证据核验门控推理引擎（纯 CPU 毫秒级，严格只读）。

设计边界：
- 任务 1（宏阶段切分）：基于 4 状态粘性 GaussianHMM Viterbi 解码与 PELT 变点检测，
  将多模态滑动窗序列归并为连续行为阶段并定位关键转折点；
- 任务 2（证据核验门控）：基于 8 维客观采集覆盖与时序质量特征，结合
  IsolationForest 分布外检测、逻辑回归与分布无关 Split Conformal Prediction
  （alpha=0.10，理论覆盖率 90%）判定证据阶段并输出白盒加性特征归因；
- 严格不计算任何总分、不做人物对比或主观特质归因；空日志输入严格确定性返回且耗时为 0.0 ms。
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from report_frontend import behavior_regime_model as _brm
from report_frontend import l1_quality as _l1_quality
from report_frontend import signal_quality_clf as _sqc

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "models"
DEFAULT_HMM_PATH = DEFAULT_MODEL_DIR / "referee_hmm_model.joblib"
DEFAULT_CONFORMAL_PATH = DEFAULT_MODEL_DIR / "referee_conformal_gate.joblib"
DEFAULT_METRICS_PATH = Path(__file__).resolve().parent / "referee_ml_metrics.json"

STAGE_ORDER: tuple[str, ...] = (
    "stage_1_verified",
    "stage_2_review",
    "stage_3_limited",
)

STAGE_LABELS: dict[str, str] = {
    "stage_1_verified": "阶段 I · 证据充分可核验",
    "stage_2_review": "阶段 II · 证据可用需复核",
    "stage_3_limited": "阶段 III · 证据受限待补采",
    "unavailable": "未获取有效采集数据",
}

MACRO_STAGE_ORDER: tuple[str, ...] = (
    "steady_speaking",
    "active_expressive",
    "pause_thinking",
    "low_signal",
)

MACRO_STAGE_META: dict[str, dict[str, str]] = {
    "steady_speaking": {
        "label": "平稳展开期",
        "color_key": "steady",
    },
    "active_expressive": {
        "label": "高互动活跃期",
        "color_key": "active",
    },
    "pause_thinking": {
        "label": "停顿组织期",
        "color_key": "pause",
    },
    "low_signal": {
        "label": "采集受限期",
        "color_key": "low_signal",
    },
}

EVIDENCE_FEATURE_NAMES: tuple[str, ...] = (
    "measured_ratio",
    "valid_modality_count_norm",
    "signal_usable_ratio",
    "cross_modal_overlap_ratio",
    "face_valid_frame_ratio",
    "audio_speech_coverage",
    "gesture_continuity_ratio",
    "dynamics_stability_score",
)

EVIDENCE_FEATURE_LABELS: dict[str, str] = {
    "measured_ratio": "核心指标实测比例",
    "valid_modality_count_norm": "有效模态齐备度",
    "signal_usable_ratio": "传感窗口信号可用率",
    "cross_modal_overlap_ratio": "跨模态有效重合度",
    "face_valid_frame_ratio": "面部有效检出比例",
    "audio_speech_coverage": "语音发声连续覆盖",
    "gesture_continuity_ratio": "上肢姿态有效覆盖",
    "dynamics_stability_score": "时序采样与动态稳定性",
}

NON_SCORING_GUARANTEE = "仅做行为阶段切分与证据充分度核验，不计算任何总分或人物评价"
DISCLAIMER_TEXT = (
    "本模块仅对客观动作节律做时序分段并对采集证据充分度做不确定性核验，"
    "不输出任何数值汇总或主观特质结论；遇不确定或异常分布时自动转交人工复核。"
)

_HMM_CACHE: dict[str, Any] = {"path": None, "mtime": None, "artifact": None}
_GATE_CACHE: dict[str, Any] = {"path": None, "mtime": None, "artifact": None}
_METRICS_CACHE: dict[str, Any] = {"path": None, "mtime": None, "metrics": None}


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _clip01(val: Any, default: float = 0.0) -> float:
    v = _safe_float(val, default)
    return max(0.0, min(1.0, v))


def _load_joblib_cached(path: Path, cache: dict[str, Any], required_keys: tuple[str, ...]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        mtime = path.stat().st_mtime_ns
        if cache["path"] == str(path) and cache["mtime"] == mtime:
            return cache["artifact"]
        import joblib
        artifact = joblib.load(path)
        if not isinstance(artifact, dict) or not all(k in artifact for k in required_keys):
            return None
        cache.update({"path": str(path), "mtime": mtime, "artifact": artifact})
        return artifact
    except Exception:
        return None


def _load_metrics_cached(metrics_path: Path) -> dict[str, Any] | None:
    if not metrics_path.is_file():
        return None
    try:
        mtime = metrics_path.stat().st_mtime_ns
        if _METRICS_CACHE["path"] == str(metrics_path) and _METRICS_CACHE["mtime"] == mtime:
            return _METRICS_CACHE["metrics"]
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if isinstance(metrics, dict):
            _METRICS_CACHE.update({"path": str(metrics_path), "mtime": mtime, "metrics": metrics})
            return metrics
    except Exception:
        return None
    return None


def _has_any_log_rows(logs: dict[str, Any] | None) -> bool:
    if not isinstance(logs, dict) or not logs:
        return False
    for val in logs.values():
        if isinstance(val, pd.DataFrame) and len(val) > 0:
            return True
    return False


def _pick_voice_df(logs: dict[str, Any] | None) -> pd.DataFrame | None:
    if not isinstance(logs, dict):
        return None
    for key in ("voice", "voice_interview", "voice_research"):
        df = logs.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    return None


def _df_usable_ratio(df: pd.DataFrame | None, primary_cols: tuple[str, ...], fallback_prefixes: tuple[str, ...]) -> float:
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return 0.0
    present_primary = [c for c in primary_cols if c in df.columns]
    if present_primary:
        return float(df[present_primary].notna().any(axis=1).mean())
    fallback_cols = [
        c for c in df.columns
        if any(c.startswith(p) or p in c for p in fallback_prefixes)
    ]
    if fallback_cols:
        return float(df[fallback_cols].notna().any(axis=1).mean())
    return 0.0


def _series_to_seconds(df: pd.DataFrame | None, col: str = "timestamp") -> np.ndarray:
    if not isinstance(df, pd.DataFrame) or df.empty or col not in df.columns:
        return np.array([], dtype=float)
    s = df[col]
    if pd.api.types.is_datetime64_any_dtype(s):
        ns = s.astype("int64").to_numpy(dtype=float)
        # 过滤 NaT（min int64）以及被 pd.to_datetime 错误截断为 1970-01-01 + <1ms 的相对秒
        valid = np.isfinite(ns) & (ns > -9e18)
        if not valid.any():
            return np.array([], dtype=float)
        if float(np.nanmax(np.abs(ns[valid]))) < 1e9:
            return np.array([], dtype=float)
        out = np.full(len(ns), np.nan, dtype=float)
        out[valid] = ns[valid] / 1e9
        return out
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def _compute_fg_jaccard(face_df: pd.DataFrame | None, gesture_df: pd.DataFrame | None) -> float | None:
    if not isinstance(face_df, pd.DataFrame) or not isinstance(gesture_df, pd.DataFrame):
        return None
    if face_df.empty or gesture_df.empty:
        return None
    ft = _series_to_seconds(face_df, "timestamp")
    gt = _series_to_seconds(gesture_df, "timestamp")
    ft_fin = ft[np.isfinite(ft)]
    gt_fin = gt[np.isfinite(gt)]
    if ft_fin.size == 0 or gt_fin.size == 0:
        return None
    if abs(float(ft_fin.min()) - float(gt_fin.min())) > 3600.0:
        ft0 = float(ft_fin.min())
        gt0 = float(gt_fin.min())
    else:
        ft0 = gt0 = min(float(ft_fin.min()), float(gt_fin.min()))

    def _valid_sec_set(df: pd.DataFrame, ts: np.ndarray, t_orig: float, primary: tuple[str, ...], prefixes: tuple[str, ...]) -> set[int]:
        mask = np.isfinite(ts)
        cols = [c for c in primary if c in df.columns]
        if not cols:
            cols = [c for c in df.columns if any(c.startswith(p) or p in c for p in prefixes)]
        if cols:
            mask &= df[cols].notna().any(axis=1).to_numpy(dtype=bool)
        valid = ts[mask] - t_orig
        if valid.size == 0:
            return set()
        return set(np.floor(valid).astype(int).tolist())

    sf = _valid_sec_set(face_df, ft, ft0, _l1_quality.COVERAGE_COLUMNS["face"], ("au", "bs_", "gaze_", "head_"))
    sg = _valid_sec_set(gesture_df, gt, gt0, _l1_quality.COVERAGE_COLUMNS["gesture"], ("jitter", "angle", "spread", "hand_visible"))
    union = len(sf | sg)
    if union == 0:
        return 0.0
    return float(len(sf & sg) / union)


def _compute_dynamics_stability(face_df: pd.DataFrame | None, gesture_df: pd.DataFrame | None) -> float:
    dt_cvs: list[float] = []
    art_rates: list[float] = []
    has_any_ts = False

    for df in (face_df, gesture_df):
        ts = _series_to_seconds(df, "timestamp")
        if ts.size >= 3:
            has_any_ts = True
            dt = np.diff(ts)
            dt_seg = dt[np.isfinite(dt) & (dt > 0.0) & (dt <= 10.0)]
            if dt_seg.size >= 2 and float(np.mean(dt_seg)) > 0.0:
                dt_cvs.append(float(np.std(dt_seg) / np.mean(dt_seg)))

    if isinstance(face_df, pd.DataFrame) and len(face_df) >= 2 and "head_yaw" in face_df.columns:
        yaw = pd.to_numeric(face_df["head_yaw"], errors="coerce").to_numpy(dtype=float)
        dy = np.abs(np.diff(yaw))
        dy_fin = dy[np.isfinite(dy)]
        if dy_fin.size > 0:
            art_rates.append(float((dy_fin > 0.5).mean()))

    if isinstance(gesture_df, pd.DataFrame) and len(gesture_df) >= 1:
        for col in ("left_wrist_jitter_world", "right_wrist_jitter_world", "left_hand_jitter", "right_hand_jitter"):
            if col in gesture_df.columns:
                arr = pd.to_numeric(gesture_df[col], errors="coerce").to_numpy(dtype=float)
                arr_fin = arr[np.isfinite(arr)]
                if arr_fin.size > 0:
                    art_rates.append(float((arr_fin > 0.35).mean()))

    if dt_cvs:
        w_dt_cv = float(np.mean(dt_cvs))
    elif has_any_ts:
        w_dt_cv = 1.0
    else:
        has_rows = (isinstance(face_df, pd.DataFrame) and len(face_df) > 0) or (
            isinstance(gesture_df, pd.DataFrame) and len(gesture_df) > 0
        )
        w_dt_cv = 0.25 if has_rows else 1.0

    art_r = float(np.mean(art_rates)) if art_rates else 0.0
    score = 0.65 * (1.0 / (1.0 + max(0.0, w_dt_cv))) + 0.35 * (1.0 - max(0.0, min(1.0, art_r)))
    return _clip01(score, 0.0)


def extract_evidence_features(
    logs: dict[str, Any] | None,
    session_id: str | None = None,
    l1_summary: dict[str, Any] | None = None,
    feature_v2: dict[str, Any] | None = None,
    *,
    voice_series_list: Any = None,
    l1_result: Any = None,
    cmc_payload: dict[str, Any] | None = None,
    sq_payload: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """提取 8 维客观证据质量特征，训练与线上推理共享唯一口径。"""
    logs = logs or {}
    face_df = logs.get("face")
    gesture_df = logs.get("gesture")
    voice_df = _pick_voice_df(logs)

    l1_dict: dict[str, Any] = {}
    if isinstance(l1_summary, dict):
        if isinstance(l1_summary.get("l1"), dict):
            l1_dict = l1_summary["l1"]
        else:
            l1_dict = l1_summary
    fv2 = feature_v2 if isinstance(feature_v2, dict) else {}

    # 加载整场语音序列（如未传入）
    vs_list = voice_series_list
    if vs_list is None and session_id:
        try:
            from report_frontend import voice_series as _vs_mod
            vs_list = _vs_mod.load_series(session_id)
        except Exception:
            vs_list = None

    # 1. face_valid_frame_ratio & gesture_continuity_ratio
    mod_cov = l1_dict.get("modality_coverage") if isinstance(l1_dict.get("modality_coverage"), dict) else None
    if mod_cov and isinstance(mod_cov.get("face"), dict) and mod_cov["face"].get("rows", 0) > 0:
        fv_raw = _clip01(mod_cov["face"].get("usable_ratio", 0.0))
        if fv_raw == 0.0 and isinstance(face_df, pd.DataFrame) and len(face_df) > 0:
            if not any(c in face_df.columns for c in _l1_quality.COVERAGE_COLUMNS["face"]):
                fv_raw = _df_usable_ratio(face_df, _l1_quality.COVERAGE_COLUMNS["face"], ("au", "bs_", "gaze_", "head_"))
    else:
        fv_raw = _df_usable_ratio(face_df, _l1_quality.COVERAGE_COLUMNS["face"], ("au", "bs_", "gaze_", "head_"))

    if mod_cov and isinstance(mod_cov.get("gesture"), dict) and mod_cov["gesture"].get("rows", 0) > 0:
        gv_raw = _clip01(mod_cov["gesture"].get("usable_ratio", 0.0))
        if gv_raw == 0.0 and isinstance(gesture_df, pd.DataFrame) and len(gesture_df) > 0:
            if not any(c in gesture_df.columns for c in _l1_quality.COVERAGE_COLUMNS["gesture"]):
                gv_raw = _df_usable_ratio(gesture_df, _l1_quality.COVERAGE_COLUMNS["gesture"], ("jitter", "angle", "spread", "hand_visible"))
    else:
        gv_raw = _df_usable_ratio(gesture_df, _l1_quality.COVERAGE_COLUMNS["gesture"], ("jitter", "angle", "spread", "hand_visible"))

    if "face_valid_frame_ratio" in fv2:
        fv_raw = _clip01(fv2["face_valid_frame_ratio"], fv_raw)
    if "gesture_continuity_ratio" in fv2:
        gv_raw = _clip01(fv2["gesture_continuity_ratio"], gv_raw)

    # 2. audio_speech_coverage
    speech_cov = 0.0
    used_whole_series = False
    if "audio_speech_coverage" in fv2:
        speech_cov = _clip01(fv2["audio_speech_coverage"])
    else:
        if vs_list is not None:
            cand = vs_list if isinstance(vs_list, (list, tuple)) else [vs_list]
            whole = [
                item for item in cand
                if item is not None
                and getattr(item, "index", None) is None
                and getattr(item, "f0", None) is not None
                and len(getattr(item, "f0")) > 0
            ]
            if whole:
                f0 = np.asarray(whole[0].f0, dtype=float)
                speech_cov = _clip01(float(np.isfinite(f0).mean()))
                used_whole_series = True
        if speech_cov == 0.0 and isinstance(voice_df, pd.DataFrame) and not voice_df.empty:
            if "speech_duration_sec" in voice_df.columns and "duration_sec" in voice_df.columns:
                sd = pd.to_numeric(voice_df["speech_duration_sec"], errors="coerce").sum()
                td = pd.to_numeric(voice_df["duration_sec"], errors="coerce").sum()
                if td > 0.0 and math.isfinite(sd) and math.isfinite(td):
                    speech_cov = _clip01(sd / td)
            elif "voiced_ratio" in voice_df.columns:
                vr = pd.to_numeric(voice_df["voiced_ratio"], errors="coerce").dropna()
                if len(vr) > 0:
                    speech_cov = _clip01(float(vr.mean()))

    # 3. measured_ratio & valid_modality_count_norm
    measured_ratio: float | None = None
    mod_norm: float | None = None
    if "measured_ratio" in fv2:
        measured_ratio = _clip01(fv2["measured_ratio"])
    if "valid_modality_count_norm" in fv2:
        mod_norm = _clip01(fv2["valid_modality_count_norm"])

    if measured_ratio is None or mod_norm is None:
        if "n_measured" in l1_dict and "n_total" in l1_dict and _safe_float(l1_dict["n_total"], 0.0) > 0:
            if measured_ratio is None:
                measured_ratio = _clip01(_safe_float(l1_dict["n_measured"]) / max(_safe_float(l1_dict["n_total"], 1.0), 1.0))
            if mod_norm is None and isinstance(l1_dict.get("modality_summary"), dict):
                active_cnt = sum(
                    1 for m in ("face", "gesture", "voice", "text")
                    if isinstance(l1_dict["modality_summary"].get(m), dict)
                    and _safe_float(l1_dict["modality_summary"][m].get("n_measured", 0)) > 0
                )
                mod_norm = _clip01(active_cnt / 4.0)

    res_obj = l1_result
    if (measured_ratio is None or mod_norm is None or (cmc_payload is None and "cross_modal_coverage" not in l1_dict)) and _has_any_log_rows(logs):
        if res_obj is None:
            try:
                from report_frontend.l1_session import compute_session_l1
                res_obj = compute_session_l1(session_id or "eval_session", logs=logs)
            except Exception:
                res_obj = None

    if measured_ratio is None or mod_norm is None:
        if res_obj is not None and getattr(res_obj, "indicators", None):
            inds = res_obj.indicators
            if measured_ratio is None:
                measured_ratio = _clip01(sum(1 for i in inds if i.has_value) / max(len(inds), 1))
            if mod_norm is None:
                fmap = {"F": "face", "G": "gesture", "V": "voice", "T": "text"}
                active = {fmap[i.family[0]] for i in inds if i.has_value and i.family and i.family[0] in fmap}
                mod_norm = _clip01(len(active) / 4.0)

    if measured_ratio is None or mod_norm is None:
        # 轻量合成兜底估计
        active_mods = 0
        if fv_raw > 0.05:
            active_mods += 1
        if gv_raw > 0.05:
            active_mods += 1
        if speech_cov > 0.05 or (isinstance(voice_df, pd.DataFrame) and len(voice_df) > 0):
            active_mods += 1
        if isinstance(voice_df, pd.DataFrame) and "transcript" in voice_df.columns:
            if voice_df["transcript"].dropna().astype(str).str.strip().ne("").any():
                active_mods += 1
        if mod_norm is None:
            mod_norm = _clip01(active_mods / 4.0)
        if measured_ratio is None:
            measured_ratio = _clip01(0.45 * mod_norm + 0.30 * fv_raw + 0.25 * gv_raw)

    # 4. signal_usable_ratio
    sq_ratio: float | None = None
    if "signal_usable_ratio" in fv2:
        sq_ratio = _clip01(fv2["signal_usable_ratio"])
    elif isinstance(sq_payload, dict) and sq_payload.get("usable_window_ratio") is not None:
        sq_ratio = _clip01(sq_payload.get("usable_window_ratio"))
    elif isinstance(l1_dict.get("signal_quality_audit"), dict) and l1_dict["signal_quality_audit"].get("usable_window_ratio") is not None:
        sq_ratio = _clip01(l1_dict["signal_quality_audit"].get("usable_window_ratio"))
    elif (isinstance(face_df, pd.DataFrame) and len(face_df) >= _sqc.WINDOW_SIZE) or (
        isinstance(gesture_df, pd.DataFrame) and len(gesture_df) >= _sqc.WINDOW_SIZE
    ):
        sq_out = _sqc.audit_session_signal_quality(logs, session_id)
        if sq_out.get("usable_window_ratio") is not None:
            sq_ratio = _clip01(sq_out.get("usable_window_ratio"))

    if sq_ratio is None:
        vals = [v for v in (fv_raw, gv_raw) if v > 0.0]
        sq_ratio = _clip01(float(np.mean(vals)) if vals else 0.0)

    # 5. cross_modal_overlap_ratio
    if "cross_modal_overlap_ratio" in fv2:
        cm_ov = _clip01(fv2["cross_modal_overlap_ratio"])
    else:
        cmc = cmc_payload if isinstance(cmc_payload, dict) else (
            l1_dict.get("cross_modal_coverage") if isinstance(l1_dict.get("cross_modal_coverage"), dict) else None
        )
        if cmc is None and res_obj is not None:
            try:
                from report_frontend.cross_modal_coverage import evaluate_cross_modal_coverage
                cmc = evaluate_cross_modal_coverage(res_obj, logs=logs)
            except Exception:
                cmc = None
        fg_jacc = None
        fv_span = None
        if isinstance(cmc, dict) and isinstance(cmc.get("session"), dict):
            pairs = cmc["session"].get("pairs") or {}
            fg_jacc = (pairs.get("face_gesture") or {}).get("jaccard")
            fv_span = (pairs.get("face_voice") or {}).get("span_ratio")
        if fg_jacc is None:
            fg_jacc = _compute_fg_jaccard(face_df, gesture_df)
        if fv_span is None and used_whole_series and vs_list is not None:
            cand = vs_list if isinstance(vs_list, (list, tuple)) else [vs_list]
            whole = [x for x in cand if x is not None and getattr(x, "index", None) is None and getattr(x, "f0", None) is not None]
            if whole:
                aud_dur = len(whole[0].f0) * float(getattr(whole[0], "frame_sec", 0.04) or 0.04)
                vis_dur = 0.0
                for df in (face_df, gesture_df):
                    if isinstance(df, pd.DataFrame) and "timestamp" in df.columns and len(df) >= 2:
                        ts = pd.to_numeric(df["timestamp"], errors="coerce").dropna()
                        if len(ts) >= 2:
                            vis_dur = max(vis_dur, float(ts.max() - ts.min()))
                if vis_dur > 0.0 and aud_dur > 0.0:
                    fv_span = min(vis_dur, aud_dur) / max(vis_dur, aud_dur)

        vis_ov = _clip01(fg_jacc) if (fg_jacc is not None and math.isfinite(_safe_float(fg_jacc, float("nan")))) else min(fv_raw, gv_raw)
        sp_eff = min(1.0, speech_cov / 0.30) if speech_cov > 0.0 else 0.0
        cm_ov = min(vis_ov, sp_eff)
        if fv_span is not None and math.isfinite(_safe_float(fv_span, float("nan"))) and float(fv_span) < 0.70 and used_whole_series:
            cm_ov = min(cm_ov, _clip01(fv_span))
        cm_ov = _clip01(cm_ov)

    # 6. dynamics_stability_score
    if "dynamics_stability_score" in fv2:
        dyn_stab = _clip01(fv2["dynamics_stability_score"])
    else:
        dyn_stab = _compute_dynamics_stability(face_df, gesture_df)

    feat_map = {
        "measured_ratio": round(float(measured_ratio), 4),
        "valid_modality_count_norm": round(float(mod_norm), 4),
        "signal_usable_ratio": round(float(sq_ratio), 4),
        "cross_modal_overlap_ratio": round(float(cm_ov), 4),
        "face_valid_frame_ratio": round(float(fv_raw), 4),
        "audio_speech_coverage": round(float(speech_cov), 4),
        "gesture_continuity_ratio": round(float(gv_raw), 4),
        "dynamics_stability_score": round(float(dyn_stab), 4),
    }
    vec = np.asarray([feat_map[name] for name in EVIDENCE_FEATURE_NAMES], dtype=float)
    return vec, feat_map


def _rule_evidence_stage(feat_map: dict[str, float]) -> tuple[str, dict[str, float]]:
    meas = feat_map["measured_ratio"]
    n_mods = feat_map["valid_modality_count_norm"] * 4.0
    sq_u = feat_map["signal_usable_ratio"]
    cm_ov = feat_map["cross_modal_overlap_ratio"]
    fv = feat_map["face_valid_frame_ratio"]
    gv = feat_map["gesture_continuity_ratio"]
    sp = feat_map["audio_speech_coverage"]

    if meas >= 0.70 and n_mods >= 3.0 and sq_u >= 0.65 and cm_ov >= 0.55 and fv >= 0.60 and gv >= 0.60:
        return "stage_1_verified", {"stage_1_verified": 0.80, "stage_2_review": 0.16, "stage_3_limited": 0.04}
    if meas >= 0.25 and (n_mods >= 2.0 or sq_u >= 0.50 or sp >= 0.25):
        return "stage_2_review", {"stage_1_verified": 0.18, "stage_2_review": 0.68, "stage_3_limited": 0.14}
    return "stage_3_limited", {"stage_1_verified": 0.04, "stage_2_review": 0.16, "stage_3_limited": 0.80}


def _evaluate_evidence_gate(
    feat_vec: np.ndarray,
    feat_map: dict[str, float],
    gate_artifact: dict[str, Any] | None,
    metrics_data: dict[str, Any] | None,
) -> dict[str, Any]:
    emp_cov: float | None = None
    if isinstance(metrics_data, dict):
        gate_m = metrics_data.get("evidence_gate")
        if isinstance(gate_m, dict) and gate_m.get("empirical_conformal_coverage") is not None:
            emp_cov = _safe_float(gate_m["empirical_conformal_coverage"], 0.90)

    X_row = feat_vec.reshape(1, -1)
    if gate_artifact is not None:
        scaler = gate_artifact["scaler"]
        clf = gate_artifact["classifier"]
        iso_scaler = gate_artifact["iso_scaler"]
        iso = gate_artifact["isolation_forest"]
        q_hat = float(gate_artifact["q_hat"])
        alpha = float(gate_artifact.get("alpha", 0.10))
        classes = list(gate_artifact.get("classes", STAGE_ORDER))
        if emp_cov is None and gate_artifact.get("empirical_cv_coverage") is not None:
            emp_cov = _safe_float(gate_artifact["empirical_cv_coverage"], 0.90)

        Xz = scaler.transform(X_row)
        probs_arr = np.asarray(clf.predict_proba(Xz)[0], dtype=float)
        probs_arr = np.clip(probs_arr, 1e-6, 1.0)
        probs_arr = probs_arr / float(probs_arr.sum())
        stage_probs = {cls: round(float(probs_arr[i]), 4) for i, cls in enumerate(classes)}
        top_idx = int(np.argmax(probs_arr))
        stage_code = classes[top_idx]

        # Split Conformal Prediction 集合
        in_set_indices = [i for i, p in enumerate(probs_arr) if (1.0 - float(p)) <= q_hat]
        empty_cset = len(in_set_indices) == 0
        if empty_cset:
            in_set_indices = [top_idx]
        in_set_indices.sort(key=lambda i: probs_arr[i], reverse=True)
        conformal_set = [classes[i] for i in in_set_indices]

        X_iso_z = iso_scaler.transform(X_row)
        ood_score = round(float(-iso.decision_function(X_iso_z)[0]), 4)
        ood_anomaly_flag = bool(int(iso.predict(X_iso_z)[0]) == -1)
        uncertainty_escalated = bool(len(conformal_set) > 1 or empty_cset or ood_anomaly_flag)

        # 白盒加性特征归因：沿阶段 I 证据支持对比方向计算逐维贡献
        coef = np.asarray(clf.coef_, dtype=float)
        i_s1 = classes.index("stage_1_verified") if "stage_1_verified" in classes else 0
        i_s3 = classes.index("stage_3_limited") if "stage_3_limited" in classes else (len(classes) - 1)
        w_diff = coef[i_s1] - coef[i_s3]
        w_contrast = np.where(w_diff > 0.0, w_diff, np.maximum(coef[i_s1], 0.05))
        raw_contribs = w_contrast * Xz[0]
    else:
        alpha = 0.10
        stage_code, stage_probs = _rule_evidence_stage(feat_map)
        conformal_set = [stage_code] if stage_code != "stage_2_review" else ["stage_2_review", "stage_1_verified"]
        ood_score = None
        ood_anomaly_flag = False
        uncertainty_escalated = len(conformal_set) > 1
        # 规则回退下的标准化中心偏差归因
        raw_contribs = np.asarray([(feat_vec[j] - 0.55) * 1.5 for j in range(len(EVIDENCE_FEATURE_NAMES))], dtype=float)

    attributions: list[dict[str, Any]] = []
    for j, fname in enumerate(EVIDENCE_FEATURE_NAMES):
        contrib = round(float(raw_contribs[j]), 4)
        val_pct = feat_map[fname] * 100.0
        if fname == "valid_modality_count_norm":
            n_mods = int(round(feat_map[fname] * 4.0))
            val_disp = f"{n_mods}/4 路"
        else:
            val_disp = f"{val_pct:.1f}%"
        attributions.append({
            "feature_key": fname,
            "label": EVIDENCE_FEATURE_LABELS[fname],
            "value_display": val_disp,
            "contribution": contrib,
            "direction": "正向支撑" if contrib >= 0.0 else "待补强",
        })
    attributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)

    conformal_set_labels = [STAGE_LABELS[c] for c in conformal_set]
    pos_top = [a["label"] for a in attributions if a["direction"] == "正向支撑"][:2]
    neg_top = [a["label"] for a in attributions if a["direction"] == "待补强"][:2]

    parts: list[str] = [f"当前证据核验归为{STAGE_LABELS[stage_code]}"]
    if ood_anomaly_flag:
        parts.append("检测到跨模态采集分布异常，已触发人工复核提醒")
    elif uncertainty_escalated:
        parts.append(f"保形预测保留 {len(conformal_set)} 个候选阶段，建议人工复核边界项")
    else:
        parts.append("保形集合收敛至单一阶段，采集结构清晰")
    if pos_top:
        parts.append(f"主要正向支撑为{'与'.join(pos_top)}")
    if neg_top:
        parts.append(f"待补强环节为{'与'.join(neg_top)}")

    return {
        "stage_code": stage_code,
        "stage_label": STAGE_LABELS[stage_code],
        "stage_probs": stage_probs,
        "conformal_set": conformal_set,
        "conformal_set_labels": conformal_set_labels,
        "uncertainty_escalated": uncertainty_escalated,
        "alpha": round(alpha, 2),
        "target_coverage": round(1.0 - alpha, 2),
        "empirical_cv_coverage": round(emp_cov, 4) if emp_cov is not None else None,
        "ood_anomaly_flag": ood_anomaly_flag,
        "ood_score": ood_score,
        "attributions": attributions,
        "summary_plain": "，".join(parts) + "。",
    }


def _compute_transition_entropy(states: list[str]) -> float:
    n = len(states)
    if n < 2:
        return 0.0
    state_to_idx = {st: i for i, st in enumerate(MACRO_STAGE_ORDER)}
    counts = np.zeros((len(MACRO_STAGE_ORDER), len(MACRO_STAGE_ORDER)), dtype=float)
    for a, b in zip(states[:-1], states[1:]):
        ia = state_to_idx.get(a)
        ib = state_to_idx.get(b)
        if ia is not None and ib is not None:
            counts[ia, ib] += 1.0
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    row_sums = counts.sum(axis=1)
    entropy = 0.0
    for i in range(len(MACRO_STAGE_ORDER)):
        if row_sums[i] <= 0.0:
            continue
        pi = row_sums[i] / total
        probs = counts[i] / row_sums[i]
        nz = probs > 0.0
        h_i = -float(np.sum(probs[nz] * np.log(probs[nz])))
        entropy += pi * h_i
    return round(max(0.0, entropy), 4)


def _describe_changepoint_cues(pre_mean: np.ndarray, post_mean: np.ndarray) -> str:
    diff = post_mean - pre_mean
    cue_map: dict[str, tuple[str, str]] = {
        "face_valid_ratio": ("面部有效检出提升", "面部有效检出下降"),
        "mouth_open_mean": ("口部表达开合增强", "口部动作转平缓"),
        "mouth_motion_std": ("口部动态变化增强", "口部节律趋平稳"),
        "gaze_dev_mean": ("视线偏移切换增多", "视线注视趋集中"),
        "head_motion_std": ("头部动作幅度增大", "头部姿态趋稳定"),
        "upper_body_motion": ("上肢活动幅值上升", "上肢动作明显收敛"),
        "hand_visible_or_spread": ("手势幅度展开", "手势动作回落"),
        "voiced_ratio": ("发声连续度提升", "发声停顿明显增多"),
    }
    ranked = sorted(
        enumerate(_brm.FEATURE_NAMES),
        key=lambda item: abs(float(diff[item[0]])),
        reverse=True,
    )
    cues: list[str] = []
    seen_groups: set[str] = set()
    for idx, fname in ranked:
        d = float(diff[idx])
        if abs(d) < 0.35:
            continue
        grp = "mouth" if fname.startswith("mouth_") else fname
        if grp in seen_groups:
            continue
        seen_groups.add(grp)
        pos_txt, neg_txt = cue_map.get(fname, ("行为节律增强", "行为节律趋平缓"))
        cues.append(pos_txt if d > 0.0 else neg_txt)
        if len(cues) >= 2:
            break
    return "，".join(cues) if cues else "多模态动作节律出现结构切换"


def _detect_pelt_changepoints(
    X_scaled: np.ndarray,
    meta: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    n = int(len(X_scaled))
    if n < 8 or len(meta) != n:
        return []
    try:
        import ruptures as rpt
    except Exception:
        return []

    if n > 120:
        stride_b = int(math.ceil(n / 100.0))
        m = n // stride_b
        if m < 8:
            return []
        X_in = X_scaled[: m * stride_b].reshape(m, stride_b, -1).mean(axis=1)
        scale_f = stride_b
        jump = 2 if len(X_in) > 50 else 1
    else:
        X_in = X_scaled
        scale_f = 1
        jump = 2 if n > 50 else 1

    pen = max(1.8, min(3.6, 0.65 * math.log(max(len(X_in), 4))))
    try:
        bkps = rpt.Pelt(model="rbf", min_size=3, jump=jump).fit_predict(X_in, pen=pen)
    except Exception:
        return []

    change_points: list[dict[str, Any]] = []
    for raw_b in bkps[:-1]:
        w_idx = int(min(n - 3, max(3, raw_b * scale_f)))
        pre = X_scaled[max(0, w_idx - 5) : w_idx].mean(axis=0)
        post = X_scaled[w_idx : min(n, w_idx + 5)].mean(axis=0)
        mag = round(float(np.linalg.norm(post - pre)), 3)
        t_sec = round(float(meta[w_idx]["start_sec"]), 2)
        change_points.append({
            "time_sec": t_sec,
            "window_idx": w_idx,
            "magnitude": mag,
            "cue_summary": _describe_changepoint_cues(pre, post),
        })
    return change_points


def _evaluate_stage_segmentation(
    logs: dict[str, Any],
    session_id: str | None,
    hmm_artifact: dict[str, Any] | None,
    *,
    voice_list: Any = None,
) -> dict[str, Any]:
    face_df = logs.get("face")
    gesture_df = logs.get("gesture")
    if voice_list is None and session_id:
        try:
            from report_frontend import voice_series as _vs_mod
            voice_list = _vs_mod.load_series(session_id)
        except Exception:
            voice_list = None

    X, meta = _brm.extract_regime_windows(face_df, gesture_df, voice_series=voice_list)
    if len(X) == 0:
        return {
            "available": False,
            "method": "GaussianHMM (4状态Markov) + PELT变点检测",
            "n_windows": 0,
            "total_duration_sec": 0.0,
            "transition_count": 0,
            "mean_dwell_sec": 0.0,
            "transition_entropy": 0.0,
            "flicker_reduction_pct": None,
            "macro_stages": [],
            "change_points": [],
            "summary_plain": "连续有效观测帧不足，暂未切分宏观行为阶段。",
        }

    if hmm_artifact is not None:
        prep = hmm_artifact["preprocessor"]
        hmm = hmm_artifact["hmm"]
        state_map = hmm_artifact["state_map"]
        X_scaled = prep.transform(X)
        hmm_ids = hmm.predict(X_scaled)
        states = [state_map[int(cid)] for cid in hmm_ids]

        gmm_ref = hmm_artifact.get("gmm_ref")
        gmm_map = hmm_artifact.get("gmm_map")
        if gmm_ref is not None and gmm_map is not None:
            gmm_ids = gmm_ref.predict(X_scaled)
            gmm_states = [gmm_map[int(cid)] for cid in gmm_ids]
            gmm_trans = sum(1 for a, b in zip(gmm_states[:-1], gmm_states[1:]) if a != b)
            hmm_trans_raw = sum(1 for a, b in zip(states[:-1], states[1:]) if a != b)
            if gmm_trans > 0:
                flicker_red = round(((gmm_trans - hmm_trans_raw) / float(gmm_trans)) * 100.0, 2)
            else:
                flicker_red = 0.0
        else:
            flicker_red = None
    else:
        raw_states = _brm._rule_predict_states(X)
        states = _brm.smooth_state_sequence(raw_states, window=3)
        X_scaled = np.nan_to_num(X, nan=0.0)
        col_std = np.where(X_scaled.std(axis=0) > 1e-6, X_scaled.std(axis=0), 1.0)
        X_scaled = (X_scaled - X_scaled.mean(axis=0)) / col_std
        flicker_red = None

    raw_segments = _brm.merge_state_segments(states, meta)
    total_duration_sec = round(sum(float(s["duration_sec"]) for s in raw_segments), 2)

    macro_stages: list[dict[str, Any]] = []
    n_segs = max(len(raw_segments), 1)
    for seg in raw_segments:
        st = seg["state"]
        info = MACRO_STAGE_META[st]
        dur = float(seg["duration_sec"])
        ratio = round(dur / total_duration_sec, 4) if total_duration_sec > 0.0 else round(1.0 / n_segs, 4)
        macro_stages.append({
            "stage": st,
            "label": info["label"],
            "start_sec": float(seg["start_sec"]),
            "end_sec": float(seg["end_sec"]),
            "duration_sec": dur,
            "color_key": info["color_key"],
            "ratio": ratio,
        })

    # 若开场阶段出现明显早期适应波动且后续切入平稳展开期，则自然标注「适应建立期」
    adaptation_noted = False
    if len(macro_stages) >= 2:
        first_seg = macro_stages[0]
        second_seg = macro_stages[1]
        early_cap = max(35.0, 0.25 * max(total_duration_sec, 1.0))
        if (
            first_seg["stage"] != "steady_speaking"
            and second_seg["stage"] == "steady_speaking"
            and first_seg["duration_sec"] <= early_cap
        ):
            first_seg["annotation"] = "适应建立期"
            adaptation_noted = True

    transition_count = max(0, len(macro_stages) - 1)
    mean_dwell_sec = round(total_duration_sec / len(macro_stages), 2) if macro_stages else 0.0
    transition_entropy = _compute_transition_entropy(states)
    change_points = _detect_pelt_changepoints(X_scaled, meta)

    dur_by_stage = Counter()
    for seg in macro_stages:
        dur_by_stage[seg["stage"]] += float(seg["duration_sec"])
    ranked_stages = sorted(
        ((st, dur_by_stage.get(st, 0.0) / total_duration_sec if total_duration_sec > 0 else 0.0) for st in MACRO_STAGE_ORDER),
        key=lambda item: item[1],
        reverse=True,
    )
    ratio_parts = [
        f"{MACRO_STAGE_META[st]['label']}占比 {round(r * 100)}%"
        for st, r in ranked_stages
        if round(r * 100) > 0
    ]
    summary_segs = [f"整场切分为 {len(macro_stages)} 个连续行为阶段"]
    if ratio_parts:
        summary_segs.append("，".join(ratio_parts[:3]))
    if adaptation_noted:
        summary_segs.append("开场前段呈现适应建立期后切入平稳节奏")
    if change_points:
        summary_segs.append(f"共定位 {len(change_points)} 处行为转折点")
    else:
        summary_segs.append("全程节律过渡平缓")

    return {
        "available": True,
        "method": "GaussianHMM (4状态Markov) + PELT变点检测",
        "n_windows": int(len(X)),
        "total_duration_sec": total_duration_sec,
        "transition_count": int(transition_count),
        "mean_dwell_sec": mean_dwell_sec,
        "transition_entropy": transition_entropy,
        "flicker_reduction_pct": flicker_red,
        "macro_stages": macro_stages,
        "change_points": change_points,
        "summary_plain": "，".join(summary_segs) + "。",
    }


def _empty_result(session_id: str | None, metrics_data: dict[str, Any] | None) -> dict[str, Any]:
    emp_cov: float | None = None
    if isinstance(metrics_data, dict):
        gm = metrics_data.get("evidence_gate")
        if isinstance(gm, dict) and gm.get("empirical_conformal_coverage") is not None:
            emp_cov = round(_safe_float(gm["empirical_conformal_coverage"], 0.90), 4)
    return {
        "available": False,
        "advisory_only": True,
        "non_scoring_guarantee": NON_SCORING_GUARANTEE,
        "session_id": session_id,
        "evidence_gate": {
            "stage_code": "unavailable",
            "stage_label": STAGE_LABELS["unavailable"],
            "stage_probs": {c: 0.0 for c in STAGE_ORDER},
            "conformal_set": [],
            "conformal_set_labels": [],
            "uncertainty_escalated": False,
            "alpha": 0.10,
            "target_coverage": 0.90,
            "empirical_cv_coverage": emp_cov,
            "ood_anomaly_flag": False,
            "ood_score": None,
            "attributions": [],
            "summary_plain": "当前场次暂无有效多模态观测日志，暂未执行证据充分度核验。",
        },
        "stage_segmentation": {
            "available": False,
            "method": "GaussianHMM (4状态Markov) + PELT变点检测",
            "n_windows": 0,
            "total_duration_sec": 0.0,
            "transition_count": 0,
            "mean_dwell_sec": 0.0,
            "transition_entropy": 0.0,
            "flicker_reduction_pct": None,
            "macro_stages": [],
            "change_points": [],
            "summary_plain": "当前场次暂无有效连续观测帧，暂未切分宏观行为阶段。",
        },
        "model_metrics": metrics_data,
        "disclaimer": DISCLAIMER_TEXT,
        "inference_ms": 0.0,
    }


def evaluate_referee_ml(
    logs: dict[str, pd.DataFrame] | None,
    session_id: str | None = None,
    l1_summary: dict[str, Any] | None = None,
    feature_v2: dict[str, Any] | None = None,
    *,
    model_dir: Path | None = None,
) -> dict[str, Any]:
    """对单场多模态采集会话执行纯 CPU 宏观阶段切分与证据核验门控。"""
    mdir = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
    metrics_path = mdir / "referee_ml_metrics.json"
    if not metrics_path.is_file():
        metrics_path = DEFAULT_METRICS_PATH
    metrics_data = _load_metrics_cached(metrics_path)

    has_rows = _has_any_log_rows(logs)
    has_summary = isinstance(l1_summary, dict) and bool(l1_summary)
    has_fv2 = isinstance(feature_v2, dict) and bool(feature_v2)

    if not has_rows and not has_summary and not has_fv2:
        return _empty_result(session_id, metrics_data)

    t0 = time.perf_counter()

    hmm_artifact = _load_joblib_cached(
        mdir / "referee_hmm_model.joblib",
        _HMM_CACHE,
        ("preprocessor", "hmm", "state_map"),
    )
    gate_artifact = _load_joblib_cached(
        mdir / "referee_conformal_gate.joblib",
        _GATE_CACHE,
        ("scaler", "classifier", "iso_scaler", "isolation_forest", "q_hat"),
    )

    voice_list = None
    if session_id:
        try:
            from report_frontend import voice_series as _vs_mod
            voice_list = _vs_mod.load_series(session_id)
        except Exception:
            voice_list = None

    feat_vec, feat_map = extract_evidence_features(
        logs,
        session_id=session_id,
        l1_summary=l1_summary,
        feature_v2=feature_v2,
        voice_series_list=voice_list,
    )
    evidence_gate = _evaluate_evidence_gate(feat_vec, feat_map, gate_artifact, metrics_data)
    stage_segmentation = _evaluate_stage_segmentation(
        logs or {}, session_id, hmm_artifact, voice_list=voice_list
    )

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "available": True,
        "advisory_only": True,
        "non_scoring_guarantee": NON_SCORING_GUARANTEE,
        "session_id": session_id,
        "evidence_gate": evidence_gate,
        "stage_segmentation": stage_segmentation,
        "model_metrics": metrics_data,
        "disclaimer": DISCLAIMER_TEXT,
        "inference_ms": elapsed_ms,
    }
