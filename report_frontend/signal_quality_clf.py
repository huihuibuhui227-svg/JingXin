# report_frontend/signal_quality_clf.py
"""自监督采集信号质量异常检测器（纯 CPU 毫秒级，仅作仪器质检提示，绝不参与评定）。

架构定位（见 `07_深度学习怎么用与创新点怎么立.md` §4）：
- 本模块属于**仪器质检辅助层**，仅对 L0 滑动时间窗（10 帧）内的面部与上肢遥测统计量
  判断「该时间窗是否存在遮挡、画面冻结、时间戳抖动或传感器异常尖峰」；
- 严格只读（`advisory_only: True`），不修改任何 L1 数值、置信度档位或维度结果；
- 无受试者身份特征，训练验证严格按 `session_id` 分组（GroupKFold），防止同场泄漏。
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "signal_quality_clf.joblib"
DEFAULT_METRICS_PATH = Path(__file__).resolve().parent / "signal_quality_metrics.json"

WINDOW_SIZE = 10
WINDOW_STRIDE = 5
USABILITY_THRESHOLD = 0.50

FEATURE_NAMES: tuple[str, ...] = (
    "face_valid_ratio",
    "au_finite_ratio",
    "au_temporal_diff_mean",
    "au_cross_channel_std",
    "gaze_finite_ratio",
    "gaze_std",
    "head_yaw_abs_p90",
    "head_yaw_diff_std",
    "face_dt_cv",
    "gesture_valid_ratio",
    "wrist_jitter_finite_ratio",
    "wrist_jitter_p95",
)

_MODEL_CACHE: dict[str, Any] = {"path": None, "mtime": None, "model": None}


def _num_series(df: pd.DataFrame | None, col: str) -> np.ndarray:
    if df is None or df.empty or col not in df.columns:
        return np.array([], dtype=float)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _safe_stat(arr: np.ndarray, fn, default: float = float("nan")) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return default
    val = float(fn(finite))
    return val if math.isfinite(val) else default


def extract_window_features(
    face_df: pd.DataFrame | None,
    gesture_df: pd.DataFrame | None,
    *,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """将单场面部/手势 L0 表切分为滑动时间窗并提取 12 维纯信号质量统计量。"""
    n_face = len(face_df) if isinstance(face_df, pd.DataFrame) else 0
    n_gest = len(gesture_df) if isinstance(gesture_df, pd.DataFrame) else 0
    n_total = max(n_face, n_gest)
    if n_total < window_size:
        return np.empty((0, len(FEATURE_NAMES)), dtype=float), []

    au_cols = (
        [c for c in face_df.columns if c.startswith("au") and "_" in c][:8]
        if isinstance(face_df, pd.DataFrame) and not face_df.empty
        else []
    )

    feats: list[list[float]] = []
    meta: list[dict[str, Any]] = []

    for start in range(0, n_total - window_size + 1, stride):
        end = start + window_size
        f_win = face_df.iloc[start:end] if (isinstance(face_df, pd.DataFrame) and start < n_face) else None
        g_win = gesture_df.iloc[start:end] if (isinstance(gesture_df, pd.DataFrame) and start < n_gest) else None

        # 1. 面部覆盖率与 AU 统计
        if f_win is not None and len(f_win) > 0 and au_cols:
            au_mat = np.column_stack([_num_series(f_win, c) for c in au_cols])
            row_valid = np.isfinite(au_mat).any(axis=1)
            face_valid_ratio = float(row_valid.mean())
            au_finite_ratio = float(np.isfinite(au_mat).mean())
            if au_mat.shape[0] >= 2 and np.isfinite(au_mat).any():
                diffs = np.abs(np.diff(au_mat, axis=0))
                au_diff_mean = _safe_stat(diffs.ravel(), np.mean, default=0.0)
            else:
                au_diff_mean = float("nan")
            col_stds = [
                float(np.std(au_mat[:, j][np.isfinite(au_mat[:, j])]))
                for j in range(au_mat.shape[1])
                if np.isfinite(au_mat[:, j]).sum() >= 2
            ]
            au_cross_std = float(np.mean(col_stds)) if col_stds else float("nan")
        else:
            face_valid_ratio = 0.0
            au_finite_ratio = 0.0
            au_diff_mean = float("nan")
            au_cross_std = float("nan")

        # 2. 视线与头部姿态稳定性
        gx = _num_series(f_win, "gaze_direction_x")
        gy = _num_series(f_win, "gaze_direction_y")
        gaze_finite_ratio = float(np.isfinite(gx).mean()) if gx.size else 0.0
        gaze_std_x = _safe_stat(gx, np.std, default=float("nan"))
        gaze_std_y = _safe_stat(gy, np.std, default=float("nan"))
        gaze_std = float(np.nanmean([gaze_std_x, gaze_std_y])) if (
            math.isfinite(gaze_std_x) or math.isfinite(gaze_std_y)
        ) else float("nan")

        yaw = _num_series(f_win, "head_yaw")
        yaw_abs_p90 = _safe_stat(np.abs(yaw), lambda a: np.percentile(a, 90), default=float("nan"))
        yaw_diff_std = _safe_stat(np.diff(yaw), np.std, default=float("nan")) if yaw.size >= 2 else float("nan")

        ts = _num_series(f_win, "timestamp")
        if ts.size >= 3:
            dt = np.diff(ts)
            dt_pos = dt[np.isfinite(dt) & (dt > 0)]
            if dt_pos.size >= 2 and float(np.mean(dt_pos)) > 0:
                face_dt_cv = float(np.std(dt_pos) / np.mean(dt_pos))
            else:
                face_dt_cv = 1.0
        else:
            face_dt_cv = float("nan")

        # 3. 手势/体态覆盖率与手腕抖动量级
        lw = _num_series(g_win, "left_wrist_jitter_world")
        if lw.size == 0:
            lw = _num_series(g_win, "left_hand_jitter")
        rw = _num_series(g_win, "right_wrist_jitter_world")
        if rw.size == 0:
            rw = _num_series(g_win, "right_hand_jitter")

        if g_win is not None and len(g_win) > 0:
            cand_cols = [c for c in ("left_shoulder_angle", "head_tilt_angle", "left_arm_angle",
                                     "left_hand_jitter", "right_hand_jitter") if c in g_win.columns]
            if cand_cols:
                g_mat = np.column_stack([_num_series(g_win, c) for c in cand_cols])
                gesture_valid_ratio = float(np.isfinite(g_mat).any(axis=1).mean())
            else:
                gesture_valid_ratio = 0.0
        else:
            gesture_valid_ratio = 0.0

        both_w = np.concatenate([lw, rw]) if (lw.size or rw.size) else np.array([], dtype=float)
        wrist_finite = float(np.isfinite(both_w).mean()) if both_w.size else 0.0
        wrist_p95 = _safe_stat(both_w, lambda a: np.percentile(a, 95), default=float("nan"))

        row = [
            face_valid_ratio,
            au_finite_ratio,
            au_diff_mean,
            au_cross_std,
            gaze_finite_ratio,
            gaze_std,
            yaw_abs_p90,
            yaw_diff_std,
            face_dt_cv,
            gesture_valid_ratio,
            wrist_finite,
            wrist_p95,
        ]
        feats.append(row)
        meta.append({"start_row": start, "end_row": end})

    return np.asarray(feats, dtype=float), meta


def _diagnose_window_hints(row: np.ndarray) -> list[str]:
    """依据物理信号特征解释窗口为何疑似信号降质（供运维与采集核验参考）。"""
    hints: list[str] = []
    face_cov, _au_fin, au_diff, _au_std, _gz_fin, _gz_std, yaw_p90, _yd, dt_cv, gest_cov, _wf, w_p95 = row
    if math.isfinite(face_cov) and face_cov < 0.5:
        hints.append("面部检出中断/疑似遮挡")
    if math.isfinite(au_diff) and math.isfinite(face_cov) and face_cov >= 0.5 and au_diff < 1e-4:
        hints.append("画面或动作序列疑似冻结")
    if math.isfinite(dt_cv) and dt_cv > 0.65:
        hints.append("采集帧间隔抖动偏大")
    if math.isfinite(yaw_p90) and yaw_p90 > 0.65:
        hints.append("头部大幅偏转出镜")
    if math.isfinite(gest_cov) and gest_cov < 0.35:
        hints.append("上肢与体态检出率偏低")
    if math.isfinite(w_p95) and w_p95 > 0.45:
        hints.append("上肢坐标出现高频尖峰噪点")
    return hints or ["多特征联合偏离常规采集分布"]


def _rule_usability_prob(X: np.ndarray) -> np.ndarray:
    """无模型文件时的纯规则降级估计（仅返回 0~1 可用度参考）。"""
    probs = np.ones(len(X), dtype=float)
    for i, row in enumerate(X):
        face_cov, _au_fin, au_diff, _au_std, _gz_fin, _gz_std, yaw_p90, _yd, dt_cv, gest_cov, _wf, w_p95 = row
        pen = 0.0
        if not math.isfinite(face_cov) or face_cov < 0.5:
            pen += 0.60
        if math.isfinite(au_diff) and face_cov >= 0.5 and au_diff < 1e-4:
            pen += 0.55
        if math.isfinite(dt_cv) and dt_cv > 0.65:
            pen += 0.25
        if math.isfinite(yaw_p90) and yaw_p90 > 0.65:
            pen += 0.20
        if math.isfinite(gest_cov) and gest_cov < 0.35:
            pen += 0.15
        if math.isfinite(w_p95) and w_p95 > 0.45:
            pen += 0.20
        probs[i] = max(0.0, min(1.0, 1.0 - pen))
    return probs


def _load_model(model_path: Path = DEFAULT_MODEL_PATH):
    if not model_path.is_file():
        return None
    try:
        mtime = model_path.stat().st_mtime_ns
        if _MODEL_CACHE["path"] == str(model_path) and _MODEL_CACHE["mtime"] == mtime:
            return _MODEL_CACHE["model"]
        import joblib
        mdl = joblib.load(model_path)
        _MODEL_CACHE.update({"path": str(model_path), "mtime": mtime, "model": mdl})
        return mdl
    except Exception:
        return None


def _load_cv_metrics(metrics_path: Path = DEFAULT_METRICS_PATH) -> dict[str, Any] | None:
    if not metrics_path.is_file():
        return None
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def audit_session_signal_quality(
    logs: dict[str, pd.DataFrame] | None,
    session_id: str | None = None,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
) -> dict[str, Any]:
    """对单场 L0 遥测执行毫秒级采集质量质检（只读提示，不改写任何指标或裁决）。"""
    t0 = time.perf_counter()
    face_df = (logs or {}).get("face")
    gesture_df = (logs or {}).get("gesture")
    X, meta = extract_window_features(face_df, gesture_df)
    cv_metrics = _load_cv_metrics(metrics_path)

    base_disclaimer = (
        "仅用于检测采集端画面遮挡、画面冻结、帧间隔抖动或坐标尖峰等信号质量异常，"
        "严格不参与任何行为指标计算、置信度裁决或人物评定。"
    )

    if len(X) == 0:
        return {
            "status": "insufficient_windows",
            "advisory_only": True,
            "session_id": session_id,
            "n_windows": 0,
            "usable_window_ratio": None,
            "usable_windows": 0,
            "flagged_windows": [],
            "fault_hint_counts": {},
            "summary_plain": "本场有效连续帧数不足一个质检时间窗（10 帧），未执行滑动窗信号质量检测。",
            "disclaimer": base_disclaimer,
            "model_used": False,
            "model_type": None,
            "cv_metrics": cv_metrics,
            "inference_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    mdl = _load_model(model_path)
    if mdl is not None:
        probs = np.asarray(mdl.predict_proba(X)[:, 1], dtype=float)
        mode_status = "ready"
        model_used = True
        model_type = type(mdl).__name__
    else:
        probs = _rule_usability_prob(X)
        mode_status = "fallback_rule"
        model_used = False
        model_type = "rule_fallback"

    usable_mask = probs >= USABILITY_THRESHOLD
    n_usable = int(usable_mask.sum())
    ratio = round(float(n_usable) / len(X), 4)

    flagged: list[dict[str, Any]] = []
    hint_counts: dict[str, int] = {}
    for idx, (p, m, row) in enumerate(zip(probs, meta, X)):
        if p < USABILITY_THRESHOLD:
            hints = _diagnose_window_hints(row)
            for h in hints:
                hint_counts[h] = hint_counts.get(h, 0) + 1
            if len(flagged) < 12:
                flagged.append({
                    "window_index": idx,
                    "start_row": m["start_row"],
                    "end_row": m["end_row"],
                    "usability_prob": round(float(p), 3),
                    "hints": hints,
                })

    if not flagged:
        summary_plain = (
            f"共巡检 {len(X)} 个采集滑动窗（每窗 {WINDOW_SIZE} 帧），"
            f"信号可用率 {ratio * 100:.1f}%（{n_usable}/{len(X)}），未发现明显遮挡、冻结或尖峰异常。"
        )
    else:
        top_hint = max(hint_counts.items(), key=lambda kv: kv[1])[0]
        summary_plain = (
            f"共巡检 {len(X)} 个采集滑动窗，信号可用率 {ratio * 100:.1f}%（{n_usable}/{len(X)}），"
            f"检出 {len(X) - n_usable} 个疑似信号降质窗（主要提示：{top_hint}）。"
        )

    return {
        "status": mode_status,
        "advisory_only": True,
        "session_id": session_id,
        "n_windows": int(len(X)),
        "usable_windows": n_usable,
        "usable_window_ratio": ratio,
        "flagged_count": int(len(X) - n_usable),
        "flagged_windows": flagged,
        "fault_hint_counts": hint_counts,
        "summary_plain": summary_plain,
        "disclaimer": base_disclaimer,
        "model_used": model_used,
        "model_type": model_type,
        "threshold": USABILITY_THRESHOLD,
        "cv_metrics": cv_metrics,
        "inference_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }
