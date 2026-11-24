# report_frontend/blendshape_calibration.py
"""面部关键点几何代理与面部动作模型系数的会内时序去噪与头部姿态去偏校准模块。

本模块对会话内有效面部观测帧执行画面质量门控、零相位 Savitzky-Golay 平滑、
静息基线对齐与头部姿态协变量岭回归残差化，输出各面部动作构念的校准前后
一致性及融合可用性分级，仅用于跨仪器一致性校核，不修改主表读数。
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from report_frontend.face_dual_instrument_audit import (
    CONSTRUCT_PAIRS,
    FLOOR_EPS,
    FLOOR_RATE_INACTIVE_THRESHOLD,
    STD_EPS,
    ConstructPairSpec,
    _num_array,
    _paired_bs_array,
)

MIN_RHO_FRAMES: int = 4
MIN_CALIBRATION_FRAMES: int = 10
MIN_FUSION_FRAMES: int = 20
FUSION_RHO_THRESHOLD: float = 0.55
ASSIST_RHO_THRESHOLD: float = 0.30
BLUR_MIN_THRESHOLD: float = 15.0
MAX_SAVGOL_WINDOW: int = 7
SAVGOL_POLYORDER: int = 2
RIDGE_ALPHA: float = 1.0

_DISCLAIMER: str = (
    "本校准结果仅反映面部关键点几何测算与面部动作模型输出之间的帧级一致性变化，"
    "不构成对受测者的任何能力或状态判定。"
)


def _finite_or_none(val: float | None) -> float | None:
    if val is None:
        return None
    f = float(val)
    return f if math.isfinite(f) else None


def _round_or_none(val: float | None, digits: int = 6) -> float | None:
    f = _finite_or_none(val)
    return None if f is None else round(f, digits)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < MIN_RHO_FRAMES or y.size < MIN_RHO_FRAMES or x.size != y.size:
        return None
    if float(np.std(x)) <= STD_EPS or float(np.std(y)) <= STD_EPS:
        return None
    stat = spearmanr(x, y).statistic
    return _finite_or_none(stat)


def _smooth_series(values: np.ndarray) -> np.ndarray:
    """对单条连续序列执行零相位时序平滑，过短序列回退到滚动中位数。"""
    n = int(values.size)
    if n >= 5:
        win = min(MAX_SAVGOL_WINDOW, n if n % 2 == 1 else n - 1)
        return savgol_filter(values, window_length=win, polyorder=SAVGOL_POLYORDER).astype(float)
    if n >= 3:
        return (
            pd.Series(values, dtype=float)
            .rolling(window=3, center=True, min_periods=1)
            .median()
            .to_numpy(dtype=float)
        )
    return values.astype(float, copy=True)


def _detect_pose_mode(face_df: pd.DataFrame) -> str:
    yaw_3d = _num_array(face_df, "head_yaw_3d")
    pitch_3d = _num_array(face_df, "head_pitch_3d")
    if yaw_3d is not None and pitch_3d is not None:
        if bool((np.isfinite(yaw_3d) & np.isfinite(pitch_3d)).any()):
            return "3d_matrix"

    yaw_2d = _num_array(face_df, "head_yaw")
    pitch_2d = _num_array(face_df, "head_pitch")
    scale = _num_array(face_df, "face_scale")
    has_2d_pose = (
        yaw_2d is not None
        and pitch_2d is not None
        and bool((np.isfinite(yaw_2d) & np.isfinite(pitch_2d)).any())
    )
    has_scale = scale is not None and bool((np.isfinite(scale) & (scale > 0.0)).any())
    if has_2d_pose or has_scale:
        return "2d_legacy"
    return "unavailable"


def _build_quality_mask(face_df: pd.DataFrame, bs_cols: list[str]) -> np.ndarray:
    n_rows = len(face_df)
    mask = np.ones(n_rows, dtype=bool)

    q_ok = _num_array(face_df, "frame_quality_ok")
    if q_ok is not None and bool(np.isfinite(q_ok).any()):
        mask &= q_ok == 1.0

    blur = _num_array(face_df, "frame_blur_laplacian")
    if blur is not None and bool(np.isfinite(blur).any()):
        mask &= np.isfinite(blur) & (blur >= BLUR_MIN_THRESHOLD)

    scale = _num_array(face_df, "face_scale")
    if scale is not None and bool(np.isfinite(scale).any()):
        mask &= np.isfinite(scale) & (scale > 0.0)

    bs_stacked = np.column_stack(
        [pd.to_numeric(face_df[c], errors="coerce").to_numpy(dtype=float) for c in bs_cols]
    )
    mask &= np.isfinite(bs_stacked).any(axis=1)
    return mask


def _confounder_candidates(pose_mode: str) -> tuple[str, ...]:
    if pose_mode == "3d_matrix":
        return ("head_yaw_3d", "head_pitch_3d", "head_roll_3d", "face_scale")
    if pose_mode == "2d_legacy":
        return ("head_yaw", "head_pitch", "face_scale")
    return ()


def _select_confounders(
    face_df: pd.DataFrame,
    base_ok: np.ndarray,
    pose_mode: str,
) -> tuple[list[np.ndarray], np.ndarray]:
    candidates = _confounder_candidates(pose_mode)
    if not candidates or not bool(base_ok.any()):
        return [], base_ok

    n_base = int(base_ok.sum())
    selected_arrays: list[np.ndarray] = []
    ok = base_ok.copy()
    for col in candidates:
        arr = _num_array(face_df, col)
        if arr is None:
            continue
        finite_on_base = int((base_ok & np.isfinite(arr)).sum())
        if finite_on_base == 0:
            continue
        # 可选协变量（如缺失较多的尺度或横滚角）不因局部缺值而丢弃整批有效帧
        if finite_on_base < max(1, int(0.8 * n_base)) and col in ("head_roll_3d", "face_scale"):
            continue
        selected_arrays.append(arr)
        ok &= np.isfinite(arr)

    return selected_arrays, ok


def _build_construct_note(
    *,
    missing_cols: bool,
    n_valid: int,
    inactive_reason: str | None,
    eligibility: str,
    calibrated_rho: float | None,
    pose_r2: float | None,
) -> str:
    if missing_cols or n_valid < MIN_RHO_FRAMES:
        return "当前场次缺少该动作的完整观测帧，暂无法完成跨方式核验。"
    if inactive_reason is not None:
        return f"{inactive_reason}，建议在采集三维头部姿态后继续核验。"
    assert calibrated_rho is not None
    if eligibility == "已校准可融合":
        if pose_r2 is not None and pose_r2 >= 0.05:
            return (
                f"经时序去噪与头部姿态去偏后，两侧一致度达到 {calibrated_rho:.2f}，"
                f"头部姿态约解释了 {pose_r2:.0%} 的原始变化，已达到稳定融合条件。"
            )
        return f"经时序去噪与头部姿态去偏后，两侧一致度达到 {calibrated_rho:.2f}，已达到稳定融合条件。"
    if eligibility == "辅助对照":
        return f"校准后两侧一致度为 {calibrated_rho:.2f}，存在一定测量口径差异，建议仅作辅助对照使用。"
    return f"校准后两侧一致度为 {calibrated_rho:.2f}，波动差异仍较大，需依托三维头部姿态重采样本进一步核验。"


def _calibrate_single_construct(
    face_df: pd.DataFrame,
    clean_mask: np.ndarray,
    pose_mode: str,
    spec: ConstructPairSpec,
) -> dict[str, Any]:
    geo = _num_array(face_df, spec.geometric_col)
    bs = _paired_bs_array(face_df, spec.blendshape_cols)
    if geo is None or bs is None:
        return {
            "key": spec.construct,
            "label": spec.label,
            "n_valid": 0,
            "raw_rho": None,
            "calibrated_rho": None,
            "rho_delta": None,
            "pose_r2": None,
            "eligibility": "待3D重采验证",
            "note": _build_construct_note(
                missing_cols=True,
                n_valid=0,
                inactive_reason=None,
                eligibility="待3D重采验证",
                calibrated_rho=None,
                pose_r2=None,
            ),
        }

    base_ok = clean_mask & np.isfinite(geo) & np.isfinite(bs)
    conf_arrays, ok = _select_confounders(face_df, base_ok, pose_mode)
    n_valid = int(ok.sum())
    if n_valid < MIN_RHO_FRAMES:
        return {
            "key": spec.construct,
            "label": spec.label,
            "n_valid": n_valid,
            "raw_rho": None,
            "calibrated_rho": None,
            "rho_delta": None,
            "pose_r2": None,
            "eligibility": "待3D重采验证",
            "note": _build_construct_note(
                missing_cols=False,
                n_valid=n_valid,
                inactive_reason=None,
                eligibility="待3D重采验证",
                calibrated_rho=None,
                pose_r2=None,
            ),
        }

    g_raw = geo[ok]
    b_raw = bs[ok]
    g_aligned = g_raw * float(spec.geo_polarity)

    geo_floor_rate = float(np.mean(g_raw <= FLOOR_EPS))
    bs_floor_rate = float(np.mean(b_raw <= FLOOR_EPS))
    raw_rho = _safe_spearman(g_aligned, b_raw)

    g_smooth = _smooth_series(g_aligned)
    b_smooth = _smooth_series(b_raw)

    g_shifted = g_smooth - float(np.percentile(g_smooth, 10))
    b_shifted = b_smooth - float(np.percentile(b_smooth, 10))

    active_conf = [arr[ok] for arr in conf_arrays if float(np.std(arr[ok])) > STD_EPS]
    if active_conf and n_valid >= len(active_conf) + 2:
        X = np.column_stack(active_conf)
        X_std = (X - X.mean(axis=0)) / X.std(axis=0)
        g_pred = Ridge(alpha=RIDGE_ALPHA).fit(X_std, g_shifted).predict(X_std)
        b_pred = Ridge(alpha=RIDGE_ALPHA).fit(X_std, b_shifted).predict(X_std)
        g_res = g_shifted - g_pred
        b_res = b_shifted - b_pred

        ss_tot = float(np.sum((b_raw - float(np.mean(b_raw))) ** 2))
        if ss_tot > 1e-12:
            b_raw_pred = Ridge(alpha=RIDGE_ALPHA).fit(X_std, b_raw).predict(X_std)
            ss_res = float(np.sum((b_raw - b_raw_pred) ** 2))
            pose_r2: float | None = max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
        else:
            pose_r2 = 0.0
    else:
        g_res = g_shifted
        b_res = b_shifted
        pose_r2 = 0.0 if pose_mode != "unavailable" else None

    calibrated_rho = _safe_spearman(g_res, b_res)
    rho_delta = (
        (calibrated_rho - raw_rho)
        if (calibrated_rho is not None and raw_rho is not None)
        else None
    )

    inactive_reason: str | None = None
    if n_valid < MIN_CALIBRATION_FRAMES:
        inactive_reason = f"有效连续观测仅 {n_valid} 帧，样本量偏少"
    elif float(np.std(g_raw)) <= STD_EPS or float(np.std(b_raw)) <= STD_EPS:
        inactive_reason = "单侧或双侧动作序列几乎无波动"
    elif geo_floor_rate > FLOOR_RATE_INACTIVE_THRESHOLD:
        inactive_reason = "几何测算序列贴底比例过高"
    elif bs_floor_rate > FLOOR_RATE_INACTIVE_THRESHOLD:
        inactive_reason = "动作模型系数贴底比例过高，通道几乎未激活"
    elif calibrated_rho is None:
        inactive_reason = "校准后序列波动接近零，无法建立稳健相关"

    if inactive_reason is not None:
        eligibility = "待3D重采验证"
    elif calibrated_rho is not None and calibrated_rho >= FUSION_RHO_THRESHOLD and n_valid >= MIN_FUSION_FRAMES:
        eligibility = "已校准可融合"
    elif calibrated_rho is not None and calibrated_rho >= ASSIST_RHO_THRESHOLD:
        eligibility = "辅助对照"
    else:
        eligibility = "待3D重采验证"

    return {
        "key": spec.construct,
        "label": spec.label,
        "n_valid": n_valid,
        "raw_rho": _round_or_none(raw_rho),
        "calibrated_rho": _round_or_none(calibrated_rho),
        "rho_delta": _round_or_none(rho_delta),
        "pose_r2": _round_or_none(pose_r2),
        "eligibility": eligibility,
        "note": _build_construct_note(
            missing_cols=False,
            n_valid=n_valid,
            inactive_reason=inactive_reason,
            eligibility=eligibility,
            calibrated_rho=calibrated_rho,
            pose_r2=pose_r2,
        ),
    }


def _unavailable_result(
    *,
    pose_mode: str = "unavailable",
    n_raw_frames: int = 0,
    n_clean_frames: int = 0,
    quality_pass_rate: float | None = None,
    summary_plain: str | None = None,
) -> dict[str, Any]:
    return {
        "available": False,
        "pose_mode": pose_mode,
        "n_raw_frames": int(n_raw_frames),
        "n_clean_frames": int(n_clean_frames),
        "quality_pass_rate": _round_or_none(quality_pass_rate),
        "eligible_construct_count": 0,
        "summary_plain": summary_plain or "当前场次缺少可用的面部动作模型系数数据，未执行校准对照。",
        "constructs": [],
        "disclaimer": _DISCLAIMER,
        "inference_ms": 0.0,
    }


def calibrate_session_blendshapes(logs: dict | None) -> dict[str, Any]:
    """对单场面部日志执行画面质量过滤、平滑滤波与头部姿态残差校准。"""
    face_df = logs.get("face") if isinstance(logs, dict) else None
    if face_df is None or not isinstance(face_df, pd.DataFrame) or face_df.empty:
        return _unavailable_result()

    bs_cols = [c for c in face_df.columns if isinstance(c, str) and c.startswith("bs_")]
    if not bs_cols:
        return _unavailable_result(n_raw_frames=len(face_df))

    has_valid_bs = any(
        bool(np.isfinite(pd.to_numeric(face_df[c], errors="coerce").to_numpy(dtype=float)).any())
        for c in bs_cols
    )
    if not has_valid_bs:
        return _unavailable_result(
            n_raw_frames=len(face_df),
            summary_plain="当前场次面部动作模型系数列无有效观测帧，未执行校准对照。",
        )

    t0 = time.perf_counter()
    n_raw_frames = int(len(face_df))
    pose_mode = _detect_pose_mode(face_df)
    clean_mask = _build_quality_mask(face_df, bs_cols)
    n_clean_frames = int(clean_mask.sum())
    quality_pass_rate = float(n_clean_frames / n_raw_frames) if n_raw_frames > 0 else None

    if n_clean_frames == 0:
        return _unavailable_result(
            pose_mode=pose_mode,
            n_raw_frames=n_raw_frames,
            n_clean_frames=0,
            quality_pass_rate=quality_pass_rate,
            summary_plain="当前场次全部帧均未通过画面质量门控，未执行校准对照。",
        )

    constructs = [
        _calibrate_single_construct(face_df, clean_mask, pose_mode, spec)
        for spec in CONSTRUCT_PAIRS
    ]

    eligible_count = sum(1 for c in constructs if c["eligibility"] == "已校准可融合")
    assist_count = sum(1 for c in constructs if c["eligibility"] == "辅助对照")
    pending_count = sum(1 for c in constructs if c["eligibility"] == "待3D重采验证")

    pose_label_map = {
        "3d_matrix": "三维姿态矩阵",
        "2d_legacy": "二维姿态代理",
        "unavailable": "无姿态参照",
    }
    pose_label = pose_label_map.get(pose_mode, "无姿态参照")
    summary_plain = (
        f"本场次共纳入 {n_clean_frames} 帧有效画面，在 {pose_label} 姿态基线校准后，"
        f"{len(constructs)} 项面部动作中有 {eligible_count} 项达到融合对照标准，"
        f"{assist_count} 项适合作为辅助对照，{pending_count} 项需在三维姿态重采后进一步核验。"
    )

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    return {
        "available": True,
        "pose_mode": pose_mode,
        "n_raw_frames": n_raw_frames,
        "n_clean_frames": n_clean_frames,
        "quality_pass_rate": _round_or_none(quality_pass_rate),
        "eligible_construct_count": eligible_count,
        "summary_plain": summary_plain,
        "constructs": constructs,
        "disclaimer": _DISCLAIMER,
        "inference_ms": elapsed_ms,
    }
