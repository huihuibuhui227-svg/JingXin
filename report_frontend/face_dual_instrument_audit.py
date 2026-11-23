# report_frontend/face_dual_instrument_audit.py
"""面部双仪器一致性审计与头部姿态残差诊断模块。

本模块对面部关键点几何代理列与 52 维模型系数列进行逐帧秩相关与帧间差分相关性对照，
同时提供嘴角下压几何列相对俯仰角代理的线性回归残差诊断，以及外部 FACS AU CSV 对接入口。
所有结果严格用于双仪器分轨透明披露与离线标定排查，不修改任何主报告指标读数。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MIN_VALID_FRAMES: int = 10
FLOOR_EPS: float = 1e-6
FLOOR_RATE_INACTIVE_THRESHOLD: float = 0.95
STD_EPS: float = 1e-9
RHO_HIGH: float = 0.7
RHO_MODERATE: float = 0.4


@dataclass(frozen=True)
class ConstructPairSpec:
    """面部几何代理与模型系数的配对构念定义。"""

    construct: str
    label: str
    geometric_col: str
    blendshape_cols: tuple[str, ...]
    geo_polarity: int = 1


CONSTRUCT_PAIRS: tuple[ConstructPairSpec, ...] = (
    ConstructPairSpec(
        construct="eye_squeeze",
        label="眼周收紧",
        geometric_col="au7_eye_squeeze",
        blendshape_cols=("bs_eyeSquintLeft", "bs_eyeSquintRight"),
        geo_polarity=1,
    ),
    ConstructPairSpec(
        construct="smile",
        label="嘴角上扬",
        geometric_col="au12_smile",
        blendshape_cols=("bs_mouthSmileLeft", "bs_mouthSmileRight"),
        geo_polarity=1,
    ),
    ConstructPairSpec(
        construct="mouth_open",
        label="口部张开",
        geometric_col="au25_mouth_open",
        blendshape_cols=("bs_jawOpen",),
        geo_polarity=1,
    ),
    ConstructPairSpec(
        construct="lip_compression",
        label="唇部压紧",
        geometric_col="au23_lip_compression",
        blendshape_cols=("bs_mouthPressLeft", "bs_mouthPressRight"),
        geo_polarity=-1,
    ),
    ConstructPairSpec(
        construct="brow_furrow",
        label="眉间收缩",
        geometric_col="au4_frown",
        blendshape_cols=("bs_browDownLeft", "bs_browDownRight"),
        geo_polarity=-1,
    ),
)


@dataclass(frozen=True)
class ExternalAuSpec:
    """外部 FACS 动作单元与本系统双仪器列的映射定义。"""

    canonical: str
    aliases: tuple[str, ...]
    label: str
    geometric_col: str
    blendshape_cols: tuple[str, ...]
    geo_polarity: int = 1


_EXTERNAL_AU_SPECS: tuple[ExternalAuSpec, ...] = (
    ExternalAuSpec(
        canonical="AU04_r",
        aliases=("au04_r", "au04", "au4_r", "au4"),
        label="眉间收缩",
        geometric_col="au4_frown",
        blendshape_cols=("bs_browDownLeft", "bs_browDownRight"),
        geo_polarity=-1,
    ),
    ExternalAuSpec(
        canonical="AU06_r",
        aliases=("au06_r", "au06", "au6_r", "au6"),
        label="面颊抬起",
        geometric_col="au6_cheek_raise",
        blendshape_cols=("bs_cheekSquintLeft", "bs_cheekSquintRight"),
        geo_polarity=1,
    ),
    ExternalAuSpec(
        canonical="AU07_r",
        aliases=("au07_r", "au07", "au7_r", "au7"),
        label="眼周收紧",
        geometric_col="au7_eye_squeeze",
        blendshape_cols=("bs_eyeSquintLeft", "bs_eyeSquintRight"),
        geo_polarity=1,
    ),
    ExternalAuSpec(
        canonical="AU10_r",
        aliases=("au10_r", "au10"),
        label="上唇抬起",
        geometric_col="au10_upper_lip_raise",
        blendshape_cols=("bs_mouthUpperUpLeft", "bs_mouthUpperUpRight"),
        geo_polarity=1,
    ),
    ExternalAuSpec(
        canonical="AU12_r",
        aliases=("au12_r", "au12"),
        label="嘴角上扬",
        geometric_col="au12_smile",
        blendshape_cols=("bs_mouthSmileLeft", "bs_mouthSmileRight"),
        geo_polarity=1,
    ),
    ExternalAuSpec(
        canonical="AU15_r",
        aliases=("au15_r", "au15"),
        label="嘴角下压",
        geometric_col="au15_mouth_down",
        blendshape_cols=("bs_mouthFrownLeft", "bs_mouthFrownRight"),
        geo_polarity=1,
    ),
    ExternalAuSpec(
        canonical="AU23_r",
        aliases=("au23_r", "au23"),
        label="唇部压紧",
        geometric_col="au23_lip_compression",
        blendshape_cols=("bs_mouthPressLeft", "bs_mouthPressRight"),
        geo_polarity=-1,
    ),
    ExternalAuSpec(
        canonical="AU25_r",
        aliases=("au25_r", "au25"),
        label="口部张开",
        geometric_col="au25_mouth_open",
        blendshape_cols=("bs_jawOpen",),
        geo_polarity=1,
    ),
    ExternalAuSpec(
        canonical="AU26_r",
        aliases=("au26_r", "au26"),
        label="下颌下降",
        geometric_col="au26_jaw_drop",
        blendshape_cols=("bs_jawOpen",),
        geo_polarity=1,
    ),
)


def _finite_or_none(val: float | None) -> float | None:
    if val is None:
        return None
    f = float(val)
    return f if math.isfinite(f) else None


def _round_or_none(val: float | None, digits: int = 6) -> float | None:
    f = _finite_or_none(val)
    return None if f is None else round(f, digits)


def _num_array(df: pd.DataFrame, col: str) -> np.ndarray | None:
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _paired_bs_array(df: pd.DataFrame, cols: tuple[str, ...]) -> np.ndarray | None:
    if not cols or any(c not in df.columns for c in cols):
        return None
    arrays = [pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in cols]
    stacked = np.column_stack(arrays)
    ok = np.isfinite(stacked).all(axis=1)
    out = np.full(len(df), np.nan, dtype=float)
    if ok.any():
        out[ok] = stacked[ok].mean(axis=1)
    return out


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 3 or y.size < 3 or x.size != y.size:
        return None
    if float(np.std(x)) <= STD_EPS or float(np.std(y)) <= STD_EPS:
        return None
    stat = spearmanr(x, y).statistic
    return _finite_or_none(stat)


def _delta_spearman(x_full: np.ndarray, y_full: np.ndarray, ok: np.ndarray) -> float | None:
    if int(ok.sum()) < 4:
        return None
    ok_adj = ok[1:] & ok[:-1]
    if int(ok_adj.sum()) >= 3:
        dx = np.diff(x_full)[ok_adj]
        dy = np.diff(y_full)[ok_adj]
    else:
        dx = np.diff(x_full[ok])
        dy = np.diff(y_full[ok])
    return _safe_spearman(dx, dy)


def _classify_agreement(
    n_valid: int,
    x_std: float,
    y_std: float,
    x_floor_rate: float,
    y_floor_rate: float,
    rho: float | None,
) -> tuple[str, str | None]:
    if n_valid < MIN_VALID_FRAMES:
        return "通道未激活", f"有效配对帧数不足 {MIN_VALID_FRAMES} 帧"
    if x_std <= STD_EPS or y_std <= STD_EPS:
        return "通道未激活", "单侧或双侧序列方差接近零"
    if x_floor_rate > FLOOR_RATE_INACTIVE_THRESHOLD:
        return "通道未激活", "几何通道贴底占比超过百分之九十五"
    if y_floor_rate > FLOOR_RATE_INACTIVE_THRESHOLD:
        return "通道未激活", "模型通道贴底占比超过百分之九十五"
    if rho is None:
        return "通道未激活", "相关系数无法计算"
    if rho >= RHO_HIGH:
        return "高度一致", None
    if rho >= RHO_MODERATE:
        return "中度一致", None
    return "分歧较大", None


def _audit_single_pair(face_df: pd.DataFrame, spec: ConstructPairSpec) -> dict[str, Any]:
    geo = _num_array(face_df, spec.geometric_col)
    bs = _paired_bs_array(face_df, spec.blendshape_cols)

    if geo is None or bs is None:
        missing = []
        if geo is None:
            missing.append(spec.geometric_col)
        if bs is None:
            missing.extend(c for c in spec.blendshape_cols if c not in face_df.columns)
        return {
            "construct": spec.construct,
            "label": spec.label,
            "geometric_col": spec.geometric_col,
            "blendshape_cols": list(spec.blendshape_cols),
            "geo_polarity": spec.geo_polarity,
            "n_valid": 0,
            "spearman_rho": None,
            "raw_spearman_rho": None,
            "delta_spearman_rho": None,
            "raw_delta_spearman_rho": None,
            "geo_floor_rate": None,
            "bs_floor_rate": None,
            "agreement_level": "通道未激活",
            "inactive_reason": f"缺少列 {', '.join(missing)}",
        }

    ok = np.isfinite(geo) & np.isfinite(bs)
    n_valid = int(ok.sum())
    if n_valid == 0:
        return {
            "construct": spec.construct,
            "label": spec.label,
            "geometric_col": spec.geometric_col,
            "blendshape_cols": list(spec.blendshape_cols),
            "geo_polarity": spec.geo_polarity,
            "n_valid": 0,
            "spearman_rho": None,
            "raw_spearman_rho": None,
            "delta_spearman_rho": None,
            "raw_delta_spearman_rho": None,
            "geo_floor_rate": None,
            "bs_floor_rate": None,
            "agreement_level": "通道未激活",
            "inactive_reason": "有效配对帧数为零",
        }

    ga = geo[ok]
    ba = bs[ok]
    geo_floor_rate = float(np.mean(ga <= FLOOR_EPS))
    bs_floor_rate = float(np.mean(ba <= FLOOR_EPS))
    geo_std = float(np.std(ga))
    bs_std = float(np.std(ba))

    raw_rho = _safe_spearman(ga, ba) if n_valid >= MIN_VALID_FRAMES else None
    raw_drho = _delta_spearman(geo, bs, ok) if n_valid >= MIN_VALID_FRAMES else None
    aligned_rho = None if raw_rho is None else spec.geo_polarity * raw_rho
    aligned_drho = None if raw_drho is None else spec.geo_polarity * raw_drho

    level, inactive_reason = _classify_agreement(
        n_valid=n_valid,
        x_std=geo_std,
        y_std=bs_std,
        x_floor_rate=geo_floor_rate,
        y_floor_rate=bs_floor_rate,
        rho=aligned_rho,
    )

    return {
        "construct": spec.construct,
        "label": spec.label,
        "geometric_col": spec.geometric_col,
        "blendshape_cols": list(spec.blendshape_cols),
        "geo_polarity": spec.geo_polarity,
        "n_valid": n_valid,
        "spearman_rho": _round_or_none(aligned_rho),
        "raw_spearman_rho": _round_or_none(raw_rho),
        "delta_spearman_rho": _round_or_none(aligned_drho),
        "raw_delta_spearman_rho": _round_or_none(raw_drho),
        "geo_floor_rate": round(geo_floor_rate, 6),
        "bs_floor_rate": round(bs_floor_rate, 6),
        "agreement_level": level,
        "inactive_reason": inactive_reason,
    }


def _compute_au15_pitch_diagnostic(face_df: pd.DataFrame) -> dict[str, Any] | None:
    """计算 AU15 嘴角下压几何列对俯仰角代理的回归残差诊断信息，仅作诊断用途。"""
    if "au15_mouth_down" not in face_df.columns or "head_pitch" not in face_df.columns:
        return None

    au15 = _num_array(face_df, "au15_mouth_down")
    pitch = _num_array(face_df, "head_pitch")
    assert au15 is not None and pitch is not None

    ok_ap = np.isfinite(au15) & np.isfinite(pitch)
    n_valid = int(ok_ap.sum())
    if n_valid == 0:
        return {
            "available": False,
            "diagnostic_only": True,
            "status": "diagnostic_only",
            "n_valid": 0,
            "raw_floor_rate": None,
            "au15_pitch_spearman_rho": None,
            "pitch_regression_slope": None,
            "pitch_regression_intercept": None,
            "pitch_r_squared": None,
            "n_valid_with_bs": 0,
            "bs_frown_floor_rate": None,
            "raw_bs_frown_spearman_rho": None,
            "residual_bs_frown_spearman_rho": None,
            "diagnostic_note": "诊断参考项：当前场次缺少有效嘴角下压或俯仰角观测帧，未计算残差。",
        }

    yv = au15[ok_ap]
    xv = pitch[ok_ap]
    raw_floor_rate = float(np.mean(yv <= FLOOR_EPS))

    if n_valid < MIN_VALID_FRAMES:
        return {
            "available": False,
            "diagnostic_only": True,
            "status": "diagnostic_only",
            "n_valid": n_valid,
            "raw_floor_rate": round(raw_floor_rate, 6),
            "au15_pitch_spearman_rho": _round_or_none(_safe_spearman(yv, xv)),
            "pitch_regression_slope": None,
            "pitch_regression_intercept": None,
            "pitch_r_squared": None,
            "n_valid_with_bs": 0,
            "bs_frown_floor_rate": None,
            "raw_bs_frown_spearman_rho": None,
            "residual_bs_frown_spearman_rho": None,
            "diagnostic_note": f"诊断参考项：有效帧数仅为 {n_valid} 帧，不足 {MIN_VALID_FRAMES} 帧，未进行俯仰角线性残差诊断。",
        }

    if float(np.std(xv)) > STD_EPS:
        slope, intercept = np.polyfit(xv, yv, 1)
        slope_f = float(slope)
        intercept_f = float(intercept)
    else:
        slope_f = 0.0
        intercept_f = float(np.median(yv))

    yhat = slope_f * xv + intercept_f
    res_ap = yv - yhat
    ss_res = float(np.sum(res_ap ** 2))
    ss_tot = float(np.sum((yv - float(np.mean(yv))) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > STD_EPS else None
    au15_pitch_rho = _safe_spearman(yv, xv)

    bs_frown = _paired_bs_array(face_df, ("bs_mouthFrownLeft", "bs_mouthFrownRight"))
    n_valid_with_bs = 0
    bs_floor_rate: float | None = None
    raw_bs_rho: float | None = None
    res_bs_rho: float | None = None

    if bs_frown is not None:
        ok_all = ok_ap & np.isfinite(bs_frown)
        n_valid_with_bs = int(ok_all.sum())
        if n_valid_with_bs > 0:
            bv = bs_frown[ok_all]
            bs_floor_rate = float(np.mean(bv <= FLOOR_EPS))
            if n_valid_with_bs >= MIN_VALID_FRAMES:
                y_all = au15[ok_all]
                x_all = pitch[ok_all]
                res_all = y_all - (slope_f * x_all + intercept_f)
                raw_bs_rho = _safe_spearman(y_all, bv)
                res_bs_rho = _safe_spearman(res_all, bv)

    pitch_rho_text = f"{au15_pitch_rho:.3f}" if au15_pitch_rho is not None else "无法计算"
    res_rho_text = f"{res_bs_rho:.3f}" if res_bs_rho is not None else "无法计算"
    diagnostic_note = (
        f"诊断参考项，不参与主报告计算与解封判定：原始嘴角下压几何列零值贴底占比 {raw_floor_rate:.1%}，"
        f"与俯仰角代理相关系数 {pitch_rho_text}；扣除俯仰角线性分量后，"
        f"残差与模型皱嘴角系数相关系数 {res_rho_text}，仍不足以替代三维姿态标定。"
    )

    return {
        "available": True,
        "diagnostic_only": True,
        "status": "diagnostic_only",
        "n_valid": n_valid,
        "raw_floor_rate": round(raw_floor_rate, 6),
        "au15_pitch_spearman_rho": _round_or_none(au15_pitch_rho),
        "pitch_regression_slope": _round_or_none(slope_f),
        "pitch_regression_intercept": _round_or_none(intercept_f),
        "pitch_r_squared": _round_or_none(r2),
        "n_valid_with_bs": n_valid_with_bs,
        "bs_frown_floor_rate": None if bs_floor_rate is None else round(bs_floor_rate, 6),
        "raw_bs_frown_spearman_rho": _round_or_none(raw_bs_rho),
        "residual_bs_frown_spearman_rho": _round_or_none(res_bs_rho),
        "diagnostic_note": diagnostic_note,
    }


def _unavailable_audit(summary_plain: str | None = None) -> dict[str, Any]:
    return {
        "available": False,
        "summary_plain": summary_plain or "当前场次未包含面部模型系数列，未开展双仪器一致性对照。",
        "pairs": [],
        "au15_pitch_diagnostic": None,
    }


def audit_face_dual_instrument(logs: dict | None) -> dict[str, Any]:
    """对单场面部日志执行关键点几何代理与 52 维模型系数双仪器一致性审计。"""
    face_df = logs.get("face") if isinstance(logs, dict) else None
    if face_df is None or not isinstance(face_df, pd.DataFrame) or face_df.empty:
        return _unavailable_audit()

    bs_cols = [c for c in face_df.columns if isinstance(c, str) and c.startswith("bs_")]
    if not bs_cols:
        return _unavailable_audit()

    has_valid_bs = any(
        np.isfinite(pd.to_numeric(face_df[c], errors="coerce").to_numpy(dtype=float)).any()
        for c in bs_cols
    )
    if not has_valid_bs:
        return _unavailable_audit("当前场次面部模型系数列无有效观测帧，未开展双仪器一致性对照。")

    pairs = [_audit_single_pair(face_df, spec) for spec in CONSTRUCT_PAIRS]
    counts = {"高度一致": 0, "中度一致": 0, "分歧较大": 0, "通道未激活": 0}
    for p in pairs:
        lvl = p["agreement_level"]
        counts[lvl] = counts.get(lvl, 0) + 1

    n_high = counts["高度一致"]
    n_mod = counts["中度一致"]
    n_div_or_inactive = counts["分歧较大"] + counts["通道未激活"]
    summary_plain = (
        f"共对照 {len(pairs)} 组面部构念：{n_high} 组高度一致、{n_mod} 组中度一致、"
        f"{n_div_or_inactive} 组存在测量差异或通道未激活，主报告已按仪器分轨展示。"
    )

    return {
        "available": True,
        "summary_plain": summary_plain,
        "counts_by_level": counts,
        "pairs": pairs,
        "au15_pitch_diagnostic": _compute_au15_pitch_diagnostic(face_df),
    }


def _find_timestamp_col(df: pd.DataFrame) -> str | None:
    for cand in ("timestamp", "timestamp_sec", "time_sec", "frame_time"):
        if cand in df.columns:
            return cand
    return None


def _align_frames(face_df: pd.DataFrame, ext_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """将内部面部日志与外部 FACS CSV 帧序列对齐。"""
    ext_clean = ext_df.copy()
    ext_clean.columns = [str(c).strip() for c in ext_clean.columns]

    if len(face_df) == len(ext_clean):
        return face_df.reset_index(drop=True), ext_clean.reset_index(drop=True), "row_index"

    t_face_col = _find_timestamp_col(face_df)
    t_ext_col = _find_timestamp_col(ext_clean)
    if t_face_col and t_ext_col:
        tf = pd.to_numeric(face_df[t_face_col], errors="coerce")
        te = pd.to_numeric(ext_clean[t_ext_col], errors="coerce")
        if tf.notna().sum() >= 2 and te.notna().sum() >= 2:
            f_sub = face_df.loc[tf.notna()].copy()
            e_sub = ext_clean.loc[te.notna()].copy()
            f_t = tf.loc[tf.notna()].to_numpy(dtype=float)
            e_t = te.loc[te.notna()].to_numpy(dtype=float)
            f_min, f_max = float(np.min(f_t)), float(np.max(f_t))
            e_min, e_max = float(np.min(e_t)), float(np.max(e_t))
            if f_max < e_min - 0.5 or e_max < f_min - 0.5:
                f_t = f_t - f_min
                e_t = e_t - e_min
            f_sub["_align_t"] = f_t
            e_sub["_align_t"] = e_t
            f_sub = f_sub.sort_values("_align_t").reset_index(drop=True)
            e_sub = e_sub.sort_values("_align_t").reset_index(drop=True)
            f_sub["_orig_idx"] = np.arange(len(f_sub))
            e_renamed = e_sub.add_prefix("__ext__")
            merged = pd.merge_asof(
                f_sub[["_align_t", "_orig_idx"]],
                e_renamed,
                left_on="_align_t",
                right_on="__ext___align_t",
                direction="nearest",
                tolerance=0.5,
            )
            valid_mask = merged["__ext___align_t"].notna()
            if int(valid_mask.sum()) >= 3:
                f_aligned = f_sub.loc[valid_mask].reset_index(drop=True)
                ext_cols_map = {f"__ext__{c}": c for c in ext_clean.columns}
                e_aligned = merged.loc[valid_mask, list(ext_cols_map.keys())].rename(columns=ext_cols_map).reset_index(drop=True)
                return f_aligned, e_aligned, "timestamp_asof"

    n_min = min(len(face_df), len(ext_clean))
    return (
        face_df.iloc[:n_min].reset_index(drop=True),
        ext_clean.iloc[:n_min].reset_index(drop=True),
        "truncated_index",
    )


def _compare_pair_arrays(
    x: np.ndarray | None,
    y: np.ndarray,
    polarity: int = 1,
) -> dict[str, Any]:
    if x is None:
        return {
            "n_valid": 0,
            "spearman_rho": None,
            "delta_spearman_rho": None,
            "agreement_level": "通道未激活",
        }
    ok = np.isfinite(x) & np.isfinite(y)
    n_valid = int(ok.sum())
    if n_valid == 0:
        return {
            "n_valid": 0,
            "spearman_rho": None,
            "delta_spearman_rho": None,
            "agreement_level": "通道未激活",
        }
    xa = x[ok]
    ya = y[ok]
    raw_rho = _safe_spearman(xa, ya)
    raw_drho = _delta_spearman(x, y, ok)
    aligned_rho = None if raw_rho is None else polarity * raw_rho
    aligned_drho = None if raw_drho is None else polarity * raw_drho
    level, _ = _classify_agreement(
        n_valid=n_valid,
        x_std=float(np.std(xa)),
        y_std=float(np.std(ya)),
        x_floor_rate=float(np.mean(xa <= FLOOR_EPS)),
        y_floor_rate=float(np.mean(ya <= FLOOR_EPS)),
        rho=aligned_rho,
    )
    return {
        "n_valid": n_valid,
        "spearman_rho": _round_or_none(aligned_rho),
        "delta_spearman_rho": _round_or_none(aligned_drho),
        "agreement_level": level,
    }


def compare_external_facs_csv(
    face_df: pd.DataFrame | None,
    external_au_csv: str | Path | pd.DataFrame | None,
) -> dict[str, Any]:
    """将内部面部双通道序列与外部 FACS 动作单元序列进行对照验证。

    支持 OpenFace 的 `AU04_r` / `AU12_r` / `AU25_r` 列以及 Py-Feat 的 `AU04` / `AU12` / `AU25` 列。
    """
    if external_au_csv is None:
        return {
            "available": False,
            "alignment_mode": None,
            "matched_columns": [],
            "comparisons": [],
            "summary_plain": "未提供外部动作单元对照文件。",
        }

    if isinstance(external_au_csv, pd.DataFrame):
        ext_df = external_au_csv.copy()
    else:
        csv_path = Path(external_au_csv)
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        ext_df = pd.read_csv(csv_path)

    if face_df is None or not isinstance(face_df, pd.DataFrame) or face_df.empty or ext_df.empty:
        return {
            "available": False,
            "alignment_mode": None,
            "matched_columns": [],
            "comparisons": [],
            "summary_plain": "内部面部日志或外部对照文件为空，未开展外部校验。",
        }

    f_aligned, e_aligned, alignment_mode = _align_frames(face_df, ext_df)
    lower_lookup = {str(c).strip().lower(): str(c).strip() for c in e_aligned.columns}

    comparisons: list[dict[str, Any]] = []
    matched_columns: list[str] = []

    for spec in _EXTERNAL_AU_SPECS:
        found_col = None
        for alias in spec.aliases:
            if alias in lower_lookup:
                found_col = lower_lookup[alias]
                break
        if found_col is None:
            continue

        ext_arr = _num_array(e_aligned, found_col)
        if ext_arr is None:
            continue
        matched_columns.append(found_col)

        geo_arr = _num_array(f_aligned, spec.geometric_col)
        bs_arr = _paired_bs_array(f_aligned, spec.blendshape_cols)

        geo_cmp = _compare_pair_arrays(geo_arr, ext_arr, polarity=spec.geo_polarity)
        bs_cmp = _compare_pair_arrays(bs_arr, ext_arr, polarity=1)

        comparisons.append({
            "canonical_au": spec.canonical,
            "external_col": found_col,
            "label": spec.label,
            "geometric_col": spec.geometric_col,
            "geo_polarity": spec.geo_polarity,
            "blendshape_cols": list(spec.blendshape_cols),
            "geometric_comparison": geo_cmp,
            "blendshape_comparison": bs_cmp,
        })

    if not comparisons:
        return {
            "available": False,
            "alignment_mode": alignment_mode,
            "matched_columns": [],
            "comparisons": [],
            "summary_plain": "外部文件未匹配到可对照的标准动作单元列。",
        }

    bs_high_or_mod = sum(
        1 for c in comparisons
        if c["blendshape_comparison"]["agreement_level"] in ("高度一致", "中度一致")
    )
    geo_high_or_mod = sum(
        1 for c in comparisons
        if c["geometric_comparison"]["agreement_level"] in ("高度一致", "中度一致")
    )
    summary_plain = (
        f"外部动作单元共匹配 {len(comparisons)} 项："
        f"模型系数侧有 {bs_high_or_mod} 项达到中度以上一致，"
        f"几何代理侧有 {geo_high_or_mod} 项达到中度以上一致。"
    )

    return {
        "available": True,
        "alignment_mode": alignment_mode,
        "n_rows_face": int(len(face_df)),
        "n_rows_external": int(len(ext_df)),
        "n_rows_aligned": int(len(f_aligned)),
        "matched_columns": matched_columns,
        "comparisons": comparisons,
        "summary_plain": summary_plain,
    }
