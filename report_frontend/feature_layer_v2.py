# report_frontend/feature_layer_v2.py
"""第二层多模态动力学特征引擎（纯 CPU 毫秒级，严格只读）。

设计约束：
- 纯 CPU 运算（NumPy / SciPy / pandas / pycatch22），绝不引入 torch；
- 绝不在报告生成路径上解码音频或视频原文件，仅读取内存中的日志表与可选的 `.npz` 帧序列缓存；
- 当日志为空或不可用时严格确定性返回（`available=False`，`inference_ms=0.0`，无 NaN/Inf）。
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pycatch22
from scipy.fft import rfft, rfftfreq

from report_frontend import voice_series

_MIN_DT_SEC = 0.05
_FALLBACK_DT_SEC = 1.0
_MIN_BLUR_LAPLACIAN = 15.0
_DEFAULT_FRAME_SEC = 0.032
_MAX_CATCH22_SAMPLES = 1024

_CATCH22_STREAM_SPECS: tuple[tuple[str, str, str], ...] = (
    ("face", "au12_smile", "au12_smile"),
    ("face", "au7_eye_squeeze", "au7_eye_squeeze"),
    ("face", "gaze_direction_x", "gaze_direction_x"),
    ("gesture", "torso_jitter", "torso_jitter"),
)

_CATCH22_METRIC_ALIASES: tuple[tuple[str, ...], ...] = (
    ("CO_FirstCrossing_acf_1e_pointOfCrossing", "CO_f1ecac"),
    ("SP_Summaries_fft_area_5_1", "SP_Summaries_welch_rect_area_5_1"),
    ("SB_BinaryStats_mean_longstretch1",),
    ("MD_pNN_pnn40", "MD_hrv_classic_pnn40"),
)


def _finite_float(value: Any, digits: int = 4) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, digits)


def _num_array(df: pd.DataFrame | None, col: str) -> np.ndarray:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or col not in df.columns:
        return np.array([], dtype=float)
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        return s.to_numpy(dtype=float, na_value=np.nan)
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def _extract_timestamps_sec(df: pd.DataFrame | None) -> np.ndarray:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return np.array([], dtype=float)
    col = "timestamp" if "timestamp" in df.columns else ("unix_timestamp" if "unix_timestamp" in df.columns else None)
    if col is None:
        return np.array([], dtype=float)

    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        return s.to_numpy(dtype=float, na_value=np.nan)

    if pd.api.types.is_datetime64_any_dtype(s):
        valid = s.notna().to_numpy()
        if int(valid.sum()) < 2:
            return np.array([], dtype=float)
        ns = s.astype("datetime64[ns]").astype("int64").to_numpy()
        v_ns = ns[valid]
        diffs = np.diff(v_ns)
        pos = diffs[diffs > 0]
        # LogDataLoader 将数值秒直接传给 pd.to_datetime，导致秒数被当成纳秒写入 1970 年区间
        if 0 <= int(v_ns.max()) < 10_000_000_000 and (pos.size > 0 and float(np.median(pos)) < 1000.0):
            return np.where(valid, ns.astype(float), np.nan)
        return np.where(valid, ns.astype(float) * 1e-9, np.nan)

    num = pd.to_numeric(s, errors="coerce")
    if int(num.notna().sum()) >= 2:
        return num.to_numpy(dtype=float)

    dt_parsed = pd.to_datetime(s, errors="coerce")
    valid = dt_parsed.notna().to_numpy()
    if int(valid.sum()) >= 2:
        ns = dt_parsed.astype("datetime64[ns]").astype("int64").to_numpy()
        return np.where(valid, ns.astype(float) * 1e-9, np.nan)
    return np.array([], dtype=float)


def _compute_dt(t_sec: np.ndarray, valid_indices: np.ndarray) -> np.ndarray:
    n_intervals = max(0, len(valid_indices) - 1)
    if n_intervals == 0:
        return np.array([], dtype=float)
    if t_sec.size <= int(valid_indices[-1]):
        return np.full(n_intervals, _FALLBACK_DT_SEC, dtype=float)

    t_sel = t_sec[valid_indices]
    if not np.all(np.isfinite(t_sel)):
        return np.full(n_intervals, _FALLBACK_DT_SEC, dtype=float)

    raw_dt = np.diff(t_sel)
    if np.any(raw_dt < -1e-9) or not np.any(raw_dt > 0.0):
        return np.full(n_intervals, _FALLBACK_DT_SEC, dtype=float)

    pos_dt = raw_dt[raw_dt > 0.0]
    med_pos = max(float(np.median(pos_dt)), _MIN_DT_SEC)
    return np.where(raw_dt > 0.0, np.maximum(raw_dt, _MIN_DT_SEC), med_pos)


def _parse_quality_pass_rate(face: pd.DataFrame) -> tuple[float | None, str]:
    n_total = len(face)
    if n_total == 0:
        return None, "unavailable"

    if "frame_quality_ok" in face.columns:
        raw = face["frame_quality_ok"]
        if pd.api.types.is_bool_dtype(raw):
            valid_count = int(raw.notna().sum())
            if valid_count > 0:
                return float(raw.fillna(False).astype(bool).sum()) / n_total, "frame_quality_ok"
        else:
            num = pd.to_numeric(raw, errors="coerce")
            if int(num.notna().sum()) > 0:
                return float((num.fillna(0.0) > 0.0).sum()) / n_total, "frame_quality_ok"
            mapped = raw.astype(str).str.strip().str.lower().map(
                {"true": 1.0, "1": 1.0, "yes": 1.0, "false": 0.0, "0": 0.0, "no": 0.0}
            )
            if int(mapped.notna().sum()) > 0:
                return float((mapped.fillna(0.0) > 0.0).sum()) / n_total, "frame_quality_ok"

    if "frame_blur_laplacian" in face.columns:
        lap = _num_array(face, "frame_blur_laplacian")
        if int(np.isfinite(lap).sum()) > 0:
            passed = np.isfinite(lap) & (lap >= _MIN_BLUR_LAPLACIAN)
            return float(passed.sum()) / n_total, "frame_blur_laplacian"

    landmark_cols = [
        c for c in ("head_yaw_3d", "head_yaw", "avg_ear", "au12_smile", "gaze_direction_x", "face_scale")
        if c in face.columns
    ]
    if landmark_cols:
        mat = np.column_stack([_num_array(face, c) for c in landmark_cols])
        finite_rows = np.isfinite(mat).any(axis=1)
        return float(finite_rows.mean()), "finite_landmark"

    return None, "unavailable"


def _extract_face_kinematics(face: pd.DataFrame | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "pose_mode": "unavailable",
        "unit": "",
        "head_angular_velocity_3d_p50": None,
        "head_angular_velocity_3d_p90": None,
        "frame_quality_pass_rate": None,
        "quality_source": "unavailable",
        "n_valid_frames": 0,
        "n_total_frames": int(len(face)) if isinstance(face, pd.DataFrame) else 0,
    }
    if face is None or not isinstance(face, pd.DataFrame) or face.empty:
        return out

    pass_rate, q_src = _parse_quality_pass_rate(face)
    out["frame_quality_pass_rate"] = _finite_float(pass_rate)
    out["quality_source"] = q_src

    t_sec = _extract_timestamps_sec(face)
    has_3d = all(c in face.columns for c in ("head_yaw_3d", "head_pitch_3d", "head_roll_3d"))
    if has_3d:
        y3 = _num_array(face, "head_yaw_3d")
        p3 = _num_array(face, "head_pitch_3d")
        r3 = _num_array(face, "head_roll_3d")
        valid_3d = np.flatnonzero(np.isfinite(y3) & np.isfinite(p3) & np.isfinite(r3))
        if valid_3d.size >= 3:
            dt = _compute_dt(t_sec, valid_3d)
            disp = np.sqrt(
                np.diff(y3[valid_3d]) ** 2
                + np.diff(p3[valid_3d]) ** 2
                + np.diff(r3[valid_3d]) ** 2
            )
            vel = disp / dt
            vel = vel[np.isfinite(vel)]
            if vel.size >= 2:
                out["pose_mode"] = "3d_matrix"
                out["unit"] = "度/秒"
                out["head_angular_velocity_3d_p50"] = _finite_float(np.percentile(vel, 50))
                out["head_angular_velocity_3d_p90"] = _finite_float(np.percentile(vel, 90))
                out["n_valid_frames"] = int(valid_3d.size)
                return out

    has_2d = all(c in face.columns for c in ("head_yaw", "head_pitch"))
    if has_2d:
        y2 = _num_array(face, "head_yaw")
        p2 = _num_array(face, "head_pitch")
        valid_2d = np.flatnonzero(np.isfinite(y2) & np.isfinite(p2))
        if valid_2d.size >= 3:
            dt = _compute_dt(t_sec, valid_2d)
            disp = np.hypot(np.diff(y2[valid_2d]), np.diff(p2[valid_2d]))
            vel = disp / dt
            vel = vel[np.isfinite(vel)]
            if vel.size >= 2:
                out["pose_mode"] = "2d_legacy"
                out["unit"] = "归一化位移/秒"
                out["head_angular_velocity_3d_p50"] = _finite_float(np.percentile(vel, 50))
                out["head_angular_velocity_3d_p90"] = _finite_float(np.percentile(vel, 90))
                out["n_valid_frames"] = int(valid_2d.size)
                return out

    return out


def _load_npz_segments(
    session_id: str | None,
    l0_series_dir: Path | None,
) -> list[dict[str, Any]]:
    if not session_id or session_id == "NONE":
        return []
    if "/" in session_id or "\\" in session_id or ".." in session_id:
        return []
    base_dir = Path(l0_series_dir) if l0_series_dir is not None else voice_series.series_dir()
    npz_path = base_dir / f"{session_id}.npz"
    if not npz_path.is_file():
        return []

    segments: list[dict[str, Any]] = []
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            files = set(z.files)
            if "__meta__" in files:
                for raw_meta in z["__meta__"]:
                    parts = str(raw_meta).split("|", 2)
                    if len(parts) < 2:
                        continue
                    key, fs_str = parts[0], parts[1]
                    f0_key, rms_key = f"{key}__f0", f"{key}__rms"
                    if f0_key not in files:
                        continue
                    f0 = np.asarray(z[f0_key], dtype=float)
                    rms = np.asarray(z[rms_key], dtype=float) if rms_key in files else np.array([], dtype=float)
                    try:
                        frame_sec = float(fs_str)
                    except ValueError:
                        frame_sec = _DEFAULT_FRAME_SEC
                    if not math.isfinite(frame_sec) or frame_sec <= 0.0:
                        frame_sec = _DEFAULT_FRAME_SEC
                    vf = np.isfinite(f0) & (f0 > 0.0)
                    t = np.arange(f0.size, dtype=float) * frame_sec
                    if f0.size > 0:
                        segments.append({"t": t, "f0": f0, "rms": rms, "voiced_flag": vf, "frame_sec": frame_sec})
            elif "f0" in files:
                f0 = np.asarray(z["f0"], dtype=float)
                rms = np.asarray(z["rms"], dtype=float) if "rms" in files else np.array([], dtype=float)
                if "voiced_flag" in files:
                    vf = np.asarray(z["voiced_flag"]).astype(bool) & np.isfinite(f0) & (f0 > 0.0)
                else:
                    vf = np.isfinite(f0) & (f0 > 0.0)
                if "t" in files:
                    t = np.asarray(z["t"], dtype=float)
                    tf = t[np.isfinite(t)]
                    diffs = np.diff(tf) if tf.size >= 2 else np.array([], dtype=float)
                    pos = diffs[diffs > 0.0]
                    frame_sec = float(np.median(pos)) if pos.size > 0 else _DEFAULT_FRAME_SEC
                else:
                    if "frame_sec" in files:
                        frame_sec = float(np.asarray(z["frame_sec"]).item())
                    elif "sr" in files:
                        sr_val = float(np.asarray(z["sr"]).item())
                        frame_sec = 512.0 / sr_val if sr_val > 0 else _DEFAULT_FRAME_SEC
                    else:
                        frame_sec = _DEFAULT_FRAME_SEC
                    if not math.isfinite(frame_sec) or frame_sec <= 0.0:
                        frame_sec = _DEFAULT_FRAME_SEC
                    t = np.arange(f0.size, dtype=float) * frame_sec
                if f0.size > 0:
                    segments.append({"t": t, "f0": f0, "rms": rms, "voiced_flag": vf, "frame_sec": frame_sec})
    except Exception:
        return []
    return segments


def _robust_linear_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 3 or y.size != x.size:
        return None
    span = float(x[-1] - x[0])
    if span <= 1e-9:
        return None

    xc = x - float(np.mean(x))
    yc = y - float(np.mean(y))
    denom = float(np.sum(xc * xc))
    if denom <= 1e-12:
        return None
    slope = float(np.sum(xc * yc) / denom)
    intercept = float(np.mean(y)) - slope * float(np.mean(x))

    resid = y - (intercept + slope * x)
    med_r = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med_r))) * 1.4826
    if mad > 1e-9:
        inliers = np.abs(resid - med_r) <= 3.0 * mad
        if int(inliers.sum()) >= 3:
            xi = x[inliers]
            yi = y[inliers]
            xci = xi - float(np.mean(xi))
            yci = yi - float(np.mean(yi))
            di = float(np.sum(xci * xci))
            if di > 1e-12:
                slope = float(np.sum(xci * yci) / di)
    return slope if math.isfinite(slope) else None


def _weighted_median(values: list[float], weights: list[float]) -> float | None:
    if not values:
        return None
    pairs = sorted(zip(values, weights), key=lambda p: p[0])
    vals = np.array([p[0] for p in pairs], dtype=float)
    wts = np.maximum(np.array([p[1] for p in pairs], dtype=float), 1e-9)
    cum = np.cumsum(wts)
    half = 0.5 * cum[-1]
    idx = int(np.searchsorted(cum, half, side="left"))
    idx = min(max(idx, 0), vals.size - 1)
    return float(vals[idx])


def _extract_voice_dynamics(
    logs: dict[str, Any],
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    voice_dfs = [
        df for key in ("voice", "voice_interview", "voice_research")
        if isinstance((df := logs.get(key)), pd.DataFrame) and not df.empty
    ]
    voice_log_available = len(voice_dfs) > 0

    praat_medians: dict[str, float | None] = {}
    for col, out_key in (
        ("hnr_db", "hnr_db_median"),
        ("jitter_local", "jitter_local_median"),
        ("shimmer_local", "shimmer_local_median"),
    ):
        vals = [arr[np.isfinite(arr)] for df in voice_dfs if (arr := _num_array(df, col)).size > 0]
        cat = np.concatenate(vals) if vals else np.array([], dtype=float)
        praat_medians[out_key] = _finite_float(np.median(cat)) if cat.size > 0 else None

    praat_available = any(v is not None for v in praat_medians.values())

    f0_slope: float | None = None
    burst_regularity: float | None = None
    energy_cv: float | None = None
    voiced_ratio: float | None = None
    n_voiced_frames = 0
    n_total_frames = 0
    n_bursts = 0

    if segments:
        burst_durs: list[float] = []
        seg_slopes: list[float] = []
        seg_weights: list[float] = []
        rms_voiced_parts: list[np.ndarray] = []

        for seg in segments:
            f0 = seg["f0"]
            rms = seg["rms"]
            vf = seg["voiced_flag"]
            frame_sec = float(seg["frame_sec"])
            n_total_frames += int(f0.size)
            n_v = int(vf.sum())
            n_voiced_frames += n_v

            if vf.size > 0:
                padded = np.concatenate(([False], vf, [False]))
                starts = np.flatnonzero(~padded[:-1] & padded[1:])
                ends = np.flatnonzero(padded[:-1] & ~padded[1:])
                durs = (ends - starts) * frame_sec
                burst_durs.extend(durs.tolist())

            if n_v >= 3:
                x_voiced = np.arange(n_v, dtype=float) * frame_sec
                semi = 12.0 * np.log2(f0[vf])
                slope = _robust_linear_slope(x_voiced, semi)
                if slope is not None:
                    seg_slopes.append(slope)
                    seg_weights.append(float(n_v * frame_sec))

            if rms.size == f0.size:
                rv = rms[vf & np.isfinite(rms)]
                if rv.size > 0:
                    rms_voiced_parts.append(rv)

        n_bursts = len(burst_durs)
        if n_total_frames > 0:
            voiced_ratio = float(n_voiced_frames) / float(n_total_frames)
        if seg_slopes:
            f0_slope = _weighted_median(seg_slopes, seg_weights)
        if len(burst_durs) >= 2:
            d_arr = np.asarray(burst_durs, dtype=float)
            mean_d = float(np.mean(d_arr))
            if mean_d > 1e-9:
                cv_d = float(np.std(d_arr) / mean_d)
                burst_regularity = 1.0 / (1.0 + cv_d)
        if rms_voiced_parts:
            r_cat = np.concatenate(rms_voiced_parts)
            if r_cat.size >= 2:
                energy_cv = float(np.std(r_cat) / (float(np.mean(r_cat)) + 1e-8))

    return {
        "voice_log_available": voice_log_available,
        "praat_features_available": praat_available,
        "npz_cache_available": len(segments) > 0,
        "hnr_db_median": praat_medians["hnr_db_median"],
        "jitter_local_median": praat_medians["jitter_local_median"],
        "shimmer_local_median": praat_medians["shimmer_local_median"],
        "f0_slope_semitone_per_sec": _finite_float(f0_slope),
        "voiced_burst_regularity": _finite_float(burst_regularity),
        "energy_modulation_cv": _finite_float(energy_cv),
        "voiced_frame_ratio": _finite_float(voiced_ratio),
        "n_voiced_frames": n_voiced_frames,
        "n_voiced_bursts": n_bursts,
        "n_segments": len(segments),
    }


def _compute_sparc(motion: np.ndarray, amp_th: float = 0.05) -> float | None:
    x = motion[np.isfinite(motion)]
    if x.size < 8 or float(np.ptp(x)) <= 1e-12:
        return None
    xc = x - float(np.mean(x))
    nfft = max(256, min(2048, 1 << (int(math.ceil(math.log2(xc.size))) + 2)))
    spec = np.abs(rfft(xc, n=nfft))
    peak = float(np.max(spec))
    if peak <= 1e-12 or not math.isfinite(peak):
        return None
    spec_norm = spec / peak
    freqs = rfftfreq(nfft, d=1.0)
    above = np.flatnonzero(spec_norm >= amp_th)
    k_end = int(above[-1]) + 1 if above.size >= 2 else max(2, spec_norm.size // 4)
    s_seg = spec_norm[:k_end]
    f_seg = freqs[:k_end]
    f_span = float(f_seg[-1] - f_seg[0])
    if f_span <= 0.0:
        return 0.0
    arc = float(-np.sum(np.hypot(np.diff(f_seg) / f_span, np.diff(s_seg))))
    return arc if math.isfinite(arc) else None


def _compute_log_dimensionless_jerk(motion: np.ndarray, dt: float = 1.0) -> float | None:
    x = motion[np.isfinite(motion)]
    if x.size < 8 or float(np.ptp(x)) <= 1e-12:
        return None
    v_peak = float(np.max(np.abs(x)))
    if v_peak <= 1e-12:
        return None
    t_dur = float((x.size - 1) * dt)
    d2v = np.diff(x, n=2) / (dt ** 2)
    jerk_int = float(np.sum(d2v ** 2) * dt)
    dlj = (t_dur ** 3 / (v_peak ** 2)) * jerk_int
    if dlj <= 1e-12:
        return 0.0
    val = float(-math.log(dlj))
    return val if math.isfinite(val) else None


def _extract_gesture_smoothness(gesture: pd.DataFrame | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "sparc": None,
        "log_dimensionless_jerk": None,
        "motion_smoothness_index": None,
        "dual_hand_visible_ratio": None,
        "pose_continuity_ratio": None,
        "motion_series_length": 0,
        "n_total_frames": int(len(gesture)) if isinstance(gesture, pd.DataFrame) else 0,
    }
    if gesture is None or not isinstance(gesture, pd.DataFrame) or gesture.empty:
        return out

    primary_cols = [c for c in ("left_wrist_jitter", "right_wrist_jitter", "torso_jitter") if c in gesture.columns]
    if primary_cols:
        mat = np.column_stack([_num_array(gesture, c) for c in primary_cols])
        out["pose_continuity_ratio"] = _finite_float(np.isfinite(mat).mean())
    else:
        mat = np.empty((len(gesture), 0), dtype=float)

    motion_cols = primary_cols or [c for c in ("left_hand_jitter", "right_hand_jitter") if c in gesture.columns]
    if motion_cols:
        m_mat = mat if primary_cols else np.column_stack([_num_array(gesture, c) for c in motion_cols])
        finite_mask = np.isfinite(m_mat)
        row_has = finite_mask.any(axis=1)
        if int(row_has.sum()) >= 8:
            sub = m_mat[row_has]
            cnt = np.isfinite(sub).sum(axis=1)
            motion = np.where(np.isfinite(sub), sub, 0.0).sum(axis=1) / cnt
            out["motion_series_length"] = int(motion.size)
            sparc_val = _compute_sparc(motion)
            ldj_val = _compute_log_dimensionless_jerk(motion)
            out["sparc"] = _finite_float(sparc_val)
            out["log_dimensionless_jerk"] = _finite_float(ldj_val)
            if sparc_val is not None:
                out["motion_smoothness_index"] = _finite_float(1.0 / (1.0 + abs(sparc_val) / 10.0))

    if "hand_visible_left" in gesture.columns and "hand_visible_right" in gesture.columns:
        hl = _num_array(gesture, "hand_visible_left")
        hr = _num_array(gesture, "hand_visible_right")
        obs_any = np.isfinite(hl) | np.isfinite(hr)
        if int(obs_any.sum()) > 0:
            if mat.shape[1] > 0:
                valid_frame = np.isfinite(mat).any(axis=1) | obs_any
            else:
                valid_frame = obs_any
            denom = int(valid_frame.sum())
            if denom > 0:
                both = (np.nan_to_num(hl, nan=0.0) > 0.5) & (np.nan_to_num(hr, nan=0.0) > 0.5)
                out["dual_hand_visible_ratio"] = _finite_float(float(both.sum()) / float(denom))

    return out


def _compute_catch22_stream(arr: np.ndarray) -> dict[str, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size < 16:
        return {}
    if float(np.std(finite)) <= 1e-12 or float(np.ptp(finite)) <= 1e-12:
        return {}
    if finite.size > _MAX_CATCH22_SAMPLES:
        step = int(math.ceil(finite.size / _MAX_CATCH22_SAMPLES))
        finite = finite[::step]
    try:
        res = pycatch22.catch22_all(finite)
    except Exception:
        return {}

    names = res.get("names", [])
    values = res.get("values", [])
    mapping = dict(zip(names, values))
    selected: dict[str, float] = {}
    for aliases in _CATCH22_METRIC_ALIASES:
        for key in aliases:
            if key in mapping:
                fv = _finite_float(mapping[key])
                if fv is not None:
                    selected[key] = fv
                break
    return selected


def _lookup_alias(stream_metrics: dict[str, float], aliases: tuple[str, ...]) -> float | None:
    for key in aliases:
        if key in stream_metrics:
            return stream_metrics[key]
    return None


def _extract_catch22_dynamics(
    logs: dict[str, Any],
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates: list[tuple[str, np.ndarray]] = []
    for modality, col, stream_name in _CATCH22_STREAM_SPECS:
        df = logs.get(modality)
        arr = _num_array(df, col)
        if arr.size >= 16:
            candidates.append((stream_name, arr))

    if segments:
        f0_all = np.concatenate([s["f0"] for s in segments])
        f0_voiced = f0_all[np.isfinite(f0_all) & (f0_all > 0.0)]
        if f0_voiced.size >= 16:
            candidates.append(("voiced_f0", f0_voiced))
        rms_parts = [s["rms"] for s in segments if s["rms"].size > 0]
        if rms_parts:
            rms_all = np.concatenate(rms_parts)
            rms_fin = rms_all[np.isfinite(rms_all)]
            if rms_fin.size >= 16:
                candidates.append(("rms", rms_fin))

    streams: dict[str, dict[str, float]] = {}
    for stream_name, arr in candidates[:6]:
        desc = _compute_catch22_stream(arr)
        if desc:
            streams[stream_name] = desc

    autocorr_vals: list[float] = []
    low_freq_vals: list[float] = []
    stretch_vals: list[float] = []
    pnn_vals: list[float] = []

    for metrics in streams.values():
        v_ac = _lookup_alias(metrics, _CATCH22_METRIC_ALIASES[0])
        v_sp = _lookup_alias(metrics, _CATCH22_METRIC_ALIASES[1])
        v_sb = _lookup_alias(metrics, _CATCH22_METRIC_ALIASES[2])
        v_md = _lookup_alias(metrics, _CATCH22_METRIC_ALIASES[3])
        if v_ac is not None:
            autocorr_vals.append(v_ac)
        if v_sp is not None:
            low_freq_vals.append(v_sp)
        if v_sb is not None:
            stretch_vals.append(v_sb)
        if v_md is not None:
            pnn_vals.append(v_md)

    return {
        "n_streams": len(streams),
        "streams": streams,
        "summary": {
            "catch22_streams_computed": list(streams.keys()),
            "autocorr_timescale_median": _finite_float(np.median(autocorr_vals)) if autocorr_vals else None,
            "low_freq_power_ratio_median": _finite_float(np.median(low_freq_vals)) if low_freq_vals else None,
            "persistence_stretch_median": _finite_float(np.median(stretch_vals)) if stretch_vals else None,
            "micro_variation_rate_median": _finite_float(np.median(pnn_vals)) if pnn_vals else None,
            "autocorr_timescale_mean": _finite_float(np.mean(autocorr_vals)) if autocorr_vals else None,
            "spectral_concentration_mean": _finite_float(np.mean(low_freq_vals)) if low_freq_vals else None,
        },
    }


def _build_framework_tags(
    face_kin: dict[str, Any],
    voice_dyn: dict[str, Any],
    gesture_sm: dict[str, Any],
    catch22_dyn: dict[str, Any],
) -> list[str]:
    tags: list[str] = []
    if voice_dyn["praat_features_available"] or voice_dyn["f0_slope_semitone_per_sec"] is not None:
        tags.append("Praat-Parselmouth")
    if voice_dyn["npz_cache_available"] or voice_dyn["voiced_burst_regularity"] is not None:
        tags.append("Silero-VAD-ONNX")
    if face_kin["pose_mode"] == "3d_matrix":
        tags.append("RTMW/MediaPipe-3D")
    elif face_kin["pose_mode"] == "2d_legacy":
        tags.append("MediaPipe-2D姿态代理")
    if gesture_sm["sparc"] is not None:
        tags.append("SPARC运动平滑度")
    if catch22_dyn["n_streams"] > 0:
        tags.append("pycatch22时间序列动力学")
    return tags


def _build_highlights(
    face_kin: dict[str, Any],
    voice_dyn: dict[str, Any],
    gesture_sm: dict[str, Any],
    catch22_dyn: dict[str, Any],
) -> list[dict[str, Any]]:
    is_3d = face_kin["pose_mode"] == "3d_matrix"
    c_sum = catch22_dyn["summary"]
    candidates = [
        (
            "head_angular_velocity_3d_p50",
            "头部三维角速度中位" if is_3d else "头部二维速度代理中位",
            face_kin["head_angular_velocity_3d_p50"],
            face_kin["unit"],
            "face_kinematics",
            "反映面部朝向每秒角度变化幅度" if is_3d else "基于二维朝向代理计算的帧间变化速率",
        ),
        (
            "frame_quality_pass_rate",
            "面部有效帧通过率",
            face_kin["frame_quality_pass_rate"],
            "比例",
            "face_kinematics",
            "清晰度达标且成功检出面部关键点的帧占比",
        ),
        (
            "hnr_db_median",
            "谐噪比中位",
            voice_dyn["hnr_db_median"],
            "dB",
            "voice_dynamics",
            "声带周期性发声信号与噪声成分的比值中位数",
        ),
        (
            "f0_slope_semitone_per_sec",
            "基频半音斜率",
            voice_dyn["f0_slope_semitone_per_sec"],
            "半音/秒",
            "voice_dynamics",
            "有声时段音高在半音尺度上的线性变化速率",
        ),
        (
            "voiced_burst_regularity",
            "发声时段规整度",
            voice_dyn["voiced_burst_regularity"],
            "0-1",
            "voice_dynamics",
            "各段连续有声时长的一致性，越接近 1 越均匀",
        ),
        (
            "sparc",
            "上肢运动频谱平滑度",
            gesture_sm["sparc"],
            "SPARC",
            "gesture_smoothness",
            "动作速率频谱弧长，越接近 0 表示轨迹越平顺",
        ),
        (
            "dual_hand_visible_ratio",
            "双手同时入镜占比",
            gesture_sm["dual_hand_visible_ratio"],
            "比例",
            "gesture_smoothness",
            "左右手均被清晰检出的有效帧占比",
        ),
        (
            "low_freq_power_ratio_median",
            "多模态低频功率集中度中位",
            c_sum["low_freq_power_ratio_median"],
            "0-1",
            "catch22_dynamics",
            "跨通道时间序列慢变节律所占的能量比例中位数",
        ),
        (
            "energy_modulation_cv",
            "有声能量调制变异系数",
            voice_dyn["energy_modulation_cv"],
            "无量纲",
            "voice_dynamics",
            "发声片段短时均方根能量的相对波动幅度",
        ),
        (
            "pose_continuity_ratio",
            "上肢关键点连续覆盖率",
            gesture_sm["pose_continuity_ratio"],
            "比例",
            "gesture_smoothness",
            "手腕与躯干位移通道的有限观测覆盖均值",
        ),
    ]

    highlights: list[dict[str, Any]] = []
    for key, label, val, unit, track, note in candidates:
        if val is None:
            continue
        highlights.append({
            "key": key,
            "label": label,
            "value": val,
            "unit": unit,
            "track": track,
            "note": note,
        })
        if len(highlights) >= 8:
            break
    return highlights


def _build_summary_plain(
    available: bool,
    face_kin: dict[str, Any],
    voice_dyn: dict[str, Any],
    gesture_sm: dict[str, Any],
    catch22_dyn: dict[str, Any],
) -> str:
    if not available:
        return "本场暂未获取到可用于动力学特征提取的日志或语音序列。"

    parts: list[str] = []
    if face_kin["pose_mode"] == "3d_matrix" and face_kin["head_angular_velocity_3d_p50"] is not None:
        parts.append(
            f"面部采用三维姿态矩阵口径，头部角速度中位数为 {face_kin['head_angular_velocity_3d_p50']:.2f} 度/秒"
        )
    elif face_kin["pose_mode"] == "2d_legacy" and face_kin["head_angular_velocity_3d_p50"] is not None:
        parts.append(
            f"面部采用二维姿态代理口径，头部变化速率中位数为 {face_kin['head_angular_velocity_3d_p50']:.4f} 归一化位移/秒"
        )
    if face_kin["frame_quality_pass_rate"] is not None:
        parts.append(f"面部有效帧通过率为 {face_kin['frame_quality_pass_rate'] * 100:.1f}%")

    if voice_dyn["n_voiced_bursts"] > 0 and voice_dyn["voiced_burst_regularity"] is not None:
        parts.append(
            f"语音识别到 {voice_dyn['n_voiced_bursts']} 段连续有声片段，发声时长规整度为 {voice_dyn['voiced_burst_regularity']:.2f}"
        )
    if voice_dyn["hnr_db_median"] is not None:
        parts.append(f"语音谐噪比中位数为 {voice_dyn['hnr_db_median']:.2f} dB")

    if gesture_sm["sparc"] is not None:
        parts.append(f"上肢动作频谱平滑度为 {gesture_sm['sparc']:.2f}")
    if gesture_sm["dual_hand_visible_ratio"] is not None:
        parts.append(f"双手同时入镜占比为 {gesture_sm['dual_hand_visible_ratio'] * 100:.1f}%")

    if catch22_dyn["n_streams"] > 0:
        lfp = catch22_dyn["summary"]["low_freq_power_ratio_median"]
        if lfp is not None:
            parts.append(f"时间序列动力学共提取 {catch22_dyn['n_streams']} 条信号通道，低频功率集中度中位为 {lfp:.2f}")
        else:
            parts.append(f"时间序列动力学共提取 {catch22_dyn['n_streams']} 条信号通道")

    if not parts:
        return "本场暂未获取到可用于动力学特征提取的日志或语音序列。"
    return "；".join(parts) + "。"


def _empty_payload() -> dict[str, Any]:
    return {
        "available": False,
        "pose_mode": "unavailable",
        "framework_tags": [],
        "tracks": {
            "face_kinematics": {
                "pose_mode": "unavailable",
                "unit": "",
                "head_angular_velocity_3d_p50": None,
                "head_angular_velocity_3d_p90": None,
                "frame_quality_pass_rate": None,
                "quality_source": "unavailable",
                "n_valid_frames": 0,
                "n_total_frames": 0,
            },
            "voice_dynamics": {
                "voice_log_available": False,
                "praat_features_available": False,
                "npz_cache_available": False,
                "hnr_db_median": None,
                "jitter_local_median": None,
                "shimmer_local_median": None,
                "f0_slope_semitone_per_sec": None,
                "voiced_burst_regularity": None,
                "energy_modulation_cv": None,
                "voiced_frame_ratio": None,
                "n_voiced_frames": 0,
                "n_voiced_bursts": 0,
                "n_segments": 0,
            },
            "gesture_smoothness": {
                "sparc": None,
                "log_dimensionless_jerk": None,
                "motion_smoothness_index": None,
                "dual_hand_visible_ratio": None,
                "pose_continuity_ratio": None,
                "motion_series_length": 0,
                "n_total_frames": 0,
            },
            "catch22_dynamics": {
                "n_streams": 0,
                "streams": {},
                "summary": {
                    "catch22_streams_computed": [],
                    "autocorr_timescale_median": None,
                    "low_freq_power_ratio_median": None,
                    "persistence_stretch_median": None,
                    "micro_variation_rate_median": None,
                    "autocorr_timescale_mean": None,
                    "spectral_concentration_mean": None,
                },
            },
        },
        "highlights": [],
        "summary_plain": "本场暂未获取到可用于动力学特征提取的日志或语音序列。",
        "inference_ms": 0.0,
    }


def extract_feature_layer_v2(
    logs: dict | None,
    session_id: str | None = None,
    *,
    l0_series_dir: Path | None = None,
) -> dict[str, Any]:
    """从已加载的多模态日志与本地语音帧序列缓存提取第二层动力学特征。"""
    safe_logs = logs if isinstance(logs, dict) else {}
    has_nonempty_df = any(
        isinstance(df, pd.DataFrame) and not df.empty
        for df in safe_logs.values()
    )
    if not has_nonempty_df and (not session_id or session_id == "NONE"):
        return _empty_payload()

    t0 = time.perf_counter()
    segments = _load_npz_segments(session_id, l0_series_dir)

    if not has_nonempty_df and not segments:
        return _empty_payload()

    face_kin = _extract_face_kinematics(safe_logs.get("face"))
    voice_dyn = _extract_voice_dynamics(safe_logs, segments)
    gesture_sm = _extract_gesture_smoothness(safe_logs.get("gesture"))
    catch22_dyn = _extract_catch22_dynamics(safe_logs, segments)

    available = any((
        face_kin["pose_mode"] != "unavailable",
        face_kin["frame_quality_pass_rate"] is not None,
        voice_dyn["praat_features_available"],
        voice_dyn["npz_cache_available"],
        gesture_sm["sparc"] is not None,
        gesture_sm["dual_hand_visible_ratio"] is not None,
        gesture_sm["pose_continuity_ratio"] is not None,
        catch22_dyn["n_streams"] > 0,
    ))

    if not available:
        return _empty_payload()

    framework_tags = _build_framework_tags(face_kin, voice_dyn, gesture_sm, catch22_dyn)
    highlights = _build_highlights(face_kin, voice_dyn, gesture_sm, catch22_dyn)
    summary_plain = _build_summary_plain(available, face_kin, voice_dyn, gesture_sm, catch22_dyn)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    return {
        "available": True,
        "pose_mode": face_kin["pose_mode"],
        "framework_tags": framework_tags,
        "tracks": {
            "face_kinematics": face_kin,
            "voice_dynamics": voice_dyn,
            "gesture_smoothness": gesture_sm,
            "catch22_dynamics": catch22_dyn,
        },
        "highlights": highlights,
        "summary_plain": summary_plain,
        "inference_ms": elapsed_ms,
    }
