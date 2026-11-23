# report_frontend/behavior_regime_model.py
"""无监督多模态行为节奏分段模型（纯 CPU 毫秒级，仅描述外显动作与发声节律）。

设计边界：
- 仅基于 L0 面部、上肢姿态及本地缓存的语音浊音帧序列，提取客观可观测运动学与发声节律特征；
- 严格只读（`advisory_only: True`），不做任何主观特质归因，不产出数值汇总；
- 训练阶段按 `session_id` 分组做 GroupKFold 验证，推理阶段全本地纯 CPU 运行。
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
from numpy.lib.stride_tricks import sliding_window_view

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "behavior_regime_model.joblib"
DEFAULT_METRICS_PATH = Path(__file__).resolve().parent / "behavior_regime_metrics.json"

WINDOW_SIZE = 8
WINDOW_STRIDE = 4

FEATURE_NAMES: tuple[str, ...] = (
    "face_valid_ratio",
    "mouth_open_mean",
    "mouth_motion_std",
    "gaze_dev_mean",
    "head_motion_std",
    "upper_body_motion",
    "hand_visible_or_spread",
    "voiced_ratio",
)

STATE_ORDER: tuple[str, ...] = (
    "steady_speaking",
    "active_expressive",
    "pause_thinking",
    "low_signal",
)

STATE_META: dict[str, dict[str, str]] = {
    "steady_speaking": {
        "label": "平稳陈述态",
        "description": "面部朝向平稳，口部持续开合并伴随稳定发声，肢体动作幅度较小",
        "color": "#2E86AB",
        "color_key": "steady",
    },
    "active_expressive": {
        "label": "活跃表达态",
        "description": "头部转动、视线切换或上肢动作较为活跃，肢体配合表达明显",
        "color": "#28A745",
        "color_key": "active",
    },
    "pause_thinking": {
        "label": "静默停顿态",
        "description": "口部开合幅度较低且发声占比很少，动作平缓，处于停顿或构思阶段",
        "color": "#64748B",
        "color_key": "pause",
    },
    "low_signal": {
        "label": "信号受限态",
        "description": "面部有效观测比例偏低，存在遮挡、转头出镜或传感采样中断",
        "color": "#94A3B8",
        "color_key": "low_signal",
    },
}

_MODEL_CACHE: dict[str, Any] = {"path": None, "mtime": None, "artifact": None}
_METRICS_CACHE: dict[str, Any] = {"path": None, "mtime": None, "metrics": None}


def _num_series(df: pd.DataFrame | None, col: str) -> np.ndarray:
    if df is None or df.empty or col not in df.columns:
        return np.array([], dtype=float)
    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        return s.to_numpy(dtype=float, na_value=np.nan)
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def _pad_to(arr: np.ndarray, n_total: int, fill: float = float("nan")) -> np.ndarray:
    if arr.size >= n_total:
        return arr[:n_total]
    out = np.full(n_total, fill, dtype=arr.dtype)
    if arr.size > 0:
        out[: arr.size] = arr
    return out


def _win_mean(W: np.ndarray) -> np.ndarray:
    valid = np.isfinite(W)
    cnt = valid.sum(axis=1)
    s = np.where(valid, W, 0.0).sum(axis=1)
    out = np.full(W.shape[0], np.nan, dtype=float)
    nz = cnt > 0
    out[nz] = s[nz] / cnt[nz]
    return out


def _win_std(W: np.ndarray) -> np.ndarray:
    valid = np.isfinite(W)
    cnt = valid.sum(axis=1)
    out = np.full(W.shape[0], np.nan, dtype=float)
    ge2 = cnt >= 2
    if not ge2.any():
        return out
    Wg = W[ge2]
    vg = valid[ge2]
    cg = cnt[ge2].astype(float)
    mean = np.where(vg, Wg, 0.0).sum(axis=1, keepdims=True) / cg[:, None]
    diff = np.where(vg, Wg - mean, 0.0)
    var = (diff * diff).sum(axis=1) / cg
    out[ge2] = np.sqrt(np.maximum(var, 0.0))
    return out


def _win_abs_diff_mean(arr: np.ndarray, n_total: int, window_size: int, stride: int, n_win: int) -> np.ndarray:
    if arr.size == 0:
        return np.full(n_win, np.nan, dtype=float)
    W = sliding_window_view(_pad_to(arr, n_total), window_size)[::stride]
    fin = np.isfinite(W)
    all_fin = fin.all(axis=1)
    out = np.full(n_win, np.nan, dtype=float)
    if all_fin.any():
        out[all_fin] = np.abs(np.diff(W[all_fin], axis=1)).mean(axis=1)
    need_sparse = (~all_fin) & (fin.sum(axis=1) >= 2)
    if need_sparse.any():
        for i in np.flatnonzero(need_sparse):
            vals = W[i, fin[i]]
            out[i] = float(np.mean(np.abs(np.diff(vals))))
    return out


def _win_multi_mean(arrs: list[np.ndarray], n_total: int, window_size: int, stride: int, n_win: int) -> np.ndarray:
    if not arrs:
        return np.full(n_win, np.nan, dtype=float)
    cols = [
        _win_mean(sliding_window_view(_pad_to(a, n_total), window_size)[::stride])
        for a in arrs
    ]
    return _win_mean(np.column_stack(cols))


def _relative_timestamps(df: pd.DataFrame | None) -> np.ndarray | None:
    ts = _num_series(df, "timestamp")
    if ts.size == 0:
        return None
    fin_idx = np.flatnonzero(np.isfinite(ts))
    if fin_idx.size == 0:
        return None
    return ts - float(ts[fin_idx[0]])


def _select_whole_voice_series(voice_series: Any, session_span_sec: float):
    if voice_series is None:
        return None
    candidates = voice_series if isinstance(voice_series, (list, tuple)) else [voice_series]
    for item in candidates:
        if item is None:
            continue
        if getattr(item, "index", None) is not None:
            continue
        frame_sec = float(getattr(item, "frame_sec", 0.0) or 0.0)
        f0 = getattr(item, "f0", None)
        if frame_sec <= 0.0 or f0 is None or len(f0) == 0:
            continue
        dur = len(f0) * frame_sec
        if dur >= 0.80 * max(session_span_sec, 1.0):
            return item, frame_sec, dur, np.asarray(f0, dtype=float)
    return None


def extract_regime_windows(
    face_df: pd.DataFrame | None,
    gesture_df: pd.DataFrame | None,
    voice_series: Any = None,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """将面部、手势与可选语音序列切分为滑动时间窗并提取 8 维客观行为节律特征。"""
    n_face = len(face_df) if isinstance(face_df, pd.DataFrame) else 0
    n_gest = len(gesture_df) if isinstance(gesture_df, pd.DataFrame) else 0
    n_total = max(n_face, n_gest)
    if window_size < 2 or stride < 1 or n_total < window_size:
        return np.empty((0, len(FEATURE_NAMES)), dtype=float), []

    f_rel = _relative_timestamps(face_df)
    g_rel = _relative_timestamps(gesture_df)
    if f_rel is not None and np.isfinite(f_rel).any():
        session_span_sec = float(np.nanmax(f_rel))
    elif g_rel is not None and np.isfinite(g_rel).any():
        session_span_sec = float(np.nanmax(g_rel))
    else:
        session_span_sec = float(n_total - 1)

    # 预提取各模态数组，避免滑动窗循环内重复调用 pandas 解析
    if n_face > 0 and isinstance(face_df, pd.DataFrame):
        au_cols = [c for c in face_df.columns if c.startswith("au") and "_" in c][:8]
        extra_face = [
            c for c in ("gaze_deviation", "gaze_direction_x", "gaze_direction_y", "head_yaw", "bs_jawOpen")
            if c in face_df.columns
        ]
        check_cols = au_cols + extra_face
        if check_cols:
            mat = np.column_stack([_num_series(face_df, c) for c in check_cols])
            face_valid_mask = np.isfinite(mat).any(axis=1)
        else:
            face_valid_mask = np.zeros(n_face, dtype=bool)

        mouth_arr = _num_series(face_df, "au25_mouth_open")
        if not np.isfinite(mouth_arr).any():
            mouth_arr = _num_series(face_df, "bs_jawOpen")

        gaze_arr = _num_series(face_df, "gaze_deviation")
        if not np.isfinite(gaze_arr).any():
            gx = _num_series(face_df, "gaze_direction_x")
            gy = _num_series(face_df, "gaze_direction_y")
            if gx.size and gy.size and gx.size == gy.size:
                gaze_arr = np.hypot(gx, gy)

        head_yaw_arr = _num_series(face_df, "head_yaw")
        head_pitch_arr = _num_series(face_df, "head_pitch")
    else:
        face_valid_mask = np.zeros(0, dtype=bool)
        mouth_arr = np.array([], dtype=float)
        gaze_arr = np.array([], dtype=float)
        head_yaw_arr = np.array([], dtype=float)
        head_pitch_arr = np.array([], dtype=float)

    if n_gest > 0 and isinstance(gesture_df, pd.DataFrame):
        primary_upper = [
            _num_series(gesture_df, c)
            for c in ("left_wrist_jitter_world", "right_wrist_jitter_world", "torso_jitter")
        ]
        upper_arrs = [a for a in primary_upper if a.size and np.isfinite(a).any()]
        if not upper_arrs:
            fallback_upper = [
                _num_series(gesture_df, c)
                for c in ("left_wrist_jitter", "right_wrist_jitter", "left_hand_jitter", "right_hand_jitter")
            ]
            upper_arrs = [a for a in fallback_upper if a.size and np.isfinite(a).any()]

        spread_arrs = [
            _num_series(gesture_df, c)
            for c in ("left_hand_spread", "right_hand_spread")
        ]
        spread_arrs = [a for a in spread_arrs if a.size and np.isfinite(a).any()]
        if not spread_arrs:
            vis_arrs = [
                _num_series(gesture_df, c)
                for c in ("hand_visible_left", "hand_visible_right")
            ]
            spread_arrs = [a for a in vis_arrs if a.size and np.isfinite(a).any()]
    else:
        upper_arrs = []
        spread_arrs = []

    voice_pack = _select_whole_voice_series(voice_series, session_span_sec)
    if voice_pack is not None:
        _, v_frame_sec, v_dur, v_f0 = voice_pack
        voiced_cumsum = np.concatenate(([0], np.cumsum(np.isfinite(v_f0), dtype=np.int64)))
        v_len = len(v_f0)
    else:
        v_frame_sec = 0.0
        v_dur = 0.0
        voiced_cumsum = np.zeros(1, dtype=np.int64)
        v_len = 0

    starts = np.arange(0, n_total - window_size + 1, stride, dtype=int)
    n_win = int(len(starts))

    # 确定窗口起止相对时间
    s_secs = starts.astype(float)
    e_secs = (starts + window_size - 1).astype(float)
    t_arr = f_rel if (f_rel is not None and np.isfinite(f_rel).any()) else g_rel
    if t_arr is not None and np.isfinite(t_arr).any():
        t_W = sliding_window_view(_pad_to(t_arr, n_total), window_size)[::stride]
        t_fin = np.isfinite(t_W)
        all_t = t_fin.all(axis=1)
        if all_t.any():
            s_secs[all_t] = t_W[all_t, 0]
            e_secs[all_t] = t_W[all_t, -1]
        part_t = (~all_t) & t_fin.any(axis=1)
        if part_t.any():
            for i in np.flatnonzero(part_t):
                vals = t_W[i, t_fin[i]]
                s_secs[i] = float(vals[0])
                e_secs[i] = float(vals[-1])
    e_secs = np.maximum(e_secs, s_secs)

    # 1. face_valid_ratio
    if n_face > 0:
        fv_pad = _pad_to(face_valid_mask.astype(float), n_total, fill=0.0)
        face_valid_col = sliding_window_view(fv_pad, window_size)[::stride].sum(axis=1) / float(window_size)
    else:
        face_valid_col = np.zeros(n_win, dtype=float)

    # 2 & 3. mouth_open_mean, mouth_motion_std
    if n_face > 0 and mouth_arr.size:
        W_mo = sliding_window_view(_pad_to(mouth_arr, n_total), window_size)[::stride]
        mouth_mean_col = _win_mean(W_mo)
        mouth_std_col = _win_std(W_mo)
    else:
        mouth_mean_col = np.full(n_win, np.nan, dtype=float)
        mouth_std_col = np.full(n_win, np.nan, dtype=float)

    # 4. gaze_dev_mean
    if n_face > 0 and gaze_arr.size:
        W_gz = sliding_window_view(_pad_to(gaze_arr, n_total), window_size)[::stride]
        gaze_mean_col = _win_mean(W_gz)
    else:
        gaze_mean_col = np.full(n_win, np.nan, dtype=float)

    # 5. head_motion_std（逐帧转角变化幅值均值）
    if n_face > 0 and (head_yaw_arr.size or head_pitch_arr.size):
        h_cols = [
            _win_abs_diff_mean(arr, n_total, window_size, stride, n_win)
            for arr in (head_yaw_arr, head_pitch_arr)
            if arr.size
        ]
        head_motion_col = _win_mean(np.column_stack(h_cols)) if h_cols else np.full(n_win, np.nan, dtype=float)
    else:
        head_motion_col = np.full(n_win, np.nan, dtype=float)

    # 6. upper_body_motion
    upper_motion_col = _win_multi_mean(upper_arrs, n_total, window_size, stride, n_win)

    # 7. hand_visible_or_spread
    hand_spread_col = _win_multi_mean(spread_arrs, n_total, window_size, stride, n_win)

    # 8. voiced_ratio
    voiced_col = np.full(n_win, np.nan, dtype=float)
    if v_len > 0:
        ok = (e_secs > s_secs) & (s_secs >= 0.0) & (s_secs < v_dur) & (e_secs <= v_dur + 2.0)
        if ok.any():
            for i in np.flatnonzero(ok):
                s_s = float(s_secs[i])
                e_s = float(e_secs[i])
                i0 = max(0, int(s_s / v_frame_sec))
                i1 = min(v_len, int(math.ceil(e_s / v_frame_sec)))
                if i1 > i0 and (i1 - i0) * v_frame_sec >= 0.5 * (e_s - s_s):
                    voiced_col[i] = float(voiced_cumsum[i1] - voiced_cumsum[i0]) / float(i1 - i0)

    X = np.column_stack([
        face_valid_col,
        mouth_mean_col,
        mouth_std_col,
        gaze_mean_col,
        head_motion_col,
        upper_motion_col,
        hand_spread_col,
        voiced_col,
    ])
    X[~np.isfinite(X)] = np.nan

    meta = [
        {
            "window_index": idx,
            "start_row": int(st),
            "end_row": int(st + window_size),
            "start_sec": round(float(s_secs[idx]), 3),
            "end_sec": round(float(e_secs[idx]), 3),
        }
        for idx, st in enumerate(starts)
    ]
    return X, meta


def assign_cluster_states(
    centroids_z: np.ndarray,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> dict[int, str]:
    """根据标准化空间簇质心的物理含义，确定性映射簇序号到行为节律状态。"""
    k = int(centroids_z.shape[0])
    idx = {name: i for i, name in enumerate(feature_names)}
    remaining = set(range(k))
    mapping: dict[int, str] = {}

    if not remaining:
        return mapping

    # 1. 信号受限态：面部有效观测率最低的簇
    c_low = min(remaining, key=lambda c: float(centroids_z[c, idx["face_valid_ratio"]]))
    mapping[c_low] = "low_signal"
    remaining.remove(c_low)

    if not remaining:
        return mapping

    # 2. 活跃表达态：可见簇中口部变化、头动幅度与上肢运动之和最高者
    c_active = max(
        remaining,
        key=lambda c: float(
            centroids_z[c, idx["mouth_motion_std"]]
            + centroids_z[c, idx["upper_body_motion"]]
            + centroids_z[c, idx["head_motion_std"]]
        ),
    )
    mapping[c_active] = "active_expressive"
    remaining.remove(c_active)

    if not remaining:
        return mapping

    # 3. 静默停顿态：余下可见簇中口部开合均值、变化度与浊音占比之和最低者
    if len(remaining) >= 2:
        c_pause = min(
            remaining,
            key=lambda c: float(
                centroids_z[c, idx["mouth_open_mean"]]
                + centroids_z[c, idx["mouth_motion_std"]]
                + centroids_z[c, idx["voiced_ratio"]]
            ),
        )
        mapping[c_pause] = "pause_thinking"
        remaining.remove(c_pause)

    # 4. 平稳陈述态：其余可见稳定发声/聆听簇
    for c in sorted(remaining):
        mapping[c] = "steady_speaking"

    return mapping


def _rule_predict_states(X: np.ndarray) -> list[str]:
    """无模型文件时的纯规则降级映射（保持与 4 个客观行为态语义一致）。"""
    out: list[str] = []
    for row in X:
        fv, mo_m, mo_s, gd, hm, um, _hs, vr = row
        if not math.isfinite(fv) or fv < 0.5:
            out.append("low_signal")
        elif (
            (math.isfinite(hm) and hm > 0.04)
            or (math.isfinite(um) and um > 0.004)
            or (math.isfinite(gd) and gd > 0.35)
        ):
            out.append("active_expressive")
        elif (
            (math.isfinite(mo_m) and mo_m < 0.012 and (not math.isfinite(mo_s) or mo_s < 0.008))
            or (math.isfinite(vr) and vr < 0.15 and (not math.isfinite(mo_m) or mo_m < 0.02))
        ):
            out.append("pause_thinking")
        else:
            out.append("steady_speaking")
    return out


def smooth_state_sequence(states: list[str], window: int = 3) -> list[str]:
    """对连续滑动窗状态标签做多数平滑滤波，消除单窗毛刺跳变。"""
    n = len(states)
    if n < 3 or window < 3:
        return list(states)
    half = window // 2
    smoothed: list[str] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        nbrs = states[lo:hi]
        if len(nbrs) >= 3:
            top_state, top_count = Counter(nbrs).most_common(1)[0]
            if top_count > len(nbrs) / 2.0:
                smoothed.append(top_state)
                continue
        smoothed.append(states[i])
    return smoothed


def merge_state_segments(
    states: list[str],
    meta: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """将相邻同态滑动窗合并为连续不重叠的时间区段。"""
    if not states or len(states) != len(meta):
        return []

    raw_groups: list[dict[str, Any]] = []
    for st, m in zip(states, meta):
        s_sec = float(m["start_sec"])
        e_sec = float(m["end_sec"])
        if raw_groups and raw_groups[-1]["state"] == st:
            raw_groups[-1]["end_sec"] = max(raw_groups[-1]["end_sec"], e_sec)
        else:
            raw_groups.append({
                "state": st,
                "start_sec": s_sec,
                "end_sec": e_sec,
            })

    # 滑动窗步长小于窗宽时，在相邻异态过渡中点切分重叠区间，保证时间轴连续不重算
    for i in range(len(raw_groups) - 1):
        cur = raw_groups[i]
        nxt = raw_groups[i + 1]
        if nxt["start_sec"] < cur["end_sec"]:
            mid = 0.5 * (cur["end_sec"] + nxt["start_sec"])
            mid = max(cur["start_sec"], min(nxt["end_sec"], mid))
            cur["end_sec"] = mid
            nxt["start_sec"] = mid

    segments: list[dict[str, Any]] = []
    for grp in raw_groups:
        st = grp["state"]
        s_sec = round(float(grp["start_sec"]), 2)
        e_sec = round(float(max(grp["start_sec"], grp["end_sec"])), 2)
        dur = round(max(0.0, e_sec - s_sec), 2)
        info = STATE_META[st]
        segments.append({
            "state": st,
            "label": info["label"],
            "start_sec": s_sec,
            "end_sec": e_sec,
            "duration_sec": dur,
            "color_key": info["color_key"],
        })
    return segments


def _load_artifact(model_path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any] | None:
    if not model_path.is_file():
        return None
    try:
        mtime = model_path.stat().st_mtime_ns
        if _MODEL_CACHE["path"] == str(model_path) and _MODEL_CACHE["mtime"] == mtime:
            return _MODEL_CACHE["artifact"]
        import joblib
        artifact = joblib.load(model_path)
        if not isinstance(artifact, dict) or "preprocessor" not in artifact or "gmm" not in artifact:
            return None
        _MODEL_CACHE.update({"path": str(model_path), "mtime": mtime, "artifact": artifact})
        return artifact
    except Exception:
        return None


def _load_metrics(metrics_path: Path = DEFAULT_METRICS_PATH) -> dict[str, Any] | None:
    if not metrics_path.is_file():
        return None
    try:
        mtime = metrics_path.stat().st_mtime_ns
        if _METRICS_CACHE["path"] == str(metrics_path) and _METRICS_CACHE["mtime"] == mtime:
            return _METRICS_CACHE["metrics"]
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        _METRICS_CACHE.update({"path": str(metrics_path), "mtime": mtime, "metrics": metrics})
        return metrics
    except Exception:
        return None


def _build_summary_plain(
    segments: list[dict[str, Any]],
    state_ratios: dict[str, float],
    transition_count: int,
    total_duration_sec: float,
) -> str:
    ranked = sorted(
        ((st, state_ratios.get(st, 0.0)) for st in STATE_ORDER),
        key=lambda item: item[1],
        reverse=True,
    )
    parts = [
        f"{STATE_META[st]['label']}占比 {round(ratio * 100)}%"
        for st, ratio in ranked
        if round(ratio * 100) > 0
    ]
    if transition_count == 0:
        trans_desc = "全程节奏保持一致"
    elif total_duration_sec > 0.0 and (transition_count / max(total_duration_sec / 60.0, 1e-6)) <= 4.0:
        trans_desc = "切换平稳"
    else:
        trans_desc = "区间切换较频繁"

    if parts:
        return f"整场共记录 {len(segments)} 段行为节奏，{'，'.join(parts)}，{trans_desc}。"
    return f"整场共记录 {len(segments)} 段行为节奏，{trans_desc}。"


def segment_session_regimes(
    logs: dict[str, pd.DataFrame] | None,
    session_id: str | None = None,
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
) -> dict[str, Any]:
    """对单场 L0 遥测与可选缓存语音序列执行纯 CPU 行为节奏分段（只读提示）。"""
    t0 = time.perf_counter()
    face_df = (logs or {}).get("face")
    gesture_df = (logs or {}).get("gesture")

    voice_list = None
    if session_id:
        try:
            from report_frontend import voice_series as voice_series_mod
            voice_list = voice_series_mod.load_series(session_id)
        except Exception:
            voice_list = None

    X, meta = extract_regime_windows(face_df, gesture_df, voice_series=voice_list)
    model_metrics = _load_metrics(metrics_path)

    if len(X) == 0:
        return {
            "available": False,
            "advisory_only": True,
            "session_id": session_id,
            "n_windows": 0,
            "segments": [],
            "state_counts": {},
            "state_ratios": {},
            "state_meta": STATE_META,
            "transition_count": 0,
            "total_duration_sec": 0.0,
            "summary_plain": "有效观测帧不足，暂未切分行为节奏区间。",
            "disclaimer": "只描述外显动作与发声节奏，不做主观归因，不产出数值汇总。",
            "model_used": False,
            "model_type": None,
            "model_metrics": model_metrics,
            "inference_ms": 0.0,
        }

    artifact = _load_artifact(model_path)
    if artifact is not None:
        X_std = artifact["preprocessor"].transform(X)
        cluster_ids = artifact["gmm"].predict(X_std)
        cluster_map = artifact["cluster_to_state"]
        raw_states = [cluster_map[int(cid)] for cid in cluster_ids]
        model_used = True
        model_type = type(artifact["gmm"]).__name__
    else:
        raw_states = _rule_predict_states(X)
        model_used = False
        model_type = "rule_fallback"

    smoothed_states = smooth_state_sequence(raw_states, window=3)
    segments = merge_state_segments(smoothed_states, meta)

    win_counter = Counter(smoothed_states)
    state_counts = {st: int(win_counter.get(st, 0)) for st in STATE_ORDER}

    dur_by_state = {st: 0.0 for st in STATE_ORDER}
    for seg in segments:
        dur_by_state[seg["state"]] += float(seg["duration_sec"])
    total_duration_sec = round(sum(dur_by_state.values()), 2)

    if total_duration_sec > 0.0:
        state_ratios = {
            st: round(dur_by_state[st] / total_duration_sec, 4)
            for st in STATE_ORDER
        }
    else:
        n_win = float(len(smoothed_states))
        state_ratios = {
            st: round(state_counts[st] / n_win, 4)
            for st in STATE_ORDER
        }

    transition_count = max(0, len(segments) - 1)
    summary_plain = _build_summary_plain(segments, state_ratios, transition_count, total_duration_sec)

    return {
        "available": True,
        "advisory_only": True,
        "session_id": session_id,
        "n_windows": int(len(X)),
        "transition_count": int(transition_count),
        "total_duration_sec": total_duration_sec,
        "state_meta": STATE_META,
        "state_counts": state_counts,
        "state_ratios": state_ratios,
        "segments": segments,
        "summary_plain": summary_plain,
        "disclaimer": "只客观描述作答时的外显动作与发声节奏，不代表任何个人特质，也不参与统计汇总。",
        "model_used": model_used,
        "model_type": model_type,
        "model_metrics": model_metrics,
        "inference_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }
