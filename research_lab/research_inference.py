#!/usr/bin/env python3
"""Offline research inference API across multi-corpus multimodal models.

Provides ``predict_research_profile(face_df, gesture_df, voice_series, transcript_text)``
which maps L0 face/gesture logs, optional voice series, and optional transcript text
to research-only projections from:
  - ChaLearn First Impressions V2 behavioral model
  - RecruitView multimodal personality & interview model
  - MIT Interview observable behavioral correlates model
  - Cross-Corpus Bilingual Behavioral Regime GMM (6 shared audiovisual features)

Strictly tagged ``research_only: True``; never invokes or modifies ``report_frontend/`` scoring.
"""
from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from face_expression.models.blendshapes import BLENDSHAPE_NAMES
from research_lab.extract_multicorpus_features import (
    BILATERAL_BLENDSHAPE_GROUPS,
    KEY_GEOM_FEATURES,
)
from research_lab.train_multicorpus_models import (
    REGIME_STATE_ORDER,
    SHARED_REGIME_FEATURES,
    clean_transcript_text,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = REPO_ROOT / "models" / "research"

_ARTIFACT_CACHE: dict[str, tuple[int, Any]] = {}


def _load_joblib_cached(path: Path) -> Any | None:
    if not path.is_file():
        return None
    mtime = path.stat().st_mtime_ns
    key = str(path)
    cached = _ARTIFACT_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    obj = joblib.load(path)
    _ARTIFACT_CACHE[key] = (mtime, obj)
    return obj


def _num_arr(df: pd.DataFrame | None, col: str) -> np.ndarray:
    if df is None or df.empty or col not in df.columns:
        return np.array([], dtype=float)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _select_voice_item(voice_series: Any) -> Any | None:
    if voice_series is None:
        return None
    if isinstance(voice_series, dict):
        return voice_series
    items = voice_series if isinstance(voice_series, (list, tuple)) else [voice_series]
    whole = [x for x in items if x is not None and getattr(x, "index", None) is None]
    if whole:
        return whole[0]
    valid = [x for x in items if x is not None and getattr(x, "f0", None) is not None and len(getattr(x, "f0")) > 0]
    return valid[0] if valid else None


def summarize_l0_inputs(
    face_df: pd.DataFrame | None,
    gesture_df: pd.DataFrame | None = None,
    voice_series: Any = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Convert raw L0 DataFrames + optional VoiceSeries into aligned feature dict."""
    feats: dict[str, float] = {}
    n_face = len(face_df) if isinstance(face_df, pd.DataFrame) else 0
    n_gest = len(gesture_df) if isinstance(gesture_df, pd.DataFrame) else 0

    dt_sec = 1.0
    if isinstance(face_df, pd.DataFrame) and "timestamp" in face_df.columns:
        ts = _num_arr(face_df, "timestamp")
        ts_fin = ts[np.isfinite(ts)]
        if len(ts_fin) >= 2:
            diffs = np.diff(ts_fin)
            pos_diffs = diffs[diffs > 0]
            if len(pos_diffs) > 0:
                dt_sec = float(np.median(pos_diffs))

    feats["n_sampled_frames"] = float(n_face)
    feats["frame_dt_sec"] = float(dt_sec)

    valid_face_mask = np.zeros(n_face, dtype=bool)
    if n_face > 0 and isinstance(face_df, pd.DataFrame):
        au_cols = [c for c in face_df.columns if c.startswith("au") and "_" in c][:8]
        extra = [c for c in ("gaze_deviation", "head_yaw", "bs_jawOpen") if c in face_df.columns]
        chk = au_cols + extra
        if chk:
            mat = np.column_stack([_num_arr(face_df, c) for c in chk])
            valid_face_mask = np.isfinite(mat).any(axis=1)
        n_valid = int(valid_face_mask.sum())
        feats["n_valid_face_frames"] = float(n_valid)
        feats["face_valid_ratio"] = float(n_valid) / float(n_face)

        for k in KEY_GEOM_FEATURES:
            col = "gaze_deviation" if k == "gaze_dev" else k
            arr = _num_arr(face_df, col)
            if col == "gaze_deviation" and not np.isfinite(arr).any():
                gx = _num_arr(face_df, "gaze_direction_x")
                gy = _num_arr(face_df, "gaze_direction_y")
                if gx.size and gy.size and gx.size == gy.size:
                    arr = np.hypot(gx, gy)
            fin = arr[np.isfinite(arr)]
            if len(fin) > 0:
                feats[f"{k}_mean"] = float(np.mean(fin))
                feats[f"{k}_std"] = float(np.std(fin))

        ear = _num_arr(face_df, "avg_ear")
        ear_fin = ear[np.isfinite(ear)]
        if len(ear_fin) > 0:
            feats["ear_low_ratio"] = float(np.mean(ear_fin < 0.20))
            feats["ear_range"] = float(np.max(ear_fin) - np.min(ear_fin))

        mo = _num_arr(face_df, "au25_mouth_open")
        mo_fin = mo[np.isfinite(mo)]
        if len(mo_fin) > 0:
            feats["mouth_open_range"] = float(np.max(mo_fin) - np.min(mo_fin))

        hy = _num_arr(face_df, "head_yaw")
        hp = _num_arr(face_df, "head_pitch")
        hy_fin = hy[np.isfinite(hy)]
        hp_fin = hp[np.isfinite(hp)]
        if len(hy_fin) >= 2 and len(hp_fin) >= 2:
            step = 0.5 * (float(np.mean(np.abs(np.diff(hy_fin)))) + float(np.mean(np.abs(np.diff(hp_fin)))))
            feats["head_motion_step"] = step
            feats["head_motion_rate"] = step / max(dt_sec, 0.25)

        for name in BLENDSHAPE_NAMES:
            arr = _num_arr(face_df, f"bs_{name}")
            fin = arr[np.isfinite(arr)]
            if len(fin) > 0:
                feats[f"bs_{name}_mean"] = float(np.mean(fin))

        for gname, members in BILATERAL_BLENDSHAPE_GROUPS.items():
            cols_m = [_num_arr(face_df, f"bs_{m}") for m in members]
            cols_m = [a for a in cols_m if a.size == n_face and np.isfinite(a).any()]
            if cols_m:
                avg_s = np.nanmean(np.vstack(cols_m), axis=0)
                fin = avg_s[np.isfinite(avg_s)]
                if len(fin) > 0:
                    feats[f"{gname}_mean"] = float(np.mean(fin))
                    feats[f"{gname}_std"] = float(np.std(fin))
                    feats[f"{gname}_max"] = float(np.max(fin))
    else:
        feats["n_valid_face_frames"] = 0.0
        feats["face_valid_ratio"] = 0.0

    # Voice series summary
    v_item = _select_voice_item(voice_series)
    has_voice = False
    if v_item is not None:
        if isinstance(v_item, dict):
            f0 = np.asarray(v_item.get("f0", []), dtype=float)
            rms = np.asarray(v_item.get("rms", []), dtype=float)
            frame_sec = float(v_item.get("frame_sec", 0.032) or 0.032)
        else:
            f0 = np.asarray(getattr(v_item, "f0", []), dtype=float)
            rms = np.asarray(getattr(v_item, "rms", []), dtype=float)
            frame_sec = float(getattr(v_item, "frame_sec", 0.032) or 0.032)

        if len(f0) > 0:
            has_voice = True
            dur_sec = float(len(f0)) * frame_sec
            feats["duration_sec"] = dur_sec
            voiced = np.isfinite(f0) & (f0 > 0)
            feats["voiced_frame_ratio"] = float(np.mean(voiced))
            feats["speech_ratio"] = float(np.mean(voiced))
            f0_v = f0[voiced]
            if len(f0_v) >= 3:
                feats["pitch_mean"] = float(np.mean(f0_v))
                feats["pitch_std"] = float(np.std(f0_v))
                feats["pitch_p10"] = float(np.percentile(f0_v, 10))
                feats["pitch_p90"] = float(np.percentile(f0_v, 90))
                feats["pitch_iqr"] = float(np.percentile(f0_v, 75) - np.percentile(f0_v, 25))

            # Estimate inter-utterance pauses (>= 0.2s unvoiced gaps between voiced frames)
            min_gap_frames = max(1, int(round(0.2 / max(frame_sec, 1e-4))))
            pauses: list[float] = []
            in_gap = False
            seen_voice = False
            gap_len = 0
            for v_flag in voiced:
                if v_flag:
                    if in_gap and seen_voice and gap_len >= min_gap_frames:
                        pauses.append(gap_len * frame_sec)
                    seen_voice = True
                    in_gap = False
                    gap_len = 0
                else:
                    if seen_voice:
                        in_gap = True
                        gap_len += 1
            feats["pause_count"] = float(len(pauses))
            feats["pause_rate_per_sec"] = float(len(pauses)) / max(dur_sec, 1e-6)
            feats["pause_rate_per_min"] = float(len(pauses)) / max(dur_sec / 60.0, 1e-6)
            feats["pause_duration_mean"] = float(np.mean(pauses)) if pauses else 0.0
            feats["pause_duration_p50"] = float(np.median(pauses)) if pauses else 0.0

        if len(rms) > 0:
            r_fin = rms[np.isfinite(rms)]
            if len(r_fin) > 0:
                feats["rms_mean"] = float(np.mean(r_fin))
                feats["rms_std"] = float(np.std(r_fin))
                r_db = 20.0 * np.log10(np.maximum(r_fin, 1e-7))
                feats["rms_db_mean"] = float(np.mean(r_db))
                feats["rms_db_std"] = float(np.std(r_db))

    meta = {
        "n_face_frames": int(n_face),
        "n_gesture_frames": int(n_gest),
        "has_face": bool(n_face > 0 and feats.get("n_valid_face_frames", 0.0) > 0),
        "has_gesture": bool(n_gest > 0),
        "has_voice": has_voice,
    }
    return feats, meta


def _vectorize(feat_dict: dict[str, float], col_names: list[str]) -> np.ndarray:
    return np.asarray([[feat_dict.get(c, np.nan) for c in col_names]], dtype=float)


def _extract_shared_regime_windows(
    face_df: pd.DataFrame | None,
    voice_series: Any = None,
    window_size: int = 8,
    stride: int = 4,
) -> np.ndarray:
    if not isinstance(face_df, pd.DataFrame) or len(face_df) < window_size:
        return np.empty((0, len(SHARED_REGIME_FEATURES)), dtype=float)

    n = len(face_df)
    au_cols = [c for c in face_df.columns if c.startswith("au") and "_" in c][:8]
    extra = [c for c in ("gaze_deviation", "head_yaw", "bs_jawOpen") if c in face_df.columns]
    chk = au_cols + extra
    if chk:
        mat = np.column_stack([_num_arr(face_df, c) for c in chk])
        valid_mask = np.isfinite(mat).any(axis=1).astype(float)
    else:
        valid_mask = np.zeros(n, dtype=float)

    mouth = _num_arr(face_df, "au25_mouth_open")
    if mouth.size == 0 or not np.isfinite(mouth).any():
        mouth = _num_arr(face_df, "bs_jawOpen")
    if mouth.size == 0:
        mouth = np.full(n, np.nan)

    gaze = _num_arr(face_df, "gaze_deviation")
    if gaze.size == 0 or not np.isfinite(gaze).any():
        gx = _num_arr(face_df, "gaze_direction_x")
        gy = _num_arr(face_df, "gaze_direction_y")
        if gx.size == n and gy.size == n:
            gaze = np.hypot(gx, gy)
        else:
            gaze = np.full(n, np.nan)

    hy = _num_arr(face_df, "head_yaw")
    hp = _num_arr(face_df, "head_pitch")

    ts = _num_arr(face_df, "timestamp")
    ts_fin = ts[np.isfinite(ts)]
    if len(ts_fin) == n:
        t_rel = ts - float(ts_fin[0])
    else:
        t_rel = np.arange(n, dtype=float)

    W_fv = sliding_window_view(valid_mask, window_size)[::stride]
    W_mo = sliding_window_view(mouth, window_size)[::stride]
    W_gz = sliding_window_view(gaze, window_size)[::stride]
    W_t = sliding_window_view(t_rel, window_size)[::stride]
    n_win = len(W_fv)

    fv_col = W_fv.mean(axis=1)
    mo_mean = np.array([float(np.nanmean(r)) if np.isfinite(r).any() else np.nan for r in W_mo], dtype=float)
    mo_std = np.array([float(np.nanstd(r)) if np.isfinite(r).sum() >= 2 else np.nan for r in W_mo], dtype=float)
    gz_mean = np.array([float(np.nanmean(r)) if np.isfinite(r).any() else np.nan for r in W_gz], dtype=float)

    hm_col = np.full(n_win, np.nan, dtype=float)
    if hy.size == n or hp.size == n:
        W_hy = sliding_window_view(hy, window_size)[::stride] if hy.size == n else None
        W_hp = sliding_window_view(hp, window_size)[::stride] if hp.size == n else None
        for i in range(n_win):
            diffs = []
            for W_h in (W_hy, W_hp):
                if W_h is not None:
                    vals = W_h[i, np.isfinite(W_h[i])]
                    if len(vals) >= 2:
                        diffs.append(float(np.mean(np.abs(np.diff(vals)))))
            if diffs:
                hm_col[i] = float(np.mean(diffs))

    vr_col = np.full(n_win, np.nan, dtype=float)
    v_item = _select_voice_item(voice_series)
    if v_item is not None:
        f0 = np.asarray(v_item.get("f0", []) if isinstance(v_item, dict) else getattr(v_item, "f0", []), dtype=float)
        frame_sec = float(
            (v_item.get("frame_sec", 0.032) if isinstance(v_item, dict) else getattr(v_item, "frame_sec", 0.032)) or 0.032
        )
        if len(f0) > 0 and frame_sec > 0:
            v_dur = len(f0) * frame_sec
            v_cum = np.concatenate(([0], np.cumsum(np.isfinite(f0) & (f0 > 0), dtype=np.int64)))
            for i in range(n_win):
                s_s, e_s = float(W_t[i, 0]), float(W_t[i, -1])
                if e_s > s_s and 0.0 <= s_s < v_dur:
                    i0 = max(0, int(s_s / frame_sec))
                    i1 = min(len(f0), int(math.ceil(e_s / frame_sec)))
                    if i1 > i0:
                        vr_col[i] = float(v_cum[i1] - v_cum[i0]) / float(i1 - i0)

    return np.column_stack([fv_col, mo_mean, mo_std, gz_mean, hm_col, vr_col])


def predict_research_profile(
    face_df: pd.DataFrame | None,
    gesture_df: pd.DataFrame | None = None,
    voice_series: Any = None,
    transcript_text: str = "",
    *,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> dict[str, Any]:
    """Run offline multi-corpus research projections on an interview session."""
    t0 = time.perf_counter()
    feats, meta = summarize_l0_inputs(face_df, gesture_df, voice_series)
    cleaned_text = clean_transcript_text(transcript_text)
    has_text = bool(cleaned_text)
    meta["has_transcript"] = has_text

    # 1. ChaLearn Behavioral Projections
    chalearn_out: dict[str, Any] = {"available": False, "projections": {}}
    ch_art = _load_joblib_cached(models_dir / "chalearn_bigfive_behavioral.joblib")
    if ch_art is not None and (meta["has_face"] or meta["has_voice"]):
        X_ch = _vectorize(feats, ch_art["feature_names"])
        preds = np.clip(ch_art["pipeline"].predict(X_ch)[0], 0.0, 1.0)
        chalearn_out = {
            "available": True,
            "scale": "[0, 1]",
            "projections": {t: round(float(v), 4) for t, v in zip(ch_art["target_names"], preds)},
        }

    # 2. RecruitView Multimodal Projections
    rv_out: dict[str, Any] = {"available": False, "mode": "none", "projections": {}}
    rv_art = _load_joblib_cached(models_dir / "recruitview_bigfive_multimodal.joblib")
    if rv_art is not None and (meta["has_face"] or meta["has_voice"] or has_text):
        p_beh = None
        p_txt = None
        if meta["has_face"] or meta["has_voice"]:
            X_rv = _vectorize(feats, rv_art["behavior_feature_names"])
            p_beh = rv_art["behavior_pipeline"].predict(X_rv)[0]
        if has_text:
            X_tv = rv_art["text_vectorizer"].transform([cleaned_text])
            p_txt = rv_art["text_ridge"].predict(X_tv)[0]

        if p_beh is not None and p_txt is not None:
            mode = "multimodal_late_fusion"
            preds = np.array([
                b["intercept"] + b["w_beh"] * p_beh[j] + b["w_txt"] * p_txt[j]
                for j, b in enumerate(rv_art["late_fusion_blenders"])
            ], dtype=float)
        elif p_txt is not None:
            mode = "text_only"
            preds = p_txt
        else:
            mode = "behavior_only"
            preds = p_beh

        rv_out = {
            "available": True,
            "mode": mode,
            "projections": {t: round(float(v), 4) for t, v in zip(rv_art["target_names"], preds)},
        }

    # 3. MIT Interview Observable Correlates
    mit_out: dict[str, Any] = {"available": False, "mode": "none", "projections": {}}
    mit_art = _load_joblib_cached(models_dir / "mit_interview_multimodal.joblib")
    if mit_art is not None and (meta["has_face"] or meta["has_voice"] or has_text):
        p_beh = None
        p_txt = None
        av_pipe = mit_art.get("extracted_av_pipeline")
        av_cols = mit_art.get("extracted_av_feature_names") or []
        if av_pipe is not None and av_cols and (meta["has_face"] or meta["has_voice"]):
            X_av = _vectorize(feats, av_cols)
            p_beh = av_pipe.predict(X_av)[0]
        if has_text:
            X_tv = mit_art["text_vectorizer"].transform([cleaned_text])
            p_txt = mit_art["text_ridge"].predict(X_tv)[0]

        if p_beh is not None and p_txt is not None:
            mode = "multimodal_late_fusion"
            blenders = mit_art.get("extracted_av_late_fusion_blenders") or mit_art["late_fusion_blenders"]
            preds = np.array([
                b["intercept"] + b["w_beh"] * p_beh[j] + b["w_txt"] * p_txt[j]
                for j, b in enumerate(blenders)
            ], dtype=float)
        elif p_txt is not None:
            mode = "text_only"
            preds = p_txt
        else:
            mode = "extracted_av_only"
            preds = p_beh

        if preds is not None:
            all_proj = {t: round(float(v), 4) for t, v in zip(mit_art["target_names"], preds)}
            core = {t: all_proj[t] for t in mit_art.get("core_targets", []) if t in all_proj}
            mit_out = {
                "available": True,
                "mode": mode,
                "scale": "1-7 Likert",
                "core_correlates": core,
                "all_correlates": all_proj,
            }

    # 4. Cross-Corpus Behavioral Regime State
    regime_out: dict[str, Any] = {"available": False, "n_windows": 0, "state_ratios": {}}
    reg_art = _load_joblib_cached(models_dir / "behavior_regime_multicorpus.joblib")
    X_win = _extract_shared_regime_windows(face_df, voice_series=voice_series)
    if reg_art is not None and len(X_win) > 0:
        Z_win = reg_art["preprocessor"].transform(X_win)
        cids = reg_art["gmm"].predict(Z_win)
        cmap = reg_art["cluster_to_state"]
        states = [cmap[int(c)] for c in cids]
        counts = Counter(states)
        n_w = len(states)
        ratios = {st: round(counts.get(st, 0) / n_w, 4) for st in REGIME_STATE_ORDER}
        dominant = max(REGIME_STATE_ORDER, key=lambda st: ratios[st])
        transitions = sum(1 for i in range(1, n_w) if states[i] != states[i - 1])
        regime_out = {
            "available": True,
            "model_type": "Bilingual-Joint-GMM",
            "n_windows": int(n_w),
            "dominant_state": dominant,
            "transition_count": int(transitions),
            "state_counts": {st: int(counts.get(st, 0)) for st in REGIME_STATE_ORDER},
            "state_ratios": ratios,
        }

    return {
        "research_only": True,
        "disclaimer": (
            "Offline multi-corpus research projection only; not calibrated for "
            "clinical, psychological, or hiring decisions."
        ),
        "input_summary": {
            **meta,
            "n_extracted_summary_features": len(feats),
            "inference_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        },
        "chalearn_behavioral_projections": chalearn_out,
        "recruitview_multimodal_projections": rv_out,
        "mit_observable_correlates": mit_out,
        "cross_corpus_regime": regime_out,
    }


__all__ = ["predict_research_profile", "summarize_l0_inputs"]
