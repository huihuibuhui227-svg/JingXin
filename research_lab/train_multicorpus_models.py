#!/usr/bin/env python3
"""Multi-corpus research training and cross-corpus transfer benchmark suite.

Runs four reproducible benchmark experiments without modifying production
``report_frontend/`` code:
  1. MIT Interview (N=138, 5-fold GroupKFold by participant ID, plus Tesla P40
     FP32 PyTorch MLP via subprocess to /home/zgy/anaconda3/envs/funasr/bin/python)
  2. RecruitView (Track A full N=2,011 metadata transcripts; Track B N=160
     multimodal clips; 5-fold GroupKFold by user_no)
  3. ChaLearn First Impressions V2 (Official train -> val tuning -> test eval)
  4. Cross-Corpus Behavioral Regime Transfer (English Corpora <-> 47 Chinese
     sessions in data/logs/, zero PII)

Outputs:
  - models/research/*.joblib
  - research_lab/results/multicorpus_benchmark_20261124.json
  - research_lab/results/multicorpus_benchmark_20261124.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


def run_torch_mlp_worker(payload_npz: Path, out_json: Path, out_pred_npy: Path) -> None:
    """Train & evaluate a strict torch.float32 MLP per fold on CUDA (Tesla P40 sm_61)."""
    import torch
    import torch.nn as nn

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available in funasr environment for PyTorch MLP worker")

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = False

    class MultimodalMLP(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128, dtype=torch.float32),
                nn.LayerNorm(128, dtype=torch.float32),
                nn.SiLU(),
                nn.Dropout(0.35),
                nn.Linear(128, 64, dtype=torch.float32),
                nn.LayerNorm(64, dtype=torch.float32),
                nn.SiLU(),
                nn.Dropout(0.25),
                nn.Linear(64, out_dim, dtype=torch.float32),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    data = np.load(payload_npz, allow_pickle=False)
    n_folds = int(data["n_folds"][0])
    n_samples = int(data["n_samples"][0])
    n_targets = int(data["n_targets"][0])
    oof_preds = np.zeros((n_samples, n_targets), dtype=np.float32)

    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.perf_counter()

    for f_idx in range(n_folds):
        torch.manual_seed(42 + f_idx)
        np.random.seed(42 + f_idx)

        X_tr = torch.tensor(data[f"X_tr_{f_idx}"], dtype=torch.float32, device=device)
        y_tr_np = data[f"y_tr_{f_idx}"].astype(np.float32)
        X_te = torch.tensor(data[f"X_te_{f_idx}"], dtype=torch.float32, device=device)
        te_idx = data[f"te_idx_{f_idx}"]

        y_mean = y_tr_np.mean(axis=0, keepdims=True)
        y_std = np.maximum(y_tr_np.std(axis=0, keepdims=True), 1e-6)
        y_tr_z = torch.tensor((y_tr_np - y_mean) / y_std, dtype=torch.float32, device=device)

        model = MultimodalMLP(X_tr.shape[1], n_targets).to(device=device, dtype=torch.float32)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=3e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=160)
        loss_fn = nn.SmoothL1Loss(beta=1.0)

        model.train()
        for _ in range(160):
            opt.zero_grad(set_to_none=True)
            pred = model(X_tr)
            loss = loss_fn(pred, y_tr_z)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()

        model.eval()
        with torch.no_grad():
            pred_te_z = model(X_te).detach().cpu().numpy()
        oof_preds[te_idx] = pred_te_z * y_std + y_mean

    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - t0
    peak_mb = float(torch.cuda.max_memory_allocated(0)) / (1024.0 * 1024.0)

    np.save(out_pred_npy, oof_preds)
    meta = {
        "device_name": str(torch.cuda.get_device_name(0)),
        "torch_version": str(torch.__version__),
        "dtype": "torch.float32",
        "elapsed_sec": round(elapsed, 4),
        "peak_vram_mb": round(peak_mb, 2),
        "epochs_per_fold": 160,
        "n_folds": n_folds,
    }
    out_json.write_text(json.dumps(meta, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__" and "--torch-mlp-worker" in sys.argv:
    _idx = sys.argv.index("--torch-mlp-worker")
    run_torch_mlp_worker(Path(sys.argv[_idx + 1]), Path(sys.argv[_idx + 2]), Path(sys.argv[_idx + 3]))
    sys.exit(0)

import joblib
import pandas as pd
from scipy import sparse, stats
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.linear_model import Ridge, RidgeCV

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATASETS_ROOT = Path("/home/zgy/huihuibui/DATASETS")
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_MODELS_DIR = REPO_ROOT / "models" / "research"
FUNASR_PYTHON = Path("/home/zgy/anaconda3/envs/funasr/bin/python")

MIT_ALL_TARGETS: tuple[str, ...] = (
    "Overall",
    "RecommendHiring",
    "Colleague",
    "Engaged",
    "Excited",
    "EyeContact",
    "Smiled",
    "SpeakingRate",
    "NoFillers",
    "Friendly",
    "Paused",
    "EngagingTone",
    "StructuredAnswers",
    "Calm",
    "NotStressed",
    "Focused",
    "Authentic",
    "NotAwkward",
)

MIT_CORE_TARGETS: tuple[str, ...] = (
    "Overall",
    "RecommendHiring",
    "EyeContact",
    "Smiled",
    "SpeakingRate",
    "Paused",
    "StructuredAnswers",
    "Focused",
)

RECRUITVIEW_PERSONALITY_TARGETS: tuple[str, ...] = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
    "overall_personality",
)

RECRUITVIEW_INTERVIEW_TARGETS: tuple[str, ...] = (
    "interview_score",
    "answer_score",
    "speaking_skills",
    "confidence_score",
    "facial_expression",
    "overall_performance",
)

RECRUITVIEW_ALL_TARGETS: tuple[str, ...] = (
    RECRUITVIEW_PERSONALITY_TARGETS + RECRUITVIEW_INTERVIEW_TARGETS
)

CHALEARN_TARGETS: tuple[str, ...] = (
    "extraversion",
    "neuroticism",
    "agreeableness",
    "conscientiousness",
    "openness",
    "interview",
)

SHARED_REGIME_FEATURES: tuple[str, ...] = (
    "face_valid_ratio",
    "mouth_open_mean",
    "mouth_motion_std",
    "gaze_dev_mean",
    "head_motion_std",
    "voiced_ratio",
)

# Sampling/duration metadata columns excluded from predictive behavioral feature vectors
# so models only learn from scale-invariant behavioral ratios & acoustic statistics.
NON_PREDICTIVE_META_COLS: frozenset[str] = frozenset({
    "corpus",
    "sample_id",
    "group_id",
    "split",
    "vad_backend",
    "clean_text",
    "user_no",
    "n_sampled_frames",
    "n_valid_face_frames",
    "frame_dt_sec",
    "duration_sec",
    "pause_count",
    "bs__neutral_mean",
})

REGIME_STATE_ORDER: tuple[str, ...] = (
    "steady_speaking",
    "active_expressive",
    "pause_thinking",
    "low_signal",
)


# ---------------------------------------------------------------------------
# Evaluation & Modeling Helpers
# ---------------------------------------------------------------------------

def safe_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return 0.0
    a, b = a[mask], b[mask]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    r = float(stats.pearsonr(a, b)[0])
    return r if np.isfinite(r) else 0.0


def safe_spearmanr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return 0.0
    a, b = a[mask], b[mask]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    rho = float(stats.spearmanr(a, b)[0])
    return rho if np.isfinite(rho) else 0.0


def eval_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: tuple[str, ...] | list[str],
    *,
    minmax_spans: np.ndarray | None = None,
    subset_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    per_target: dict[str, dict[str, float]] = {}

    for j, name in enumerate(target_names):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        mae = float(np.mean(np.abs(yt - yp)))
        r = safe_pearsonr(yt, yp)
        rho = safe_spearmanr(yt, yp)
        entry = {
            "pearson_r": round(r, 4),
            "spearman_rho": round(rho, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
        }
        if minmax_spans is not None:
            span = max(float(minmax_spans[j]), 1e-6)
            entry["acc_1_minus_mae"] = round(float(1.0 - mae / span), 4)
        else:
            entry["acc_1_minus_mae"] = round(float(1.0 - mae), 4)
        per_target[name] = entry

    def _macro(names: list[str]) -> dict[str, float]:
        return {
            "pearson_r": round(float(np.mean([per_target[n]["pearson_r"] for n in names])), 4),
            "spearman_rho": round(float(np.mean([per_target[n]["spearman_rho"] for n in names])), 4),
            "rmse": round(float(np.mean([per_target[n]["rmse"] for n in names])), 4),
            "mae": round(float(np.mean([per_target[n]["mae"] for n in names])), 4),
            "acc_1_minus_mae": round(float(np.mean([per_target[n]["acc_1_minus_mae"] for n in names])), 4),
        }

    out: dict[str, Any] = {
        "macro": _macro(list(target_names)),
        "per_target": per_target,
    }
    if subset_names is not None:
        out["core_macro"] = _macro([n for n in subset_names if n in per_target])
    return out


def clean_transcript_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    # Strip [mm:ss - mm:ss] timing annotations so TF-IDF never leaks duration
    cleaned = re.sub(r"\[\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\]\s*", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _drop_all_nan_cols(X: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    keep = np.isfinite(X).any(axis=0)
    return X[:, keep], [n for n, k in zip(names, keep) if k]


def _fit_predict_ridge_oof(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    alphas: tuple[float, ...] = (1.0, 5.0, 20.0, 50.0, 150.0, 500.0),
) -> np.ndarray:
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros_like(y, dtype=float)
    for tr, te in gkf.split(X, y, groups=groups):
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("reg", RidgeCV(alphas=alphas)),
        ])
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict(X[te])
    return oof


# ---------------------------------------------------------------------------
# Experiment 1: MIT Interview
# ---------------------------------------------------------------------------

def _load_mit_official_and_text(datasets_root: Path) -> tuple[list[str], np.ndarray, np.ndarray, list[str], list[str], np.ndarray]:
    mit_root = datasets_root / "MIT_INTERVIEW_DATASET"
    ts_df = pd.read_csv(mit_root / "Labels" / "turker_scores_full_interview.csv")
    ts_df["key"] = ts_df["Participant"].astype(str).str.lower()
    agg_scores = ts_df.groupby("key")[list(MIT_ALL_TARGETS)].mean()

    keys = sorted(p.stem.lower() for p in (mit_root / "Videos").glob("*.avi") if p.stem.lower() in agg_scores.index)

    # Prosody (exact key match before q<idx>, fixing legacy prefix bug & dropping all-NaN loudness)
    pros_df = pd.read_csv(mit_root / "Prosody" / "prosodic_features.csv")
    pros_df["key"] = (
        pros_df["participant&question"]
        .astype(str)
        .str.lower()
        .str.extract(r"^(pp?\d+)q", expand=False)
    )
    pros_num_cols = [
        c for c in pros_df.columns
        if c not in ("participant&question", "key") and pros_df[c].notna().any()
    ]

    # Transcripts (header=None so p1 is not dropped as column header)
    tr_df = pd.read_csv(
        mit_root / "Labels" / "interview_transcripts_by_turkers.csv",
        header=None,
        names=["participant", "raw_transcript"],
    )
    tr_map: dict[str, str] = {}
    for _, row in tr_df.iterrows():
        k = str(row["participant"]).strip().lower()
        raw = str(row["raw_transcript"]) if isinstance(row["raw_transcript"], str) else ""
        turns = []
        for seg in raw.split("|"):
            seg = seg.strip()
            if seg.lower().startswith("interviewee"):
                seg = seg.split(":", 1)[-1].strip()
                if seg and "[???]" not in seg:
                    turns.append(seg)
        tr_map[k] = " ".join(turns)

    feat_dicts: list[dict[str, float]] = []
    texts: list[str] = []

    for k in keys:
        d: dict[str, float] = {}
        # 1. Facial Features
        ff_path = mit_root / "Facial_Features" / f"{k.upper()}.csv"
        if ff_path.exists():
            fdf = pd.read_csv(ff_path).dropna(axis=1, how="all")
            for col in fdf.columns:
                s = pd.to_numeric(fdf[col], errors="coerce").dropna()
                if len(s) >= 5:
                    base = re.sub(r"\W+", "_", str(col)).strip("_") or "f"
                    d[f"ff_{base}_mean"] = float(s.mean())
                    d[f"ff_{base}_std"] = float(s.std(ddof=0))
                    d[f"ff_{base}_min"] = float(s.min())
                    d[f"ff_{base}_max"] = float(s.max())

        # 2. Prosody
        p_sub = pros_df[pros_df["key"] == k]
        for col in pros_num_cols:
            s = pd.to_numeric(p_sub[col], errors="coerce").dropna()
            if len(s) >= 2:
                base = re.sub(r"\W+", "_", str(col)).strip("_")
                d[f"pros_{base}_mean"] = float(s.mean())
                d[f"pros_{base}_std"] = float(s.std(ddof=0))

        # 3. SmileData
        sub_dir = "post" if k.startswith("pp") else "pre"
        pid_num = re.search(r"\d+", k).group()
        sm_path = mit_root / "SmileData" / sub_dir / f"Smoothed-features-P{pid_num}.txt"
        if sm_path.exists():
            sdf = pd.read_csv(sm_path, sep=r"\s+", header=None, engine="python").iloc[:, :3]
            for j, nm in enumerate(("smile", "nod", "shake")):
                s = pd.to_numeric(sdf[j], errors="coerce").dropna()
                if len(s) >= 5:
                    d[f"shore_{nm}_mean"] = float(s.mean())
                    d[f"shore_{nm}_std"] = float(s.std(ddof=0))
                    d[f"shore_{nm}_max"] = float(s.max())
                    d[f"shore_{nm}_pct_pos"] = float((s > 0).mean())

        feat_dicts.append(d)
        texts.append(tr_map.get(k, ""))

    col_names = sorted({c for d in feat_dicts for c in d})
    X_off = np.full((len(keys), len(col_names)), np.nan, dtype=float)
    for i, d in enumerate(feat_dicts):
        for j, c in enumerate(col_names):
            if c in d:
                X_off[i, j] = d[c]

    Y = agg_scores.loc[keys, list(MIT_ALL_TARGETS)].to_numpy(dtype=float)
    groups = np.asarray([int(re.search(r"\d+", k).group()) for k in keys], dtype=int)
    return keys, X_off, col_names, texts, groups, Y


def run_experiment_1_mit(
    datasets_root: Path,
    cache_dir: Path,
    models_dir: Path,
) -> dict[str, Any]:
    print("\n=== Experiment 1: MIT Interview (N=138, 5-fold GroupKFold by Participant) ===", flush=True)
    keys, X_off, off_cols, texts, groups, Y = _load_mit_official_and_text(datasets_root)

    # Merge newly extracted MediaPipe + Silero + Parselmouth features if available
    mit_cache_csv = cache_dir / "mit_multimodal_features.csv"
    X_ext = np.empty((len(keys), 0), dtype=float)
    ext_cols: list[str] = []
    if mit_cache_csv.exists():
        df_ext = pd.read_csv(mit_cache_csv, dtype={"sample_id": str, "group_id": str})
        df_ext["sample_id"] = df_ext["sample_id"].astype(str).str.lower()
        df_ext = df_ext.set_index("sample_id").reindex(keys)
        ext_cols = [
            c for c in df_ext.columns
            if c not in NON_PREDICTIVE_META_COLS and not c.startswith("target_") and pd.api.types.is_numeric_dtype(df_ext[c])
        ]
        X_ext = df_ext[ext_cols].to_numpy(dtype=float)
        X_ext, ext_cols = _drop_all_nan_cols(X_ext, ext_cols)

    # Combined behavior = Official (Prosody + Facial + SmileData) + Extracted AV
    if X_ext.shape[1] > 0:
        X_beh = np.hstack([X_off, X_ext])
        beh_cols = off_cols + [f"mp_{c}" for c in ext_cols]
    else:
        X_beh = X_off
        beh_cols = off_cols

    n_samples, n_targets = Y.shape
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(X_beh, Y, groups=groups))

    oof_preds: dict[str, np.ndarray] = {
        "Behavior-Ridge": np.zeros_like(Y),
        "Behavior-RF": np.zeros_like(Y),
        "Behavior-HistGBT": np.zeros_like(Y),
        "Text-TFIDF-Ridge": np.zeros_like(Y),
        "Early-Fusion-Ridge": np.zeros_like(Y),
        "Late-Fusion": np.zeros_like(Y),
    }
    if X_ext.shape[1] > 0:
        oof_preds["Extracted-AV-Ridge"] = np.zeros_like(Y)

    # Prepare fold arrays for GPU PyTorch FP32 MLP (strictly fit preprocessing inside each train fold)
    mlp_payload: dict[str, Any] = {
        "n_folds": np.array([len(folds)], dtype=np.int32),
        "n_samples": np.array([n_samples], dtype=np.int32),
        "n_targets": np.array([n_targets], dtype=np.int32),
    }

    for f_idx, (tr, te) in enumerate(folds):
        # 1. Behavior models (clip extreme z-scores on p > n MIT tables)
        beh_imp = SimpleImputer(strategy="median")
        beh_sc = StandardScaler()
        Xb_tr = np.clip(beh_sc.fit_transform(beh_imp.fit_transform(X_beh[tr])), -5.0, 5.0)
        Xb_te = np.clip(beh_sc.transform(beh_imp.transform(X_beh[te])), -5.0, 5.0)

        ridge_b = RidgeCV(alphas=np.logspace(1, 5, 16)).fit(Xb_tr, Y[tr])
        oof_preds["Behavior-Ridge"][te] = ridge_b.predict(Xb_te)

        rf_b = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, max_features="sqrt", n_jobs=-1, random_state=42 + f_idx
        ).fit(Xb_tr, Y[tr])
        oof_preds["Behavior-RF"][te] = rf_b.predict(Xb_te)

        hgbt_b = MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=150, max_depth=3, min_samples_leaf=8, learning_rate=0.05, l2_regularization=1.0, random_state=42 + f_idx
            )
        ).fit(Xb_tr, Y[tr])
        oof_preds["Behavior-HistGBT"][te] = hgbt_b.predict(Xb_te)

        if X_ext.shape[1] > 0:
            imp_e = SimpleImputer(strategy="median")
            sc_e = StandardScaler()
            Xe_tr = np.clip(sc_e.fit_transform(imp_e.fit_transform(X_ext[tr])), -5.0, 5.0)
            Xe_te = np.clip(sc_e.transform(imp_e.transform(X_ext[te])), -5.0, 5.0)
            oof_preds["Extracted-AV-Ridge"][te] = RidgeCV(alphas=np.logspace(0, 4, 15)).fit(Xe_tr, Y[tr]).predict(Xe_te)

        # 2. Text TF-IDF (fit strictly on train fold)
        vec = TfidfVectorizer(max_features=1200, stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=2)
        Xt_tr = vec.fit_transform([texts[i] for i in tr])
        Xt_te = vec.transform([texts[i] for i in te])
        ridge_t = RidgeCV(alphas=np.logspace(-1, 3, 12)).fit(Xt_tr, Y[tr])
        oof_preds["Text-TFIDF-Ridge"][te] = ridge_t.predict(Xt_te)

        # 3. Early Fusion (Behavior + Text SVD)
        n_svd = min(32, max(2, Xt_tr.shape[1] - 1), len(tr) - 2)
        svd = TruncatedSVD(n_components=n_svd, random_state=42)
        svd_sc = StandardScaler()
        Xts_tr = np.clip(svd_sc.fit_transform(svd.fit_transform(Xt_tr)), -5.0, 5.0)
        Xts_te = np.clip(svd_sc.transform(svd.transform(Xt_te)), -5.0, 5.0)

        Xf_tr = np.hstack([Xb_tr, Xts_tr])
        Xf_te = np.hstack([Xb_te, Xts_te])
        oof_preds["Early-Fusion-Ridge"][te] = RidgeCV(alphas=np.logspace(1, 5, 16)).fit(Xf_tr, Y[tr]).predict(Xf_te)

        # Pack PCA-regularized fold payload for Tesla P40 FP32 MLP (zero leakage)
        n_pca = min(48, len(tr) - 2, Xb_tr.shape[1])
        pca_b = PCA(n_components=n_pca, random_state=42)
        pca_sc = StandardScaler()
        Xp_tr = pca_sc.fit_transform(pca_b.fit_transform(Xb_tr))
        Xp_te = pca_sc.transform(pca_b.transform(Xb_te))
        Xmlp_tr = np.hstack([Xp_tr, Xts_tr[:, : min(24, Xts_tr.shape[1])]])
        Xmlp_te = np.hstack([Xp_te, Xts_te[:, : min(24, Xts_te.shape[1])]])
        mlp_payload[f"X_tr_{f_idx}"] = Xmlp_tr.astype(np.float32)
        mlp_payload[f"y_tr_{f_idx}"] = Y[tr].astype(np.float32)
        mlp_payload[f"X_te_{f_idx}"] = Xmlp_te.astype(np.float32)
        mlp_payload[f"te_idx_{f_idx}"] = te.astype(np.int64)

        # 4. Zero-leakage Late Fusion using inner GroupKFold OOF inside train fold
        inner_gkf = GroupKFold(n_splits=min(4, len(np.unique(groups[tr]))))
        in_rf = np.zeros_like(Y[tr])
        in_hg = np.zeros_like(Y[tr])
        in_txt = np.zeros_like(Y[tr])
        for itr, ite in inner_gkf.split(Xb_tr, Y[tr], groups=groups[tr]):
            in_rf[ite] = RandomForestRegressor(
                n_estimators=180, min_samples_leaf=3, max_features="sqrt", n_jobs=-1, random_state=42
            ).fit(Xb_tr[itr], Y[tr][itr]).predict(Xb_tr[ite])
            in_hg[ite] = MultiOutputRegressor(
                HistGradientBoostingRegressor(max_iter=120, max_depth=3, min_samples_leaf=8, learning_rate=0.05, random_state=42)
            ).fit(Xb_tr[itr], Y[tr][itr]).predict(Xb_tr[ite])
            ivec = TfidfVectorizer(max_features=1200, stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=2)
            iXtr = ivec.fit_transform([texts[tr[i]] for i in itr])
            iXte = ivec.transform([texts[tr[i]] for i in ite])
            in_txt[ite] = RidgeCV(alphas=np.logspace(-1, 3, 10)).fit(iXtr, Y[tr][itr]).predict(iXte)

        lf_pred = np.zeros_like(Y[te])
        for d in range(n_targets):
            S_in = np.column_stack([in_rf[:, d], in_hg[:, d], in_txt[:, d]])
            S_te = np.column_stack([
                oof_preds["Behavior-RF"][te, d],
                oof_preds["Behavior-HistGBT"][te, d],
                oof_preds["Text-TFIDF-Ridge"][te, d],
            ])
            blender = Ridge(alpha=15.0, positive=True).fit(S_in, Y[tr, d])
            lf_pred[:, d] = blender.predict(S_te)
        oof_preds["Late-Fusion"][te] = lf_pred

    # Run Tesla P40 GPU FP32 PyTorch MLP subprocess
    gpu_mlp_meta: dict[str, Any] = {"available": False}
    if FUNASR_PYTHON.exists():
        with tempfile.TemporaryDirectory() as tmpd:
            tmp_path = Path(tmpd)
            npz_in = tmp_path / "mit_mlp_folds.npz"
            meta_out = tmp_path / "mit_mlp_meta.json"
            pred_out = tmp_path / "mit_mlp_oof.npy"
            np.savez_compressed(npz_in, **mlp_payload)
            cmd = [
                str(FUNASR_PYTHON),
                str(Path(__file__).resolve()),
                "--torch-mlp-worker",
                str(npz_in),
                str(meta_out),
                str(pred_out),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode == 0 and pred_out.exists() and meta_out.exists():
                oof_preds["GPU-FP32-MLP"] = np.load(pred_out).astype(float)
                gpu_mlp_meta = json.loads(meta_out.read_text(encoding="utf-8"))
                gpu_mlp_meta["available"] = True
            else:
                gpu_mlp_meta["error"] = (proc.stderr or proc.stdout)[-400:]

    # Evaluate all models
    model_results: dict[str, Any] = {}
    for m_name, preds in oof_preds.items():
        model_results[m_name] = eval_regression(Y, preds, MIT_ALL_TARGETS, subset_names=MIT_CORE_TARGETS)
        mr = model_results[m_name]["macro"]["pearson_r"]
        cmr = model_results[m_name]["core_macro"]["pearson_r"]
        rmse = model_results[m_name]["macro"]["rmse"]
        print(f"  [MIT] {m_name:20s} macro_r={mr:.4f} core_r={cmr:.4f} macro_rmse={rmse:.4f}", flush=True)

    best_name = max(model_results, key=lambda k: model_results[k]["macro"]["pearson_r"])

    # Fit full-dataset artifact for offline research_inference (saving both behavior & text heads + extracted-AV head)
    full_beh_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("reg", RidgeCV(alphas=np.logspace(-1, 4, 15))),
    ]).fit(X_beh, Y)

    full_txt_vec = TfidfVectorizer(max_features=1200, stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=2)
    X_txt_full = full_txt_vec.fit_transform(texts)
    full_txt_ridge = RidgeCV(alphas=np.logspace(-1, 3, 12)).fit(X_txt_full, Y)

    ext_av_pipe = None
    ext_av_blenders: list[dict[str, float]] = []
    if X_ext.shape[1] > 0:
        ext_av_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("reg", RidgeCV(alphas=np.logspace(0, 4, 15))),
        ]).fit(X_ext, Y)
        for d in range(n_targets):
            S_e = np.column_stack([oof_preds["Extracted-AV-Ridge"][:, d], oof_preds["Text-TFIDF-Ridge"][:, d]])
            be = Ridge(alpha=15.0, positive=True).fit(S_e, Y[:, d])
            ext_av_blenders.append({"intercept": float(be.intercept_), "w_beh": float(be.coef_[0]), "w_txt": float(be.coef_[1])})

    # Learn global late-fusion blend weights from OOF predictions
    blend_models: list[dict[str, float]] = []
    for d in range(n_targets):
        S_oof = np.column_stack([oof_preds["Behavior-RF"][:, d], oof_preds["Text-TFIDF-Ridge"][:, d]])
        b = Ridge(alpha=15.0, positive=True).fit(S_oof, Y[:, d])
        blend_models.append({"intercept": float(b.intercept_), "w_beh": float(b.coef_[0]), "w_txt": float(b.coef_[1])})

    models_dir.mkdir(parents=True, exist_ok=True)
    mit_artifact = {
        "corpus": "mit_interview",
        "target_names": list(MIT_ALL_TARGETS),
        "core_targets": list(MIT_CORE_TARGETS),
        "behavior_feature_names": beh_cols,
        "extracted_av_feature_names": ext_cols,
        "behavior_pipeline": full_beh_pipe,
        "extracted_av_pipeline": ext_av_pipe,
        "extracted_av_late_fusion_blenders": ext_av_blenders,
        "text_vectorizer": full_txt_vec,
        "text_ridge": full_txt_ridge,
        "late_fusion_blenders": blend_models,
        "best_cv_model": best_name,
    }
    mit_model_path = models_dir / "mit_interview_multimodal.joblib"
    joblib.dump(mit_artifact, mit_model_path, compress=3)

    return {
        "dataset": "MIT_INTERVIEW_DATASET",
        "n_interviews": int(n_samples),
        "n_participants": int(len(np.unique(groups))),
        "cv_scheme": "GroupKFold(n_splits=5, group=participant_id)",
        "n_official_behavior_features": int(X_off.shape[1]),
        "n_extracted_av_features": int(X_ext.shape[1]),
        "core_targets": list(MIT_CORE_TARGETS),
        "all_targets": list(MIT_ALL_TARGETS),
        "best_model": best_name,
        "gpu_mlp_hardware": gpu_mlp_meta,
        "models": model_results,
        "saved_model_path": str(mit_model_path),
    }


# ---------------------------------------------------------------------------
# Experiment 2: RecruitView
# ---------------------------------------------------------------------------

def run_experiment_2_recruitview(
    datasets_root: Path,
    cache_dir: Path,
    models_dir: Path,
) -> dict[str, Any]:
    print("\n=== Experiment 2: RecruitView (Track A N=2011 Text + Track B N=160 Multimodal) ===", flush=True)
    rv_root = datasets_root / "RecruitView"
    raw_rows = [json.loads(line) for line in open(rv_root / "metadata.jsonl", encoding="utf-8")]
    df_meta = pd.DataFrame(raw_rows)
    df_meta["id"] = df_meta["id"].astype(str)
    df_meta["user_no"] = df_meta["user_no"].astype(str)
    df_meta["clean_text"] = df_meta["transcript"].map(clean_transcript_text)

    Y_all = df_meta[list(RECRUITVIEW_ALL_TARGETS)].to_numpy(dtype=float)
    minmax_spans = np.maximum(Y_all.max(axis=0) - Y_all.min(axis=0), 1e-6)
    groups_all = df_meta["user_no"].to_numpy()
    texts_all = df_meta["clean_text"].tolist()

    # --- Track A: Full N=2,011 metadata transcripts ---
    gkf_a = GroupKFold(n_splits=5)
    oof_a_ridge = np.zeros_like(Y_all)
    oof_a_hgbt = np.zeros_like(Y_all)

    for f_idx, (tr, te) in enumerate(gkf_a.split(Y_all, Y_all, groups=groups_all)):
        vec = TfidfVectorizer(max_features=3000, stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=2)
        Xtr_t = vec.fit_transform([texts_all[i] for i in tr])
        Xte_t = vec.transform([texts_all[i] for i in te])
        oof_a_ridge[te] = RidgeCV(alphas=[0.5, 1.0, 5.0, 15.0, 50.0]).fit(Xtr_t, Y_all[tr]).predict(Xte_t)

        svd = TruncatedSVD(n_components=128, random_state=42)
        Ztr = svd.fit_transform(Xtr_t)
        Zte = svd.transform(Xte_t)
        hgbt = MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=150, max_depth=4, min_samples_leaf=15, learning_rate=0.05, random_state=42 + f_idx)
        ).fit(Ztr, Y_all[tr])
        oof_a_hgbt[te] = hgbt.predict(Zte)

    track_a_results = {
        "Text-TFIDF-Ridge": eval_regression(
            Y_all, oof_a_ridge, RECRUITVIEW_ALL_TARGETS, minmax_spans=minmax_spans, subset_names=RECRUITVIEW_PERSONALITY_TARGETS
        ),
        "Text-SVD-HistGBT": eval_regression(
            Y_all, oof_a_hgbt, RECRUITVIEW_ALL_TARGETS, minmax_spans=minmax_spans, subset_names=RECRUITVIEW_PERSONALITY_TARGETS
        ),
    }
    for nm, res in track_a_results.items():
        print(f"  [RV Track A N={len(df_meta)}] {nm:20s} macro_r={res['macro']['pearson_r']:.4f} ocean_r={res['core_macro']['pearson_r']:.4f} rmse={res['macro']['rmse']:.4f}", flush=True)

    # --- Track B: Extracted Multimodal Clips (N=160) ---
    rv_csv = cache_dir / "recruitview_multimodal_features.csv"
    df_feat = pd.read_csv(rv_csv, dtype={"sample_id": str, "group_id": str}).copy()
    df_feat["sample_id"] = df_feat["sample_id"].astype(str)
    meta_indexed = df_meta.set_index("id")
    clean_texts_b = meta_indexed.reindex(df_feat["sample_id"])["clean_text"].fillna("").tolist()
    users_b = meta_indexed.reindex(df_feat["sample_id"])["user_no"].fillna(df_feat["group_id"]).astype(str).to_numpy()

    beh_cols = [
        c for c in df_feat.columns
        if c not in NON_PREDICTIVE_META_COLS and not c.startswith("target_") and pd.api.types.is_numeric_dtype(df_feat[c])
    ]
    X_b = df_feat[beh_cols].to_numpy(dtype=float)
    X_b, beh_cols = _drop_all_nan_cols(X_b, beh_cols)
    Y_b = df_feat[[f"target_{t}" for t in RECRUITVIEW_ALL_TARGETS]].to_numpy(dtype=float)
    groups_b = users_b
    texts_b = clean_texts_b

    gkf_b = GroupKFold(n_splits=5)
    folds_b = list(gkf_b.split(X_b, Y_b, groups=groups_b))
    oof_b: dict[str, np.ndarray] = {
        "Behavior-Ridge": np.zeros_like(Y_b),
        "Behavior-RF": np.zeros_like(Y_b),
        "Behavior-HistGBT": np.zeros_like(Y_b),
        "Text-TFIDF-Ridge": np.zeros_like(Y_b),
        "Early-Fusion-Ridge": np.zeros_like(Y_b),
        "Late-Fusion": np.zeros_like(Y_b),
    }

    for f_idx, (tr, te) in enumerate(folds_b):
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        Xb_tr = np.clip(sc.fit_transform(imp.fit_transform(X_b[tr])), -5.0, 5.0)
        Xb_te = np.clip(sc.transform(imp.transform(X_b[te])), -5.0, 5.0)

        oof_b["Behavior-Ridge"][te] = RidgeCV(alphas=np.logspace(1, 5, 16)).fit(Xb_tr, Y_b[tr]).predict(Xb_te)
        oof_b["Behavior-RF"][te] = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, max_features="sqrt", n_jobs=-1, random_state=42 + f_idx
        ).fit(Xb_tr, Y_b[tr]).predict(Xb_te)
        oof_b["Behavior-HistGBT"][te] = MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=150, max_depth=3, min_samples_leaf=8, learning_rate=0.05, random_state=42 + f_idx)
        ).fit(Xb_tr, Y_b[tr]).predict(Xb_te)

        vec = TfidfVectorizer(max_features=800, stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=3)
        Xt_tr = vec.fit_transform([texts_b[i] for i in tr])
        Xt_te = vec.transform([texts_b[i] for i in te])
        oof_b["Text-TFIDF-Ridge"][te] = RidgeCV(alphas=np.logspace(0, 3, 15)).fit(Xt_tr, Y_b[tr]).predict(Xt_te)

        n_svd = min(32, max(2, Xt_tr.shape[1] - 1), len(tr) - 2)
        svd = TruncatedSVD(n_components=n_svd, random_state=42)
        svd_sc = StandardScaler()
        Xts_tr = np.clip(svd_sc.fit_transform(svd.fit_transform(Xt_tr)), -5.0, 5.0)
        Xts_te = np.clip(svd_sc.transform(svd.transform(Xt_te)), -5.0, 5.0)
        Xf_tr = np.hstack([Xb_tr, Xts_tr])
        Xf_te = np.hstack([Xb_te, Xts_te])
        oof_b["Early-Fusion-Ridge"][te] = RidgeCV(alphas=np.logspace(1, 5, 16)).fit(Xf_tr, Y_b[tr]).predict(Xf_te)

        # Zero-leakage inner GroupKFold stacking for Late-Fusion (non-negative convex blend)
        inner_gkf = GroupKFold(n_splits=min(4, len(np.unique(groups_b[tr]))))
        in_rf = np.zeros_like(Y_b[tr])
        in_hgbt = np.zeros_like(Y_b[tr])
        in_txt = np.zeros_like(Y_b[tr])
        for itr, ite in inner_gkf.split(Xb_tr, Y_b[tr], groups=groups_b[tr]):
            in_rf[ite] = RandomForestRegressor(
                n_estimators=180, min_samples_leaf=3, max_features="sqrt", n_jobs=-1, random_state=42
            ).fit(Xb_tr[itr], Y_b[tr][itr]).predict(Xb_tr[ite])
            in_hgbt[ite] = MultiOutputRegressor(
                HistGradientBoostingRegressor(max_iter=120, max_depth=3, min_samples_leaf=8, learning_rate=0.05, random_state=42)
            ).fit(Xb_tr[itr], Y_b[tr][itr]).predict(Xb_tr[ite])
            ivec = TfidfVectorizer(max_features=800, stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=3)
            iXtr = ivec.fit_transform([texts_b[tr[i]] for i in itr])
            iXte = ivec.transform([texts_b[tr[i]] for i in ite])
            in_txt[ite] = RidgeCV(alphas=np.logspace(0, 3, 12)).fit(iXtr, Y_b[tr][itr]).predict(iXte)

        for d in range(Y_b.shape[1]):
            S_in = np.column_stack([in_rf[:, d], in_hgbt[:, d], in_txt[:, d]])
            S_te = np.column_stack([
                oof_b["Behavior-RF"][te, d],
                oof_b["Behavior-HistGBT"][te, d],
                oof_b["Text-TFIDF-Ridge"][te, d],
            ])
            b = Ridge(alpha=20.0, positive=True).fit(S_in, Y_b[tr, d])
            oof_b["Late-Fusion"][te, d] = b.predict(S_te)

    track_b_results: dict[str, Any] = {}
    for nm, preds in oof_b.items():
        track_b_results[nm] = eval_regression(
            Y_b, preds, RECRUITVIEW_ALL_TARGETS, minmax_spans=minmax_spans, subset_names=RECRUITVIEW_PERSONALITY_TARGETS
        )
        mr = track_b_results[nm]["macro"]["pearson_r"]
        oc = track_b_results[nm]["core_macro"]["pearson_r"]
        acc = track_b_results[nm]["macro"]["acc_1_minus_mae"]
        rmse = track_b_results[nm]["macro"]["rmse"]
        print(f"  [RV Track B N={len(df_feat)}] {nm:20s} macro_r={mr:.4f} ocean_r={oc:.4f} acc(1-MAE_norm)={acc:.4f} rmse={rmse:.4f}", flush=True)

    best_b_name = max(track_b_results, key=lambda k: track_b_results[k]["macro"]["pearson_r"])

    # Fit deployable research artifact (full Track A text model + Track B RF behavior model + positive blend weights)
    full_track_a_vec = TfidfVectorizer(max_features=3000, stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=2)
    X_ta_full = full_track_a_vec.fit_transform(texts_all)
    full_track_a_ridge = RidgeCV(alphas=[0.5, 1.0, 5.0, 15.0, 50.0]).fit(X_ta_full, Y_all)

    full_beh_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("reg", RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features="sqrt", n_jobs=-1, random_state=42)),
    ]).fit(X_b, Y_b)

    blend_models: list[dict[str, float]] = []
    for d in range(Y_b.shape[1]):
        S_oof = np.column_stack([oof_b["Behavior-RF"][:, d], oof_b["Text-TFIDF-Ridge"][:, d]])
        b = Ridge(alpha=15.0, positive=True).fit(S_oof, Y_b[:, d])
        blend_models.append({"intercept": float(b.intercept_), "w_beh": float(b.coef_[0]), "w_txt": float(b.coef_[1])})

    models_dir.mkdir(parents=True, exist_ok=True)
    rv_model_path = models_dir / "recruitview_bigfive_multimodal.joblib"
    rv_artifact = {
        "corpus": "recruitview",
        "target_names": list(RECRUITVIEW_ALL_TARGETS),
        "personality_targets": list(RECRUITVIEW_PERSONALITY_TARGETS),
        "interview_targets": list(RECRUITVIEW_INTERVIEW_TARGETS),
        "behavior_feature_names": beh_cols,
        "behavior_pipeline": full_beh_pipe,
        "text_vectorizer": full_track_a_vec,
        "text_ridge": full_track_a_ridge,
        "late_fusion_blenders": blend_models,
        "best_track_b_model": best_b_name,
    }
    joblib.dump(rv_artifact, rv_model_path, compress=3)

    return {
        "dataset": "RecruitView",
        "cv_scheme": "GroupKFold(n_splits=5, group=user_no)",
        "targets_personality": list(RECRUITVIEW_PERSONALITY_TARGETS),
        "targets_interview": list(RECRUITVIEW_INTERVIEW_TARGETS),
        "track_a_full_metadata_text": {
            "n_clips": int(len(df_meta)),
            "n_users": int(df_meta["user_no"].nunique()),
            "models": track_a_results,
        },
        "track_b_multimodal": {
            "n_clips": int(len(df_feat)),
            "n_users": int(len(np.unique(groups_b))),
            "n_behavior_features": int(len(beh_cols)),
            "best_model": best_b_name,
            "models": track_b_results,
        },
        "saved_model_path": str(rv_model_path),
    }


# ---------------------------------------------------------------------------
# Experiment 3: ChaLearn First Impressions V2 (Official train / val / test)
# ---------------------------------------------------------------------------

def _split_chalearn_modalities(feat_cols: list[str]) -> tuple[list[str], list[str]]:
    acoustic_prefixes = (
        "duration_sec", "speech_ratio", "pause_", "rms_", "pitch_", "voiced_frame_ratio", "hnr_db", "jitter_", "shimmer_",
    )
    aud_cols = [c for c in feat_cols if any(c.startswith(p) or c == p for p in acoustic_prefixes)]
    vis_cols = [c for c in feat_cols if c not in set(aud_cols)]
    return vis_cols, aud_cols


def run_experiment_3_chalearn(
    cache_dir: Path,
    models_dir: Path,
) -> dict[str, Any]:
    print("\n=== Experiment 3: ChaLearn First Impressions V2 (Official train -> val tune -> test eval) ===", flush=True)
    ch_csv = cache_dir / "chalearn_multimodal_features.csv"
    df = pd.read_csv(ch_csv, dtype={"sample_id": str, "group_id": str})

    raw_feat_cols = [
        c for c in df.columns
        if c not in NON_PREDICTIVE_META_COLS and not c.startswith("target_") and pd.api.types.is_numeric_dtype(df[c])
    ]

    tr_mask = df["split"] == "train"
    va_mask = df["split"] == "val"
    te_mask = df["split"] == "test"

    X_raw = df[raw_feat_cols].to_numpy(dtype=float)
    # Only drop columns that are completely NaN in train
    keep_tr = np.isfinite(X_raw[tr_mask]).any(axis=0)
    feat_cols = [c for c, k in zip(raw_feat_cols, keep_tr) if k]
    vis_cols, aud_cols = _split_chalearn_modalities(feat_cols)

    target_cols = [f"target_{t}" for t in CHALEARN_TARGETS]
    Y_tr = df.loc[tr_mask, target_cols].to_numpy(dtype=float)
    Y_va = df.loc[va_mask, target_cols].to_numpy(dtype=float)
    Y_te = df.loc[te_mask, target_cols].to_numpy(dtype=float)

    def _fit_and_eval_ridge(cols: list[str], tag: str) -> tuple[dict[str, Any], float, Any]:
        Xtr = df.loc[tr_mask, cols].to_numpy(dtype=float)
        Xva = df.loc[va_mask, cols].to_numpy(dtype=float)
        Xte = df.loc[te_mask, cols].to_numpy(dtype=float)

        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(imp.fit_transform(Xtr))
        Xva_s = sc.transform(imp.transform(Xva))
        Xte_s = sc.transform(imp.transform(Xte))

        best_alpha = 10.0
        best_val_score = -1e9
        for alpha in (0.5, 1.0, 5.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0):
            reg = Ridge(alpha=alpha).fit(Xtr_s, Y_tr)
            p_va = reg.predict(Xva_s)
            val_acc = float(1.0 - np.mean(np.abs(Y_va - p_va)))
            val_r = float(np.mean([safe_pearsonr(Y_va[:, d], p_va[:, d]) for d in range(Y_va.shape[1])]))
            # Joint selection objective: ChaLearn accuracy + correlation
            score = val_acc + 0.25 * val_r
            if score > best_val_score:
                best_val_score = score
                best_alpha = alpha

        reg_final = Ridge(alpha=best_alpha).fit(Xtr_s, Y_tr)
        p_te = np.clip(reg_final.predict(Xte_s), 0.0, 1.0)
        res = eval_regression(Y_te, p_te, CHALEARN_TARGETS)
        res["tuned_hyperparams"] = {"alpha": float(best_alpha)}
        return res, float(best_alpha), Pipeline([("imp", imp), ("sc", sc), ("reg", reg_final)])

    av_ridge_res, best_alpha, av_ridge_pipe = _fit_and_eval_ridge(feat_cols, "AudioVisual-Ridge")
    vis_ridge_res, _, _ = _fit_and_eval_ridge(vis_cols, "Visual-only-Ridge")
    aud_ridge_res, _, _ = _fit_and_eval_ridge(aud_cols, "Acoustic-only-Ridge")

    # RandomForest & HistGBT tuned on val
    Xtr_all = df.loc[tr_mask, feat_cols].to_numpy(dtype=float)
    Xva_all = df.loc[va_mask, feat_cols].to_numpy(dtype=float)
    Xte_all = df.loc[te_mask, feat_cols].to_numpy(dtype=float)
    imp_all = SimpleImputer(strategy="median").fit(Xtr_all)
    Xtr_i = imp_all.transform(Xtr_all)
    Xva_i = imp_all.transform(Xva_all)
    Xte_i = imp_all.transform(Xte_all)

    best_rf_cfg = (3, "sqrt")
    best_rf_score = -1e9
    for msl in (2, 4, 8):
        for mf in ("sqrt", 0.3):
            rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=msl, max_features=mf, n_jobs=-1, random_state=42).fit(Xtr_i, Y_tr)
            p_va = rf.predict(Xva_i)
            score = float(1.0 - np.mean(np.abs(Y_va - p_va))) + 0.25 * float(np.mean([safe_pearsonr(Y_va[:, d], p_va[:, d]) for d in range(Y_va.shape[1])]))
            if score > best_rf_score:
                best_rf_score = score
                best_rf_cfg = (msl, mf)
    rf_final = RandomForestRegressor(
        n_estimators=400, min_samples_leaf=best_rf_cfg[0], max_features=best_rf_cfg[1], n_jobs=-1, random_state=42
    ).fit(Xtr_i, Y_tr)
    rf_res = eval_regression(Y_te, np.clip(rf_final.predict(Xte_i), 0.0, 1.0), CHALEARN_TARGETS)
    rf_res["tuned_hyperparams"] = {"min_samples_leaf": best_rf_cfg[0], "max_features": best_rf_cfg[1]}

    best_hg_cfg = (3, 0.05, 1.0)
    best_hg_score = -1e9
    for md in (2, 3, 4):
        for lr in (0.03, 0.06):
            for l2 in (0.5, 2.0):
                hg = MultiOutputRegressor(
                    HistGradientBoostingRegressor(max_iter=150, max_depth=md, min_samples_leaf=10, learning_rate=lr, l2_regularization=l2, random_state=42)
                ).fit(Xtr_i, Y_tr)
                p_va = hg.predict(Xva_i)
                score = float(1.0 - np.mean(np.abs(Y_va - p_va))) + 0.25 * float(np.mean([safe_pearsonr(Y_va[:, d], p_va[:, d]) for d in range(Y_va.shape[1])]))
                if score > best_hg_score:
                    best_hg_score = score
                    best_hg_cfg = (md, lr, l2)
    hg_final = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_iter=150, max_depth=best_hg_cfg[0], min_samples_leaf=10, learning_rate=best_hg_cfg[1], l2_regularization=best_hg_cfg[2], random_state=42
        )
    ).fit(Xtr_i, Y_tr)
    hg_res = eval_regression(Y_te, np.clip(hg_final.predict(Xte_i), 0.0, 1.0), CHALEARN_TARGETS)
    hg_res["tuned_hyperparams"] = {"max_depth": best_hg_cfg[0], "learning_rate": best_hg_cfg[1], "l2_regularization": best_hg_cfg[2]}

    models_out = {
        "AudioVisual-Ridge": av_ridge_res,
        "AudioVisual-RF": rf_res,
        "AudioVisual-HistGBT": hg_res,
        "Visual-only-Ridge": vis_ridge_res,
        "Acoustic-only-Ridge": aud_ridge_res,
    }
    for nm, r in models_out.items():
        print(
            f"  [ChaLearn test N={int(te_mask.sum())}] {nm:20s} 1-MAE={r['macro']['acc_1_minus_mae']:.4f} "
            f"macro_r={r['macro']['pearson_r']:.4f} rmse={r['macro']['rmse']:.4f}",
            flush=True,
        )

    best_name = max(models_out, key=lambda k: models_out[k]["macro"]["acc_1_minus_mae"] + 0.25 * models_out[k]["macro"]["pearson_r"])

    # Refit best tuned model on train+val ONLY AFTER test evaluation is complete
    tv_mask = tr_mask | va_mask
    X_tv = df.loc[tv_mask, feat_cols].to_numpy(dtype=float)
    Y_tv = df.loc[tv_mask, target_cols].to_numpy(dtype=float)
    if best_name == "AudioVisual-RF":
        saved_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("reg", RandomForestRegressor(
                n_estimators=400, min_samples_leaf=best_rf_cfg[0], max_features=best_rf_cfg[1], n_jobs=-1, random_state=42
            )),
        ]).fit(X_tv, Y_tv)
    elif best_name == "AudioVisual-HistGBT":
        saved_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("reg", MultiOutputRegressor(
                HistGradientBoostingRegressor(
                    max_iter=150, max_depth=best_hg_cfg[0], min_samples_leaf=10, learning_rate=best_hg_cfg[1], l2_regularization=best_hg_cfg[2], random_state=42
                )
            )),
        ]).fit(X_tv, Y_tv)
    else:
        saved_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("reg", Ridge(alpha=best_alpha)),
        ]).fit(X_tv, Y_tv)

    models_dir.mkdir(parents=True, exist_ok=True)
    ch_model_path = models_dir / "chalearn_bigfive_behavioral.joblib"
    joblib.dump(
        {
            "corpus": "chalearn_first_impressions_v2",
            "target_names": list(CHALEARN_TARGETS),
            "feature_names": feat_cols,
            "visual_feature_names": vis_cols,
            "acoustic_feature_names": aud_cols,
            "pipeline": saved_pipe,
            "best_test_model": best_name,
            "saved_model_fit_split": "train+val",
        },
        ch_model_path,
        compress=3,
    )

    return {
        "dataset": "ChaLearn_First_Impressions_V2",
        "split_counts": {
            "train": int(tr_mask.sum()),
            "val": int(va_mask.sum()),
            "test": int(te_mask.sum()),
        },
        "n_visual_features": int(len(vis_cols)),
        "n_acoustic_features": int(len(aud_cols)),
        "n_total_features": int(len(feat_cols)),
        "targets": list(CHALEARN_TARGETS),
        "best_model": best_name,
        "models": models_out,
        "saved_model_path": str(ch_model_path),
    }


# ---------------------------------------------------------------------------
# Experiment 4: Cross-Corpus Behavioral Regime Transfer
# ---------------------------------------------------------------------------

def assign_audiovisual_cluster_states(
    centroids_z: np.ndarray,
    feature_names: tuple[str, ...] = SHARED_REGIME_FEATURES,
) -> dict[int, str]:
    """Deterministic physical mapping of K=4 clusters to behavioral rhythm states."""
    k = int(centroids_z.shape[0])
    idx = {n: i for i, n in enumerate(feature_names)}
    remaining = set(range(k))
    mapping: dict[int, str] = {}

    # 1. low_signal: lowest face_valid_ratio
    c_low = min(remaining, key=lambda c: float(centroids_z[c, idx["face_valid_ratio"]]))
    mapping[c_low] = "low_signal"
    remaining.remove(c_low)
    if not remaining:
        return mapping

    # 2. active_expressive: highest mouth_motion_std + head_motion_std + gaze_dev_mean
    c_active = max(
        remaining,
        key=lambda c: float(
            centroids_z[c, idx["mouth_motion_std"]]
            + centroids_z[c, idx["head_motion_std"]]
            + 0.5 * centroids_z[c, idx["gaze_dev_mean"]]
        ),
    )
    mapping[c_active] = "active_expressive"
    remaining.remove(c_active)
    if not remaining:
        return mapping

    # 3. pause_thinking: lowest mouth_open_mean + mouth_motion_std + voiced_ratio
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

    for c in sorted(remaining):
        mapping[c] = "steady_speaking"
    return mapping


def _build_regime_prep_and_gmm() -> tuple[Pipeline, GaussianMixture]:
    prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("power", PowerTransformer(method="yeo-johnson")),
        ("scaler", StandardScaler()),
    ])
    gmm = GaussianMixture(
        n_components=4,
        covariance_type="diag",
        reg_covar=0.15,
        n_init=5,
        random_state=42,
    )
    return prep, gmm


def _load_english_regime_matrix(cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    frames: list[np.ndarray] = []
    corpora: list[np.ndarray] = []
    for corpus_name in ("mit", "recruitview", "chalearn"):
        p = cache_dir / f"{corpus_name}_multimodal_features.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, dtype={"sample_id": str, "group_id": str})
        hm_col = "head_motion_rate" if "head_motion_rate" in df.columns else "head_motion_step"
        mat = np.column_stack([
            pd.to_numeric(df["face_valid_ratio"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(df["au25_mouth_open_mean"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(df["au25_mouth_open_std"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(df["gaze_dev_mean"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(df[hm_col], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(df["voiced_frame_ratio"], errors="coerce").to_numpy(dtype=float),
        ])
        frames.append(mat)
        corpora.append(np.full(len(df), corpus_name, dtype=object))
    if not frames:
        raise RuntimeError("No extracted English feature caches found for Experiment 4.")
    return np.vstack(frames), np.concatenate(corpora)


def run_experiment_4_regime_transfer(
    cache_dir: Path,
    models_dir: Path,
) -> dict[str, Any]:
    print("\n=== Experiment 4: Cross-Corpus Behavioral Regime Transfer (English <-> 47 Chinese Sessions) ===", flush=True)
    from tools.train_behavior_regime import build_dataset as build_chinese_dataset
    from report_frontend.behavior_regime_model import FEATURE_NAMES as FULL_CH_FEATURES

    X_ch_full, groups_ch, n_ch_sessions, n_ch_voice = build_chinese_dataset()
    ch_idx = [FULL_CH_FEATURES.index(name) for name in SHARED_REGIME_FEATURES]
    X_ch = X_ch_full[:, ch_idx].astype(float)
    X_en, corpora_en = _load_english_regime_matrix(cache_dir)

    # 1. Chinese-only K=4 GMM (full reference + 5-fold GroupKFold)
    prep_ch_full, gmm_ch_full = _build_regime_prep_and_gmm()
    X_ch_z = prep_ch_full.fit_transform(X_ch)
    gmm_ch_full.fit(X_ch_z)
    ch_ref_cids = gmm_ch_full.predict(X_ch_z)
    ch_ref_map = assign_audiovisual_cluster_states(gmm_ch_full.means_, SHARED_REGIME_FEATURES)
    ch_ref_states = np.asarray([ch_ref_map[int(c)] for c in ch_ref_cids], dtype=object)

    gkf = GroupKFold(n_splits=5)
    ch_cv_sil, ch_cv_db, ch_cv_ari, ch_cv_nmi, ch_cv_agr = [], [], [], [], []
    for tr, te in gkf.split(X_ch, groups=groups_ch):
        p_f, g_f = _build_regime_prep_and_gmm()
        Z_tr = p_f.fit_transform(X_ch[tr])
        Z_te = p_f.transform(X_ch[te])
        g_f.fit(Z_tr)
        te_cids = g_f.predict(Z_te)
        f_map = assign_audiovisual_cluster_states(g_f.means_, SHARED_REGIME_FEATURES)
        te_states = np.asarray([f_map[int(c)] for c in te_cids], dtype=object)
        ch_cv_sil.append(float(silhouette_score(Z_te, te_cids)))
        ch_cv_db.append(float(davies_bouldin_score(Z_te, te_cids)))
        ch_cv_ari.append(float(adjusted_rand_score(ch_ref_states[te], te_states)))
        ch_cv_nmi.append(float(normalized_mutual_info_score(ch_ref_states[te], te_states)))
        ch_cv_agr.append(float(np.mean(ch_ref_states[te] == te_states)))

    # 2. English-pretrained K=4 GMM zero-shot evaluated on 47 Chinese sessions
    prep_en, gmm_en = _build_regime_prep_and_gmm()
    X_en_z = prep_en.fit_transform(X_en)
    gmm_en.fit(X_en_z)
    en_map = assign_audiovisual_cluster_states(gmm_en.means_, SHARED_REGIME_FEATURES)

    X_ch_in_en_z = prep_en.transform(X_ch)
    en_on_ch_cids = gmm_en.predict(X_ch_in_en_z)
    en_on_ch_states = np.asarray([en_map[int(c)] for c in en_on_ch_cids], dtype=object)

    en_n_clusters = len(np.unique(en_on_ch_cids))
    en_sil = float(silhouette_score(X_ch_in_en_z, en_on_ch_cids)) if en_n_clusters >= 2 else 0.0
    en_db = float(davies_bouldin_score(X_ch_in_en_z, en_on_ch_cids)) if en_n_clusters >= 2 else 0.0
    en_ari = float(adjusted_rand_score(ch_ref_states, en_on_ch_states))
    en_nmi = float(normalized_mutual_info_score(ch_ref_states, en_on_ch_states))
    en_agr = float(np.mean(ch_ref_states == en_on_ch_states))

    # 3. Bilingual Joint English+Chinese K=4 GMM (balanced English + 4/5 Chinese train -> 1/5 Chinese val)
    bi_cv_sil, bi_cv_db, bi_cv_ari, bi_cv_nmi, bi_cv_agr = [], [], [], [], []
    # Replicate English samples inside training so English (~600 clips) carries ~30% weight alongside ~4,500 Chinese windows
    en_repeat = max(1, int(round(0.35 * len(X_ch) / max(len(X_en), 1))))
    X_en_bal = np.tile(X_en, (en_repeat, 1))

    for tr, te in gkf.split(X_ch, groups=groups_ch):
        p_b, g_b = _build_regime_prep_and_gmm()
        X_joint_tr = np.vstack([X_ch[tr], X_en_bal])
        Z_joint_tr = p_b.fit_transform(X_joint_tr)
        Z_ch_te = p_b.transform(X_ch[te])
        g_b.fit(Z_joint_tr)
        te_cids = g_b.predict(Z_ch_te)
        b_map = assign_audiovisual_cluster_states(g_b.means_, SHARED_REGIME_FEATURES)
        te_states = np.asarray([b_map[int(c)] for c in te_cids], dtype=object)

        bi_cv_sil.append(float(silhouette_score(Z_ch_te, te_cids)))
        bi_cv_db.append(float(davies_bouldin_score(Z_ch_te, te_cids)))
        bi_cv_ari.append(float(adjusted_rand_score(ch_ref_states[te], te_states)))
        bi_cv_nmi.append(float(normalized_mutual_info_score(ch_ref_states[te], te_states)))
        bi_cv_agr.append(float(np.mean(ch_ref_states[te] == te_states)))

    # Fit full Bilingual model for deployment
    prep_bi_full, gmm_bi_full = _build_regime_prep_and_gmm()
    X_joint_full = np.vstack([X_ch, X_en_bal])
    Z_joint_full = prep_bi_full.fit_transform(X_joint_full)
    gmm_bi_full.fit(Z_joint_full)
    bi_full_map = assign_audiovisual_cluster_states(gmm_bi_full.means_, SHARED_REGIME_FEATURES)
    bi_on_ch_cids = gmm_bi_full.predict(prep_bi_full.transform(X_ch))
    bi_on_ch_states = np.asarray([bi_full_map[int(c)] for c in bi_on_ch_cids], dtype=object)

    models_dir.mkdir(parents=True, exist_ok=True)
    regime_model_path = models_dir / "behavior_regime_multicorpus.joblib"
    joblib.dump(
        {
            "model_type": "GaussianMixture",
            "n_components": 4,
            "feature_names": list(SHARED_REGIME_FEATURES),
            "state_order": list(REGIME_STATE_ORDER),
            "preprocessor": prep_bi_full,
            "gmm": gmm_bi_full,
            "cluster_to_state": bi_full_map,
            "english_only_preprocessor": prep_en,
            "english_only_gmm": gmm_en,
            "english_only_cluster_to_state": en_map,
        },
        regime_model_path,
        compress=3,
    )

    def _dist(states: np.ndarray) -> dict[str, float]:
        n = max(len(states), 1)
        return {st: round(float(np.sum(states == st) / n), 4) for st in REGIME_STATE_ORDER}

    results = {
        "shared_features": list(SHARED_REGIME_FEATURES),
        "chinese_corpus": {
            "n_sessions": int(n_ch_sessions),
            "n_sessions_with_voice": int(n_ch_voice),
            "n_windows": int(len(X_ch)),
        },
        "english_corpus": {
            "n_clips": int(len(X_en)),
            "breakdown": {c: int(np.sum(corpora_en == c)) for c in ("mit", "recruitview", "chalearn") if np.any(corpora_en == c)},
        },
        "regimes": {
            "Chinese-only-GMM (5-fold GroupKFold)": {
                "val_silhouette_mean": round(float(np.mean(ch_cv_sil)), 4),
                "val_davies_bouldin_mean": round(float(np.mean(ch_cv_db)), 4),
                "ari_vs_chinese_ref": round(float(np.mean(ch_cv_ari)), 4),
                "nmi_vs_chinese_ref": round(float(np.mean(ch_cv_nmi)), 4),
                "state_agreement_vs_chinese_ref": round(float(np.mean(ch_cv_agr)), 4),
                "chinese_state_distribution": _dist(ch_ref_states),
            },
            "English-pretrained-GMM (Zero-Shot on Chinese)": {
                "silhouette_on_chinese": round(en_sil, 4),
                "davies_bouldin_on_chinese": round(en_db, 4),
                "ari_vs_chinese_ref": round(en_ari, 4),
                "nmi_vs_chinese_ref": round(en_nmi, 4),
                "state_agreement_vs_chinese_ref": round(en_agr, 4),
                "chinese_state_distribution": _dist(en_on_ch_states),
            },
            "Bilingual-Joint-GMM (5-fold GroupKFold on Chinese)": {
                "val_silhouette_mean": round(float(np.mean(bi_cv_sil)), 4),
                "val_davies_bouldin_mean": round(float(np.mean(bi_cv_db)), 4),
                "ari_vs_chinese_ref": round(float(np.mean(bi_cv_ari)), 4),
                "nmi_vs_chinese_ref": round(float(np.mean(bi_cv_nmi)), 4),
                "state_agreement_vs_chinese_ref": round(float(np.mean(bi_cv_agr)), 4),
                "chinese_state_distribution": _dist(bi_on_ch_states),
            },
        },
        "saved_model_path": str(regime_model_path),
    }

    for k, v in results["regimes"].items():
        sil_k = v.get("val_silhouette_mean", v.get("silhouette_on_chinese"))
        print(
            f"  [Regime] {k:46s} sil={sil_k:.4f} ARI={v['ari_vs_chinese_ref']:.4f} agree={v['state_agreement_vs_chinese_ref']:.4f}",
            flush=True,
        )

    return results


# ---------------------------------------------------------------------------
# Report Rendering (JSON + Markdown, strict allow_nan=False, zero PII)
# ---------------------------------------------------------------------------

def render_markdown_report(bench: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Multi-Corpus Research Benchmark Report (2026-11-24)")
    lines.append("")
    lines.append("> **Research-Only Notice**: All models and benchmarks below are strictly offline research experiments across public English multimodal corpora and anonymized local L0 kinematics windows (zero PII, excluded from `report_frontend/` scoring).")
    lines.append("")

    # Extraction summary
    ext = bench.get("extraction_summary", {})
    if ext:
        lines.append("## 0. Multimodal Feature Extraction Throughput")
        lines.append("")
        lines.append(f"- **Total Clips Extracted**: {ext.get('total_clips_succeeded', 0)} / {ext.get('total_clips_scheduled', 0)} (failed: {ext.get('total_clips_failed', 0)})")
        lines.append(f"- **Wall Time & Throughput**: {ext.get('elapsed_sec', 0)}s ({ext.get('clips_per_sec', 0)} clips/sec across {ext.get('workers', 0)} CPU workers, {ext.get('frames_per_clip', 0)} frames/clip)")
        lines.append("")
        lines.append("| Corpus | Clips | Features | Mean Face Detect | Silero ONNX Clips |")
        lines.append("|---|---:|---:|---:|---:|")
        for c_name, c_info in ext.get("corpora", {}).items():
            lines.append(
                f"| {c_name} | {c_info['n_clips']} | {c_info['n_features']} | "
                f"{c_info['mean_face_valid_ratio']:.3f} | {c_info['silero_onnx_count']}/{c_info['n_clips']} |"
            )
        lines.append("")

    # Exp 1: MIT
    e1 = bench["experiment_1_mit"]
    lines.append("## 1. Experiment 1 · MIT Interview (`N=138`, 69 Participants, 5-Fold GroupKFold)")
    lines.append("")
    gpu = e1.get("gpu_mlp_hardware", {})
    if gpu.get("available"):
        lines.append(f"- **GPU Hardware Benchmark**: `{gpu['device_name']}` (`torch {gpu['torch_version']}`, strict `{gpu['dtype']}`, peak VRAM `{gpu['peak_vram_mb']} MB`, 5-fold total `{gpu['elapsed_sec']}s`)")
    lines.append(f"- **Best Model (Macro-18 Pearson r)**: `{e1['best_model']}`")
    lines.append("")
    lines.append("| Model | Macro-18 $r$ | Macro-18 $\\rho$ | Core-8 $r$ | Macro-18 RMSE |")
    lines.append("|---|---:|---:|---:|---:|")
    for m_name, m_res in e1["models"].items():
        lines.append(
            f"| {m_name} | {m_res['macro']['pearson_r']:.4f} | {m_res['macro']['spearman_rho']:.4f} | "
            f"{m_res['core_macro']['pearson_r']:.4f} | {m_res['macro']['rmse']:.4f} |"
        )
    lines.append("")
    lines.append(f"### Per-Target Breakdown for `{e1['best_model']}` (Core-8 Highlights)")
    lines.append("")
    lines.append("| Target | Pearson $r$ | Spearman $\\rho$ | RMSE |")
    lines.append("|---|---:|---:|---:|")
    best_e1_pt = e1["models"][e1["best_model"]]["per_target"]
    for t in e1["core_targets"]:
        row = best_e1_pt[t]
        lines.append(f"| {t} | {row['pearson_r']:.4f} | {row['spearman_rho']:.4f} | {row['rmse']:.4f} |")
    lines.append("")

    # Exp 2: RecruitView
    e2 = bench["experiment_2_recruitview"]
    lines.append("## 2. Experiment 2 · RecruitView (`N=2,011` Text + `N=160` Multimodal, 5-Fold GroupKFold by `user_no`)")
    lines.append("")
    ta = e2["track_a_full_metadata_text"]
    lines.append(f"### Track A · Full Transcript Corpus (`N={ta['n_clips']}`, `{ta['n_users']}` Unique Users)")
    lines.append("")
    lines.append("| Model | Macro-12 $r$ | Macro-12 $\\rho$ | BigFive-6 $r$ | Macro-12 RMSE |")
    lines.append("|---|---:|---:|---:|---:|")
    for m_name, m_res in ta["models"].items():
        lines.append(
            f"| {m_name} | {m_res['macro']['pearson_r']:.4f} | {m_res['macro']['spearman_rho']:.4f} | "
            f"{m_res['core_macro']['pearson_r']:.4f} | {m_res['macro']['rmse']:.4f} |"
        )
    lines.append("")

    tb = e2["track_b_multimodal"]
    lines.append(f"### Track B · Multimodal Subset (`N={tb['n_clips']}`, `{tb['n_users']}` Unique Users, `{tb['n_behavior_features']}` Behavior Features)")
    lines.append("")
    lines.append("| Model | Macro-12 $r$ | BigFive-6 $r$ | 1 - MAE_norm | Macro-12 RMSE |")
    lines.append("|---|---:|---:|---:|---:|")
    for m_name, m_res in tb["models"].items():
        lines.append(
            f"| {m_name} | {m_res['macro']['pearson_r']:.4f} | {m_res['core_macro']['pearson_r']:.4f} | "
            f"{m_res['macro']['acc_1_minus_mae']:.4f} | {m_res['macro']['rmse']:.4f} |"
        )
    lines.append("")

    # Exp 3: ChaLearn
    e3 = bench["experiment_3_chalearn"]
    sc = e3["split_counts"]
    lines.append(f"## 3. Experiment 3 · ChaLearn First Impressions V2 (`train={sc['train']}`, `val={sc['val']}`, `test={sc['test']}`)")
    lines.append("")
    lines.append(f"- **Best Model**: `{e3['best_model']}` (`{e3['n_visual_features']}` visual + `{e3['n_acoustic_features']}` acoustic features)")
    lines.append("")
    lines.append("| Model | Official `1 - MAE` | Pearson $r$ | Spearman $\\rho$ | RMSE |")
    lines.append("|---|---:|---:|---:|---:|")
    for m_name, m_res in e3["models"].items():
        lines.append(
            f"| {m_name} | {m_res['macro']['acc_1_minus_mae']:.4f} | {m_res['macro']['pearson_r']:.4f} | "
            f"{m_res['macro']['spearman_rho']:.4f} | {m_res['macro']['rmse']:.4f} |"
        )
    lines.append("")
    lines.append(f"### Per-Trait ChaLearn Test Metrics (`{e3['best_model']}`)")
    lines.append("")
    lines.append("| Trait | `1 - MAE` | Pearson $r$ | Spearman $\\rho$ | RMSE |")
    lines.append("|---|---:|---:|---:|---:|")
    for t in e3["targets"]:
        row = e3["models"][e3["best_model"]]["per_target"][t]
        lines.append(f"| {t} | {row['acc_1_minus_mae']:.4f} | {row['pearson_r']:.4f} | {row['spearman_rho']:.4f} | {row['rmse']:.4f} |")
    lines.append("")

    # Exp 4: Cross-Corpus Regime Transfer
    e4 = bench["experiment_4_regime_transfer"]
    lines.append("## 4. Experiment 4 · Cross-Corpus Behavioral Regime Transfer (`English <-> 47 Chinese Sessions`)")
    lines.append("")
    lines.append(
        f"- **Chinese Reference Corpus**: {e4['chinese_corpus']['n_sessions']} sessions "
        f"({e4['chinese_corpus']['n_windows']} windows, {e4['chinese_corpus']['n_sessions_with_voice']} with voice)"
    )
    lines.append(f"- **English Transfer Corpus**: {e4['english_corpus']['n_clips']} clips ({e4['english_corpus']['breakdown']})")
    lines.append("")
    lines.append("| Regime Model | Silhouette (Chinese) | Davies-Bouldin | ARI vs Chinese | NMI vs Chinese | Agreement |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r_name, r_info in e4["regimes"].items():
        sil = r_info.get("val_silhouette_mean", r_info.get("silhouette_on_chinese"))
        db = r_info.get("val_davies_bouldin_mean", r_info.get("davies_bouldin_on_chinese"))
        lines.append(
            f"| {r_name} | {sil:.4f} | {db:.4f} | "
            f"{r_info['ari_vs_chinese_ref']:.4f} | {r_info['nmi_vs_chinese_ref']:.4f} | {r_info['state_agreement_vs_chinese_ref']:.4f} |"
        )
    lines.append("")

    # Saved model artifacts
    lines.append("## 5. Saved Research Model Artifacts (`models/research/`)")
    lines.append("")
    lines.append("| Artifact | Size (KB) |")
    lines.append("|---|---:|")
    for name, sz in bench.get("saved_model_sizes_kb", {}).items():
        lines.append(f"| `{name}` | {sz:.1f} |")
    lines.append("")
    return "\n".join(lines)


def run_all_experiments(
    *,
    datasets_root: Path = DEFAULT_DATASETS_ROOT,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    models_dir: Path = DEFAULT_MODELS_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    extraction_summary: dict[str, Any] = {}
    bench_cache = cache_dir / "extraction_benchmark.json"
    if bench_cache.exists():
        extraction_summary = json.loads(bench_cache.read_text(encoding="utf-8"))

    e1 = run_experiment_1_mit(datasets_root, cache_dir, models_dir)
    e2 = run_experiment_2_recruitview(datasets_root, cache_dir, models_dir)
    e3 = run_experiment_3_chalearn(cache_dir, models_dir)
    e4 = run_experiment_4_regime_transfer(cache_dir, models_dir)

    model_sizes_kb: dict[str, float] = {}
    for p in sorted(models_dir.glob("*.joblib")):
        model_sizes_kb[p.name] = round(p.stat().st_size / 1024.0, 2)

    benchmark: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "research_only": True,
        "total_training_elapsed_sec": round(time.perf_counter() - t0, 2),
        "extraction_summary": extraction_summary,
        "experiment_1_mit": e1,
        "experiment_2_recruitview": e2,
        "experiment_3_chalearn": e3,
        "experiment_4_regime_transfer": e4,
        "saved_model_sizes_kb": model_sizes_kb,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / "multicorpus_benchmark_20261124.json"
    md_path = results_dir / "multicorpus_benchmark_20261124.md"

    json_path.write_text(json.dumps(benchmark, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    md_path.write_text(render_markdown_report(benchmark), encoding="utf-8")
    print(f"\nSaved benchmark JSON -> {json_path}", flush=True)
    print(f"Saved benchmark MD   -> {md_path}", flush=True)
    return benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-corpus multimodal research training suite")
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--torch-mlp-worker",
        nargs=3,
        metavar=("PAYLOAD_NPZ", "OUT_JSON", "OUT_PRED_NPY"),
        help="Internal worker mode invoked with funasr CUDA Python",
    )
    args = parser.parse_args()

    if args.torch_mlp_worker:
        npz_p, json_p, npy_p = [Path(x) for x in args.torch_mlp_worker]
        run_torch_mlp_worker(npz_p, json_p, npy_p)
        return

    run_all_experiments(
        datasets_root=args.datasets_root,
        cache_dir=args.cache_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
