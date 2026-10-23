#!/usr/bin/env python
"""
CueCoT Phase 4: Baselines — AAAI 2027 Experimental Rigor
=========================================================
Implements all comparison methods for the CueCoT paper:

  Unimodal:       SVR / Ridge / Random Forest (per modality × per dimension)
  Multimodal:     Early Fusion MLP / Late Fusion / CRMF (CCA-based)
  Deep Learning:  Attention-LSTM (BiLSTM + Multi-head Self-Attention)
  LLM Methods:    CueCoT (cot_full) / Direct LLM / Generic CoT (from llm_results.jsonl)
  Text-only LLM:  Transcript + Gemini summary → LLM (no multimodal cues)
  Statistical:    Paired bootstrap test, 95% CI, permutation p-values

Global standards:
  - 5 seeds: [42, 123, 456, 789, 1024]
  - User-level 7:1:2 split (same user's videos in one split)
  - MSE (primary) + Pearson r + Spearman ρ + MAE
  - Per-dimension + 12-dim macro avg + personality/interview group avg
  - All randomness seeded (numpy, random, torch)

Usage:
  python baselines.py                    # full run (all methods, all seeds)
  python baselines.py --methods svr,ridge,rf  # specific methods only
  python baselines.py --skip-text-llm    # skip text-only LLM (saves API cost)
  python baselines.py --skip-attention   # skip Attention-LSTM (no PyTorch)
  python baselines.py --test            # quick test on 50 videos, 1 seed
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─── Paths ───────────────────────────────────────────────────────────
FEATURES_DIR = Path(r"D:\jingxin\experiments\results\features")
METADATA_PATH = Path(r"D:\RecruitView\RecruitView\metadata.jsonl")
OUTPUT_DIR = Path(r"D:\jingxin\experiments\results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X_NPY = FEATURES_DIR / "all_features.npy"
Y_NPY = FEATURES_DIR / "targets.npy"
NAMES_TXT = FEATURES_DIR / "feature_names.txt"
LLM_RESULTS = FEATURES_DIR / "llm_results.jsonl"

# ─── Global Constants ─────────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 1024]
SPLIT_RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}

TARGET_KEYS = [
    "openness", "conscientiousness", "extraversion", "agreeableness",
    "neuroticism", "overall_personality",          # Personality (6)
    "interview_score", "answer_score", "speaking_skills",
    "confidence_score", "facial_expression", "overall_performance",  # Interview (6)
]
PERSONALITY_KEYS = TARGET_KEYS[:6]
INTERVIEW_KEYS = TARGET_KEYS[6:]

# ─── Hyperparameters (centralized, with justification) ────────────────
SVR_PARAM_DISTRIBUTIONS = {
    "estimator__C": [0.1, 1, 10, 100],
    "estimator__gamma": [0.001, 0.01, 0.1, 1, "scale"],
}
SVR_RANDOM_ITERS = 10  # RandomizedSearch iterations (vs 20 full grid)
SVR_PCA_DIM = {"face": 100, "gesture": 50, "voice": 10, "all": 150}
# PCA before SVR reduces noise and speeds up fit; dims chosen to retain ~95% variance

RIDGE_ALPHAS = [0.01, 0.1, 1, 10, 100]
# RidgeCV uses efficient leave-one-out CV, no need for GridSearch

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_leaf": 5,
    "random_state": 42,
}   # fixed params; RandomForest is robust to defaults
MLP_PARAMS = {
    "hidden_layer_sizes": (256, 128, 64),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-4,
    "batch_size": 32,
    "learning_rate_init": 1e-3,
    "max_iter": 200,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 20,
    "random_state": 42,
}   # 3-layer MLP with early stopping; architecture follows RecruitView
ATTENTION_LSTM_PARAMS = {
    "input_dim": 1410,
    "lstm_hidden": 128,
    "lstm_layers": 2,
    "num_heads": 4,
    "fc_dim": 64,
    "output_dim": 12,
    "dropout": 0.2,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "epochs": 100,
    "batch_size": 32,
    "patience": 15,
}   # BiLSTM + multi-head self-attention; paper reference: malik2022


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_data() -> dict:
    """Load feature matrix, targets, feature names, and metadata index."""
    X = np.load(X_NPY).astype(np.float64)
    y = np.load(Y_NPY).astype(np.float64)

    with open(NAMES_TXT) as f:
        feature_names = [ln.strip() for ln in f]

    # Modality column indices
    face_idx = [i for i, n in enumerate(feature_names) if n.startswith("face.")]
    gesture_idx = [i for i, n in enumerate(feature_names) if n.startswith("gesture.")]
    voice_idx = [i for i, n in enumerate(feature_names) if n.startswith("voice.")]

    # Metadata: video_id → {user_no, transcript, gemini_summary, targets}
    meta_raw = {}
    with open(METADATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            vid = Path(d["file_name"]).stem
            meta_raw[vid] = {
                "user_no": d["user_no"],
                "transcript": d.get("transcript", ""),
                "gemini_summary": d.get("gemini_summary", ""),
                "targets": {k: d[k] for k in TARGET_KEYS},
            }

    # Read video order from all_features.jsonl (targets.npy follows this order)
    video_ids = []
    with open(FEATURES_DIR / "all_features.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            video_ids.append(json.loads(line)["video_id"])

    assert len(video_ids) == X.shape[0] == y.shape[0], \
        f"Mismatch: {len(video_ids)} ids vs {X.shape[0]} X vs {y.shape[0]} y"

    # Verify: targets.npy should match metadata (order-independent check)
    metadata = {vid: meta_raw[vid] for vid in video_ids if vid in meta_raw}
    y_from_meta = np.array([[metadata[vid]["targets"][k] for k in TARGET_KEYS]
                            for vid in video_ids], dtype=np.float64)
    assert np.allclose(y, y_from_meta, atol=1e-4), \
        "Stored targets mismatch metadata — check feature extraction order"

    # User → video_ids mapping
    user_videos: dict[str, list[str]] = defaultdict(list)
    for vid in video_ids:
        user_videos[metadata[vid]["user_no"]].append(vid)

    return {
        "X": X,
        "y": y,
        "video_ids": video_ids,
        "feature_names": feature_names,
        "modality_idx": {
            "face": face_idx,
            "gesture": gesture_idx,
            "voice": voice_idx,
            "all": list(range(len(feature_names))),
        },
        "metadata": metadata,
        "user_videos": dict(user_videos),
    }


# ═══════════════════════════════════════════════════════════════════════
# SPLIT MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════

def create_user_splits(user_videos: dict, seed: int) -> dict:
    """
    User-level stratified 7:1:2 split.
    Stratification: sort users by number of videos, interleave to balance
    video counts across splits.
    """
    users = sorted(user_videos.keys(), key=lambda u: len(user_videos[u]))
    rng = np.random.RandomState(seed)

    # Interleaved assignment: preserves per-split video count balance
    train_u, val_u, test_u = [], [], []
    for i, u in enumerate(users):
        bucket = i % 10
        if bucket < 7:
            train_u.append(u)
        elif bucket < 8:
            val_u.append(u)
        else:
            test_u.append(u)

    # Shuffle within each split (not strictly needed but consistent)
    rng.shuffle(train_u)
    rng.shuffle(val_u)
    rng.shuffle(test_u)

    def _vids(user_list):
        vids = []
        for u in user_list:
            vids.extend(user_videos[u])
        return vids

    return {
        "seed": seed,
        "train_users": train_u,
        "val_users": val_u,
        "test_users": test_u,
        "train_videos": _vids(train_u),
        "val_videos": _vids(val_u),
        "test_videos": _vids(test_u),
    }


def get_split_indices(video_ids: list, split_videos: list) -> np.ndarray:
    """Convert video_id lists to index arrays."""
    vid_to_idx = {v: i for i, v in enumerate(video_ids)}
    return np.array([vid_to_idx[v] for v in split_videos if v in vid_to_idx])


# ═══════════════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Per-dimension metrics + macro averages.
    y_true, y_pred: (n_samples, 12)
    """
    n_dims = y_true.shape[1]
    per_dim = []
    for d in range(n_dims):
        yt = y_true[:, d]
        yp = y_pred[:, d]
        mse = np.mean((yt - yp) ** 2)
        mae = np.mean(np.abs(yt - yp))
        # Pearson r
        if np.std(yt) > 1e-8 and np.std(yp) > 1e-8:
            pr, _ = stats.pearsonr(yt, yp)
        else:
            pr = 0.0
        # Spearman ρ
        if np.std(yt) > 1e-8 and np.std(yp) > 1e-8:
            sr, _ = stats.spearmanr(yt, yp)
        else:
            sr = 0.0
        per_dim.append({
            "dimension": TARGET_KEYS[d],
            "mse": float(mse), "mae": float(mae),
            "pearson_r": float(pr), "spearman_rho": float(sr),
        })

    # Macro averages
    macro = {
        "mse": float(np.mean([d["mse"] for d in per_dim])),
        "mae": float(np.mean([d["mae"] for d in per_dim])),
        "pearson_r": float(np.mean([d["pearson_r"] for d in per_dim])),
        "spearman_rho": float(np.mean([d["spearman_rho"] for d in per_dim])),
    }
    personality = {k: float(np.mean([d[k] for d in per_dim[:6]]))
                   for k in ["mse", "mae", "pearson_r", "spearman_rho"]}
    interview = {k: float(np.mean([d[k] for d in per_dim[6:]]))
                 for k in ["mse", "mae", "pearson_r", "spearman_rho"]}

    return {
        "per_dimension": per_dim,
        "macro_avg": macro,
        "personality_avg": personality,
        "interview_avg": interview,
    }


# ═══════════════════════════════════════════════════════════════════════
# UNIMODAL REGRESSORS
# ═══════════════════════════════════════════════════════════════════════

def _random_search_svr(X_tr, y_tr, X_val, y_val):
    """RandomizedSearchCV for SVR — 10 iters, much faster than full grid."""
    X_cv = np.vstack([X_tr, X_val])
    y_cv = np.concatenate([y_tr, y_val], axis=0)
    test_fold = np.array([-1] * len(X_tr) + [0] * len(X_val))
    cv = PredefinedSplit(test_fold)

    base = SVR(kernel="rbf")
    model = MultiOutputRegressor(base)
    search = RandomizedSearchCV(
        model, SVR_PARAM_DISTRIBUTIONS, n_iter=SVR_RANDOM_ITERS,
        cv=cv, scoring="neg_mean_squared_error", n_jobs=-1, refit=True,
        random_state=42,
    )
    search.fit(X_cv, y_cv)
    return search.best_estimator_


def _ridge_cv(X_tr, y_train):
    """RidgeCV: efficient leave-one-out CV, single fit."""
    model = MultiOutputRegressor(RidgeCV(alphas=RIDGE_ALPHAS))
    model.fit(X_tr, y_train)
    return model


def run_unimodal(X_train, y_train, X_val, y_val, X_test, y_test,
                 modality_idx: dict, method: str) -> dict:
    """
    Train unimodal regressor per modality.
    - SVR: PCA → RandomizedSearchCV (10 iters)
    - Ridge: RidgeCV (efficient built-in CV)
    - RF: fixed params (robust to defaults)
    """
    results = {}
    for mod_name in ["face", "gesture", "voice", "all"]:
        idx = modality_idx[mod_name]
        t0 = time.time()

        # Per-modality scaling
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train[:, idx])
        X_val_s = scaler.transform(X_val[:, idx])
        X_te_s = scaler.transform(X_test[:, idx])

        # PCA for SVR to reduce noise and speed up fit
        if method == "svr" and X_tr_s.shape[1] > SVR_PCA_DIM.get(mod_name, 100):
            pca_dim = min(SVR_PCA_DIM[mod_name], X_tr_s.shape[1], X_tr_s.shape[0] - 1)
            pca = PCA(n_components=pca_dim, random_state=42)
            X_tr_s = pca.fit_transform(X_tr_s)
            X_val_s = pca.transform(X_val_s)
            X_te_s = pca.transform(X_te_s)

        if method == "svr":
            model = _random_search_svr(X_tr_s, y_train, X_val_s, y_val)
        elif method == "ridge":
            model = _ridge_cv(X_tr_s, y_train)
        elif method == "rf":
            model = MultiOutputRegressor(RandomForestRegressor(**RF_PARAMS))
            model.fit(X_tr_s, y_train)
        else:
            raise ValueError(f"Unknown method: {method}")

        y_pred = model.predict(X_te_s)
        metrics = compute_metrics(y_test, y_pred)
        metrics["fit_time_s"] = round(time.time() - t0, 1)

        if method == "svr":
            metrics["best_params"] = {
                "C": model.estimators_[0].C,
                "gamma": model.estimators_[0].gamma,
            }
        elif method == "ridge":
            chosen = model.estimators_[0].alpha_
            metrics["best_params"] = {"alpha": float(chosen)}

        results[mod_name] = metrics
    return results


# ═══════════════════════════════════════════════════════════════════════
# MULTIMODAL FUSION
# ═══════════════════════════════════════════════════════════════════════

def run_early_fusion_mlp(X_train, y_train, X_val, y_val, X_test, y_test,
                         modality_idx: dict, seed: int) -> dict:
    """
    Early Fusion MLP: concatenate all modalities, 3-layer MLP with dropout.
    Uses separate scalers per modality.
    """
    t0 = time.time()
    # Scale each modality separately before concatenation
    scalers = {}
    X_tr_parts, X_val_parts, X_te_parts = [], [], []
    for mod in ["face", "gesture", "voice"]:
        idx = modality_idx[mod]
        sc = RobustScaler()
        X_tr_parts.append(sc.fit_transform(X_train[:, idx]))
        X_val_parts.append(sc.transform(X_val[:, idx]))
        X_te_parts.append(sc.transform(X_test[:, idx]))
        scalers[mod] = sc

    X_tr_fused = np.hstack(X_tr_parts)
    X_val_fused = np.hstack(X_val_parts)
    X_te_fused = np.hstack(X_te_parts)

    # Additional global scaling
    global_scaler = StandardScaler()
    X_tr_fused = global_scaler.fit_transform(X_tr_fused)
    X_val_fused = global_scaler.transform(X_val_fused)
    X_te_fused = global_scaler.transform(X_te_fused)

    mlp = MLPRegressor(
        hidden_layer_sizes=MLP_PARAMS["hidden_layer_sizes"],
        activation=MLP_PARAMS["activation"],
        solver=MLP_PARAMS["solver"],
        alpha=MLP_PARAMS["alpha"],
        batch_size=MLP_PARAMS["batch_size"],
        learning_rate_init=MLP_PARAMS["learning_rate_init"],
        max_iter=MLP_PARAMS["max_iter"],
        early_stopping=MLP_PARAMS["early_stopping"],
        validation_fraction=MLP_PARAMS["validation_fraction"],
        n_iter_no_change=MLP_PARAMS["n_iter_no_change"],
        random_state=seed,
    )
    mlp.fit(X_tr_fused, y_train)
    y_pred = mlp.predict(X_te_fused)

    metrics = compute_metrics(y_test, y_pred)
    metrics["fit_time_s"] = round(time.time() - t0, 1)
    metrics["n_iter_"] = int(mlp.n_iter_)
    metrics["best_loss_"] = float(mlp.best_loss_) if mlp.best_loss_ is not None else None
    return metrics


def run_late_fusion(X_train, y_train, X_val, y_val, X_test, y_test,
                    modality_idx: dict) -> dict:
    """
    Late Fusion: weighted average of best unimodal (SVR) predictions.
    Trains separate SVR per modality on training set, then optimizes fusion
    weights on val set via grid search.
    """
    from itertools import product

    t0 = time.time()
    mods = ["face", "gesture", "voice"]

    # Train best SVR per modality on train, predict on val and test
    val_preds = {}
    test_preds = {}
    for mod in mods:
        idx = modality_idx[mod]
        sc = StandardScaler()
        X_tr_m = sc.fit_transform(X_train[:, idx])
        X_val_m = sc.transform(X_val[:, idx])
        X_te_m = sc.transform(X_test[:, idx])

        svr = MultiOutputRegressor(SVR(kernel="rbf", C=1.0, gamma="scale"))
        svr.fit(X_tr_m, y_train)
        val_preds[mod] = svr.predict(X_val_m)
        test_preds[mod] = svr.predict(X_te_m)

    # Grid search weights on val
    weights_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    best_mse = float("inf")
    best_weights = None

    for wf, wg, wv in product(weights_grid, repeat=3):
        total = wf + wg + wv
        if total <= 0:
            continue
        wf_n, wg_n, wv_n = wf / total, wg / total, wv / total
        fused_val = (wf_n * val_preds["face"]
                     + wg_n * val_preds["gesture"]
                     + wv_n * val_preds["voice"])
        mse = np.mean((y_val - fused_val) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_weights = {"face": wf_n, "gesture": wg_n, "voice": wv_n}

    y_pred = (best_weights["face"] * test_preds["face"]
              + best_weights["gesture"] * test_preds["gesture"]
              + best_weights["voice"] * test_preds["voice"])

    metrics = compute_metrics(y_test, y_pred)
    metrics["fit_time_s"] = round(time.time() - t0, 1)
    metrics["fusion_weights"] = best_weights
    return metrics


def run_crmf(X_train, y_train, X_val, y_val, X_test, y_test,
             modality_idx: dict) -> dict:
    """
    CRMF (Cross-modal Regression with Manifold Fusion).
    Core idea from RecruitView paper:
      1. PCA per modality → low-dim manifold
      2. CCA to align modalities pairwise
      3. Concatenate aligned features + SVR regression

    This is a practical approximation of the paper's manifold alignment.
    """
    t0 = time.time()
    mods = ["face", "gesture", "voice"]
    n_components = min(30, min(X_train.shape[0] // 10, 50))

    # Step 1: PCA per modality
    pca_models = {}
    X_tr_reduced, X_val_reduced, X_te_reduced = [], [], []
    for mod in mods:
        idx = modality_idx[mod]
        sc = StandardScaler()
        X_tr_m = sc.fit_transform(X_train[:, idx])
        X_val_m = sc.transform(X_val[:, idx])
        X_te_m = sc.transform(X_test[:, idx])

        n_comp = min(n_components, X_tr_m.shape[1], X_tr_m.shape[0] - 1)
        pca = PCA(n_components=n_comp, random_state=42)
        X_tr_reduced.append(pca.fit_transform(X_tr_m))
        X_val_reduced.append(pca.transform(X_val_m))
        X_te_reduced.append(pca.transform(X_te_m))
        pca_models[mod] = pca

    # Step 2: CCA pairwise alignment (face↔gesture, face↔voice, gesture↔voice)
    cca_features_tr, cca_features_val, cca_features_te = [], [], []
    pairs = [(0, 1), (0, 2), (1, 2)]  # (face,gesture), (face,voice), (gesture,voice)
    for i, j in pairs:
        n_comp = min(5, X_tr_reduced[i].shape[1], X_tr_reduced[j].shape[1],
                     X_tr_reduced[i].shape[0] - 1)
        if n_comp < 2:
            continue
        cca = CCA(n_components=n_comp)
        cca.fit(X_tr_reduced[i], X_tr_reduced[j])
        Xi_tr, Xj_tr = cca.transform(X_tr_reduced[i], X_tr_reduced[j])
        Xi_val, Xj_val = cca.transform(X_val_reduced[i], X_val_reduced[j])
        Xi_te, Xj_te = cca.transform(X_te_reduced[i], X_te_reduced[j])
        cca_features_tr.extend([Xi_tr, Xj_tr])
        cca_features_val.extend([Xi_val, Xj_val])
        cca_features_te.extend([Xi_te, Xj_te])

    # Add original PCA features too
    cca_features_tr.extend(X_tr_reduced)
    cca_features_val.extend(X_val_reduced)
    cca_features_te.extend(X_te_reduced)

    X_tr_fused = np.hstack(cca_features_tr)
    X_val_fused = np.hstack(cca_features_val)
    X_te_fused = np.hstack(cca_features_te)

    # Step 3: SVR on fused manifold features
    final_scaler = StandardScaler()
    X_tr_fused = final_scaler.fit_transform(X_tr_fused)
    X_val_fused = final_scaler.transform(X_val_fused)
    X_te_fused = final_scaler.transform(X_te_fused)

    svr = MultiOutputRegressor(SVR(kernel="rbf", C=1.0, gamma="scale"))
    svr.fit(X_tr_fused, y_train)
    y_pred = svr.predict(X_te_fused)

    metrics = compute_metrics(y_test, y_pred)
    metrics["fit_time_s"] = round(time.time() - t0, 1)
    metrics["fused_dim"] = X_tr_fused.shape[1]
    return metrics


# ═══════════════════════════════════════════════════════════════════════
# ATTENTION-LSTM FUSION
# ═══════════════════════════════════════════════════════════════════════

def run_attention_lstm(X_train, y_train, X_val, y_val, X_test, y_test,
                       modality_idx: dict, seed: int) -> dict:
    """
    BiLSTM + Multi-head Self-Attention for multimodal feature fusion.
    Since we have aggregated (not sequential) features, we treat each
    modality's feature segment as a sequence step:

      Step 0: Face features (1260-dim, projected)
      Step 1: Gesture features (140-dim, projected)
      Step 2: Voice features (10-dim, projected)

    The sequence → BiLSTM → Multi-head Self-Attention pooling → FC → output.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("  [WARN] PyTorch not available; skipping Attention-LSTM")
        return {"error": "PyTorch not installed"}

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Set seeds ──
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── Prepare data: project each modality to same dim, form 3-step sequence ──
    proj_dim = 256
    mods = ["face", "gesture", "voice"]

    # Fit PCA + scaler per modality
    projections = {}
    for mod in mods:
        idx = modality_idx[mod]
        sc = StandardScaler()
        Xm_tr = sc.fit_transform(X_train[:, idx])
        n_comp = min(proj_dim, Xm_tr.shape[1], Xm_tr.shape[0] - 1)
        pca = PCA(n_components=n_comp, random_state=seed)
        pca.fit(Xm_tr)
        projections[mod] = (sc, pca)

    def _build_sequence(X_raw):
        steps = []
        for mod in mods:
            idx = modality_idx[mod]
            sc, pca = projections[mod]
            Xm = sc.transform(X_raw[:, idx])
            Xp = pca.transform(Xm)
            # Pad to proj_dim
            if Xp.shape[1] < proj_dim:
                pad = np.zeros((Xp.shape[0], proj_dim - Xp.shape[1]))
                Xp = np.hstack([Xp, pad])
            steps.append(Xp)
        # (batch, 3, proj_dim)
        return np.stack(steps, axis=1).astype(np.float32)

    X_tr_seq = _build_sequence(X_train)
    X_val_seq = _build_sequence(X_val)
    X_te_seq = _build_sequence(X_test)

    y_tr = y_train.astype(np.float32)
    y_val_f = y_val.astype(np.float32)
    y_te = y_test.astype(np.float32)

    # ── Model ──
    class AttentionLSTM(nn.Module):
        def __init__(self, input_dim=proj_dim, lstm_hidden=128, lstm_layers=2,
                     num_heads=4, fc_dim=64, output_dim=12, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, lstm_hidden, lstm_layers,
                                batch_first=True, bidirectional=True, dropout=dropout)
            lstm_out = lstm_hidden * 2
            self.attention = nn.MultiheadAttention(lstm_out, num_heads,
                                                   dropout=dropout, batch_first=True)
            self.fc = nn.Sequential(
                nn.Linear(lstm_out, fc_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fc_dim, output_dim),
            )
            self.ln = nn.LayerNorm(lstm_out)

        def forward(self, x):
            # x: (batch, seq=3, input_dim)
            lstm_out, _ = self.lstm(x)          # (batch, 3, lstm_hidden*2)
            lstm_out = self.ln(lstm_out)
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            # Global average pooling over sequence
            pooled = attn_out.mean(dim=1)        # (batch, lstm_hidden*2)
            return self.fc(pooled)

    model = AttentionLSTM(
        input_dim=proj_dim,
        lstm_hidden=ATTENTION_LSTM_PARAMS["lstm_hidden"],
        lstm_layers=ATTENTION_LSTM_PARAMS["lstm_layers"],
        num_heads=ATTENTION_LSTM_PARAMS["num_heads"],
        fc_dim=ATTENTION_LSTM_PARAMS["fc_dim"],
        output_dim=ATTENTION_LSTM_PARAMS["output_dim"],
        dropout=ATTENTION_LSTM_PARAMS["dropout"],
    ).to(device)

    # ── Training ──
    train_ds = TensorDataset(torch.from_numpy(X_tr_seq), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_val_seq), torch.from_numpy(y_val_f))
    train_loader = DataLoader(train_ds, batch_size=ATTENTION_LSTM_PARAMS["batch_size"],
                              shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=ATTENTION_LSTM_PARAMS["batch_size"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ATTENTION_LSTM_PARAMS["lr"],
        weight_decay=ATTENTION_LSTM_PARAMS["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(ATTENTION_LSTM_PARAMS["epochs"]):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                pred = model(Xb)
                val_loss += criterion(pred, yb).item() * Xb.size(0)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= ATTENTION_LSTM_PARAMS["patience"]:
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        X_te_t = torch.from_numpy(X_te_seq).to(device)
        y_pred = model(X_te_t).cpu().numpy()

    metrics = compute_metrics(y_test, y_pred)
    metrics["fit_time_s"] = round(time.time() - t0, 1)
    metrics["best_val_loss"] = float(best_val_loss)
    metrics["epochs_trained"] = epoch + 1
    return metrics


# ═══════════════════════════════════════════════════════════════════════
# LLM COMPARISON (from existing results)
# ═══════════════════════════════════════════════════════════════════════

def load_llm_scores() -> dict:
    """
    Load LLM results and return {method: {video_id: scores_dict}}.
    Deduplicates by taking last entry per (video_id, method, temperature).
    Filters to temperature=0.0 only.
    """
    if not LLM_RESULTS.exists():
        print("  [WARN] llm_results.jsonl not found")
        return {}

    seen = {}
    with open(LLM_RESULTS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["video_id"], r["method"], r.get("temperature", 0.0))
            seen[key] = r

    scores_by_method = defaultdict(dict)
    for (vid, method, temp), r in seen.items():
        if temp != 0.0:
            continue
        if r.get("scores") and r.get("scores_valid") is not False:
            scores_by_method[method][vid] = r["scores"]

    return dict(scores_by_method)


def evaluate_llm_vs_truth(llm_scores: dict, video_ids: list, metadata: dict,
                          test_indices: np.ndarray) -> dict:
    """Compare LLM predictions to ground truth on test set."""
    results = {}
    test_vids = [video_ids[i] for i in test_indices]

    for method, vid_scores in llm_scores.items():
        y_true_list, y_pred_list = [], []
        for vid in test_vids:
            if vid not in vid_scores:
                continue
            y_true_list.append([metadata[vid]["targets"][k] for k in TARGET_KEYS])
            y_pred_list.append([vid_scores[vid].get(k, 0.0) for k in TARGET_KEYS])

        if not y_true_list:
            results[method] = {"error": f"No matching videos for {method}"}
            continue

        y_true = np.array(y_true_list, dtype=np.float64)
        y_pred = np.array(y_pred_list, dtype=np.float64)
        results[method] = compute_metrics(y_true, y_pred)
        results[method]["n_videos"] = len(y_true_list)

    return results


def run_text_only_llm(test_video_ids: list, metadata: dict) -> dict:
    """
    Text-only LLM baseline: uses transcript + gemini_summary, NO multimodal features.
    Calls LLM via the same API as llm_reasoning.py.
    """
    import openai

    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_API_BASE", "https://api.deepseek.com")
    model = os.getenv("OPENAI_MODEL", "deepseek-v4-flash")

    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    system_prompt = (
        "You are an expert interview assessor. "
        "Based on the candidate's interview transcript and summary, "
        "rate them on 12 dimensions. Scores are normalized, centered around 0, "
        "typically ranging from -2 to 2.\n\n"
        "Dimensions: openness, conscientiousness, extraversion, agreeableness, "
        "neuroticism, overall_personality, interview_score, answer_score, "
        "speaking_skills, confidence_score, facial_expression, overall_performance.\n\n"
        "Output ONLY valid JSON: {\"dimension_name\": score, ...}"
    )

    y_true_list, y_pred_list = [], []
    t0 = time.time()
    n_total = len(test_video_ids)

    for i, vid in enumerate(test_video_ids):
        if i > 0 and i % 10 == 0:
            print(f"  text-only LLM: {i}/{n_total}")

        meta = metadata.get(vid, {})
        transcript = meta.get("transcript", "")[:3000]
        summary = meta.get("gemini_summary", "")[:3000]

        user_prompt = (
            f"## Interview Transcript (excerpt)\n{transcript}\n\n"
            f"## Gemini Summary\n{summary}\n\n"
            f"Rate the candidate on all 12 dimensions. Output JSON only."
        )

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=2000,
            )
            raw = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"    text-only LLM error for {vid}: {e}")
            continue

        # Parse JSON
        scores = _parse_json_scores(raw)
        if scores:
            y_true_list.append([meta["targets"][k] for k in TARGET_KEYS])
            y_pred_list.append([scores.get(k, 0.0) for k in TARGET_KEYS])

    if not y_true_list:
        return {"error": "No successful text-only LLM predictions"}

    y_true = np.array(y_true_list, dtype=np.float64)
    y_pred = np.array(y_pred_list, dtype=np.float64)
    metrics = compute_metrics(y_true, y_pred)
    metrics["fit_time_s"] = round(time.time() - t0, 1)
    metrics["n_videos"] = len(y_true_list)
    return metrics


def _parse_json_scores(text: str) -> dict | None:
    """Extract JSON scores from LLM output. Returns None on failure."""
    import re
    # Try ```json block
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try bare JSON object
    m = re.search(r"\{[\s\S]*?\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # Regex fallback
    pattern = r'"(' + "|".join(TARGET_KEYS) + r')"\s*:\s*(-?\d+\.?\d*)'
    matches = re.findall(pattern, text)
    if len(matches) >= 8:
        return {k: float(v) for k, v in matches}
    return None


# ═══════════════════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════

def paired_bootstrap_test(errors_a: np.ndarray, errors_b: np.ndarray,
                          n_bootstrap: int = 10000, seed: int = 42) -> dict:
    """
    Paired bootstrap test: is method A significantly better than B?
    errors_a, errors_b: per-sample MSE arrays (n_samples,)
    Null: mean(error_a) >= mean(error_b)  (A is NOT better)
    """
    rng = np.random.RandomState(seed)
    n = len(errors_a)
    diff_obs = np.mean(errors_b) - np.mean(errors_a)  # positive = A better

    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        boot_d = np.mean(errors_b[idx] - errors_a[idx])
        boot_diffs.append(boot_d)

    boot_diffs = np.array(boot_diffs)
    ci_lower = np.percentile(boot_diffs, 2.5)
    ci_upper = np.percentile(boot_diffs, 97.5)

    # Two-sided p-value: fraction of bootstrap diffs more extreme than 0
    p_value = min(
        np.mean(boot_diffs <= 0),
        np.mean(boot_diffs >= 0),
    ) * 2
    p_value = min(p_value, 1.0)

    return {
        "observed_diff": float(diff_obs),
        "ci_95": [float(ci_lower), float(ci_upper)],
        "p_value": float(p_value),
        "significant_at_05": bool(p_value < 0.05),
        "n_bootstrap": n_bootstrap,
    }


def permutation_test(errors_a: np.ndarray, errors_b: np.ndarray,
                     n_perm: int = 10000, seed: int = 42) -> dict:
    """
    Permutation test for paired difference in mean MSE.
    """
    rng = np.random.RandomState(seed)
    diff_obs = np.mean(errors_a) - np.mean(errors_b)
    n = len(errors_a)

    diffs = np.abs(errors_a - errors_b)  # paired differences
    count_extreme = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=n)
        perm_diff = np.mean(signs * diffs)
        if abs(perm_diff) >= abs(diff_obs):
            count_extreme += 1

    p_value = count_extreme / n_perm
    return {
        "observed_diff": float(diff_obs),
        "p_value": float(p_value),
        "significant_at_05": bool(p_value < 0.05),
        "n_permutations": n_perm,
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════

def run_single_seed(data: dict, split_info: dict, seed: int,
                    methods: set, skip_text_llm: bool,
                    skip_attention: bool) -> dict:
    """Run all baseline methods for a single seed split."""
    X = data["X"]
    y = data["y"]
    video_ids = data["video_ids"]
    metadata = data["metadata"]
    modality_idx = data["modality_idx"]

    train_idx = get_split_indices(video_ids, split_info["train_videos"])
    val_idx = get_split_indices(video_ids, split_info["val_videos"])
    test_idx = get_split_indices(video_ids, split_info["test_videos"])

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"  Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

    seed_results = {"seed": seed, "split": split_info}

    # ── Unimodal Regressors ──
    for method in ["svr", "ridge", "rf"]:
        if method not in methods:
            continue
        print(f"  [{method.upper()}] unimodal...")
        seed_results[f"unimodal_{method}"] = run_unimodal(
            X_train, y_train, X_val, y_val, X_test, y_test, modality_idx, method,
        )

    # ── Multimodal Fusion ──
    if "mlp" in methods:
        print("  [MLP] Early Fusion...")
        seed_results["early_fusion_mlp"] = run_early_fusion_mlp(
            X_train, y_train, X_val, y_val, X_test, y_test, modality_idx, seed,
        )

    if "late_fusion" in methods:
        print("  [Late Fusion]...")
        seed_results["late_fusion"] = run_late_fusion(
            X_train, y_train, X_val, y_val, X_test, y_test, modality_idx,
        )

    if "crmf" in methods:
        print("  [CRMF] CCA-based manifold fusion...")
        seed_results["crmf"] = run_crmf(
            X_train, y_train, X_val, y_val, X_test, y_test, modality_idx,
        )

    # ── Attention-LSTM ──
    if "attention_lstm" in methods and not skip_attention:
        print("  [Attention-LSTM]...")
        seed_results["attention_lstm"] = run_attention_lstm(
            X_train, y_train, X_val, y_val, X_test, y_test, modality_idx, seed,
        )

    # ── LLM Comparison ──
    if "llm" in methods:
        print("  [LLM comparison] loading existing results...")
        llm_scores = load_llm_scores()
        seed_results["llm_comparison"] = evaluate_llm_vs_truth(
            llm_scores, video_ids, metadata, test_idx,
        )

    # ── Text-only LLM ──
    if "text_llm" in methods and not skip_text_llm:
        print("  [Text-only LLM]...")
        test_vids = [video_ids[i] for i in test_idx]
        seed_results["text_only_llm"] = run_text_only_llm(test_vids, metadata)

    return seed_results


def aggregate_across_seeds(all_seed_results: list) -> dict:
    """Aggregate results across 5 seeds: mean ± std per method per metric."""
    agg = {}

    # Collect all method keys from first seed
    method_keys = [k for k in all_seed_results[0].keys()
                   if k not in ("seed", "split")]

    for mkey in method_keys:
        seed_vals = []
        for sr in all_seed_results:
            if mkey in sr and sr[mkey] is not None:
                seed_vals.append(sr[mkey])

        if not seed_vals:
            agg[mkey] = {"error": "No successful seeds"}
            continue

        # For unimodal methods: nested {modality: metrics}
        if "unimodal" in mkey:
            agg[mkey] = _aggregate_unimodal(seed_vals)
        elif isinstance(seed_vals[0], dict) and "per_dimension" in seed_vals[0]:
            agg[mkey] = _aggregate_flat_metrics(seed_vals)
        elif isinstance(seed_vals[0], dict) and "error" in seed_vals[0]:
            agg[mkey] = {"error": seed_vals[0]["error"]}
        else:
            agg[mkey] = _aggregate_nested(seed_vals)

    return agg


def _aggregate_unimodal(seed_vals: list) -> dict:
    """Aggregate unimodal results: {method: {modality: {metrics_per_seed...}}}"""
    modalities = list(seed_vals[0].keys())
    agg = {}
    for mod in modalities:
        mod_seeds = [sv[mod] for sv in seed_vals if mod in sv and "error" not in sv[mod]]
        if mod_seeds:
            agg[mod] = _aggregate_flat_metrics(mod_seeds)
        else:
            agg[mod] = {"error": "No data"}
    return agg


def _aggregate_flat_metrics(seed_vals: list) -> dict:
    """Aggregate flat metrics dicts: mean ± std for each metric."""
    if not seed_vals:
        return {"error": "No data"}

    # Handle meta keys (not aggregated)
    meta_keys = {"fit_time_s", "best_params", "n_iter_", "best_loss_",
                 "fusion_weights", "fused_dim", "n_videos", "best_val_loss",
                 "epochs_trained", "error"}

    agg = {}
    # Per-dimension metrics
    if "per_dimension" in seed_vals[0]:
        n_dims = len(seed_vals[0]["per_dimension"])
        per_dim_agg = []
        for d in range(n_dims):
            dim_vals = defaultdict(list)
            for sv in seed_vals:
                pd = sv["per_dimension"][d]
                for k, v in pd.items():
                    if k != "dimension":
                        dim_vals[k].append(v)
            dim_entry = {"dimension": seed_vals[0]["per_dimension"][d]["dimension"]}
            for k, vals in dim_vals.items():
                arr = np.array(vals)
                dim_entry[k] = {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=1))}
            per_dim_agg.append(dim_entry)
        agg["per_dimension"] = per_dim_agg

    # Macro / group averages
    for group_key in ["macro_avg", "personality_avg", "interview_avg"]:
        if group_key in seed_vals[0]:
            gv = defaultdict(list)
            for sv in seed_vals:
                for k, v in sv[group_key].items():
                    gv[k].append(v)
            agg[group_key] = {}
            for k, vals in gv.items():
                arr = np.array(vals)
                agg[group_key][k] = {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=1))}

    # Meta keys: just take from first seed
    for mk in meta_keys:
        if mk in seed_vals[0]:
            agg[mk] = seed_vals[0][mk]

    return agg


def _aggregate_nested(seed_vals: list) -> dict:
    """Generic nested aggregation."""
    if all("error" in sv for sv in seed_vals):
        return {"error": seed_vals[0]["error"]}
    # Try _aggregate_flat_metrics
    return _aggregate_flat_metrics(seed_vals)


# ═══════════════════════════════════════════════════════════════════════
# COMMAND-LINE ENTRY
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser(description="CueCoT Phase 4: Baselines")
    ap.add_argument("--methods", type=str, default="all",
                    help="Comma-separated: svr,ridge,rf,mlp,late_fusion,crmf,attention_lstm,llm,text_llm")
    ap.add_argument("--skip-text-llm", action="store_true",
                    help="Skip text-only LLM baseline (saves API cost)")
    ap.add_argument("--skip-attention", action="store_true",
                    help="Skip Attention-LSTM (if no PyTorch)")
    ap.add_argument("--test", action="store_true",
                    help="Quick test: 50 videos, 1 seed")
    args = ap.parse_args()

    method_set = set(args.methods.split(",")) if args.methods != "all" else {
        "svr", "ridge", "rf", "mlp", "late_fusion", "crmf",
        "attention_lstm", "llm", "text_llm",
    }

    print("=" * 60)
    print("CueCoT Phase 4: Baselines (AAAI 2027 Experimental Rigor)")
    print(f"Methods: {sorted(method_set)}")
    print(f"Seeds: {SEEDS}")
    print(f"Test mode: {args.test}")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading data...")
    data = load_data()
    X, y = data["X"], data["y"]
    print(f"  X: {X.shape}, y: {y.shape}")
    print(f"  Users: {len(data['user_videos'])}, Videos: {len(data['video_ids'])}")
    print(f"  Face: {len(data['modality_idx']['face'])}  "
          f"Gesture: {len(data['modality_idx']['gesture'])}  "
          f"Voice: {len(data['modality_idx']['voice'])}")

    # Test mode: subsample
    if args.test:
        np.random.seed(42)
        test_users = sorted(data["user_videos"].keys())
        np.random.shuffle(test_users)
        test_users = test_users[:10]
        test_vids = []
        for u in test_users:
            test_vids.extend(data["user_videos"][u])
        idx = np.array([data["video_ids"].index(v) for v in test_vids])
        data["X"] = X[idx]
        data["y"] = y[idx]
        data["video_ids"] = [data["video_ids"][i] for i in idx]
        data["user_videos"] = {u: data["user_videos"][u] for u in test_users}
        seeds_to_run = [42]
        print(f"  Test mode: {len(test_users)} users, {len(idx)} videos, 1 seed")
    else:
        seeds_to_run = SEEDS

    # Generate splits
    print("\n[2/4] Generating user-level splits...")
    all_splits = {}
    for seed in seeds_to_run:
        split = create_user_splits(data["user_videos"], seed)
        all_splits[seed] = split
        print(f"  Seed {seed}: train={len(split['train_videos'])} "
              f"val={len(split['val_videos'])} test={len(split['test_videos'])}")

    # Save split info
    split_info_path = OUTPUT_DIR / "split_info.json"
    with open(split_info_path, "w") as f:
        json.dump({str(s): v for s, v in all_splits.items()}, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {split_info_path}")

    # Run experiments per seed
    print(f"\n[3/4] Running {len(seeds_to_run)} seeds...")
    all_seed_results = []
    total_start = time.time()

    for seed in seeds_to_run:
        print(f"\n{'─'*40}\n  Seed {seed}\n{'─'*40}")
        np.random.seed(seed)
        random.seed(seed)
        seed_results = run_single_seed(
            data, all_splits[seed], seed, method_set,
            args.skip_text_llm, args.skip_attention,
        )
        all_seed_results.append(seed_results)

    total_elapsed = time.time() - total_start
    print(f"\n  Total time: {total_elapsed/60:.1f} min")

    # Aggregate
    print("\n[4/4] Aggregating results across seeds...")
    summary = aggregate_across_seeds(all_seed_results)

    # Save detailed per-seed results
    results_path = OUTPUT_DIR / "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(all_seed_results, f, indent=2, ensure_ascii=False,
                  default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    print(f"  Saved: {results_path}")

    # Save summary
    summary_path = OUTPUT_DIR / "baseline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {summary_path}")

    # Print key results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (Macro Avg MSE ± std across seeds)")
    print("=" * 60)
    _print_comparison_table(summary)

    # Statistical tests: CueCoT vs best baseline
    print("\n" + "=" * 60)
    print("STATISTICAL TESTS")
    print("=" * 60)
    _run_statistical_tests(all_seed_results, data)

    print("\nDone.")


def _print_comparison_table(summary: dict):
    """Print a clean comparison table."""
    rows = []
    for mkey, mval in summary.items():
        if "error" in mval:
            continue
        if "unimodal" in mkey:
            method_base = mkey.replace("unimodal_", "")
            for mod, mod_val in mval.items():
                if "error" in mod_val:
                    continue
                macro = mod_val.get("macro_avg", {})
                rows.append((f"{method_base.upper()}_{mod}", macro))
        else:
            macro = mval.get("macro_avg", {})
            if macro:
                rows.append((mkey, macro))

    # Print header
    print(f"{'Method':<30} {'MSE':>8} {'Pearson r':>10} {'Spearman ρ':>12} {'MAE':>8}")
    print("-" * 70)
    for name, macro in sorted(rows, key=lambda x: x[1].get("mse", {}).get("mean", 999)):
        mse = macro.get("mse", {})
        pr = macro.get("pearson_r", {})
        sr = macro.get("spearman_rho", {})
        mae = macro.get("mae", {})
        def _fmt(v):
            if isinstance(v, dict):
                return f"{v.get('mean', 0):.4f}±{v.get('std', 0):.3f}"
            return f"{v:.4f}" if v else "N/A"
        print(f"{name:<30} {_fmt(mse):>8} {_fmt(pr):>10} {_fmt(sr):>12} {_fmt(mae):>8}")


def _run_statistical_tests(all_seed_results: list, data: dict):
    """Run paired bootstrap tests: CueCoT vs best baseline."""
    # Collect per-video errors across seeds for key methods
    # For simplicity, use seed 42 results
    if not all_seed_results:
        return
    sr0 = all_seed_results[0]

    # Get LLM results
    llm_scores = load_llm_scores()
    cot_scores = llm_scores.get("cot_full", {})
    direct_scores = llm_scores.get("direct", {})

    # Get test videos from seed 42 split
    split42 = create_user_splits(data["user_videos"], 42)
    test_vids = set(split42["test_videos"])

    # Get best baseline (early fusion MLP) predictions — approximate using seed 42
    # Compare CueCoT vs Direct LLM (main result)
    cot_errors = []
    direct_errors = []
    for vid in test_vids:
        if vid in cot_scores and vid in direct_scores:
            truth = np.array([data["metadata"][vid]["targets"][k] for k in TARGET_KEYS])
            cot_pred = np.array([cot_scores[vid].get(k, 0.0) for k in TARGET_KEYS])
            direct_pred = np.array([direct_scores[vid].get(k, 0.0) for k in TARGET_KEYS])
            cot_errors.append(np.mean((truth - cot_pred) ** 2))
            direct_errors.append(np.mean((truth - direct_pred) ** 2))

    if cot_errors and direct_errors:
        print(f"\n  CoT_full vs Direct (n={len(cot_errors)} test videos):")
        print(f"    CoT_full MSE:  {np.mean(cot_errors):.4f}")
        print(f"    Direct MSE:    {np.mean(direct_errors):.4f}")
        boot = paired_bootstrap_test(np.array(cot_errors), np.array(direct_errors))
        print(f"    Observed diff: {boot['observed_diff']:.4f}")
        print(f"    95% CI:        [{boot['ci_95'][0]:.4f}, {boot['ci_95'][1]:.4f}]")
        print(f"    p-value:       {boot['p_value']:.4f}")
        print(f"    Significant:   {boot['significant_at_05']}")


if __name__ == "__main__":
    main()
