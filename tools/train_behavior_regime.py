#!/usr/bin/env python3
"""训练并验证《衡才智析》无监督行为节奏分段模型（纯 CPU + GroupKFold）。

训练边界：
- 仅使用 `data/logs/` 下真实会话的 L0 面部/手势运动学特征与缓存语音浊音序列（零受试者姓名/学号/文本）；
- 按 `session_id` 分组执行 5 折 `GroupKFold`，评估训练集与验证集 Silhouette 系数、
  Davies-Bouldin 指数，以及各折确定性映射后状态标签相对于全量模型的 ARI/NMI 稳定性；
- 产出紧凑模型 `models/behavior_regime_model.joblib` 与零 PII 验证指标
  `report_frontend/behavior_regime_metrics.json`。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from report_frontend.behavior_regime_model import (
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    FEATURE_NAMES,
    STATE_META,
    STATE_ORDER,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    assign_cluster_states,
    extract_regime_windows,
)
from report_frontend.data_loader import LogDataLoader
from report_frontend.voice_series import load_series


def _build_preprocessor() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("power", PowerTransformer(method="yeo-johnson")),
        ("scaler", StandardScaler()),
    ])


def _build_gmm(n_components: int = 4) -> GaussianMixture:
    return GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        reg_covar=0.1,
        n_init=5,
        random_state=42,
    )


def build_dataset(log_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, int, int]:
    loader = LogDataLoader(str(log_dir) if log_dir else None)
    groups = loader._scan_and_group_files()
    face_by_sid = {f["session_id"]: f["path"] for f in groups["face"]}
    gest_by_sid = {f["session_id"]: f["path"] for f in groups["gesture"]}

    all_sids = sorted(set(face_by_sid) | set(gest_by_sid))
    X_list: list[np.ndarray] = []
    g_list: list[np.ndarray] = []
    used_sessions = 0
    sessions_with_voice = 0

    for sid in all_sids:
        fdf = None
        gdf = None
        if sid in face_by_sid:
            try:
                fdf = pd.read_csv(face_by_sid[sid], low_memory=False)
            except Exception:
                fdf = None
        if sid in gest_by_sid:
            try:
                gdf = pd.read_csv(gest_by_sid[sid], low_memory=False)
            except Exception:
                gdf = None

        vs = load_series(sid)
        X_s, _ = extract_regime_windows(
            fdf,
            gdf,
            voice_series=vs,
            window_size=WINDOW_SIZE,
            stride=WINDOW_STRIDE,
        )
        if len(X_s) == 0:
            continue

        X_list.append(X_s)
        g_list.append(np.full(len(X_s), sid, dtype=object))
        used_sessions += 1
        if np.isfinite(X_s[:, -1]).any():
            sessions_with_voice += 1

    if not X_list:
        raise RuntimeError("未找到可用于训练行为节奏分段模型的真实 L0 会话日志")

    return np.vstack(X_list), np.concatenate(g_list), used_sessions, sessions_with_voice


def train_and_save(
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    X, groups, n_sessions, n_sessions_voice = build_dataset(log_dir=log_dir)
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    if n_splits < 2:
        raise RuntimeError(f"有效会话数不足以执行 GroupKFold（当前 {len(unique_groups)} 场）")

    # 先拟合全量模型并执行确定性物理状态映射，作为跨折稳定性参照
    full_prep = _build_preprocessor()
    X_full_z = full_prep.fit_transform(X)
    full_gmm = _build_gmm(n_components=4).fit(X_full_z)
    full_cluster_ids = full_gmm.predict(X_full_z)
    full_mapping = assign_cluster_states(full_gmm.means_, FEATURE_NAMES)
    full_states = np.asarray([full_mapping[int(c)] for c in full_cluster_ids], dtype=object)

    full_sil = float(silhouette_score(X_full_z, full_cluster_ids))
    full_db = float(davies_bouldin_score(X_full_z, full_cluster_ids))

    # 5 折按 session_id 分组交叉验证
    gkf = GroupKFold(n_splits=n_splits)
    train_sils, val_sils, val_dbs, aris, nmis, agreements = [], [], [], [], [], []

    for tr_idx, va_idx in gkf.split(X, groups=groups):
        fold_prep = _build_preprocessor()
        X_tr = fold_prep.fit_transform(X[tr_idx])
        X_va = fold_prep.transform(X[va_idx])

        fold_gmm = _build_gmm(n_components=4).fit(X_tr)
        tr_cids = fold_gmm.predict(X_tr)
        va_cids = fold_gmm.predict(X_va)

        if len(np.unique(tr_cids)) > 1:
            train_sils.append(float(silhouette_score(X_tr, tr_cids)))
        if len(np.unique(va_cids)) > 1:
            val_sils.append(float(silhouette_score(X_va, va_cids)))
            val_dbs.append(float(davies_bouldin_score(X_va, va_cids)))

        fold_mapping = assign_cluster_states(fold_gmm.means_, FEATURE_NAMES)
        va_states = np.asarray([fold_mapping[int(c)] for c in va_cids], dtype=object)

        aris.append(float(adjusted_rand_score(full_states[va_idx], va_states)))
        nmis.append(float(normalized_mutual_info_score(full_states[va_idx], va_states)))
        agreements.append(float((va_states == full_states[va_idx]).mean()))

    val_sil_mean = float(np.mean(val_sils)) if val_sils else full_sil
    if val_sil_mean < 0.15:
        raise RuntimeError(f"验证集 Silhouette 过低（{val_sil_mean:.4f} < 0.15），请检查特征或聚类配置")

    # 测量纯 CPU 推理耗时（30 窗典型片段）
    sample_batch = X[: min(30, len(X))]
    t_inf0 = time.perf_counter()
    for _ in range(30):
        Xs_b = full_prep.transform(sample_batch)
        full_gmm.predict(Xs_b)
    inf_ms_30 = ((time.perf_counter() - t_inf0) / 30.0) * 1000.0

    # 保存紧凑产物
    centroids_z_by_state: dict[str, dict[str, float]] = {}
    for cid, state in sorted(full_mapping.items(), key=lambda kv: STATE_ORDER.index(kv[1])):
        centroids_z_by_state[state] = {
            fname: round(float(full_gmm.means_[cid, j]), 4)
            for j, fname in enumerate(FEATURE_NAMES)
        }

    artifact = {
        "version": "1.0.0",
        "feature_names": list(FEATURE_NAMES),
        "window_size": WINDOW_SIZE,
        "stride": WINDOW_STRIDE,
        "preprocessor": full_prep,
        "gmm": full_gmm,
        "cluster_to_state": full_mapping,
        "state_meta": STATE_META,
        "centroids_z_by_state": centroids_z_by_state,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path, compress=3)
    model_kb = round(model_path.stat().st_size / 1024.0, 1)

    n_windows = int(len(X))
    state_distribution: dict[str, dict[str, Any]] = {}
    for st in STATE_ORDER:
        cnt = int((full_states == st).sum())
        state_distribution[st] = {
            "label": STATE_META[st]["label"],
            "count": cnt,
            "ratio": round(cnt / float(n_windows), 4),
        }

    metrics: dict[str, Any] = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_scope": "unsupervised_behavioral_regime_segmentation_only",
        "scoring_participation": False,
        "model_type": type(full_gmm).__name__,
        "n_components": int(full_gmm.n_components),
        "covariance_type": full_gmm.covariance_type,
        "reg_covar": float(full_gmm.reg_covar),
        "model_size_kb": model_kb,
        "window_size": WINDOW_SIZE,
        "stride": WINDOW_STRIDE,
        "n_sessions": int(n_sessions),
        "n_sessions_with_voice_series": int(n_sessions_voice),
        "n_windows": n_windows,
        "feature_names": list(FEATURE_NAMES),
        "cv_scheme": f"GroupKFold(n_splits={n_splits}, group=session_id)",
        "full_silhouette": round(full_sil, 4),
        "full_davies_bouldin": round(full_db, 4),
        "train_silhouette_mean": round(float(np.mean(train_sils)), 4) if train_sils else None,
        "val_silhouette_mean": round(val_sil_mean, 4),
        "val_silhouette_std": round(float(np.std(val_sils)), 4) if val_sils else None,
        "val_davies_bouldin_mean": round(float(np.mean(val_dbs)), 4) if val_dbs else None,
        "cluster_stability_ari_mean": round(float(np.mean(aris)), 4),
        "cluster_stability_nmi_mean": round(float(np.mean(nmis)), 4),
        "state_agreement_mean": round(float(np.mean(agreements)), 4),
        "cpu_inference_ms_per_30_windows": round(inf_ms_30, 3),
        "inference_ms": round(inf_ms_30, 3),
        "train_elapsed_sec": round(time.perf_counter() - t0, 2),
        "state_distribution": state_distribution,
        "cluster_centroids_z": centroids_z_by_state,
        "boundary_note": (
            "本模型仅在纯 CPU 下对多模态外显动作与发声节律做无监督区间切分，"
            "不代表任何个人特质，也不产出任何数值汇总。"
        ),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    out = train_and_save()
    print(json.dumps(out, ensure_ascii=False, indent=2))
