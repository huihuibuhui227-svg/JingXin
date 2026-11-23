#!/usr/bin/env python3
"""训练并验证《衡才智析》自监督采集信号质量异常检测模型（纯 CPU + GroupKFold）。

训练边界（严格遵守红线）：
- 仅使用 `data/logs/` 下真实会话的 L0 物理信号统计量（零受试者姓名/学号/文本）；
- 正样本：真实高质量采集滑动窗；负样本：真实低质量窗 + 4 类物理故障合成注入
  （镜头遮挡丢帧、画面缓存冻结、时间戳抖动、传感器尖峰噪点）；
- 交叉验证严格按 `session_id` 分组（GroupKFold），确保同一场数据绝不跨折泄漏；
- 产出紧凑模型 `models/signal_quality_clf.joblib` 与零 PII 验证指标
  `report_frontend/signal_quality_metrics.json`。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from report_frontend.data_loader import LogDataLoader
from report_frontend.signal_quality_clf import (
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    FEATURE_NAMES,
    extract_window_features,
)


def _inject_faults(
    face_df: pd.DataFrame,
    gesture_df: pd.DataFrame | None,
    rng: np.random.Generator,
) -> list[tuple[pd.DataFrame, pd.DataFrame | None]]:
    """构造 4 类物理采集故障副本用于自监督信号质检训练。"""
    faults: list[tuple[pd.DataFrame, pd.DataFrame | None]] = []
    au_cols = [c for c in face_df.columns if c.startswith("au") and "_" in c]
    gaze_cols = [c for c in ("gaze_direction_x", "gaze_direction_y") if c in face_df.columns]

    # 1. 摄像头遮挡 / 连续丢检（65%~90% 帧置 NaN）
    f_occ = face_df.copy()
    mask = rng.random(len(f_occ)) < rng.uniform(0.65, 0.90)
    for c in au_cols + gaze_cols:
        f_occ.loc[mask, c] = np.nan
    faults.append((f_occ, gesture_df))

    # 2. 画面冻结 / 缓存卡死（整段重复首帧值，时序差分为 0）
    f_frz = face_df.copy()
    if len(f_frz) > 1:
        for c in au_cols + gaze_cols:
            first_val = pd.to_numeric(f_frz[c], errors="coerce").dropna()
            if len(first_val):
                f_frz[c] = float(first_val.iloc[0])
    faults.append((f_frz, gesture_df))

    # 3. 时间戳剧烈抖动 / 掉包堆积（dt CV 显著飙升） + 头部偏转出镜
    f_jit = face_df.copy()
    if "timestamp" in f_jit.columns and len(f_jit) > 2:
        gaps = rng.choice([0.05, 0.1, 4.5, 9.0], size=len(f_jit))
        f_jit["timestamp"] = 1700000000.0 + np.cumsum(gaps)
    if "head_yaw" in f_jit.columns:
        f_jit["head_yaw"] = rng.uniform(0.75, 1.35, size=len(f_jit))
    faults.append((f_jit, gesture_df))

    # 4. 上肢传感器尖峰与丢失（手势表缺检或坐标尖峰）
    if gesture_df is not None and not gesture_df.empty:
        g_bad = gesture_df.copy()
        for c in ("left_wrist_jitter_world", "right_wrist_jitter_world", "left_hand_jitter", "right_hand_jitter"):
            if c in g_bad.columns:
                g_bad[c] = rng.uniform(0.60, 2.50, size=len(g_bad))
        faults.append((face_df, g_bad))

    return faults


def build_dataset(log_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    loader = LogDataLoader(str(log_dir) if log_dir else None)
    groups = loader._scan_and_group_files()
    face_by_sid = {f["session_id"]: f["path"] for f in groups["face"]}
    gest_by_sid = {f["session_id"]: f["path"] for f in groups["gesture"]}

    rng = np.random.default_rng(42)
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    g_list: list[np.ndarray] = []
    used_sessions = 0

    for sid in sorted(face_by_sid):
        try:
            fdf = pd.read_csv(face_by_sid[sid], low_memory=False)
        except Exception:
            continue
        if len(fdf) < 15:
            continue
        gdf = None
        if sid in gest_by_sid:
            try:
                gdf = pd.read_csv(gest_by_sid[sid], low_memory=False)
            except Exception:
                gdf = None

        X_real, _ = extract_window_features(fdf, gdf)
        if len(X_real) == 0:
            continue

        # 真实窗中筛选明显健康的窗作为正样本，天然损坏窗归为负样本
        face_cov = X_real[:, 0]
        au_diff = X_real[:, 2]
        dt_cv = X_real[:, 8]
        clean_mask = (
            np.isfinite(face_cov) & (face_cov >= 0.70)
            & (~np.isfinite(au_diff) | (au_diff > 1e-4))
            & (~np.isfinite(dt_cv) | (dt_cv <= 0.60))
        )
        broken_mask = (
            (face_cov < 0.35)
            | (np.isfinite(au_diff) & (face_cov >= 0.5) & (au_diff <= 1e-5))
        )

        if clean_mask.any():
            Xc = X_real[clean_mask]
            X_list.append(Xc)
            y_list.append(np.ones(len(Xc), dtype=int))
            g_list.append(np.full(len(Xc), sid, dtype=object))
        if broken_mask.any():
            Xb = X_real[broken_mask]
            X_list.append(Xb)
            y_list.append(np.zeros(len(Xb), dtype=int))
            g_list.append(np.full(len(Xb), sid, dtype=object))

        # 故障注入负样本（按正样本规模平衡采样）
        n_clean = max(int(clean_mask.sum()), 4)
        fault_windows: list[np.ndarray] = []
        for f_bad, g_bad in _inject_faults(fdf, gdf, rng):
            Xf, _ = extract_window_features(f_bad, g_bad)
            if len(Xf):
                fault_windows.append(Xf)
        if fault_windows:
            Xf_all = np.vstack(fault_windows)
            take = min(len(Xf_all), max(n_clean, 8))
            idx = rng.choice(len(Xf_all), size=take, replace=False)
            X_list.append(Xf_all[idx])
            y_list.append(np.zeros(take, dtype=int))
            g_list.append(np.full(take, sid, dtype=object))

        used_sessions += 1

    if not X_list:
        raise RuntimeError("未找到可用于训练采集信号质检模型的真实 L0 会话日志")

    return np.vstack(X_list), np.concatenate(y_list), np.concatenate(g_list), used_sessions


def train_and_save(
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
) -> dict:
    t0 = time.perf_counter()
    X, y, groups, n_sessions = build_dataset()
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    if n_splits < 2:
        raise RuntimeError(f"有效会话数不足以执行 GroupKFold（当前 {len(unique_groups)} 场）")

    gkf = GroupKFold(n_splits=n_splits)
    aucs, f1s, precs, recs = [], [], [], []
    oof_probs = np.zeros(len(y), dtype=float)

    for tr_idx, va_idx in gkf.split(X, y, groups):
        clf = HistGradientBoostingClassifier(
            max_depth=4,
            max_iter=80,
            learning_rate=0.08,
            min_samples_leaf=15,
            l2_regularization=1.0,
            random_state=42,
        )
        clf.fit(X[tr_idx], y[tr_idx])
        probs = clf.predict_proba(X[va_idx])[:, 1]
        oof_probs[va_idx] = probs
        preds = (probs >= 0.5).astype(int)
        if len(np.unique(y[va_idx])) == 2:
            aucs.append(float(roc_auc_score(y[va_idx], probs)))
        f1s.append(float(f1_score(y[va_idx], preds, zero_division=0)))
        precs.append(float(precision_score(y[va_idx], preds, zero_division=0)))
        recs.append(float(recall_score(y[va_idx], preds, zero_division=0)))

    final_clf = HistGradientBoostingClassifier(
        max_depth=4,
        max_iter=80,
        learning_rate=0.08,
        min_samples_leaf=15,
        l2_regularization=1.0,
        random_state=42,
    )
    final_clf.fit(X, y)

    # 测量单场（30 窗）纯 CPU 推理耗时
    sample_batch = X[: min(30, len(X))]
    t_inf0 = time.perf_counter()
    for _ in range(20):
        final_clf.predict_proba(sample_batch)
    inf_ms_per_session = ((time.perf_counter() - t_inf0) / 20.0) * 1000.0

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_clf, model_path, compress=3)
    model_kb = round(model_path.stat().st_size / 1024.0, 1)

    metrics = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_scope": "capture_signal_quality_audit_only",
        "scoring_participation": False,
        "model_type": type(final_clf).__name__,
        "model_size_kb": model_kb,
        "n_sessions": int(n_sessions),
        "n_windows_total": int(len(y)),
        "n_positive_usable": int((y == 1).sum()),
        "n_negative_degraded": int((y == 0).sum()),
        "feature_names": list(FEATURE_NAMES),
        "cv_scheme": f"GroupKFold(n_splits={n_splits}, group=session_id)",
        "cv_roc_auc_mean": round(float(np.mean(aucs)), 4) if aucs else None,
        "cv_roc_auc_std": round(float(np.std(aucs)), 4) if aucs else None,
        "cv_f1_mean": round(float(np.mean(f1s)), 4),
        "cv_precision_mean": round(float(np.mean(precs)), 4),
        "cv_recall_mean": round(float(np.mean(recs)), 4),
        "cpu_inference_ms_per_30_windows": round(inf_ms_per_session, 3),
        "train_elapsed_sec": round(time.perf_counter() - t0, 2),
        "boundary_note": (
            "本模型仅利用自监督故障注入评估采集端物理信号可用性（遮挡/冻结/抖动/尖峰），"
            "不代表任何能力预测效果，且绝不进入白盒裁判层。"
        ),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    m = train_and_save()
    print(json.dumps(m, ensure_ascii=False, indent=2))
