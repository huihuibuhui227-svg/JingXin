#!/usr/bin/env python3
"""训练并验证《衡才智析》Layer 3 ML 行为阶段切分与证据核验门控模型（纯 CPU + StratifiedGroupKFold）。

训练边界（严格非评分、零人物评价、零 PII）：
- 仅读取 `data/logs/`、`data/l0_series/` 与 `data/transcripts/` 下真实会话的客观运动学、发声序列与覆盖审计结果；
- Model 1 (`models/referee_hmm_model.joblib`)：
  在 47 场真实会话的 5,666 个多模态滑动窗上拟合 4 状态粘性 GaussianHMM（Viterbi 阶段解码）
  并对比帧独立 GMM 的跳变抖动抑制率、平均驻留时长与 5 折 GroupKFold 一致性；
- Model 2 (`models/referee_conformal_gate.joblib`)：
  100% 基于 47 场真实会话（0 条合成样本），以非循环的下游五维 L1 证据审计 G1-G4 物理置信度上限
  （`coverage.confidence_cap`：高 / 中 / 低或不足）作为监督信号，
  拟合 StandardScaler + LogisticRegression(C=2.0) + IsolationForest(contamination=0.08) +
  Split Conformal Prediction（alpha=0.10，理论覆盖率 90%），严守 5 折 StratifiedGroupKFold(session_id) 验证，
  并补充剔除 `measured_ratio` 的 7 维原始传感遥测非循环消融实验；
- 产物：
  1. models/referee_hmm_model.joblib
  2. models/referee_conformal_gate.joblib
  3. report_frontend/referee_ml_metrics.json
  4. /home/zgy/JingXin/07_三方向与评审筹备/系统补强/referee_ml_benchmark_20261124.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from report_frontend import behavior_regime_model as brm
from report_frontend import referee_ml_engine as rme
from report_frontend.cross_modal_coverage import evaluate_cross_modal_coverage
from report_frontend.data_loader import LogDataLoader
from report_frontend.l1_report import build_l1_report
from report_frontend.l1_session import compute_session_l1
from report_frontend.signal_quality_clf import audit_session_signal_quality
from report_frontend.voice_series import load_series

DEFAULT_BENCHMARK_PATH = Path("/home/zgy/JingXin/07_三方向与评审筹备/系统补强/referee_ml_benchmark_20261124.json")


def _build_hmm_preprocessor() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("power", PowerTransformer(method="yeo-johnson")),
        ("scaler", StandardScaler()),
    ])


def _fit_sticky_hmm(
    X_z: np.ndarray,
    lengths: list[int],
    *,
    sticky_diag: float = 0.99,
    min_covar: float = 0.25,
) -> tuple[GaussianHMM, GaussianMixture, dict[int, str], dict[int, str]]:
    gmm = GaussianMixture(
        n_components=4,
        covariance_type="diag",
        reg_covar=0.1,
        n_init=5,
        random_state=42,
    ).fit(X_z)
    gmm_map = brm.assign_cluster_states(gmm.means_, brm.FEATURE_NAMES)

    off = (1.0 - sticky_diag) / 3.0
    init_trans = np.full((4, 4), off, dtype=float)
    np.fill_diagonal(init_trans, sticky_diag)

    hmm = GaussianHMM(
        n_components=4,
        covariance_type="diag",
        n_iter=150,
        tol=0.5,
        min_covar=min_covar,
        init_params="",
        params="mc",
        random_state=42,
    )
    hmm.startprob_ = np.full(4, 0.25, dtype=float)
    hmm.transmat_ = init_trans
    hmm.means_ = gmm.means_.copy()
    hmm.covars_ = np.maximum(gmm.covariances_.copy(), min_covar)
    hmm.fit(X_z, lengths=lengths)
    hmm_map = brm.assign_cluster_states(hmm.means_, brm.FEATURE_NAMES)
    return hmm, gmm, hmm_map, gmm_map


def _derive_downstream_audit_stage(l1_report_payload: dict[str, Any]) -> str:
    """基于下游五维 L1 证据审计上限（G1-G4 物理质量门控产物）导出非循环证据阶段标签。"""
    cov = l1_report_payload.get("coverage") if isinstance(l1_report_payload, dict) else None
    cap = str((cov or {}).get("confidence_cap") or "").strip()
    if cap == "高":
        return "stage_1_verified"
    if cap == "中":
        return "stage_2_review"
    return "stage_3_limited"


def _generate_corrupted_ood_samples(X_ref: np.ndarray, seed: int) -> np.ndarray:
    """构造物理矛盾/采集损坏的分布外测试样本，用于评估 IsolationForest 的 OOD 检出 AUROC。"""
    rng = np.random.default_rng(seed)
    n = len(X_ref)
    half = max(1, n)
    # 类型 A：高实测宣称但各路底层有效观测接近零（结构性矛盾）
    c1 = np.column_stack([
        rng.uniform(0.85, 1.0, half),
        rng.uniform(0.75, 1.0, half),
        rng.uniform(0.0, 0.08, half),
        rng.uniform(0.0, 0.05, half),
        rng.uniform(0.0, 0.05, half),
        rng.uniform(0.0, 0.05, half),
        rng.uniform(0.0, 0.05, half),
        rng.uniform(0.02, 0.15, half),
    ])
    # 类型 B：模态宣称齐全但时间戳极端抖动且完全无跨模态重合
    c2 = np.column_stack([
        rng.uniform(0.0, 0.15, half),
        rng.uniform(0.75, 1.0, half),
        rng.uniform(0.80, 1.0, half),
        rng.uniform(0.0, 0.05, half),
        rng.uniform(0.85, 1.0, half),
        rng.uniform(0.0, 0.05, half),
        rng.uniform(0.0, 0.05, half),
        rng.uniform(0.03, 0.18, half),
    ])
    return np.vstack([c1, c2])


def load_real_sessions(log_dir: Path | None = None) -> list[dict[str, Any]]:
    default_logs_dir = (REPO / "data" / "logs").resolve()
    resolved_dir = Path(log_dir).resolve() if log_dir is not None else default_logs_dir
    is_default_corpus = resolved_dir == default_logs_dir

    loader = LogDataLoader(str(log_dir) if log_dir else None)
    groups = loader._scan_and_group_files()
    face_by_sid = {f["session_id"]: f["path"] for f in groups["face"]}
    gest_by_sid = {f["session_id"]: f["path"] for f in groups["gesture"]}
    iv_by_sid = {f["session_id"]: f["path"] for f in groups["interview"]}
    rs_by_sid = {f["session_id"]: f["path"] for f in groups["research"]}

    fg_sids = sorted(set(face_by_sid) | set(gest_by_sid))
    sessions: list[dict[str, Any]] = []

    for sid in fg_sids:
        logs: dict[str, pd.DataFrame] = {}
        for key, mp in (
            ("face", face_by_sid),
            ("gesture", gest_by_sid),
            ("voice_interview", iv_by_sid),
            ("voice_research", rs_by_sid),
        ):
            if sid in mp:
                try:
                    logs[key] = pd.read_csv(mp[sid], low_memory=False)
                except Exception:
                    try:
                        df_safe = loader._read_csv_safe(mp[sid])
                        if df_safe is not None and not df_safe.empty:
                            logs[key] = df_safe
                    except Exception:
                        pass

        vs = load_series(sid)
        X_w, meta_w = brm.extract_regime_windows(logs.get("face"), logs.get("gesture"), voice_series=vs)
        if len(X_w) == 0:
            continue

        l1_res = compute_session_l1(sid, logs=logs)
        cmc = evaluate_cross_modal_coverage(l1_res, logs=logs)
        sq = audit_session_signal_quality(logs, sid)
        ev_vec, ev_map = rme.extract_evidence_features(
            logs,
            session_id=sid,
            voice_series_list=vs,
            l1_result=l1_res,
            cmc_payload=cmc,
            sq_payload=sq,
        )

        # 下游五维 L1 证据审计真相标签：
        # 真实语料直接以 build_l1_report(sid) 的 G1-G4 置信度上限为非循环监督信号
        # （避免部分历史场次 voice_interview CSV 无时基/序列缓存触发校验 ValueError）；
        # 若单元测试传入自定义 log_dir，则优先带入 logs 并保留安全兜底。
        if is_default_corpus:
            l1_report_payload = build_l1_report(sid)
        else:
            try:
                l1_report_payload = build_l1_report(sid, logs=logs)
            except Exception:
                l1_report_payload = build_l1_report(sid)

        ev_label = _derive_downstream_audit_stage(l1_report_payload)
        sessions.append({
            "session_id": sid,
            "logs": logs,
            "X_windows": X_w,
            "meta_windows": meta_w,
            "ev_vec": ev_vec,
            "ev_map": ev_map,
            "ev_label": ev_label,
            "l1_report": l1_report_payload,
            "has_voice_series": bool(np.isfinite(X_w[:, -1]).any()),
        })

    if not sessions:
        raise RuntimeError("未找到可用于训练 Layer 3 模型的真实多模态会话日志")
    return sessions


def _iter_grouped_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits_target: int,
    shuffle: bool = False,
    random_state: int = 42,
) -> tuple[str, list[tuple[np.ndarray, np.ndarray]]]:
    """构建分层分组折分；若测试夹具某类样本过少则优雅回退至 GroupKFold / StratifiedKFold / KFold。"""
    uniq_groups = np.unique(groups)
    n_groups = int(len(uniq_groups))
    uniq_classes = np.unique(y)

    if len(uniq_classes) >= 2 and n_groups >= 2:
        min_groups_per_class = min(int(len(np.unique(groups[y == c]))) for c in uniq_classes)
        k_sgkf = min(n_splits_target, n_groups, min_groups_per_class)
        if k_sgkf >= 2:
            splitter = StratifiedGroupKFold(
                n_splits=k_sgkf,
                shuffle=shuffle,
                random_state=random_state if shuffle else None,
            )
            return f"StratifiedGroupKFold(n_splits={k_sgkf}, group=session_id)", list(
                splitter.split(X, y, groups=groups)
            )

    if n_groups >= 2:
        k_gkf = min(n_splits_target, n_groups)
        splitter = GroupKFold(n_splits=k_gkf)
        return f"GroupKFold(n_splits={k_gkf}, group=session_id)", list(splitter.split(X, y, groups=groups))

    if len(uniq_classes) >= 2:
        min_cls = min(int((y == c).sum()) for c in uniq_classes)
        k_skf = min(n_splits_target, min_cls)
        if k_skf >= 2:
            splitter = StratifiedKFold(n_splits=k_skf, shuffle=shuffle, random_state=random_state)
            return f"StratifiedKFold(n_splits={k_skf})", list(splitter.split(X, y))

    k_kf = min(n_splits_target, len(X))
    if k_kf < 2:
        raise RuntimeError(f"样本数不足以执行交叉验证: {len(X)}")
    splitter = KFold(n_splits=k_kf, shuffle=shuffle, random_state=random_state if shuffle else None)
    return f"KFold(n_splits={k_kf})", list(splitter.split(X))


def _predict_full_probs(clf: LogisticRegression, X_scaled: np.ndarray, n_classes: int) -> np.ndarray:
    """将分类器预测概率按完整类别槽位 [0..n_classes-1] 对齐，兼容小规模合成夹具缺类折。"""
    raw_p = np.asarray(clf.predict_proba(X_scaled), dtype=float)
    if raw_p.shape[1] == n_classes and np.array_equal(clf.classes_, np.arange(n_classes)):
        return raw_p
    full_p = np.zeros((len(X_scaled), n_classes), dtype=float)
    for col_idx, cls_id in enumerate(clf.classes_):
        full_p[:, int(cls_id)] = raw_p[:, col_idx]
    return full_p


def _fit_gate_and_conformal(
    X_ev: np.ndarray,
    y_ev: np.ndarray,
    sess_ids: np.ndarray,
    classes: list[str],
    *,
    alpha: float = 0.10,
    C: float = 2.0,
    n_splits_outer: int = 5,
    n_splits_inner: int = 4,
    eval_ood: bool = True,
) -> dict[str, Any]:
    """100% 在真实会话上执行外层 StratifiedGroupKFold + 内层 Split Conformal 校准与全量拟合。"""
    n_samples = int(len(y_ev))
    n_classes = int(len(classes))
    outer_desc, outer_splits = _iter_grouped_splits(
        X_ev, y_ev, sess_ids, n_splits_target=n_splits_outer, shuffle=False, random_state=42
    )

    oof_preds = np.zeros(n_samples, dtype=int)
    oof_probs = np.zeros((n_samples, n_classes), dtype=float)
    oof_in_set = np.zeros(n_samples, dtype=bool)
    oof_set_size = np.zeros(n_samples, dtype=int)
    oof_escalated = np.zeros(n_samples, dtype=bool)
    fold_q_hats: list[float] = []
    fold_ood_aurocs: list[float] = []
    oof_real_nonconf = np.zeros(n_samples, dtype=float)

    for fold_idx, (tr_i, va_i) in enumerate(outer_splits):
        # 内层分组分层折分仅在真实训练集 tr_i 上收集无验证泄漏的保形校准残差
        _, inner_splits = _iter_grouped_splits(
            X_ev[tr_i],
            y_ev[tr_i],
            sess_ids[tr_i],
            n_splits_target=n_splits_inner,
            shuffle=True,
            random_state=42 + fold_idx,
        )
        cal_scores: list[float] = []
        for itr, ical in inner_splits:
            if len(np.unique(y_ev[tr_i[itr]])) < 2:
                continue
            sc_i = StandardScaler().fit(X_ev[tr_i[itr]])
            clf_i = LogisticRegression(C=C, max_iter=500, random_state=42).fit(
                sc_i.transform(X_ev[tr_i[itr]]), y_ev[tr_i[itr]]
            )
            p_cal = _predict_full_probs(clf_i, sc_i.transform(X_ev[tr_i[ical]]), n_classes)
            cal_scores.extend((1.0 - p_cal[np.arange(len(ical)), y_ev[tr_i[ical]]]).tolist())

        if not cal_scores:
            cal_scores = [0.5]
        cal_arr = np.asarray(cal_scores, dtype=float)
        n_cal = len(cal_arr)
        q_level = min(1.0, math.ceil((n_cal + 1) * (1.0 - alpha)) / float(n_cal))
        q_hat = float(np.quantile(cal_arr, q_level, method="higher"))
        fold_q_hats.append(q_hat)

        scaler_f = StandardScaler().fit(X_ev[tr_i])
        clf_f = LogisticRegression(C=C, max_iter=500, random_state=42).fit(
            scaler_f.transform(X_ev[tr_i]), y_ev[tr_i]
        )

        iso_scaler_f = StandardScaler().fit(X_ev[tr_i])
        iso_f = IsolationForest(n_estimators=40, contamination=0.08, random_state=42).fit(
            iso_scaler_f.transform(X_ev[tr_i])
        )

        va_p = _predict_full_probs(clf_f, scaler_f.transform(X_ev[va_i]), n_classes)
        va_pred = np.argmax(va_p, axis=1)
        oof_probs[va_i] = va_p
        oof_preds[va_i] = va_pred

        va_iso_z = iso_scaler_f.transform(X_ev[va_i])
        va_ood_sc = -iso_f.decision_function(va_iso_z)
        va_ood_flag = iso_f.predict(va_iso_z) == -1

        for j, vi in enumerate(va_i):
            oof_real_nonconf[vi] = float(1.0 - va_p[j, y_ev[vi]])
            cset = np.flatnonzero(1.0 - va_p[j] <= q_hat)
            empty = len(cset) == 0
            if empty:
                cset = np.array([va_pred[j]])
            oof_in_set[vi] = bool(y_ev[vi] in cset)
            oof_set_size[vi] = int(len(cset))
            oof_escalated[vi] = bool((len(cset) > 1) or empty or bool(va_ood_flag[j]))

        if eval_ood and X_ev.shape[1] == len(rme.EVIDENCE_FEATURE_NAMES):
            ood_X = _generate_corrupted_ood_samples(X_ev[va_i], seed=1000 + fold_idx)
            ood_sc = -iso_f.decision_function(iso_scaler_f.transform(ood_X))
            y_ood = np.concatenate([np.zeros(len(va_ood_sc)), np.ones(len(ood_sc))])
            sc_ood = np.concatenate([va_ood_sc, ood_sc])
            fold_ood_aurocs.append(float(roc_auc_score(y_ood, sc_ood)))

    cv_acc = float(accuracy_score(y_ev, oof_preds))
    cv_macro_f1 = float(f1_score(y_ev, oof_preds, average="macro", zero_division=0))
    emp_cov = float(oof_in_set.mean())
    mean_set_size = float(oof_set_size.mean())
    escalation_rate = float(oof_escalated.mean())
    cm = confusion_matrix(y_ev, oof_preds, labels=list(range(n_classes))).tolist()

    ood_auroc_mean = float(np.mean(fold_ood_aurocs)) if fold_ood_aurocs else None
    ood_auroc_std = float(np.std(fold_ood_aurocs)) if fold_ood_aurocs else None

    # 全量交付模型：100% 使用真实样本 OOF 残差计算有限样本分位数 q_hat
    final_q_level = min(1.0, math.ceil((n_samples + 1) * (1.0 - alpha)) / float(n_samples))
    final_q_hat = float(np.quantile(oof_real_nonconf, final_q_level, method="higher"))

    final_scaler = StandardScaler().fit(X_ev)
    final_clf = LogisticRegression(C=C, max_iter=500, random_state=42).fit(
        final_scaler.transform(X_ev), y_ev
    )
    final_iso_scaler = StandardScaler().fit(X_ev)
    final_iso = IsolationForest(n_estimators=40, contamination=0.08, random_state=42).fit(
        final_iso_scaler.transform(X_ev)
    )

    return {
        "cv_scheme": outer_desc,
        "outer_splits": int(len(outer_splits)),
        "cv_accuracy": cv_acc,
        "cv_macro_f1": cv_macro_f1,
        "empirical_conformal_coverage": emp_cov,
        "mean_conformal_set_size": mean_set_size,
        "uncertainty_escalation_rate": escalation_rate,
        "confusion_matrix": cm,
        "fold_q_hats": fold_q_hats,
        "final_q_hat": final_q_hat,
        "ood_auroc_mean": ood_auroc_mean,
        "ood_auroc_std": ood_auroc_std,
        "scaler": final_scaler,
        "classifier": final_clf,
        "iso_scaler": final_iso_scaler,
        "isolation_forest": final_iso,
    }


def train_and_save(
    *,
    model_dir: Path = rme.DEFAULT_MODEL_DIR,
    metrics_path: Path = rme.DEFAULT_METRICS_PATH,
    benchmark_path: Path = DEFAULT_BENCHMARK_PATH,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    sessions = load_real_sessions(log_dir=log_dir)
    n_sessions = len(sessions)
    n_windows = int(sum(len(s["X_windows"]) for s in sessions))
    n_sessions_voice = int(sum(1 for s in sessions if s["has_voice_series"]))

    sess_ids = np.asarray([s["session_id"] for s in sessions], dtype=object)
    n_splits = min(5, n_sessions)
    if n_splits < 2:
        raise RuntimeError(f"有效会话数不足以执行 GroupKFold: {n_sessions}")

    # ── 1. 训练并评估 Model 1: Sticky 4-State GaussianHMM + PELT ──────────────
    X_all_win = np.vstack([s["X_windows"] for s in sessions])
    lengths_all = [len(s["X_windows"]) for s in sessions]

    full_prep = _build_hmm_preprocessor()
    Xz_all = full_prep.fit_transform(X_all_win)
    full_hmm, full_gmm, full_hmm_map, full_gmm_map = _fit_sticky_hmm(Xz_all, lengths_all)

    full_hmm_states: list[str] = []
    full_gmm_trans = 0
    full_hmm_trans = 0
    gmm_dwells: list[float] = []
    hmm_dwells: list[float] = []
    pelt_cps_counts: list[int] = []
    pelt_ms_list: list[float] = []

    offset = 0
    for s in sessions:
        L = len(s["X_windows"])
        Xs_z = Xz_all[offset : offset + L]
        offset += L

        g_st = [full_gmm_map[int(c)] for c in full_gmm.predict(Xs_z)]
        h_st = [full_hmm_map[int(c)] for c in full_hmm.predict(Xs_z)]
        full_hmm_states.extend(h_st)

        full_gmm_trans += sum(1 for a, b in zip(g_st[:-1], g_st[1:]) if a != b)
        full_hmm_trans += sum(1 for a, b in zip(h_st[:-1], h_st[1:]) if a != b)
        for seg in brm.merge_state_segments(g_st, s["meta_windows"]):
            gmm_dwells.append(float(seg["duration_sec"]))
        for seg in brm.merge_state_segments(h_st, s["meta_windows"]):
            hmm_dwells.append(float(seg["duration_sec"]))

        tp0 = time.perf_counter()
        cps = rme._detect_pelt_changepoints(Xs_z, s["meta_windows"])
        pelt_ms_list.append((time.perf_counter() - tp0) * 1000.0)
        pelt_cps_counts.append(len(cps))

    full_hmm_states_arr = np.asarray(full_hmm_states, dtype=object)

    # 5-fold GroupKFold for HMM vs GMM
    gkf = GroupKFold(n_splits=n_splits)
    cv_aris: list[float] = []
    cv_agreements: list[float] = []
    cv_gmm_trans = 0
    cv_hmm_trans = 0
    cv_gmm_dwells: list[float] = []
    cv_hmm_dwells: list[float] = []

    for tr_si, va_si in gkf.split(sess_ids, groups=sess_ids):
        tr_set = set(sess_ids[tr_si])
        tr_X = np.vstack([s["X_windows"] for s in sessions if s["session_id"] in tr_set])
        tr_lens = [len(s["X_windows"]) for s in sessions if s["session_id"] in tr_set]

        f_prep = _build_hmm_preprocessor()
        tr_Xz = f_prep.fit_transform(tr_X)
        f_hmm, f_gmm, f_hmm_map, f_gmm_map = _fit_sticky_hmm(tr_Xz, tr_lens)

        va_pred: list[str] = []
        va_ref: list[str] = []
        off_ref = 0
        for s in sessions:
            L = len(s["X_windows"])
            if s["session_id"] not in tr_set:
                va_Xz = f_prep.transform(s["X_windows"])
                g_st = [f_gmm_map[int(c)] for c in f_gmm.predict(va_Xz)]
                h_st = [f_hmm_map[int(c)] for c in f_hmm.predict(va_Xz)]
                va_pred.extend(h_st)
                va_ref.extend(full_hmm_states_arr[off_ref : off_ref + L].tolist())

                cv_gmm_trans += sum(1 for a, b in zip(g_st[:-1], g_st[1:]) if a != b)
                cv_hmm_trans += sum(1 for a, b in zip(h_st[:-1], h_st[1:]) if a != b)
                for seg in brm.merge_state_segments(g_st, s["meta_windows"]):
                    cv_gmm_dwells.append(float(seg["duration_sec"]))
                for seg in brm.merge_state_segments(h_st, s["meta_windows"]):
                    cv_hmm_dwells.append(float(seg["duration_sec"]))
            off_ref += L

        cv_aris.append(float(adjusted_rand_score(va_ref, va_pred)))
        cv_agreements.append(float(np.mean(np.asarray(va_ref) == np.asarray(va_pred))))

    cv_flicker_red_pct = (1.0 - cv_hmm_trans / max(cv_gmm_trans, 1)) * 100.0
    full_flicker_red_pct = (1.0 - full_hmm_trans / max(full_gmm_trans, 1)) * 100.0

    hmm_artifact = {
        "version": "1.0.0",
        "feature_names": list(brm.FEATURE_NAMES),
        "window_size": brm.WINDOW_SIZE,
        "stride": brm.WINDOW_STRIDE,
        "preprocessor": full_prep,
        "hmm": full_hmm,
        "state_map": full_hmm_map,
        "gmm_ref": full_gmm,
        "gmm_map": full_gmm_map,
        "macro_stage_meta": rme.MACRO_STAGE_META,
        "transmat_diag": [round(float(v), 4) for v in np.diag(full_hmm.transmat_)],
    }
    hmm_path = model_dir / "referee_hmm_model.joblib"
    hmm_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(hmm_artifact, hmm_path, compress=3)
    hmm_kb = round(hmm_path.stat().st_size / 1024.0, 2)

    # ── 2. 训练并评估 Model 2: 100% 真实会话非循环监督 + Split Conformal ──────
    X_ev = np.vstack([s["ev_vec"] for s in sessions])
    classes = list(rme.STAGE_ORDER)
    cls_to_i = {c: i for i, c in enumerate(classes)}
    y_ev_str = np.asarray([s["ev_label"] for s in sessions], dtype=object)
    y_ev = np.asarray([cls_to_i[str(c)] for c in y_ev_str], dtype=int)

    alpha = 0.10
    gate_res = _fit_gate_and_conformal(
        X_ev,
        y_ev,
        sess_ids,
        classes,
        alpha=alpha,
        C=2.0,
        n_splits_outer=n_splits,
        n_splits_inner=4,
        eval_ood=True,
    )

    # 非循环原始传感遥测消融实验：剔除 measured_ratio，仅用 7 维原始传感与时序对齐特征预测下游审计上限
    feature_names_all = list(rme.EVIDENCE_FEATURE_NAMES)
    excluded_features = ["measured_ratio"]
    keep_indices = [i for i, fname in enumerate(feature_names_all) if fname not in excluded_features]
    ablation_features = [feature_names_all[i] for i in keep_indices]
    ablation_res = _fit_gate_and_conformal(
        X_ev[:, keep_indices],
        y_ev,
        sess_ids,
        classes,
        alpha=alpha,
        C=2.0,
        n_splits_outer=n_splits,
        n_splits_inner=4,
        eval_ood=False,
    )

    gate_artifact = {
        "version": "1.1.0",
        "supervision_protocol": "downstream_l1_g1_g4_audit_outcome",
        "feature_names": feature_names_all,
        "classes": classes,
        "stage_labels": rme.STAGE_LABELS,
        "scaler": gate_res["scaler"],
        "classifier": gate_res["classifier"],
        "iso_scaler": gate_res["iso_scaler"],
        "isolation_forest": gate_res["isolation_forest"],
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "q_hat": gate_res["final_q_hat"],
        "empirical_cv_coverage": round(gate_res["empirical_conformal_coverage"], 4),
    }
    gate_path = model_dir / "referee_conformal_gate.joblib"
    joblib.dump(gate_artifact, gate_path, compress=3)
    gate_kb = round(gate_path.stat().st_size / 1024.0, 2)

    # ── 3. 端到端纯 CPU 热推理延迟基准 ────────────────────────────────────────
    rep_sess = min(sessions, key=lambda s: abs(len(s["X_windows"]) - 75))
    try:
        rep_l1 = build_l1_report(rep_sess["session_id"], logs=rep_sess["logs"])
    except Exception:
        rep_l1 = rep_sess["l1_report"]

    _ = rme.evaluate_referee_ml(
        rep_sess["logs"], session_id=rep_sess["session_id"], l1_summary=rep_l1, model_dir=model_dir
    )
    hot_latencies: list[float] = []
    for _ in range(15):
        t_i0 = time.perf_counter()
        out_probe = rme.evaluate_referee_ml(
            rep_sess["logs"], session_id=rep_sess["session_id"], l1_summary=rep_l1, model_dir=model_dir
        )
        hot_latencies.append((time.perf_counter() - t_i0) * 1000.0)
        json.dumps(out_probe, allow_nan=False)

    standalone_latencies: list[float] = []
    for _ in range(5):
        t_s0 = time.perf_counter()
        out_standalone = rme.evaluate_referee_ml(
            rep_sess["logs"], session_id=rep_sess["session_id"], model_dir=model_dir
        )
        standalone_latencies.append((time.perf_counter() - t_s0) * 1000.0)
        json.dumps(out_standalone, allow_nan=False)

    stage_dist = {
        cls: int((y_ev_str == cls).sum())
        for cls in classes
    }

    metrics: dict[str, Any] = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_scope": "macro_behavioral_stage_segmentation_and_conformal_evidence_gating",
        "scoring_participation": False,
        "non_scoring_guarantee": rme.NON_SCORING_GUARANTEE,
        "dataset": {
            "n_real_sessions": int(n_sessions),
            "n_real_windows": int(n_windows),
            "n_real_sessions_with_voice_series": int(n_sessions_voice),
            "n_synthetic_boundary_calibration": 0,
            "synthetic_usage_note": (
                "已取消合成边界样本，全部模型训练、分层分组交叉验证与 Split Conformal 校准均 100% 基于真实多模态会话样本。"
            ),
            "real_stage_distribution": stage_dist,
        },
        "stage_segmentation": {
            "model_type": "GaussianHMM + ruptures.Pelt(rbf)",
            "model_file": "models/referee_hmm_model.joblib",
            "model_size_kb": hmm_kb,
            "n_states": 4,
            "sticky_self_transition_diag": [round(float(v), 4) for v in np.diag(full_hmm.transmat_)],
            "cv_scheme": f"GroupKFold(n_splits={n_splits}, group=session_id)",
            "gmm_cv_transition_count": int(cv_gmm_trans),
            "hmm_cv_transition_count": int(cv_hmm_trans),
            "flicker_reduction_pct": round(cv_flicker_red_pct, 2),
            "full_corpus_flicker_reduction_pct": round(full_flicker_red_pct, 2),
            "gmm_mean_dwell_sec": round(float(np.mean(cv_gmm_dwells)), 2),
            "hmm_mean_dwell_sec": round(float(np.mean(cv_hmm_dwells)), 2),
            "cross_fold_viterbi_ari_mean": round(float(np.mean(cv_aris)), 4),
            "cross_fold_viterbi_agreement_mean": round(float(np.mean(cv_agreements)), 4),
            "pelt_mean_changepoints_per_session": round(float(np.mean(pelt_cps_counts)), 2),
            "pelt_median_ms_per_session": round(float(np.median(pelt_ms_list)), 3),
        },
        "evidence_gate": {
            "model_type": "StandardScaler + LogisticRegression(C=2.0) + IsolationForest + SplitConformal",
            "model_file": "models/referee_conformal_gate.joblib",
            "model_size_kb": gate_kb,
            "supervision_protocol": "downstream_l1_g1_g4_audit_outcome",
            "supervision_note_plain": (
                "真实训练标签严格取自下游五维 L1 证据审计的 G1 至 G4 物理置信度上限，"
                "包含高、中与低或不足三档，完全脱离输入特征阈值规则并消除循环监督。"
            ),
            "feature_names": feature_names_all,
            "cv_scheme": f"{gate_res['cv_scheme']} on {n_sessions} real sessions",
            "alpha": alpha,
            "target_coverage": round(1.0 - alpha, 2),
            "final_q_hat": round(gate_res["final_q_hat"], 4),
            "fold_q_hats": [round(q, 4) for q in gate_res["fold_q_hats"]],
            "cv_accuracy": round(gate_res["cv_accuracy"], 4),
            "cv_macro_f1": round(gate_res["cv_macro_f1"], 4),
            "empirical_conformal_coverage": round(gate_res["empirical_conformal_coverage"], 4),
            "mean_conformal_set_size": round(gate_res["mean_conformal_set_size"], 3),
            "uncertainty_escalation_rate": round(gate_res["uncertainty_escalation_rate"], 4),
            "ood_auroc_mean": (
                round(gate_res["ood_auroc_mean"], 4) if gate_res["ood_auroc_mean"] is not None else None
            ),
            "ood_auroc_std": (
                round(gate_res["ood_auroc_std"], 4) if gate_res["ood_auroc_std"] is not None else None
            ),
            "confusion_matrix": {
                "labels": classes,
                "matrix": gate_res["confusion_matrix"],
            },
            "non_circular_raw_telemetry_ablation": {
                "n_features": int(len(ablation_features)),
                "excluded_features": excluded_features,
                "feature_names": ablation_features,
                "cv_scheme": f"{ablation_res['cv_scheme']} on {n_sessions} real sessions",
                "cv_accuracy": round(ablation_res["cv_accuracy"], 4),
                "cv_macro_f1": round(ablation_res["cv_macro_f1"], 4),
                "empirical_conformal_coverage": round(ablation_res["empirical_conformal_coverage"], 4),
                "mean_conformal_set_size": round(ablation_res["mean_conformal_set_size"], 3),
                "confusion_matrix": {
                    "labels": classes,
                    "matrix": ablation_res["confusion_matrix"],
                },
            },
        },
        "cpu_runtime": {
            "benchmark_session_windows": int(len(rep_sess["X_windows"])),
            "warm_inference_ms_median": round(float(np.median(hot_latencies)), 2),
            "warm_inference_ms_p95": round(float(np.percentile(hot_latencies, 95)), 2),
            "standalone_inference_ms_median": round(float(np.median(standalone_latencies)), 2),
            "train_elapsed_sec": round(time.perf_counter() - t_start, 2),
        },
        "boundary_note": rme.DISCLAIMER_TEXT,
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_json = json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    metrics_path.write_text(metrics_json, encoding="utf-8")

    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.write_text(metrics_json, encoding="utf-8")
    return metrics


if __name__ == "__main__":
    out = train_and_save()
    print(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False))
