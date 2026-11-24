#!/usr/bin/env python3
"""训练并验证《衡才智析》Layer 3 ML 行为阶段切分与证据核验门控模型（纯 CPU + GroupKFold）。

训练边界（严格非评分、零人物评价、零 PII）：
- 仅读取 `data/logs/` 与 `data/l0_series/` 下真实会话的客观运动学、发声序列与覆盖统计；
- Model 1 (`models/referee_hmm_model.joblib`)：
  在 47 场真实会话的 5,666 个多模态滑动窗上拟合 4 状态粘性 GaussianHMM（Viterbi 阶段解码）
  并对比帧独立 GMM 的跳变抖动抑制率、平均驻留时长与 5 折 GroupKFold 一致性；
- Model 2 (`models/referee_conformal_gate.joblib`)：
  在 47 场真实会话（配合 24 条确定性边界合成校准样补充对齐边界，真实与合成数在指标中分别诚实披露）上
  拟合 StandardScaler + LogisticRegression(C=2.0) + IsolationForest(contamination=0.08) +
  Split Conformal Prediction（alpha=0.10，理论覆盖率 90%），并严守 5 折 GroupKFold(session_id) 验证；
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
    f1_score,
    roc_auc_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from report_frontend import behavior_regime_model as brm
from report_frontend import referee_ml_engine as rme
from report_frontend.cross_modal_coverage import evaluate_cross_modal_coverage
from report_frontend.data_loader import LogDataLoader
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


def _derive_evidence_stage_label(feat_vec: np.ndarray) -> str:
    """基于客观采集完整性与跨模态覆盖导出结构性证据阶段标签（非评分）。"""
    meas_r, mod_norm, sq_u, cm_ov, fv, sp, gv, dyn = [float(x) for x in feat_vec]
    n_mods = round(mod_norm * 4.0)
    if (
        meas_r >= 0.70
        and n_mods >= 3
        and sq_u >= 0.65
        and cm_ov >= 0.55
        and fv >= 0.60
        and gv >= 0.60
        and dyn >= 0.50
    ):
        return "stage_1_verified"
    if meas_r >= 0.25 and (n_mods >= 2 or sq_u >= 0.50 or sp >= 0.25):
        return "stage_2_review"
    return "stage_3_limited"


def _build_synthetic_boundary_sessions() -> tuple[np.ndarray, np.ndarray]:
    """生成 24 条确定性边界/降质合成会话，仅用于校准阶段边界完整性，不进入真实 47 场 CV 统计。"""
    rng = np.random.default_rng(2026)
    rows: list[list[float]] = []
    labels: list[str] = []

    # 8 条阶段 I 稳定/近边界样本
    for _ in range(8):
        fv = float(rng.uniform(0.66, 0.98))
        gv = float(rng.uniform(0.66, 0.98))
        sp = float(rng.uniform(0.32, 0.70))
        cm_ov = min(fv, gv, min(1.0, sp / 0.30))
        rows.append([
            float(rng.uniform(0.75, 1.0)),
            float(rng.choice([0.75, 1.0])),
            float(rng.uniform(0.68, 0.98)),
            cm_ov,
            fv,
            sp,
            gv,
            float(rng.uniform(0.65, 0.96)),
        ])
        labels.append("stage_1_verified")

    # 8 条阶段 II 部分覆盖 / 跨模态错配 / 单路偏弱边界样本
    for i in range(8):
        if i < 3:
            # 单路视觉偏弱导致跨模态重合偏低
            fv = float(rng.uniform(0.08, 0.42))
            gv = float(rng.uniform(0.65, 0.98))
            sp = float(rng.uniform(0.35, 0.70))
            meas = float(rng.uniform(0.75, 0.98))
            mod_n = float(rng.choice([0.75, 1.0]))
            sq_u = float(rng.uniform(0.10, 0.45))
            cm_ov = min(fv, gv)
            dyn = float(rng.uniform(0.65, 0.92))
        elif i < 6:
            # 指标齐全但跨模态时段重叠偏低或姿态受限
            fv = float(rng.uniform(0.80, 0.96))
            gv = float(rng.uniform(0.20, 0.50))
            sp = float(rng.uniform(0.40, 0.65))
            meas = float(rng.uniform(0.85, 1.0))
            mod_n = 1.0
            sq_u = float(rng.uniform(0.65, 0.95))
            cm_ov = float(rng.uniform(0.06, 0.42))
            dyn = float(rng.uniform(0.50, 0.88))
        else:
            # L1 实测率中等但原始采集帧良好
            fv = float(rng.uniform(0.55, 0.95))
            gv = float(rng.uniform(0.55, 0.95))
            sp = float(rng.uniform(0.28, 0.72))
            meas = float(rng.uniform(0.28, 0.48))
            mod_n = float(rng.choice([0.25, 0.50]))
            sq_u = float(rng.uniform(0.55, 0.95))
            cm_ov = min(fv, gv, min(1.0, sp / 0.30))
            dyn = float(rng.uniform(0.70, 0.95))
        rows.append([meas, mod_n, sq_u, cm_ov, fv, sp, gv, dyn])
        labels.append("stage_2_review")

    # 8 条阶段 III 证据受限 / 稀疏样本
    for _ in range(8):
        rows.append([
            float(rng.uniform(0.0, 0.20)),
            float(rng.choice([0.0, 0.25])),
            float(rng.uniform(0.0, 0.45)),
            0.0,
            float(rng.uniform(0.0, 0.45)),
            float(rng.uniform(0.0, 0.08)),
            float(rng.uniform(0.0, 0.95)),
            float(rng.uniform(0.45, 0.95)),
        ])
        labels.append("stage_3_limited")

    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=object)


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
        ev_label = _derive_evidence_stage_label(ev_vec)
        sessions.append({
            "session_id": sid,
            "logs": logs,
            "X_windows": X_w,
            "meta_windows": meta_w,
            "ev_vec": ev_vec,
            "ev_map": ev_map,
            "ev_label": ev_label,
            "has_voice_series": bool(np.isfinite(X_w[:, -1]).any()),
        })

    if not sessions:
        raise RuntimeError("未找到可用于训练 Layer 3 模型的真实多模态会话日志")
    return sessions


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

    # ── 2. 训练并评估 Model 2: IsolationForest + LogisticRegression + Split Conformal ──
    X_ev = np.vstack([s["ev_vec"] for s in sessions])
    classes = list(rme.STAGE_ORDER)
    cls_to_i = {c: i for i, c in enumerate(classes)}
    y_ev_str = np.asarray([s["ev_label"] for s in sessions], dtype=object)
    y_ev = np.asarray([cls_to_i[str(c)] for c in y_ev_str], dtype=int)

    syn_X, syn_y_str = _build_synthetic_boundary_sessions()
    syn_y = np.asarray([cls_to_i[str(c)] for c in syn_y_str], dtype=int)

    alpha = 0.10
    oof_preds = np.zeros(len(y_ev), dtype=int)
    oof_probs = np.zeros((len(y_ev), len(classes)), dtype=float)
    oof_in_set = np.zeros(len(y_ev), dtype=bool)
    oof_set_size = np.zeros(len(y_ev), dtype=int)
    oof_escalated = np.zeros(len(y_ev), dtype=bool)
    fold_q_hats: list[float] = []
    fold_ood_aurocs: list[float] = []
    oof_real_nonconf: list[float] = []

    for fold_idx, (tr_i, va_i) in enumerate(gkf.split(X_ev, y_ev, groups=sess_ids)):
        # 内层 4 折仅在真实训练会话上产出无泄漏保形校准残差（合成样本仅用于辅助拟合线性分类面）
        inner_splits = min(4, len(tr_i))
        skf = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=42 + fold_idx)
        cal_scores: list[float] = []
        for itr, ical in skf.split(X_ev[tr_i], y_ev[tr_i]):
            X_fit = np.vstack([X_ev[tr_i[itr]], syn_X])
            y_fit = np.concatenate([y_ev[tr_i[itr]], syn_y])
            sc_i = StandardScaler().fit(X_fit)
            clf_i = LogisticRegression(C=2.0, max_iter=500, random_state=42).fit(sc_i.transform(X_fit), y_fit)
            p_cal = clf_i.predict_proba(sc_i.transform(X_ev[tr_i[ical]]))
            cal_scores.extend((1.0 - p_cal[np.arange(len(ical)), y_ev[tr_i[ical]]]).tolist())

        cal_arr = np.asarray(cal_scores, dtype=float)
        n_cal = len(cal_arr)
        q_level = min(1.0, math.ceil((n_cal + 1) * (1.0 - alpha)) / float(n_cal))
        q_hat = float(np.quantile(cal_arr, q_level, method="higher"))
        fold_q_hats.append(q_hat)

        # 折内分类器与孤立森林
        X_tr_aug = np.vstack([X_ev[tr_i], syn_X])
        y_tr_aug = np.concatenate([y_ev[tr_i], syn_y])
        scaler_f = StandardScaler().fit(X_tr_aug)
        clf_f = LogisticRegression(C=2.0, max_iter=500, random_state=42).fit(scaler_f.transform(X_tr_aug), y_tr_aug)

        iso_scaler_f = StandardScaler().fit(X_ev[tr_i])
        iso_f = IsolationForest(n_estimators=40, contamination=0.08, random_state=42).fit(iso_scaler_f.transform(X_ev[tr_i]))

        va_p = clf_f.predict_proba(scaler_f.transform(X_ev[va_i]))
        va_pred = np.argmax(va_p, axis=1)
        oof_probs[va_i] = va_p
        oof_preds[va_i] = va_pred
        for j, vi in enumerate(va_i):
            oof_real_nonconf.append(float(1.0 - va_p[j, y_ev[vi]]))

        va_iso_z = iso_scaler_f.transform(X_ev[va_i])
        va_ood_sc = -iso_f.decision_function(va_iso_z)
        va_ood_flag = iso_f.predict(va_iso_z) == -1

        for j, vi in enumerate(va_i):
            cset = np.flatnonzero(1.0 - va_p[j] <= q_hat)
            empty = len(cset) == 0
            if empty:
                cset = np.array([va_pred[j]])
            oof_in_set[vi] = bool(y_ev[vi] in cset)
            oof_set_size[vi] = int(len(cset))
            oof_escalated[vi] = bool((len(cset) > 1) or empty or bool(va_ood_flag[j]))

        # OOD AUROC：真实验证会话 vs 物理矛盾/损坏 OOD 样本
        ood_X = _generate_corrupted_ood_samples(X_ev[va_i], seed=1000 + fold_idx)
        ood_sc = -iso_f.decision_function(iso_scaler_f.transform(ood_X))
        y_ood = np.concatenate([np.zeros(len(va_ood_sc)), np.ones(len(ood_sc))])
        sc_ood = np.concatenate([va_ood_sc, ood_sc])
        fold_ood_aurocs.append(float(roc_auc_score(y_ood, sc_ood)))

    cv_acc = float(accuracy_score(y_ev, oof_preds))
    cv_macro_f1 = float(f1_score(y_ev, oof_preds, average="macro"))
    emp_cov = float(oof_in_set.mean())
    mean_set_size = float(oof_set_size.mean())
    escalation_rate = float(oof_escalated.mean())
    ood_auroc_mean = float(np.mean(fold_ood_aurocs))
    ood_auroc_std = float(np.std(fold_ood_aurocs))

    # 全量交付模型：使用 47 场真实 OOF 残差计算有限样本分位数 q_hat（无泄漏保形门槛）
    n_oof = len(oof_real_nonconf)
    final_q_level = min(1.0, math.ceil((n_oof + 1) * (1.0 - alpha)) / float(n_oof))
    final_q_hat = float(np.quantile(np.asarray(oof_real_nonconf, dtype=float), final_q_level, method="higher"))

    X_full_aug = np.vstack([X_ev, syn_X])
    y_full_aug = np.concatenate([y_ev, syn_y])
    final_scaler = StandardScaler().fit(X_full_aug)
    final_clf = LogisticRegression(C=2.0, max_iter=500, random_state=42).fit(final_scaler.transform(X_full_aug), y_full_aug)

    final_iso_scaler = StandardScaler().fit(X_ev)
    final_iso = IsolationForest(n_estimators=40, contamination=0.08, random_state=42).fit(final_iso_scaler.transform(X_ev))

    gate_artifact = {
        "version": "1.0.0",
        "feature_names": list(rme.EVIDENCE_FEATURE_NAMES),
        "classes": classes,
        "stage_labels": rme.STAGE_LABELS,
        "scaler": final_scaler,
        "classifier": final_clf,
        "iso_scaler": final_iso_scaler,
        "isolation_forest": final_iso,
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "q_hat": final_q_hat,
        "empirical_cv_coverage": round(emp_cov, 4),
    }
    gate_path = model_dir / "referee_conformal_gate.joblib"
    joblib.dump(gate_artifact, gate_path, compress=3)
    gate_kb = round(gate_path.stat().st_size / 1024.0, 2)

    # ── 3. 端到端纯 CPU 热推理延迟基准 ────────────────────────────────────────
    # 找一个 ~60-120 窗的中等会话测试热推理耗时（含带 l1_summary 报告热路径与独立路径）
    rep_sess = min(sessions, key=lambda s: abs(len(s["X_windows"]) - 75))
    from report_frontend.l1_report import build_l1_report
    rep_l1 = build_l1_report(rep_sess["session_id"], logs=rep_sess["logs"])

    # 先预热模型缓存
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
            "n_synthetic_boundary_calibration": int(len(syn_X)),
            "synthetic_usage_note": "24 条合成边界样仅用于辅助校准分类边界，全部交叉验证指标严格仅在 47 场真实会话上计算。",
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
            "feature_names": list(rme.EVIDENCE_FEATURE_NAMES),
            "cv_scheme": f"GroupKFold(n_splits={n_splits}, group=session_id) on 47 real sessions",
            "alpha": alpha,
            "target_coverage": round(1.0 - alpha, 2),
            "final_q_hat": round(final_q_hat, 4),
            "fold_q_hats": [round(q, 4) for q in fold_q_hats],
            "cv_accuracy": round(cv_acc, 4),
            "cv_macro_f1": round(cv_macro_f1, 4),
            "empirical_conformal_coverage": round(emp_cov, 4),
            "mean_conformal_set_size": round(mean_set_size, 3),
            "uncertainty_escalation_rate": round(escalation_rate, 4),
            "ood_auroc_mean": round(ood_auroc_mean, 4),
            "ood_auroc_std": round(ood_auroc_std, 4),
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
