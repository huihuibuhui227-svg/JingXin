#!/usr/bin/env python
"""RecruitView 融合增益探针: 信号浓缩(行为PCA + 文本SVD)后拼接是否让文本互补更突出。
先用 3 seeds 筛配置; 胜出再 5 seeds。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import run_fusion as R

SCREEN = [42, 123, 456]
OUT = Path(__file__).resolve().parent / "results"
RF = dict(n_estimators=200, min_samples_leaf=5, max_depth=None, n_jobs=-1, random_state=0)


def pearson(a, b):
    return stats.pearsonr(a, b)[0]


def macro(y, p):
    return float(np.mean([pearson(y[:, d], p[:, d]) for d in range(y.shape[1])]))


def main():
    X, y, names, vids, texts, users = R.load()
    res = {}
    for seed in SCREEN:
        tr, va, te = R.user_splits(users, seed)
        # 文本 SVD(在 train 上 fit)
        vec = TfidfVectorizer(max_features=2500, stop_words="english",
                              sublinear_tf=True, min_df=2)
        Xt_tr = vec.fit_transform([texts[i] for i in tr])
        svd = TruncatedSVD(n_components=200, random_state=0)
        tsvd_tr = svd.fit_transform(Xt_tr)  # fit then reuse 100/200 = first k cols
        Xt_te_all = vec.transform([texts[i] for i in te])
        tsvd_te = svd.transform(Xt_te_all)
        # 行为 PCA
        sc = StandardScaler().fit(X[tr])
        b_tr, b_te = sc.transform(X[tr]), sc.transform(X[te])
        pca = PCA(n_components=150, random_state=0)
        bp_tr = pca.fit_transform(b_tr); bp_te = pca.transform(b_te)
        ytr, yte = y[tr], y[te]

        def rf(A_tr, A_te):
            return RandomForestRegressor(**RF).fit(A_tr, ytr).predict(A_te)

        r = {}
        r["beh_PCA150"] = macro(yte, rf(bp_tr, bp_te))
        r["beh_full"] = macro(yte, rf(b_tr, b_te))
        for k in (100, 200):
            r[f"text_SVD{k}"] = macro(yte, rf(tsvd_tr[:, :k], tsvd_te[:, :k]))
            r[f"fus_PCA150+SVD{k}"] = macro(
                yte, rf(np.hstack([bp_tr, tsvd_tr[:, :k]]), np.hstack([bp_te, tsvd_te[:, :k]])))
            r[f"fus_full+SVD{k}"] = macro(
                yte, rf(np.hstack([b_tr, tsvd_tr[:, :k]]), np.hstack([b_te, tsvd_te[:, :k]])))
        res[str(seed)] = r
        print(f"[seed {seed}] " + "  ".join(f"{k}={v:.3f}" for k, v in r.items()), flush=True)

    # 汇总
    order = list(next(iter(res.values())).keys())
    summ = {}
    for k in order:
        arr = np.array([res[s][k] for s in res])
        summ[k] = {"mean": float(arr.mean()), "std": float(arr.std())}
        print(f"{k:20s} {arr.mean():.4f}±{arr.std():.4f}")
    json.dump({"per_seed": res, "summary": summ}, open(OUT / "boost_trials.json", "w"), indent=1)


if __name__ == "__main__":
    main()
