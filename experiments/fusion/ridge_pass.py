#!/usr/bin/env python
"""RecruitView 融合 — Ridge(线性)对照 pass。文本/行为/融合同用线性, 检验互补叠加。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import sparse, stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import run_fusion as R

OUT = Path(__file__).resolve().parent / "results"


def pearson(a, b):
    return stats.pearsonr(a, b)[0]


def eval_dims(yt, yp):
    row = {}
    for d, k in enumerate(R.TARGET_KEYS):
        row[k] = float(pearson(yt[:, d], yp[:, d]))
    row["macro_r"] = float(np.mean([row[k] for k in R.TARGET_KEYS]))
    return row


def main():
    X, y, names, vids, texts, users = R.load()
    per_seed = {}
    for seed in R.SEEDS:
        tr, va, te = R.user_splits(users, seed)
        vec = TfidfVectorizer(max_features=800, stop_words="english",
                              sublinear_tf=True, min_df=3)
        Xtr_t = vec.fit_transform([texts[i] for i in tr])
        Xte_t = vec.transform([texts[i] for i in te])
        sc = StandardScaler().fit(X[tr])
        btr, bte = sc.transform(X[tr]), sc.transform(X[te])
        ytr, yte = y[tr], y[te]
        ftr = sparse.hstack([sparse.csr_matrix(btr), Xtr_t], format="csr")
        fte = sparse.hstack([sparse.csr_matrix(bte), Xte_t], format="csr")
        rows = {}
        for tag, A_tr, A_te in [("text", Xtr_t, Xte_t), ("beh", btr, bte),
                                ("fus", ftr, fte)]:
            m = Ridge(alpha=1.0).fit(A_tr, ytr)
            rows[tag] = eval_dims(yte, m.predict(A_te))
        per_seed[str(seed)] = rows
        print(f"[seed {seed}] text={rows['text']['macro_r']:.3f} "
              f"beh={rows['beh']['macro_r']:.3f} fus={rows['fus']['macro_r']:.3f}", flush=True)

    def ms(k):
        return np.array([per_seed[str(s)][k]["macro_r"] for s in R.SEEDS])
    summ = {k: {"macro_mean": float(ms(k).mean()), "macro_std": float(ms(k).std())}
            for k in ["text", "beh", "fus"]}
    summ["fus-text"] = float(ms("fus").mean() - ms("text").mean())
    summ["fus-beh"] = float(ms("fus").mean() - ms("beh").mean())
    summ["seeds_fus_gt"] = {"text": int(np.sum(ms("fus") > ms("text"))),
                            "beh": int(np.sum(ms("fus") > ms("beh")))}
    json.dump({"summary": summ, "per_seed": per_seed},
              open(OUT / "ridge_result.json", "w"), indent=1)
    print("=== Ridge 汇总 ===")
    for k in ["text", "beh", "fus"]:
        print(f"  {k:5s} {summ[k]['macro_mean']:.4f}±{summ[k]['macro_std']:.4f}")
    print(f"  fus-text={summ['fus-text']:+.4f} fus-beh={summ['fus-beh']:+.4f} "
          f"seeds(fus>t,fus>b)={summ['seeds_fus_gt']}")


if __name__ == "__main__":
    main()
