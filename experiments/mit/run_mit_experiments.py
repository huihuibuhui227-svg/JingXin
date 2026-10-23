#!/usr/bin/env python
"""
MIT — 同范式复现: 文本/行为/融合
================================
与 RecruitView 相同协议:
  * person-level 7:1:2 划分(69 人, 每人 pre/post 两场只进一个 split), 5 seeds
  * TF-IDF 只在 train 文本 fit
  * 主模型 RF; Ridge 亦报告
n 很小(≈138), 结果偏噪 → 只用于方向性复现(文本≥行为? 融合≥单模态?)
"""
from __future__ import annotations
import json, re, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import sparse, stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

OUT = Path(__file__).resolve().parent / "out"
SEEDS = [42, 123, 456, 789, 1024]


def pearson(a, b):
    return stats.pearsonr(a, b)[0]


def eval_dims(yt, yp, names):
    rows = {}
    for d, k in enumerate(names):
        rows[k] = {"pearson_r": float(pearson(yt[:, d], yp[:, d])),
                   "mse": float(np.mean((yt[:, d] - yp[:, d]) ** 2))}
    rows["macro_r"] = float(np.mean([rows[k]["pearson_r"] for k in names]))
    return rows


def person_splits(persons, seed):
    """person-level 7:1:2, 每 seed 真随机重分组。返回三组 person 集合"""
    people = sorted(persons)
    rng = np.random.RandomState(seed)
    rng.shuffle(people)
    n = len(people)
    n_tr = int(round(0.7 * n)); n_va = int(round(0.1 * n))
    return set(people[:n_tr]), set(people[n_tr:n_tr + n_va]), set(people[n_tr + n_va:])


def csr(a):
    return a if sparse.issparse(a) else sparse.csr_matrix(a)


def main():
    keys = json.load(open(OUT / "keys.json"))["keys"]
    persons = {k: int(re.search(r"\d+", k).group()) for k in keys}
    texts = json.load(open(OUT / "texts.json", encoding="utf-8"))["texts"]
    names = json.load(open(OUT / "keys.json"))["cols"]
    X = np.load(OUT / "X_behavior.npy")
    y = np.load(OUT / "y.npy")
    ynames = json.load(open(OUT / "texts.json", encoding="utf-8"))["targets"]
    assert len(keys) == X.shape[0] == y.shape[0]

    per_seed = {}
    for seed in SEEDS:
        trp, vap, tep = person_splits(set(persons.values()), seed)
        tr = [i for i, k in enumerate(keys) if persons[k] in trp]
        va = [i for i, k in enumerate(keys) if persons[k] in vap]
        te = [i for i, k in enumerate(keys) if persons[k] in tep]
        # TF-IDF (train-only)
        vec = TfidfVectorizer(max_features=1000, stop_words="english",
                              sublinear_tf=True, min_df=1, ngram_range=(1, 2))
        Xtr_t = vec.fit_transform([texts[i] for i in tr])
        Xva_t = vec.transform([texts[i] for i in va])
        Xte_t = vec.transform([texts[i] for i in te])
        beh_tr, beh_va, beh_te = X[tr], X[va], X[te]
        fu_tr = sparse.hstack([csr(beh_tr), Xtr_t], format="csr")
        fu_va = sparse.hstack([csr(beh_va), Xva_t], format="csr")
        fu_te = sparse.hstack([csr(beh_te), Xte_t], format="csr")
        ytr, yva, yte = y[tr], y[va], y[te]

        def rf_fit(A):
            return RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                         max_depth=None, n_jobs=-1, random_state=42).fit(A, ytr)
        def ridge_fit(A):
            return Ridge(alpha=1.0).fit(A, ytr)

        rows = {}
        preds = {}
        for tag, A_tr, A_va, A_te in [("text", Xtr_t, Xva_t, Xte_t),
                                      ("beh", beh_tr, beh_va, beh_te),
                                      ("fus", fu_tr, fu_va, fu_te)]:
            m_rf = rf_fit(A_tr); p_rf = m_rf.predict(A_te)
            m_rd = ridge_fit(A_tr); p_rd = m_rd.predict(A_te)
            rows[f"{tag}_rf"] = eval_dims(yte, p_rf, ynames)
            rows[f"{tag}_ridge"] = eval_dims(yte, p_rd, ynames)
            preds[tag] = (p_rf, p_rd, m_rf.predict(A_va), m_rd.predict(A_va))
        # late fusion: RF 主模型 + Ridge 加权
        pt, pva_t = preds["text"][0], preds["text"][2]
        pb, pva_b = preds["beh"][0], preds["beh"][2]
        lf = np.zeros_like(pt)
        for d in range(y.shape[1]):
            w = Ridge(alpha=1.0).fit(np.column_stack([pva_t[:, d], pva_b[:, d]]), yva[:, d])
            lf[:, d] = w.predict(np.column_stack([pt[:, d], pb[:, d]]))
        rows["late_fusion"] = eval_dims(yte, lf, ynames)
        rows["_n"] = {"train": len(tr), "val": len(va), "test": len(te), "text_dim": Xtr_t.shape[1]}
        per_seed[str(seed)] = rows
        print(f"[seed {seed}] n_test={len(te)} "
              f"text={rows['text_rf']['macro_r']:.3f} "
              f"beh={rows['beh_rf']['macro_r']:.3f} "
              f"fus={rows['fus_rf']['macro_r']:.3f} "
              f"late={rows['late_fusion']['macro_r']:.3f} "
              f"(text_dim={Xtr_t.shape[1]})")

    def ms(key):
        return np.array([per_seed[str(s)][key]["macro_r"] for s in SEEDS])
    summ = {}
    for key in ["text_rf", "beh_rf", "fus_rf", "late_fusion", "text_ridge", "beh_ridge", "fus_ridge"]:
        mm = ms(key)
        summ[key] = {"macro_mean": float(mm.mean()), "macro_std": float(mm.std()),
                     "per_seed": [float(x) for x in mm]}
    for a, b in [("fus_rf", "text_rf"), ("fus_rf", "beh_rf"), ("beh_rf", "text_rf")]:
        summ[f"{a} - {b}"] = float(ms(a).mean() - ms(b).mean())
    json.dump({"summary": summ, "per_seed": per_seed}, open(OUT / "mit_result.json", "w"),
              indent=1, ensure_ascii=False)
    print("\n=== MIT 汇总(5 seeds) ===")
    for key in ["text_rf", "beh_rf", "fus_rf", "late_fusion"]:
        print(f"  {key:12s} macro = {summ[key]['macro_mean']:.3f}±{summ[key]['macro_std']:.3f}")
    print(f"  beh-text = {summ['beh_rf - text_rf']:+.3f}   fus-text = {summ['fus_rf - text_rf']:+.3f}   fus-beh = {summ['fus_rf - beh_rf']:+.3f}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"耗时 {time.time()-t0:.0f}s")
