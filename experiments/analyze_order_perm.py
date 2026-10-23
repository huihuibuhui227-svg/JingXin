#!/usr/bin/env python
"""Analyze the order-permutation experiment (gpt-5.6-sol).

Reads llm_results_codexapis_orderperm.jsonl (perm = -1 original order, 0..K shuffled),
and reports how stable the model's scores are under within-sample bullet-order
permutations (original vs each shuffle), plus a cross-channel sanity check of the
sol-original scores against the luna direct_v2 scores on the same videos.
Run anytime; results printed for the rows present so far.
"""
import json
import os
import numpy as np

FEA = os.path.expanduser("~/jingxin/experiments/results/features")
ORDER = f"{FEA}/llm_results_codexapis_orderperm.jsonl"
TARGETS = ["openness", "conscientiousness", "extraversion", "agreeableness",
           "neuroticism", "overall_personality", "interview_score", "answer_score",
           "speaking_skills", "confidence_score", "facial_expression", "overall_performance"]


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def load_order():
    by = {}   # vid -> {perm: scores}
    with open(ORDER, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("scores"):
                by.setdefault(r["video_id"], {})[int(r["perm"])] = r["scores"]
    return by


def mean12(s):
    return float(np.mean([s[d] for d in TARGETS if d in s]))


def metrics(orig, perm):
    common = [v for v in orig if v in perm]
    if not common:
        return None
    # per-dim pooled macro
    dim_r = []
    for d in TARGETS:
        a = [orig[v][d] for v in common if d in orig[v] and d in perm[v]]
        b = [perm[v][d] for v in common if d in orig[v] and d in perm[v]]
        if len(a) >= 3:
            dim_r.append(pearson(a, b))
    # pooled all obs
    A, B = [], []
    for v in common:
        for d in TARGETS:
            if d in orig[v] and d in perm[v]:
                A.append(orig[v][d]); B.append(perm[v][d])
    # per-video r over dims
    pv = [pearson([orig[v][d] for d in TARGETS if d in orig[v]],
                  [perm[v][d] for d in TARGETS if d in perm[v]])
          for v in common]
    pv = [x for x in pv if np.isfinite(x)]
    ms = pearson([mean12(orig[v]) for v in common], [mean12(perm[v]) for v in common])
    return {
        "n": len(common),
        "mean_score_r": ms,
        "per_dim_pooled_macro": float(np.nanmean(dim_r)),
        "pooled_r": pearson(A, B),
        "per_video_r_mean": float(np.nanmean(pv)),
    }


def load_luna_direct():
    out = {}
    with open(f"{FEA}/llm_results_codexapis.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("method") == "direct_v2" and r.get("scores"):
                out[r["video_id"]] = r["scores"]
    return out


def main():
    by = load_order()
    orig = {v: s[-1] for v, s in by.items() if -1 in s}
    perms = sorted({p for s in by.values() for p in s if p >= 0})
    print(f"sol orig videos={len(orig)}  perm indices present={perms}")
    res = {}
    for p in perms:
        perm = {v: s[p] for v, s in by.items() if p in s}
        m = metrics(orig, perm)
        if m:
            res[p] = m
            print(f"[orig vs perm{p}] n={m['n']} mean_score_r={m['mean_score_r']:+.3f} "
                  f"per_dim_pooled={m['per_dim_pooled_macro']:+.3f} pooled={m['pooled_r']:+.3f} "
                  f"per_video={m['per_video_r_mean']:+.3f}")
    # cross-channel sanity: sol-original vs luna direct_v2 on common videos (mean score)
    luna = load_luna_direct()
    common = [v for v in orig if v in luna]
    if common:
        ms = pearson([mean12(orig[v]) for v in common], [mean12(luna[v]) for v in common])
        print(f"[sol-orig vs luna-direct_v2] n={len(common)} mean_score_r={ms:+.3f} (通道一致性 sanity)")
    res["_orig_n"] = len(orig)
    with open(f"{FEA}/orderperm_analysis.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
