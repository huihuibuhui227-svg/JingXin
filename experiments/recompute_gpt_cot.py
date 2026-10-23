#!/usr/bin/env python
"""Recompute GPT CoT (codexapis cot) stats with FIRST-occurrence dedup, and the
DS-CoT vs GPT-CoT cross-model correlations. Writes a small summary JSON.

Motivation: llm_results_codexapis_cot.jsonl contains duplicate rows for 128/200
videos (resume/concurrency rewrites). Reading code that uses `d[vid] = ...` keeps
the LAST occurrence; we switch to the FIRST occurrence (paper decision), which
changes GPT-CoT-vs-expert macro from about -0.111 to about -0.131 (11/12 dims neg).
"""
import json
import os
import sys
import numpy as np

FEA = os.path.expanduser("~/jingxin/experiments/results/features")
TARGETS = ["openness", "conscientiousness", "extraversion", "agreeableness",
           "neuroticism", "overall_personality", "interview_score", "answer_score",
           "speaking_skills", "confidence_score", "facial_expression", "overall_performance"]


def load_features():
    with open(f"{FEA}/all_features.jsonl", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    idx = {r["video_id"]: i for i, r in enumerate(rows)}
    targets = np.load(f"{FEA}/targets.npy")  # 2011 x 12, row order == all_features
    return idx, targets


def load_scores(path, methods, dedup="first"):
    """Return {video_id: {dim: score}} for the first matching method row per video."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            m = r.get("method")
            if methods and m not in methods:
                continue
            if not r.get("scores"):
                continue
            vid = r["video_id"]
            if dedup == "first" and vid in out:
                continue
            if dedup == "last":
                out[vid] = r["scores"]
            else:
                out.setdefault(vid, r["scores"])
    return out


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def dim_pearson_vs_gt(scores, idx, targets, dim):
    di = TARGETS.index(dim)
    ys, gs = [], []
    for vid, s in scores.items():
        i = idx.get(vid)
        if i is None or dim not in s or not np.isfinite(targets[i, di]):
            continue
        ys.append(s[dim]); gs.append(targets[i, di])
    return pearson(ys, gs), len(ys)


def cross_model_dim(sa, sb, dim):
    vals = [(sa[v][dim], sb[v][dim]) for v in sa if v in sb and dim in sa[v] and dim in sb[v]]
    if not vals:
        return float("nan"), 0
    a, b = zip(*vals)
    return pearson(a, b), len(vals)


def main():
    idx, targets = load_features()

    # GPT CoT (codexapis) - FIRST dedup  / also report last for comparison
    gpt_cot_first = load_scores(f"{FEA}/llm_results_codexapis_cot.jsonl", {"cot_full_v3"}, "first")
    gpt_cot_last = load_scores(f"{FEA}/llm_results_codexapis_cot.jsonl", {"cot_full_v3"}, "last")
    print(f"GPT CoT unique first-dedup videos: {len(gpt_cot_first)} ; last-wins videos: {len(gpt_cot_last)}")

    for label, sc in [("FIRST", gpt_cot_first), ("LAST", gpt_cot_last)]:
        rs, neg = [], 0
        for d in TARGETS:
            r, n = dim_pearson_vs_gt(sc, idx, targets, d)
            rs.append(r)
            if r < 0:
                neg += 1
        print(f"[GPT CoT {label}] macro={np.nanmean(rs):.4f} neg={neg}/12 n_firstdim={len(sc)}")
        print("   per-dim:", " ".join(f"{r:+.3f}" for r in rs))

    # DS CoT (DeepSeek cot_full) first dedup for cross-model
    ds_cot = load_scores(f"{FEA}/llm_results.jsonl", {"cot_full"}, "first")
    print(f"DS CoT unique videos: {len(ds_cot)}")

    out = {}
    dims_r = []
    for d in TARGETS:
        r, n = cross_model_dim(ds_cot, gpt_cot_first, d)
        dims_r.append(r)
    pooled = np.nanmean(dims_r)  # per-dim pooled macro
    # pooled across all obs (concatenate)  == "pooled r"
    a = [(ds_cot[v][d], gpt_cot_first[v][d]) for v in ds_cot if v in gpt_cot_first for d in TARGETS
         if d in ds_cot[v] and d in gpt_cot_first[v]]
    pa, pb = zip(*a)
    pooled_r_all = pearson(pa, pb)
    # per-video: pearson over 12 dims per video then mean
    pv = []
    for v in ds_cot:
        if v in gpt_cot_first:
            xa = [ds_cot[v][d] for d in TARGETS if d in ds_cot[v]]
            xb = [gpt_cot_first[v][d] for d in TARGETS if d in gpt_cot_first[v]]
            if len(xa) >= 6 and len(xb) >= 6:
                rr = pearson(xa, xb)
                if np.isfinite(rr):
                    pv.append(rr)
    common = len([v for v in ds_cot if v in gpt_cot_first])
    print(f"[DS↔GPT CoT] n_common={common} per-dim-pooled-macro={pooled:.4f} "
          f"pooled-r(all)={pooled_r_all:.4f} per-video-mean={np.mean(pv):.4f}")

    # aggregate GPT CoT (first) vs expert table values we want to persist
    rs_first = [dim_pearson_vs_gt(gpt_cot_first, idx, targets, d)[0] for d in TARGETS]
    rs_last = [dim_pearson_vs_gt(gpt_cot_last, idx, targets, d)[0] for d in TARGETS]
    out = {
        "gpt_cot_first_dedup": {
            "n": len(gpt_cot_first),
            "macro_vs_expert": float(np.nanmean(rs_first)),
            "neg_dims": sum(1 for r in rs_first if r < 0),
            "per_dim_vs_expert": {d: (float(r) if np.isfinite(r) else None) for d, r in zip(TARGETS, rs_first)},
        },
        "gpt_cot_last_dedup": {
            "n": len(gpt_cot_last),
            "macro_vs_expert": float(np.nanmean(rs_last)),
            "neg_dims": sum(1 for r in rs_last if r < 0),
        },
        "ds_cot": {"n": len(ds_cot)},
        "ds_vs_gpt_cot": {
            "n_common": common,
            "per_dim_pooled_macro": float(pooled),
            "pooled_r": float(pooled_r_all),
            "per_video_mean_r": float(np.mean(pv)),
        },
    }
    dst = f"{FEA}/gpt_cot_recomputed.json"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
