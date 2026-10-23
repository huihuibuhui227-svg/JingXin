#!/usr/bin/env python
"""
CueCoT Supplementary Experiments — Supporting New Narrative
============================================================
Experiment 1: Transcript-only baseline (text vs. behavior)
Experiment 2: RF feature importance (what drives predictions?)
Experiment 3: Within-user score consistency
Experiment 4: LLM reasoning chain quality analysis
"""

import json
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from collections import defaultdict

FEATURES_DIR = Path(r"D:\jingxin\experiments\results\features")
METADATA_PATH = Path(r"D:\RecruitView\RecruitView\metadata.jsonl")
OUTPUT_DIR = Path(r"D:\jingxin\experiments\results")

TARGET_KEYS = [
    "openness", "conscientiousness", "extraversion", "agreeableness",
    "neuroticism", "overall_personality",
    "interview_score", "answer_score", "speaking_skills",
    "confidence_score", "facial_expression", "overall_performance",
]
PERSONALITY = TARGET_KEYS[:6]
INTERVIEW = TARGET_KEYS[6:]


def load_data():
    X = np.load(FEATURES_DIR / "all_features.npy").astype(np.float64)
    y = np.load(FEATURES_DIR / "targets.npy").astype(np.float64)
    with open(FEATURES_DIR / "feature_names.txt") as f:
        names = [l.strip() for l in f]
    with open(FEATURES_DIR / "all_features.jsonl") as f:
        vids = [json.loads(l)["video_id"] for l in f if l.strip()]

    meta = {}
    with open(METADATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            meta[Path(d["file_name"]).stem] = d

    return X, y, names, vids, meta


def experiment_1_text_vs_behavior(X, y, names, vids, meta):
    """
    Compare text-only (transcript + gemini_summary) vs. behavioral features.
    Uses TF-IDF on text → Ridge regression.
    """
    print("=" * 60)
    print("EXPERIMENT 1: Text-only vs. Behavioral Features")
    print("=" * 60)

    # Build text features
    texts = []
    for vid in vids:
        m = meta.get(vid, {})
        transcript = m.get("transcript", "") or ""
        summary = m.get("gemini_summary", "") or ""
        texts.append(transcript + " " + summary)

    # TF-IDF
    vec = TfidfVectorizer(max_features=500, stop_words="english", sublinear_tf=True)
    X_text = vec.fit_transform(texts).toarray()
    print(f"  Text features: {X_text.shape[1]} TF-IDF dims")

    # Behavioral features
    scaler = StandardScaler()
    X_beh = scaler.fit_transform(X)
    print(f"  Behavioral features: {X_beh.shape[1]} dims")

    # Compare: 5-fold CV Ridge for each
    results = {}
    for name, X_feat in [("Text-only (TF-IDF)", X_text), ("Behavioral (1410d)", X_beh)]:
        row = {}
        for d, dim in enumerate(TARGET_KEYS):
            ridge = Ridge(alpha=1.0)
            y_pred = cross_val_predict(ridge, X_feat, y[:, d], cv=5)
            r_val, _ = stats.pearsonr(y[:, d], y_pred)
            mse = np.mean((y[:, d] - y_pred) ** 2)
            row[dim] = {"pearson_r": float(r_val), "mse": float(mse)}

        # Macro averages
        row["macro_r"] = float(np.mean([row[d]["pearson_r"] for d in TARGET_KEYS]))
        row["macro_mse"] = float(np.mean([row[d]["mse"] for d in TARGET_KEYS]))
        row["personality_r"] = float(np.mean([row[d]["pearson_r"] for d in PERSONALITY]))
        row["interview_r"] = float(np.mean([row[d]["pearson_r"] for d in INTERVIEW]))
        results[name] = row

    # Print comparison
    print(f"\n  {'':<25} {'Text r':>8} {'Behav r':>8} {'Winner':>10}")
    print("  " + "-" * 55)
    for dim in TARGET_KEYS:
        tr = results["Text-only (TF-IDF)"][dim]["pearson_r"]
        br = results["Behavioral (1410d)"][dim]["pearson_r"]
        winner = "TEXT" if tr > br else "BEHAV"
        print(f"  {dim:<25} {tr:>8.4f} {br:>8.4f} {winner:>10}")

    print(f"\n  {'Macro Avg':<25} "
          f"{results['Text-only (TF-IDF)']['macro_r']:>8.4f} "
          f"{results['Behavioral (1410d)']['macro_r']:>8.4f}")
    print(f"  {'Personality Avg':<25} "
          f"{results['Text-only (TF-IDF)']['personality_r']:>8.4f} "
          f"{results['Behavioral (1410d)']['personality_r']:>8.4f}")
    print(f"  {'Interview Avg':<25} "
          f"{results['Text-only (TF-IDF)']['interview_r']:>8.4f} "
          f"{results['Behavioral (1410d)']['interview_r']:>8.4f}")

    return results


def experiment_2_feature_importance(X, y, names, vids, meta):
    """
    Train RF on all features, report top-30 most important features.
    Categorize: behavioral signal vs. statistical artifact.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: RF Feature Importance Analysis")
    print("=" * 60)

    rf = RandomForestRegressor(n_estimators=200, max_depth=15,
                               random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # Aggregate importance across 12 targets
    importances = np.mean([rf.estimators_[i].feature_importances_
                           for i in range(len(rf.estimators_))], axis=0)

    # Also per-dimension
    per_dim_importance = {}
    for d, dim in enumerate(TARGET_KEYS):
        rf_d = RandomForestRegressor(n_estimators=200, max_depth=15,
                                     random_state=42, n_jobs=-1)
        rf_d.fit(X, y[:, d])
        per_dim_importance[dim] = rf_d.feature_importances_

    # Top 30 overall
    top_idx = np.argsort(importances)[::-1][:30]
    print(f"\n  Top-30 Features (aggregated importance):")
    print(f"  {'Rank':<5} {'Feature':<55} {'Importance':>10} {'Category':>20}")
    print("  " + "-" * 95)
    for rank, idx in enumerate(top_idx, 1):
        name = names[idx][:54]
        imp = importances[idx]
        # Categorize
        if any(kw in name for kw in ['mean', 'std', 'avg']):
            cat = "central_tendency"
        elif any(kw in name for kw in ['slope', 'trend']):
            cat = "temporal_trend"
        elif any(kw in name for kw in ['volatility', 'change_rate']):
            cat = "variability"
        elif any(kw in name for kw in ['min', 'max', 'range']):
            cat = "extremum"
        else:
            cat = "other"
        print(f"  {rank:<5} {name:<55} {imp:>10.4f} {cat:>20}")

    # Category summary
    cats = defaultdict(float)
    cat_counts = defaultdict(int)
    for idx in range(len(names)):
        name = names[idx]
        if any(kw in name for kw in ['mean', 'std', 'avg']):
            c = "central_tendency"
        elif any(kw in name for kw in ['slope', 'trend']):
            c = "temporal_trend"
        elif any(kw in name for kw in ['volatility', 'change_rate']):
            c = "variability"
        elif any(kw in name for kw in ['min', 'max', 'range']):
            c = "extremum"
        else:
            c = "other"
        cats[c] += importances[idx]
        cat_counts[c] += 1

    print(f"\n  Importance by feature category:")
    total = sum(cats.values())
    for c in sorted(cats, key=cats.get, reverse=True):
        print(f"    {c:<20}: {cats[c]/total*100:>5.1f}% ({cat_counts[c]} features)")

    # Per-dimension top features
    print(f"\n  Top-5 features per dimension:")
    for d, dim in enumerate(TARGET_KEYS[:6]):  # First 6 dimensions
        imp_d = per_dim_importance[dim]
        top5 = np.argsort(imp_d)[::-1][:5]
        print(f"    {dim}:")
        for idx in top5:
            print(f"      {names[idx][:60]} ({imp_d[idx]:.4f})")

    return importances, per_dim_importance


def experiment_3_within_user_consistency(vids, meta):
    """
    Check if same user gets consistent scores across questions.
    High within-user variance = raters evaluate question-specific quality,
    not stable personality traits.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Within-User Score Consistency")
    print("=" * 60)

    user_scores = defaultdict(list)
    for vid in vids:
        m = meta.get(vid, {})
        u = m.get("user_no", "unknown")
        scores = [m.get(k, 0) for k in TARGET_KEYS]
        user_scores[u].append(scores)

    # Only users with >= 3 videos
    multi_video_users = {u: s for u, s in user_scores.items() if len(s) >= 3}
    print(f"  Users with >= 3 videos: {len(multi_video_users)}/{len(user_scores)}")

    # ICC(1) — simplified: variance ratio
    within_var = []
    between_var = []
    for u, scores in multi_video_users.items():
        arr = np.array(scores)
        user_mean = arr.mean(axis=0)
        within_var.append(arr.var(axis=0, ddof=1))
        between_var.append((arr - user_mean).var(axis=0, ddof=1) if arr.shape[0] > 1
                            else np.zeros_like(user_mean))

    within_var = np.array(within_var)
    # ICC-like: between_var / (between_var + within_var)
    total_var_per_dim = np.var([np.mean(s, axis=0) for s in multi_video_users.values()],
                                axis=0, ddof=1)

    print(f"\n  Per-dimension analysis (within-user std / between-user std):")
    print(f"  {'Dimension':<25} {'Within-σ':>10} {'Between-σ':>10} {'ICC-like':>10}")
    print("  " + "-" * 60)
    for d, dim in enumerate(TARGET_KEYS):
        w_std = np.sqrt(np.mean([v[d] for v in within_var if not np.isnan(v[d])]))
        b_std = np.sqrt(max(0, total_var_per_dim[d]))
        icc = b_std**2 / (b_std**2 + w_std**2) if (b_std**2 + w_std**2) > 0 else 0
        print(f"  {dim:<25} {w_std:>10.4f} {b_std:>10.4f} {icc:>10.4f}")


def experiment_4_llm_reasoning_quality():
    """
    Analyze LLM reasoning chains: do they follow psychological common sense?
    Sample a few CoT chains and check consistency.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: LLM Reasoning Chain Analysis")
    print("=" * 60)

    with open(FEATURES_DIR / "llm_results.jsonl", encoding="utf-8") as f:
        results = [json.loads(l) for l in f if l.strip()]

    # Dedup
    seen = {}
    for r in results:
        k = (r["video_id"], r["method"], r.get("temperature", 0.0))
        seen[k] = r

    # Sample 5 cot_full with json_ok
    cot_ok = [(vid, r) for (vid, m, t), r in seen.items()
              if m == "cot_full" and r.get("parse_status") == "json_ok"]
    print(f"  CoT full with JSON OK: {len(cot_ok)}")

    # Analyze: do LLM scores correlate with each other in expected ways?
    # e.g., extraversion and confidence should be positively correlated
    all_scores = defaultdict(list)
    for vid, r in cot_ok:
        sc = r.get("scores", {})
        for dim, val in sc.items():
            if val is not None:
                all_scores[dim].append(val)

    print(f"\n  LLM Score Inter-correlations (expected patterns):")
    pairs = [
        ("extraversion", "confidence_score", "positive"),
        ("neuroticism", "confidence_score", "negative"),
        ("speaking_skills", "interview_score", "positive"),
        ("openness", "extraversion", "positive"),
        ("neuroticism", "extraversion", "negative"),
    ]
    print(f"  {'Pair':<40} {'Expected':>10} {'Actual r':>10} {'Match?':>10}")
    print("  " + "-" * 75)
    for d1, d2, expected in pairs:
        if d1 in all_scores and d2 in all_scores:
            a1 = np.array(all_scores[d1])
            a2 = np.array(all_scores[d2])
            r_val, _ = stats.pearsonr(a1, a2)
            if expected == "positive":
                match = "YES" if r_val > 0 else "NO"
            else:
                match = "YES" if r_val < 0 else "NO"
            print(f"  {d1} vs {d2:<15} {expected:>10} {r_val:>10.4f} {match:>10}")

    # Check one sample reasoning chain
    print(f"\n  Sample CoT reasoning chain (vid_0020):")
    for vid, r in cot_ok:
        if vid == "vid_0020":
            raw = r.get("raw_output", "")
            # Print first 800 chars
            print(f"  {raw[:800]}...")
            break


def main():
    print("CueCoT Supplementary Experiments")
    print("=" * 60)
    print()

    X, y, names, vids, meta = load_data()
    print(f"Loaded: X={X.shape}, y={y.shape}, {len(meta)} metadata entries")
    print()

    # Run experiments
    exp1_results = experiment_1_text_vs_behavior(X, y, names, vids, meta)
    exp2_results = experiment_2_feature_importance(X, y, names, vids, meta)
    experiment_3_within_user_consistency(vids, meta)
    experiment_4_llm_reasoning_quality()

    # Save results
    output = {
        "experiment_1_text_vs_behavior": exp1_results,
    }
    with open(OUTPUT_DIR / "supplementary_results.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False,
                  default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    print(f"\nSaved: {OUTPUT_DIR / 'supplementary_results.json'}")


if __name__ == "__main__":
    main()
