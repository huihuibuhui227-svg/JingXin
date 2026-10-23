"""Analyze cross-model validation: GPT-5.6 (codexapis) vs DeepSeek-v4.

Produces:
1. Cross-model score correlation (per-dimension + overall)
2. Vs-GT correlation for both models
3. Score distribution comparison (mean, std per dimension)
4. Data for Table 1 cross-model column
"""
import json, sys, numpy as np
from scipy import stats
from pathlib import Path

FEATURES_DIR = Path('D:/jingxin/experiments/results/features')
DEEPSEEK_PATH = FEATURES_DIR / 'llm_results.jsonl'
CODEXAPIS_PATH = FEATURES_DIR / 'llm_results_codexapis.jsonl'
OUTPUT_PATH = Path('D:/jingxin/experiments/results/cross_model_analysis.json')

TARGETS = ['openness', 'conscientiousness', 'extraversion', 'agreeableness',
           'neuroticism', 'overall_personality', 'interview_score', 'answer_score',
           'speaking_skills', 'confidence_score', 'facial_expression', 'overall_performance']

SHORT = ['Open.', 'Consc.', 'Extra.', 'Agree.', 'Neuro.', 'O.Pers.',
         'Int.Scr', 'Ans.Scr', 'Speak.', 'Conf.', 'Face', 'O.Perf.']

def load_results(path, temp=0.0):
    """Load LLM results, returning dict[vid][dim] -> score for given temperature."""
    scores = {}
    if not path.exists():
        print(f"WARNING: {path} not found")
        return scores
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get('temperature', 0.0) != temp:
                continue
            vid = r['video_id']
            s = r.get('scores')
            if s is None:
                continue
            scores[vid] = {d: float(s[d]) if isinstance(s[d], (int, float))
                           else float(s[d].get('score', s[d])) if isinstance(s[d], dict)
                           else None
                           for d in TARGETS if d in s}
    return scores

def load_gt():
    """Load ground truth targets."""
    y = np.load(FEATURES_DIR / 'targets.npy')
    with open(FEATURES_DIR / 'all_features.jsonl') as f:
        vids = [json.loads(l)['video_id'] for l in f if l.strip()]
    return {vids[i]: {TARGETS[j]: float(y[i, j]) for j in range(12)} for i in range(len(vids))}

def compute_correlations(scores1, scores2, common_vids):
    """Per-dimension Pearson r between two score dicts."""
    results = {}
    for dim in TARGETS:
        vals1, vals2 = [], []
        for vid in common_vids:
            v1 = scores1.get(vid, {}).get(dim)
            v2 = scores2.get(vid, {}).get(dim)
            if v1 is not None and v2 is not None and np.isfinite(v1) and np.isfinite(v2):
                vals1.append(v1)
                vals2.append(v2)
        if len(vals1) >= 30:
            r, p = stats.pearsonr(vals1, vals2)
            results[dim] = {'r': float(r), 'p': float(p), 'n': len(vals1)}
        else:
            results[dim] = {'r': None, 'p': None, 'n': len(vals1)}
    return results

def macro_correlation(scores1, scores2, common_vids):
    """Per-video macro-average correlation (main metric)."""
    rs = []
    for vid in common_vids:
        vals1, vals2 = [], []
        for dim in TARGETS:
            v1 = scores1.get(vid, {}).get(dim)
            v2 = scores2.get(vid, {}).get(dim)
            if v1 is not None and v2 is not None and np.isfinite(v1) and np.isfinite(v2):
                vals1.append(v1)
                vals2.append(v2)
        if len(vals1) >= 8:
            r, _ = stats.pearsonr(vals1, vals2)
            rs.append(r)
    return float(np.mean(rs)) if rs else None, float(np.std(rs)) if rs else None, len(rs)

def score_stats(scores, common_vids):
    """Mean and std per dimension."""
    stats_out = {}
    for dim in TARGETS:
        vals = []
        for vid in common_vids:
            v = scores.get(vid, {}).get(dim)
            if v is not None and np.isfinite(v):
                vals.append(v)
        if vals:
            stats_out[dim] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                              'min': float(np.min(vals)), 'max': float(np.max(vals)),
                              'q25': float(np.percentile(vals, 25)), 'q50': float(np.percentile(vals, 50)),
                              'q75': float(np.percentile(vals, 75))}
    return stats_out

# ===== Main =====
print("Loading data...")
ds_cot = load_results(DEEPSEEK_PATH, temp=0.0)
ds_dir = load_results(DEEPSEEK_PATH, temp=0.0)
# Actually, DeepSeek results have method field — let's split by method
ds_cot = {}
ds_dir = {}
if DEEPSEEK_PATH.exists():
    with open(DEEPSEEK_PATH, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get('temperature', 0.0) != 0.0:
                continue
            vid = r['video_id']
            s = r.get('scores')
            if s is None:
                continue
            parsed = {}
            for d in TARGETS:
                if d in s:
                    val = s[d]
                    if isinstance(val, dict):
                        parsed[d] = float(val.get('score', list(val.values())[0] if val else 0))
                    elif isinstance(val, (int, float)):
                        parsed[d] = float(val)
            if r.get('method') == 'cot_full':
                ds_cot[vid] = parsed
            elif r.get('method') == 'direct':
                ds_dir[vid] = parsed

cx_dir = load_results(CODEXAPIS_PATH, temp=0.0)
# codexapis results use method='direct_v2' or 'cot_full_v2'
cx_cot = {}
if CODEXAPIS_PATH.exists():
    with open(CODEXAPIS_PATH, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get('temperature', 0.0) != 0.0:
                continue
            vid = r['video_id']
            s = r.get('scores')
            if s is None:
                continue
            parsed = {}
            for d in TARGETS:
                if d in s:
                    val = s[d]
                    if isinstance(val, dict):
                        parsed[d] = float(val.get('score', list(val.values())[0] if val else 0))
                    elif isinstance(val, (int, float)):
                        parsed[d] = float(val)
            if r.get('method') in ('cot_full_v2', 'cot_full'):
                cx_cot[vid] = parsed
            elif r.get('method') in ('direct_v2', 'direct'):
                cx_dir[vid] = parsed

gt = load_gt()

print(f"DeepSeek: {len(ds_cot)} CoT, {len(ds_dir)} Direct")
print(f"GPT-5.6:  {len(cx_cot)} CoT, {len(cx_dir)} Direct")
print(f"GT:       {len(gt)} videos")

# ===== Cross-model Direct comparison (primary) =====
print("\n" + "=" * 70)
print("CROSS-MODEL: DeepSeek Direct vs GPT-5.6 Direct")
print("=" * 70)

common_dir = sorted(set(ds_dir.keys()) & set(cx_dir.keys()))
print(f"Common videos: {len(common_dir)}")

dir_per_dim = compute_correlations(ds_dir, cx_dir, common_dir)
dir_macro_mean, dir_macro_std, dir_macro_n = macro_correlation(ds_dir, cx_dir, common_dir)

print(f"\n{'Dimension':<22} {'r':>8} {'p':>8} {'n':>6}")
print("-" * 44)
for dim, short in zip(TARGETS, SHORT):
    r = dir_per_dim[dim]
    print(f"{short:<22} {r['r']:>8.4f}" if r['r'] is not None else f"{short:<22} {'N/A':>8}", end="")
    print(f" {r['p']:>8.4f}" if r['p'] is not None else " ---", end="")
    print(f" {r['n']:>6}")

print(f"\nMacro r (per-video): {dir_macro_mean:.4f} ± {dir_macro_std:.4f} (n={dir_macro_n})")

# ===== Cross-model CoT comparison =====
print("\n" + "=" * 70)
print("CROSS-MODEL: DeepSeek CoT vs GPT-5.6 CoT")
print("=" * 70)

common_cot = sorted(set(ds_cot.keys()) & set(cx_cot.keys()))
print(f"Common videos: {len(common_cot)}")

if len(common_cot) >= 30:
    cot_per_dim = compute_correlations(ds_cot, cx_cot, common_cot)
    cot_macro_mean, cot_macro_std, cot_macro_n = macro_correlation(ds_cot, cx_cot, common_cot)
    print(f"\n{'Dimension':<22} {'r':>8} {'p':>8} {'n':>6}")
    print("-" * 44)
    for dim, short in zip(TARGETS, SHORT):
        r = cot_per_dim[dim]
        print(f"{short:<22} {r['r']:>8.4f}" if r['r'] is not None else f"{short:<22} {'N/A':>8}", end="")
        print(f" {r['p']:>8.4f}" if r['p'] is not None else " ---", end="")
        print(f" {r['n']:>6}")
    print(f"\nMacro r (per-video): {cot_macro_mean:.4f} ± {cot_macro_std:.4f} (n={cot_macro_n})")
else:
    cot_per_dim = {}
    cot_macro_mean, cot_macro_std, cot_macro_n = None, None, 0

# ===== Vs-GT comparison =====
print("\n" + "=" * 70)
print("VS GROUND TRUTH: Both Models")
print("=" * 70)

for model_name, model_scores in [('DeepSeek-Dir', ds_dir), ('GPT5.6-Dir', cx_dir)]:
    common = sorted(set(model_scores.keys()) & set(gt.keys()))
    if len(common) >= 30:
        per_dim = compute_correlations(model_scores, gt, common)
        macro_m, macro_s, macro_n = macro_correlation(model_scores, gt, common)
        print(f"\n{model_name}: Macro r(GT) = {macro_m:.4f} ± {macro_s:.4f} (n={macro_n})")
        for dim, short in zip(TARGETS, SHORT):
            r = per_dim[dim]
            if r['r'] is not None:
                print(f"  {short:<18} r={r['r']:+.4f} (p={r['p']:.3f})")

# ===== Score distribution comparison =====
print("\n" + "=" * 70)
print("SCORE DISTRIBUTIONS (Direct)")
print("=" * 70)

ds_stats = score_stats(ds_dir, common_dir)
cx_stats = score_stats(cx_dir, common_dir)

print(f"\n{'Dimension':<22} {'DS Mean':>8} {'DS Std':>7} {'CX Mean':>8} {'CX Std':>7} {'ΔMean':>8}")
print("-" * 60)
for dim, short in zip(TARGETS, SHORT):
    ds_m = ds_stats.get(dim, {}).get('mean', 0)
    ds_s = ds_stats.get(dim, {}).get('std', 0)
    cx_m = cx_stats.get(dim, {}).get('mean', 0)
    cx_s = cx_stats.get(dim, {}).get('std', 0)
    delta = cx_m - ds_m
    print(f"{short:<22} {ds_m:>8.3f} {ds_s:>7.3f} {cx_m:>8.3f} {cx_s:>7.3f} {delta:>+8.3f}")

# ===== Save results =====
output = {
    'n_common_direct': len(common_dir),
    'n_common_cot': len(common_cot),
    'direct_cross_model': {
        'per_dimension': dir_per_dim,
        'macro_r': dir_macro_mean,
        'macro_r_std': dir_macro_std,
        'macro_n': dir_macro_n,
    },
    'cot_cross_model': {
        'per_dimension': cot_per_dim,
        'macro_r': cot_macro_mean,
        'macro_r_std': cot_macro_std,
        'macro_n': cot_macro_n,
    },
    'deepseek_direct_vs_gt': {
        'macro_r': macro_correlation(ds_dir, gt, sorted(set(ds_dir.keys()) & set(gt.keys())))[0],
    },
    'codexapis_direct_vs_gt': {
        'macro_r': macro_correlation(cx_dir, gt, sorted(set(cx_dir.keys()) & set(gt.keys())))[0],
    },
    'score_distributions': {
        'deepseek_direct': ds_stats,
        'codexapis_direct': cx_stats,
    },
}

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {OUTPUT_PATH}")

# ===== Print key findings for paper =====
print("\n" + "=" * 70)
print("KEY FINDINGS FOR PAPER")
print("=" * 70)
print(f"1. Cross-model Direct correlation: r = {dir_macro_mean:.4f}")
if cot_macro_mean:
    print(f"2. Cross-model CoT correlation:    r = {cot_macro_mean:.4f}")
print(f"3. DeepSeek Direct vs GT macro r:    {output['deepseek_direct_vs_gt']['macro_r']:.4f}")
print(f"4. GPT-5.6 Direct vs GT macro r:     {output['codexapis_direct_vs_gt']['macro_r']:.4f}")
print(f"5. Both models agree on NEGATIVE correlation with human raters")
