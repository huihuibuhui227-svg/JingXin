"""Calibration Analysis: Can affine transform fix LLM-GT misalignment?

Tests:
1. Per-dimension OLS: GT = a + b * LLM_score (fit on train, eval on test)
2. Sign reversal: GT ~ a + b * (-LLM_score)
3. Multi-output ridge calibration (12 dims jointly)
4. Report raw r vs calibrated r per dimension

Key question: Is the negative correlation just a scale/polarity issue,
or a genuine ranking failure?
"""
import json, sys
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

CONDA_PY = '/c/Users/Lenovo/.conda/envs/jingxin/python'

FEATURES_DIR = 'D:/jingxin/experiments/results/features'
RESULTS_DIR = 'D:/jingxin/experiments/results'
TARGETS = ['openness', 'conscientiousness', 'extraversion', 'agreeableness',
           'neuroticism', 'overall_personality', 'interview_score', 'answer_score',
           'speaking_skills', 'confidence_score', 'facial_expression', 'overall_performance']


def load_data():
    # Load GT targets
    targets = np.load(f'{FEATURES_DIR}/targets.npy')  # shape: (n_videos, 12)

    # Load split info
    with open(f'{RESULTS_DIR}/split_info.json') as f:
        splits = json.load(f)

    # Load DeepSeek Direct results (only keep valid, prefer first or last doesn't matter if one is valid)
    ds_results = {}
    with open(f'{FEATURES_DIR}/llm_results.jsonl', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('method') == 'direct' and r.get('temperature', 0.0) == 0.0:
                if r.get('scores') is not None and len(r['scores']) >= 8:
                    ds_results[r['video_id']] = r['scores']

    # Load GPT-5.6 Direct results
    gpt_results = {}
    with open(f'{FEATURES_DIR}/llm_results_codexapis.jsonl', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('scores') is not None and len(r['scores']) >= 8:
                gpt_results[r['video_id']] = r['scores']

    # Load GPT-5.6 CoT results (200 subset)
    gpt_cot_results = {}
    try:
        with open(f'{FEATURES_DIR}/llm_results_codexapis_cot.jsonl', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                gpt_cot_results[r['video_id']] = r['scores']
    except FileNotFoundError:
        pass

    # Load all features metadata for video_id mapping
    video_ids = []
    with open(f'{FEATURES_DIR}/all_features.jsonl', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            video_ids.append(d['video_id'])

    return targets, splits, video_ids, ds_results, gpt_results, gpt_cot_results


def build_matrices(video_ids, targets_array, llm_results, test_ratio=0.3, seed=42, min_train=50):
    """Build aligned GT and LLM score matrices with random split.

    Uses simple random split since pre-computed split_info.json is test-only (10 users).
    For diagnostic purposes (checking if calibration fixes sign), this is sufficient.
    """
    rng = np.random.RandomState(seed)

    # Map video_id to index
    vid_to_idx = {v: i for i, v in enumerate(video_ids)}

    # Collect all aligned rows
    all_gt, all_llm, all_vids = [], [], []
    for vid in llm_results:
        if vid in vid_to_idx and llm_results[vid] is not None:
            idx = vid_to_idx[vid]
            gt = targets_array[idx]
            scores = llm_results[vid]
            llm = np.array([scores.get(t, 0.0) if isinstance(scores, dict) else 0.0 for t in TARGETS])
            if not (np.any(np.isnan(gt)) or np.any(np.isnan(llm))):
                all_gt.append(gt)
                all_llm.append(llm)
                all_vids.append(vid)

    all_gt = np.array(all_gt)
    all_llm = np.array(all_llm)
    n = len(all_vids)

    # Random split
    indices = rng.permutation(n)
    n_test = max(int(n * test_ratio), 1)
    n_train = n - n_test

    if n_train < min_train:
        print(f"  WARNING: Only {n_train} train samples (need {min_train}). Using all data.")
        return all_llm, all_gt, all_llm, all_gt

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, y_train = all_llm[train_idx], all_gt[train_idx]
    X_test, y_test = all_llm[test_idx], all_gt[test_idx]
    return X_train, y_train, X_test, y_test


def run_calibration(gt_train, llm_train, gt_test, llm_test, method_name):
    """Run per-dimension and multi-output calibration."""
    n_dims = len(TARGETS)
    results = {
        'method': method_name,
        'per_dimension': {},
        'macro': {},
    }

    # --- Per-dimension OLS ---
    raw_rs, cal_rs, sign_flip_rs = [], [], []
    for d in range(n_dims):
        y_tr = gt_train[:, d]
        x_tr = llm_train[:, d].reshape(-1, 1)
        y_te = gt_test[:, d]
        x_te = llm_test[:, d].reshape(-1, 1)

        # Raw correlation
        raw_r, _ = pearsonr(y_te.ravel(), x_te.ravel())
        raw_rs.append(raw_r)

        # Affine calibration: GT = a + b * LLM
        ols = LinearRegression()
        ols.fit(x_tr, y_tr)
        cal_pred = ols.predict(x_te)
        cal_r, _ = pearsonr(y_te, cal_pred.ravel())
        cal_rs.append(cal_r)

        # Sign-flipped calibration
        ols_flip = LinearRegression()
        ols_flip.fit(-x_tr, y_tr)
        flip_pred = ols_flip.predict(-x_te)
        flip_r, _ = pearsonr(y_te, flip_pred.ravel())
        sign_flip_rs.append(flip_r)

        results['per_dimension'][TARGETS[d]] = {
            'raw_r': round(raw_r, 4),
            'calibrated_r': round(cal_r, 4),
            'sign_flipped_r': round(flip_r, 4),
            'slope': round(ols.coef_[0], 4),
            'intercept': round(ols.intercept_, 4),
            'improvement': round(cal_r - raw_r, 4),
        }

    # --- Multi-output Ridge (12 dims jointly) ---
    ridge = Ridge(alpha=1.0)
    ridge.fit(llm_train, gt_train)
    ridge_pred = ridge.predict(llm_test)

    multi_rs = []
    for d in range(n_dims):
        r, _ = pearsonr(gt_test[:, d], ridge_pred[:, d])
        multi_rs.append(r)

    # --- Sign-reversed multi-output ---
    ridge_flip = Ridge(alpha=1.0)
    ridge_flip.fit(-llm_train, gt_train)
    ridge_flip_pred = ridge_flip.predict(-llm_test)

    flip_multi_rs = []
    for d in range(n_dims):
        r, _ = pearsonr(gt_test[:, d], ridge_flip_pred[:, d])
        flip_multi_rs.append(r)

    results['macro'] = {
        'raw_macro_r': round(float(np.mean(raw_rs)), 4),
        'calibrated_macro_r': round(float(np.mean(cal_rs)), 4),
        'sign_flipped_macro_r': round(float(np.mean(sign_flip_rs)), 4),
        'ridge_multi_macro_r': round(float(np.mean(multi_rs)), 4),
        'ridge_flip_multi_macro_r': round(float(np.mean(flip_multi_rs)), 4),
    }

    # Key diagnostic: does calibration make r positive?
    n_positive_raw = sum(1 for r in raw_rs if r > 0)
    n_positive_cal = sum(1 for r in cal_rs if r > 0)
    n_positive_flip = sum(1 for r in sign_flip_rs if r > 0)

    results['diagnostic'] = {
        'n_dims_positive_raw': n_positive_raw,
        'n_dims_positive_calibrated': n_positive_cal,
        'n_dims_positive_sign_flipped': n_positive_flip,
        'calibration_fixes_sign': n_positive_cal > n_positive_raw,
        'conclusion': (
            'Scale/polarity issue — calibration substantially improves alignment'
            if n_positive_cal >= n_dims // 2
            else 'Genuine ranking mismatch — calibration cannot fix the sign'
        ),
    }

    return results


def main():
    print("Loading data...")
    targets, _, video_ids, ds_results, gpt_results, gpt_cot_results = load_data()

    all_calibrations = {}

    # 1. DeepSeek Direct calibration
    print("\n" + "=" * 60)
    print("1. DeepSeek Direct Calibration")
    print("=" * 60)
    name = 'deepseek_direct'
    X_train, y_train, X_test, y_test = build_matrices(
        video_ids, targets, ds_results)
    print(f"Train: {X_train.shape[0]} videos, Test: {X_test.shape[0]} videos")
    result = run_calibration(y_train, X_train, y_test, X_test, name)
    all_calibrations[name] = result
    print(f"  Raw macro r: {result['macro']['raw_macro_r']}")
    print(f"  Calibrated macro r: {result['macro']['calibrated_macro_r']}")
    print(f"  Sign-flipped macro r: {result['macro']['sign_flipped_macro_r']}")
    print(f"  Ridge multi macro r: {result['macro']['ridge_multi_macro_r']}")
    print(f"  {result['diagnostic']['conclusion']}")

    # 2. GPT-5.6 Direct calibration
    print("\n" + "=" * 60)
    print("2. GPT-5.6 Direct Calibration")
    print("=" * 60)
    name = 'gpt56_direct'
    X_train2, y_train2, X_test2, y_test2 = build_matrices(
        video_ids, targets, gpt_results)
    print(f"Train: {X_train2.shape[0]} videos, Test: {X_test2.shape[0]} videos")
    result2 = run_calibration(y_train2, X_train2, y_test2, X_test2, name)
    all_calibrations[name] = result2
    print(f"  Raw macro r: {result2['macro']['raw_macro_r']}")
    print(f"  Calibrated macro r: {result2['macro']['calibrated_macro_r']}")
    print(f"  Sign-flipped macro r: {result2['macro']['sign_flipped_macro_r']}")
    print(f"  Ridge multi macro r: {result2['macro']['ridge_multi_macro_r']}")
    print(f"  {result2['diagnostic']['conclusion']}")

    # 3. GPT-5.6 CoT calibration (200 subset only)
    if gpt_cot_results:
        print("\n" + "=" * 60)
        print("3. GPT-5.6 CoT Calibration (200 subset)")
        print("=" * 60)
        name = 'gpt56_cot'
        X_train3, y_train3, X_test3, y_test3 = build_matrices(
            video_ids, targets, gpt_cot_results)
        print(f"Train: {X_train3.shape[0]} videos, Test: {X_test3.shape[0]} videos")
        if X_train3.shape[0] > 5 and X_test3.shape[0] > 5:
            result3 = run_calibration(y_train3, X_train3, y_test3, X_test3, name)
            all_calibrations[name] = result3
            print(f"  Raw macro r: {result3['macro']['raw_macro_r']}")
            print(f"  Calibrated macro r: {result3['macro']['calibrated_macro_r']}")
            print(f"  {result3['diagnostic']['conclusion']}")
        else:
            print("  Insufficient data for calibration")

    # 4. Per-dimension summary table
    print("\n" + "=" * 60)
    print("4. Per-Dimension Summary (DeepSeek Direct)")
    print("=" * 60)
    print(f"{'Dimension':<25} {'Raw r':>8} {'Calib r':>8} {'Flip r':>8} {'Slope':>8} {'Verdict'}")
    print("-" * 70)
    for dim in TARGETS:
        d = all_calibrations['deepseek_direct']['per_dimension'][dim]
        verdict = 'FIXED' if d['improvement'] > 0.05 else ('FLIP' if d['sign_flipped_r'] > d['calibrated_r'] else 'STUCK')
        print(f"{dim:<25} {d['raw_r']:>+8.4f} {d['calibrated_r']:>+8.4f} {d['sign_flipped_r']:>+8.4f} {d['slope']:>+8.4f} {verdict}")

    # Save
    output_path = f'{RESULTS_DIR}/calibration_analysis.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_calibrations, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {output_path}")

    return all_calibrations


if __name__ == '__main__':
    results = main()
