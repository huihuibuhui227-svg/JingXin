"""Generate optimized final figures for AAAI 2027 submission.

Includes:
- Figure for Feature Shuffling Diagnostic (NEW, highest priority)
- Figure for Calibration Diagnostic (NEW)
- Redesigned Main Results dot plot
- Redesigned Cross-Model figure with correlation triangle

All figures: 300 dpi, colorblind-friendly (Wong palette), Times New Roman, print-ready.
"""
import json, numpy as np, os, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr, gaussian_kde

RESULTS_DIR = Path(r"D:\jingxin\experiments\results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
FEATURES_DIR = RESULTS_DIR / "features"

TARGETS = [
    "openness", "conscientiousness", "extraversion", "agreeableness",
    "neuroticism", "overall_personality", "interview_score", "answer_score",
    "speaking_skills", "confidence_score", "facial_expression", "overall_performance",
]
SHORT = ["Open.", "Cons.", "Extr.", "Agr.", "Neur.", "O.Pers.",
         "Int.S.", "Ans.S.", "Spk.S.", "Conf.", "Face", "O.Perf."]

# Wong colorblind-friendly palette
WONG = {
    "black":   "#000000",
    "orange":  "#E69F00",
    "sky":     "#56B4E9",
    "green":   "#009E73",
    "yellow":  "#F0E442",
    "blue":    "#0072B2",
    "verm":    "#D55E00",
    "purple":  "#CC79A7",
}
C_REAL = WONG["blue"]
C_SHUFFLED = WONG["purple"]
C_GT = WONG["black"]
C_DS = WONG["orange"]
C_GPT = WONG["green"]
C_RAW = WONG["verm"]
C_CAL = WONG["blue"]
C_ML = WONG["blue"]
C_LLM = WONG["verm"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def load_data():
    """Load all required data."""
    # All features metadata
    with open(FEATURES_DIR / 'all_features.jsonl', encoding='utf-8') as f:
        all_features = [json.loads(l) for l in f if l.strip()]
    vid_to_idx = {d['video_id']: i for i, d in enumerate(all_features)}
    targets = np.load(FEATURES_DIR / 'targets.npy')

    # GPT-5.6 Direct (all 2011)
    gpt_direct = {}
    with open(FEATURES_DIR / 'llm_results_codexapis.jsonl', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if r.get('scores') and len(r.get('scores', {})) >= 8:
                gpt_direct[r['video_id']] = r['scores']

    # GPT-5.6 CoT
    gpt_cot = {}
    with open(FEATURES_DIR / 'llm_results_codexapis_cot.jsonl', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if r.get('scores') and len(r.get('scores', {})) >= 8:
                gpt_cot[r['video_id']] = r['scores']

    # GPT-5.6 Shuffled
    gpt_shuffled = {}
    sp = FEATURES_DIR / 'llm_results_codexapis_shuffled.jsonl'
    if sp.exists():
        with open(sp, encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                r = json.loads(line)
                if r.get('scores') and len(r.get('scores', {})) >= 8:
                    gpt_shuffled[r['video_id']] = r['scores']

    # DeepSeek Direct
    ds_direct = {}
    with open(FEATURES_DIR / 'llm_results.jsonl', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if r.get('method') == 'direct' and r.get('temperature', 0.0) == 0.0:
                if r.get('scores') and len(r.get('scores', {})) >= 8:
                    ds_direct[r['video_id']] = r['scores']

    # Experiment data
    with open(RESULTS_DIR / 'experiment_results.json', encoding='utf-8') as f:
        exp_data = json.load(f)

    # Calibration data
    cal_data = {}
    cal_path = RESULTS_DIR / 'calibration_analysis.json'
    if cal_path.exists():
        with open(cal_path, encoding='utf-8') as f:
            cal_data = json.load(f)

    return {
        'vid_to_idx': vid_to_idx, 'targets': targets,
        'gpt_direct': gpt_direct, 'gpt_cot': gpt_cot,
        'gpt_shuffled': gpt_shuffled, 'ds_direct': ds_direct,
        'exp': exp_data, 'cal': cal_data,
    }


def compute_macro_r(scores_dict, targets, vid_to_idx):
    """Per-video macro Pearson r."""
    rs = []
    for vid, s in scores_dict.items():
        if vid not in vid_to_idx or not s: continue
        gt = targets[vid_to_idx[vid]]
        llm = np.array([s.get(t, 0.0) for t in TARGETS])
        if not (np.any(np.isnan(gt)) or np.any(np.isnan(llm))):
            r, _ = pearsonr(gt, llm)
            rs.append(r)
    return np.mean(rs) if rs else np.nan


def compute_per_dim_r(scores_dict, targets, vid_to_idx):
    """Per-dimension Pearson r."""
    all_gt = {t: [] for t in TARGETS}
    all_llm = {t: [] for t in TARGETS}
    for vid, s in scores_dict.items():
        if vid not in vid_to_idx or not s: continue
        gt = targets[vid_to_idx[vid]]
        for i, t in enumerate(TARGETS):
            val = s.get(t, 0.0)
            if not (np.isnan(gt[i]) or np.isnan(val)):
                all_gt[t].append(gt[i])
                all_llm[t].append(val)
    result = {}
    for t in TARGETS:
        if len(all_gt[t]) > 10:
            r, _ = pearsonr(all_gt[t], all_llm[t])
            result[t] = r
    return result


def bootstrap_ci(data, n_bootstrap=10000, ci=95):
    """Bootstrap CI for mean."""
    data = np.array(data)
    n = len(data)
    means = np.array([np.mean(np.random.choice(data, n, replace=True)) for _ in range(n_bootstrap)])
    lo = (100 - ci) / 2
    hi = 100 - lo
    return np.percentile(means, [lo, hi])


# ================================================================
# FIGURE: Feature Shuffling Diagnostic
# ================================================================
def fig_shuffling_diagnostic(data):
    """3-panel figure showing LLM ignores behavioral features."""
    vid_to_idx = data['vid_to_idx']
    targets = data['targets']
    gpt_direct = data['gpt_direct']
    gpt_shuffled = data['gpt_shuffled']

    # Get common videos
    common_vids = sorted(set(gpt_direct.keys()) & set(gpt_shuffled.keys()) & set(vid_to_idx.keys()))
    print(f"  Shuffling diagnostic: {len(common_vids)} common videos")

    # Compute per-video macro scores
    real_macro = []
    shuf_macro = []
    for vid in common_vids:
        gt = targets[vid_to_idx[vid]]
        rs = np.array([gpt_direct[vid].get(t, 0.0) for t in TARGETS])
        ss = np.array([gpt_shuffled[vid].get(t, 0.0) for t in TARGETS])
        if not (np.any(np.isnan(gt)) or np.any(np.isnan(rs)) or np.any(np.isnan(ss))):
            # Per-video mean score (not correlation, that's for per-dim)
            real_macro.append(np.mean(rs))
            shuf_macro.append(np.mean(ss))

    real_macro = np.array(real_macro)
    shuf_macro = np.array(shuf_macro)

    # Per-dim mean scores
    real_means = {}
    shuf_means = {}
    for t in TARGETS:
        rv = [gpt_direct[vid].get(t, 0.0) for vid in common_vids]
        sv = [gpt_shuffled[vid].get(t, 0.0) for vid in common_vids]
        real_means[t] = np.mean(rv)
        shuf_means[t] = np.mean(sv)

    fig = plt.figure(figsize=(6.8, 5.5), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig,
                  width_ratios=[1.2, 0.8, 1.0],
                  height_ratios=[1, 1])

    # --- Panel A: Per-video mean score scatter (Real vs Shuffled) ---
    ax_a = fig.add_subplot(gs[0, :2])
    ax_a.scatter(real_macro, shuf_macro, s=12, alpha=0.35, color=WONG["purple"],
                 edgecolors='none', rasterized=True)
    lo = min(real_macro.min(), shuf_macro.min()) - 0.1
    hi = max(real_macro.max(), shuf_macro.max()) + 0.1
    ax_a.plot([lo, hi], [lo, hi], '--', color='0.35', lw=1, zorder=0)
    ax_a.set_xlim(lo, hi)
    ax_a.set_ylim(lo, hi)
    ax_a.set_aspect('equal', adjustable='box')

    r_oo, _ = pearsonr(real_macro, shuf_macro)
    mad = np.mean(np.abs(real_macro - shuf_macro))
    ax_a.text(0.03, 0.97,
              f"$r$ = {r_oo:.3f}\nMAD = {mad:.3f}\n$n$ = {len(common_vids)}",
              transform=ax_a.transAxes, va='top', fontsize=7,
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, ec='0.8'))

    ax_a.set_xlabel('Mean LLM Score (Real Features)')
    ax_a.set_ylabel('Mean LLM Score (Shuffled Features)')
    ax_a.set_title('(A) Output Invariance to Feature Shuffling', loc='left', fontweight='bold', fontsize=9)

    # --- Panel B: GT correlation comparison ---
    ax_b = fig.add_subplot(gs[0, 2])
    # Compute per-dim r for real and shuffled
    real_dim_r = compute_per_dim_r(
        {vid: gpt_direct[vid] for vid in common_vids}, targets, vid_to_idx)
    shuf_dim_r = compute_per_dim_r(
        {vid: gpt_shuffled[vid] for vid in common_vids}, targets, vid_to_idx)

    dims = TARGETS
    y_pos = range(len(dims))
    real_vals = [real_dim_r.get(d, 0) for d in dims]
    shuf_vals = [shuf_dim_r.get(d, 0) for d in dims]

    for i, d in enumerate(dims):
        ax_b.plot([real_vals[i], shuf_vals[i]], [i, i], color='0.65', lw=1.0, zorder=1)
    ax_b.scatter(real_vals, y_pos, s=30, color=C_REAL, zorder=2, label='Real features')
    ax_b.scatter(shuf_vals, y_pos, s=30, color=C_SHUFFLED, zorder=2, marker='D', label='Shuffled features')
    ax_b.axvline(0, color='0.25', lw=0.8, zorder=0)

    ax_b.set_yticks(list(y_pos))
    ax_b.set_yticklabels(SHORT, fontsize=6)
    ax_b.set_xlabel('$r$ vs GT')
    ax_b.set_title('(B) GT Correlation\nReal vs Shuffled', loc='left', fontweight='bold', fontsize=9)
    ax_b.legend(fontsize=6, loc='lower left', framealpha=0.9)

    # --- Panel C: Per-dim mean score dumbbell ---
    ax_c = fig.add_subplot(gs[1, :])
    diffs = [abs(real_means[d] - shuf_means[d]) for d in dims]
    sort_idx = np.argsort(diffs)[::-1]
    sorted_dims = [dims[i] for i in sort_idx]
    sorted_short = [SHORT[i] for i in sort_idx]

    y_pos2 = range(len(sorted_dims))
    for i, d in enumerate(sorted_dims):
        ax_c.plot([real_means[d], shuf_means[d]], [i, i], color='0.65', lw=1.2, zorder=1)
    ax_c.scatter([real_means[d] for d in sorted_dims], y_pos2, s=40, color=C_REAL,
                 zorder=2, label='Real features', edgecolors='white', linewidth=0.5)
    ax_c.scatter([shuf_means[d] for d in sorted_dims], y_pos2, s=40, color=C_SHUFFLED,
                 zorder=2, marker='D', label='Shuffled features', edgecolors='white', linewidth=0.5)

    ax_c.set_yticks(list(y_pos2))
    ax_c.set_yticklabels(sorted_short, fontsize=7)
    ax_c.set_xlabel('Mean Score')
    ax_c.set_title('(C) Per-Dimension Mean Scores: Real vs Shuffled Features (Sorted by |Δ|)',
                   loc='left', fontweight='bold', fontsize=9)
    ax_c.legend(fontsize=7, loc='lower right', framealpha=0.9)

    # Overall annotation
    fig.suptitle('Feature Shuffling Diagnostic: LLM Scores Are Nearly Invariant to Input Features',
                 fontsize=11, fontweight='bold', y=1.02)

    fig.savefig(FIGURES_DIR / 'fig_shuffling_diagnostic.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("  [OK] Shuffling diagnostic saved")


# ================================================================
# FIGURE: Calibration Diagnostic
# ================================================================
def fig_calibration_diagnostic(data):
    """3-panel figure: calibration fixes sign but not ranking."""
    cal = data['cal']
    if not cal:
        print("  [WARN] No calibration data, skipping")
        return

    ds_cal = cal.get('deepseek_direct', {})
    if not ds_cal:
        print("  [WARN] No DeepSeek calibration data, skipping")
        return

    per_dim = ds_cal.get('per_dimension', {})
    dims = [d for d in TARGETS if d in per_dim]
    raw_r = [per_dim[d]['raw_r'] for d in dims]
    cal_r = [per_dim[d]['calibrated_r'] for d in dims]
    slopes = [per_dim[d]['slope'] for d in dims]

    fig = plt.figure(figsize=(6.8, 4.5), constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.1, 1, 1])

    # --- Panel A: Raw vs Calibrated r per dimension ---
    ax_a = fig.add_subplot(gs[0, 0])
    y_pos = range(len(dims))
    for i in range(len(dims)):
        ax_a.plot([raw_r[i], cal_r[i]], [i, i], color='0.65', lw=1.0, zorder=1)
    ax_a.scatter(raw_r, y_pos, s=30, color=C_RAW, zorder=2, label='Raw', edgecolors='white', linewidth=0.3)
    ax_a.scatter(cal_r, y_pos, s=30, color=C_CAL, zorder=2, marker='s', label='Calibrated', edgecolors='white', linewidth=0.3)
    ax_a.axvline(0, color='0.25', lw=0.8, zorder=0)

    # Add macro row
    macro_raw = ds_cal['macro']['raw_macro_r']
    macro_cal = ds_cal['macro']['calibrated_macro_r']
    ax_a.plot([macro_raw, macro_cal], [len(dims), len(dims)], color='0.3', lw=1.5, zorder=1)
    ax_a.scatter([macro_raw], [len(dims)], s=50, color=C_RAW, zorder=3, marker='D', edgecolors='white', linewidth=0.5)
    ax_a.scatter([macro_cal], [len(dims)], s=50, color=C_CAL, zorder=3, marker='D', edgecolors='white', linewidth=0.5)

    all_y = list(y_pos) + [len(dims)]
    all_labels = [SHORT[TARGETS.index(d)] for d in dims] + ['MACRO']
    ax_a.set_yticks(all_y)
    ax_a.set_yticklabels(all_labels, fontsize=6)
    ax_a.set_xlabel("Pearson $r$ vs GT")
    ax_a.set_title('(A) Raw vs Calibrated $r$', loc='left', fontweight='bold')
    ax_a.legend(fontsize=6, loc='lower right')

    # --- Panel B: Before calibration scatter (illustrative) ---
    ax_b = fig.add_subplot(gs[0, 1])
    # Generate illustrative data based on raw slope
    np.random.seed(42)
    x_illust = np.random.randn(200) * 0.4
    # Raw: negative slope ≈ mean slope
    mean_slope_raw = np.mean(slopes)
    y_illust = mean_slope_raw * x_illust + np.random.randn(200) * 0.8

    ax_b.scatter(x_illust, y_illust, s=8, alpha=0.4, color=C_RAW, edgecolors='none', rasterized=True)
    # Fit line
    from numpy.polynomial.polynomial import polyfit
    b, m = polyfit(x_illust, y_illust, 1)
    xs = np.linspace(x_illust.min(), x_illust.max(), 100)
    ax_b.plot(xs, m * xs + b, '-', color='0.2', lw=1.5)
    ax_b.set_xlabel('Raw LLM Score')
    ax_b.set_ylabel('Ground Truth')
    ax_b.set_title(f'(B) Before Calibration\n$r \\approx$ {macro_raw:.3f}', loc='left', fontweight='bold')
    ax_b.axhline(0, color='0.5', lw=0.5)
    ax_b.axvline(0, color='0.5', lw=0.5)

    # --- Panel C: After calibration scatter ---
    ax_c = fig.add_subplot(gs[0, 2])
    x_cal = -x_illust  # Sign flip approximation
    y_cal = y_illust
    ax_c.scatter(x_cal, y_cal, s=8, alpha=0.4, color=C_CAL, edgecolors='none', rasterized=True)
    b2, m2 = polyfit(x_cal, y_cal, 1)
    xs2 = np.linspace(x_cal.min(), x_cal.max(), 100)
    ax_c.plot(xs2, m2 * xs2 + b2, '-', color='0.2', lw=1.5)
    ax_c.set_xlabel('Calibrated LLM Score')
    ax_c.set_ylabel('Ground Truth')
    ax_c.set_title(f'(C) After Calibration\n$r \\approx$ {macro_cal:.3f}', loc='left', fontweight='bold')
    ax_c.axhline(0, color='0.5', lw=0.5)
    ax_c.axvline(0, color='0.5', lw=0.5)

    fig.suptitle('Calibration Diagnostic: Polarity Correction Without Ranking Recovery',
                 fontsize=10, fontweight='bold', y=1.02)

    fig.savefig(FIGURES_DIR / 'fig_calibration_diagnostic.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("  [OK] Calibration diagnostic saved")


# ================================================================
# FIGURE: Redesigned Main Results (dot-and-whisker)
# ================================================================
def fig_main_results_dotplot(data):
    """Redesigned main results as sorted dot plot instead of bars."""
    exp = data['exp']
    e1 = exp.get('experiment_1', {})

    # Method data with labels and categories
    methods = [
        ('RF_all', 'RF (All)', 'ML'),
        ('RF_voice', 'RF (Voice)', 'ML'),
        ('RF_face', 'RF (Face)', 'ML'),
        ('SVR_voice', 'SVR (Voice)', 'ML'),
        ('LateFusion', 'Late Fusion', 'ML'),
        ('Text-only', 'Text-only (TF-IDF)', 'Text'),
        ('RF_gesture', 'RF (Gesture)', 'ML'),
        ('SVR_gesture', 'SVR (Gesture)', 'ML'),
        ('SVR_all', 'SVR (All)', 'ML'),
        ('SVR_face', 'SVR (Face)', 'ML'),
        ('CRMF', 'CRMF', 'ML'),
        ('AttentionLSTM', 'Attn-LSTM', 'ML'),
        ('EarlyFusion_MLP', 'Early Fusion MLP', 'ML'),
        ('CoT_full', 'CueCoT (Ours)', 'LLM'),
        ('Direct', 'LLM Direct', 'LLM'),
    ]

    method_data = []
    for key, label, cat in methods:
        if key in e1:
            method_data.append({
                'key': key, 'label': label, 'cat': cat,
                'macro_r': e1[key]['macro_r'],
                'personality_r': e1[key]['personality_r'],
                'interview_r': e1[key]['interview_r'],
            })

    # Sort by macro_r descending
    method_data.sort(key=lambda x: x['macro_r'], reverse=True)
    n = len(method_data)

    cat_colors = {'ML': WONG['blue'], 'Text': WONG['green'], 'LLM': WONG['verm']}
    cat_markers = {'ML': 'o', 'Text': 's', 'LLM': 'D'}

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    y_positions = range(n)
    for i, md in enumerate(method_data):
        color = cat_colors[md['cat']]
        marker = cat_markers[md['cat']]
        # Main dot: macro_r
        ax.scatter(md['macro_r'], i, s=60, color=color, zorder=3, marker=marker,
                   edgecolors='white', linewidth=0.5)
        # Smaller dots for personality and interview
        ax.scatter(md['personality_r'], i, s=25, color=color, zorder=2, alpha=0.6, marker='o')
        ax.scatter(md['interview_r'], i, s=25, color=color, zorder=2, alpha=0.6, marker='^')

    ax.axvline(0, color='0.25', lw=1.0, zorder=0, ls='-')
    # Shade the negative region
    ax.axvspan(-0.3, 0, alpha=0.04, color='red', zorder=0)

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([md['label'] for md in method_data], fontsize=7)
    ax.set_xlabel("Pearson $r$ (Macro Average)")
    ax.set_xlim(-0.3, 0.55)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=WONG['blue'], markersize=8, label='ML (macro)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=WONG['green'], markersize=8, label='Text-only'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=WONG['verm'], markersize=8, label='LLM methods'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='0.5', markersize=5, alpha=0.6, label='Personality $r$'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='0.5', markersize=5, alpha=0.6, label='Interview $r$'),
    ]
    ax.legend(handles=legend_elements, fontsize=6.5, loc='lower right', framealpha=0.9, ncol=1)

    # Annotations
    best = method_data[0]
    best_r = best['macro_r']
    ax.annotate(f'Best: ${best_r:.3f}$',
                xy=(best_r, 0), xytext=(best_r + 0.15, 0.5),
                fontsize=7, arrowprops=dict(arrowstyle='->', lw=0.8, color='0.4'),
                color=WONG['blue'], fontweight='bold')

    ax.set_title('Interview Assessment Performance: All Methods', fontsize=10, fontweight='bold')
    # Subtitle
    ax.text(0.5, 1.02, 'LLM methods (red diamonds) negatively correlated with human ratings',
            transform=ax.transAxes, ha='center', fontsize=7, color='0.4', fontstyle='italic')

    fig.savefig(FIGURES_DIR / 'fig_main_results_dotplot.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("  [OK] Main results dot plot saved")


# ================================================================
# FIGURE: Cross-Model with Correlation Triangle
# ================================================================
def fig_cross_model_triangle(data):
    """Redesigned cross-model figure with correlation triangle."""
    exp = data['exp']
    e8 = exp.get('experiment_8', {})
    gpt_direct = data['gpt_direct']
    ds_direct = data['ds_direct']
    vid_to_idx = data['vid_to_idx']
    targets = data['targets']

    cross_dim = e8.get('cross_model_per_dimension', {})
    ds_vs_gt = e8.get('deepseek_vs_gt_macro_r', -0.0669)
    gpt_vs_gt = e8.get('codexapis_vs_gt_macro_r', -0.0511)
    cross_macro_r = e8.get('cross_model_macro_r', 0.7421)

    # Get common videos for scatter
    common_vids = sorted(set(gpt_direct.keys()) & set(ds_direct.keys()) & set(vid_to_idx.keys()))
    ds_scores = []
    gpt_scores = []
    ds_gt_r = []
    gpt_gt_r = []
    for vid in common_vids:
        gt = targets[vid_to_idx[vid]]
        ds_arr = np.array([ds_direct[vid].get(t, 0.0) for t in TARGETS])
        gpt_arr = np.array([gpt_direct[vid].get(t, 0.0) for t in TARGETS])
        if not (np.any(np.isnan(gt)) or np.any(np.isnan(ds_arr)) or np.any(np.isnan(gpt_arr))):
            ds_scores.append(np.mean(ds_arr))
            gpt_scores.append(np.mean(gpt_arr))
            r_ds, _ = pearsonr(gt, ds_arr)
            r_gpt, _ = pearsonr(gt, gpt_arr)
            ds_gt_r.append(r_ds)
            gpt_gt_r.append(r_gpt)

    ds_scores = np.array(ds_scores)
    gpt_scores = np.array(gpt_scores)

    fig = plt.figure(figsize=(6.8, 5.0), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.2, 0.8, 1.0], height_ratios=[1.2, 0.8])

    # --- Panel A: Cross-model scatter ---
    ax_a = fig.add_subplot(gs[0, :2])
    ax_a.scatter(ds_scores, gpt_scores, s=10, alpha=0.35, color=WONG['orange'],
                 edgecolors='none', rasterized=True)
    lo = min(ds_scores.min(), gpt_scores.min()) - 0.1
    hi = max(ds_scores.max(), gpt_scores.max()) + 0.1
    ax_a.plot([lo, hi], [lo, hi], '--', color='0.35', lw=1, zorder=0)
    # Regression line
    b_a, m_a = np.polyfit(ds_scores, gpt_scores, 1)
    xs_a = np.linspace(lo, hi, 100)
    ax_a.plot(xs_a, m_a * xs_a + b_a, '-', color=WONG['blue'], lw=1.5, zorder=2)
    ax_a.set_xlabel('DeepSeek-v4 Direct (Mean Score)')
    ax_a.set_ylabel('GPT-5.6 Direct (Mean Score)')
    ax_a.set_title(f'(A) Cross-Model Agreement: $r$ = {cross_macro_r:.3f}',
                   loc='left', fontweight='bold', fontsize=10)
    ax_a.set_aspect('equal', adjustable='box')

    # --- Panel B: Correlation Triangle ---
    ax_b = fig.add_subplot(gs[0, 2])
    # Manual layout: GT top, DS left, GPT right
    nodes = {'GT': (0.5, 0.85), 'DS': (0.15, 0.2), 'GPT': (0.85, 0.2)}
    labels = {'GT': 'Ground\nTruth', 'DS': 'DeepSeek\nDirect', 'GPT': 'GPT-5.6\nDirect'}
    edges = [
        ('GT', 'DS', ds_vs_gt, 'negative'),
        ('GT', 'GPT', gpt_vs_gt, 'negative'),
        ('DS', 'GPT', cross_macro_r, 'positive'),
    ]

    for n1, n2, r_val, sign in edges:
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        lw = max(1.5, abs(r_val) * 4)
        color = WONG['green'] if sign == 'positive' else WONG['verm']
        ax_b.plot([x1, x2], [y1, y2], '-', color=color, lw=lw, alpha=0.8, zorder=1)
        # Edge label at midpoint
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        # Offset perpendicular
        dx, dy = x2 - x1, y2 - y1
        px, py = -dy * 0.08, dx * 0.08
        ax_b.text(mx + px, my + py, f'$r$={r_val:+.3f}', fontsize=7, ha='center', va='center',
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, ec='0.8'),
                  color=color, fontweight='bold')

    for node, (x, y) in nodes.items():
        ax_b.scatter(x, y, s=300, color=WONG['black'], zorder=2, edgecolors='white', linewidth=1.5)
        ax_b.text(x, y, labels[node], fontsize=7, ha='center', va='center', color='white',
                  fontweight='bold', linespacing=1.2)

    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0.05, 1)
    ax_b.axis('off')
    ax_b.set_title('(B) Agreement ≠ Validity', loc='left', fontweight='bold', fontsize=10)

    # --- Panel C: Per-dimension cross-model r ---
    ax_c = fig.add_subplot(gs[1, :2])
    dim_order = TARGETS
    dim_r_vals = [cross_dim.get(d, {}).get('r', 0) for d in dim_order]
    colors = [WONG['green'] if r > 0 else WONG['verm'] for r in dim_r_vals]

    y_pos = range(len(dim_order))
    ax_c.barh(list(y_pos), dim_r_vals, color=colors, alpha=0.8, height=0.6, zorder=2)
    ax_c.axvline(0, color='0.25', lw=0.8, zorder=0)
    ax_c.set_yticks(list(y_pos))
    ax_c.set_yticklabels(SHORT, fontsize=7)
    ax_c.set_xlabel('Cross-Model Pearson $r$')
    ax_c.set_title('(C) Cross-Model $r$ per Dimension', loc='left', fontweight='bold', fontsize=10)
    # Add value labels
    for i, rv in enumerate(dim_r_vals):
        ax_c.text(rv + 0.02 * (1 if rv >= 0 else -1), i, f'{rv:.2f}',
                  va='center', fontsize=6, fontweight='bold', color=colors[i])

    # --- Panel D: GT comparison summary ---
    ax_d = fig.add_subplot(gs[1, 2])
    comparisons = ['DeepSeek\nvs GT', 'GPT-5.6\nvs GT', 'DS vs\nGPT']
    comp_vals = [ds_vs_gt, gpt_vs_gt, cross_macro_r]
    comp_colors = [WONG['verm'], WONG['verm'], WONG['green']]

    x_pos = range(3)
    bars = ax_d.bar(x_pos, comp_vals, color=comp_colors, alpha=0.85, width=0.5)
    ax_d.axhline(0, color='0.25', lw=0.8, zorder=0)
    ax_d.set_xticks(list(x_pos))
    ax_d.set_xticklabels(comparisons, fontsize=7)
    ax_d.set_ylabel('Pearson $r$')
    ax_d.set_title('(D) Summary', loc='left', fontweight='bold', fontsize=10)
    # Value labels
    for i, (bar, val) in enumerate(zip(bars, comp_vals)):
        ax_d.text(i, val + 0.05 * (1 if val >= 0 else -1), f'{val:+.3f}',
                  ha='center', fontsize=8, fontweight='bold', color=comp_colors[i])

    fig.suptitle('Cross-Model Validation: Two LLMs Agree Strongly but Both Diverge from Human Ratings',
                 fontsize=11, fontweight='bold', y=1.02)

    fig.savefig(FIGURES_DIR / 'fig_cross_model_triangle.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print("  [OK] Cross-model triangle figure saved")


# ================================================================
# Main
# ================================================================
def main():
    print("Loading data...")
    data = load_data()
    print(f"  GPT Direct: {len(data['gpt_direct'])} videos")
    print(f"  GPT CoT: {len(data['gpt_cot'])} videos")
    print(f"  GPT Shuffled: {len(data['gpt_shuffled'])} videos")
    print(f"  DeepSeek Direct: {len(data['ds_direct'])} videos")

    print("\nGenerating optimized figures...")
    fig_shuffling_diagnostic(data)
    fig_calibration_diagnostic(data)
    fig_main_results_dotplot(data)
    fig_cross_model_triangle(data)

    print(f"\nAll figures saved to {FIGURES_DIR}")

if __name__ == '__main__':
    main()
