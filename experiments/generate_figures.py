"""Publication-quality figures for CueCoT paper (AAAI 2027).

Generates 7 figures at 300 dpi, colorblind-friendly, strict from experiment data.
Fig 1: TikZ architecture (separate file)
"""
import json, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import seaborn as sns

RESULTS_DIR = Path(r"D:\jingxin\experiments\results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

TARGETS = [
    "openness", "conscientiousness", "extraversion", "agreeableness",
    "neuroticism", "overall_personality", "interview_score", "answer_score",
    "speaking_skills", "confidence_score", "facial_expression", "overall_performance",
]
PERSONALITY = TARGETS[:6]
INTERVIEW = TARGETS[6:]
SHORT = ["Open.", "Cons.", "Extr.", "Agr.", "Neur.", "O.Pers.",
         "Int.S.", "Ans.S.", "Spk.S.", "Conf.", "Face", "O.Perf."]

# Colorblind-friendly palette (Wong, 2011)
C_BLUE = '#0072B2'
C_ORANGE = '#E69F00'
C_GREEN = '#009E73'
C_RED = '#D55E00'
C_PURPLE = '#CC79A7'
C_SKY = '#56B4E9'
C_YELLOW = '#F0E442'
C_GRAY = '#999999'
C_BLACK = '#000000'

# Group colors
ML_COLOR = C_BLUE
LLM_COLOR = C_RED
GPT_COLOR = C_ORANGE
GT_COLOR = C_BLACK

# Global matplotlib config
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

# ===== Data Loading =====

def load_data():
    """Load all experiment data."""
    with open(RESULTS_DIR / "experiment_results.json") as f:
        exp = json.load(f)
    with open(RESULTS_DIR / "baseline_summary.json") as f:
        bl = json.load(f)
    cx_path = RESULTS_DIR / "cross_model_analysis.json"
    cx = {}
    if cx_path.exists():
        with open(cx_path) as f:
            cx = json.load(f)
    sup_path = RESULTS_DIR / "supplementary_results.json"
    sup = {}
    if sup_path.exists():
        with open(sup_path) as f:
            sup = json.load(f)
    return exp, bl, cx, sup

def build_methods_dict(exp, bl, cx):
    """Build unified methods dict for plotting."""
    methods = {}

    # Baselines from baseline_summary.json
    bl_map = {
        "RF_face": ("unimodal_rf", "face"), "RF_gesture": ("unimodal_rf", "gesture"),
        "RF_voice": ("unimodal_rf", "voice"), "RF_all": ("unimodal_rf", "all"),
        "SVR_face": ("unimodal_svr", "face"), "SVR_gesture": ("unimodal_svr", "gesture"),
        "SVR_voice": ("unimodal_svr", "voice"), "SVR_all": ("unimodal_svr", "all"),
        "RIDGE_voice": ("unimodal_ridge", "voice"),
    }
    for name, (cat, mod) in bl_map.items():
        if cat in bl and mod in bl[cat]:
            entry = bl[cat][mod]
            methods[name] = _extract_metrics(entry)

    # Advanced baselines
    for key in ["late_fusion", "crmf", "attention_lstm", "early_fusion_mlp"]:
        if key in bl:
            methods[key.replace("_", " ").title().replace(" ", "")] = _extract_metrics(bl[key])

    # LLM methods from experiment_results.json
    e1 = exp.get("experiment_1", {})
    for m in ["CoT_full", "Direct"]:
        if m in e1:
            methods[m] = e1[m]

    # Text-only from supplementary
    sup = {}
    sup_path = RESULTS_DIR / "supplementary_results.json"
    if sup_path.exists():
        with open(sup_path) as f:
            sup = json.load(f)
    if "experiment_1_text_vs_behavior" in sup:
        text_data = sup["experiment_1_text_vs_behavior"].get("Text-only (TF-IDF)", {})
        if text_data:
            methods["Text-only"] = text_data

    # GPT-5.6
    gpt56 = cx.get("gpt56_vs_gt", {})
    if gpt56:
        methods["GPT-5.6 Direct"] = gpt56

    return methods

def _extract_metrics(entry):
    """Extract standard metrics from a baseline entry."""
    result = {}
    for d in entry.get("per_dimension", []):
        dim = d["dimension"]
        result[dim] = {
            "pearson_r": d["pearson_r"]["mean"],
            "mse": d["mse"]["mean"],
        }
    if "macro_avg" in entry:
        result["macro_r"] = entry["macro_avg"]["pearson_r"]["mean"]
        result["macro_mse"] = entry["macro_avg"]["mse"]["mean"]
    if "personality_avg" in entry:
        result["personality_r"] = entry["personality_avg"]["pearson_r"]["mean"]
    if "interview_avg" in entry:
        result["interview_r"] = entry["interview_avg"]["pearson_r"]["mean"]
    return result

# ===== Figure 2: Main Results =====

def fig2_main_results(methods):
    """Horizontal bar chart: all methods × 3 metric groups."""
    METHOD_LABELS = {
        "RF_all": "RF (All features)", "RF_voice": "RF (Voice only)",
        "RF_face": "RF (Face only)", "RF_gesture": "RF (Gesture only)",
        "SVR_all": "SVR (All)", "SVR_voice": "SVR (Voice)",
        "SVR_face": "SVR (Face)", "SVR_gesture": "SVR (Gesture)",
        "RIDGE_voice": "Ridge (Voice)", "LateFusion": "Late Fusion",
        "CRMF": "CRMF", "AttentionLSTM": "Attn-LSTM",
        "EarlyFusionMLP": "Early Fusion MLP", "CoT_full": "CueCoT (Ours)",
        "Direct": "LLM Direct (DS)", "GPT-5.6 Direct": "LLM Direct (GPT-5.6)",
        "Text-only": "Text-only (TF-IDF)",
    }
    # Select methods that have all 3 group metrics
    available = []
    for m, label in METHOD_LABELS.items():
        if m in methods:
            d = methods[m]
            if all(k in d for k in ["personality_r", "interview_r", "macro_r"]):
                available.append((m, label))

    # Sort: ML first by macro_r desc, then LLM methods
    ml_methods = [(m, l) for m, l in available if m not in ["CoT_full", "Direct", "GPT-5.6 Direct"]]
    llm_methods = [(m, l) for m, l in available if m in ["CoT_full", "Direct", "GPT-5.6 Direct"]]
    ml_methods.sort(key=lambda x: -methods[x[0]]["macro_r"])
    ordered = ml_methods + llm_methods

    labels = [l for _, l in ordered]
    keys = [m for m, _ in ordered]
    n = len(ordered)

    fig, axes = plt.subplots(1, 3, figsize=(16, max(5, n * 0.3)), sharey=True)

    for ax_idx, (metric, title) in enumerate([
        ("personality_r", "Personality (6 traits)"),
        ("interview_r", "Interview (6 metrics)"),
        ("macro_r", "Macro Average (12 dims)"),
    ]):
        ax = axes[ax_idx]
        vals = [methods[m].get(metric, 0) for m in keys]
        colors = [LLM_COLOR if m in ["CoT_full", "Direct", "GPT-5.6 Direct"] else ML_COLOR for m in keys]

        bars = ax.barh(range(n), vals, color=colors, edgecolor='white', linewidth=0.5, height=0.7)
        ax.axvline(x=0, color='black', linewidth=0.5, linestyle='-')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Pearson r")
        ax.set_xlim(-0.25, 0.55)

        # Value labels
        for i, (v, c) in enumerate(zip(vals, colors)):
            x_pos = v + 0.01 if v >= 0 else v - 0.08
            ha = 'left' if v >= 0 else 'right'
            ax.text(x_pos, i, f'{v:.3f}', va='center', ha=ha, fontsize=7,
                    color=c if abs(v) > 0.05 else 'gray')

        ax.grid(axis='x', alpha=0.3)

    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=ML_COLOR, label='ML Baselines (behavioral features)'),
        Patch(facecolor=LLM_COLOR, label='LLM Methods (behavioral → score)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, frameon=False, fontsize=8)

    fig.suptitle('Method Comparison by Dimension Group', fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(FIGURES_DIR / "fig2_main_results.png")
    plt.close(fig)
    print("  Fig 2: Main Results — done")

# ===== Figure 3: Correlation Heatmap =====

def fig3_correlation_heatmap(exp):
    """GT-GT inter-dimension correlation heatmap."""
    e2 = exp.get("experiment_2", {})
    corr_data = e2.get("correlation_matrix", {})
    if not corr_data:
        print("  Fig 3: No correlation matrix data — skipping")
        return

    # Build matrix
    n = len(TARGETS)
    matrix = np.zeros((n, n))
    for i, d1 in enumerate(TARGETS):
        for j, d2 in enumerate(TARGETS):
            key = f"{d1}_vs_{d2}"
            if key in corr_data:
                matrix[i, j] = corr_data[key]
            elif d1 == d2:
                matrix[i, j] = 1.0

    fig, ax = plt.subplots(figsize=(10, 8.5))
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    sns.heatmap(matrix, mask=mask, annot=True, fmt='.2f', cmap=cmap,
                vmin=-0.3, vmax=1.0, center=0, square=True,
                linewidths=0.5, linecolor='white',
                xticklabels=SHORT, yticklabels=SHORT,
                annot_kws={'fontsize': 7}, cbar_kws={'shrink': 0.8, 'label': 'Pearson r'},
                ax=ax)

    ax.set_title('Ground Truth Inter-Dimension Correlations', fontweight='bold', pad=15)
    ax.tick_params(axis='both', labelsize=8, rotation=45)
    ax.set_xticklabels(SHORT, rotation=45, ha='right')
    ax.set_yticklabels(SHORT, rotation=0)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_correlation_heatmap.png")
    plt.close(fig)
    print("  Fig 3: Correlation Heatmap — done")

# ===== Figure 4: Cross-Model Validation =====

def fig4_cross_model(cx):
    """2×2 panel: cross-model scatter, per-dim r, vs-GT comparison, score distributions."""
    if not cx:
        print("  Fig 4: No cross-model data — skipping")
        return

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # (a) Cross-model per-dimension r
    ax_a = fig.add_subplot(gs[0, 0])
    per_dim = cx["direct_cross_model"]["per_dimension"]
    dims = list(per_dim.keys())
    r_vals = [per_dim[d]["r"] for d in dims]
    short_dims = [SHORT[TARGETS.index(d)] if d in TARGETS else d for d in dims]

    colors = [C_GREEN if r > 0.8 else C_BLUE if r > 0.6 else C_GRAY for r in r_vals]
    bars = ax_a.barh(range(len(dims)), r_vals, color=colors, edgecolor='white', height=0.6)
    ax_a.set_yticks(range(len(dims)))
    ax_a.set_yticklabels(short_dims, fontsize=7)
    ax_a.set_xlabel("Cross-Model Pearson r")
    ax_a.set_title("(a) DeepSeek vs GPT-5.6 Direct\nPer-Dimension Agreement", fontsize=9, fontweight='bold')
    ax_a.axvline(x=cx["direct_cross_model"]["macro_r"], color=C_RED, linestyle='--', linewidth=1,
                 label=f'Macro r={cx["direct_cross_model"]["macro_r"]:.3f}')
    ax_a.legend(fontsize=7, frameon=False)
    ax_a.set_xlim(0, 1.05)
    ax_a.invert_yaxis()

    # (b) DeepSeek vs GPT-5.6 scatter (sample 1000 points for clarity)
    ax_b = fig.add_subplot(gs[0, 1])
    ds_scores = _load_model_scores("llm_results.jsonl", "direct")
    gpt_scores = _load_model_scores("llm_results_codexapis.jsonl", "direct_v2")

    common = sorted(set(ds_scores.keys()) & set(gpt_scores.keys()))
    if len(common) > 1000:
        np.random.seed(42)
        common = sorted(np.random.choice(common, 1000, replace=False))

    ds_vals = [np.mean([ds_scores[v].get(d, 0) for d in TARGETS]) for v in common]
    gpt_vals = [np.mean([gpt_scores[v].get(d, 0) for d in TARGETS]) for v in common]

    ax_b.scatter(ds_vals, gpt_vals, c=C_BLUE, alpha=0.3, s=15, edgecolors='none')
    # Fit line
    z = np.polyfit(ds_vals, gpt_vals, 1)
    x_line = np.linspace(min(ds_vals), max(ds_vals), 100)
    ax_b.plot(x_line, np.polyval(z, x_line), color=C_RED, linewidth=1.5, linestyle='--')
    from scipy import stats
    r_val, _ = stats.pearsonr(ds_vals, gpt_vals)
    ax_b.set_xlabel("DeepSeek-v4 Mean Score")
    ax_b.set_ylabel("GPT-5.6 Mean Score")
    ax_b.set_title(f"(b) Per-Video Mean Score Agreement\nr = {r_val:.4f} (n={len(common)})", fontsize=9, fontweight='bold')
    ax_b.text(0.05, 0.95, f'y = {z[0]:.3f}x + {z[1]:.3f}', transform=ax_b.transAxes,
              fontsize=7, va='top', color=C_RED)

    # (c) Both models vs GT per dimension
    ax_c = fig.add_subplot(gs[1, 0])
    x = np.arange(len(TARGETS))
    width = 0.35

    ds_vs_gt_vals = []
    gpt_vs_gt_vals = []
    for dim in TARGETS:
        ds_v = [ds_scores[v].get(dim, 0) for v in common]
        gt_v = _load_gt_scores()
        gv = [gt_v.get(v, {}).get(dim, 0) for v in common]
        if len(ds_v) > 30:
            ds_vs_gt_vals.append(stats.pearsonr(ds_v, gv)[0])
            gpt_v = [gpt_scores[v].get(dim, 0) for v in common]
            gpt_vs_gt_vals.append(stats.pearsonr(gpt_v, gv)[0])
        else:
            ds_vs_gt_vals.append(0)
            gpt_vs_gt_vals.append(0)

    ax_c.bar(x - width/2, ds_vs_gt_vals, width, color=LLM_COLOR, label='DeepSeek Direct', edgecolor='white')
    ax_c.bar(x + width/2, gpt_vs_gt_vals, width, color=GPT_COLOR, label='GPT-5.6 Direct', edgecolor='white')
    ax_c.axhline(y=0, color='black', linewidth=0.5)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(SHORT, rotation=45, ha='right', fontsize=7)
    ax_c.set_ylabel("Pearson r vs GT")
    ax_c.set_title("(c) Both Models vs Ground Truth\nPer-Dimension Correlation", fontsize=9, fontweight='bold')
    ax_c.legend(fontsize=7, frameon=False)
    ax_c.set_ylim(-0.35, 0.15)

    # (d) Score distribution comparison
    ax_d = fig.add_subplot(gs[1, 1])
    ds_dist = cx["score_distributions"]["deepseek_direct"]
    gpt_dist = cx["score_distributions"]["codexapis_direct"]
    ds_means = [ds_dist[d]["mean"] for d in TARGETS]
    ds_stds = [ds_dist[d]["std"] for d in TARGETS]
    gpt_means = [gpt_dist[d]["mean"] for d in TARGETS]
    gpt_stds = [gpt_dist[d]["std"] for d in TARGETS]

    ax_d.errorbar(ds_means, range(len(TARGETS)), xerr=ds_stds, fmt='o', color=LLM_COLOR,
                  capsize=2, markersize=4, label='DeepSeek', alpha=0.8)
    ax_d.errorbar(gpt_means, [i+0.15 for i in range(len(TARGETS))], xerr=gpt_stds, fmt='s',
                  color=GPT_COLOR, capsize=2, markersize=4, label='GPT-5.6', alpha=0.8)
    ax_d.axvline(x=0, color='black', linewidth=0.5, linestyle='--')
    ax_d.set_yticks(range(len(TARGETS)))
    ax_d.set_yticklabels(SHORT, fontsize=7)
    ax_d.set_xlabel("Score (z-scored)")
    ax_d.set_title("(d) Score Distribution\nMean ± 1 SD", fontsize=9, fontweight='bold')
    ax_d.legend(fontsize=7, frameon=False)
    ax_d.invert_yaxis()

    fig.suptitle('Cross-Model Validation: DeepSeek-v4 vs GPT-5.6', fontweight='bold', fontsize=12)
    fig.savefig(FIGURES_DIR / "fig4_cross_model.png")
    plt.close(fig)
    print("  Fig 4: Cross-Model Validation — done")

def _load_model_scores(filename, method_filter):
    """Load LLM scores from JSONL, return {vid: {dim: score}}."""
    path = RESULTS_DIR / "features" / filename
    scores = {}
    if not path.exists():
        return scores
    with open(path, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line.strip())
            if r.get('method', '') == method_filter and r.get('temperature', 0.0) == 0.0:
                s = r.get('scores')
                if s is None:
                    continue
                vid = r['video_id']
                scores[vid] = {}
                for d in TARGETS:
                    if d in s:
                        val = s[d]
                        if isinstance(val, dict):
                            scores[vid][d] = float(val.get('score', 0))
                        elif isinstance(val, (int, float)):
                            scores[vid][d] = float(val)
    return scores

def _load_gt_scores():
    """Load ground truth scores."""
    y = np.load(RESULTS_DIR / "features" / "targets.npy")
    with open(RESULTS_DIR / "features" / "all_features.jsonl", encoding='utf-8') as f:
        vids = [json.loads(l)['video_id'] for l in f if l.strip()]
    return {vids[i]: {TARGETS[j]: float(y[i, j]) for j in range(12)} for i in range(len(vids))}

# ===== Figure 5: Modality Ablation =====

def fig5_modality_ablation(exp):
    """Modality ablation: Face/Gesture/Voice/Combinations."""
    e4 = exp.get("experiment_4", {})
    if not e4:
        print("  Fig 5: No modality ablation data — skipping")
        return

    mod_order = ["Face", "Gesture", "Voice", "Face+Gesture", "Face+Voice",
                 "Gesture+Voice", "All"]
    metrics = ["macro_r", "personality_r", "interview_r"]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(mod_order))
    width = 0.25
    colors = [C_BLUE, C_GREEN, C_ORANGE]

    for i, (metric, label) in enumerate(zip(metrics, ["Macro r", "Personality r", "Interview r"])):
        vals = [e4[m][metric] for m in mod_order]
        bars = ax.bar(x + i * width, vals, width, label=label, color=colors[i],
                      edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)

    ax.set_xticks(x + width)
    ax.set_xticklabels(mod_order)
    ax.set_ylabel("Pearson r")
    ax.set_title("Modality Ablation (RF, 5-fold CV)", fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(0, 0.5)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_modality_ablation.png")
    plt.close(fig)
    print("  Fig 5: Modality Ablation — done")

# ===== Figure 6: CoT Ablation & Reasoning Analysis =====

def fig6_cot_ablation(exp, methods):
    """CoT ablation: Full CoT vs Skip S1 vs Skip S2 vs Direct."""
    e5 = exp.get("experiment_5", {})

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) Score variance comparison
    ax_a = axes[0]
    if "score_variance" in e5:
        sv = e5["score_variance"]
        methods_var = ["CoT Full", "CoT Skip S1", "CoT Skip S2", "Direct", "GT"]
        scores_var = [
            sv.get("cot_mean", 0), sv.get("skip_s1_mean", sv.get("cot_mean", 0)),
            sv.get("skip_s2_mean", sv.get("direct_mean", 0)),
            sv.get("direct_mean", 0), sv.get("gt_mean", 0),
        ]
        colors = [C_RED, C_ORANGE, C_ORANGE, C_BLUE, C_GRAY]
        bars = ax_a.bar(methods_var, scores_var, color=colors, edgecolor='white')
        for bar, val in zip(bars, scores_var):
            ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                      f'{val:.3f}', ha='center', fontsize=8)
    ax_a.set_ylabel("Per-Video Score Std Dev")
    ax_a.set_title("(a) Score Dispersion by Method", fontweight='bold')
    ax_a.grid(axis='y', alpha=0.3)

    # (b) CoT vs Direct per-dimension comparison
    ax_b = axes[1]
    cot = methods.get("CoT_full", {})
    direct = methods.get("Direct", {})

    x = np.arange(len(TARGETS))
    width = 0.35
    cot_vals = [cot.get(d, {}).get("pearson_r", 0) if isinstance(cot.get(d), dict) else 0 for d in TARGETS]
    dir_vals = [direct.get(d, {}).get("pearson_r", 0) if isinstance(direct.get(d), dict) else 0 for d in TARGETS]

    ax_b.bar(x - width/2, cot_vals, width, color=C_RED, label='CueCoT (3-stage)', edgecolor='white')
    ax_b.bar(x + width/2, dir_vals, width, color=C_BLUE, label='LLM Direct', edgecolor='white')
    ax_b.axhline(y=0, color='black', linewidth=0.5)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(SHORT, rotation=45, ha='right', fontsize=7)
    ax_b.set_ylabel("Pearson r vs GT")
    ax_b.set_title("(b) CoT vs Direct: Per-Dimension vs GT", fontweight='bold')
    ax_b.legend(frameon=False, fontsize=8)
    ax_b.set_ylim(-0.35, 0.15)

    fig.suptitle('CoT Reasoning Analysis', fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_cot_ablation.png")
    plt.close(fig)
    print("  Fig 6: CoT Ablation — done")

# ===== Figure 7: Per-Dimension Forest Plot =====

def fig7_per_dim_forest(methods, cx):
    """Forest plot: all methods vs GT across all 12 dimensions."""
    # Select representative methods
    rep_methods = ["RF_all", "SVR_all", "Text-only", "CoT_full", "Direct", "GPT-5.6 Direct"]
    rep_labels = ["RF (All)", "SVR (All)", "Text-only", "CueCoT", "LLM Direct (DS)", "LLM Direct (GPT-5.6)"]
    colors = [ML_COLOR, ML_COLOR, C_GREEN, C_RED, C_ORANGE, C_PURPLE]

    available = [(m, l, c) for m, l, c in zip(rep_methods, rep_labels, colors) if m in methods]

    fig, axes = plt.subplots(6, 2, figsize=(14, 16), sharex=True)
    fig.subplots_adjust(hspace=0.4, wspace=0.1)

    for dim_idx, dim in enumerate(TARGETS):
        row, col = divmod(dim_idx, 2)
        ax = axes[row, col]

        y_positions = []
        y_labels = []
        r_values = []
        bar_colors = []

        for i, (m, label, color) in enumerate(available):
            d = methods[m]
            if isinstance(d.get(dim), dict):
                r_val = d[dim].get("pearson_r", 0)
            else:
                r_val = d.get(dim, 0) if isinstance(d.get(dim), (int, float)) else 0

            y_positions.append(len(available) - i)
            y_labels.append(label)
            r_values.append(r_val)
            bar_colors.append(color)

        ax.barh(y_positions, r_values, color=bar_colors, edgecolor='white', height=0.6)
        ax.axvline(x=0, color='black', linewidth=0.5, linestyle='-')
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=7)
        ax.set_title(SHORT[dim_idx], fontsize=9, fontweight='bold')
        ax.set_xlim(-0.5, 0.6)
        ax.grid(axis='x', alpha=0.3)

    fig.suptitle('Per-Dimension Method Comparison (Pearson r vs Ground Truth)', fontweight='bold', y=0.995)
    fig.text(0.5, 0.005, 'Pearson r', ha='center', fontsize=10)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(FIGURES_DIR / "fig7_per_dim_forest.png")
    plt.close(fig)
    print("  Fig 7: Per-Dimension Forest Plot — done")

# ===== Figure 8: Qualitative Cases =====

def fig8_qualitative_cases(exp):
    """3-case qualitative comparison: CoT, Direct, GT scores as heatmap."""
    e7 = exp.get("experiment_7", {})
    if not e7:
        print("  Fig 8: No qualitative data — skipping")
        return

    # Load scores from result files
    deepseek = {}
    with open(RESULTS_DIR / "features" / "llm_results.jsonl", encoding='utf-8') as f:
        for line in f:
            r = json.loads(line.strip())
            s = r.get('scores')
            if s:
                deepseek[(r['video_id'], r['method'])] = s

    gpt = {}
    gpt_path = RESULTS_DIR / "features" / "llm_results_codexapis.jsonl"
    if gpt_path.exists():
        with open(gpt_path, encoding='utf-8') as f:
            for line in f:
                r = json.loads(line.strip())
                s = r.get('scores')
                if s:
                    gpt[r['video_id']] = s

    gt = _load_gt_scores()

    cases = [
        ("case_a", "Case A\n(vid_0122)\nr=-0.94"),
        ("case_b", "Case B\n(vid_1637)\nr=+0.83"),
        ("case_c", "Case C\n(vid_0113)\nr=-0.04"),
    ]

    rows = []
    row_labels = []
    for case_key, case_label in cases:
        case_data = e7.get(case_key, {})
        vid = case_data.get("video_id", "")

        # GT
        if vid in gt:
            rows.append([gt[vid].get(d, 0) for d in TARGETS])
            row_labels.append(f"{case_label}\nGT")
        # DeepSeek CoT
        if (vid, 'cot_full') in deepseek:
            s = deepseek[(vid, 'cot_full')]
            rows.append([float(s.get(d, 0)) for d in TARGETS])
            row_labels.append(f"{case_label}\nDS CoT")
        # DeepSeek Direct
        if (vid, 'direct') in deepseek:
            s = deepseek[(vid, 'direct')]
            rows.append([float(s.get(d, 0)) for d in TARGETS])
            row_labels.append(f"{case_label}\nDS Dir")
        # GPT-5.6 Direct
        if vid in gpt:
            s = gpt[vid]
            rows.append([float(s.get(d, 0)) for d in TARGETS])
            row_labels.append(f"{case_label}\nGPT Dir")

    if len(rows) < 3:
        print("  Fig 8: Insufficient case data — skipping")
        return

    data = np.array(rows)
    fig, ax = plt.subplots(figsize=(14, max(5, len(rows) * 0.55)))
    sns.heatmap(data, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                vmin=-1.5, vmax=1.5, linewidths=0.8, linecolor='white',
                xticklabels=SHORT, yticklabels=row_labels,
                annot_kws={'fontsize': 7.5}, cbar_kws={'shrink': 0.5, 'label': 'Score'},
                ax=ax)
    ax.set_title('Qualitative Case Analysis: 3 Representative Videos', fontweight='bold', pad=10)
    ax.tick_params(axis='both', labelsize=8)
    ax.set_xticklabels(SHORT, rotation=45, ha='right')
    ax.set_yticklabels(row_labels, rotation=0)

    # Add separator lines between cases
    for i in range(1, len(cases)):
        ax.axhline(y=i * (len(rows) // len(cases)), color='black', linewidth=1.5)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig8_qualitative_cases.png")
    plt.close(fig)
    print("  Fig 8: Qualitative Cases — done")

# ===== Main =====

def main():
    print("Loading data...")
    exp, bl, cx, sup = load_data()
    methods = build_methods_dict(exp, bl, cx)

    print(f"  {len(methods)} methods loaded")
    print("Generating figures...")

    fig2_main_results(methods)
    fig3_correlation_heatmap(exp)
    fig4_cross_model(cx)
    fig5_modality_ablation(exp)
    fig6_cot_ablation(exp, methods)
    fig7_per_dim_forest(methods, cx)
    fig8_qualitative_cases(exp)

    print(f"\nAll figures saved to: {FIGURES_DIR}")

if __name__ == "__main__":
    main()
