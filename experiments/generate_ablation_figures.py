"""Supplementary ablation figures — deeper analysis beyond the 8 main figures.

Generates 6 additional figures:
  A1: Per-dimension modality contribution heatmap (7 modalities × 12 dims)
  A2: JS divergence CoT vs Direct per dimension (why CoT is worse)
  A3: Score distribution density (violin plots CoT/Direct/GT)
  A4: Modality marginal gain (waterfall: what each modality adds)
  A5: CoT vs Direct inter-dim correlation matrices (3-panel)
  A6: Modality complementarity matrix (F×G×V synergy)
"""
import json, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

# Colorblind-friendly
C_BLUE = '#0072B2'
C_ORANGE = '#E69F00'
C_GREEN = '#009E73'
C_RED = '#D55E00'
C_PURPLE = '#CC79A7'
C_SKY = '#56B4E9'
C_GRAY = '#999999'

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 10, "axes.titlesize": 11, "legend.fontsize": 8,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.spines.top": False, "axes.spines.right": False,
})

def load_data():
    with open(RESULTS_DIR / "experiment_results.json") as f:
        exp = json.load(f)
    with open(RESULTS_DIR / "baseline_summary.json") as f:
        bl = json.load(f)
    return exp, bl

# ===== A1: Per-Dimension Modality Contribution =====

def fig_a1_modality_heatmap(exp):
    """Heatmap: 7 modality combinations × 12 dimensions."""
    e4 = exp["experiment_4"]
    mods = ["Face", "Gesture", "Voice", "Face+Gesture", "Face+Voice", "Gesture+Voice", "All"]
    mod_labels = ["Face", "Gesture", "Voice", "F+G", "F+V", "G+V", "All"]

    data = np.zeros((len(mods), len(TARGETS)))
    for i, mod in enumerate(mods):
        for j, dim in enumerate(TARGETS):
            data[i, j] = e4[mod]["per_dimension"].get(dim, 0)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    sns.heatmap(data, annot=True, fmt='.3f', cmap='YlOrRd',
                vmin=0.15, vmax=0.45, linewidths=0.5, linecolor='white',
                xticklabels=SHORT, yticklabels=mod_labels,
                annot_kws={'fontsize': 7.5}, cbar_kws={'shrink': 0.8, 'label': 'Pearson r'},
                ax=ax)
    ax.set_title('Per-Dimension Modality Contribution (RF, 5-fold CV)', fontweight='bold', pad=10)
    ax.set_ylabel('Modality Combination')
    ax.tick_params(axis='both', labelsize=8)

    # Highlight best modality per dimension
    for j in range(len(TARGETS)):
        best_i = np.argmax(data[:, j])
        ax.add_patch(plt.Rectangle((j, best_i), 1, 1, fill=False, edgecolor=C_RED, lw=2))

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_a1_modality_heatmap.png")
    plt.close(fig)
    print("  A1: Modality Contribution Heatmap — done")

# ===== A2: JS Divergence CoT vs Direct =====

def fig_a2_js_divergence(exp):
    """Bar chart: JS divergence to GT, CoT vs Direct per dimension."""
    e5 = exp["experiment_5"]
    kl = e5["kl_divergence"]

    cot_js = [kl[d]["cot_vs_gt_js"] for d in TARGETS]
    dir_js = [kl[d]["direct_vs_gt_js"] for d in TARGETS]
    delta_js = [c - d for c, d in zip(cot_js, dir_js)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    x = np.arange(len(TARGETS))
    width = 0.35

    # Left: absolute JS divergence
    ax1.bar(x - width/2, cot_js, width, color=C_RED, label='CoT vs GT', edgecolor='white')
    ax1.bar(x + width/2, dir_js, width, color=C_BLUE, label='Direct vs GT', edgecolor='white')
    ax1.set_xticks(x)
    ax1.set_xticklabels(SHORT, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Jensen-Shannon Divergence')
    ax1.set_title('(a) Distribution Distance to GT', fontweight='bold')
    ax1.legend(frameon=False, fontsize=8)
    ax1.axhline(y=np.mean(cot_js), color=C_RED, linestyle='--', alpha=0.5, linewidth=0.8)
    ax1.axhline(y=np.mean(dir_js), color=C_BLUE, linestyle='--', alpha=0.5, linewidth=0.8)
    ax1.grid(axis='y', alpha=0.3)

    # Right: delta (CoT - Direct), positive = CoT worse
    colors = [C_RED if d > 0 else C_GREEN for d in delta_js]
    ax2.bar(x, delta_js, color=colors, edgecolor='white')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(SHORT, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Δ JS Divergence (CoT - Direct)')
    ax2.set_title(f'(b) CoT Δ vs Direct (mean Δ={np.mean(delta_js):.4f}) — ALL positive', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle('CoT Makes Scores LESS Aligned with GT — Not More', fontweight='bold', fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_a2_js_divergence.png")
    plt.close(fig)
    print("  A2: JS Divergence Analysis — done")

# ===== A3: Score Distribution Density =====

def fig_a3_score_density(exp):
    """Violin/box plot: score distributions CoT vs Direct vs GT."""
    e5 = exp["experiment_5"]

    # Load actual scores
    all_cot = []
    all_dir = []
    all_gt = []

    with open(RESULTS_DIR / "features" / "llm_results.jsonl", encoding='utf-8') as f:
        for line in f:
            r = json.loads(line.strip())
            s = r.get('scores')
            if s is None:
                continue
            if r['method'] == 'cot_full' and r.get('temperature', 0.0) == 0.0:
                all_cot.extend([float(s[d]) for d in TARGETS if d in s])
            elif r['method'] == 'direct' and r.get('temperature', 0.0) == 0.0:
                all_dir.extend([float(s[d]) for d in TARGETS if d in s])

    y = np.load(RESULTS_DIR / "features" / "targets.npy")
    all_gt = y.flatten().tolist()
    all_gt = [float(v) for v in all_gt if abs(v) < 5.0]  # clip outliers

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    # Select 6 representative dimensions
    rep_dims = ["extraversion", "neuroticism", "speaking_skills", "confidence_score",
                "conscientiousness", "answer_score"]
    rep_short = ["Extraversion", "Neuroticism", "Speaking Skills", "Confidence",
                 "Conscientiousness", "Answer Score"]

    for idx, (dim, label) in enumerate(zip(rep_dims, rep_short)):
        ax = axes[idx // 3, idx % 3]

        # Extract per-dim scores
        cot_vals = []
        dir_vals = []
        gt_vals = []

        with open(RESULTS_DIR / "features" / "llm_results.jsonl", encoding='utf-8') as f:
            for line in f:
                r = json.loads(line.strip())
                s = r.get('scores')
                if s is None or dim not in s:
                    continue
                if r['method'] == 'cot_full' and r.get('temperature', 0.0) == 0.0:
                    cot_vals.append(float(s[dim]))
                elif r['method'] == 'direct' and r.get('temperature', 0.0) == 0.0:
                    dir_vals.append(float(s[dim]))

        target_idx = TARGETS.index(dim)
        gt_vals = [float(y[i, target_idx]) for i in range(y.shape[0]) if abs(y[i, target_idx]) < 5.0]

        parts = ax.violinplot([gt_vals, dir_vals, cot_vals], positions=[0, 1, 2],
                              showmeans=True, showmedians=True)
        for pc, color in zip(parts['bodies'], [C_GRAY, C_BLUE, C_RED]):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['GT', 'Direct', 'CoT'], fontsize=8)
        ax.set_title(label, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Score Distribution Density: GT vs Direct vs CoT (6 representative dimensions)',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_a3_score_density.png")
    plt.close(fig)
    print("  A3: Score Distribution Density — done")

# ===== A4: Modality Marginal Gain =====

def fig_a4_modality_marginal(exp):
    """Waterfall: marginal contribution of each modality to macro_r."""
    e4 = exp["experiment_4"]

    # Marginal gains
    baseline = 0.0
    face_gain = e4["Face"].get("macro_r", 0) if "macro_r" in e4["Face"] else \
                np.mean(list(e4["Face"]["per_dimension"].values()))
    gesture_gain = e4["Gesture"].get("macro_r", 0) if "macro_r" in e4["Gesture"] else \
                   np.mean(list(e4["Gesture"]["per_dimension"].values()))
    voice_gain = e4["Voice"].get("macro_r", 0) if "macro_r" in e4["Voice"] else \
                 np.mean(list(e4["Voice"]["per_dimension"].values()))
    fg_gain = e4["Face+Gesture"].get("macro_r", 0) if "macro_r" in e4["Face+Gesture"] else \
              np.mean(list(e4["Face+Gesture"]["per_dimension"].values()))
    fv_gain = e4["Face+Voice"].get("macro_r", 0) if "macro_r" in e4["Face+Voice"] else \
              np.mean(list(e4["Face+Voice"]["per_dimension"].values()))
    gv_gain = e4["Gesture+Voice"].get("macro_r", 0) if "macro_r" in e4["Gesture+Voice"] else \
              np.mean(list(e4["Gesture+Voice"]["per_dimension"].values()))
    all_gain = e4["All"].get("macro_r", 0) if "macro_r" in e4["All"] else \
               np.mean(list(e4["All"]["per_dimension"].values()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: absolute values ranked
    mods = ['Face', 'Gesture', 'Voice', 'F+G', 'F+V', 'G+V', 'All']
    vals = [face_gain, gesture_gain, voice_gain, fg_gain, fv_gain, gv_gain, all_gain]
    colors = [C_SKY, C_ORANGE, C_GREEN, C_PURPLE, C_BLUE, C_RED, '#333333']

    sorted_idx = np.argsort(vals)
    ax1.barh([mods[i] for i in sorted_idx], [vals[i] for i in sorted_idx],
             color=[colors[i] for i in sorted_idx], edgecolor='white')
    ax1.set_xlabel('Mean Pearson r across 12 dimensions')
    ax1.set_title('(a) Modality Ranking (RF)', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # Right: synergy analysis
    ax2.barh(['Voice alone', '+ Face', '+ Gesture', '= All three'],
             [voice_gain, fv_gain - voice_gain, all_gain - fv_gain, all_gain],
             color=[C_GREEN, C_SKY, C_ORANGE, '#333333'], edgecolor='white',
             left=[0, voice_gain, fv_gain, 0])
    ax2.set_xlabel('Cumulative Pearson r')
    ax2.set_title('(b) Modality Synergy: What Each Adds', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    # Annotations
    ax2.text(fv_gain - 0.01, 1, f'+{fv_gain - voice_gain:.4f}', va='center', ha='right', fontsize=8, color=C_SKY)
    ax2.text(all_gain - 0.01, 0, f'+{all_gain - fv_gain:.4f}', va='center', ha='right', fontsize=8, color=C_ORANGE)

    fig.suptitle('Modality Contribution & Synergy Analysis', fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_a4_modality_marginal.png")
    plt.close(fig)
    print("  A4: Modality Marginal Gain — done")

# ===== A5: CoT vs Direct Inter-Dim Correlation =====

def fig_a5_correlation_matrices(exp):
    """3-panel: CoT / Direct / GT inter-dimension correlation matrices."""
    # Load scores
    cot_scores = {d: [] for d in TARGETS}
    dir_scores = {d: [] for d in TARGETS}
    gt_scores = {d: [] for d in TARGETS}

    with open(RESULTS_DIR / "features" / "llm_results.jsonl", encoding='utf-8') as f:
        for line in f:
            r = json.loads(line.strip())
            s = r.get('scores')
            if s is None:
                continue
            if r['method'] == 'cot_full' and r.get('temperature', 0.0) == 0.0:
                for d in TARGETS:
                    if d in s:
                        cot_scores[d].append(float(s[d]))
            elif r['method'] == 'direct' and r.get('temperature', 0.0) == 0.0:
                for d in TARGETS:
                    if d in s:
                        dir_scores[d].append(float(s[d]))

    y = np.load(RESULTS_DIR / "features" / "targets.npy")
    for j, d in enumerate(TARGETS):
        gt_scores[d] = [float(y[i, j]) for i in range(y.shape[0]) if abs(y[i, j]) < 5.0]

    def build_corr_matrix(scores_dict):
        n = len(TARGETS)
        mat = np.zeros((n, n))
        for i, d1 in enumerate(TARGETS):
            for j, d2 in enumerate(TARGETS):
                v1 = scores_dict[d1]
                v2 = scores_dict[d2]
                min_len = min(len(v1), len(v2))
                if min_len > 10:
                    mat[i, j] = np.corrcoef(v1[:min_len], v2[:min_len])[0, 1]
        return mat

    cot_mat = build_corr_matrix(cot_scores)
    dir_mat = build_corr_matrix(dir_scores)
    gt_mat = build_corr_matrix(gt_scores)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    matrices = [cot_mat, dir_mat, gt_mat]
    titles = ['CueCoT (3-stage)', 'LLM Direct', 'Ground Truth']
    means = [np.mean(np.abs(cot_mat)), np.mean(np.abs(dir_mat)), np.mean(np.abs(gt_mat))]

    for ax, mat, title, mean_r in zip(axes, matrices, titles, means):
        sns.heatmap(mat, annot=False, cmap='RdBu_r', vmin=-0.2, vmax=1.0, center=0,
                    square=True, linewidths=0.3, linecolor='white',
                    xticklabels=SHORT, yticklabels=SHORT,
                    cbar_kws={'shrink': 0.7}, ax=ax)
        ax.set_title(f'{title}\n(mean |r| = {mean_r:.3f})', fontweight='bold', fontsize=10)
        ax.tick_params(axis='both', labelsize=6, rotation=45)
        ax.set_xticklabels(SHORT, rotation=45, ha='right')
        ax.set_yticklabels(SHORT, rotation=0)

    fig.suptitle('Inter-Dimension Correlation Matrices', fontweight='bold', fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_a5_correlation_matrices.png")
    plt.close(fig)
    print("  A5: Correlation Matrices — done")

# ===== A6: Modality Complementarity =====

def fig_a6_modality_complementarity(exp):
    """Show which dimensions benefit from which modality pairings."""
    e4 = exp["experiment_4"]

    # Compute complementarity: (F+V) - max(F, V) for each dimension
    # Positive = synergy, Negative = redundancy
    face = {d: e4["Face"]["per_dimension"][d] for d in TARGETS}
    gesture = {d: e4["Gesture"]["per_dimension"][d] for d in TARGETS}
    voice = {d: e4["Voice"]["per_dimension"][d] for d in TARGETS}
    all3 = {d: e4["All"]["per_dimension"][d] for d in TARGETS}
    fv = {d: e4["Face+Voice"]["per_dimension"][d] for d in TARGETS}

    fig, ax = plt.subplots(figsize=(14, 5.5))

    x = np.arange(len(TARGETS))
    width = 0.25

    # Plot each modality's per-dim r
    ax.bar(x - width, [voice[d] for d in TARGETS], width, color=C_GREEN, label='Voice alone', edgecolor='white')
    ax.bar(x, [face[d] for d in TARGETS], width, color=C_SKY, label='Face alone', edgecolor='white')
    ax.bar(x + width, [gesture[d] for d in TARGETS], width, color=C_ORANGE, label='Gesture alone', edgecolor='white')

    # Mark All combination with stars
    for j, dim in enumerate(TARGETS):
        best_single = max(face[dim], gesture[dim], voice[dim])
        synergy = all3[dim] - best_single
        if synergy > 0.01:
            ax.annotate(f'+{synergy:.3f}', (j, all3[dim] + 0.01), ha='center', fontsize=7,
                        color=C_RED, fontweight='bold')
        ax.scatter(j, all3[dim], marker='*', color=C_RED, s=80, zorder=5)

    ax.scatter([], [], marker='*', color=C_RED, s=80, label='All 3 combined')

    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Pearson r')
    ax.set_title('Modality Complementarity: Which Dimensions Benefit from Multi-Modal Fusion?', fontweight='bold')
    ax.legend(frameon=False, fontsize=8, ncol=4)
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_a6_modality_complementarity.png")
    plt.close(fig)
    print("  A6: Modality Complementarity — done")

# ===== Main =====

def main():
    print("Generating supplementary ablation figures...")
    exp, bl = load_data()

    fig_a1_modality_heatmap(exp)
    fig_a2_js_divergence(exp)
    fig_a3_score_density(exp)
    fig_a4_modality_marginal(exp)
    fig_a5_correlation_matrices(exp)
    fig_a6_modality_complementarity(exp)

    print(f"\nAll supplementary figures saved to: {FIGURES_DIR}")

if __name__ == "__main__":
    main()
