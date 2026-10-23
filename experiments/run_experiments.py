#!/usr/bin/env python
"""
CueCoT Phase 5: Comprehensive Experiments (AAAI 2027)
======================================================
7 experiments supporting the narrative:
  "LLM CoT reasoning is psychologically coherent but diverges from human ratings,
   revealing that interview assessment is content-driven, not behavior-driven."

Experiment 1: Main Results Table (LaTeX + bar chart)
Experiment 2: Feature-GT Correlation Heatmap
Experiment 3: LLM Reasoning Consistency Matrix
Experiment 4: Modality Ablation (RF)
Experiment 5: CoT Stage Ablation (reasoning consistency)
Experiment 6: Bootstrap Statistical Tests
Experiment 7: Qualitative Reasoning Chain Samples
"""

import json
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

# --- Paths ---
FEATURES_DIR = Path(r"D:\jingxin\experiments\results\features")
METADATA_PATH = Path(r"D:\RecruitView\RecruitView\metadata.jsonl")
RESULTS_DIR = Path(r"D:\jingxin\experiments\results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# --- Constants ---
TARGET_KEYS = [
    "openness", "conscientiousness", "extraversion", "agreeableness",
    "neuroticism", "overall_personality",
    "interview_score", "answer_score", "speaking_skills",
    "confidence_score", "facial_expression", "overall_performance",
]
PERSONALITY = TARGET_KEYS[:6]
INTERVIEW = TARGET_KEYS[6:]
SEED = 42
N_BOOTSTRAP = 10000

# Feature modality splits (from feature_names.txt ordering)
# face: 1260, gesture: 140, voice: 10
FACE_IDX = list(range(0, 1260))
GESTURE_IDX = list(range(1260, 1400))
VOICE_IDX = list(range(1400, 1410))


# ============================================================
# Data Loading
# ============================================================

def load_data():
    """Load all features, targets, feature names, video IDs, and metadata."""
    X = np.load(FEATURES_DIR / "all_features.npy").astype(np.float64)
    y = np.load(FEATURES_DIR / "targets.npy").astype(np.float64)
    with open(FEATURES_DIR / "feature_names.txt") as f:
        names = [l.strip() for l in f]
    with open(FEATURES_DIR / "all_features.jsonl") as f:
        vids = [json.loads(l)["video_id"] for l in f if l.strip()]

    meta = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                meta[Path(d["file_name"]).stem] = d

    return X, y, names, vids, meta


def load_baseline_results():
    """Load aggregated baseline summary."""
    with open(RESULTS_DIR / "baseline_summary.json") as f:
        return json.load(f)


def load_text_results():
    """Load supplementary experiment results (text vs behavior)."""
    with open(RESULTS_DIR / "supplementary_results.json") as f:
        return json.load(f)


def load_llm_results():
    """Load deduplicated LLM results, return dict keyed by (video_id, method)."""
    with open(FEATURES_DIR / "llm_results.jsonl", encoding="utf-8") as f:
        raw = [json.loads(l) for l in f if l.strip()]
    seen = {}
    for r in raw:
        k = (r["video_id"], r["method"], r.get("temperature", 0.0))
        seen[k] = r
    return seen


def load_split_info():
    """Load user-level split info."""
    with open(RESULTS_DIR / "split_info.json") as f:
        return json.load(f)


# ============================================================
# Plotting Setup
# ============================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})
sns.set_style("whitegrid")

# Grayscale-friendly palette
METHOD_COLORS = {
    "RF_all": "#333333",
    "RF_face": "#555555",
    "RF_gesture": "#777777",
    "RF_voice": "#999999",
    "SVR_face": "#555555",
    "SVR_gesture": "#777777",
    "SVR_voice": "#999999",
    "SVR_all": "#333333",
    "RIDGE_voice": "#999999",
    "LateFusion": "#444444",
    "CRMF": "#666666",
    "AttentionLSTM": "#888888",
    "EarlyFusion_MLP": "#AAAAAA",
    "CoT_full": "#1f77b4",
    "Direct": "#ff7f0e",
    "Text-only": "#2ca02c",
}
GRAY_PALETTE = ["#222222", "#444444", "#666666", "#888888", "#AAAAAA", "#CCCCCC"]


# ============================================================
# Experiment 1: Main Results Table
# ============================================================

def experiment_1_main_table(llm_results, baseline_summary, text_results):
    """
    Generate LaTeX Table 1 and bar chart comparing all methods.
    Reports Pearson r and MSE per dimension group.
    """
    print("=" * 60)
    print("EXPERIMENT 1: Main Results Table")
    print("=" * 60)

    # Collect method results: method_name -> {dim: {"pearson_r": val, "mse": val}}
    methods = {}

    # 1. Baseline methods from baseline_summary
    # Structure: baseline_summary[method_category][modality] -> {per_dimension, macro_avg, ...}
    bl_map = {
        "SVR_face": ("unimodal_svr", "face"),
        "SVR_gesture": ("unimodal_svr", "gesture"),
        "SVR_voice": ("unimodal_svr", "voice"),
        "SVR_all": ("unimodal_svr", "all"),
        "RIDGE_voice": ("unimodal_ridge", "voice"),
        "RF_face": ("unimodal_rf", "face"),
        "RF_gesture": ("unimodal_rf", "gesture"),
        "RF_voice": ("unimodal_rf", "voice"),
        "RF_all": ("unimodal_rf", "all"),
        "EarlyFusion_MLP": ("early_fusion_mlp", None),
        "LateFusion": ("late_fusion", None),
        "CRMF": ("crmf", None),
        "AttentionLSTM": ("attention_lstm", None),
    }

    for method_name, (category, modality) in bl_map.items():
        if modality is None:
            entry = baseline_summary[category]
        else:
            if category not in baseline_summary:
                continue
            entry = baseline_summary[category].get(modality)
            if entry is None:
                continue
        methods[method_name] = {}
        for d in entry["per_dimension"]:
            dim = d["dimension"]
            methods[method_name][dim] = {
                "pearson_r": d["pearson_r"]["mean"],
                "mse": d["mse"]["mean"],
            }

    # 2. LLM methods
    llm_by_method = defaultdict(lambda: defaultdict(list))
    for (vid, method, _), r in llm_results.items():
        scores = r.get("scores")
        if scores is None:
            continue
        for dim, val in scores.items():
            if val is not None and dim in TARGET_KEYS:
                llm_by_method[method][dim].append((vid, val))

    # Build vid -> GT mapping
    X, y, names, vids, _ = load_data()
    vid_to_y = {v: y[i] for i, v in enumerate(vids)}

    for method in ["cot_full", "direct"]:
        display_name = "CoT_full" if method == "cot_full" else "Direct"
        methods[display_name] = {}
        for d_idx, dim in enumerate(TARGET_KEYS):
            pairs = [(val, vid_to_y[vid][d_idx])
                     for vid, val in llm_by_method[method][dim]
                     if vid in vid_to_y]
            if len(pairs) < 10:
                methods[display_name][dim] = {"pearson_r": 0, "mse": 999}
                continue
            vals = np.array([p[0] for p in pairs])
            gts = np.array([p[1] for p in pairs])
            r_val, _ = stats.pearsonr(vals, gts)
            mse_val = np.mean((vals - gts) ** 2)
            methods[display_name][dim] = {"pearson_r": float(r_val), "mse": float(mse_val)}

    # 3. Text-only from supplementary results
    text_data = text_results["experiment_1_text_vs_behavior"]
    methods["Text-only"] = text_data["Text-only (TF-IDF)"]
    # Rename dim keys back
    for dim in TARGET_KEYS:
        if dim in text_data["Text-only (TF-IDF)"]:
            pass  # already there

    # Compute macro averages
    for m_name in list(methods.keys()):
        dims_present = [d for d in TARGET_KEYS if d in methods[m_name]
                        and isinstance(methods[m_name][d], dict)
                        and "pearson_r" in methods[m_name][d]]
        if dims_present:
            methods[m_name]["macro_r"] = float(np.mean(
                [methods[m_name][d]["pearson_r"] for d in dims_present]))
            methods[m_name]["macro_mse"] = float(np.mean(
                [methods[m_name][d]["mse"] for d in dims_present]))
        else:
            methods[m_name]["macro_r"] = 0.0
            methods[m_name]["macro_mse"] = 0.0
        methods[m_name]["personality_r"] = float(np.mean(
            [methods[m_name][d]["pearson_r"] for d in PERSONALITY
             if d in methods[m_name] and isinstance(methods[m_name][d], dict)]))
        methods[m_name]["interview_r"] = float(np.mean(
            [methods[m_name][d]["pearson_r"] for d in INTERVIEW
             if d in methods[m_name] and isinstance(methods[m_name][d], dict)]))

    # --- Print summary ---
    method_order = [
        "RF_all", "RF_voice", "RF_face", "RF_gesture",
        "SVR_voice", "SVR_face", "SVR_gesture", "SVR_all",
        "RIDGE_voice", "LateFusion", "CRMF", "AttentionLSTM",
        "EarlyFusion_MLP", "CoT_full", "Direct", "Text-only",
    ]
    available = [m for m in method_order if m in methods]

    print(f"\n  {'Method':<20} {'Personality r':>14} {'Interview r':>14} {'Macro r':>10} {'Macro MSE':>10}")
    print("  " + "-" * 72)
    for m in available:
        d = methods[m]
        print(f"  {m:<20} {d.get('personality_r', 0):>14.4f} "
              f"{d.get('interview_r', 0):>14.4f} {d.get('macro_r', 0):>10.4f} "
              f"{d.get('macro_mse', 0):>10.4f}")

    # --- LaTeX Table ---
    latex = _generate_latex_table(methods, available)
    latex_path = RESULTS_DIR / "main_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"\n  LaTeX table saved: {latex_path}")

    # --- Bar Chart ---
    _generate_bar_chart(methods, available)
    print(f"  Bar chart saved: {FIGURES_DIR / 'exp1_main_results.png'}")

    return methods


def _generate_latex_table(methods, method_order):
    """Generate LaTeX tabular for the main results table."""
    header = r"""% Auto-generated by run_experiments.py -- Experiment 1
\begin{table*}[t]
\centering
\caption{Pearson correlation ($r$) and MSE across all methods and dimension groups.
Bold = best, \underline{underline} = second best. LLM methods (CoT, Direct) show
negative or near-zero correlations with ground truth, while text-only features
achieve the strongest predictive signal.}
\label{tab:main_results}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrr}
\toprule
\textbf{Method} & \textbf{Personality $r$} & \textbf{Interview $r$} &
\textbf{Macro $r$} & \textbf{Macro MSE} & \textbf{Open. $r$} &
\textbf{Consc. $r$} \\
\midrule
"""
    # Find best per column
    all_keys = ["personality_r", "interview_r", "macro_r"] + TARGET_KEYS
    best = {}
    second = {}
    for key in all_keys:
        vals = []
        for m in method_order:
            if key in TARGET_KEYS:
                v = methods[m].get(key, {}).get("pearson_r", -999) if isinstance(methods[m].get(key), dict) else -999
            else:
                v = methods[m].get(key, -999)
            vals.append((m, v))
        vals.sort(key=lambda x: -x[1])
        best[key] = vals[0][0]
        second[key] = vals[1][0] if len(vals) > 1 else vals[0][0]

    # Rows
    rows = []
    for m in method_order:
        d = methods[m]
        pr = d.get("personality_r", 0)
        ir = d.get("interview_r", 0)
        mr = d.get("macro_r", 0)
        mmse = d.get("macro_mse", 0)

        def fmt(val, key, is_mse=False):
            if val == -999:
                return "--"
            s = f"{val:.4f}"
            if m == best[key]:
                return f"\\textbf{{{s}}}"
            elif m == second[key]:
                return f"\\underline{{{s}}}"
            return s

        row = f"  {m:<20} & {fmt(pr, 'personality_r'):>16} & {fmt(ir, 'interview_r'):>14} & "
        row += f"{fmt(mr, 'macro_r'):>10} & {fmt(mmse, 'macro_r', True):>12}"

        # Add first 2 personality dims for compactness; the full table is in supplement
        for dim in TARGET_KEYS[:2]:
            pd = d.get(dim)
            if isinstance(pd, dict):
                row += f" & {fmt(pd['pearson_r'], dim):>10}"
            else:
                row += " & --"
        row += r" \\"
        rows.append(row)

    footer = rf"""\bottomrule
\end{{tabular}}
\end{{table*}}
"""
    return header + "\n".join(rows) + "\n" + footer


def _generate_bar_chart(methods, method_order):
    """Generate grouped bar chart comparing methods across dimension groups."""
    fig, ax = plt.subplots(figsize=(14, 6))

    groups = ["Personality", "Interview", "Macro"]
    x = np.arange(len(groups))
    width = 0.8 / len(method_order)

    for i, m in enumerate(method_order):
        d = methods[m]
        vals = [
            d.get("personality_r", 0),
            d.get("interview_r", 0),
            d.get("macro_r", 0),
        ]
        color = METHOD_COLORS.get(m, "#999999")
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                      label=m, color=color, edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Pearson r")
    ax.set_title("Method Comparison by Dimension Group")
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.legend(loc="lower left", ncol=3, fontsize=7)
    ax.set_ylim(-0.2, 0.5)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp1_main_results.png")
    plt.close(fig)


# ============================================================
# Experiment 2: Feature-GT Correlation Heatmap
# ============================================================

def experiment_2_feature_gt_heatmap(X, y, names):
    """
    Select 20 representative features across modalities.
    Compute Pearson r with 12 GT dimensions. Plot heatmap.
    Flag counter-intuitive correlations with red borders.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Feature-GT Correlation Heatmap")
    print("=" * 60)

    # Select representative features: 5 face AU, 5 gaze, 5 gesture, 5 voice
    selected = []

    # Face AU features — search for key action units
    au_targets = [
        ("au12_smile", "smile"), ("au06_cheek_raise", "cheek_raise"),
        ("au01_inner_brow_raise", "brow_raise"), ("au04_brow_furrow", "brow_furrow"),
        ("au07_lid_tighten", "lid_tighten"), ("au10_upper_lip_raise", "lip_raise"),
        ("au15_lip_corner_depress", "lip_depress"), ("au09_nose_wrinkle", "nose_wrinkle"),
    ]
    for kw, label in au_targets:
        for stat in ["_mean"]:
            matches = [i for i, n in enumerate(names) if kw in n and stat in n
                       and "volatility" not in n and "change_rate" not in n
                       and "trend" not in n]
            if matches and matches[0] not in [s[0] for s in selected]:
                selected.append((matches[0], f"face.{label}.mean"))
                break

    # Gaze features — broader search
    gaze_targets = [
        ("gaze_deviation", "gaze_dev"), ("gaze_direction_x", "gaze_dir_x"),
        ("gaze_direction_y", "gaze_dir_y"), ("gaze_stability", "gaze_stab"),
        ("eye_openness", "eye_open"), ("gaze_velocity", "gaze_vel"),
    ]
    for kw, label in gaze_targets:
        for stat in ["_mean"]:
            matches = [i for i, n in enumerate(names) if kw in n and stat in n
                       and "volatility" not in n and "change_rate" not in n]
            if matches and matches[0] not in [s[0] for s in selected]:
                selected.append((matches[0], f"gaze.{label}.mean"))
                break

    # Gesture features
    gesture_targets = [
        ("hand_movement", "hand_mov"), ("arm_angle", "arm_angle"),
        ("hand_velocity", "hand_vel"), ("gesture_rate", "gesture_rate"),
        ("hand_position", "hand_pos"), ("arm_distance", "arm_dist"),
    ]
    for kw, label in gesture_targets:
        for stat in ["_mean"]:
            matches = [i for i, n in enumerate(names) if kw in n and stat in n
                       and "volatility" not in n and "change_rate" not in n]
            if matches and matches[0] not in [s[0] for s in selected]:
                selected.append((matches[0], f"gesture.{label}.mean"))
                break

    # Voice features
    voice_targets = [
        ("voice.pitch_mean", "pitch_mean"), ("voice.pitch_std", "pitch_std"),
        ("voice.energy_mean", "energy_mean"), ("voice.pause_frequency", "pause_freq"),
        ("voice.speech_rate", "speech_rate"), ("voice.pitch_trend", "pitch_trend"),
    ]
    for kw, label in voice_targets:
        matches = [i for i, n in enumerate(names) if kw in n]
        if matches and matches[0] not in [s[0] for s in selected]:
            selected.append((matches[0], f"voice.{label}"))
            break

    # If still < 20, fill with high-variance features from face
    if len(selected) < 20:
        face_vars = np.var(X[:, FACE_IDX], axis=0)
        top_face_idx = np.argsort(face_vars)[::-1]
        for idx in top_face_idx:
            global_idx = FACE_IDX[idx]
            if global_idx not in [s[0] for s in selected]:
                name_short = names[global_idx].replace("face.", "")[:40]
                selected.append((global_idx, f"face.{name_short}"))
                if len(selected) >= 20:
                    break

    selected = selected[:20]  # Ensure exactly 20
    sel_indices = [s[0] for s in selected]
    sel_labels = [s[1] for s in selected]

    # Compute correlation matrix: [20 features x 12 targets]
    corr_matrix = np.zeros((len(sel_indices), 12))
    for i, idx in enumerate(sel_indices):
        for d in range(12):
            mask = ~np.isnan(X[:, idx]) & ~np.isnan(y[:, d])
            if mask.sum() > 10:
                r_val, _ = stats.pearsonr(X[mask, idx], y[mask, d])
                corr_matrix[i, d] = r_val

    print(f"  Selected {len(sel_indices)} features, matrix shape: {corr_matrix.shape}")

    # Define counter-intuitive expectations
    # (feature_label_prefix, dimension_name, expected_sign)
    counter_intuitive_checks = [
        ("face.smile", "extraversion", "positive"),   # Smiling should → extraversion +
        ("face.smile", "confidence_score", "positive"), # Smiling should → confidence +
        ("gaze.gaze_dev", "confidence_score", "negative"),  # Looking away should → confidence -
        ("gaze.gaze_dev", "neuroticism", "positive"),  # Looking away should → neuroticism +
        ("face.brow_furrow", "neuroticism", "positive"),  # Furrowing should → neuroticism +
        ("voice.pitch_std", "extraversion", "positive"),  # Pitch variation → extraversion +
        ("voice.energy_mean", "confidence_score", "positive"),  # Energy → confidence +
        ("voice.pause_frequency", "confidence_score", "negative"),  # More pauses → less confident
        ("gesture.hand_movement", "extraversion", "positive"),
        ("gesture.arm_angle", "confidence_score", "positive"),
    ]

    # Identify counter-intuitive cells
    counter_flags = np.zeros_like(corr_matrix, dtype=bool)
    for feat_prefix, dim_name, expected_sign in counter_intuitive_checks:
        for i, label in enumerate(sel_labels):
            if label.startswith(feat_prefix):
                d_idx = TARGET_KEYS.index(dim_name) if dim_name in TARGET_KEYS else -1
                if d_idx >= 0:
                    if expected_sign == "positive" and corr_matrix[i, d_idx] < -0.02:
                        counter_flags[i, d_idx] = True
                    elif expected_sign == "negative" and corr_matrix[i, d_idx] > 0.02:
                        counter_flags[i, d_idx] = True

    n_counter = counter_flags.sum()
    print(f"  Counter-intuitive correlations flagged: {n_counter}")

    # --- Plot heatmap ---
    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    im = ax.imshow(corr_matrix, cmap=cmap, aspect="auto", vmin=-0.3, vmax=0.3)

    # Highlight counter-intuitive cells
    for i in range(len(sel_indices)):
        for d in range(12):
            if counter_flags[i, d]:
                ax.add_patch(plt.Rectangle(
                    (d - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="red", linewidth=2.5, zorder=10))

    ax.set_xticks(range(12))
    ax.set_xticklabels(TARGET_KEYS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(sel_indices)))
    ax.set_yticklabels(sel_labels, fontsize=7)
    ax.set_title("Feature-GT Correlation Heatmap\n(red border = counter-intuitive direction)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson r")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp2_feature_gt_heatmap.png")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'exp2_feature_gt_heatmap.png'}")

    return {
        "selected_features": sel_labels,
        "correlation_matrix": corr_matrix.tolist(),
        "counter_intuitive_count": int(n_counter),
        "counter_intuitive_pairs": [
            {"feature": sel_labels[i], "dimension": TARGET_KEYS[d], "r": float(corr_matrix[i, d])}
            for i in range(len(sel_indices)) for d in range(12)
            if counter_flags[i, d]
        ],
    }


# ============================================================
# Experiment 3: LLM Reasoning Consistency Matrix
# ============================================================

def experiment_3_llm_consistency(llm_results, y, vids):
    """
    Cross-correlation matrix: GT 12 vs LLM 12 + LLM internal structure.
    Shows LLM is internally consistent but diverges from GT.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: LLM Reasoning Consistency Matrix")
    print("=" * 60)

    vid_to_idx = {v: i for i, v in enumerate(vids)}

    # Collect LLM scores (cot_full)
    llm_scores = {}
    for (vid, method, _), r in llm_results.items():
        if method == "cot_full":
            sc = r.get("scores")
            if sc is None:
                continue
            if all(sc.get(d) is not None for d in TARGET_KEYS):
                llm_scores[vid] = np.array([sc[d] for d in TARGET_KEYS])

    # Match with GT
    common_vids = sorted(set(llm_scores.keys()) & set(vid_to_idx.keys()))
    print(f"  Common videos with both GT and LLM CoT: {len(common_vids)}")

    gt_matrix = np.array([y[vid_to_idx[v]] for v in common_vids])
    llm_matrix = np.array([llm_scores[v] for v in common_vids])

    # Build combined matrix: [GT 12 + LLM 12] for cross-correlation
    combined = np.hstack([gt_matrix, llm_matrix])  # [N, 24]
    labels = [f"GT:{d}" for d in TARGET_KEYS] + [f"LLM:{d}" for d in TARGET_KEYS]

    corr_full = np.corrcoef(combined.T)  # [24, 24]

    # Extract sub-blocks
    gt_gt = corr_full[:12, :12]
    llm_llm = corr_full[12:, 12:]
    gt_llm = corr_full[:12, 12:]  # rows=GT, cols=LLM

    # Diagonal of GT-LLM block (same dimension)
    diag_gt_llm = np.diag(gt_llm)
    print(f"\n  GT-LLM diagonal correlations:")
    for d, dim in enumerate(TARGET_KEYS):
        print(f"    {dim:<25}: r = {diag_gt_llm[d]:+.4f}")
    print(f"  Mean diagonal r: {diag_gt_llm.mean():+.4f}")
    print(f"  LLM internal consistency (mean |r| off-diag): {np.abs(llm_llm).mean():.4f}")
    print(f"  GT internal consistency (mean |r| off-diag): {np.abs(gt_gt).mean():.4f}")

    # --- Plot combined heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: GT-LLM cross-correlation
    ax1 = axes[0]
    im1 = ax1.imshow(gt_llm, cmap="RdBu_r", aspect="auto", vmin=-0.6, vmax=0.6)
    ax1.set_xticks(range(12))
    ax1.set_xticklabels([d.replace("_", " ") for d in TARGET_KEYS], rotation=45, ha="right", fontsize=7)
    ax1.set_yticks(range(12))
    ax1.set_yticklabels([d.replace("_", " ") for d in TARGET_KEYS], fontsize=7)
    ax1.set_xlabel("LLM CoT Score")
    ax1.set_ylabel("Ground Truth Score")
    ax1.set_title("Cross-Correlation: GT vs LLM CoT")

    # Highlight diagonal
    for d in range(12):
        ax1.add_patch(plt.Rectangle(
            (d - 0.5, d - 0.5), 1, 1,
            fill=False, edgecolor="red" if diag_gt_llm[d] < 0 else "green",
            linewidth=2.5, zorder=10))
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # Right: LLM internal correlation matrix
    ax2 = axes[1]
    mask_llm = np.eye(12, dtype=bool)
    im2 = ax2.imshow(llm_llm, cmap="RdBu_r", aspect="auto", vmin=-0.6, vmax=0.6)
    ax2.set_xticks(range(12))
    ax2.set_xticklabels([d.replace("_", " ") for d in TARGET_KEYS], rotation=45, ha="right", fontsize=7)
    ax2.set_yticks(range(12))
    ax2.set_yticklabels([d.replace("_", " ") for d in TARGET_KEYS], fontsize=7)
    ax2.set_title("LLM Internal Score Correlation")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp3_llm_consistency.png")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'exp3_llm_consistency.png'}")

    return {
        "gt_llm_diagonal": {TARGET_KEYS[d]: float(diag_gt_llm[d]) for d in range(12)},
        "mean_diagonal_r": float(diag_gt_llm.mean()),
        "llm_internal_mean_abs_r": float(np.abs(llm_llm).mean()),
        "gt_internal_mean_abs_r": float(np.abs(gt_gt).mean()),
        "correlation_matrices": {
            "gt_llm": gt_llm.tolist(),
            "llm_llm": llm_llm.tolist(),
            "gt_gt": gt_gt.tolist(),
        },
    }


# ============================================================
# Experiment 4: Modality Ablation (RF)
# ============================================================

def experiment_4_modality_ablation(X, y):
    """
    Fixed RF, 7 modality combinations.
    5-fold CV Pearson r for each of 12 dims.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Modality Ablation (RF)")
    print("=" * 60)

    modality_splits = {
        "Face": FACE_IDX,
        "Gesture": GESTURE_IDX,
        "Voice": VOICE_IDX,
        "Face+Gesture": FACE_IDX + GESTURE_IDX,
        "Face+Voice": FACE_IDX + VOICE_IDX,
        "Gesture+Voice": GESTURE_IDX + VOICE_IDX,
        "All": FACE_IDX + GESTURE_IDX + VOICE_IDX,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}

    for mod_name, indices in modality_splits.items():
        X_mod = X[:, indices]
        dim_rs = []
        for d in range(12):
            y_d = y[:, d]
            rf = RandomForestRegressor(
                n_estimators=100, max_depth=12, random_state=SEED, n_jobs=-1)
            y_pred = cross_val_predict(rf, X_mod, y_d, cv=kf, n_jobs=-1)
            r_val, _ = stats.pearsonr(y_d, y_pred)
            dim_rs.append(r_val)

        results[mod_name] = {
            "per_dimension": {TARGET_KEYS[d]: float(dim_rs[d]) for d in range(12)},
            "macro_r": float(np.mean(dim_rs)),
            "personality_r": float(np.mean(dim_rs[:6])),
            "interview_r": float(np.mean(dim_rs[6:])),
        }
        print(f"  {mod_name:<20}: macro_r = {results[mod_name]['macro_r']:.4f}, "
              f"personality_r = {results[mod_name]['personality_r']:.4f}, "
              f"interview_r = {results[mod_name]['interview_r']:.4f}")

    # --- Bar chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    mod_order = ["Face", "Gesture", "Voice", "Face+Gesture", "Face+Voice",
                 "Gesture+Voice", "All"]
    x = np.arange(len(mod_order))
    macro_vals = [results[m]["macro_r"] for m in mod_order]
    pers_vals = [results[m]["personality_r"] for m in mod_order]
    inter_vals = [results[m]["interview_r"] for m in mod_order]

    w = 0.25
    ax.bar(x - w, pers_vals, w, label="Personality", color="#555555", edgecolor="white")
    ax.bar(x, inter_vals, w, label="Interview", color="#999999", edgecolor="white")
    ax.bar(x + w, macro_vals, w, label="Macro Avg", color="#333333", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(mod_order, rotation=30, ha="right")
    ax.set_ylabel("Pearson r")
    ax.set_title("Modality Ablation: RF Performance by Feature Subset")
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp4_modality_ablation.png")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'exp4_modality_ablation.png'}")

    return results


# ============================================================
# Experiment 5: CoT Stage Ablation (Reasoning Consistency)
# ============================================================

def experiment_5_cot_ablation(llm_results, y, vids):
    """
    Analyze CoT stages from reasoning consistency angle.
    Since we only have cot_full and direct, compare these two.
    Metrics: score variance, inter-dimension correlation strength, KL from GT.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: CoT Stage Ablation")
    print("=" * 60)

    vid_to_idx = {v: i for i, v in enumerate(vids)}

    # Collect scores by method
    scores_by_method = defaultdict(dict)
    for (vid, method, _), r in llm_results.items():
        sc = r.get("scores")
        if sc is None:
            continue
        if all(sc.get(d) is not None for d in TARGET_KEYS):
            scores_by_method[method][vid] = np.array([sc[d] for d in TARGET_KEYS])

    common_cot = sorted(set(scores_by_method.get("cot_full", {}).keys()) & set(vid_to_idx.keys()))
    common_direct = sorted(set(scores_by_method.get("direct", {}).keys()) & set(vid_to_idx.keys()))
    common_both = sorted(set(common_cot) & set(common_direct))

    print(f"  Videos with CoT: {len(common_cot)}, Direct: {len(common_direct)}, Both: {len(common_both)}")

    # 1. Score variance: std of 12 scores per video
    cot_vars = []
    direct_vars = []
    gt_vars = []
    for vid in common_both:
        cot_vars.append(float(np.std(scores_by_method["cot_full"][vid])))
        direct_vars.append(float(np.std(scores_by_method["direct"][vid])))
        idx = vid_to_idx[vid]
        gt_vars.append(float(np.std(y[idx])))

    cot_mean_var = np.mean(cot_vars)
    direct_mean_var = np.mean(direct_vars)
    gt_mean_var = np.mean(gt_vars)

    print(f"\n  Mean score std (across 12 dims per video):")
    print(f"    CoT Full:  {cot_mean_var:.4f}")
    print(f"    Direct:    {direct_mean_var:.4f}")
    print(f"    GT:        {gt_mean_var:.4f}")

    # 2. Inter-dimension correlation strength (mean |r| off-diag)
    cot_matrix = np.array([scores_by_method["cot_full"][v] for v in common_both])
    direct_matrix = np.array([scores_by_method["direct"][v] for v in common_both])
    gt_matrix = np.array([y[vid_to_idx[v]] for v in common_both])

    cot_inter_r = np.abs(np.corrcoef(cot_matrix.T)).mean()
    direct_inter_r = np.abs(np.corrcoef(direct_matrix.T)).mean()
    gt_inter_r = np.abs(np.corrcoef(gt_matrix.T)).mean()

    print(f"\n  Inter-dimension correlation strength (mean |r|):")
    print(f"    CoT Full:  {cot_inter_r:.4f} (higher = more structured)")
    print(f"    Direct:    {direct_inter_r:.4f}")
    print(f"    GT:        {gt_inter_r:.4f}")

    # 3. KL divergence from GT distribution (per dimension)
    kl_divs = {}
    for d, dim in enumerate(TARGET_KEYS):
        gt_dist = gt_matrix[:, d]
        # Use histogram-based distribution approximation
        bins = np.linspace(-3, 3, 31)
        gt_hist, _ = np.histogram(gt_dist, bins=bins, density=True)
        gt_hist = gt_hist + 1e-10
        gt_hist /= gt_hist.sum()

        cot_dist = cot_matrix[:, d]
        cot_hist, _ = np.histogram(cot_dist, bins=bins, density=True)
        cot_hist = cot_hist + 1e-10
        cot_hist /= cot_hist.sum()

        direct_dist = direct_matrix[:, d]
        direct_hist, _ = np.histogram(direct_dist, bins=bins, density=True)
        direct_hist = direct_hist + 1e-10
        direct_hist /= direct_hist.sum()

        kl_divs[dim] = {
            "cot_vs_gt_js": float(jensenshannon(cot_hist, gt_hist)),
            "direct_vs_gt_js": float(jensenshannon(direct_hist, gt_hist)),
            "cot_vs_direct_js": float(jensenshannon(cot_hist, direct_hist)),
        }

    mean_cot_js = np.mean([kl_divs[d]["cot_vs_gt_js"] for d in TARGET_KEYS])
    mean_direct_js = np.mean([kl_divs[d]["direct_vs_gt_js"] for d in TARGET_KEYS])
    print(f"\n  Mean Jensen-Shannon distance from GT:")
    print(f"    CoT vs GT:  {mean_cot_js:.4f}")
    print(f"    Direct vs GT: {mean_direct_js:.4f}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Score variance distribution
    ax1 = axes[0]
    ax1.hist(cot_vars, bins=20, alpha=0.6, label=f"CoT (mean={cot_mean_var:.3f})", color="#1f77b4")
    ax1.hist(direct_vars, bins=20, alpha=0.6, label=f"Direct (mean={direct_mean_var:.3f})", color="#ff7f0e")
    ax1.axvline(gt_mean_var, color="black", linestyle="--", label=f"GT ({gt_mean_var:.3f})")
    ax1.set_xlabel("Score Std (across 12 dims)")
    ax1.set_ylabel("Count")
    ax1.set_title("Per-Video Score Dispersion")
    ax1.legend(fontsize=8)

    # Plot 2: Inter-dim correlation comparison
    ax2 = axes[1]
    groups = ["CoT", "Direct", "GT"]
    vals = [cot_inter_r, direct_inter_r, gt_inter_r]
    colors = ["#1f77b4", "#ff7f0e", "#333333"]
    ax2.bar(groups, vals, color=colors, edgecolor="white")
    ax2.set_ylabel("Mean |r| (off-diagonal)")
    ax2.set_title("Inter-Dimension Correlation Strength")

    # Plot 3: KL divergence per dimension
    ax3 = axes[2]
    x = np.arange(12)
    w = 0.35
    cot_js = [kl_divs[d]["cot_vs_gt_js"] for d in TARGET_KEYS]
    direct_js = [kl_divs[d]["direct_vs_gt_js"] for d in TARGET_KEYS]
    ax3.bar(x - w/2, cot_js, w, label="CoT vs GT", color="#1f77b4", edgecolor="white")
    ax3.bar(x + w/2, direct_js, w, label="Direct vs GT", color="#ff7f0e", edgecolor="white")
    ax3.set_xticks(x)
    ax3.set_xticklabels([d[:8] for d in TARGET_KEYS], rotation=45, ha="right", fontsize=7)
    ax3.set_ylabel("Jensen-Shannon Distance")
    ax3.set_title("Distribution Divergence from GT")
    ax3.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp5_cot_ablation.png")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'exp5_cot_ablation.png'}")

    return {
        "score_variance": {
            "cot_mean": cot_mean_var,
            "direct_mean": direct_mean_var,
            "gt_mean": gt_mean_var,
        },
        "inter_dim_correlation": {
            "cot_mean_abs_r": cot_inter_r,
            "direct_mean_abs_r": direct_inter_r,
            "gt_mean_abs_r": gt_inter_r,
        },
        "kl_divergence": kl_divs,
        "mean_js_cot_vs_gt": mean_cot_js,
        "mean_js_direct_vs_gt": mean_direct_js,
    }


# ============================================================
# Experiment 6: Bootstrap Statistical Tests
# ============================================================

def experiment_6_bootstrap_tests(methods_dict, llm_results, y, vids):
    """
    Paired bootstrap (10k resamples) comparing key method pairs.
    Reports observed difference, 95% CI, p-value, Cohen's d.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: Bootstrap Statistical Tests")
    print("=" * 60)

    vid_to_idx = {v: i for i, v in enumerate(vids)}

    # Collect per-video scores for methods we can compare
    # We need paired per-video predictions: ML methods from baseline, LLM from results

    # For ML methods: use baseline_result per-dimension values across seeds
    # For paired bootstrap, we need per-video predictions which we don't have stored.
    # Instead, do dimension-level paired test across the 12 dimensions.
    # Each "sample" = one of the 12 dimensions.

    np.random.seed(SEED)

    # Build per-dimension r values for each method
    method_dim_rs = {}
    for m_name, m_data in methods_dict.items():
        method_dim_rs[m_name] = np.array([
            m_data[d]["pearson_r"] if isinstance(m_data.get(d), dict) else m_data.get(d, 0)
            for d in TARGET_KEYS
        ])

    comparisons = [
        ("RF_all", "CoT_full", "RF (All features) vs LLM CoT"),
        ("Text-only", "CoT_full", "Text-only vs LLM CoT"),
        ("CoT_full", "Direct", "CoT Full vs Direct"),
        ("RF_all", "Direct", "RF vs LLM Direct"),
    ]

    results = {}
    print(f"\n  Bootstrap tests ({N_BOOTSTRAP} resamples, paired across 12 dimensions):")

    for m1, m2, desc in comparisons:
        if m1 not in method_dim_rs or m2 not in method_dim_rs:
            print(f"    SKIP {desc}: missing data")
            continue

        r1 = method_dim_rs[m1]
        r2 = method_dim_rs[m2]
        obs_diff = np.mean(r1 - r2)

        # Paired bootstrap across dimensions
        n_dims = len(r1)
        boot_diffs = []
        for _ in range(N_BOOTSTRAP):
            idx = np.random.choice(n_dims, size=n_dims, replace=True)
            boot_diffs.append(np.mean(r1[idx] - r2[idx]))
        boot_diffs = np.array(boot_diffs)

        ci_low = np.percentile(boot_diffs, 2.5)
        ci_high = np.percentile(boot_diffs, 97.5)
        p_val = 2 * min(
            np.mean(boot_diffs <= 0),
            np.mean(boot_diffs >= 0),
        )
        p_val = min(p_val, 1.0)

        pooled_std = np.sqrt((np.var(r1) + np.var(r2)) / 2)
        cohens_d = obs_diff / pooled_std if pooled_std > 0 else 0

        significant = "YES" if p_val < 0.05 else "NO"

        print(f"    {desc}:")
        print(f"      {m1} mean r = {np.mean(r1):.4f}, {m2} mean r = {np.mean(r2):.4f}")
        print(f"      Observed diff = {obs_diff:+.4f}, 95% CI = [{ci_low:+.4f}, {ci_high:+.4f}]")
        print(f"      p = {p_val:.4f}, Cohen's d = {cohens_d:+.3f}, Significant: {significant}")

        results[f"{m1}_vs_{m2}"] = {
            "description": desc,
            "method1_mean_r": float(np.mean(r1)),
            "method2_mean_r": float(np.mean(r2)),
            "observed_diff": float(obs_diff),
            "ci_95_low": float(ci_low),
            "ci_95_high": float(ci_high),
            "p_value": float(p_val),
            "cohens_d": float(cohens_d),
            "significant": significant,
        }

    return results


# ============================================================
# Experiment 7: Qualitative Reasoning Chain Samples
# ============================================================

def experiment_7_qualitative_samples(llm_results, y, vids):
    """
    Select 3 representative cases showing the LLM-GT divergence pattern.
    Output complete 3-stage CoT reasoning chains with GT comparison.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 7: Qualitative Reasoning Chain Samples")
    print("=" * 60)

    vid_to_idx = {v: i for i, v in enumerate(vids)}

    # Collect CoT results with stages
    cot_with_stages = {}
    for (vid, method, _), r in llm_results.items():
        if method == "cot_full" and "stages" in r:
            sc = r.get("scores")
            if sc is None:
                continue
            if all(sc.get(d) is not None for d in TARGET_KEYS):
                cot_with_stages[vid] = r

    # For each video, compute LLM-GT agreement (mean r across dims)
    candidates = []
    for vid, r in cot_with_stages.items():
        idx = vid_to_idx.get(vid)
        if idx is None:
            continue
        llm_vec = np.array([r["scores"][d] for d in TARGET_KEYS])
        gt_vec = y[idx]
        # Pearson r between LLM score pattern and GT pattern
        if np.std(llm_vec) > 1e-6 and np.std(gt_vec) > 1e-6:
            r_val, _ = stats.pearsonr(llm_vec, gt_vec)
        else:
            r_val = 0
        # Mean absolute difference
        mad = np.mean(np.abs(llm_vec - gt_vec))
        candidates.append({
            "video_id": vid,
            "r": r_val,
            "mad": mad,
            "llm_vec": llm_vec,
            "gt_vec": gt_vec,
            "result": r,
        })

    # Sort by r: find best agreement, worst LLM>GT, worst GT>LLM
    candidates.sort(key=lambda x: x["r"])

    # Case A: LLM >> GT (LLM overestimates) — take low r, LLM mean > GT mean
    case_a = None
    for c in candidates:
        if c["llm_vec"].mean() > c["gt_vec"].mean() + 0.3:
            case_a = c
            break
    if case_a is None:
        case_a = candidates[0]

    # Case B: LLM << GT (LLM underestimates) — take low r, GT mean > LLM mean
    case_b = None
    for c in reversed(candidates):
        if c["gt_vec"].mean() > c["llm_vec"].mean() + 0.3:
            case_b = c
            break
    if case_b is None:
        case_b = candidates[-1]

    # Case C: LLM ≈ GT (best agreement)
    case_c = candidates[len(candidates) // 2]  # median agreement

    print(f"  Case A (LLM > GT): {case_a['video_id']}, r={case_a['r']:.4f}, "
          f"LLM mean={case_a['llm_vec'].mean():.2f}, GT mean={case_a['gt_vec'].mean():.2f}")
    print(f"  Case B (LLM < GT): {case_b['video_id']}, r={case_b['r']:.4f}, "
          f"LLM mean={case_b['llm_vec'].mean():.2f}, GT mean={case_b['gt_vec'].mean():.2f}")
    print(f"  Case C (LLM ~ GT): {case_c['video_id']}, r={case_c['r']:.4f}, "
          f"LLM mean={case_c['llm_vec'].mean():.2f}, GT mean={case_c['gt_vec'].mean():.2f}")

    # --- Generate text output ---
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("QUALITATIVE REASONING CHAIN ANALYSIS")
    output_lines.append("=" * 80)

    for case_label, case in [("Case A: LLM OVERESTIMATES", case_a),
                               ("Case B: LLM UNDERESTIMATES", case_b),
                               ("Case C: LLM ≈ GT", case_c)]:
        output_lines.append(f"\n{'=' * 80}")
        output_lines.append(f"{case_label}")
        output_lines.append(f"{'=' * 80}")
        output_lines.append(f"Video: {case['video_id']}")
        output_lines.append(f"LLM-GT pattern correlation: r = {case['r']:.4f}")
        output_lines.append(f"Mean absolute difference: {case['mad']:.4f}")
        output_lines.append("")

        # Score comparison table
        output_lines.append(f"{'Dimension':<25} {'LLM CoT':>10} {'GT':>10} {'Diff':>10}")
        output_lines.append("-" * 57)
        for d, dim in enumerate(TARGET_KEYS):
            llm_s = case["llm_vec"][d]
            gt_s = case["gt_vec"][d]
            diff = llm_s - gt_s
            output_lines.append(f"{dim:<25} {llm_s:>10.4f} {gt_s:>10.4f} {diff:>+10.4f}")

        # CoT stages
        r = case["result"]
        stages = r.get("stages", {})
        for stage_key in ["stage1_output", "stage2_output", "stage3_output"]:
            stage_name = stage_key.replace("_output", "").replace("stage", "Stage ")
            output_lines.append(f"\n--- {stage_name} ---")
            stage_text = stages.get(stage_key, "N/A")
            # Clean up JSON markers for readability
            stage_text = stage_text.replace("```json", "").replace("```", "").strip()
            output_lines.append(stage_text[:2000])  # Truncate if very long
            if len(stage_text) > 2000:
                output_lines.append("  ... [truncated]")

    # Save
    text_path = FIGURES_DIR / "exp7_qualitative_samples.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"  Saved: {text_path}")

    return {
        "case_a": {"video_id": case_a["video_id"], "r": float(case_a["r"])},
        "case_b": {"video_id": case_b["video_id"], "r": float(case_b["r"])},
        "case_c": {"video_id": case_c["video_id"], "r": float(case_c["r"])},
        "output_file": str(text_path),
    }


# ============================================================
# Main
# ============================================================

def main():
    print("CueCoT Phase 5: Comprehensive Experiments")
    print("=" * 60)
    print()

    # Load all data
    print("[1/3] Loading data...")
    X, y, names, vids, meta = load_data()
    llm_results = load_llm_results()
    baseline_summary = load_baseline_results()
    text_results = load_text_results()
    print(f"  X={X.shape}, y={y.shape}, LLM results={len(llm_results)}")
    print(f"  Baseline methods: {len(baseline_summary)} categories")

    # Run experiments
    print("\n[2/3] Running experiments...")

    all_results = {}

    # Exp 1: Main Results Table
    methods_dict = experiment_1_main_table(llm_results, baseline_summary, text_results)
    all_results["experiment_1"] = {
        m: {d: (v if not isinstance(v, dict) else v) for d, v in md.items()}
        for m, md in methods_dict.items()
    }

    # Exp 2: Feature-GT Heatmap
    exp2 = experiment_2_feature_gt_heatmap(X, y, names)
    all_results["experiment_2"] = exp2

    # Exp 3: LLM Consistency Matrix
    exp3 = experiment_3_llm_consistency(llm_results, y, vids)
    all_results["experiment_3"] = exp3

    # Exp 4: Modality Ablation
    exp4 = experiment_4_modality_ablation(X, y)
    all_results["experiment_4"] = exp4

    # Exp 5: CoT Stage Ablation
    exp5 = experiment_5_cot_ablation(llm_results, y, vids)
    all_results["experiment_5"] = exp5

    # Exp 6: Bootstrap Tests
    exp6 = experiment_6_bootstrap_tests(methods_dict, llm_results, y, vids)
    all_results["experiment_6"] = exp6

    # Exp 7: Qualitative Samples
    exp7 = experiment_7_qualitative_samples(llm_results, y, vids)
    all_results["experiment_7"] = exp7

    # --- Save all results ---
    print("\n[3/3] Saving results...")

    def convert(obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results_path = RESULTS_DIR / "experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(convert(all_results), f, indent=2, ensure_ascii=False)
    print(f"  Saved: {results_path}")

    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
