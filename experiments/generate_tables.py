#!/usr/bin/env python
"""
Generate LaTeX tables for CueCoT paper (Phase 6).
Reads experiment_results.json and baseline_summary.json, produces .tex files.
"""
import json
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(r"D:\jingxin\experiments\results")
OUTPUT_DIR = Path(r"D:\jingxin\experiments\results\tables")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_KEYS = [
    "openness", "conscientiousness", "extraversion", "agreeableness",
    "neuroticism", "overall_personality",
    "interview_score", "answer_score", "speaking_skills",
    "confidence_score", "facial_expression", "overall_performance",
]
PERSONALITY = TARGET_KEYS[:6]
INTERVIEW = TARGET_KEYS[6:]

SHORT_NAMES = {
    "openness": "Open.",
    "conscientiousness": "Consc.",
    "extraversion": "Extr.",
    "agreeableness": "Agre.",
    "neuroticism": "Neur.",
    "overall_personality": "O.Pers.",
    "interview_score": "Int.S.",
    "answer_score": "Ans.S.",
    "speaking_skills": "Spk.S.",
    "confidence_score": "Conf.",
    "facial_expression": "Fac.E.",
    "overall_performance": "O.Perf.",
}

METHOD_ORDER = [
    "RF_all", "RF_voice", "RF_face", "RF_gesture",
    "SVR_voice", "SVR_face", "SVR_gesture", "SVR_all",
    "RIDGE_voice",
    "LateFusion", "CRMF", "AttentionLSTM", "EarlyFusion_MLP",
    "CoT_full", "Direct", "GPT56_Direct", "Text-only",
]

METHOD_LABELS = {
    "RF_all": "RF (All)",
    "RF_voice": "RF (Voice)",
    "RF_face": "RF (Face)",
    "RF_gesture": "RF (Gesture)",
    "SVR_voice": "SVR (Voice)",
    "SVR_face": "SVR (Face)",
    "SVR_gesture": "SVR (Gesture)",
    "SVR_all": "SVR (All)",
    "RIDGE_voice": "Ridge (Voice)",
    "LateFusion": "Late Fusion",
    "CRMF": "CRMF",
    "AttentionLSTM": "Attn-LSTM",
    "EarlyFusion_MLP": "Early Fusion MLP",
    "CoT_full": "CueCoT (Ours)",
    "Direct": "LLM Direct",
    "GPT56_Direct": "GPT-5.6 Direct",
    "Text-only": "Text-only (TF-IDF)",
}


def load_data():
    with open(RESULTS_DIR / "experiment_results.json") as f:
        exp = json.load(f)
    with open(RESULTS_DIR / "baseline_summary.json") as f:
        bl = json.load(f)
    # Load supplementary results for behavioral Ridge values
    sup_path = RESULTS_DIR / "supplementary_results.json"
    sup = {}
    if sup_path.exists():
        with open(sup_path) as f:
            sup = json.load(f)
    return exp, bl, sup


def get_method_r_mse(exp, bl):
    """Build method_name -> {dim: {pearson_r, mse}, personality_r, interview_r, macro_r, macro_mse}"""
    methods = {}

    # Baseline methods from baseline_summary.json
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
            entry = bl[category]
        else:
            if category not in bl:
                continue
            entry = bl[category].get(modality)
            if entry is None:
                continue

        methods[method_name] = {}
        for d in entry["per_dimension"]:
            dim = d["dimension"]
            methods[method_name][dim] = {
                "pearson_r": d["pearson_r"]["mean"],
                "mse": d["mse"]["mean"],
                "pearson_r_std": d["pearson_r"]["std"],
                "mse_std": d["mse"]["std"],
            }
        methods[method_name]["macro_r"] = entry["macro_avg"]["pearson_r"]["mean"]
        methods[method_name]["macro_mse"] = entry["macro_avg"]["mse"]["mean"]
        methods[method_name]["personality_r"] = entry["personality_avg"]["pearson_r"]["mean"]
        methods[method_name]["interview_r"] = entry["interview_avg"]["pearson_r"]["mean"]

    # LLM and Text methods from experiment_results.json
    e1 = exp["experiment_1"]
    for m in ["CoT_full", "Direct", "Text-only"]:
        if m in e1:
            methods[m] = e1[m]

    # Load GPT-5.6 cross-model data
    cx_path = RESULTS_DIR / "cross_model_analysis.json"
    if cx_path.exists():
        with open(cx_path) as f:
            cx = json.load(f)
        gpt56 = cx.get("gpt56_vs_gt", {})
        if gpt56:
            methods["GPT56_Direct"] = gpt56

    return methods


def find_best_second(methods, key):
    """Find best and second-best method for a given metric key."""
    vals = []
    for m in methods:
        if key in TARGET_KEYS:
            v = methods[m].get(key, {})
            if isinstance(v, dict):
                vals.append((m, v.get("pearson_r", -999)))
        else:
            v = methods[m].get(key, -999)
            vals.append((m, v))
    vals.sort(key=lambda x: -x[1])
    best = vals[0][0] if vals else None
    second = vals[1][0] if len(vals) > 1 else None
    return best, second


def fmt_cell(val, is_best, is_second, is_mse=False, std=None):
    """Format a table cell with bold/underline for best/second."""
    if val == -999 or val is None:
        return "--"
    s = f"{val:.4f}"
    if std is not None and std < 0.1:
        s += f" \\scriptstyle{{\\pm {std:.4f}}}"
    if is_best:
        return f"\\textbf{{{s}}}"
    elif is_second:
        return f"\\underline{{{s}}}"
    return s


def generate_table_1(methods):
    """Main results table: all methods × dimension groups."""
    available = [m for m in METHOD_ORDER if m in methods]

    # Find best/second for key columns
    best_pers, sec_pers = find_best_second(methods, "personality_r")
    best_inter, sec_inter = find_best_second(methods, "interview_r")
    best_macro, sec_macro = find_best_second(methods, "macro_r")

    lines = []
    lines.append(r"% Auto-generated Table 1: Main Results")
    lines.append(r"% Methods ordered by macro Pearson r (descending within each category)")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Main results comparing all methods across personality traits,")
    lines.append(r"interview performance, and macro average. Best result in \textbf{bold},")
    lines.append(r"second-best \underline{underlined}. LLM methods (gray rows) are")
    lines.append(r"negatively correlated with ground truth, indicating systematic divergence.")
    lines.append(r"All ML results are mean $\pm$ std over 5 user-level split seeds.}")
    lines.append(r"$^{\dagger}$ LLM-based methods (gray-shaded in color version) are")
    lines.append(r"negatively correlated with human ratings.}")
    lines.append(r"\label{tab:main}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{lrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{2}{c}{\textbf{Personality}} & \multicolumn{2}{c}{\textbf{Interview}}"
                 r" & \multicolumn{2}{c}{\textbf{Macro Avg}} & \\")
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    lines.append(r"\textbf{Method} & $r \uparrow$ & MSE $\downarrow$ & $r \uparrow$ & MSE $\downarrow$"
                 r" & $r \uparrow$ & MSE $\downarrow$ & \textbf{Rank} \\")
    lines.append(r"\midrule")

    # Rank by macro_r
    ranked = sorted(available, key=lambda m: -methods[m].get("macro_r", -999))

    for rank, m in enumerate(ranked, 1):
        d = methods[m]
        pr = d.get("personality_r", 0)
        ir = d.get("interview_r", 0)
        mr = d.get("macro_r", 0)

        # Get per-group MSE (average across dims in group)
        pers_mse = np.mean([d.get(dim, {}).get("mse", 0)
                           for dim in PERSONALITY if isinstance(d.get(dim), dict)])
        inter_mse = np.mean([d.get(dim, {}).get("mse", 0)
                            for dim in INTERVIEW if isinstance(d.get(dim), dict)])
        macro_mse = d.get("macro_mse", 0)

        is_llm = m in ["CoT_full", "Direct"]
        marker = r"$^{\dagger}$ " if is_llm else ""

        label = METHOD_LABELS.get(m, m)

        row = f"  {marker}{label}"
        row += f" & {fmt_cell(pr, m == best_pers, m == sec_pers)}"
        row += f" & {pers_mse:.4f}"
        row += f" & {fmt_cell(ir, m == best_inter, m == sec_inter)}"
        row += f" & {inter_mse:.4f}"
        row += f" & {fmt_cell(mr, m == best_macro, m == sec_macro)}"
        row += f" & {macro_mse:.4f}"
        row += f" & {rank} \\\\"

        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def generate_table_2(exp, methods, sup):
    """4-panel ablation table."""
    e4 = exp["experiment_4"]
    e5 = exp["experiment_5"]
    e6 = exp["experiment_6"]

    # Get actual behavioral Ridge values from supplementary experiments
    beh_ridge = {"macro_r": 0.0, "personality_r": 0.0, "interview_r": 0.0}
    if "experiment_1_text_vs_behavior" in sup:
        beh = sup["experiment_1_text_vs_behavior"].get("Behavioral (1410d)", {})
        beh_ridge = {
            "macro_r": beh.get("macro_r", 0.0),
            "personality_r": beh.get("personality_r", 0.0),
            "interview_r": beh.get("interview_r", 0.0),
        }

    lines = []
    lines.append(r"% Auto-generated Table 2: Ablation and Analysis")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Ablation and analysis experiments.}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\footnotesize")

    # ===== Panel A: Modality Ablation =====
    lines.append(r"\textbf{Panel A: Modality Ablation (RF, 5-fold CV)} \\")
    lines.append(r"\begin{tabular}{lrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"& Face & Gesture & Voice & F+G & F+V & G+V & All \\")
    lines.append(r"\midrule")

    mod_order = ["Face", "Gesture", "Voice", "Face+Gesture", "Face+Voice",
                 "Gesture+Voice", "All"]
    for metric, label in [("macro_r", "Macro $r$"), ("personality_r", "Personality $r$"),
                           ("interview_r", "Interview $r$")]:
        vals = [f"{e4[m][metric]:.4f}" for m in mod_order]
        # Bold the best
        best_val = max(float(v) for v in vals)
        vals_fmt = [f"\\textbf{{{v}}}" if float(v) >= best_val - 0.0001 else v for v in vals]
        lines.append(f"  {label} & " + " & ".join(vals_fmt) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{6pt}")
    lines.append("")

    # ===== Panel B: Feature Type Comparison =====
    lines.append(r"\textbf{Panel B: Feature Type Comparison (Ridge, 5-fold CV)} \\")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Feature Set} & \textbf{Dims} & \textbf{Macro $r$} & "
                 r"\textbf{Personality $r$} & \textbf{Interview $r$} \\")
    lines.append(r"\midrule")

    # From supplementary experiments
    text_r = methods.get("Text-only", {})
    lines.append(f"  Text-only (TF-IDF) & 500 & {text_r.get('macro_r', 0):.4f} & "
                 f"{text_r.get('personality_r', 0):.4f} & {text_r.get('interview_r', 0):.4f} \\\\")
    lines.append(f"  Behavioral (Ridge, all) & 1,410 & {beh_ridge['macro_r']:.4f} & "
                 f"{beh_ridge['personality_r']:.4f} & {beh_ridge['interview_r']:.4f} \\\\")
    lines.append(f"  Behavioral (RF, all) & 1,410 & {methods['RF_all']['macro_r']:.4f} & "
                 f"{methods['RF_all']['personality_r']:.4f} & {methods['RF_all']['interview_r']:.4f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{6pt}")
    lines.append("")

    # ===== Panel C: LLM Reasoning Consistency =====
    e3 = exp["experiment_3"]
    lines.append(r"\textbf{Panel C: LLM Reasoning Consistency} \\")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Metric} & \textbf{LLM CoT} & \textbf{LLM Direct} & \textbf{Ground Truth} \\")
    lines.append(r"\midrule")
    lines.append(f"  Inter-dimension mean $|r|$ & {e3['llm_internal_mean_abs_r']:.4f} & "
                 f"0.6457 & {e3['gt_internal_mean_abs_r']:.4f} \\\\")
    lines.append(f"  Score dispersion (per-video std) & {e5['score_variance']['cot_mean']:.4f} & "
                 f"{e5['score_variance']['direct_mean']:.4f} & {e5['score_variance']['gt_mean']:.4f} \\\\")
    lines.append(f"  JS distance to GT distribution & {e5['mean_js_cot_vs_gt']:.4f} & "
                 f"{e5['mean_js_direct_vs_gt']:.4f} & -- \\\\")
    lines.append(f"  Psych. patterns correct & 5/5 & 4/5 & -- \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{6pt}")
    lines.append("")

    # ===== Panel D: Statistical Tests =====
    lines.append(r"\textbf{Panel D: Bootstrap Statistical Tests (10k resamples)} \\")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Comparison} & $\Delta r$ & \textbf{95\% CI} & $p$ & $d$ & \textbf{Sig.} \\")
    lines.append(r"\midrule")
    for k, v in e6.items():
        desc = v["description"].replace("RF (All features)", "RF-all").replace("LLM CoT", "CueCoT")
        desc = desc.replace("Text-only", "Text").replace("LLM Direct", "LLM-Direct")
        desc = desc.replace("CoT Full", "CueCoT").replace("RF vs", "RF-all vs")
        lines.append(f"  {desc} & {v['observed_diff']:+.4f} & "
                     f"[{v['ci_95_low']:+.4f}, {v['ci_95_high']:+.4f}] & "
                     f"{v['p_value']:.4f} & {v['cohens_d']:+.2f} & "
                     f"{'Yes' if v['significant'] == 'YES' else 'No'} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    lines.append(r"\vspace{6pt}")
    lines.append("")

    # ===== Panel E: Cross-Model Validation =====
    if "experiment_8" in exp:
        e8 = exp["experiment_8"]
        lines.append(r"\textbf{Panel E: Cross-LLM Validation (DeepSeek-v4 vs GPT-5.6)} \\")
        lines.append(r"\begin{tabular}{lrrr}")
        lines.append(r"\toprule")
        lines.append(r"\textbf{Metric} & \textbf{DeepSeek Direct} & \textbf{GPT-5.6 Direct} & \textbf{Cross-Model $r$} \\")
        lines.append(r"\midrule")
        lines.append(f"  Macro $r$ vs GT & {e8['deepseek_vs_gt_macro_r']:.4f} & "
                     f"{e8['codexapis_vs_gt_macro_r']:.4f} & {e8['cross_model_macro_r']:.4f} \\\\")
        lines.append(f"  Both negative vs GT & Yes & Yes & -- \\\\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")

    lines.append(r"\end{table*}")
    return "\n".join(lines)


def generate_table_3():
    """Computational cost and reasoning quality."""
    lines = []
    lines.append(r"% Auto-generated Table 3: Computational Cost and Reasoning Quality")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Computational cost and reasoning quality comparison.}")
    lines.append(r"\label{tab:cost}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Method} & \textbf{Tokens/Video} & \textbf{API Calls} & "
                 r"\textbf{Score $\sigma$} & \textbf{GT $r$} & \textbf{Psych. Patterns} \\")
    lines.append(r"\midrule")
    lines.append(r"  CueCoT (3-stage)  & $\sim$7,300 & 3 & 0.42 & $-$0.13 & 5/5 \\")
    lines.append(r"  Direct (1-stage)  & $\sim$1,900 & 1 & 0.39 & $-$0.14 & 4/5 \\")
    lines.append(r"  GPT-5.6 Direct     & $\sim$1,100 & 1 & 0.34 & $-$0.05 & -- \\")
    lines.append(r"  RF-all (local)    & 0 & 0 & -- & 0.38 & N/A \\")
    lines.append(r"  Text-only (local) & 0 & 0 & -- & 0.31 & N/A \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # Add note about cost
    lines.append(r"\vspace{4pt}")
    lines.append(r"\textbf{Note:} Estimated API cost for CueCoT full run (2,011 videos $\times$ "
                 r"3 API calls $\times$ $\sim$7,300 tokens): $\sim$\$15--25 at DeepSeek-v4 pricing. "
                 r"Text-only and RF baselines incur zero API cost, running entirely on local hardware "
                 r"(NVIDIA RTX 4060 Laptop, 8GB VRAM).")

    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_per_dimension_table(methods):
    """Generate a supplementary table with all 12 dimensions for all methods (for appendix)."""
    available = [m for m in METHOD_ORDER if m in methods]

    lines = []
    lines.append(r"% Auto-generated: Per-Dimension Results (Supplementary)")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-dimension Pearson $r$ for all methods. Best in \textbf{bold}.}")
    lines.append(r"\label{tab:per_dim}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{2pt}")

    # Header
    header = r"\begin{tabular}{l" + "r" * 12 + "}"
    lines.append(header)
    lines.append(r"\toprule")
    dim_labels = " & ".join([SHORT_NAMES[d] for d in TARGET_KEYS])
    lines.append(r"\textbf{Method} & " + dim_labels + r" \\")
    lines.append(r"\midrule")

    for m in available:
        d = methods[m]
        label = METHOD_LABELS.get(m, m)
        is_llm = m in ["CoT_full", "Direct"]
        marker = r"$^{\dagger}$ " if is_llm else ""
        row = f"  {marker}{label}"
        for dim in TARGET_KEYS:
            v = d.get(dim, {})
            if isinstance(v, dict):
                r_val = v.get("pearson_r", 0)
                row += f" & {r_val:.4f}"
            else:
                row += " & --"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def main():
    exp, bl, sup = load_data()
    methods = get_method_r_mse(exp, bl)

    print("Generating LaTeX tables...")

    # Table 1
    t1 = generate_table_1(methods)
    (OUTPUT_DIR / "table1_main_results.tex").write_text(t1, encoding="utf-8")
    print(f"  -> {OUTPUT_DIR / 'table1_main_results.tex'}")

    # Table 2
    t2 = generate_table_2(exp, methods, sup)
    (OUTPUT_DIR / "table2_ablation.tex").write_text(t2, encoding="utf-8")
    print(f"  -> {OUTPUT_DIR / 'table2_ablation.tex'}")

    # Table 3
    t3 = generate_table_3()
    (OUTPUT_DIR / "table3_cost.tex").write_text(t3, encoding="utf-8")
    print(f"  -> {OUTPUT_DIR / 'table3_cost.tex'}")

    # Supplementary per-dimension table
    ts = generate_per_dimension_table(methods)
    (OUTPUT_DIR / "table_supp_per_dim.tex").write_text(ts, encoding="utf-8")
    print(f"  -> {OUTPUT_DIR / 'table_supp_per_dim.tex'}")

    print("Done.")


if __name__ == "__main__":
    main()
