"""Prepare a summary for GPT-5.6 to review the paper."""
import json

with open('D:/jingxin/experiments/results/experiment_results.json') as f:
    data = json.load(f)

e1 = data['experiment_1']
methods_summary = {}
for m in ['RF_all', 'RF_voice', 'RF_face', 'CoT_full', 'Direct', 'Text-only', 'LateFusion', 'AttentionLSTM', 'EarlyFusion_MLP']:
    if m in e1:
        methods_summary[m] = {
            'macro_r': round(e1[m]['macro_r'], 4),
            'personality_r': round(e1[m]['personality_r'], 4),
            'interview_r': round(e1[m]['interview_r'], 4),
        }

e4 = data['experiment_4']
modality = {k: round(v['macro_r'], 4) for k, v in e4.items()}

e5 = data['experiment_5']
e6 = data['experiment_6']
e8 = data['experiment_8']

summary = {
    'methods': methods_summary,
    'modality_ablation': modality,
    'cot_vs_direct': {
        'score_std': e5['score_variance'],
        'js_to_gt': {'cot': round(e5['mean_js_cot_vs_gt'], 4), 'direct': round(e5['mean_js_direct_vs_gt'], 4)},
        'p_value': e6['CoT_full_vs_Direct']['p_value'],
        'cohens_d': round(e6['CoT_full_vs_Direct']['cohens_d'], 2),
    },
    'cross_model': {
        'deepseek_vs_gpt56_direct_r': round(e8['cross_model_macro_r'], 4),
        'deepseek_vs_gt_r': round(e8['deepseek_vs_gt_macro_r'], 4),
        'gpt56_vs_gt_r': round(e8['codexapis_vs_gt_macro_r'], 4),
    },
    'bootstrap': {k: {'p': v['p_value'], 'd': round(v['cohens_d'], 2)} for k, v in e6.items()},
}

print(json.dumps(summary, indent=2))
