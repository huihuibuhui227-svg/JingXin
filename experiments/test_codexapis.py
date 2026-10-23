"""Test codexapis.com GPT-5.6 on CueCoT 3-stage CoT, and compare vs existing DeepSeek results."""
import os
import json, sys, time, re, numpy as np
sys.path.insert(0, 'D:/jingxin')
from openai import OpenAI
from experiments.cot_prompts import CoTPromptBuilder, build_stage2_user_message, build_stage3_user_message
from scipy import stats

# ===== Config =====
CODECXAPIS_KEY = os.getenv("CODEXAPIS_KEY", "")
CODECXAPIS_URL = 'https://www.codexapis.com/v1'
MODEL = 'gpt-5.6-luna'
N_VIDEOS = 5

FEATURES_DIR = 'D:/jingxin/experiments/results/features'
LLM_RESULTS_PATH = f'{FEATURES_DIR}/llm_results.jsonl'

TARGETS = ['openness', 'conscientiousness', 'extraversion', 'agreeableness',
           'neuroticism', 'overall_personality', 'interview_score', 'answer_score',
           'speaking_skills', 'confidence_score', 'facial_expression', 'overall_performance']

# ===== Load data =====
client = OpenAI(api_key=CODECXAPIS_KEY, base_url=CODECXAPIS_URL)
builder = CoTPromptBuilder()

with open(f'{FEATURES_DIR}/all_features.jsonl') as f:
    features = [json.loads(l) for l in f if l.strip()]

y = np.load(f'{FEATURES_DIR}/targets.npy')
with open(f'{FEATURES_DIR}/all_features.jsonl') as f:
    vids_order = [json.loads(l)['video_id'] for l in f if l.strip()]
vid_to_idx = {v: i for i, v in enumerate(vids_order)}

# Load existing DeepSeek CoT results for comparison
deepseek_cot = {}
deepseek_direct = {}
with open(LLM_RESULTS_PATH, encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        vid, method, temp = r['video_id'], r['method'], r.get('temperature', 0.0)
        if temp != 0.0:
            continue
        scores = r.get('scores')
        if scores is None:
            continue
        if method == 'cot_full':
            deepseek_cot[vid] = {d: scores.get(d) for d in TARGETS}
        elif method == 'direct':
            deepseek_direct[vid] = {d: scores.get(d) for d in TARGETS}

def call_llm(messages, label="", max_tokens=2000):
    """Single API call with timing."""
    msgs = [{'role': m['role'], 'content': m['content']} for m in messages]
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=msgs, temperature=0.0, max_tokens=max_tokens)
        raw = resp.choices[0].message.content or ''
        elapsed = time.time() - t0
        tokens = resp.usage.total_tokens
        print(f"    {label}: {elapsed:.1f}s, {tokens} tokens", flush=True)
        return raw, elapsed, tokens
    except Exception as e:
        print(f"    {label} ERROR: {e}", flush=True)
        return None, 0, 0

def parse_scores(text):
    """Extract 12 scores from JSON/raw text."""
    scores = {}
    # Try JSON block first
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if not json_match:
        json_match = re.search(r'(\{[^{}]*"openness"[^{}]*\})', text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            for d in TARGETS:
                if d in parsed and parsed[d] is not None:
                    scores[d] = float(parsed[d])
            if len(scores) >= 8:
                return scores
        except json.JSONDecodeError:
            pass
    # Fallback: regex
    for dim in TARGETS:
        m = re.search(rf'"{dim}"\s*:\s*(-?\d+\.?\d*)', text)
        if m:
            scores[dim] = float(m.group(1))
    return scores

def run_cot_full(feature_dict, vid):
    """Run full 3-stage CoT."""
    stages = builder.build_full_cot(feature_dict)
    total_tokens = 0
    total_time = 0
    chain = {}

    # Stage 1
    raw1, t1, tok1 = call_llm(stages[0], "Stage1")
    if raw1 is None:
        return None, {}, 0, 0
    chain['stage1_output'] = raw1[:2000]
    total_time += t1
    total_tokens += tok1

    # Stage 2
    s2_msgs = stages[1] + build_stage2_user_message(raw1)
    raw2, t2, tok2 = call_llm(s2_msgs, "Stage2")
    if raw2 is None:
        return None, {}, total_time, total_tokens
    chain['stage2_output'] = raw2[:2000]
    total_time += t2
    total_tokens += tok2

    # Stage 3
    s3_msgs = stages[2] + build_stage3_user_message(raw1, raw2)
    raw3, t3, tok3 = call_llm(s3_msgs, "Stage3")
    if raw3 is None:
        return None, {}, total_time, total_tokens
    chain['stage3_output'] = raw3[:2000]
    total_time += t3
    total_tokens += tok3

    scores = parse_scores(raw3)
    return scores, chain, total_time, total_tokens

def run_direct(feature_dict, vid):
    """Run Direct baseline."""
    msgs = builder.build_direct(feature_dict)
    raw, elapsed, tokens = call_llm(msgs, "Direct")
    if raw is None:
        return None, elapsed, tokens
    scores = parse_scores(raw)
    return scores, elapsed, tokens

# ===== Pick 5 diverse test videos =====
# Use the 3 qualitative analysis cases + 2 more
test_vids = ['vid_0122', 'vid_1637', 'vid_0113', 'vid_0001', 'vid_0110']
test_features = []
for vid in test_vids:
    for d in features:
        if d['video_id'] == vid:
            test_features.append(d)
            break
# Fallback: first 5
if len(test_features) < N_VIDEOS:
    test_features = features[:N_VIDEOS]

print(f"Testing codexapis.com ({MODEL}) — CueCoT 3-stage CoT + Direct")
print(f"Videos: {[d['video_id'] for d in test_features]}")
print(f"Existing DeepSeek CoT entries: {len(deepseek_cot)}, Direct: {len(deepseek_direct)}")
print("=" * 70, flush=True)

results = []
for d in test_features:
    vid = d['video_id']
    print(f"\n{'='*50}")
    print(f"  {vid}", flush=True)
    print(f"{'='*50}")

    # --- Full CoT ---
    print("  [CoT Full]", flush=True)
    scores_cot, chain_cot, t_cot, tok_cot = run_cot_full(d, vid)

    # --- Direct ---
    print("  [Direct]", flush=True)
    scores_dir, t_dir, tok_dir = run_direct(d, vid)

    # --- GT ---
    idx = vid_to_idx.get(vid)
    gt = {TARGETS[i]: float(y[idx, i]) for i in range(12)} if idx is not None else {}

    # --- DeepSeek comparison ---
    ds_cot = deepseek_cot.get(vid, {})
    ds_dir = deepseek_direct.get(vid, {})

    # --- Compute correlations ---
    def compute_r(s1, s2):
        common = [d for d in TARGETS if d in s1 and d in s2
                  and s1[d] is not None and s2[d] is not None]
        if len(common) < 5:
            return None
        a1 = [s1[d] for d in common]
        a2 = [s2[d] for d in common]
        r, _ = stats.pearsonr(a1, a2)
        return r

    r_cot_vs_gt = compute_r(scores_cot or {}, gt)
    r_dir_vs_gt = compute_r(scores_dir or {}, gt)
    r_cot_vs_ds = compute_r(scores_cot or {}, ds_cot)
    r_dir_vs_ds = compute_r(scores_dir or {}, ds_dir) if ds_dir else None

    n_cot = len(scores_cot) if scores_cot else 0
    n_dir = len(scores_dir) if scores_dir else 0

    print(f"\n  --- {vid} Summary ---", flush=True)
    print(f"  CoT: {n_cot}/12 scores, r_vs_GT={r_cot_vs_gt:.3f}" if r_cot_vs_gt is not None else f"  CoT: {n_cot}/12 scores, r=N/A")
    print(f"  Direct: {n_dir}/12 scores, r_vs_GT={r_dir_vs_gt:.3f}" if r_dir_vs_gt is not None else f"  Direct: {n_dir}/12 scores, r=N/A")
    print(f"  CoT vs DeepSeek-CoT r={r_cot_vs_ds:.3f}" if r_cot_vs_ds is not None else "  CoT vs DeepSeek-CoT: N/A")
    print(f"  Direct vs DeepSeek-Direct r={r_dir_vs_ds:.3f}" if r_dir_vs_ds is not None else "  Direct vs DeepSeek-Direct: N/A")
    print(f"  CoT time={t_cot:.0f}s, tokens={tok_cot} | Direct time={t_dir:.0f}s, tokens={tok_dir}", flush=True)

    results.append({
        'video_id': vid,
        'cot_scores': scores_cot,
        'direct_scores': scores_dir,
        'gt': gt,
        'deepseek_cot': ds_cot,
        'deepseek_direct': ds_dir,
        'r_cot_vs_gt': float(r_cot_vs_gt) if r_cot_vs_gt is not None else None,
        'r_dir_vs_gt': float(r_dir_vs_gt) if r_dir_vs_gt is not None else None,
        'r_cot_vs_deepseek': float(r_cot_vs_ds) if r_cot_vs_ds is not None else None,
    })

# ===== Final Summary =====
print("\n" + "=" * 70)
print("FINAL COMPARISON: codexapis GPT-5.6 vs DeepSeek-v4 vs GT")
print("=" * 70)
print(f"\n{'Video':<12} {'CoT r(GT)':>10} {'Dir r(GT)':>10} {'CoT r(DS)':>10} {'CoT #scores':>12} {'Dir #scores':>12}")
print("-" * 68)
for r in results:
    print(f"{r['video_id']:<12} {str(r['r_cot_vs_gt']):>10} {str(r['r_dir_vs_gt']):>10} "
          f"{str(r['r_cot_vs_deepseek']):>10} "
          f"{len(r['cot_scores'] or {}):>12} {len(r['direct_scores'] or {}):>12}")

# Aggregate
valid_cot = [r['r_cot_vs_gt'] for r in results if r['r_cot_vs_gt'] is not None]
valid_dir = [r['r_dir_vs_gt'] for r in results if r['r_dir_vs_gt'] is not None]
if valid_cot:
    print(f"\nMean CoT r(GT): {np.mean(valid_cot):.4f}  (DeepSeek CoT macro_r = -0.128)")
if valid_dir:
    print(f"Mean Direct r(GT): {np.mean(valid_dir):.4f}  (DeepSeek Direct macro_r = -0.140)")

# Save detailed results
out_path = 'D:/jingxin/experiments/results/codexapis_test.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
print(f"\nSaved: {out_path}")
