"""Feature Shuffling Control Experiment.

Shuffles behavioral feature descriptions across videos and runs GPT-5.6 Direct
on the mismatched features. If the LLM uses the behavioral signal, shuffled features
should produce worse (more random) predictions. If LLM scores are similar regardless,
the prompt is not effectively using the behavioral information.

Runs on the same 200-video stratified subset as the CoT experiment for comparability.
Uses Direct mode (1 API call per video) for speed.
"""
import os
import json, sys, time, re, os, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
sys.path.insert(0, 'D:/jingxin')
from openai import OpenAI

# ===== Config =====
API_KEY = os.getenv("CODEXAPIS_KEY", "")
API_URL = 'https://www.codexapis.com/v1'
MODEL = 'gpt-5.6-luna'
METHOD = 'direct_shuffled'
NUM_WORKERS = 8
SUBSET_SIZE = 200
SEED = 42

FEATURES_DIR = 'D:/jingxin/experiments/results/features'
OUTPUT_PATH = f'{FEATURES_DIR}/llm_results_codexapis_shuffled.jsonl'
PROGRESS_PATH = 'D:/jingxin/experiments/results/codexapis_shuffled_progress.json'

TARGETS = ['openness', 'conscientiousness', 'extraversion', 'agreeableness',
           'neuroticism', 'overall_personality', 'interview_score', 'answer_score',
           'speaking_skills', 'confidence_score', 'facial_expression', 'overall_performance']

# Direct prompt (same as V2 proven prompt)
DIRECT_SYSTEM = """You are an expert interview assessor. Given behavioral feature measurements from a single interview video, directly assign scores on 12 standardized dimensions.

## ABOUT THE FEATURES
The measurements below are computed from video/audio analysis. Values are percentile-normalized across a population of 2,011 interview candidates. A value near the median indicates a typical candidate on that feature.

## SCORING RUBRIC (Z-SCORED)
  +2.0 = exceptional (top 2-3%)
  +1.0 = clearly above average
   0.0 = typical/average candidate
  -1.0 = clearly below average
  -2.0 = severely deficient (bottom 2-3%)

## CRITICAL RULES
1. Evaluate each dimension INDEPENDENTLY. A candidate can have excellent delivery (high speaking_skills) but weak answers (low answer_score). Do NOT let one domain dominate all scores.
2. Behavioral features (voice, face, gesture) directly inform: speaking_skills, confidence_score, facial_expression, extraversion.
3. Behavioral features provide only INDIRECT evidence for: answer_score, interview_score, conscientiousness. Acknowledge this uncertainty — scores on these dimensions should be closer to 0 unless behavioral signals are extreme.
4. DO NOT default to 0 for every dimension. Differentiate. The expected standard deviation of your 12 scores should be at least 0.3.
5. BEWARE THE HALO EFFECT: frequent smiling and open gestures do NOT guarantee good answers.

## DIMENSIONS
1. openness (intellectual curiosity)  2. conscientiousness (discipline)
3. extraversion (sociability)         4. agreeableness (warmth, cooperation)
5. neuroticism (anxiety — higher=more) 6. overall_personality
7. interview_score (overall)          8. answer_score (answer quality — INDIRECT)
9. speaking_skills (fluency — DIRECT) 10. confidence_score (self-assurance)
11. facial_expression                 12. overall_performance

## OUTPUT FORMAT
Return ONLY this exact JSON structure (no other text):
{
  "openness": <float>,
  "conscientiousness": <float>,
  "extraversion": <float>,
  "agreeableness": <float>,
  "neuroticism": <float>,
  "overall_personality": <float>,
  "interview_score": <float>,
  "answer_score": <float>,
  "speaking_skills": <float>,
  "confidence_score": <float>,
  "facial_expression": <float>,
  "overall_performance": <float>
}"""

# ===== Thread-safe state =====
write_lock = Lock()
progress_lock = Lock()
completed = set()


def save_progress():
    with progress_lock:
        with open(PROGRESS_PATH, 'w') as f:
            json.dump(sorted(completed), f)


def save_result(result):
    with write_lock:
        with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()


def parse_scores(text):
    """Parse flat JSON scores — same as Direct V2 parser."""
    scores = {}
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if not m:
        m = re.search(r'(\{[^{}]*"openness"[^{}]*\})', text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(1))
            for d in TARGETS:
                if d in parsed:
                    val = parsed[d]
                    if isinstance(val, dict):
                        scores[d] = float(val.get('score', val))
                    elif isinstance(val, (int, float)):
                        scores[d] = float(val)
            if len(scores) >= 8:
                return scores
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    for dim in TARGETS:
        m = re.search(rf'"{dim}"\s*:\s*(-?\d+\.?\d*)', text)
        if m:
            scores[dim] = float(m.group(1))
    return scores


def call_api(client, messages, max_tokens=2000):
    msgs = [{'role': m['role'], 'content': m['content']} for m in messages]
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=msgs, temperature=0.0, max_tokens=max_tokens)
            raw = resp.choices[0].message.content or ''
            return raw, resp.usage.total_tokens
        except Exception as e:
            wait = 2 ** attempt
            time.sleep(wait)
    raise RuntimeError("All retries failed")


def process_one(vid, feature_text):
    """Process a single video with shuffled features."""
    t0 = time.time()
    client = OpenAI(api_key=API_KEY, base_url=API_URL)

    try:
        messages = [
            {"role": "system", "content": DIRECT_SYSTEM},
            {"role": "user", "content": feature_text + "\n\n## Task\nAssign scores directly. Consider each dimension independently. Output ONLY the JSON."},
        ]
        raw, total_tok = call_api(client, messages)
        scores = parse_scores(raw)

        result = {
            'video_id': vid, 'method': METHOD, 'model': MODEL,
            'temperature': 0.0, 'scores': scores,
            'tokens_used': total_tok, 'parse_success': len(scores) == 12,
            '_shuffled': True,
        }

        elapsed = time.time() - t0
        with progress_lock:
            completed.add(vid)
        n_parsed = len(scores) if scores else 0
        print(f"[{len(completed)}/{SUBSET_SIZE}] {vid} | {n_parsed}/12 | {elapsed:.0f}s | {total_tok} tok", flush=True)
        return result

    except Exception as e:
        print(f"[ERROR] {vid}: {e}", flush=True)
        return None


def format_features(features):
    """Simple feature-to-text formatter (same as what Direct baseline sees)."""
    sys.path.insert(0, 'D:/jingxin/experiments')
    from cot_prompts import FeatureDescriber
    desc = FeatureDescriber.describe(features)
    lines = [
        "## Behavioral Feature Measurements",
        "Computed from video/audio of one candidate answering interview questions.",
        "Values are percentile-normalized across 2,011 videos. Median ≈ typical candidate.",
        "",
        desc["summary"],
        "",
        "## Detailed Feature Values",
    ]
    for obs in desc["observations"]:
        lines.append(f"- {obs['description']}")
    return "\n".join(lines)


# ===== Init =====
with open(f'{FEATURES_DIR}/all_features.jsonl') as f:
    all_features = [json.loads(l) for l in f if l.strip()]

# Stratified sample (same as CoT subset)
stride = max(1, len(all_features) // SUBSET_SIZE)
subset = all_features[::stride][:SUBSET_SIZE]
print(f"Full dataset: {len(all_features)} videos")
print(f"Shuffled subset: {len(subset)} videos (stride={stride})")

# Create shuffled mapping: each video gets another video's features
rng = random.Random(SEED)
original_vids = [d['video_id'] for d in subset]
shuffled_indices = list(range(len(subset)))
rng.shuffle(shuffled_indices)

shuffled_features = {}
for i, vid in enumerate(original_vids):
    shuffled_features[vid] = subset[shuffled_indices[i]]

# Ensure no self-matches
self_matches = sum(1 for vid in original_vids if shuffled_features[vid]['video_id'] == vid)
print(f"Self-matches (should be near 0): {self_matches}")

# Resume support
if os.path.exists(PROGRESS_PATH):
    with open(PROGRESS_PATH) as f:
        completed = set(json.load(f))
    print(f"Resuming: {len(completed)} already done")

# Also load existing results if any
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                completed.add(r['video_id'])

remaining = [(d['video_id'], d) for d in subset if d['video_id'] not in completed]
print(f"Remaining: {len(remaining)} | Workers: {NUM_WORKERS}")
print(f"Estimated time: {len(remaining) * 4 / NUM_WORKERS / 60:.0f}min")

# ===== Prepare tasks with shuffled features =====
tasks = []
for vid, d in remaining:
    # Get shuffled features (LLM doesn't know features belong to a different video)
    shuffled_d = shuffled_features[vid]
    feature_text = format_features(shuffled_d)
    tasks.append((vid, feature_text))

print(f"Tasks prepared: {len(tasks)}")
print(f"Example: vid={tasks[0][0]} will receive features from another video")

# ===== Parallel execution =====
t0_total = time.time()
n_done = 0

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {executor.submit(process_one, vid, ft): vid for vid, ft in tasks}
    for future in as_completed(futures):
        vid = futures[future]
        try:
            result = future.result()
            if result is not None:
                save_result(result)
            n_done += 1
            if n_done % 20 == 0:
                save_progress()
                elapsed = time.time() - t0_total
                rate = n_done / elapsed * 60
                eta = (len(tasks) - n_done) / rate if rate > 0 else 0
                print(f"  [{n_done}/{len(tasks)}] {rate:.0f}/min | ETA {eta:.0f}min", flush=True)
        except Exception as e:
            print(f"[FATAL] {vid}: {e}", flush=True)
            n_done += 1

# ===== Final =====
save_progress()
elapsed_total = time.time() - t0_total
print(f"\nShuffled Done: {len(completed)}/{SUBSET_SIZE} in {elapsed_total/60:.1f}min")
print(f"Results: {OUTPUT_PATH}")

# ===== Quick comparison =====
print("\n" + "=" * 60)
print("QUICK COMPARISON: Real vs Shuffled Features")
print("=" * 60)

import numpy as np
from scipy.stats import pearsonr

# Load targets
targets = np.load(f'{FEATURES_DIR}/targets.npy')
vid_to_idx = {d['video_id']: i for i, d in enumerate(all_features)}

# Load real GPT-5.6 Direct results for same subset
real_scores = {}
with open(f'{FEATURES_DIR}/llm_results_codexapis.jsonl', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get('scores') and r['video_id'] in {d['video_id'] for d in subset}:
            real_scores[r['video_id']] = r['scores']

# Load shuffled results
shuf_scores = {}
with open(OUTPUT_PATH, encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get('scores'):
            shuf_scores[r['video_id']] = r['scores']

# Compute macro r for both
def compute_macro_r(scores_dict, targets, vid_to_idx):
    rs = []
    for vid, s in scores_dict.items():
        if vid in vid_to_idx and s:
            gt = targets[vid_to_idx[vid]]
            llm = np.array([s.get(t, 0.0) for t in TARGETS])
            if not np.any(np.isnan(gt)) and not np.any(np.isnan(llm)):
                r, _ = pearsonr(gt, llm)
                rs.append(r)
    return np.mean(rs) if rs else float('nan')

real_r = compute_macro_r(real_scores, targets, vid_to_idx)
shuf_r = compute_macro_r(shuf_scores, targets, vid_to_idx)

print(f"Real features macro r: {real_r:.4f} (n={len(real_scores)})")
print(f"Shuffled features macro r: {shuf_r:.4f} (n={len(shuf_scores)})")
print(f"Difference (real - shuffled): {real_r - shuf_r:.4f}")
print(f"Conclusion: {'LLM IS using behavioral signal' if abs(real_r - shuf_r) > 0.03 else 'LLM may NOT be using behavioral signal effectively'}")
