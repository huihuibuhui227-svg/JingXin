"""Batch run: codexapis GPT-5.6 with optimized v2 prompts on ALL videos (parallel).

16-thread parallel version — I/O bound API calls, ~10x speedup.
"""
import os
import json, sys, time, re, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
sys.path.insert(0, 'D:/jingxin')
from openai import OpenAI
from experiments.cot_prompts_v2 import (
    CoTPromptBuilderV2, build_stage2_user_message, build_stage3_user_message
)

# ===== Config =====
API_KEY = os.getenv("CODEXAPIS_KEY", "")
API_URL = 'https://www.codexapis.com/v1'
MODEL = 'gpt-5.6-luna'
METHOD = 'direct_v2'
NUM_WORKERS = 16

FEATURES_DIR = 'D:/jingxin/experiments/results/features'
OUTPUT_PATH = f'{FEATURES_DIR}/llm_results_codexapis.jsonl'
PROGRESS_PATH = 'D:/jingxin/experiments/results/codexapis_progress.json'

TARGETS = ['openness', 'conscientiousness', 'extraversion', 'agreeableness',
           'neuroticism', 'overall_personality', 'interview_score', 'answer_score',
           'speaking_skills', 'confidence_score', 'facial_expression', 'overall_performance']

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
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=msgs, temperature=0.0, max_tokens=max_tokens)
            raw = resp.choices[0].message.content or ''
            return raw, resp.usage.total_tokens
        except Exception as e:
            wait = 2 ** attempt
            time.sleep(wait)
    raise RuntimeError("All retries failed")

def process_one(d):
    """Process a single video — runs in worker thread."""
    vid = d['video_id']
    t0 = time.time()

    # Each thread needs its own client (OpenAI client is not thread-safe)
    client = OpenAI(api_key=API_KEY, base_url=API_URL)
    builder = CoTPromptBuilderV2()

    try:
        if METHOD == 'direct_v2':
            msgs = builder.build_direct(d)
            raw, tokens = call_api(client, msgs)
            scores = parse_scores(raw)
            result = {
                'video_id': vid, 'method': METHOD, 'model': MODEL,
                'temperature': 0.0, 'scores': scores, 'raw_output': raw[:500],
                'tokens_used': tokens, 'parse_success': len(scores) == 12,
            }
        elif METHOD == 'cot_full_v2':
            stages = builder.build_full_cot(d)
            chain = {}
            total_tok = 0
            raw1, tok1 = call_api(client, stages[0])
            chain['stage1_output'] = raw1[:2000]
            total_tok += tok1
            s2_msgs = stages[1] + build_stage2_user_message(raw1)
            raw2, tok2 = call_api(client, s2_msgs)
            chain['stage2_output'] = raw2[:2000]
            total_tok += tok2
            s3_msgs = stages[2] + build_stage3_user_message(raw1, raw2)
            raw3, tok3 = call_api(client, s3_msgs)
            chain['stage3_output'] = raw3[:2000]
            total_tok += tok3
            scores = parse_scores(raw3)
            result = {
                'video_id': vid, 'method': METHOD, 'model': MODEL,
                'temperature': 0.0, 'scores': scores, 'stages': chain,
                'tokens_used': total_tok, 'parse_success': len(scores) == 12,
            }

        elapsed = time.time() - t0
        with progress_lock:
            completed.add(vid)
        n_parsed = len(scores) if scores else 0
        print(f"[{len(completed)}/2011] {vid} | {n_parsed}/12 | {elapsed:.0f}s", flush=True)
        return result

    except Exception as e:
        print(f"[ERROR] {vid}: {e}", flush=True)
        return None

# ===== Init =====
with open(f'{FEATURES_DIR}/all_features.jsonl') as f:
    all_features = [json.loads(l) for l in f if l.strip()]
print(f"Loaded {len(all_features)} videos")

if os.path.exists(PROGRESS_PATH):
    with open(PROGRESS_PATH) as f:
        completed = set(json.load(f))
    print(f"Resuming: {len(completed)} already done")

# Filter out completed
remaining = [d for d in all_features if d['video_id'] not in completed]
print(f"Remaining: {len(remaining)} videos")
print(f"Workers: {NUM_WORKERS} | Method: {METHOD}")
print(f"Estimated time: {len(remaining) * 8 / NUM_WORKERS / 60:.0f}min")

# ===== Parallel execution =====
t0_total = time.time()
n_done = 0

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {executor.submit(process_one, d): d['video_id'] for d in remaining}
    for future in as_completed(futures):
        vid = futures[future]
        try:
            result = future.result()
            if result is not None:
                save_result(result)
            n_done += 1
            if n_done % 100 == 0:
                save_progress()
                elapsed = time.time() - t0_total
                rate = n_done / elapsed * 60
                eta = (len(remaining) - n_done) / rate
                print(f"  [{n_done}/{len(remaining)}] {rate:.0f}/min | ETA {eta:.0f}min", flush=True)
        except Exception as e:
            print(f"[FATAL] {vid}: {e}", flush=True)
            n_done += 1

# ===== Final =====
save_progress()
elapsed_total = time.time() - t0_total
print(f"\nDone: {len(completed)}/{len(all_features)} in {elapsed_total/60:.1f}min")
print(f"Results: {OUTPUT_PATH}")
