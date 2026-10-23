"""Quick test: gpt-5.6 via hanhegufei proxy on 5 videos."""
import os
import json, sys, time, re
sys.path.insert(0, 'D:/jingxin')
from openai import OpenAI
from experiments.cot_prompts import CoTPromptBuilder

client = OpenAI(
    api_key=os.getenv("HANHEGUFEl_KEY", os.getenv("CODEXAPIS_KEY", "")),
    base_url='https://api.hanhegufei.online/v1',
)

with open('D:/jingxin/experiments/results/features/all_features.jsonl') as f:
    features = [json.loads(l) for l in f if l.strip()]

builder = CoTPromptBuilder()
TARGETS = ['openness','conscientiousness','extraversion','agreeableness',
    'neuroticism','overall_personality','interview_score','answer_score',
    'speaking_skills','confidence_score','facial_expression','overall_performance']

y = __import__('numpy').load('D:/jingxin/experiments/results/features/targets.npy')
with open('D:/jingxin/experiments/results/features/all_features.jsonl') as f:
    vids_order = [json.loads(l)['video_id'] for l in f if l.strip()]
vid_to_idx = {v: i for i, v in enumerate(vids_order)}

print("Testing gpt-5.6 on 5 videos...", flush=True)
results = []
for d in features[:5]:
    vid = d['video_id']
    print(f"  Processing {vid}...", flush=True)

    msgs = builder.build_direct(d)
    msgs_formatted = [
        {'role': 'system' if m['role'] == 'system' else 'user', 'content': m['content']}
        for m in msgs
    ]

    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model='gpt-5.6',
            messages=msgs_formatted,
            temperature=0.0,
            max_tokens=1000,
        )
        raw = resp.choices[0].message.content or ''
        elapsed = time.time() - t0

        scores = {}
        for dim in TARGETS:
            m = re.search(rf'"{dim}"\s*:\s*(-?\d+\.?\d*)', raw)
            if m:
                scores[dim] = float(m.group(1))

        idx = vid_to_idx.get(vid)
        gt = {TARGETS[i]: float(y[idx, i]) for i in range(12)} if idx is not None else {}

        print(f"    {len(scores)}/12 parsed in {elapsed:.0f}s", flush=True)
        results.append({'video_id': vid, 'scores': scores, 'gt': gt})
    except Exception as e:
        print(f"    ERROR: {e}", flush=True)

print(f"\nDone: {len(results)}/5 successful", flush=True)
if results:
    # Compare with GT
    from scipy import stats
    for r in results[:2]:
        print(f"\n--- {r['video_id']} ---")
        for dim in TARGETS:
            s = r['scores'].get(dim, '?')
            g = r['gt'].get(dim, '?')
            print(f"  {dim:<25} LLM={s:>8}  GT={g:>8}")
