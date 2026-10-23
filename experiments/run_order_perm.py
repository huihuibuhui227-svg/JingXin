#!/usr/bin/env python
"""True within-sample input-order permutation control (GPT-5.6 family, Direct v2).

Scientific goal
---------------
The paper's "input-order sensitivity" claim requires the *same* behavioral feature
content to be re-presented to the model in a different *order*. The pre-existing
shuffle run (run_feature_shuffle.py) instead swapped features ACROSS videos
(cross-candidate mismatch) — that is not an order test.

This script re-runs the exact direct_v2 prompt (cot_prompts_v2.DIRECT_SYSTEM +
CoTPromptBuilderV2-style feature text) but shuffles, per video and per permutation
index, only the *order of the "- <desc>" bullets* under "## Detailed Feature Values".
Header/summary and every bullet's content are unchanged. If the model is order-robust,
permuted-order scores should correlate ~ +1.0 with the same model's original-order
scores; if it anchors on presentation order, the correlation drops.

Design
------
- Model: gpt-5.6-sol (the luna channel on the provider is temporarily unavailable; the
  codexapis aliases map to the same gpt-5.6 family). To keep the test internally
  consistent we run BOTH the original-order condition (perm=-1) and K_PERM shuffled-order
  conditions on the SAME 200-video subset, all on gpt-5.6-sol.
- Same prompt/content as direct_v2 except bullet order. If the model is order-robust,
  shuffled-order scores correlate ~ +1.0 with the same model's original-order scores.
Usage:
    python run_order_perm.py --smoke     # quick check: first ~8 videos x (orig, perm0)
    python run_order_perm.py             # full: 200 videos x (orig + K_PERM shuffles)
"""
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI  # noqa: E402
from cot_prompts import FeatureDescriber  # noqa: E402
from cot_prompts_v2 import DIRECT_SYSTEM  # noqa: E402

# ===== Config =====
API_KEY = os.getenv("CODEXAPIS_KEY", "")
API_URL = os.getenv("CODEXAPIS_URL", "https://www.codexapis.com/v1")
MODEL = os.getenv("GPT_MODEL", "gpt-5.6-sol")  # luna channel was down -> sol (see session)
METHOD_PREFIX = "direct_orderperm"
NUM_WORKERS = 12
SUBSET_SIZE = 200
SEED = 42
K_PERM = 2          # shuffled-order conditions (plus one original-order condition, perm=-1)
SMOKE = "--smoke" in sys.argv
SMOKE_N = 8

FEATURES_DIR = os.path.expanduser("~/jingxin/experiments/results/features")
OUTPUT_PATH = f"{FEATURES_DIR}/llm_results_codexapis_orderperm.jsonl"
PROGRESS_PATH = f"{FEATURES_DIR}/codexapis_orderperm_progress.json"

TARGETS = ["openness", "conscientiousness", "extraversion", "agreeableness",
           "neuroticism", "overall_personality", "interview_score", "answer_score",
           "speaking_skills", "confidence_score", "facial_expression", "overall_performance"]

TASK_APPENDIX = ("\n\n## Task\nAssign scores directly. Consider each dimension "
                 "independently. Output ONLY the JSON.")

write_lock = Lock()
progress_lock = Lock()
completed = set()


# ===== Prompt construction (identical to direct_v2 except bullet ORDER) =====
def build_permuted_messages(row: dict, perm_idx: int) -> list[dict]:
    desc = FeatureDescriber.describe(row)
    obs = list(desc["observations"])  # keep bullet contents identical
    if perm_idx >= 0:                 # perm_idx = -1  -> original (canonical) order
        rng = random.Random(100_000 + perm_idx)  # deterministic per perm index
        rng.shuffle(obs)
    lines = [
        "## Behavioral Feature Measurements",
        "Computed from video/audio of one candidate answering interview questions.",
        "Values are percentile-normalized across 2,011 videos. Median ≈ typical candidate.",
        "",
        desc["summary"],
        "",
        "## Detailed Feature Values",
    ]
    for o in obs:
        lines.append(f"- {o['description']}")
    feature_text = "\n".join(lines)
    return [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": feature_text + TASK_APPENDIX},
    ]


def parse_scores(text: str) -> dict:
    scores = {}
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        m = re.search(r"(\{[^{}]*\"openness\"[^{}]*\})", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(1))
            for d in TARGETS:
                if d in parsed:
                    val = parsed[d]
                    if isinstance(val, dict):
                        scores[d] = float(val.get("score", val))
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
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=messages, temperature=0.0, max_tokens=max_tokens)
            raw = resp.choices[0].message.content or ""
            return raw, resp.usage.total_tokens
        except Exception:
            time.sleep(2 ** attempt)
    raise RuntimeError("All retries failed")


def process_one(row, perm_idx):
    vid = row["video_id"]
    client = OpenAI(api_key=API_KEY, base_url=API_URL)
    msgs = build_permuted_messages(row, perm_idx)
    raw, tok = call_api(client, msgs)
    scores = parse_scores(raw)
    method = "direct_v2_orig" if perm_idx < 0 else f"{METHOD_PREFIX}_p{perm_idx}"
    return {
        "video_id": vid,
        "perm": perm_idx,
        "method": method,
        "model": MODEL,
        "temperature": 0.0,
        "scores": scores,
        "tokens_used": tok,
        "parse_success": len(scores) == 12,
    }


# ===== IO helpers =====
def save_result(result):
    with write_lock:
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()


def load_subset():
    rows = []
    with open(f"{FEATURES_DIR}/all_features.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    stride = max(1, len(rows) // SUBSET_SIZE)
    subset = rows[::stride][:SUBSET_SIZE]
    return rows, subset


def load_completed():
    keys = set()
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            keys = set(json.load(f))
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    keys.add(f"{r['video_id']}__{r['perm']}")
    return keys


# ===== Main =====
def main():
    rows, subset = load_subset()
    if SMOKE:
        subset = subset[:SMOKE_N]
        perms = [-1, 0]                     # original + one shuffle
        label = f"SMOKE (first {len(subset)} videos x {perms})"
    else:
        perms = [-1] + list(range(K_PERM))  # original-order + K_PERM shuffles
        label = f"FULL ({len(subset)} videos x {len(perms)} conds = {len(subset) * len(perms)} calls, model={MODEL})"

    completed.update(load_completed())
    tasks = [(row, p) for row in subset for p in perms
             if f"{row['video_id']}__{p}" not in completed]
    print(f"[{label}] subset={len(subset)} perms={perms} todo={len(tasks)}", flush=True)

    if not tasks:
        print("Nothing to do.", flush=True)
        return

    def run(args):
        row, p = args
        vid = row["video_id"]
        key = f"{vid}__{p}"
        t0 = time.time()
        try:
            res = process_one(row, p)
            if res is not None and res.get("scores"):
                save_result(res)
                with progress_lock:
                    completed.add(key)
                n = len(res["scores"])
                print(f"[{key}] parsed {n}/12 | {time.time() - t0:.0f}s | {res['tokens_used']} tok", flush=True)
                return True
            print(f"[{key}] EMPTY scores — retried, skipping", flush=True)
            return False
        except Exception as e:
            print(f"[{key}] ERROR: {e}", flush=True)
            return False

    n_ok = 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        for ok in ex.map(run, tasks):
            n_ok += int(ok)
            if not SMOKE and n_ok % 60 == 0:
                with open(PROGRESS_PATH, "w") as f:
                    json.dump(sorted(completed), f)

    with open(PROGRESS_PATH, "w") as f:
        json.dump(sorted(completed), f)
    print(f"Done. ok={n_ok}/{len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
