import os
import requests, json, time

API_KEY = os.getenv("CODEXAPIS_KEY", "")
BASE_URL = "https://www.codexapis.com/v1"

with open("D:/Latex/texlive/LATEX/CueCoT.tex", "r", encoding="utf-8") as f:
    paper = f.read()
with open("D:/Latex/texlive/LATEX/table1_main_results.tex", "r", encoding="utf-8") as f:
    table = f.read()
with open("D:/Latex/texlive/LATEX/aaai2027.bib", "r", encoding="utf-8") as f:
    bib = f.read()

abstract = paper[paper.find("\\begin{abstract}"):paper.find("\\end{abstract}")+len("\\end{abstract}")]
intro = paper[paper.find("\\section{Introduction}"):paper.find("\\section{Related Work}")]
related = paper[paper.find("\\section{Related Work}"):paper.find("\\section{Methodology}")]
method = paper[paper.find("\\section{Methodology}"):paper.find("\\section{Experiments}")]
experiments = paper[paper.find("\\section{Experiments}"):paper.find("\\section{Discussion}")]
discussion = paper[paper.find("\\section{Discussion}"):paper.find("\\bibliography{")]
ethics_sec = paper[paper.find("\\section*{Ethical Statement}"):]

def call_gpt(prompt, label=""):
    print(f"  [{label}] Prompt: {len(prompt)} chars", flush=True)
    try:
        resp = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": "gpt-5.6-sol", "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.3, "max_tokens": 1500},
            timeout=180
        )
        print(f"  [{label}] HTTP {resp.status_code}, {len(resp.text)} bytes", flush=True)
        if resp.status_code != 200:
            print(f"  [{label}] ERROR BODY: {resp.text[:200]}", flush=True)
            return f"HTTP {resp.status_code}: {resp.text[:300]}"
        result = resp.json()
        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        return f"API ERROR: {json.dumps(result)[:300]}"
    except Exception as e:
        return f"EXCEPTION: {e}"

# ROUND 1: Compact
print("=" * 60)
print("ROUND 1: Abstract Claims vs Evidence")
print("=" * 60)
prompt1 = f"""AAAII 2027 paper review. Check every abstract claim against experimental data.

ABSTRACT:
{abstract}

TABLE 1:
{table}

EXPERIMENTAL FACTS:
- Single dataset: RecruitView (2011 videos, 331 participants)
- 17 methods: RF/SVR/Ridge (Face/Voice/Gesture/All), Late Fusion, Attn-LSTM, CRMF, Early Fusion MLP, TF-IDF, GPT-5.6 Direct, CueCoT, LLM Direct
- Best: RF(All) r=0.3829. Best LLM: GPT-5.6 r=-0.1517
- ALL LLM methods negatively correlated with human ratings
- Feature shuffling: near-perfect invariance (LLM ignores behavioral input)
- No human evaluation done
- CueCoT vs Direct: NOT significant (p=0.085)

TASK: List every claim in the abstract. For each, mark SUPPORTED / PARTIALLY / CONTRADICTED with evidence."""

r1 = call_gpt(prompt1, "R1")
print(r1)
print()
time.sleep(2)

# ROUND 2: Methodology
print("=" * 60)
print("ROUND 2: Methodology & Experiments")
print("=" * 60)
prompt2 = f"""AAAII 2027 review - Methodology and Experiments.

METHOD (abridged):
{method[:2500]}

EXPERIMENTS (abridged):
{experiments[:3000]}

QUESTIONS:
1. Abstract says "projected into the LLM's token space as structured soft prompts." Method describes percentile-based natural language descriptions (300-500 tokens). Is this a discrepancy?
2. User-level 5-seed split: proper? GPT-5.6 on ALL videos without splits - valid comparison?
3. Feature extraction: MediaPipe + librosa - adequate for AAAI 2027?
4. TF-IDF baseline: too weak? Should use embeddings?
5. 14 baselines claimed: count them. Are they all meaningful comparisons?
6. Statistical tests: bootstrap CIs, Cohen's d - appropriate?
7. Rate experimental rigor 1-10."""

r2 = call_gpt(prompt2, "R2")
print(r2)
print()
time.sleep(2)

# ROUND 3
print("=" * 60)
print("ROUND 3: References")
print("=" * 60)
prompt3 = f"""AAAII 2027 review - References check.

RELATED WORK:
{related[:1500]}

BIB:
{bib[:3000]}

QUESTIONS:
1. Any fabricated or unverifiable references?
2. Key missing citations?
3. Is "first to apply CoT to behavioral assessment" claim justified?
4. RecruitView citation correct?
5. Rate reference quality 1-10."""

r3 = call_gpt(prompt3, "R3")
print(r3)
print()
time.sleep(2)

# ROUND 4
print("=" * 60)
print("ROUND 4: Discussion, Ethics & Final Verdict")
print("=" * 60)
prompt4 = f"""AAAII 2027 review - Final verdict.

DISCUSSION:
{discussion[:2500]}

ETHICS:
{ethics_sec[:1000]}

INTRO (framing):
{intro[:1500]}

QUESTIONS:
1. Paper reframes FAILED method as contribution ("behavioral-signal assumption is wrong"). Honest? Valid?
2. Are three explanations well-supported?
3. Abstract-body tension: handled well?
4. Ethics adequate?
5. Strengths: cross-model, shuffling controls.
6. Weaknesses: negative results, single dataset, no human eval.

FINAL VERDICT (pick one):
STRONG ACCEPT / WEAK ACCEPT / BORDERLINE / WEAK REJECT / STRONG REJECT

Justify in 3-5 sentences. This is the FINAL call."""

r4 = call_gpt(prompt4, "R4")
print(r4)

# Save
full = f"""============================================================
ROUND 1: Abstract Claims vs Evidence
============================================================
{r1}

============================================================
ROUND 2: Methodology & Experiments
============================================================
{r2}

============================================================
ROUND 3: References
============================================================
{r3}

============================================================
ROUND 4: Discussion, Ethics & Final Verdict
============================================================
{r4}
"""
with open("D:/jingxin/experiments/_gpt_review_batch.txt", "w", encoding="utf-8") as f:
    f.write(full)
print("\n[DONE]")
