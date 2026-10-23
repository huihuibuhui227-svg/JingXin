import os
import requests, json, sys

API_KEY = os.getenv("CODEXAPIS_KEY", "")
BASE_URL = "https://www.codexapis.com/v1"

with open("D:/Latex/texlive/LATEX/CueCoT.tex", "r", encoding="utf-8") as f:
    paper = f.read()
with open("D:/Latex/texlive/LATEX/table1_main_results.tex", "r", encoding="utf-8") as f:
    table = f.read()
with open("D:/Latex/texlive/LATEX/aaai2027.bib", "r", encoding="utf-8") as f:
    bib = f.read()

# Split paper into manageable sections
abstract = paper[paper.find("\\begin{abstract}"):paper.find("\\end{abstract}")+len("\\end{abstract}")]
intro = paper[paper.find("\\section{Introduction}"):paper.find("\\section{Related Work}")]
related = paper[paper.find("\\section{Related Work}"):paper.find("\\section{Methodology}")]
method = paper[paper.find("\\section{Methodology}"):paper.find("\\section{Experiments}")]
experiments = paper[paper.find("\\section{Experiments}"):paper.find("\\section{Discussion}")]
discussion = paper[paper.find("\\section{Discussion}"):paper.find("\\section*{Ethical Statement}")]
ethics = paper[paper.find("\\section*{Ethical Statement}"):paper.find("\\bibliography{")]

review_prompt = f"""You are a senior AAAI 2027 area chair conducting a final rigorous review. This is a DESK REJECT decision: you must determine if this paper meets the bar for publication at AAAI.

The paper proposes CueCoT, an LLM Chain-of-Thought framework for automated interview assessment using multimodal behavioral cues (facial, gesture, voice).

Evaluate it critically on ALL of the following dimensions. Be harsh and honest—this is a top-tier AI conference.

## COMPLETE PAPER:
ABSTRACT:
{abstract}

INTRODUCTION:
{intro}

RELATED WORK:
{related}

METHODOLOGY:
{method}

EXPERIMENTS:
{experiments}

DISCUSSION:
{discussion}

ETHICAL STATEMENT:
{ethics}

MAIN RESULTS TABLE:
{table}

BIBLIOGRAPHY:
{bib}

---

## REVIEW QUESTIONS (answer each with score 1-5 and detailed justification):

### 1. TECHNICAL SOUNDNESS (1-5)
Are the methods correct? Are the experiments properly designed? Are statistical tests appropriate?
- User-level splits (good), bootstrap tests (good), but check: is the feature extraction pipeline valid?
- The shuffling experiment: is the conclusion "LLM ignores behavioral signal" justified by Δr=0.009?

### 2. NOVELTY & SIGNIFICANCE (1-5)
Is this the first to apply CoT to behavioral interview assessment? Is the contribution meaningful?
- The paper essentially reports a NEGATIVE result (LLM CoT doesn't work for this task). Is that contribution enough for AAAI?
- The behavioral-signal assumption critique: novel or already known?

### 3. CLAIM-EVIDENCE ALIGNMENT (1-5)
Do the claims in the abstract match what the experiments actually show?
- Check every sentence in the abstract against the experimental data.
- Does the body honestly report negative results?

### 4. EXPERIMENTAL RIGOR (1-5)
- 17 methods compared (comprehensive)
- Only ONE dataset (RecruitView)—is this sufficient?
- TF-IDF baseline: is this strong enough? Why not use sentence embeddings?
- No human evaluation—how damaging is this?
- GPT-5.6 run only on all 2011 videos (no 5-seed split)—is this a valid comparison?
- The "14 baselines" claim: count them and verify.

### 5. REFERENCE QUALITY (1-5)
- Check for fabricated references. Are all verifiable?
- The deleted huang2024crmf: was the method properly re-attributed?
- Any missing key citations?

### 6. WRITING QUALITY (1-5)
- Clarity, organization, figure quality
- Only 2 figures: sufficient?
- The abstract-body tension: handled well or problematic?

### 7. ETHICAL CONSIDERATIONS
- Are ethical risks properly discussed?
- The paper argues behavioral signals are weak predictors—does this strengthen or weaken the ethical argument?

## FINAL VERDICT:
- [ ] Strong Accept
- [ ] Weak Accept
- [ ] Borderline
- [ ] Weak Reject
- [ ] Strong Reject

Then give a 3-5 sentence summary for the authors explaining your decision.
Focus on whether this paper, AS IS, should be accepted to AAAI 2027.
"""

print(f"Prompt: {len(review_prompt)} chars", flush=True)

resp = requests.post(
    f"{BASE_URL}/chat/completions",
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    json={"model": "gpt-5.6-sol", "messages": [{"role": "user", "content": review_prompt}],
          "temperature": 0.3, "max_tokens": 4000},
    timeout=300
)

print(f"Status: {resp.status_code}, Length: {len(resp.text)}", flush=True)
try:
    result = resp.json()
    if "choices" in result:
        output = result["choices"][0]["message"]["content"]
        print(output)
        with open("D:/jingxin/experiments/_gpt_review_final.txt", "w", encoding="utf-8") as f:
            f.write(output)
        print("\n[DONE]")
    else:
        print("ERROR:", json.dumps(result, indent=2)[:800])
except Exception as e:
    print(f"Error: {e}\nRaw: {resp.text[:300]}")
