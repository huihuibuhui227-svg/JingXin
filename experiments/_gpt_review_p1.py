import os
import requests, json

API_KEY = os.getenv("CODEXAPIS_KEY", "")
BASE_URL = "https://www.codexapis.com/v1"

with open("D:/Latex/texlive/LATEX/CueCoT.tex", "r", encoding="utf-8") as f:
    paper = f.read()
with open("D:/Latex/texlive/LATEX/table1_main_results.tex", "r", encoding="utf-8") as f:
    table = f.read()
with open("D:/Latex/texlive/LATEX/aaai2027.bib", "r", encoding="utf-8") as f:
    bib = f.read()

# Only send key sections
abstract = paper[paper.find("\\begin{abstract}"):paper.find("\\end{abstract}")+len("\\end{abstract}")]
experiments = paper[paper.find("\\section{Experiments}"):paper.find("\\section{Discussion}")]
discussion = paper[paper.find("\\section{Discussion}"):paper.find("\\bibliography{")]

# PART 1: Abstract, Claims vs Data, Table
prompt1 = f"""You are a senior AAAI 2027 area chair (track: AI for Social Impact). Perform a rigorous DESK REJECT review.

TASK: Check every factual claim in the abstract against the experimental data and table. Flag ALL discrepancies.

ABSTRACT:
{abstract}

MAIN RESULTS TABLE (Table 1):
{table}

EXPERIMENTAL FINDINGS (from paper body):
{experiments}

ANALYZE:
1. List every factual claim in the abstract (sentence by sentence). For each, state whether the experimental data SUPPORTS, PARTIALLY SUPPORTS, or CONTRADICTS it. Be specific with numbers.

2. The paper uses only ONE dataset (RecruitView). The abstract says "a public interview benchmark" — is this accurate? Any other dataset issues?

3. Table integrity: Are all computed values internally consistent? For each method, verify macro = (personality + interview)/2. Flag any anomalies.

4. The 14 "baselines" + 3 LLM methods = 17 methods. Are all properly categorized? Are the "LLM methods negatively correlated" claims accurate?

5. The body says CueCoT vs Direct is "not statistically significant" (p=0.085). Does this undermine the paper's contribution?

6. Rate CLAIM-EVIDENCE ALIGNMENT on a scale of 1-10. Justify.

Output a structured report.
"""

print(f"Part 1 prompt: {len(prompt1)} chars", flush=True)

resp = requests.post(
    f"{BASE_URL}/chat/completions",
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    json={"model": "gpt-5.6-sol", "messages": [{"role": "user", "content": prompt1}],
          "temperature": 0.3, "max_tokens": 2500},
    timeout=300
)

print(f"Status: {resp.status_code}, Length: {len(resp.text)}", flush=True)
result = resp.json()
if "choices" in result:
    output = result["choices"][0]["message"]["content"]
    print(output)
    with open("D:/jingxin/experiments/_gpt_review_p1.txt", "w", encoding="utf-8") as f:
        f.write(output)
else:
    print("ERROR:", json.dumps(result, indent=2)[:500])
