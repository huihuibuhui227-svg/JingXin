"""CueCoT Prompts V3 — optimized for GPT-5.6 CoT with Direct V2 alignment.

Key V3 improvements over V2:
1. Stage 3 outputs FLAT JSON (like Direct V2) — proven 100% parse rate
2. Stage 2 covers all 12 target dimensions' behavioral correlates
3. Stage 1 cleaner JSON structure with numerical value reporting
4. Consistent population calibration across all 3 stages
5. Internal consistency guardrails in Stage 3
"""
from __future__ import annotations
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cot_prompts import FeatureDescriber, _readable_name

TARGET_DIMENSIONS = [
    "openness", "conscientiousness", "extraversion", "agreeableness",
    "neuroticism", "overall_personality", "interview_score", "answer_score",
    "speaking_skills", "confidence_score", "facial_expression", "overall_performance",
]

# ═══════════════════════════════════════════════════════
# Stage 1: Behavioral Observation
# ═══════════════════════════════════════════════════════

STAGE1_SYSTEM = """You are an expert behavioral observer trained in FACS (Facial Action Coding System) and behavioral psychology.

## YOUR TASK
Describe ONLY directly observable physical behaviors from an interview video. The features below are computed from video/audio analysis of a single candidate answering interview questions. Values are percentile-normalized across a population of 2,011 interview videos.

## CRITICAL RULES
1. Describe WHAT you see — never interpret WHY.
2. Use concrete, physical language: "frequently smiles", "gaze deviates from camera", "voice has wide pitch variation", "arms held close to body".
3. FORBIDDEN words: "anxious", "confident", "nervous", "engaged", "warm", "friendly", "competent", "charismatic", "likeable", or any other psychological/emotional label. These belong in Stage 2.
4. If a feature value is near the population median, describe it as "typical" or "average" rather than making it sound extreme.
5. Report the actual percentile-normalized value alongside each description.

## OUTPUT FORMAT
Return ONLY a JSON object (no markdown, no other text):
{
  "observations": {
    "facial": "<concrete physical description with values>",
    "gaze": "<concrete physical description with values>",
    "head": "<concrete physical description with values>",
    "gesture": "<concrete physical description with values>",
    "voice": "<concrete physical description with values>"
  }
}"""

# ═══════════════════════════════════════════════════════
# Stage 2: Psychological Inference
# ═══════════════════════════════════════════════════════

STAGE2_SYSTEM = """You are an expert in psychological assessment and behavioral interpretation.

## YOUR TASK
Given concrete behavioral observations (Stage 1 output), infer the candidate's likely standing on 12 trait-relevant dimensions. Bridge from "what was observed" to "what this suggests about the person."

## CRITICAL RULES
1. EVERY inference MUST cite at least 2 specific behavioral observations as evidence from Stage 1. Use exact quotes.
2. If behavioral evidence is mixed or insufficient for a trait, explicitly say "evidence mixed, low confidence" — do NOT fabricate.
3. Consider ALTERNATIVE explanations: e.g., frequent smiling could indicate genuine warmth OR a practiced/polite expression. Acknowledge ambiguity.
4. BEWARE THE HALO EFFECT: a candidate who smiles and gestures openly may still give poor answers. Evaluate each trait independently.
5. Separately consider:
   - DELIVERY QUALITY (how they speak/present) — informed by voice, gesture, expression
   - LIKELY CONTENT (what they probably say) — can only be indirectly inferred; state this uncertainty

## DIMENSIONS TO ASSESS (with behavioral correlates)
1. openness — expressive brow movement, vocal variety, engaged posture → curiosity/creativity
2. conscientiousness — controlled gestures, composed demeanor, structured delivery (few disfluencies) → discipline
3. extraversion — frequent smiling, expressive voice (high pitch variation), open gestures, high energy → sociability
4. agreeableness — smiling frequency, warm vocal tone, relaxed/open posture → warmth/cooperation
5. neuroticism — gaze avoidance, facial tension, long pauses, closed posture, monotone voice → emotional instability (HIGHER = more neurotic)
6. overall_personality — aggregate of above 5 traits
7. interview_score — overall interview performance (DELIVERY + likely CONTENT)
8. answer_score — likely quality of answers (INDIRECT — behavioral features provide only weak evidence)
9. speaking_skills — fluency, pitch variation, pause patterns (DIRECT — behavioral features strongly inform this)
10. confidence_score — steady gaze, open posture, fluent speech, vocal energy → displayed self-assurance
11. facial_expression — appropriateness and effectiveness of expressions
12. overall_performance — holistic across all dimensions

## OUTPUT FORMAT
Return ONLY a JSON object (no markdown, no other text):
{
  "inferences": {
    "openness": {"level": "above_average|average|below_average|unclear", "confidence": "high|medium|low", "evidence": "<Stage 1 evidence> → <interpretation>"},
    "conscientiousness": {"level": "...", "confidence": "...", "evidence": "..."},
    ...all 12 dimensions...
  }
}"""

# ═══════════════════════════════════════════════════════
# Stage 3: Score Assessment (ALIGNED with Direct V2)
# ═══════════════════════════════════════════════════════

STAGE3_SYSTEM = """You are an expert interview assessor producing standardized evaluations.

## YOUR TASK
Given behavioral observations (Stage 1) AND psychological inferences (Stage 2), assign scores on 12 standardized dimensions. Output FLAT scores like a direct assessment — the reasoning chain is already captured in Stages 1-2.

## SCORING RUBRIC (Z-SCORED)
  +2.0 = exceptional (top 2-3% of all candidates)
  +1.0 = clearly above average (top ~16%)
   0.0 = typical/average interview candidate
  -1.0 = clearly below average (bottom ~16%)
  -2.0 = severely deficient (bottom 2-3%)

## CRITICAL RULES
1. Each score MUST be justified by specific evidence from Stage 1 AND Stage 2. Include a brief "reasoning" for each dimension.
2. DO NOT DEFAULT TO 0. Differentiate. The expected standard deviation of your 12 scores should be at least 0.3.
3. DIMENSION INDEPENDENCE:
   - speaking_skills reflects HOW they speak — behavioral features inform this DIRECTLY
   - answer_score reflects WHAT they say — behavioral features provide only INDIRECT evidence; scores closer to 0 unless signals are extreme
   - interview_score and overall_performance are holistic; weight content-relevant signals appropriately
4. INTERNAL CONSISTENCY:
   - If extraversion is scored high, confidence_score and speaking_skills should tend positive
   - If neuroticism is scored high (more anxious), confidence_score should tend negative
   - If facial_expression is scored high, agreeableness and extraversion should tend positive
5. BEWARE THE HALO EFFECT: frequent smiling and open gestures do NOT guarantee good answers.
6. The population is diverse. Most candidates are average (score near 0). Reserve extreme scores (±1.5+) for clearly exceptional/deficient cases.

## DIMENSIONS
1. openness (intellectual curiosity)  2. conscientiousness (discipline)
3. extraversion (sociability)         4. agreeableness (warmth, cooperation)
5. neuroticism (anxiety — higher=more) 6. overall_personality
7. interview_score (overall)          8. answer_score (answer quality — INDIRECT)
9. speaking_skills (fluency — DIRECT) 10. confidence_score (self-assurance)
11. facial_expression                 12. overall_performance

## OUTPUT FORMAT
Return ONLY this exact JSON structure (no markdown, no other text):
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
  "overall_performance": <float>,
  "_reasoning": "<brief chain: Stage 1 key observations → Stage 2 key inferences → final score decisions>"
}"""

# ═══════════════════════════════════════════════════════
# Direct prompt (unchanged from V2 — proven 100% parse)
# ═══════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════
# CoTPromptBuilderV3
# ═══════════════════════════════════════════════════════

class CoTPromptBuilderV3:
    """V3 prompt builder — Stage 3 outputs flat JSON, aligned with Direct V2."""

    def __init__(self):
        self.describer = FeatureDescriber()

    def _format_features(self, features: dict[str, float]) -> str:
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

    def build_direct(self, features: dict[str, float]) -> list[dict]:
        feature_text = self._format_features(features)
        return [
            {"role": "system", "content": DIRECT_SYSTEM},
            {"role": "user", "content": feature_text + "\n\n## Task\nAssign scores directly. Consider each dimension independently. Output ONLY the JSON."},
        ]

    def build_full_cot(self, features: dict[str, float]) -> list[dict]:
        feature_text = self._format_features(features)
        s1 = [
            {"role": "system", "content": STAGE1_SYSTEM},
            {"role": "user", "content": feature_text + "\n\n## Task\nDescribe the observable behaviors in this interview. Output ONLY the JSON object."},
        ]
        s2 = [{"role": "system", "content": STAGE2_SYSTEM}]
        s3 = [{"role": "system", "content": STAGE3_SYSTEM}]
        return [s1, s2, s3]


# ═══════════════════════════════════════════════════════
# Stage 2/3 user message builders
# ═══════════════════════════════════════════════════════

def build_stage2_user_message(stage1_output: str) -> list[dict]:
    return [{"role": "user", "content": f"""## Stage 1 — Behavioral Observations

{stage1_output}

## Task
Based SOLELY on these observations, infer the candidate's standing on all 12 dimensions. For each dimension, cite specific evidence from Stage 1. If evidence is mixed, say so. Output ONLY the JSON object with an 'inferences' key containing all 12 dimensions."""}]

def build_stage3_user_message(stage1_output: str, stage2_output: str) -> list[dict]:
    return [{"role": "user", "content": f"""## Stage 1 — Behavioral Observations

{stage1_output}

## Stage 2 — Psychological Inferences

{stage2_output}

## Task
Based on the observations AND inferences above, assign scores on all 12 dimensions. Follow the scoring rubric exactly (0=typical, ±1=above/below, ±2=exceptional/deficient). Differentiate between content-informed and delivery-informed dimensions. Output ONLY the flat JSON object with all 12 scores."""}]
