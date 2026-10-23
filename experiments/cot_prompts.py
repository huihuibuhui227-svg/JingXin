#!/usr/bin/env python
"""
CueCoT Prompt 模板 — CueCoT 实验阶段三
=======================================
将聚合后的特征向量转换为 LLM 可理解的 Chain-of-Thought 提示词。

提供：
  - FeatureDescriber: 特征向量 → 自然语言行为描述
  - 5 种 Prompt 变体: Full CoT (3-stage), Direct, Generic CoT, Skip S1, Skip S2
  - 2 个 Few-shot 完整推理链示例

用法：
  from cot_prompts import CoTPromptBuilder
  builder = CoTPromptBuilder()
  prompts = builder.build_full_cot(feature_dict)  # 返回 3 组 messages
"""

from __future__ import annotations

import json
from typing import Any

# ═══════════════════════════════════════════════════════
# 12 维目标维度
# ═══════════════════════════════════════════════════════

TARGET_DIMENSIONS = [
    "openness", "conscientiousness", "extraversion", "agreeableness",
    "neuroticism", "overall_personality", "interview_score", "answer_score",
    "speaking_skills", "confidence_score", "facial_expression", "overall_performance",
]

DIMENSION_DESCRIPTIONS = {
    "openness": "Openness to experience — intellectual curiosity, creativity, preference for novelty",
    "conscientiousness": "Conscientiousness — organization, diligence, self-discipline, reliability",
    "extraversion": "Extraversion — sociability, assertiveness, energy, enthusiasm",
    "agreeableness": "Agreeableness — cooperation, empathy, trust, modesty",
    "neuroticism": "Neuroticism — emotional instability, anxiety, moodiness (higher = more neurotic)",
    "overall_personality": "Overall personality composite score",
    "interview_score": "Overall interview performance rating",
    "answer_score": "Quality and relevance of answers to interview questions",
    "speaking_skills": "Verbal communication fluency, clarity, and effectiveness",
    "confidence_score": "Displayed confidence and self-assurance",
    "facial_expression": "Appropriateness and effectiveness of facial expressions",
    "overall_performance": "Holistic performance evaluation across all dimensions",
}


# ═══════════════════════════════════════════════════════
# FeatureDescriber: 特征向量 → 自然语言
# ═══════════════════════════════════════════════════════

class FeatureDescriber:
    """将聚合后的特征向量转换为分组自然语言行为观察。

    使用 Phase 2 分析得出的总体百分位数阈值来判断"高于平均"/"低于平均"。
    """

    # 阈值定义：key → (low_threshold, high_threshold, label_low, label_mid, label_high)
    THRESHOLDS = {
        "face.au12_smile_mean": (0.02, 0.07, "rarely smiles", "occasionally smiles", "frequently smiles"),
        "face.au1_inner_brow_raise_mean": (0.13, 0.19, "minimal brow movement", "moderate brow raises", "frequent expressive brow raises"),
        "face.au4_frown_mean": (0.001, 0.01, "minimal frowning", "occasional frowning", "noticeable frowning"),
        "face.gaze_deviation_mean": (0.42, 0.62, "steady, direct gaze", "moderate gaze variation", "frequently looks away from camera"),
        "face.tension_score_mean": (0.40, 0.54, "relaxed facial expression", "mild facial tension", "visibly tense facial expression"),
        "face.emotion_anxiety_mean": (0.15, 0.40, "low anxiety signals", "moderate anxiety cues", "elevated anxiety indicators"),
        "face.head_yaw_mean": (0.24, 0.29, "minimal head turning", "moderate head movement", "frequent head turning"),
        "gesture.left_arm_arm_angle_mean": (15.0, 32.0, "arms held close to body", "moderate arm openness", "wide, open arm gestures"),
        "gesture.right_arm_arm_angle_mean": (15.0, 32.0, "arms held close to body", "moderate arm openness", "wide, open arm gestures"),
        "gesture.upper_body_head_jitter_mean": (0.006, 0.015, "very steady head", "slight head movement", "noticeable head tremor or sway"),
        "gesture.upper_body_torso_stability_mean": (0.92, 0.97, "mobile upper body with postural shifts", "moderate torso stability", "very rigid, still posture"),
        "gesture.shoulder_shrug_level_mean": (0.005, 0.05, "relaxed shoulders", "occasional shoulder raises", "frequent shoulder shrugging"),
        "voice.pitch_mean": (123.0, 193.0, "low-pitched voice", "mid-range voice pitch", "high-pitched voice"),
        "voice.pitch_std": (18.0, 56.0, "monotone, flat delivery", "moderate pitch variation", "highly expressive, varied intonation"),
        "voice.pitch_trend": (-15.0, 2.0, "falling intonation (statements concluding)", "steady pitch level", "rising intonation (engaging, questioning)"),
        "voice.pause_frequency": (24.0, 34.0, "few pauses, fluent speech", "moderate pause frequency", "frequent pauses, hesitant speech"),
        "voice.pause_duration_mean": (0.35, 0.52, "short pauses, quick transitions", "moderate pause duration", "long pauses between utterances"),
        "voice.energy_mean": (0.025, 0.038, "soft, quiet voice", "moderate voice energy", "loud, energetic voice"),
    }

    @classmethod
    def describe(cls, features: dict[str, float]) -> dict:
        """将特征扁平字典转换为分组自然语言观察。

        返回 {"groups": {...}, "observations": [...], "summary": "..."}
        """
        observations = []
        groups: dict[str, list[str]] = {
            "facial_expression": [],
            "eye_gaze": [],
            "head_movement": [],
            "gesture_posture": [],
            "voice_prosody": [],
        }

        for feature_key, (lo, hi, label_lo, label_mid, label_hi) in cls.THRESHOLDS.items():
            value = features.get(feature_key)
            if value is None:
                continue

            if value < lo:
                desc = label_lo
            elif value < hi:
                desc = label_mid
            else:
                desc = label_hi

            # 缩短特征名为可读标签
            readable = _readable_name(feature_key)
            observation_text = f"{readable}: {desc} (value={value:.3f})"
            observations.append({"feature": feature_key, "value": round(value, 4), "description": desc})

            # 分组
            group = cls._feature_group(feature_key)
            groups[group].append(observation_text)

        # 构建摘要段落
        summary_parts = []
        if groups["facial_expression"]:
            summary_parts.append("Facial: " + "; ".join(groups["facial_expression"]))
        if groups["eye_gaze"]:
            summary_parts.append("Gaze: " + "; ".join(groups["eye_gaze"]))
        if groups["head_movement"]:
            summary_parts.append("Head: " + "; ".join(groups["head_movement"]))
        if groups["gesture_posture"]:
            summary_parts.append("Posture/Gesture: " + "; ".join(groups["gesture_posture"]))
        if groups["voice_prosody"]:
            summary_parts.append("Voice: " + "; ".join(groups["voice_prosody"]))

        return {
            "groups": groups,
            "observations": observations,
            "summary": "\n".join(summary_parts),
        }

    @staticmethod
    def _feature_group(key: str) -> str:
        if any(k in key for k in ("smile", "frown", "au1_", "au4_", "tension", "emotion_anxiety")):
            return "facial_expression"
        if "gaze" in key:
            return "eye_gaze"
        if "head" in key:
            return "head_movement"
        if any(k in key for k in ("arm_", "shoulder", "torso", "head_jitter")):
            return "gesture_posture"
        if key.startswith("voice."):
            return "voice_prosody"
        return "facial_expression"


def _readable_name(key: str) -> str:
    """将 'face.au12_smile_mean' 转为 'Smile frequency'。"""
    # 去掉前缀
    clean = key.replace("face.", "").replace("gesture.", "").replace("voice.", "")
    # 去掉后缀 _mean
    clean = clean.replace("_mean", "").replace("_std", "")
    # 映射常见特征到可读名
    mapping = {
        "au12_smile": "Smile frequency",
        "au1_inner_brow_raise": "Brow raises (surprise/emphasis)",
        "au4_frown": "Frowning",
        "gaze_deviation": "Gaze stability",
        "tension_score": "Facial tension",
        "emotion_anxiety": "Anxiety indicators",
        "head_yaw": "Head turning",
        "left_arm_arm_angle": "Left arm openness",
        "right_arm_arm_angle": "Right arm openness",
        "upper_body_head_jitter": "Head steadiness",
        "upper_body_torso_stability": "Posture rigidity",
        "shoulder_shrug_level": "Shoulder shrugging",
        "pitch_mean": "Voice pitch level",
        "pitch_std": "Pitch variation (expressiveness)",
        "pitch_trend": "Intonation trend",
        "pause_frequency": "Speech fluency (pauses)",
        "pause_duration_mean": "Pause duration",
        "energy_mean": "Voice energy/loudness",
    }
    return mapping.get(clean, clean.replace("_", " ").title())


# ═══════════════════════════════════════════════════════
# CoTPromptBuilder: 5 种 Prompt 变体
# ═══════════════════════════════════════════════════════

class CoTPromptBuilder:
    """构建 CoT 各阶段的 System / User 消息对。"""

    # ---------- 共享的分数输出格式说明 ----------
    SCORE_FORMAT_INSTRUCTION = """## Output Format
Return a JSON object with exactly 12 scores. Each score must be a float.
Scores are normalized (z-scored), typically ranging from -2.0 to +2.0:
  - positive = above average on this dimension
  - negative = below average
  - near 0 = average

Dimensions:
1. openness           — intellectual curiosity, creativity
2. conscientiousness  — organization, diligence, reliability
3. extraversion       — sociability, assertiveness, energy
4. agreeableness      — cooperation, empathy, modesty
5. neuroticism        — emotional instability (higher = more anxious/moody)
6. overall_personality— personality composite
7. interview_score    — overall interview performance
8. answer_score       — quality of answers
9. speaking_skills    — verbal fluency and clarity
10. confidence_score  — displayed confidence
11. facial_expression — appropriateness of expressions
12. overall_performance— holistic evaluation

Output ONLY this JSON (no other text):
```json
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
}
```"""

    # ---------- Few-shot 示例 ----------
    FEWSHOT_EXAMPLE_A = """## Example A: Anxious, Low-Confidence Candidate

### Stage 1 — Behavioral Observations
```json
{
  "observations": {
    "facial": "Rarely smiles (mean=0.008). Mild facial tension (0.47). Elevated anxiety indicators (0.52). Minimal frowning.",
    "gaze": "Frequently looks away from camera (gaze deviation=0.68).",
    "head": "Moderate head movement (yaw=0.27).",
    "gesture": "Arms held close to body (left_angle=12.3°, right_angle=14.1°). Slight head tremor (jitter=0.017). Moderately rigid posture (stability=0.94).",
    "voice": "Low-pitched voice (107Hz). Monotone delivery (pitch_std=15.2). Falling intonation (-18.3). Frequent pauses (37.1/min) with long durations (0.58s)."
  }
}
```

### Stage 2 — Psychological Inferences
```json
{
  "inferences": [
    {"trait": "anxiety", "level": "high", "evidence": "Elevated anxiety indicators (0.52), frequent gaze avoidance (0.68), frequent long pauses (37/min, 0.58s avg), monotone voice (pitch_std=15.2)."},
    {"trait": "confidence", "level": "low", "evidence": "Closed arm posture (angles 12-14°), gaze avoidance, frequent hesitations, falling intonation suggesting lack of assertiveness."},
    {"trait": "extraversion", "level": "low", "evidence": "Monotone delivery, closed posture, rare smiling, low vocal energy — consistent with introversion."},
    {"trait": "conscientiousness", "level": "moderate", "evidence": "Despite nervousness, maintains some structure — tension suggests effort to perform well, indicating diligence."},
    {"trait": "openness", "level": "unclear", "evidence": "Insufficient behavioral signals specific to intellectual curiosity; anxiety may mask openness cues."}
  ]
}
```

### Stage 3 — Scores
```json
{
  "openness": -0.15,
  "conscientiousness": 0.10,
  "extraversion": -0.72,
  "agreeableness": 0.05,
  "neuroticism": 0.85,
  "overall_personality": -0.23,
  "interview_score": -0.55,
  "answer_score": -0.30,
  "speaking_skills": -0.68,
  "confidence_score": -0.80,
  "facial_expression": -0.45,
  "overall_performance": -0.58
}
```"""

    FEWSHOT_EXAMPLE_B = """## Example B: Engaging, High-Energy Candidate

### Stage 1 — Behavioral Observations
```json
{
  "observations": {
    "facial": "Frequently smiles (mean=0.095). Expressive brow raises (0.22). Relaxed expression (tension=0.32). Low anxiety indicators (0.12).",
    "gaze": "Steady, direct gaze at camera (gaze deviation=0.36).",
    "head": "Moderate head movement (yaw=0.26).",
    "gesture": "Wide, open arm gestures (left_angle=68.5°, right_angle=72.1°). Steady head (jitter=0.005). Relaxed shoulders (shrug=0.003).",
    "voice": "Mid-range pitch (178Hz). Highly expressive intonation (pitch_std=62.4). Rising intonation trend (+8.5). Fluent speech, few pauses (20.3/min). Moderate voice energy."
  }
}
```

### Stage 2 — Psychological Inferences
```json
{
  "inferences": [
    {"trait": "extraversion", "level": "high", "evidence": "Frequent smiling (0.095), expressive intonation (pitch_std=62.4), wide open gestures (arm angles 68-72°), direct gaze — all hallmarks of extraversion."},
    {"trait": "confidence", "level": "high", "evidence": "Steady eye contact (deviation=0.36), relaxed expression, open posture, fluent speech (20 pauses/min) — projects self-assurance."},
    {"trait": "anxiety", "level": "low", "evidence": "Low anxiety indicators (0.12), relaxed facial tension (0.32), steady head, few speech hesitations — minimal nervousness."},
    {"trait": "openness", "level": "moderate-high", "evidence": "Expressive brow raises suggesting engagement and curiosity, varied intonation indicating cognitive flexibility."},
    {"trait": "speaking_skills", "level": "high", "evidence": "Fluent delivery with few pauses, expressive pitch variation, rising intonation maintaining listener engagement."}
  ]
}
```

### Stage 3 — Scores
```json
{
  "openness": 0.65,
  "conscientiousness": 0.40,
  "extraversion": 0.88,
  "agreeableness": 0.55,
  "neuroticism": -0.70,
  "overall_personality": 0.50,
  "interview_score": 0.75,
  "answer_score": 0.55,
  "speaking_skills": 0.82,
  "confidence_score": 0.78,
  "facial_expression": 0.70,
  "overall_performance": 0.72
}
```"""

    # ---------- Stage System Prompts ----------

    STAGE1_SYSTEM = f"""You are an expert behavioral observer trained in the Facial Action Coding System (FACS) and behavioral psychology. Your task is to observe and describe multimodal behavioral cues from an interview video.

## CRITICAL RULE
Describe ONLY directly observable physical behaviors. Do NOT infer psychological states (no "anxious", "confident", "engaged"). Use purely descriptive language: "smiles frequently", "looks away from camera", "speaks with varied pitch", "holds arms close to body".

## Behavior Categories to Observe
1. **Facial expressions**: smiling, frowning, brow raises, facial tension, anxiety indicators
2. **Eye gaze**: gaze stability, direction, deviation from camera
3. **Head movement**: turning, tilting, steadiness/tremor
4. **Gesture & posture**: arm angles, shoulder movement, torso stability, head jitter
5. **Voice prosody**: pitch level, pitch variation (monotone vs expressive), intonation trend, pause frequency, pause duration, voice energy

## Output Format
Return a JSON object with an "observations" key containing one string per category. Be specific and reference the actual values.

{FEWSHOT_EXAMPLE_A.split('### Stage 1')[1].split('### Stage 2')[0].strip() if '### Stage 1' in FEWSHOT_EXAMPLE_A else ''}"""

    STAGE2_SYSTEM = f"""You are an expert in psychological assessment and behavioral interpretation. Given detailed behavioral observations from an interview video, your task is to infer the candidate's psychological states.

## Key Principle
Every inference MUST cite specific behavioral evidence from the observations provided. Do not guess or speculate beyond what the evidence supports. If evidence is insufficient for a particular inference, state that explicitly.

## Traits to Consider
- **Anxiety/Nervousness**: gaze avoidance, vocal tremor, frequent pauses, closed posture, facial tension
- **Confidence**: steady gaze, open posture, fluent speech, relaxed expression
- **Extraversion**: frequent smiling, expressive voice, open gestures, high energy
- **Conscientiousness**: structured responses, controlled gestures, composed demeanor
- **Openness**: expressive brow movement, vocal variety, engaged posture
- **Agreeableness**: smiling, warm tone, relaxed posture

## Output Format
Return a JSON object with an "inferences" array. Each inference has "trait", "level" (high/moderate/low/unclear), and "evidence" (string citing specific observations).

{FEWSHOT_EXAMPLE_A.split('### Stage 2')[1].split('### Stage 3')[0].strip() if '### Stage 2' in FEWSHOT_EXAMPLE_A else ''}"""

    STAGE3_SYSTEM = f"""You are an expert interview assessor. Given behavioral observations AND psychological inferences, assign scores on 12 standardized dimensions.

## Scoring Guidelines
- Scores are z-score normalized (centered at 0, typical range -2 to +2)
- Base your scores on the psychological inferences, which are themselves grounded in observable behavior
- Each score should be consistent with the trait inferences (e.g., if anxiety is inferred as high, neuroticism should be positive; if confidence is high, confidence_score should be positive)
- Do NOT default to 0 — make a judgment based on the available evidence

{SCORE_FORMAT_INSTRUCTION}

{FEWSHOT_EXAMPLE_A.split('### Stage 3')[1].strip() if '### Stage 3' in FEWSHOT_EXAMPLE_A else ''}"""

    DIRECT_SYSTEM = f"""You are an expert interview assessor. Given behavioral feature measurements from an interview video, directly assign scores on 12 standardized dimensions.

{SCORE_FORMAT_INSTRUCTION}

{FEWSHOT_EXAMPLE_B.split('### Stage 3')[1].strip() if '### Stage 3' in FEWSHOT_EXAMPLE_B else ''}"""

    GENERIC_COT_SYSTEM = f"""You are an expert interview assessor. Given behavioral feature measurements from an interview video, think through your assessment step by step, then assign scores.

First, describe what you observe in the features. Then, infer what these behaviors suggest about the candidate. Finally, assign scores.

{SCORE_FORMAT_INSTRUCTION}"""

    SKIP_S1_SYSTEM = f"""You are an expert interview assessor. Given behavioral feature measurements, directly infer psychological states and assign scores. Skip the behavioral description step.

First, infer the candidate's psychological traits from the features. Then, assign scores based on your inferences.

{SCORE_FORMAT_INSTRUCTION}"""

    SKIP_S2_SYSTEM = f"""You are an expert interview assessor. Given behavioral feature measurements, first describe the observable behaviors, then directly assign scores (skip the inference step).

{SCORE_FORMAT_INSTRUCTION}"""

    def __init__(self):
        self.describer = FeatureDescriber()

    def _format_features(self, features: dict[str, float]) -> str:
        """格式化特征为 LLM 可读的描述文本。"""
        desc = FeatureDescriber.describe(features)
        lines = ["## Behavioral Feature Measurements\n"]
        lines.append(desc["summary"])
        lines.append("\n## Detailed Observations\n")
        for obs in desc["observations"]:
            lines.append(f"- {obs['description']}")
        return "\n".join(lines)

    # ---------- Public API: 5 种 Prompt 变体 ----------

    def build_full_cot(self, features: dict[str, float]) -> list[dict]:
        """返回 3 组 messages，分别对应 Stage 1/2/3。"""
        feature_text = self._format_features(features)

        s1_messages = [
            {"role": "system", "content": self.STAGE1_SYSTEM},
            {"role": "user", "content": feature_text + "\n\nDescribe the observable behaviors in this interview. Output ONLY the JSON."},
        ]

        s2_messages = [
            {"role": "system", "content": self.STAGE2_SYSTEM},
            # Stage 2 的 user message 会在拿到 Stage 1 输出后拼接
        ]

        s3_messages = [
            {"role": "system", "content": self.STAGE3_SYSTEM},
            # Stage 3 的 user message 会在拿到 Stage 1 + Stage 2 输出后拼接
        ]

        return [s1_messages, s2_messages, s3_messages]

    def build_direct(self, features: dict[str, float]) -> list[dict]:
        """Direct LLM baseline: 特征 + 直接打分，无 CoT。"""
        feature_text = self._format_features(features)
        return [{
            "role": "system",
            "content": self.DIRECT_SYSTEM,
        }, {
            "role": "user",
            "content": feature_text + "\n\nAssign scores directly. Output ONLY the JSON scores.",
        }]

    def build_generic_cot(self, features: dict[str, float]) -> list[dict]:
        """Generic CoT baseline: "Let's think step by step"。"""
        feature_text = self._format_features(features)
        return [{
            "role": "system",
            "content": self.GENERIC_COT_SYSTEM,
        }, {
            "role": "user",
            "content": feature_text + "\n\nLet's think step by step. First observe, then infer, then score. Output the complete reasoning followed by the JSON scores.",
        }]

    def build_skip_s1(self, features: dict[str, float]) -> list[dict]:
        """Ablation: 跳过 Stage 1（观察），直接从特征推断 + 打分。"""
        feature_text = self._format_features(features)
        return [{
            "role": "system",
            "content": self.SKIP_S1_SYSTEM,
        }, {
            "role": "user",
            "content": feature_text + "\n\nInfer psychological traits from these features, then assign scores. Output the inferences and JSON scores.",
        }]

    def build_skip_s2(self, features: dict[str, float]) -> list[dict]:
        """Ablation: 跳过 Stage 2（推理），直接观察 → 打分。"""
        feature_text = self._format_features(features)
        return [{
            "role": "system",
            "content": self.SKIP_S2_SYSTEM,
        }, {
            "role": "user",
            "content": feature_text + "\n\nDescribe observable behaviors, then assign scores directly. Output the observations and JSON scores.",
        }]


# ═══════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════

def build_stage2_user_message(stage1_output: str) -> list[dict]:
    """用 Stage 1 的输出构建 Stage 2 的 user message。"""
    return [{
        "role": "user",
        "content": f"## Stage 1 — Behavioral Observations\n\n{stage1_output}\n\nBased on these observations, infer the candidate's psychological states. Output ONLY the JSON.",
    }]


def build_stage3_user_message(stage1_output: str, stage2_output: str) -> list[dict]:
    """用 Stage 1 + Stage 2 输出构建 Stage 3 的 user message。"""
    return [{
        "role": "user",
        "content": f"## Stage 1 — Behavioral Observations\n\n{stage1_output}\n\n## Stage 2 — Psychological Inferences\n\n{stage2_output}\n\nBased on the observations and inferences above, assign scores on all 12 dimensions. Output ONLY the JSON scores.",
    }]


# ═══════════════════════════════════════════════════════
# 测试入口
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # 加载第一个视频的特征向量
    features_path = "results/features"
    sample = f"{features_path}/vid_0001/feature_vector.json"

    if not __import__("os").path.exists(sample):
        print(f"Sample not found: {sample}")
        print("Run aggregate_features.py first.")
        sys.exit(1)

    with open(sample, encoding="utf-8") as f:
        fv = json.load(f)

    # 重建扁平特征
    flat = {}
    for face_k, face_v in fv.get("face", {}).items():
        if isinstance(face_v, dict):
            for stat_k, stat_v in face_v.items():
                flat[f"face.{face_k}_{stat_k}"] = stat_v
    for ges_k, ges_v in fv.get("gesture", {}).items():
        if isinstance(ges_v, dict):
            for stat_k, stat_v in ges_v.items():
                flat[f"gesture.{ges_k}_{stat_k}"] = stat_v
    for voi_k, voi_v in fv.get("voice", {}).items():
        flat[f"voice.{voi_k}"] = voi_v

    builder = CoTPromptBuilder()

    # 测试 FeatureDescriber
    print("=" * 60)
    print("FEATURE DESCRIPTION")
    print("=" * 60)
    desc = FeatureDescriber.describe(flat)
    print(desc["summary"])
    print()

    # 测试 Full CoT
    print("=" * 60)
    print("FULL COT — STAGE 1 SYSTEM PROMPT (first 500 chars)")
    print("=" * 60)
    stages = builder.build_full_cot(flat)
    print(stages[0][0]["content"][:500])
    print("...")
    print()

    # 测试 Direct
    print("=" * 60)
    print("DIRECT — USER PROMPT (first 500 chars)")
    print("=" * 60)
    direct = builder.build_direct(flat)
    print(direct[1]["content"][:500])
    print("...")

    print()
    print("All prompt variants generated successfully.")
