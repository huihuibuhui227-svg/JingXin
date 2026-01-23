from dataclasses import dataclass
from typing import Dict, List
from .features import AUFeatures, TemporalStats, MicroExpressionResult

@dataclass
class EmotionResult:
    emotion_vector: Dict[str, float]
    dominant_emotion: str
    confidence: float
    composite_emotions: List[str]
    psychological_summary: str

@dataclass
class TensionResult:
    tension_score: float
    tension_level: str
    tension_sources: Dict[str, float]

@dataclass
class AnalysisFrameResult:
    session_id: str
    timestamp: float
    focus_score: float
    au_features: AUFeatures
    temporal_stats: TemporalStats
    micro_expressions: MicroExpressionResult
    emotion_result: EmotionResult
    tension_result: TensionResult

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "focus_score": self.focus_score,
            "symmetry_score": self.au_features.symmetry_score,
            **{k: round(float(v), 3) for k, v in self.au_features.__dict__.items()},
            "psychological_signals": self.tension_result.__dict__,
            "micro_expressions": self.micro_expressions.data,
            "temporal_stats": self.temporal_stats.data,
            "emotion_vector": self.emotion_result.emotion_vector,
            "dominant_emotion": self.emotion_result.dominant_emotion,
            "confidence": self.emotion_result.confidence
        }