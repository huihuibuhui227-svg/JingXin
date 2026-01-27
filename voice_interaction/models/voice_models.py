"""
语音交互数据模型

定义语音识别、语音合成和评估相关的数据模型
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np


@dataclass
class AudioData:
    """音频数据模型"""
    data: np.ndarray
    sample_rate: int = 16000
    duration: float = 0.0
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def __post_init__(self):
        """初始化后处理"""
        if self.duration == 0.0 and len(self.data) > 0:
            self.duration = len(self.data) / self.sample_rate

    @property
    def is_empty(self) -> bool:
        """检查音频是否为空"""
        return len(self.data) == 0

    @property
    def length_seconds(self) -> float:
        """获取音频长度（秒）"""
        return self.duration


@dataclass
class ProsodyFeatures:
    """语音韵律特征模型"""
    # 音高特征
    pitch_mean: float = 0.0
    pitch_std: float = 0.0
    pitch_trend: float = 0.0
    pitch_direction: str = "无法判断"

    # 能量特征
    energy_mean: float = 0.0
    energy_std: float = 0.0

    # 流畅度特征
    speech_ratio: float = 0.0
    duration_sec: float = 0.0

    # 停顿特征
    pause_duration_mean: float = 0.0
    pause_duration_max: float = 0.0
    pause_frequency: float = 0.0

    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "pitch_mean": self.pitch_mean,
            "pitch_std": self.pitch_std,
            "pitch_trend": self.pitch_trend,
            "pitch_direction": self.pitch_direction,
            "energy_mean": self.energy_mean,
            "energy_std": self.energy_std,
            "speech_ratio": self.speech_ratio,
            "duration_sec": self.duration_sec,
            "pause_duration_mean": self.pause_duration_mean,
            "pause_duration_max": self.pause_duration_max,
            "pause_frequency": self.pause_frequency,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProsodyFeatures':
        """从字典创建实例"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SpeechRecognitionResult:
    """语音识别结果模型"""
    text: str = ""
    confidence: float = 0.0
    is_final: bool = False
    audio_data: Optional[AudioData] = None
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    @property
    def is_valid(self) -> bool:
        """检查结果是否有效"""
        return len(self.text.strip()) > 0

    @property
    def has_audio(self) -> bool:
        """检查是否有关联的音频数据"""
        return self.audio_data is not None and not self.audio_data.is_empty


@dataclass
class ProsodyAnalysisResult:
    """语音韵律分析结果模型"""
    is_valid: bool = False
    overall_score: float = 0.0
    feedback: str = ""

    # 音高分析
    pitch_quality: str = ""
    pitch_feedback: str = ""
    pitch_mean: float = 0.0
    pitch_variation: float = 0.0
    pitch_direction: str = ""

    # 能量分析
    volume_quality: str = ""
    volume_feedback: str = ""
    energy_mean: float = 0.0
    energy_variation: float = 0.0

    # 流畅度分析
    fluency_quality: str = ""
    fluency_feedback: str = ""
    pause_quality: str = ""
    pause_feedback: str = ""
    speech_ratio: float = 0.0
    pause_frequency: float = 0.0
    pause_duration_mean: float = 0.0

    # 元数据
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "is_valid": self.is_valid,
            "overall_score": self.overall_score,
            "feedback": self.feedback,
            "pitch": {
                "quality": self.pitch_quality,
                "feedback": self.pitch_feedback,
                "mean": self.pitch_mean,
                "variation": self.pitch_variation,
                "direction": self.pitch_direction
            },
            "energy": {
                "quality": self.volume_quality,
                "feedback": self.volume_feedback,
                "mean": self.energy_mean,
                "variation": self.energy_variation
            },
            "fluency": {
                "quality": self.fluency_quality,
                "feedback": self.fluency_feedback,
                "pause_quality": self.pause_quality,
                "pause_feedback": self.pause_feedback,
                "speech_ratio": self.speech_ratio,
                "pause_frequency": self.pause_frequency,
                "pause_duration_mean": self.pause_duration_mean
            },
            "timestamp": self.timestamp
        }


@dataclass
class QuestionAnswerPair:
    """问答对模型"""
    question: str = ""
    answer: str = ""
    prosody_features: Optional[ProsodyFeatures] = None
    prosody_analysis: Optional[ProsodyAnalysisResult] = None
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    @property
    def has_valid_answer(self) -> bool:
        """检查是否有有效回答"""
        return len(self.answer.strip()) > 0 and self.answer != "[无有效回答]"

    @property
    def has_prosody_data(self) -> bool:
        """检查是否有语音特征数据"""
        return self.prosody_features is not None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "question": self.question,
            "answer": self.answer,
            "timestamp": self.timestamp
        }

        if self.prosody_features:
            result["prosody_features"] = self.prosody_features.to_dict()

        if self.prosody_analysis:
            result["prosody_analysis"] = self.prosody_analysis.to_dict()

        return result


@dataclass
class AssessmentResult:
    """评估结果模型"""
    text: str = ""
    is_valid: bool = True
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "text": self.text,
            "is_valid": self.is_valid,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class InterviewSession:
    """面试会话模型"""
    session_id: str = ""
    qa_pairs: List[QuestionAnswerPair] = field(default_factory=list)
    start_time: float = field(default_factory=lambda: datetime.now().timestamp())
    end_time: Optional[float] = None

    def add_qa_pair(self, question: str, answer: str, 
                   prosody_features: Optional[ProsodyFeatures] = None,
                   prosody_analysis: Optional[ProsodyAnalysisResult] = None) -> None:
        """添加问答对"""
        qa_pair = QuestionAnswerPair(
            question=question,
            answer=answer,
            prosody_features=prosody_features,
            prosody_analysis=prosody_analysis
        )
        self.qa_pairs.append(qa_pair)

    @property
    def duration(self) -> float:
        """获取会话持续时间"""
        end = self.end_time if self.end_time else datetime.now().timestamp()
        return end - self.start_time

    @property
    def question_count(self) -> int:
        """获取问题数量"""
        return len(self.qa_pairs)

    @property
    def valid_answer_count(self) -> int:
        """获取有效回答数量"""
        return sum(1 for qa in self.qa_pairs if qa.has_valid_answer)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "question_count": self.question_count,
            "valid_answer_count": self.valid_answer_count,
            "qa_pairs": [qa.to_dict() for qa in self.qa_pairs]
        }
