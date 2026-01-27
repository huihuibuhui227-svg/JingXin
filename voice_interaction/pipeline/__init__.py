"""
处理管道模块

提供语音处理的完整流程管道
"""

from .voice_pipeline import VoiceProcessingPipeline
from .speech_recognition_pipeline import SpeechRecognitionPipeline
from .tts_pipeline import TTSPipeline
from .assessment_pipeline import (
    AssessmentPipeline,
    InterviewAssessmentPipeline,
    ResearchAssessmentPipeline
)

__all__ = [
    'VoiceProcessingPipeline',
    'SpeechRecognitionPipeline',
    'TTSPipeline',
    'AssessmentPipeline',
    'InterviewAssessmentPipeline',
    'ResearchAssessmentPipeline'
]
