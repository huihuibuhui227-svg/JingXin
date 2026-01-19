
# Voice Interaction Module
"""
语音交互模块

提供语音识别、语音合成和面试评估功能
"""

__version__ = "1.0.0"
__author__ = "JingXin Team"

from .analyzers import SpeechRecognizer, TTSEngine
from .assessment import InterviewAssessment, ResearchAssessment

__all__ = [
    'SpeechRecognizer',
    'TTSEngine',
    'InterviewAssessment',
    'ResearchAssessment'
]
