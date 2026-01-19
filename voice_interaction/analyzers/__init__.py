
# Analyzers模块初始化
"""
语音分析器模块

提供语音识别和语音合成功能
"""

from .speech_recognizer import SpeechRecognizer
from .tts_engine import TTSEngine

__all__ = ['SpeechRecognizer', 'TTSEngine']
