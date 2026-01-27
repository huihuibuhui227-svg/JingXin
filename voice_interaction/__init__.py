
# Voice Interaction Module
"""
语音交互模块

提供语音识别、语音合成和面试评估功能

模块结构：
- core: 核心分析引擎（特征提取和分析）
  - feature_extraction: 特征提取
  - analysis: 情绪和语音分析
- models: 数据模型定义
- pipeline: 处理管道
  - voice_pipeline: 完整的语音处理流程
  - speech_recognition_pipeline: 语音识别流程
  - tts_pipeline: 文本转语音流程
  - assessment_pipeline: 评估流程
- api: FastAPI接口层
- utils: 工具函数
- examples: 示例代码
- config: 配置文件
"""

__version__ = "2.0.0"
__author__ = "JingXin Team"

# 新结构导出
from .core.feature_extraction.prosody_extractor import ProsodyFeatureExtractor
from .core.analysis.prosody_analyzer import ProsodyAnalyzer
from .models.voice_models import (
    AudioData,
    ProsodyFeatures,
    ProsodyAnalysisResult,
    SpeechRecognitionResult,
    QuestionAnswerPair,
    AssessmentResult,
    InterviewSession
)
from .pipeline.voice_pipeline import VoiceProcessingPipeline
from .pipeline.speech_recognition_pipeline import SpeechRecognitionPipeline
from .pipeline.tts_pipeline import TTSPipeline
from .pipeline.assessment_pipeline import (
    AssessmentPipeline,
    InterviewAssessmentPipeline,
    ResearchAssessmentPipeline
)

__all__ = [
    # 新结构
    'ProsodyFeatureExtractor',
    'ProsodyAnalyzer',
    'AudioData',
    'ProsodyFeatures',
    'ProsodyAnalysisResult',
    'SpeechRecognitionResult',
    'QuestionAnswerPair',
    'AssessmentResult',
    'InterviewSession',
    'VoiceProcessingPipeline',
    'SpeechRecognitionPipeline',
    'TTSPipeline',
    'AssessmentPipeline',
    'InterviewAssessmentPipeline',
    'ResearchAssessmentPipeline'
]
