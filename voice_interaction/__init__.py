
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

导出是【惰性】的（PEP 562，见 Ruling M1-1）：父包总在任何子模块之前被导入，
所以这里 eager import 会让 `import voice_interaction.asr.session` 顺带构造
TTS / vosk / librosa 管线 —— 既慢，又可能因为与被测代码无关的原因失败。
`from voice_interaction import X` 的写法与 `__all__` 保持不变。
"""

__version__ = "2.0.0"
__author__ = "JingXin Team"

# 名字 -> (子模块, 模块内属性名)。真正的 import 推迟到属性被访问时。
_LAZY_EXPORTS = {
    "ProsodyFeatureExtractor": ("voice_interaction.core.feature_extraction.prosody_extractor",
                                "ProsodyFeatureExtractor"),
    "ProsodyAnalyzer": ("voice_interaction.core.analysis.prosody_analyzer", "ProsodyAnalyzer"),
    "AudioData": ("voice_interaction.models.voice_models", "AudioData"),
    "ProsodyFeatures": ("voice_interaction.models.voice_models", "ProsodyFeatures"),
    "ProsodyAnalysisResult": ("voice_interaction.models.voice_models", "ProsodyAnalysisResult"),
    "SpeechRecognitionResult": ("voice_interaction.models.voice_models", "SpeechRecognitionResult"),
    "QuestionAnswerPair": ("voice_interaction.models.voice_models", "QuestionAnswerPair"),
    "AssessmentResult": ("voice_interaction.models.voice_models", "AssessmentResult"),
    "InterviewSession": ("voice_interaction.models.voice_models", "InterviewSession"),
    "VoiceProcessingPipeline": ("voice_interaction.pipeline.voice_pipeline", "VoiceProcessingPipeline"),
    "SpeechRecognitionPipeline": ("voice_interaction.pipeline.speech_recognition_pipeline",
                                  "SpeechRecognitionPipeline"),
    "TTSPipeline": ("voice_interaction.pipeline.tts_pipeline", "TTSPipeline"),
    "AssessmentPipeline": ("voice_interaction.pipeline.assessment_pipeline", "AssessmentPipeline"),
    "InterviewAssessmentPipeline": ("voice_interaction.pipeline.assessment_pipeline",
                                    "InterviewAssessmentPipeline"),
    "ResearchAssessmentPipeline": ("voice_interaction.pipeline.assessment_pipeline",
                                   "ResearchAssessmentPipeline"),
}

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


def __getattr__(name: str):
    """PEP 562:首次访问时才导入对应子模块,并缓存到 globals()。"""
    try:
        module_path, attr = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module_path), attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
