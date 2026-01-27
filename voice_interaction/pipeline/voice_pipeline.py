"""
语音处理管道

提供从音频输入到分析输出的完整处理流程
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from ..core.feature_extraction.prosody_extractor import ProsodyFeatureExtractor
from ..core.analysis.prosody_analyzer import ProsodyAnalyzer
from ..models.voice_models import (
    AudioData, 
    ProsodyFeatures, 
    ProsodyAnalysisResult,
    SpeechRecognitionResult
)


class VoiceProcessingPipeline:
    """语音处理管道"""

    def __init__(self, sample_rate: int = 16000):
        """
        初始化处理管道

        参数:
            sample_rate: 音频采样率
        """
        self.sample_rate = sample_rate
        self.feature_extractor = ProsodyFeatureExtractor(sample_rate)
        self.analyzer = ProsodyAnalyzer()

    def process_audio(
        self, 
        audio_data: np.ndarray,
        extract_features: bool = True,
        analyze: bool = True
    ) -> Dict[str, Any]:
        """
        处理音频数据

        参数:
            audio_data: 音频数据
            extract_features: 是否提取特征
            analyze: 是否进行分析

        返回:
            处理结果字典
        """
        result = {
            "audio": AudioData(audio_data, self.sample_rate),
            "features": None,
            "analysis": None
        }

        # 提取特征
        if extract_features:
            features_dict = self.feature_extractor.extract_all_features(audio_data)
            result["features"] = ProsodyFeatures.from_dict(features_dict)

        # 分析特征
        if analyze and result["features"]:
            analysis_dict = self.analyzer.analyze_all(features_dict)
            result["analysis"] = self._convert_to_analysis_result(analysis_dict)

        return result

    def extract_features_only(self, audio_data: np.ndarray) -> ProsodyFeatures:
        """
        仅提取特征

        参数:
            audio_data: 音频数据

        返回:
            语音特征对象
        """
        features_dict = self.feature_extractor.extract_all_features(audio_data)
        return ProsodyFeatures.from_dict(features_dict)

    def analyze_features_only(self, features: ProsodyFeatures) -> ProsodyAnalysisResult:
        """
        仅分析特征

        参数:
            features: 语音特征对象

        返回:
            分析结果对象
        """
        features_dict = features.to_dict()
        analysis_dict = self.analyzer.analyze_all(features_dict)
        return self._convert_to_analysis_result(analysis_dict)

    def process_recognition_result(
        self,
        recognition_result: SpeechRecognitionResult,
        analyze_prosody: bool = True
    ) -> Dict[str, Any]:
        """
        处理语音识别结果

        参数:
            recognition_result: 语音识别结果
            analyze_prosody: 是否分析语音特征

        返回:
            处理结果字典
        """
        result = {
            "recognition": recognition_result,
            "features": None,
            "analysis": None
        }

        # 如果有音频数据且需要分析
        if analyze_prosody and recognition_result.has_audio:
            audio_data = recognition_result.audio_data.data
            features = self.extract_features_only(audio_data)
            result["features"] = features

            analysis = self.analyze_features_only(features)
            result["analysis"] = analysis

        return result

    def process_multiple_audio(
        self,
        audio_list: list,
        extract_features: bool = True,
        analyze: bool = True
    ) -> Dict[str, Any]:
        """
        处理多个音频

        参数:
            audio_list: 音频数据列表
            extract_features: 是否提取特征
            analyze: 是否进行分析

        返回:
            处理结果字典
        """
        individual_results = []
        features_list = []

        # 处理每个音频
        for audio_data in audio_list:
            result = self.process_audio(audio_data, extract_features, analyze)
            individual_results.append(result)
            if result["features"]:
                features_list.append(result["features"].to_dict())

        # 综合分析
        overall_analysis = None
        if analyze and features_list:
            analysis_dict = self.analyzer.analyze_multiple(features_list)
            overall_analysis = self._convert_to_analysis_result(analysis_dict)

        return {
            "individual_results": individual_results,
            "overall_analysis": overall_analysis,
            "count": len(audio_list)
        }

    def _convert_to_analysis_result(self, analysis_dict: Dict[str, Any]) -> ProsodyAnalysisResult:
        """
        将分析字典转换为分析结果对象

        参数:
            analysis_dict: 分析字典

        返回:
            分析结果对象
        """
        pitch = analysis_dict.get("pitch", {})
        energy = analysis_dict.get("energy", {})
        fluency = analysis_dict.get("fluency", {})

        return ProsodyAnalysisResult(
            is_valid=analysis_dict.get("is_valid", False),
            overall_score=analysis_dict.get("overall_score", 0.0),
            feedback=analysis_dict.get("feedback", ""),
            pitch_quality=pitch.get("pitch_quality", ""),
            pitch_feedback=pitch.get("pitch_feedback", ""),
            pitch_mean=pitch.get("pitch_mean", 0.0),
            pitch_variation=pitch.get("pitch_variation", 0.0),
            pitch_direction=pitch.get("pitch_direction", ""),
            volume_quality=energy.get("volume_quality", ""),
            volume_feedback=energy.get("volume_feedback", ""),
            energy_mean=energy.get("energy_mean", 0.0),
            energy_variation=energy.get("energy_variation", 0.0),
            fluency_quality=fluency.get("fluency_quality", ""),
            fluency_feedback=fluency.get("fluency_feedback", ""),
            pause_quality=fluency.get("pause_quality", ""),
            pause_feedback=fluency.get("pause_feedback", ""),
            speech_ratio=fluency.get("speech_ratio", 0.0),
            pause_frequency=fluency.get("pause_frequency", 0.0),
            pause_duration_mean=fluency.get("pause_duration_mean", 0.0)
        )
