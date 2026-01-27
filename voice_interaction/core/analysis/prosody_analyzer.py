"""
语音韵律特征分析器

负责对提取的语音特征进行分析和评估
"""

import numpy as np
from typing import Dict, Any, List, Optional


class ProsodyAnalyzer:
    """语音韵律特征分析器"""

    def __init__(self):
        """初始化分析器"""
        pass

    def analyze_pitch(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析音高特征

        参数:
            features: 包含音高特征的字典

        返回:
            分析结果字典
        """
        pitch_mean = features.get("pitch_mean", 0.0)
        pitch_std = features.get("pitch_std", 0.0)
        pitch_direction = features.get("pitch_direction", "无法判断")

        # 音高变化评估
        if pitch_std < 20:
            pitch_quality = "平缓"
            pitch_feedback = "语调平缓，可能显得不够自信"
        elif pitch_std > 40:
            pitch_quality = "丰富"
            pitch_feedback = "语调起伏大，富有表现力"
        else:
            pitch_quality = "自然"
            pitch_feedback = "语调自然，有适度变化"

        return {
            "pitch_quality": pitch_quality,
            "pitch_feedback": pitch_feedback,
            "pitch_mean": pitch_mean,
            "pitch_variation": pitch_std,
            "pitch_direction": pitch_direction
        }

    def analyze_energy(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析能量特征

        参数:
            features: 包含能量特征的字典

        返回:
            分析结果字典
        """
        energy_mean = features.get("energy_mean", 0.0)
        energy_std = features.get("energy_std", 0.0)

        # 音量评估
        if energy_mean < 0.5:
            volume_quality = "偏轻"
            volume_feedback = "声音偏轻，建议适当提高音量"
        elif energy_mean > 0.8:
            volume_quality = "洪亮"
            volume_feedback = "声音洪亮，表达清晰"
        else:
            volume_quality = "适中"
            volume_feedback = "音量适中"

        return {
            "volume_quality": volume_quality,
            "volume_feedback": volume_feedback,
            "energy_mean": energy_mean,
            "energy_variation": energy_std
        }

    def analyze_speech_fluency(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析说话流畅度

        参数:
            features: 包含流畅度特征的字典

        返回:
            分析结果字典
        """
        speech_ratio = features.get("speech_ratio", 0.0)
        pause_frequency = features.get("pause_frequency", 0.0)
        pause_duration_mean = features.get("pause_duration_mean", 0.0)

        # 流畅度评估
        if speech_ratio > 0.6:
            fluency_quality = "流畅"
            fluency_feedback = "表达流畅，停顿合理"
        elif speech_ratio > 0.3:
            fluency_quality = "较连贯"
            fluency_feedback = "表达较连贯，存在少量停顿"
        else:
            fluency_quality = "犹豫"
            fluency_feedback = "停顿较多，略显犹豫"

        # 停顿频率评估
        if pause_frequency > 10:
            pause_quality = "频繁"
            pause_feedback = "停顿较为频繁，建议加强表达的连贯性"
        elif pause_frequency > 5:
            pause_quality = "适中"
            pause_feedback = "停顿频率适中"
        else:
            pause_quality = "较少"
            pause_feedback = "停顿较少，表达连贯"

        return {
            "fluency_quality": fluency_quality,
            "fluency_feedback": fluency_feedback,
            "pause_quality": pause_quality,
            "pause_feedback": pause_feedback,
            "speech_ratio": speech_ratio,
            "pause_frequency": pause_frequency,
            "pause_duration_mean": pause_duration_mean
        }

    def analyze_all(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        综合分析所有语音特征

        参数:
            features: 包含所有特征的字典

        返回:
            综合分析结果字典
        """
        if not features:
            return {
                "is_valid": False,
                "error": "特征数据为空"
            }

        # 分析各类特征
        pitch_analysis = self.analyze_pitch(features)
        energy_analysis = self.analyze_energy(features)
        fluency_analysis = self.analyze_speech_fluency(features)

        # 综合评分
        overall_score = self._calculate_overall_score(
            pitch_analysis, energy_analysis, fluency_analysis
        )

        # 生成综合反馈
        feedback = self._generate_feedback(
            pitch_analysis, energy_analysis, fluency_analysis
        )

        return {
            "is_valid": True,
            "overall_score": overall_score,
            "feedback": feedback,
            "pitch": pitch_analysis,
            "energy": energy_analysis,
            "fluency": fluency_analysis
        }

    def _calculate_overall_score(
        self, 
        pitch_analysis: Dict[str, Any],
        energy_analysis: Dict[str, Any],
        fluency_analysis: Dict[str, Any]
    ) -> float:
        """
        计算综合评分

        参数:
            pitch_analysis: 音高分析结果
            energy_analysis: 能量分析结果
            fluency_analysis: 流畅度分析结果

        返回:
            综合评分 (0-100)
        """
        # 音高评分
        pitch_std = pitch_analysis.get("pitch_variation", 0.0)
        pitch_score = min(100, max(0, 50 + (pitch_std - 20) * 2.5))

        # 能量评分
        energy_mean = energy_analysis.get("energy_mean", 0.5)
        energy_score = min(100, max(0, energy_mean * 100))

        # 流畅度评分
        speech_ratio = fluency_analysis.get("speech_ratio", 0.0)
        fluency_score = min(100, max(0, speech_ratio * 100))

        # 加权平均
        overall = pitch_score * 0.3 + energy_score * 0.3 + fluency_score * 0.4

        return round(overall, 1)

    def _generate_feedback(
        self,
        pitch_analysis: Dict[str, Any],
        energy_analysis: Dict[str, Any],
        fluency_analysis: Dict[str, Any]
    ) -> str:
        """
        生成综合反馈

        参数:
            pitch_analysis: 音高分析结果
            energy_analysis: 能量分析结果
            fluency_analysis: 流畅度分析结果

        返回:
            综合反馈文本
        """
        feedback_parts = []

        # 音高反馈
        pitch_feedback = pitch_analysis.get("pitch_feedback", "")
        if pitch_feedback:
            feedback_parts.append(pitch_feedback)

        # 能量反馈
        energy_feedback = energy_analysis.get("volume_feedback", "")
        if energy_feedback:
            feedback_parts.append(energy_feedback)

        # 流畅度反馈
        fluency_feedback = fluency_analysis.get("fluency_feedback", "")
        if fluency_feedback:
            feedback_parts.append(fluency_feedback)

        return "；".join(feedback_parts)

    def analyze_multiple(self, features_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析多次语音特征

        参数:
            features_list: 特征列表

        返回:
            综合分析结果
        """
        if not features_list:
            return {
                "is_valid": False,
                "error": "特征列表为空"
            }

        # 分析每次语音
        individual_analyses = []
        for features in features_list:
            analysis = self.analyze_all(features)
            individual_analyses.append(analysis)

        # 计算平均特征
        avg_features = self._calculate_average_features(features_list)

        # 综合分析
        overall_analysis = self.analyze_all(avg_features)

        # 统计信息
        overall_analysis["individual_analyses"] = individual_analyses
        overall_analysis["count"] = len(features_list)

        return overall_analysis

    def _calculate_average_features(self, features_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算平均特征

        参数:
            features_list: 特征列表

        返回:
            平均特征字典
        """
        numeric_keys = [
            "pitch_mean", "pitch_std", "pitch_trend",
            "energy_mean", "energy_std",
            "speech_ratio", "duration_sec",
            "pause_duration_mean", "pause_duration_max", "pause_frequency"
        ]

        avg_features = {}
        for key in numeric_keys:
            values = [f.get(key, 0.0) for f in features_list if key in f]
            if values:
                avg_features[key] = round(float(np.mean(values)), 2)

        # 保留非数值字段（使用最后一次出现的值）
        for features in features_list:
            for key, value in features.items():
                if key not in numeric_keys and key not in avg_features:
                    avg_features[key] = value

        return avg_features
