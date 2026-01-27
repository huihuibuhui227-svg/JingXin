"""
语音韵律特征提取器

负责从原始音频中提取音高、能量、语速等基础特征
"""

import numpy as np
import librosa
from typing import Dict, Any


class ProsodyFeatureExtractor:
    """语音韵律特征提取器"""

    def __init__(self, sample_rate: int = 16000):
        """
        初始化特征提取器

        参数:
            sample_rate: 音频采样率
        """
        self.sample_rate = sample_rate
        self.fmin = librosa.note_to_hz('C2')
        self.fmax = librosa.note_to_hz('C7')

    def extract_pitch_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        提取音高相关特征

        参数:
            audio: 音频数据

        返回:
            包含音高特征的字典
        """
        if len(audio) == 0:
            return {
                "pitch_mean": 0.0,
                "pitch_std": 0.0,
                "pitch_trend": 0.0,
                "pitch_direction": "无法判断"
            }

        # 计算基频
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=self.fmin, fmax=self.fmax, sr=self.sample_rate
        )
        f0_voiced = f0[voiced_flag]

        if len(f0_voiced) == 0:
            return {
                "pitch_mean": 0.0,
                "pitch_std": 0.0,
                "pitch_trend": 0.0,
                "pitch_direction": "无法判断"
            }

        pitch_mean = float(np.mean(f0_voiced))
        pitch_std = float(np.std(f0_voiced))

        # 分析语调趋势
        if len(f0_voiced) > 1:
            n = len(f0_voiced)
            first_third = np.mean(f0_voiced[:n//3])
            last_third = np.mean(f0_voiced[-n//3:])
            pitch_trend = last_third - first_third

            # 判断语调趋势
            if pitch_trend > 10:
                pitch_direction = "上扬"
            elif pitch_trend < -10:
                pitch_direction = "下降"
            else:
                pitch_direction = "平稳"
        else:
            pitch_trend = 0.0
            pitch_direction = "无法判断"

        return {
            "pitch_mean": round(pitch_mean, 2),
            "pitch_std": round(pitch_std, 2),
            "pitch_trend": round(pitch_trend, 2),
            "pitch_direction": pitch_direction
        }

    def extract_energy_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        提取能量相关特征

        参数:
            audio: 音频数据

        返回:
            包含能量特征的字典
        """
        if len(audio) == 0:
            return {
                "energy_mean": 0.0,
                "energy_std": 0.0
            }

        rms = librosa.feature.rms(y=audio)[0]
        energy_mean = float(np.mean(rms))
        energy_std = float(np.std(rms))

        return {
            "energy_mean": round(energy_mean, 4),
            "energy_std": round(energy_std, 4)
        }

    def extract_speech_ratio(self, audio: np.ndarray) -> float:
        """
        计算说话时间占比（语速/流畅度）

        参数:
            audio: 音频数据

        返回:
            说话时间占比 (0-1)
        """
        if len(audio) == 0:
            return 0.0

        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        non_silent = np.sum(spectral_centroids > np.mean(spectral_centroids) * 0.1)
        speech_ratio = float(non_silent / len(spectral_centroids)) if len(spectral_centroids) > 0 else 0.0

        return round(speech_ratio, 2)

    def extract_pause_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        提取停顿相关特征

        参数:
            audio: 音频数据

        返回:
            包含停顿特征的字典
        """
        if len(audio) == 0:
            return {
                "pause_duration_mean": 0.0,
                "pause_duration_max": 0.0,
                "pause_frequency": 0.0
            }

        rms = librosa.feature.rms(y=audio)[0]
        duration = len(audio) / self.sample_rate

        # 使用能量阈值检测停顿
        energy_threshold = np.mean(rms) * 0.3
        is_speech = rms > energy_threshold

        # 找到所有停顿区间
        pause_intervals = []
        in_pause = False
        pause_start = 0

        for i, speech in enumerate(is_speech):
            if not speech and not in_pause:
                in_pause = True
                pause_start = i
            elif speech and in_pause:
                in_pause = False
                pause_duration = (i - pause_start) * (duration / len(rms))
                if pause_duration > 0.1:
                    pause_intervals.append(pause_duration)

        # 计算停顿统计
        if len(pause_intervals) > 0:
            pause_duration_mean = round(float(np.mean(pause_intervals)), 2)
            pause_duration_max = round(float(np.max(pause_intervals)), 2)
            pause_frequency = round(len(pause_intervals) / duration * 60, 2)
        else:
            pause_duration_mean = 0.0
            pause_duration_max = 0.0
            pause_frequency = 0.0

        return {
            "pause_duration_mean": pause_duration_mean,
            "pause_duration_max": pause_duration_max,
            "pause_frequency": pause_frequency
        }

    def extract_all_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        提取所有韵律特征

        参数:
            audio: 音频数据

        返回:
            包含所有特征的字典
        """
        if len(audio) == 0:
            return {}

        # 提取各类特征
        pitch_features = self.extract_pitch_features(audio)
        energy_features = self.extract_energy_features(audio)
        speech_ratio = self.extract_speech_ratio(audio)
        pause_features = self.extract_pause_features(audio)
        duration = len(audio) / self.sample_rate

        # 合并所有特征
        all_features = {
            **pitch_features,
            **energy_features,
            "speech_ratio": speech_ratio,
            "duration_sec": round(duration, 2),
            **pause_features
        }

        return all_features
