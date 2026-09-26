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

        # 计算基频。★ 第 3 个返回值 `voiced_prob`(逐帧「是浊音」的概率)**此前被 `_` 丢掉** ——
        # `docs/superpowers/specs/2026-09-21-jingxin-feature-redesign-design.md:235` 的依据栏点名的就是它(«pyin 无能量门限且丢 `voiced_prob`»):
        # 它是给 pitch 族加门控的原料,也是日志列 `voiced_prob_mean` 的产出方(Task 5)。
        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio, fmin=self.fmin, fmax=self.fmax, sr=self.sample_rate
        )
        f0_voiced = f0[voiced_flag]

        # ★ 零浊音帧(全静音):`voiced_flag` 全 `False` ⟹ `f0_voiced` 是空的、`f0` 全 `nan`。
        # 这一支里**两个量必须分开处置**(plan `docs/superpowers/plans/2026-09-26-m3-0-l0-column-table.md:38`):
        #   · `voiced_prob_mean` = **0.0 是真值**(确实一个浊音帧都没有,pyin 的概率恒 0)⟹
        #     **不做特殊处理**。把它写成空等于把「量到了 0」降级成「没测到」,而 0 在这里
        #     是最强的断言(「这不是语音」)。
        #   · `pitch_mean` = **`None`(落盘成空串)** —— `f0` 全 `nan`,均值**无定义**。
        #     写 0 就是本项目一路在杀的「把没测到写成 0」:0 Hz 是个会被下游当成真值的数。
        #     ⚠️ 同分支的 `pitch_std` / `pitch_trend` / `pitch_p10` / `pitch_p90` 是**同类缺陷**
        #     (同样是 `f0_voiced` 上的统计量),本任务未改(plan:38 只点了 `pitch_mean`)。
        if len(f0_voiced) == 0:
            return {
                "pitch_mean": None,
                "pitch_std": 0.0,
                "pitch_trend": 0.0,
                "pitch_direction": "无法判断",
                "voiced_prob_mean": float(np.mean(voiced_prob)),
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

        # 分位数:与 mean/std 不同,它们**不受极端帧拖拽**,而且是"会话内归一"的原料
        # (报告的 energy/pitch 封停理由之一就是"跨会话不可比,需会话内归一")。
        p10 = float(np.percentile(f0_voiced, 10)) if len(f0_voiced) else 0.0
        p90 = float(np.percentile(f0_voiced, 90)) if len(f0_voiced) else 0.0

        # ★ 浊音概率的均值:取**全部帧**(不是"浊音帧的均值")。只在 `voiced_flag` 为真的帧上
        # 取均值会与 `voiced_flag` 的含义重合,而且全静音段会变成"无值";取全部帧的均值才同时
        # 表达「这声音有多像语音」与「有多少帧像语音」,全静音时它是 0.0 —— 一个真值。
        # 不四舍五入:概率的低端正是门控要用的地方(实测:真素材 17 段 0.0102~0.0514,
        # 合成正弦段 0.6245,白噪声 0.0100 —— 四位小数才分得开噪声底与真回答)。
        voiced_prob_mean = float(np.mean(voiced_prob))
        return {
            "pitch_mean": round(pitch_mean, 2),
            "pitch_std": round(pitch_std, 2),
            "pitch_trend": round(pitch_trend, 2),
            "pitch_direction": pitch_direction,
            "pitch_p10": round(p10, 2),
            "pitch_p90": round(p90, 2),
            "voiced_prob_mean": voiced_prob_mean
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

        e10 = float(np.percentile(rms, 10))
        e90 = float(np.percentile(rms, 90))
        return {
            "energy_mean": round(energy_mean, 4),
            "energy_std": round(energy_std, 4),
            "energy_p10": round(e10, 4),
            "energy_p90": round(e90, 4)
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
                "pause_frequency": 0.0,
                "speech_duration_sec": 0.0,
                "speech_onset_sec": None
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

        # ★ **有声时长**(秒):同一个能量门算出来的,此前只在函数内部用过就丢了。
        # 它是 `speech_ratio` 的绝对量版本 —— 而 `speech_ratio` 是**自指阈值**
        # (它的封停理由:87.9% 恰为 1.0),绝对秒数没有这个问题。
        # 也是"语速"的分母(字数 ÷ 有声秒)。
        frame_sec = duration / len(rms) if len(rms) else 0.0
        speech_seconds = round(float(np.count_nonzero(is_speech)) * frame_sec, 2)
        # ★ **首次开口在音频里的位置**(秒):反应延迟要用它。
        #   "录音开始"到"真的开口"之间那段前导静音,不是反应时间 —— 但推首次开口
        #   的**墙钟**必须把它算进去,否则整段前导静音都会被算成反应时间。
        first_voiced = next((i for i, v in enumerate(is_speech) if v), None)
        speech_onset = round(first_voiced * frame_sec, 2) if first_voiced is not None else None
        return {
            "pause_duration_mean": pause_duration_mean,
            "pause_duration_max": pause_duration_max,
            "pause_frequency": pause_frequency,
            "speech_duration_sec": speech_seconds,
            "speech_onset_sec": speech_onset
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
