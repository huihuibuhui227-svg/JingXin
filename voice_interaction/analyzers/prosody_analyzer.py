# voice_interaction/analyzers/prosody_analyzer.py
"""
语音语调分析器
"""

import numpy as np
import librosa

def analyze_prosody(audio: np.ndarray, sr: int = 16000) -> dict:
    if len(audio) == 0:
        return {}

    # 1. 音高（基频）
    f0, voiced_flag, _ = librosa.pyin(
        audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
    )
    f0_voiced = f0[voiced_flag]
    pitch_mean = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
    pitch_std = float(np.std(f0_voiced)) if len(f0_voiced) > 0 else 0.0

    # 2. 能量
    rms = librosa.feature.rms(y=audio)[0]
    energy_mean = float(np.mean(rms))
    energy_std = float(np.std(rms))

    # 3. 说话时间占比（语速/流畅度）
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    non_silent = np.sum(spectral_centroids > np.mean(spectral_centroids) * 0.1)
    speech_ratio = float(non_silent / len(spectral_centroids)) if len(spectral_centroids) > 0 else 0.0

    # 4. 总时长
    duration = len(audio) / sr

    return {
        "pitch_mean": round(pitch_mean, 2),
        "pitch_variation": round(pitch_std, 2),
        "energy_mean": round(energy_mean, 4),
        "energy_variation": round(energy_std, 4),
        "speech_ratio": round(speech_ratio, 2),
        "duration_sec": round(duration, 2)
    }