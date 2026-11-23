"""Pure NumPy + ONNXRuntime wrapper around the bundled Silero VAD v6 ONNX model.

This module intentionally locates ``silero_vad.onnx`` via ``importlib.util.find_spec``
without importing ``silero_vad``, because ``silero_vad.__init__`` eagerly imports
``torch`` (which is not installed in this environment).
"""
from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
from typing import Any

import numpy as np

_PAUSE_MIN_GAP_SEC = 0.2
_SESSION_LOCK = threading.Lock()
_CACHED_SESSION: Any = None
_SESSION_LOAD_FAILED = False


def _locate_silero_onnx() -> Path:
    """Locate ``silero_vad/data/silero_vad.onnx`` without importing ``silero_vad``."""
    spec = importlib.util.find_spec("silero_vad")
    if spec is None or not spec.origin:
        raise FileNotFoundError("silero_vad package spec not found")
    onnx_path = Path(spec.origin).resolve().parent / "data" / "silero_vad.onnx"
    if not onnx_path.is_file():
        raise FileNotFoundError(f"silero_vad.onnx not found at {onnx_path}")
    return onnx_path


def _get_onnx_session() -> Any:
    """Return a cached CPU ORT InferenceSession, or raise if unavailable."""
    global _CACHED_SESSION, _SESSION_LOAD_FAILED
    if _CACHED_SESSION is not None:
        return _CACHED_SESSION
    with _SESSION_LOCK:
        if _CACHED_SESSION is not None:
            return _CACHED_SESSION
        if _SESSION_LOAD_FAILED:
            raise RuntimeError("Silero ONNX session previously failed to load")
        try:
            import onnxruntime as ort

            onnx_path = _locate_silero_onnx()
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            providers = (
                ["CPUExecutionProvider"]
                if "CPUExecutionProvider" in ort.get_available_providers()
                else None
            )
            _CACHED_SESSION = ort.InferenceSession(
                str(onnx_path),
                sess_options=opts,
                providers=providers,
            )
            return _CACHED_SESSION
        except Exception:
            _SESSION_LOAD_FAILED = True
            raise


def _reset_cached_session_for_tests() -> None:
    """Reset the cached ONNX session (for unit tests only)."""
    global _CACHED_SESSION, _SESSION_LOAD_FAILED
    with _SESSION_LOCK:
        _CACHED_SESSION = None
        _SESSION_LOAD_FAILED = False


def _prepare_audio(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Normalize input audio to 1-D float32 at 8000 Hz or 16000 Hz."""
    if audio is None:
        return np.zeros(0, dtype=np.float32), 16000
    arr = np.asarray(audio)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32), 16000
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / 32768.0
    else:
        arr = arr.astype(np.float32, copy=False)
    if arr.ndim > 1:
        # Collapse multi-channel audio along the smaller channel axis.
        if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[-1]:
            arr = np.mean(arr, axis=0)
        else:
            arr = np.mean(arr, axis=-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0).reshape(-1)

    if sr <= 0:
        raise ValueError(f"Sample rate must be positive, got {sr}")
    if sr in (8000, 16000):
        return arr, sr
    if sr > 16000 and sr % 16000 == 0:
        step = sr // 16000
        return arr[::step].astype(np.float32, copy=False), 16000

    target_sr = 16000
    target_len = max(1, int(round(len(arr) * target_sr / float(sr))))
    xp = np.linspace(0.0, 1.0, num=len(arr), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    resampled = np.interp(x_new, xp, arr).astype(np.float32)
    return resampled, target_sr


def _run_silero_onnx_probs(audio: np.ndarray, sr: int) -> list[float]:
    """Run frame-by-frame Silero ONNX inference returning per-chunk speech probabilities."""
    session = _get_onnx_session()
    window_size = 512 if sr == 16000 else 256
    context_size = 64 if sr == 16000 else 32

    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, context_size), dtype=np.float32)
    sr_arr = np.array(sr, dtype=np.int64)

    probs: list[float] = []
    n_samples = len(audio)
    for start in range(0, n_samples, window_size):
        chunk = audio[start:start + window_size]
        if len(chunk) < window_size:
            chunk = np.pad(chunk, (0, window_size - len(chunk)), mode="constant")
        x = np.concatenate([context, chunk[np.newaxis, :]], axis=1).astype(np.float32, copy=False)
        out, state = session.run(None, {"input": x, "state": state, "sr": sr_arr})
        probs.append(float(np.clip(out[0, 0], 0.0, 1.0)))
        context = x[:, -context_size:]
    return probs


def _run_rms_fallback_probs(audio: np.ndarray, sr: int) -> list[float]:
    """Compute adaptive RMS-based pseudo-probabilities when ONNX is unavailable."""
    window_size = 512 if sr == 16000 else 256
    n_samples = len(audio)
    rms_vals: list[float] = []
    for start in range(0, n_samples, window_size):
        chunk = audio[start:start + window_size]
        rms_vals.append(float(np.sqrt(np.mean(chunk * chunk))) if len(chunk) else 0.0)
    if not rms_vals:
        return []
    rms_arr = np.asarray(rms_vals, dtype=np.float64)
    noise_floor = float(np.percentile(rms_arr, 20))
    peak = float(np.percentile(rms_arr, 95))
    if peak <= 1e-6:
        return [0.0] * len(rms_vals)
    # Adaptive scale relative to noise floor and mean energy so silence stays low
    # and speech-like bursts smoothly cross the 0.5 threshold.
    ref = max(noise_floor * 2.5, float(np.mean(rms_arr)) * 0.6, 1e-4)
    probs = np.clip(rms_arr / (rms_arr + ref) * 1.25, 0.0, 1.0)
    return [float(p) for p in probs]


def _segments_from_probs(
    frame_probs: list[float],
    *,
    audio_length_samples: int,
    sr: int,
    threshold: float,
    neg_threshold: float,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
) -> list[tuple[int, int]]:
    """Run Silero hysteresis state machine and merge/filter raw sample segments."""
    window_size = 512 if sr == 16000 else 256
    min_speech_samples = sr * max(0, min_speech_duration_ms) / 1000.0
    min_silence_samples = sr * max(0, min_silence_duration_ms) / 1000.0

    triggered = False
    current_start = 0
    temp_end = 0
    raw_segments: list[tuple[int, int]] = []

    for i, prob in enumerate(frame_probs):
        cur_sample = window_size * i
        if prob >= threshold and temp_end:
            temp_end = 0
        if prob >= threshold and not triggered:
            triggered = True
            current_start = cur_sample
            continue
        if prob < neg_threshold and triggered:
            if temp_end == 0:
                temp_end = cur_sample
            if (cur_sample - temp_end) >= min_silence_samples:
                if (temp_end - current_start) >= min_speech_samples:
                    raw_segments.append((current_start, temp_end))
                triggered = False
                temp_end = 0

    if triggered:
        end_sample = temp_end if (temp_end and (audio_length_samples - temp_end) >= min_silence_samples) else audio_length_samples
        if (end_sample - current_start) >= min_speech_samples:
            raw_segments.append((current_start, min(end_sample, audio_length_samples)))

    # Defensive merge pass for adjacent segments closer than min_silence_duration_ms.
    merged: list[tuple[int, int]] = []
    for start, end in raw_segments:
        if merged and (start - merged[-1][1]) < min_silence_samples:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    return [(s, e) for s, e in merged if (e - s) >= min_speech_samples]


def detect_speech_segments(
    audio_16k: np.ndarray,
    sr: int = 16000,
    threshold: float = 0.5,
    neg_threshold: float | None = None,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 150,
) -> dict[str, Any]:
    """Detect speech segments and inter-segment pauses using Silero VAD ONNX.

    Falls back gracefully to an adaptive RMS threshold (``vad_backend="rms_fallback"``)
    if ONNX loading or inference fails.
    """
    arr, effective_sr = _prepare_audio(audio_16k, sr)
    if len(arr) == 0:
        return {
            "speech_segments": [],
            "pause_durations_sec": [],
            "speech_ratio": 0.0,
            "speech_duration_sec": 0.0,
            "duration_sec": 0.0,
            "frame_probs": [],
            "vad_backend": "silero_vad_onnx",
        }

    eff_neg = max(float(threshold) - 0.15, 0.01) if neg_threshold is None else float(neg_threshold)

    vad_backend = "silero_vad_onnx"
    try:
        frame_probs = _run_silero_onnx_probs(arr, effective_sr)
    except Exception:
        vad_backend = "rms_fallback"
        frame_probs = _run_rms_fallback_probs(arr, effective_sr)

    sample_segments = _segments_from_probs(
        frame_probs,
        audio_length_samples=len(arr),
        sr=effective_sr,
        threshold=float(threshold),
        neg_threshold=eff_neg,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
    )

    total_duration_sec = float(len(arr)) / float(effective_sr)
    speech_segments = [
        {
            "start_sec": round(s / float(effective_sr), 4),
            "end_sec": round(min(e, len(arr)) / float(effective_sr), 4),
        }
        for s, e in sample_segments
    ]
    pause_durations_sec = [
        round(speech_segments[i + 1]["start_sec"] - speech_segments[i]["end_sec"], 4)
        for i in range(len(speech_segments) - 1)
        if (speech_segments[i + 1]["start_sec"] - speech_segments[i]["end_sec"]) >= _PAUSE_MIN_GAP_SEC
    ]

    speech_duration_sec = round(
        sum(max(0.0, seg["end_sec"] - seg["start_sec"]) for seg in speech_segments),
        4,
    )
    speech_ratio = (
        round(min(1.0, max(0.0, speech_duration_sec / total_duration_sec)), 4)
        if total_duration_sec > 0.0
        else 0.0
    )

    return {
        "speech_segments": speech_segments,
        "pause_durations_sec": pause_durations_sec,
        "speech_ratio": speech_ratio,
        "speech_duration_sec": speech_duration_sec,
        "duration_sec": round(total_duration_sec, 4),
        "frame_probs": [round(float(p), 6) for p in frame_probs],
        "vad_backend": vad_backend,
    }
