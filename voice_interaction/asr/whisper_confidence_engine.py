"""Aligned ASR word-confidence extractor powered by ``faster-whisper``.

This engine complements the live FunASR recognizer (which does not return
word-level ASR confidence probabilities) by producing same-engine aligned
word probabilities for offline auditing and backfill.
"""
from __future__ import annotations

import math
import os
from typing import Any, Iterable

import numpy as np

DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


def resolve_compute_type(device: str, compute_type: str | None) -> str:
    """Enforce hardware safety rules for CTranslate2 / faster-whisper.

    CUDA on this host does not support ``float16``. Whenever ``device`` targets
    CUDA, ``compute_type`` is forced to ``"float32"``. On CPU the default is
    ``"int8"``.
    """
    norm_device = (device or "cpu").strip().lower()
    norm_ct = (compute_type or "").strip().lower()
    if norm_device == "cuda" or norm_device.startswith("cuda:"):
        return "float32"
    if norm_device == "auto" and "float16" in norm_ct:
        return "float32"
    if norm_ct:
        return compute_type  # preserve caller's explicit non-CUDA type
    return "int8"


def summarize_word_probs(probs: Iterable[float]) -> dict[str, Any]:
    """Aggregate per-word probabilities into summary statistics."""
    clean = [float(p) for p in probs if p is not None and math.isfinite(float(p))]
    if not clean:
        return {
            "median": None,
            "p20": None,
            "low_word_ratio_lt_0_5": None,
            "n_words": 0,
        }
    arr = np.asarray(clean, dtype=np.float64)
    return {
        "median": round(float(np.median(arr)), 4),
        "p20": round(float(np.percentile(arr, 20)), 4),
        "low_word_ratio_lt_0_5": round(float(np.mean(arr < 0.5)), 4),
        "n_words": int(len(arr)),
    }


def _pcm16_to_float32(pcm: bytes) -> np.ndarray:
    """Convert 16 kHz / 16-bit / mono PCM bytes into float32 in [-1.0, 1.0]."""
    if not pcm or len(pcm) < 2:
        return np.zeros(0, dtype=np.float32)
    even_len = len(pcm) - (len(pcm) % 2)
    arr = np.frombuffer(pcm[:even_len], dtype=np.int16)
    return (arr.astype(np.float32) / 32768.0).reshape(-1)


class FasterWhisperConfidenceEngine:
    """Wrapper around ``faster_whisper.WhisperModel`` for word-level confidence."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        model: Any = None,
        **model_kwargs: Any,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = resolve_compute_type(device, compute_type)
        self._model_kwargs = dict(model_kwargs)
        self._model = model

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            **self._model_kwargs,
        )
        return self._model

    def analyze_audio(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        audio_scope: str = "qa_segment",
        return_probs: bool = False,
    ) -> dict[str, Any]:
        """Run faster-whisper with word timestamps on a mono float waveform."""
        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(arr) == 0:
            res: dict[str, Any] = {
                "asr_confidence_source": "word_prob_aligned_same_engine",
                "engine": "faster-whisper",
                "model_size": self.model_size,
                "audio_scope": audio_scope,
                "word_conf_summary": summarize_word_probs([]),
                "text": "",
            }
            if return_probs:
                res["word_probs"] = []
            return res

        if sr != 16000 and sr > 0:
            target_len = max(1, int(round(len(arr) * 16000 / float(sr))))
            xp = np.linspace(0.0, 1.0, num=len(arr), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
            arr = np.interp(x_new, xp, arr).astype(np.float32)

        model = self._get_model()
        segments_iter, _info = model.transcribe(
            arr,
            language="zh",
            word_timestamps=True,
        )

        texts: list[str] = []
        probs: list[float] = []
        for seg in segments_iter:
            seg_text = getattr(seg, "text", "") or ""
            if seg_text:
                texts.append(seg_text)
            for word in getattr(seg, "words", None) or []:
                prob = getattr(word, "probability", None)
                if prob is not None and math.isfinite(float(prob)):
                    probs.append(float(prob))

        res = {
            "asr_confidence_source": "word_prob_aligned_same_engine",
            "engine": "faster-whisper",
            "model_size": self.model_size,
            "audio_scope": audio_scope,
            "word_conf_summary": summarize_word_probs(probs),
            "text": "".join(texts).strip(),
        }
        if return_probs:
            res["word_probs"] = probs
        return res

    def analyze_pcm(
        self,
        pcm: bytes,
        audio_scope: str = "qa_segment",
        return_probs: bool = False,
    ) -> dict[str, Any]:
        """Run faster-whisper word confidence extraction on 16kHz/16bit/mono PCM."""
        audio = _pcm16_to_float32(pcm)
        return self.analyze_audio(
            audio, sr=16000, audio_scope=audio_scope, return_probs=return_probs
        )
