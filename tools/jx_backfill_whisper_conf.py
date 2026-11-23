"""Generate aligned faster-whisper word-confidence sidecars and run P0 acoustic benchmarks.

Outputs sidecar files ``data/transcripts/<sid>.whisper_conf.json`` without
modifying or overwriting original ``<sid>.json`` transcripts and without writing
any participant PII. Also benchmarks Parselmouth vs ``librosa.pyin`` + Silero VAD
on local recordings.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_OUT_DIR = REPO / "data" / "transcripts"


def session_id_of(sess_dir: Path) -> str:
    """Resolve canonical session_id from directory name without exposing PII."""
    lp = sess_dir / "label.json"
    if lp.is_file():
        try:
            sid = json.loads(lp.read_text(encoding="utf-8")).get("session_id")
            if sid:
                return str(sid)
        except Exception:
            pass
    return sess_dir.name.split("__")[-1] if "__" in sess_dir.name else sess_dir.name


def extract_pcm_16k(media_path: Path, max_sec: float | None = None) -> tuple[bytes, float]:
    """Decode WAV or WebM media to 16 kHz / 16-bit / mono PCM bytes."""
    if media_path.suffix.lower() == ".wav" and max_sec is None:
        try:
            with wave.open(str(media_path), "rb") as w:
                if w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == 16000:
                    n_frames = w.getnframes()
                    return w.readframes(n_frames), n_frames / 16000.0
        except wave.Error:
            pass

    with tempfile.TemporaryDirectory() as td:
        out_wav = Path(td) / "pcm16k.wav"
        cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(media_path)]
        if max_sec is not None and max_sec > 0:
            cmd.extend(["-t", f"{float(max_sec):.3f}"])
        cmd.extend(["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out_wav)])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg decode failed ({proc.returncode}): {proc.stderr.strip()[:300]}"
            )
        with wave.open(str(out_wav), "rb") as w:
            n_frames = w.getnframes()
            return w.readframes(n_frames), n_frames / float(w.getframerate())


def backfill_session(
    session_id: str,
    engine: Any,
    out_dir: Path = DEFAULT_OUT_DIR,
    force: bool = False,
) -> dict[str, Any]:
    """Generate ``<sid>.whisper_conf.json`` sidecar for one session (never touches ``<sid>.json``)."""
    from report_frontend import voice_series
    from voice_interaction.asr.whisper_confidence_engine import summarize_word_probs

    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar = out_dir / f"{session_id}.whisper_conf.json"
    if sidecar.is_file() and not force:
        return {"session_id": session_id, "status": "skipped", "path": str(sidecar)}

    kind, items = voice_series.audio_source(session_id)
    if kind == "none" or not items:
        return {"session_id": session_id, "status": "no_audio"}

    audio_scope = "qa_segment" if kind == "answer" else "whole_session"
    all_probs: list[float] = []
    segment_records: list[dict[str, Any]] = []
    total_audio_sec = 0.0
    t0 = time.perf_counter()

    for idx, audio_path in items:
        pcm, dur_sec = extract_pcm_16k(Path(audio_path))
        total_audio_sec += dur_sec
        res = engine.analyze_pcm(pcm, audio_scope=audio_scope, return_probs=True)
        probs = res.pop("word_probs", [])
        all_probs.extend(probs)
        segment_records.append({
            "index": idx,
            "audio_scope": audio_scope,
            "duration_sec": round(dur_sec, 2),
            "word_conf_summary": res["word_conf_summary"],
            "n_chars": len((res.get("text") or "").strip()),
        })

    payload = {
        "session_id": session_id,
        "asr_confidence_source": "word_prob_aligned_same_engine",
        "engine": "faster-whisper",
        "model_size": engine.model_size,
        "device": engine.device,
        "compute_type": engine.compute_type,
        "audio_scope": audio_scope,
        "n_audio_files": len(items),
        "audio_duration_sec": round(total_audio_sec, 2),
        "word_conf_summary": summarize_word_probs(all_probs),
        "segments": segment_records,
    }
    tmp = out_dir / f".{session_id}.whisper_conf.json.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, sidecar)

    return {
        "session_id": session_id,
        "status": "ok",
        "path": str(sidecar),
        "audio_scope": audio_scope,
        "n_audio_files": len(items),
        "word_conf_summary": payload["word_conf_summary"],
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }


def find_default_benchmark_audio() -> Path | None:
    """Locate a local recording audio file (preferring ``20261026_002930*``)."""
    import media_retention

    root = media_retention.root()
    if not root.is_dir():
        return None
    dirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("_"))
    prioritized = [d for d in dirs if d.name.startswith("20261026_002930")] + dirs
    for d in prioritized:
        aud = d / "media" / "audio"
        if aud.is_dir():
            wavs = sorted(aud.glob("*_converted.wav"))
            if wavs:
                return wavs[0]
        cam = d / "media" / "camera.webm"
        if cam.is_file():
            return cam
    return None


def run_benchmark(audio_path: Path, max_sec: float = 30.0) -> dict[str, Any]:
    """Benchmark Parselmouth vs librosa.pyin F0 extraction + Silero VAD on local audio."""
    from report_frontend import voice_series
    from voice_interaction.core.feature_extraction import silero_vad_wrapper

    pcm, dur_sec = extract_pcm_16k(audio_path, max_sec=max_sec)
    with tempfile.TemporaryDirectory() as td:
        tmp_wav = Path(td) / "bench_16k.wav"
        with wave.open(str(tmp_wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm)

        t0 = time.perf_counter()
        s_lib = voice_series.extract_file(tmp_wav, backend="librosa")
        t_lib = time.perf_counter() - t0

        t0 = time.perf_counter()
        s_pm = voice_series.extract_file(tmp_wav, backend="parselmouth")
        t_pm = time.perf_counter() - t0

    y = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    t0 = time.perf_counter()
    vad = silero_vad_wrapper.detect_speech_segments(y, sr=16000)
    t_vad = time.perf_counter() - t0

    n = min(len(s_lib.f0), len(s_pm.f0))
    f0_l = s_lib.f0[:n]
    f0_p = s_pm.f0[:n]
    mask = np.isfinite(f0_l) & np.isfinite(f0_p)
    n_mutual = int(mask.sum())
    pearson_r: float | None = None
    med_abs_diff: float | None = None
    if n_mutual >= 3:
        a, b = f0_l[mask], f0_p[mask]
        if float(np.std(a)) > 0 and float(np.std(b)) > 0:
            pearson_r = round(float(np.corrcoef(a, b)[0, 1]), 4)
        med_abs_diff = round(float(np.median(np.abs(a - b))), 2)

    return {
        "duration_sec": round(dur_sec, 2),
        "librosa_sec": round(t_lib, 4),
        "parselmouth_sec": round(t_pm, 4),
        "speedup": round(t_lib / max(t_pm, 1e-6), 2),
        "n_frames_librosa": int(len(s_lib.f0)),
        "n_frames_parselmouth": int(len(s_pm.f0)),
        "n_mutual_voiced_frames": n_mutual,
        "f0_pearson_r": pearson_r,
        "f0_median_abs_diff_hz": med_abs_diff,
        "librosa_f0_median_hz": round(float(np.nanmedian(s_lib.f0)), 2) if np.isfinite(s_lib.f0).any() else None,
        "parselmouth_f0_median_hz": round(float(np.nanmedian(s_pm.f0)), 2) if np.isfinite(s_pm.f0).any() else None,
        "silero_vad_sec": round(t_vad, 4),
        "silero_vad_backend": vad["vad_backend"],
        "silero_speech_ratio": vad["speech_ratio"],
        "silero_speech_duration_sec": vad["speech_duration_sec"],
        "silero_n_segments": len(vad["speech_segments"]),
        "silero_n_pauses": len(vad["pause_durations_sec"]),
    }


def _discover_audio_sessions(root: Path) -> list[str]:
    from report_frontend import voice_series

    sids: list[str] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        sid = session_id_of(d)
        if sid not in sids and voice_series.audio_source(sid)[0] != "none":
            sids.append(sid)
    return sids


def main() -> int:
    import media_retention
    from voice_interaction.asr.whisper_confidence_engine import FasterWhisperConfidenceEngine

    ap = argparse.ArgumentParser(description="Backfill faster-whisper confidence sidecars & benchmark P0 audio tools")
    ap.add_argument("--sessions-dir", default=None)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--session", nargs="*", default=None, help="Explicit session_id(s) to backfill")
    ap.add_argument("--limit", type=int, default=3, help="Max sessions to backfill when --session is omitted")
    ap.add_argument("--model-size", default="small")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--benchmark-only", action="store_true")
    ap.add_argument("--benchmark-audio", default=None)
    ap.add_argument("--benchmark-max-sec", type=float, default=30.0)
    args = ap.parse_args()

    if args.sessions_dir:
        os.environ["JINGXIN_RECORDINGS_DIR"] = args.sessions_dir

    if args.benchmark or args.benchmark_only:
        b_path = Path(args.benchmark_audio) if args.benchmark_audio else find_default_benchmark_audio()
        if b_path is None or not b_path.exists():
            print("✗ No local benchmark audio found", file=sys.stderr)
            if args.benchmark_only:
                return 2
        else:
            bench = run_benchmark(b_path, max_sec=args.benchmark_max_sec)
            print("Benchmark:", json.dumps(bench, ensure_ascii=False))
            if args.benchmark_only:
                return 0

    root = media_retention.root()
    if not root.is_dir():
        print(f"✗ Recordings root not found: {root}", file=sys.stderr)
        return 2

    sids = list(args.session) if args.session else _discover_audio_sessions(root)[: max(0, args.limit)]
    if not sids:
        print("No sessions with audio found.")
        return 0

    engine = FasterWhisperConfidenceEngine(
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
    )
    out_dir = Path(args.out_dir)
    n_ok = n_skip = n_fail = 0
    for sid in sids:
        try:
            res = backfill_session(sid, engine, out_dir=out_dir, force=args.force)
            print(json.dumps(res, ensure_ascii=False))
            if res["status"] == "ok":
                n_ok += 1
            else:
                n_skip += 1
        except Exception as exc:
            n_fail += 1
            print(f"✗ {sid}: {type(exc).__name__}: {exc}", file=sys.stderr)

    print(f"Done: ok={n_ok} skipped={n_skip} failed={n_fail}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
