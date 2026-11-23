#!/usr/bin/env python3
"""Wave 1 采集层端到端局部冒烟与三服务基准测试脚本（零网络/零浏览器/零 PII）。

测量并验证升级后的三条实时采集链路：
1. 面部通道（Face）：`face_expression.pipeline.video_pipeline.VideoPipeline` 在真实视频 15 帧上的耗时、检出率与 `/analyze` + `/health` 冒烟；
2. 上肢体态通道（Gesture）：`gesture_analysis.api.app` 在 `JX_GESTURE_BACKEND=mediapipe` 与 `rtmw` 下 `get_or_create_detectors` / `process_frame` / `/analyze` (`detector_backend`) / `/reset` 清理；
3. 语音通道（Voice）：`voice_interaction.api.app` 的 `prosody_features_from_pcm` 与 `/health` 在传统路径 (`librosa` + RMS) 与升级路径 (`parselmouth` + `silero_vad_onnx`) 下的耗时、RTF、声学质量特征与 `VoiceLogger` 32 列 CSV 兼容落盘。

默认输出到 `/home/zgy/JingXin/07_三方向与评审筹备/系统补强/capture_layer_v2_benchmark_20261124.json`。
绝不在输出或日志中记录任何受试者姓名、学号或含 PII 的目录路径。
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import importlib
import io
import json
import os
import platform
import re
import resource
import statistics
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
os.environ.setdefault("JOBLIB_TEMP_FOLDER", "/tmp/joblib_cache")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_ANONYMIZED_SID_RE = re.compile(r"^\d{8}_\d{6}_[a-z0-9]+$")
_ANONYMIZED_TAIL_RE = re.compile(r"(\d{8}_\d{6}_[a-z0-9]+)$")


def _rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def _cpu_info() -> dict[str, Any]:
    model = platform.processor() or "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "cpu_model": model,
        "cpu_count": os.cpu_count() or 1,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def _sanitize_sid(folder_name: str) -> str:
    if _ANONYMIZED_SID_RE.fullmatch(folder_name):
        return folder_name
    m = _ANONYMIZED_TAIL_RE.search(folder_name)
    return m.group(1) if m else "anonymized_session"


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _find_anonymized_video(rec_root: Path, explicit: Path | None) -> tuple[Path | None, str]:
    if explicit is not None:
        if explicit.is_file():
            return explicit, _sanitize_sid(explicit.parent.parent.name)
        return None, "missing"

    if not rec_root.is_dir():
        return None, "unavailable"

    preferred = rec_root / "20260925_225936_a971" / "media" / "camera.webm"
    if preferred.is_file():
        return preferred, "20260925_225936_a971"

    for d in sorted(rec_root.iterdir()):
        if not d.is_dir() or not _ANONYMIZED_SID_RE.fullmatch(d.name):
            continue
        cand = d / "media" / "camera.webm"
        if cand.is_file():
            return cand, d.name
    return None, "unavailable"


def _find_anonymized_audio(rec_root: Path, explicit: Path | None) -> tuple[Path | None, str]:
    if explicit is not None:
        if explicit.is_file():
            return explicit, _sanitize_sid(explicit.parent.parent.parent.name)
        return None, "missing"

    if not rec_root.is_dir():
        return None, "unavailable"

    # 优先使用全匿名化录制目录下的 16k WAV
    for sid in ("20260926_155559_caf0", "20260927_212143_95a8"):
        cand = rec_root / sid / "media" / "audio" / "0001_converted.wav"
        if cand.is_file():
            return cand, sid

    for d in sorted(rec_root.iterdir()):
        if not d.is_dir() or not _ANONYMIZED_SID_RE.fullmatch(d.name):
            continue
        audio_dir = d / "media" / "audio"
        if audio_dir.is_dir():
            wavs = sorted(audio_dir.glob("*_converted.wav"))
            if wavs:
                return wavs[0], d.name

    # 若仅存在含尾缀的目录，仅提取末尾匿名 ID，绝不保留目录名中的姓名/学号
    for d in sorted(rec_root.iterdir()):
        if not d.is_dir():
            continue
        m = _ANONYMIZED_TAIL_RE.search(d.name)
        if not m:
            continue
        audio_dir = d / "media" / "audio"
        if audio_dir.is_dir():
            wavs = sorted(audio_dir.glob("*_converted.wav"))
            if wavs:
                return wavs[0], m.group(1)
    return None, "unavailable"


def _sample_video_frames(video_path: Path | None, n_frames: int) -> tuple[list[Any], bytes, dict[str, Any]]:
    import cv2
    import numpy as np

    if video_path is None:
        frames = [np.full((720, 1280, 3), 128, dtype=np.uint8) for _ in range(n_frames)]
        ok, buf = cv2.imencode(".png", cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
        frame_bytes = buf.tobytes() if ok else b""
        return frames, frame_bytes, {
            "source_type": "synthetic_720p",
            "resolution": "1280x720",
            "total_decoded_frames": n_frames,
            "frames_sampled": len(frames),
        }

    cap = cv2.VideoCapture(str(video_path))
    raw_frames: list[Any] = []
    try:
        while len(raw_frames) < max(n_frames * 3, 60):
            ok, bgr = cap.read()
            if not ok:
                break
            raw_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if not raw_frames:
        frames = [np.full((720, 1280, 3), 128, dtype=np.uint8) for _ in range(n_frames)]
        ok, buf = cv2.imencode(".png", cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
        return frames, (buf.tobytes() if ok else b""), {
            "source_type": "synthetic_720p_decode_fallback",
            "resolution": "1280x720",
            "total_decoded_frames": 0,
            "frames_sampled": len(frames),
        }

    if len(raw_frames) <= n_frames:
        sampled = raw_frames
    else:
        idxs = np.linspace(0, len(raw_frames) - 1, num=n_frames, dtype=int)
        sampled = [raw_frames[int(i)] for i in idxs]

    ok, buf = cv2.imencode(".png", cv2.cvtColor(sampled[0], cv2.COLOR_RGB2BGR))
    frame_bytes = buf.tobytes() if ok else b""
    h, w = sampled[0].shape[:2]
    return sampled, frame_bytes, {
        "source_type": "anonymized_session_video",
        "media_file": "camera.webm",
        "resolution": f"{w}x{h}",
        "total_decoded_frames": len(raw_frames),
        "frames_sampled": len(sampled),
    }


def _load_pcm16_16k(audio_path: Path | None, voice_app: Any) -> tuple[bytes, dict[str, Any]]:
    import numpy as np

    if audio_path is None:
        sr = 16000
        dur = 3.0
        t = np.arange(int(dur * sr), dtype=np.float64) / sr
        sig = (0.3 * np.sin(2.0 * np.pi * 165.0 * t)).astype(np.float32)
        sig[int(1.0 * sr):int(1.5 * sr)] = 0.0
        pcm = (np.clip(sig, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        return pcm, {
            "source_type": "synthetic_16k_wav",
            "sample_rate": sr,
            "channels": 1,
            "sample_width_bytes": 2,
            "duration_sec": dur,
        }

    raw_bytes = audio_path.read_bytes()
    wav_bytes = raw_bytes if raw_bytes.startswith(b"RIFF") else voice_app._to_wav16k_bytes(raw_bytes)
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n_frames = wf.getnframes()
        pcm = wf.readframes(n_frames)

    if sr != 16000 or ch != 1 or sw != 2:
        wav_bytes = voice_app._to_wav16k_bytes(raw_bytes)
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            n_frames = wf.getnframes()
            pcm = wf.readframes(n_frames)

    return pcm, {
        "source_type": "anonymized_session_audio",
        "media_file": audio_path.name,
        "sample_rate": sr,
        "channels": ch,
        "sample_width_bytes": sw,
        "duration_sec": round(n_frames / float(sr), 2),
    }


class _MemUpload:
    """最小上传文件替身，供本地端点冒烟直调。"""

    def __init__(self, data: bytes, filename: str, content_type: str):
        self._data = data
        self.filename = filename
        self.content_type = content_type

    async def read(self) -> bytes:
        return self._data


class _MemRequest:
    """最小 Request 替身，模拟不含表单 session_id 的请求。"""

    async def form(self):
        return {}


def _bench_face_channel(frames: list[Any], jpeg_bytes: bytes, face_app: Any) -> dict[str, Any]:
    from face_expression.pipeline.video_pipeline import VideoPipeline

    t_init0 = time.perf_counter()
    pipe = VideoPipeline(session_id="smoke_face_pipe")
    init_ms = (time.perf_counter() - t_init0) * 1000.0

    t_warm0 = time.perf_counter()
    pipe.process_frame(frames[0], timestamp_ms=0)
    warmup_ms = (time.perf_counter() - t_warm0) * 1000.0
    pipe.reset()

    latencies_ms: list[float] = []
    detected = 0
    blendshape_hits = 0
    for idx, rgb in enumerate(frames):
        ts = (idx + 1) * 100
        t0 = time.perf_counter()
        res_obj, _landmarks, feat = pipe.process_frame(rgb, timestamp_ms=ts)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        if res_obj is not None:
            detected += 1
            if any(k.startswith("bs_") for k in feat):
                blendshape_hits += 1

    t_close0 = time.perf_counter()
    pipe.close()
    close_ms = (time.perf_counter() - t_close0) * 1000.0

    # 端点直调冒烟（/health + /analyze + /session/{sid}/reset）
    sid = "smoke_face_api"
    health = asyncio.run(face_app.health_check())
    resp = asyncio.run(
        face_app.analyze_frame(
            request=_MemRequest(),
            file=_MemUpload(jpeg_bytes, "frame.png", "image/png"),
            session_id=sid,
            probe=True,
        )
    )
    resp_body = json.loads(resp.body)
    reset_res = asyncio.run(face_app.reset_session(sid))

    sorted_lat = sorted(latencies_ms)
    mean_ms = statistics.fmean(latencies_ms)
    n = len(frames)
    return {
        "status": "ok",
        "frames_tested": n,
        "init_ms": round(init_ms, 2),
        "warmup_ms": round(warmup_ms, 2),
        "close_ms": round(close_ms, 2),
        "mean_ms": round(mean_ms, 2),
        "p50_ms": round(_percentile(sorted_lat, 0.50), 2),
        "p95_ms": round(_percentile(sorted_lat, 0.95), 2),
        "min_ms": round(sorted_lat[0], 2),
        "max_ms": round(sorted_lat[-1], 2),
        "throughput_fps": round(1000.0 / mean_ms, 2) if mean_ms > 0 else 0.0,
        "frames_detected": detected,
        "detection_rate": round(detected / n, 4),
        "blendshape_frames": blendshape_hits,
        "api_smoke": {
            "health_status": health.get("status"),
            "analyze_status": resp_body.get("status"),
            "session_id": resp_body.get("session_id"),
            "reset_status": reset_res.get("status"),
        },
    }


def _bench_gesture_channel(
    backend_env: str,
    frames: list[Any],
    jpeg_bytes: bytes,
    gesture_app: Any,
    rtmw_mode: str,
) -> dict[str, Any]:
    prev_backend = os.environ.get("JX_GESTURE_BACKEND")
    prev_mode = os.environ.get("JX_RTMW_MODE")
    os.environ["JX_GESTURE_BACKEND"] = backend_env
    os.environ["JX_RTMW_MODE"] = rtmw_mode

    sid = f"smoke_gesture_{backend_env}"
    try:
        gesture_app._reset_session(sid)
        t_init0 = time.perf_counter()
        analyzers = gesture_app.get_or_create_analyzers(sid)
        dets = gesture_app.get_or_create_detectors(sid)
        init_ms = (time.perf_counter() - t_init0) * 1000.0
        resolved_backend = gesture_app.detectors_backend.get(sid, "unknown")

        # 预热首帧后归零，避免预热污染统计或造成时间戳冲突
        t_warm0 = time.perf_counter()
        gesture_app.process_frame(frames[0], 0, analyzers, dets)
        warmup_ms = (time.perf_counter() - t_warm0) * 1000.0
        for det in dets.values():
            det.reset()

        engine = getattr(dets.get("pose"), "engine", None)
        inf_before = getattr(engine, "inference_count", None)

        latencies_ms: list[float] = []
        pose_hits = 0
        hand_frames = 0
        two_hand_frames = 0
        total_hand_instances = 0
        valid_pose_frames = 0

        for idx, rgb in enumerate(frames):
            ts = (idx + 1) * 100
            t0 = time.perf_counter()
            out = gesture_app.process_frame(rgb, ts, analyzers, dets)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            n_hands = int(out.get("detected_hands", 0))
            if n_hands > 0:
                hand_frames += 1
                total_hand_instances += n_hands
                if n_hands >= 2:
                    two_hand_frames += 1
            sh_res = out.get("shoulder_results")
            if sh_res is not None:
                pose_hits += 1
                if (
                    sh_res.get("is_valid")
                    and (out.get("left_arm_results") or {}).get("is_valid")
                    and (out.get("right_arm_results") or {}).get("is_valid")
                    and (out.get("upper_body_results") or {}).get("is_valid")
                ):
                    valid_pose_frames += 1

        inf_after = getattr(engine, "inference_count", None)
        shared_calls = (
            int(inf_after - inf_before)
            if inf_before is not None and inf_after is not None
            else None
        )

        # 真实端点直调冒烟：/health + /analyze + /reset（验证 detector_backend 与表清理）
        api_sid = f"{sid}_api"
        gesture_app._reset_session(api_sid)
        health = asyncio.run(gesture_app.health_check())
        api_resp = asyncio.run(
            gesture_app.analyze_image(
                request=_MemRequest(),
                file=_MemUpload(jpeg_bytes, "frame.png", "image/png"),
                session_id=api_sid,
                probe=True,
            )
        )
        t_close0 = time.perf_counter()
        gesture_app._reset_session(sid)
        reset_res = asyncio.run(gesture_app.reset_session_by_path(api_sid))
        close_ms = (time.perf_counter() - t_close0) * 1000.0

        sorted_lat = sorted(latencies_ms)
        mean_ms = statistics.fmean(latencies_ms)
        n = len(frames)
        return {
            "status": "ok",
            "backend_configured": backend_env,
            "resolved_backend": resolved_backend,
            "frames_tested": n,
            "init_ms": round(init_ms, 2),
            "warmup_ms": round(warmup_ms, 2),
            "reset_dispatch_ms": round(close_ms, 2),
            "mean_ms": round(mean_ms, 2),
            "p50_ms": round(_percentile(sorted_lat, 0.50), 2),
            "p95_ms": round(_percentile(sorted_lat, 0.95), 2),
            "min_ms": round(sorted_lat[0], 2),
            "max_ms": round(sorted_lat[-1], 2),
            "throughput_fps": round(1000.0 / mean_ms, 2) if mean_ms > 0 else 0.0,
            "pose_detected_frames": pose_hits,
            "pose_detection_rate": round(pose_hits / n, 4),
            "hand_detected_frames": hand_frames,
            "hand_detection_rate": round(hand_frames / n, 4),
            "two_hand_frames": two_hand_frames,
            "total_hand_instances": total_hand_instances,
            "valid_pose_analyzer_frames": valid_pose_frames,
            "shared_engine_inference_calls": shared_calls,
            "api_smoke": {
                "health_backend": health.get("backend"),
                "supported_backends": health.get("supported_backends"),
                "analyze_status": api_resp.get("status"),
                "detector_backend": api_resp.get("detector_backend"),
                "reset_status": reset_res.get("status"),
                "detectors_cleaned": sid not in gesture_app.detectors and api_sid not in gesture_app.detectors,
                "detectors_backend_cleaned": sid not in gesture_app.detectors_backend and api_sid not in gesture_app.detectors_backend,
            },
        }
    finally:
        if prev_backend is None:
            os.environ.pop("JX_GESTURE_BACKEND", None)
        else:
            os.environ["JX_GESTURE_BACKEND"] = prev_backend
        if prev_mode is None:
            os.environ.pop("JX_RTMW_MODE", None)
        else:
            os.environ["JX_RTMW_MODE"] = prev_mode


def _bench_voice_mode(
    pcm: bytes,
    duration_sec: float,
    repeats: int,
    *,
    pitch_backend: str,
    use_silero: bool,
    voice_app: Any,
    tmp_logs: Path,
) -> dict[str, Any]:
    prev_pitch = os.environ.get("JX_LIVE_PITCH_BACKEND")
    prev_vad = os.environ.get("JX_LIVE_USE_SILERO_VAD")
    os.environ["JX_LIVE_PITCH_BACKEND"] = pitch_backend
    os.environ["JX_LIVE_USE_SILERO_VAD"] = "1" if use_silero else "0"

    try:
        health = asyncio.run(voice_app.health_check())

        t_warm0 = time.perf_counter()
        feats = voice_app.prosody_features_from_pcm(pcm)
        warmup_ms = (time.perf_counter() - t_warm0) * 1000.0

        times_ms: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            feats = voice_app.prosody_features_from_pcm(pcm)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

        from voice_interaction.utils.logger import VoiceLogger

        tag = f"smoke_{pitch_backend}_{'silero' if use_silero else 'rms'}"
        vlog = VoiceLogger(log_type="interview", log_dir=str(tmp_logs), session_id=tag)
        vlog.log_prosody(feats, question_index=0, emotion="", feedback="")
        with open(vlog.csv_file, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        csv_ok = len(rows) == 1 and list(rows[0].keys()) == vlog.fieldnames

        mean_ms = statistics.fmean(times_ms)
        rtf = (mean_ms / 1000.0) / duration_sec if duration_sec > 0 else 0.0
        summary_keys = (
            "pitch_mean",
            "pitch_variation",
            "pitch_p10",
            "pitch_p90",
            "energy_mean",
            "energy_variation",
            "speech_duration_sec",
            "pause_duration_mean",
            "pause_duration_max",
            "pause_frequency",
            "voiced_prob_mean",
            "hnr_db",
            "jitter_local",
            "shimmer_local",
            "pitch_backend",
            "vad_backend",
        )
        output_summary = {k: feats.get(k) for k in summary_keys if k in feats}

        return {
            "status": "ok",
            "live_pitch_backend": health.get("live_pitch_backend"),
            "live_vad_backend": health.get("live_vad_backend"),
            "repeats": repeats,
            "warmup_ms": round(warmup_ms, 2),
            "mean_ms": round(mean_ms, 2),
            "p50_ms": round(statistics.median(times_ms), 2),
            "min_ms": round(min(times_ms), 2),
            "max_ms": round(max(times_ms), 2),
            "rtf": round(rtf, 4),
            "speedup_vs_realtime": round(1.0 / rtf, 2) if rtf > 0 else 0.0,
            "csv_schema_roundtrip_ok": csv_ok,
            "csv_column_count": len(vlog.fieldnames),
            "health_snapshot": health,
            "output_summary": output_summary,
        }
    finally:
        if prev_pitch is None:
            os.environ.pop("JX_LIVE_PITCH_BACKEND", None)
        else:
            os.environ["JX_LIVE_PITCH_BACKEND"] = prev_pitch
        if prev_vad is None:
            os.environ.pop("JX_LIVE_USE_SILERO_VAD", None)
        else:
            os.environ["JX_LIVE_USE_SILERO_VAD"] = prev_vad


def run_smoke(
    *,
    video_path: Path | None = None,
    audio_path: Path | None = None,
    n_frames: int = 15,
    audio_repeats: int = 3,
    rtmw_mode: str = "lightweight",
) -> dict[str, Any]:
    rec_root = Path(
        os.environ.get("JINGXIN_RECORDINGS_DIR", str(REPO.parent / "recordings"))
    )
    resolved_video, video_sid = _find_anonymized_video(rec_root, video_path)
    resolved_audio, audio_sid = _find_anonymized_audio(rec_root, audio_path)

    frames, jpeg_bytes, video_meta = _sample_video_frames(resolved_video, n_frames=n_frames)

    with tempfile.TemporaryDirectory(prefix="jx_capture_smoke_") as tmpdir_str:
        tmp_path = Path(tmpdir_str)
        tmp_data = tmp_path / "data"
        tmp_logs = tmp_data / "logs"
        tmp_rec = tmp_path / "recordings"
        tmp_logs.mkdir(parents=True, exist_ok=True)
        tmp_rec.mkdir(parents=True, exist_ok=True)

        prev_data_dir = os.environ.get("JINGXIN_DATA_DIR")
        prev_rec_dir = os.environ.get("JINGXIN_RECORDINGS_DIR")
        prev_retain = os.environ.get("JINGXIN_RETAIN_MEDIA")
        os.environ["JINGXIN_DATA_DIR"] = str(tmp_data)
        os.environ["JINGXIN_RECORDINGS_DIR"] = str(tmp_rec)
        os.environ["JINGXIN_RETAIN_MEDIA"] = "0"

        import media_retention
        media_retention._reset_for_tests()

        face_app = importlib.import_module("face_expression.api.app")
        face_logger_mod = importlib.import_module("face_expression.utils.logger")
        gesture_app = importlib.import_module("gesture_analysis.api.app")
        gesture_logger_mod = importlib.import_module("gesture_analysis.utils.logger")
        voice_app = importlib.import_module("voice_interaction.api.app")
        voice_logger_mod = importlib.import_module("voice_interaction.utils.logger")

        orig_face_logs = face_app.LOGS_DIR
        orig_face_mod_logs = face_logger_mod.LOGS_DIR
        orig_gesture_logs = gesture_app.LOGS_DIR
        orig_gesture_mod_logs = gesture_logger_mod.LOGS_DIR
        orig_voice_mod_logs = voice_logger_mod.LOGS_DIR

        face_app.LOGS_DIR = str(tmp_logs)
        face_logger_mod.LOGS_DIR = str(tmp_logs)
        gesture_app.LOGS_DIR = tmp_logs
        gesture_logger_mod.LOGS_DIR = str(tmp_logs)
        voice_logger_mod.LOGS_DIR = tmp_logs

        try:
            pcm, audio_meta = _load_pcm16_16k(resolved_audio, voice_app)
            dur_sec = float(audio_meta["duration_sec"])

            face_res = _bench_face_channel(frames, jpeg_bytes, face_app)
            gesture_mp_res = _bench_gesture_channel(
                "mediapipe", frames, jpeg_bytes, gesture_app, rtmw_mode
            )
            gesture_rtmw_res = _bench_gesture_channel(
                "rtmw", frames, jpeg_bytes, gesture_app, rtmw_mode
            )

            voice_legacy = _bench_voice_mode(
                pcm,
                dur_sec,
                audio_repeats,
                pitch_backend="librosa",
                use_silero=False,
                voice_app=voice_app,
                tmp_logs=tmp_logs,
            )
            voice_upgraded = _bench_voice_mode(
                pcm,
                dur_sec,
                audio_repeats,
                pitch_backend="parselmouth",
                use_silero=True,
                voice_app=voice_app,
                tmp_logs=tmp_logs,
            )
        finally:
            face_app.LOGS_DIR = orig_face_logs
            face_logger_mod.LOGS_DIR = orig_face_mod_logs
            gesture_app.LOGS_DIR = orig_gesture_logs
            gesture_logger_mod.LOGS_DIR = orig_gesture_mod_logs
            voice_logger_mod.LOGS_DIR = orig_voice_mod_logs
            if prev_data_dir is None:
                os.environ.pop("JINGXIN_DATA_DIR", None)
            else:
                os.environ["JINGXIN_DATA_DIR"] = prev_data_dir
            if prev_rec_dir is None:
                os.environ.pop("JINGXIN_RECORDINGS_DIR", None)
            else:
                os.environ["JINGXIN_RECORDINGS_DIR"] = prev_rec_dir
            if prev_retain is None:
                os.environ.pop("JINGXIN_RETAIN_MEDIA", None)
            else:
                os.environ["JINGXIN_RETAIN_MEDIA"] = prev_retain
            media_retention._reset_for_tests()

    speedup = (
        round(voice_legacy["mean_ms"] / voice_upgraded["mean_ms"], 2)
        if voice_upgraded["mean_ms"] > 0
        else None
    )
    report = {
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": _cpu_info(),
        "video_e2e": {
            "sample": {
                "session_id": video_sid,
                **video_meta,
            },
            "face_video_pipeline": face_res,
            "gesture_mediapipe": gesture_mp_res,
            "gesture_rtmw": gesture_rtmw_res,
            "combined_video_budget": {
                "face_plus_gesture_mp_mean_ms": round(
                    face_res["mean_ms"] + gesture_mp_res["mean_ms"], 2
                ),
                "face_plus_gesture_rtmw_mean_ms": round(
                    face_res["mean_ms"] + gesture_rtmw_res["mean_ms"], 2
                ),
            },
        },
        "audio_e2e": {
            "sample": {
                "session_id": audio_sid,
                **audio_meta,
            },
            "legacy_librosa_rms": voice_legacy,
            "upgraded_parselmouth_silero": voice_upgraded,
            "comparison": {
                "speedup_upgraded_over_legacy": speedup,
                "latency_saved_ms": round(
                    voice_legacy["mean_ms"] - voice_upgraded["mean_ms"], 2
                ),
            },
        },
        "overall_peak_rss_mb": _rss_mb(),
    }

    from tools.jx_pii_guard import scan_text

    hits = scan_text(json.dumps(report, ensure_ascii=False))
    if hits:
        raise RuntimeError(f"基准输出包含疑似 PII: {hits}")
    return report


def main() -> int:
    default_out = (
        REPO.parent
        / "07_三方向与评审筹备"
        / "系统补强"
        / "capture_layer_v2_benchmark_20261124.json"
    )
    parser = argparse.ArgumentParser(description="Wave 1 三服务采集层本地端到端基准冒烟")
    parser.add_argument("--video", type=Path, default=None, help="可选匿名视频文件路径")
    parser.add_argument("--audio", type=Path, default=None, help="可选匿名音频文件路径")
    parser.add_argument("--frames", type=int, default=15, help="视频测试采样帧数（默认 15）")
    parser.add_argument("--repeats", type=int, default=3, help="音频提取重复次数（默认 3）")
    parser.add_argument(
        "--rtmw-mode",
        default=os.getenv("JX_RTMW_MODE", "lightweight"),
        choices=["lightweight", "balanced", "performance"],
        help="RTMW 档位（默认 lightweight）",
    )
    parser.add_argument("--out", type=Path, default=default_out, help="JSON 输出路径")
    args = parser.parse_args()

    report = run_smoke(
        video_path=args.video,
        audio_path=args.audio,
        n_frames=args.frames,
        audio_repeats=args.repeats,
        rtmw_mode=args.rtmw_mode,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
