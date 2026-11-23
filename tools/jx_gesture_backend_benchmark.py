#!/usr/bin/env python3
"""MediaPipe vs RTMW/RTMPose 上肢与手部检测后端基准对比脚本（纯 CPU）。

采样本地匿名化录制视频帧（默认 `20260925_225936_a971/media/camera.webm`，最多 20 帧），
对比默认 MediaPipe（HandDetector + PoseDetector）与可选 RTMW（RTMWWholeBodyEngine
通过共享单帧缓存同时驱动 RTMWHandAdapter + RTMWPoseAdapter）在纯 CPU 下的：
  - 初始化耗时（init_ms）与释放耗时（close_ms）
  - 单帧平均 / P50 / P95 延迟（ms/frame）与吞吐量（FPS）
  - 姿态检出率、手部检出率、双手检出帧数、下游角度/分析器产出可用率
  - 单帧共享缓存命中验证（RTMW 实际推理次数 == 帧数而非 2×帧数）

绝不输出任何受试者 PII。
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import platform
import re
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_ANONYMIZED_SID_RE = re.compile(r"^\d{8}_\d{6}_[a-z0-9]+$")


def _rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def _current_rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                kb = float(line.split()[1])
                return round(kb / 1024.0, 1)
    except OSError:
        pass
    return _rss_mb()


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


def _find_anonymized_video(explicit_video: Path | None = None) -> tuple[Path | None, str]:
    if explicit_video is not None:
        if explicit_video.is_file():
            return explicit_video, explicit_video.parent.parent.name
        return None, "missing"

    rec_root = Path(
        os.environ.get("JINGXIN_RECORDINGS_DIR", str(REPO.parent / "recordings"))
    )
    if not rec_root.is_dir():
        return None, "unavailable"

    # 仅选择纯匿名 ID 目录，绝不读取带姓名/学号目录
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


def _sample_frames(video_path: Path | None, max_frames: int) -> tuple[list[Any], dict[str, Any]]:
    import cv2
    import numpy as np

    if video_path is None:
        frames = [np.full((720, 1280, 3), 128, dtype=np.uint8) for _ in range(max_frames)]
        return frames, {
            "source_type": "synthetic_720p",
            "resolution": "1280x720",
            "frames_sampled": len(frames),
        }

    cap = cv2.VideoCapture(str(video_path))
    raw_frames: list[Any] = []
    try:
        while len(raw_frames) < max(max_frames * 3, 60):
            ok, bgr = cap.read()
            if not ok:
                break
            raw_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    if not raw_frames:
        frames = [np.full((720, 1280, 3), 128, dtype=np.uint8) for _ in range(max_frames)]
        return frames, {
            "source_type": "synthetic_720p_decode_fallback",
            "resolution": "1280x720",
            "frames_sampled": len(frames),
        }

    if len(raw_frames) <= max_frames:
        sampled = raw_frames
    else:
        idxs = np.linspace(0, len(raw_frames) - 1, num=max_frames, dtype=int)
        sampled = [raw_frames[int(i)] for i in idxs]

    h, w = sampled[0].shape[:2]
    return sampled, {
        "source_type": "anonymized_session_video",
        "resolution": f"{w}x{h}",
        "total_decoded_frames": len(raw_frames),
        "frames_sampled": len(sampled),
    }


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


def _run_detector_benchmark(
    name: str,
    hands_det: Any,
    pose_det: Any,
    frames: list[Any],
    init_ms: float,
    rss_before_mb: float,
) -> dict[str, Any]:
    from gesture_analysis.core.analysis.arm_analyzer import ArmAnalyzer
    from gesture_analysis.core.analysis.hand_analyzer import HandAnalyzer
    from gesture_analysis.core.analysis.shoulder_analyzer import ShoulderAnalyzer
    from gesture_analysis.core.analysis.upper_body_analyzer import UpperBodyAnalyzer
    from gesture_analysis.core.feature_extraction.angles import (
        finger_angles,
        pose_angles,
        shoulder_width,
    )

    left_hand_an = HandAnalyzer(hand_id=0)
    right_hand_an = HandAnalyzer(hand_id=1)
    shoulder_an = ShoulderAnalyzer()
    left_arm_an = ArmAnalyzer(arm_id="left")
    right_arm_an = ArmAnalyzer(arm_id="right")
    upper_body_an = UpperBodyAnalyzer()

    # 预热 1 帧（避免首帧 JIT/图优化污染 P50/均值，单独记录首帧耗时）
    t_warm0 = time.perf_counter()
    pose_det.detect_with_world(frames[0], 0)
    hands_det.detect_with_handedness(frames[0], 0)
    warmup_ms = (time.perf_counter() - t_warm0) * 1000.0
    pose_det.reset()
    hands_det.reset()

    engine = getattr(pose_det, "engine", None)
    inf_count_before = getattr(engine, "inference_count", None)

    latencies_ms: list[float] = []
    pose_hits = 0
    hand_frames = 0
    two_hand_frames = 0
    total_hand_instances = 0
    shoulder_width_hits = 0
    pose_angles_hits = 0
    finger_angles_hits = 0
    valid_left_hand_analyzer = 0
    valid_right_hand_analyzer = 0
    valid_pose_analyzers = 0

    for idx, rgb in enumerate(frames):
        ts = (idx + 1) * 100
        t0 = time.perf_counter()
        pose_lms, _world = pose_det.detect_with_world(rgb, ts)
        hand_groups, handedness = hands_det.detect_with_handedness(rgb, ts)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)

        sw = shoulder_width(pose_lms)
        if pose_lms:
            pose_hits += 1
            shoulder_an.update(pose_lms, timestamp_ms=ts, shoulder_width=sw)
            left_arm_an.update(pose_lms, timestamp_ms=ts, shoulder_width=sw)
            right_arm_an.update(pose_lms, timestamp_ms=ts, shoulder_width=sw)
            upper_body_an.update(pose_lms, timestamp_ms=ts, shoulder_width=sw)
            if (
                shoulder_an.is_valid()
                and left_arm_an.is_valid()
                and right_arm_an.is_valid()
                and upper_body_an.is_valid()
            ):
                valid_pose_analyzers += 1
            if sw is not None and sw > 0:
                shoulder_width_hits += 1
            if pose_angles(pose_lms):
                pose_angles_hits += 1

        if hand_groups:
            hand_frames += 1
            if len(hand_groups) >= 2:
                two_hand_frames += 1
            total_hand_instances += len(hand_groups)
            used_slots: set[str] = set()
            for hi, hand_lms in enumerate(hand_groups[:2]):
                entry = handedness[hi] if hi < len(handedness) else None
                slot = None
                if entry is not None:
                    label, _conf = entry
                    side = "right" if label == "Left" else "left"
                    cand = f"{side}_hand"
                    if cand not in used_slots:
                        slot = cand
                if slot is None:
                    slot = "left_hand" if "left_hand" not in used_slots else "right_hand"
                used_slots.add(slot)
                if slot == "left_hand":
                    left_hand_an.update(hand_lms, timestamp_ms=ts, shoulder_width=sw)
                    if left_hand_an.is_valid():
                        valid_left_hand_analyzer += 1
                else:
                    right_hand_an.update(hand_lms, timestamp_ms=ts, shoulder_width=sw)
                    if right_hand_an.is_valid():
                        valid_right_hand_analyzer += 1
                if finger_angles(hand_lms):
                    finger_angles_hits += 1

    inf_count_after = getattr(engine, "inference_count", None)
    inf_calls = (
        int(inf_count_after - inf_count_before)
        if inf_count_before is not None and inf_count_after is not None
        else None
    )

    rss_after_mb = _current_rss_mb()
    t_close0 = time.perf_counter()
    hands_det.close()
    pose_det.close()
    close_ms = (time.perf_counter() - t_close0) * 1000.0

    sorted_lat = sorted(latencies_ms)
    mean_ms = statistics.fmean(latencies_ms)
    n = len(frames)

    return {
        "backend": name,
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
        "pose_detected_frames": pose_hits,
        "pose_detection_rate": round(pose_hits / n, 4),
        "hand_detected_frames": hand_frames,
        "hand_detection_rate": round(hand_frames / n, 4),
        "two_hand_frames": two_hand_frames,
        "total_hand_instances": total_hand_instances,
        "shoulder_width_valid_frames": shoulder_width_hits,
        "pose_angles_frames": pose_angles_hits,
        "finger_angles_instances": finger_angles_hits,
        "valid_pose_analyzer_frames": valid_pose_analyzers,
        "valid_left_hand_analyzer_frames": valid_left_hand_analyzer,
        "valid_right_hand_analyzer_frames": valid_right_hand_analyzer,
        "shared_engine_inference_calls": inf_calls,
        "rss_delta_mb": round(max(0.0, rss_after_mb - rss_before_mb), 1),
        "peak_rss_mb": _rss_mb(),
    }


def _init_rtmw_with_timeout(mode: str, timeout_s: float) -> tuple[Any, Any, float]:
    from gesture_analysis.core.rtmpose_backend import (
        RTMWHandAdapter,
        RTMWPoseAdapter,
        RTMWWholeBodyEngine,
    )

    def _build() -> tuple[Any, Any, float]:
        t0 = time.perf_counter()
        engine = RTMWWholeBodyEngine(mode=mode, backend="onnxruntime", device="cpu")
        engine.ensure_ready()
        hands = RTMWHandAdapter(engine)
        pose = RTMWPoseAdapter(engine)
        return hands, pose, (time.perf_counter() - t0) * 1000.0

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(_build)
    try:
        return fut.result(timeout=timeout_s)
    finally:
        pool.shutdown(wait=False)


def run_benchmark(
    *,
    video_path: Path | None = None,
    max_frames: int = 20,
    rtmw_mode: str = "lightweight",
    rtmw_timeout_s: float = 45.0,
) -> dict[str, Any]:
    from gesture_analysis.core.detectors import create_gesture_detectors

    resolved_video, session_tag = _find_anonymized_video(video_path)
    if resolved_video is not None and not _ANONYMIZED_SID_RE.fullmatch(session_tag):
        session_tag = "anonymized_custom"

    frames, sample_meta = _sample_frames(resolved_video, max_frames=max_frames)

    # 1. MediaPipe 基准
    rss_mp0 = _current_rss_mb()
    t_mp0 = time.perf_counter()
    mp_hands, mp_pose = create_gesture_detectors(backend="mediapipe")
    mp_init_ms = (time.perf_counter() - t_mp0) * 1000.0
    mp_report = _run_detector_benchmark(
        "mediapipe", mp_hands, mp_pose, frames, mp_init_ms, rss_mp0
    )

    # 2. RTMW 可选后端基准（带超时与优雅回退诊断）
    rss_rtmw0 = _current_rss_mb()
    rtmw_report: dict[str, Any]
    try:
        rtmw_hands, rtmw_pose, rtmw_init_ms = _init_rtmw_with_timeout(
            rtmw_mode, timeout_s=rtmw_timeout_s
        )
        rtmw_report = _run_detector_benchmark(
            f"rtmw:{rtmw_mode}", rtmw_hands, rtmw_pose, frames, rtmw_init_ms, rss_rtmw0
        )
    except Exception as exc:
        rtmw_report = {
            "backend": f"rtmw:{rtmw_mode}",
            "status": "unavailable_fallback_to_mediapipe",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "frames_tested": 0,
            "peak_rss_mb": _rss_mb(),
        }

    comparison: dict[str, Any] = {}
    if mp_report.get("status") == "ok" and rtmw_report.get("status") == "ok":
        mp_mean = mp_report["mean_ms"]
        rt_mean = rtmw_report["mean_ms"]
        comparison = {
            "speedup_ratio_mp_over_rtmw": round(mp_mean / rt_mean, 3) if rt_mean > 0 else None,
            "latency_diff_ms_rtmw_minus_mp": round(rt_mean - mp_mean, 2),
            "pose_detection_rate_diff": round(
                rtmw_report["pose_detection_rate"] - mp_report["pose_detection_rate"], 4
            ),
            "hand_detection_rate_diff": round(
                rtmw_report["hand_detection_rate"] - mp_report["hand_detection_rate"], 4
            ),
            "total_hand_instances_diff": (
                rtmw_report["total_hand_instances"] - mp_report["total_hand_instances"]
            ),
            "single_frame_cache_verified": (
                rtmw_report.get("shared_engine_inference_calls") == rtmw_report["frames_tested"]
            ),
        }

    return {
        "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": _cpu_info(),
        "sample": {
            "session_id": session_tag,
            **sample_meta,
        },
        "mediapipe": mp_report,
        "rtmw": rtmw_report,
        "comparison": comparison,
        "overall_peak_rss_mb": _rss_mb(),
    }


def main() -> int:
    default_out = (
        REPO.parent
        / "07_三方向与评审筹备"
        / "系统补强"
        / "gesture_backend_benchmark_20261124.json"
    )
    parser = argparse.ArgumentParser(description="MediaPipe vs RTMW 手势/上肢后端对比基准")
    parser.add_argument("--video", type=Path, default=None, help="可选匿名录制视频路径")
    parser.add_argument("--frames", type=int, default=20, help="采样帧数（默认 20）")
    parser.add_argument(
        "--rtmw-mode",
        default=os.getenv("JX_RTMW_MODE", "lightweight"),
        choices=["lightweight", "balanced", "performance"],
        help="RTMW 模型档位（默认 lightweight）",
    )
    parser.add_argument("--timeout", type=float, default=45.0, help="RTMW 初始化超时秒数")
    parser.add_argument("--out", type=Path, default=default_out, help="输出 JSON 路径")
    args = parser.parse_args()

    report = run_benchmark(
        video_path=args.video,
        max_frames=args.frames,
        rtmw_mode=args.rtmw_mode,
        rtmw_timeout_s=args.timeout,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
