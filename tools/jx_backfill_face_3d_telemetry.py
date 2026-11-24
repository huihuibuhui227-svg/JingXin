#!/usr/bin/env python3
"""Backfill real 3D head pose and optical frame quality telemetry onto face CSV logs.

Reuses production extraction helpers (`FaceDetector.detect_full`, `assess_frame_quality`,
`_extract_flat_telemetry`) over retained RGB frames (`media/face/*.jpg` via
`retention.face.jsonl`, or fallback `media/camera.webm` by timestamp), preserves all
existing 92 columns verbatim, appends the 6 Wave 2 telemetry columns matching the
98-column `DataLogger` contract, and stores pre-3D backups strictly outside `data/logs/`.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import csv
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np
from scipy.spatial import distance as dist

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from face_expression.config import FACE_MODEL
from face_expression.core.feature_extraction.landmarks import CHIN, FOREHEAD_TOP
from face_expression.pipeline.detector import FaceDetector
from face_expression.pipeline.framing import CROP_LADDER, apply_crop
from face_expression.pipeline.video_pipeline import _extract_flat_telemetry, assess_frame_quality
from face_expression.utils.logger import DataLogger

TELEMETRY_COLS: list[str] = [
    "head_yaw_3d",
    "head_pitch_3d",
    "head_roll_3d",
    "frame_brightness",
    "frame_blur_laplacian",
    "frame_quality_ok",
]

FLAGSHIP_SESSIONS: tuple[str, ...] = (
    "20261026_002930_cb79",
    "20261026_130633_0d89",
    "20260929_200333_2e84",
    "20260928_165520_90e7",
    "20260927_191325_253a",
)

DEFAULT_LOGS_DIR = REPO / "data" / "logs"
DEFAULT_RECORDINGS_DIR = REPO.parent / "recordings"
DEFAULT_BACKUP_DIR = REPO.parent / "_jxwork" / "backup_face_csv_pre3d"
MAX_ALIGN_TOLERANCE_SEC = 1.5


@dataclass(frozen=True)
class AlignedFrame:
    """Reference to one frame aligned to a CSV row."""

    source_type: str  # "jpg" | "webm"
    path: Path
    timestamp_ms: int
    frame_key: Any


def _parse_finite_float(raw: Any) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        val = float(s)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def _default_detector_factory(model_path: Path | str = FACE_MODEL) -> FaceDetector:
    return FaceDetector(model_path)


def find_recording_dir(recordings_dir: Path, session_id: str) -> Path | None:
    """Locate recording directory for `session_id` without exposing folder names in logs."""
    if not recordings_dir.is_dir():
        return None
    direct = recordings_dir / session_id
    if direct.is_dir():
        return direct
    suffix = f"__{session_id}"
    for entry in sorted(recordings_dir.iterdir(), key=lambda p: p.name):
        if entry.is_dir() and (entry.name == session_id or entry.name.endswith(suffix)):
            return entry
    return None


def discover_matched_sessions(logs_dir: Path, recordings_dir: Path) -> list[str]:
    """Discover all session_ids having both a face CSV log and a matching recording dir."""
    if not logs_dir.is_dir() or not recordings_dir.is_dir():
        return []
    matched: list[str] = []
    for csv_path in sorted(logs_dir.glob("face_au_log_*.csv")):
        sid = csv_path.stem[len("face_au_log_"):]
        if not sid or sid.startswith("NONE"):
            continue
        if find_recording_dir(recordings_dir, sid) is not None:
            matched.append(sid)
    return matched


def _already_backfilled(orig_header: list[str], rows: list[dict[str, str]]) -> bool:
    if not all(col in orig_header for col in TELEMETRY_COLS):
        return False
    return any(_parse_finite_float(r.get("head_yaw_3d")) is not None for r in rows)


def _build_output_header(orig_header: list[str], session_id: str) -> list[str]:
    orig_without = [c for c in orig_header if c not in TELEMETRY_COLS]
    out_header = orig_without + TELEMETRY_COLS
    if len(orig_without) == 92:
        assert len(out_header) == 98, f"Expected 98 columns, got {len(out_header)}"
        expected = list(DataLogger(log_type="video", session_id=session_id).fieldnames)
        assert out_header == expected, (
            f"Header mismatch against DataLogger contract for {session_id}"
        )
    return out_header


def _load_ledger_records(rec_dir: Path) -> list[dict[str, Any]]:
    ledger_path = rec_dir / "retention.face.jsonl"
    if not ledger_path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for idx, line in enumerate(ledger_path.read_text(encoding="utf-8").splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rec = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if rec.get("kind", "frame") != "frame":
            continue
        rel = rec.get("file")
        ts = rec.get("declared_ts")
        if not rel or ts is None:
            continue
        try:
            seq = int(rec.get("seq", idx))
            ts_ms = int(round(float(ts)))
        except (TypeError, ValueError):
            continue
        records.append({
            "seq": seq,
            "idx": idx,
            "path": rec_dir / str(rel),
            "declared_ts": ts_ms,
        })
    records.sort(key=lambda r: (r["seq"], r["idx"]))
    return records


def _align_frames_from_ledger(
    csv_rows: list[dict[str, str]],
    ledger: list[dict[str, Any]],
) -> list[AlignedFrame | None] | None:
    if not ledger:
        return None
    existing_any = any(rec["path"].is_file() for rec in ledger)
    if not existing_any:
        return None

    row_ts = [_parse_finite_float(r.get("timestamp")) for r in csv_rows]

    # Fast exact 1-to-1 alignment when line counts match and timestamp delta is small.
    if len(ledger) == len(csv_rows):
        diffs = [
            abs(t - rec["declared_ts"] / 1000.0)
            for t, rec in zip(row_ts, ledger)
            if t is not None
        ]
        median_diff = float(np.median(diffs)) if diffs else 0.0
        if median_diff <= MAX_ALIGN_TOLERANCE_SEC:
            aligned: list[AlignedFrame | None] = []
            for rec in ledger:
                if rec["path"].is_file():
                    aligned.append(
                        AlignedFrame(
                            source_type="jpg",
                            path=rec["path"],
                            timestamp_ms=rec["declared_ts"],
                            frame_key=("jpg", rec["seq"], str(rec["path"])),
                        )
                    )
                else:
                    aligned.append(None)
            return aligned

    # General nearest-timestamp alignment within tolerance.
    valid_recs = [rec for rec in ledger if rec["path"].is_file()]
    if not valid_recs:
        return None
    led_secs = np.array([rec["declared_ts"] / 1000.0 for rec in valid_recs], dtype=np.float64)
    # Sort by timestamp for binary search while keeping original seq in frame_key.
    order = np.argsort(led_secs, kind="mergesort")
    sorted_secs = led_secs[order]
    sorted_recs = [valid_recs[int(i)] for i in order]

    aligned = []
    for t in row_ts:
        if t is None:
            aligned.append(None)
            continue
        pos = int(np.searchsorted(sorted_secs, t))
        candidates: list[int] = []
        if pos < len(sorted_secs):
            candidates.append(pos)
        if pos > 0:
            candidates.append(pos - 1)
        best_idx = min(candidates, key=lambda k: abs(float(sorted_secs[k]) - t))
        if abs(float(sorted_secs[best_idx]) - t) <= MAX_ALIGN_TOLERANCE_SEC:
            rec = sorted_recs[best_idx]
            aligned.append(
                AlignedFrame(
                    source_type="jpg",
                    path=rec["path"],
                    timestamp_ms=rec["declared_ts"],
                    frame_key=("jpg", rec["seq"], str(rec["path"])),
                )
            )
        else:
            aligned.append(None)
    return aligned


def _align_frames_from_webm(
    csv_rows: list[dict[str, str]],
    rec_dir: Path,
) -> list[AlignedFrame | None] | None:
    webm_path = rec_dir / "media" / "camera.webm"
    if not webm_path.is_file():
        return None
    row_ts = [_parse_finite_float(r.get("timestamp")) for r in csv_rows]
    valid_ts = [t for t in row_ts if t is not None]
    if not valid_ts:
        return None
    base_ts = valid_ts[0] if valid_ts[0] > 1e9 else 0.0

    aligned: list[AlignedFrame | None] = []
    for idx, t in enumerate(row_ts):
        if t is None:
            aligned.append(None)
            continue
        rel_sec = max(0.0, t - base_ts)
        ts_ms = int(round(rel_sec * 1000.0))
        aligned.append(
            AlignedFrame(
                source_type="webm",
                path=webm_path,
                timestamp_ms=ts_ms,
                frame_key=("webm", idx, ts_ms),
            )
        )
    return aligned


def _align_session_frames(
    csv_rows: list[dict[str, str]],
    rec_dir: Path,
) -> tuple[list[AlignedFrame | None], str]:
    ledger = _load_ledger_records(rec_dir)
    from_ledger = _align_frames_from_ledger(csv_rows, ledger)
    if from_ledger is not None:
        return from_ledger, "retention_jpg"
    from_webm = _align_frames_from_webm(csv_rows, rec_dir)
    if from_webm is not None:
        return from_webm, "camera_webm"
    return [None] * len(csv_rows), "none"


def _read_aligned_bgr(
    frame: AlignedFrame,
    webm_cap: cv2.VideoCapture | None = None,
) -> np.ndarray | None:
    if frame.source_type == "jpg":
        return cv2.imread(str(frame.path))
    if frame.source_type == "webm":
        cap = webm_cap
        own_cap = False
        if cap is None:
            cap = cv2.VideoCapture(str(frame.path))
            own_cap = True
        try:
            if not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_POS_MSEC, float(frame.timestamp_ms))
            ok, bgr = cap.read()
            return bgr if ok and bgr is not None else None
        finally:
            if own_cap:
                cap.release()
    return None


def _landmarks_face_scale(landmarks: list[Any] | None) -> float | None:
    if not landmarks or len(landmarks) <= max(CHIN, FOREHEAD_TOP):
        return None
    try:
        top = landmarks[FOREHEAD_TOP]
        chin = landmarks[CHIN]
        pt_top = (float(top.x), float(top.y)) if hasattr(top, "x") else (float(top[0]), float(top[1]))
        pt_chin = (float(chin.x), float(chin.y)) if hasattr(chin, "x") else (float(chin[0]), float(chin[1]))
        val = float(dist.euclidean(pt_top, pt_chin))
        return val if math.isfinite(val) else None
    except Exception:
        return None


def _manifest_crop(logs_dir: Path, session_id: str) -> tuple[str, tuple[float, float, float, float] | None] | None:
    backfill_manifest = logs_dir / ".face_backfill_manifest.json"
    if backfill_manifest.is_file():
        try:
            data = json.loads(backfill_manifest.read_text(encoding="utf-8")) or {}
            entry = data.get(session_id)
            if isinstance(entry, dict) and entry.get("status") in ("backfilled", "rebuilt"):
                crop_name = str(entry.get("crop") or "full")
                box = entry.get("crop_box")
                if crop_name == "full" or box is None:
                    return ("full", None)
                if isinstance(box, (list, tuple)) and len(box) == 4:
                    return (crop_name, tuple(float(v) for v in box))
        except Exception:
            pass

    live_manifest = logs_dir / ".face_live_framing.json"
    if live_manifest.is_file():
        try:
            data = json.loads(live_manifest.read_text(encoding="utf-8")) or {}
            entry = data.get(session_id)
            if isinstance(entry, dict):
                crop_name = str(entry.get("crop") or "full")
                ladder = dict(CROP_LADDER)
                if crop_name in ladder:
                    return (crop_name, ladder[crop_name])
        except Exception:
            pass
    return None


def _score_crop_candidate(
    sample: list[tuple[AlignedFrame, float]],
    crop_box: tuple[float, float, float, float] | None,
    detector_factory: Callable[[], Any],
    webm_cap: cv2.VideoCapture | None = None,
) -> tuple[int, int]:
    """Return `(scale_matches, face_hits)` for a crop candidate on sampled detected rows."""
    det = detector_factory()
    prev_ts: int | None = None
    scale_matches = 0
    hits = 0
    try:
        for frame, csv_fs in sample:
            bgr = _read_aligned_bgr(frame, webm_cap=webm_cap)
            if bgr is None:
                continue
            cropped = apply_crop(bgr, crop_box)
            rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            ts_ms = frame.timestamp_ms
            if prev_ts is not None:
                if ts_ms < prev_ts:
                    det.close()
                    det = detector_factory()
                elif ts_ms == prev_ts:
                    ts_ms = prev_ts + 1
            prev_ts = ts_ms
            lm, _, _ = det.detect_full(rgb, ts_ms)
            if lm:
                hits += 1
                pred_fs = _landmarks_face_scale(lm)
                if pred_fs is not None and abs(pred_fs - csv_fs) <= 0.003:
                    scale_matches += 1
    finally:
        det.close()
    return scale_matches, hits


def _resolve_crop(
    session_id: str,
    logs_dir: Path,
    csv_rows: list[dict[str, str]],
    aligned_frames: list[AlignedFrame | None],
    detector_factory: Callable[[], Any],
    webm_cap: cv2.VideoCapture | None = None,
) -> tuple[str, tuple[float, float, float, float] | None]:
    manifest = _manifest_crop(logs_dir, session_id)
    if manifest is not None:
        return manifest

    sample: list[tuple[AlignedFrame, float]] = []
    for row, frame in zip(csv_rows, aligned_frames):
        if frame is None:
            continue
        fs = _parse_finite_float(row.get("face_scale"))
        if fs is not None and fs > 0.0:
            sample.append((frame, fs))
            if len(sample) >= 5:
                break

    if not sample:
        return ("full", None)

    full_score = _score_crop_candidate(sample, None, detector_factory, webm_cap=webm_cap)
    # Fast path: full-frame already detects all sampled rows or matches CSV face_scale.
    if full_score[0] > 0 and full_score[1] == len(sample):
        return ("full", None)
    if full_score[1] == len(sample):
        # All sampled rows detected under full frame; only switch if another crop strictly
        # matches `face_scale` (synthetic/mock detectors yield identical landmarks and stay full).
        best_name, best_box, best_score = "full", None, full_score
        for cname, cbox in CROP_LADDER[1:]:
            cand_score = _score_crop_candidate(sample, cbox, detector_factory, webm_cap=webm_cap)
            if cand_score[0] > best_score[0]:
                best_name, best_box, best_score = cname, cbox, cand_score
        return (best_name, best_box)

    best_name, best_box, best_score = "full", None, full_score
    for cname, cbox in CROP_LADDER[1:]:
        cand_score = _score_crop_candidate(sample, cbox, detector_factory, webm_cap=webm_cap)
        if cand_score > best_score and (
            cand_score[0] > 0 or (full_score[1] == 0 and cand_score[1] > full_score[1])
        ):
            best_name, best_box, best_score = cname, cbox, cand_score
    return (best_name, best_box)


def _verify_written_csv(
    tmp_path: Path,
    expected_header: list[str],
    expected_row_count: int,
) -> None:
    with open(tmp_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        got_header = list(reader.fieldnames or [])
        got_rows = list(reader)
    if got_header != expected_header:
        raise RuntimeError(
            f"Written header mismatch: got {len(got_header)} cols, expected {len(expected_header)}"
        )
    if len(got_rows) != expected_row_count:
        raise RuntimeError(
            f"Written row count mismatch: got {len(got_rows)}, expected {expected_row_count}"
        )


def backfill_session(
    session_id: str,
    *,
    logs_dir: Path = DEFAULT_LOGS_DIR,
    recordings_dir: Path = DEFAULT_RECORDINGS_DIR,
    backup_dir: Path = DEFAULT_BACKUP_DIR,
    force: bool = False,
    dry_run: bool = False,
    detector_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Backfill 3D head pose and frame quality columns onto one session's face CSV."""
    t0 = time.perf_counter()
    csv_path = logs_dir / f"face_au_log_{session_id}.csv"
    if not csv_path.is_file():
        return {"session_id": session_id, "status": "skip", "reason": "missing_csv"}

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        orig_header = list(reader.fieldnames or [])
        orig_rows = list(reader)

    if not orig_header:
        return {"session_id": session_id, "status": "skip", "reason": "empty_header"}

    if not force and _already_backfilled(orig_header, orig_rows):
        return {
            "session_id": session_id,
            "status": "skipped",
            "reason": "already_has_3d",
            "n_rows": len(orig_rows),
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        }

    out_header = _build_output_header(orig_header, session_id)
    orig_cols_without = [c for c in orig_header if c not in TELEMETRY_COLS]

    rec_dir = find_recording_dir(recordings_dir, session_id)
    if rec_dir is None:
        return {"session_id": session_id, "status": "skip", "reason": "missing_recording"}

    aligned_frames, source_kind = _align_session_frames(orig_rows, rec_dir)
    if source_kind == "none":
        return {"session_id": session_id, "status": "skip", "reason": "no_media_frames"}

    factory = detector_factory or _default_detector_factory

    webm_cap: cv2.VideoCapture | None = None
    if source_kind == "camera_webm":
        webm_path = rec_dir / "media" / "camera.webm"
        webm_cap = cv2.VideoCapture(str(webm_path))

    crop_name = "full"
    crop_box: tuple[float, float, float, float] | None = None
    detector: Any = None

    try:
        crop_name, crop_box = _resolve_crop(
            session_id,
            logs_dir,
            orig_rows,
            aligned_frames,
            factory,
            webm_cap=webm_cap,
        )

        detector = factory()
        prev_ts_ms: int | None = None
        prev_key: Any = None
        prev_flat: dict[str, Any] = {}

        out_rows: list[dict[str, Any]] = []
        n_aligned = 0
        n_pose_valid = 0
        n_quality_ok = 0

        for orig_row, frame in zip(orig_rows, aligned_frames):
            out_row: dict[str, Any] = {
                col: ("" if orig_row.get(col) is None else orig_row[col])
                for col in orig_cols_without
            }

            flat: dict[str, Any] = {}
            if frame is not None:
                bgr = _read_aligned_bgr(frame, webm_cap=webm_cap)
                if bgr is not None:
                    n_aligned += 1
                    if prev_key is not None and frame.frame_key == prev_key:
                        flat = dict(prev_flat)
                    else:
                        cropped = apply_crop(bgr, crop_box)
                        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                        ts_ms = int(frame.timestamp_ms)
                        if prev_ts_ms is not None:
                            if ts_ms < prev_ts_ms:
                                detector.close()
                                detector = factory()
                            elif ts_ms == prev_ts_ms:
                                ts_ms = prev_ts_ms + 1
                        prev_ts_ms = ts_ms

                        landmarks_norm, _, head_pose_3d = detector.detect_full(rgb, ts_ms)
                        frame_quality = assess_frame_quality(rgb, landmarks_norm)
                        flat = _extract_flat_telemetry(head_pose_3d, frame_quality)
                        prev_key = frame.frame_key
                        prev_flat = dict(flat)

            if _parse_finite_float(flat.get("head_yaw_3d")) is not None:
                n_pose_valid += 1
            if flat.get("frame_quality_ok") == 1:
                n_quality_ok += 1

            for col in TELEMETRY_COLS:
                val = flat.get(col)
                out_row[col] = "" if val is None else val

            out_rows.append(out_row)
    finally:
        if detector is not None:
            detector.close()
        if webm_cap is not None:
            webm_cap.release()

    n_rows = len(orig_rows)
    pose_valid_rate = round(n_pose_valid / n_rows, 4) if n_rows > 0 else 0.0
    quality_pass_rate = round(n_quality_ok / n_rows, 4) if n_rows > 0 else 0.0

    if dry_run:
        return {
            "session_id": session_id,
            "status": "dry_run",
            "source": source_kind,
            "crop": crop_name,
            "n_rows": n_rows,
            "n_aligned": n_aligned,
            "n_pose_valid": n_pose_valid,
            "pose_valid_rate": pose_valid_rate,
            "n_quality_ok": n_quality_ok,
            "quality_pass_rate": quality_pass_rate,
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        }

    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_file = backup_dir / csv_path.name
    if not backup_file.exists():
        shutil.copy2(csv_path, backup_file)

    tmp_path = logs_dir / f".tmp_backfill_{session_id}.csv"
    try:
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=out_header)
            writer.writeheader()
            writer.writerows(out_rows)
        _verify_written_csv(tmp_path, out_header, n_rows)
        os.replace(tmp_path, csv_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return {
        "session_id": session_id,
        "status": "backfilled",
        "source": source_kind,
        "crop": crop_name,
        "n_rows": n_rows,
        "n_aligned": n_aligned,
        "n_pose_valid": n_pose_valid,
        "pose_valid_rate": pose_valid_rate,
        "n_quality_ok": n_quality_ok,
        "quality_pass_rate": quality_pass_rate,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }


def run_backfill(
    sessions: Iterable[str],
    *,
    logs_dir: Path = DEFAULT_LOGS_DIR,
    recordings_dir: Path = DEFAULT_RECORDINGS_DIR,
    backup_dir: Path = DEFAULT_BACKUP_DIR,
    force: bool = False,
    dry_run: bool = False,
    detector_factory: Callable[[], Any] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for sid in sessions:
        res = backfill_session(
            sid,
            logs_dir=logs_dir,
            recordings_dir=recordings_dir,
            backup_dir=backup_dir,
            force=force,
            dry_run=dry_run,
            detector_factory=detector_factory,
        )
        results.append(res)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="将真实 3D 头姿与光学画面质量列回填至历史面部日志 CSV（纯 CPU，零 PII）"
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=None,
        help="待回填 session_id 列表（默认 5 场核心样例会话）",
    )
    parser.add_argument(
        "--all-matched",
        action="store_true",
        help="自动扫描 logs-dir 与 recordings-dir 匹配到的全部会话",
    )
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR, help="CSV 日志目录")
    parser.add_argument(
        "--recordings-dir", type=Path, default=DEFAULT_RECORDINGS_DIR, help="录制素材根目录"
    )
    parser.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR, help="原始 CSV 备份目录")
    parser.add_argument("--force", action="store_true", help="即使已存在有限 head_yaw_3d 值也强制重算")
    parser.add_argument("--dry-run", action="store_true", help="仅推理统计，不写入备份或覆盖 CSV")
    args = parser.parse_args(argv)

    if args.all_matched:
        sessions = discover_matched_sessions(args.logs_dir, args.recordings_dir)
    elif args.sessions:
        sessions = list(args.sessions)
    else:
        sessions = list(FLAGSHIP_SESSIONS)

    if not sessions:
        print("[jx_backfill_face_3d_telemetry] 未找到待处理会话。")
        return 0

    print(
        f"[jx_backfill_face_3d_telemetry] sessions={len(sessions)} "
        f"dry_run={args.dry_run} force={args.force}"
    )
    results = run_backfill(
        sessions,
        logs_dir=args.logs_dir,
        recordings_dir=args.recordings_dir,
        backup_dir=args.backup_dir,
        force=args.force,
        dry_run=args.dry_run,
    )
    for r in results:
        sid = r["session_id"]
        status = r["status"]
        if status in ("backfilled", "dry_run"):
            print(
                f"  - {sid}: status={status} crop={r['crop']} source={r['source']} "
                f"rows={r['n_rows']} aligned={r['n_aligned']} "
                f"pose_valid={r['n_pose_valid']}/{r['n_rows']} ({r['pose_valid_rate']:.1%}) "
                f"quality_ok={r['n_quality_ok']}/{r['n_rows']} ({r['quality_pass_rate']:.1%}) "
                f"elapsed={r['elapsed_sec']}s"
            )
        else:
            print(f"  - {sid}: status={status} reason={r.get('reason', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
