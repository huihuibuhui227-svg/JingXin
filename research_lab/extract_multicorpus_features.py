#!/usr/bin/env python3
"""Multiprocessing multimodal feature extractor across MIT Interview, RecruitView,
and ChaLearn First Impressions V2.

Extracts:
- Visual: evenly sampled frames -> MediaPipe FaceLandmarker (478-landmark geometry
  ratios, 52 ARKit-style blendshapes, 3D Euler head pose yaw/pitch/roll, frame
  brightness & Laplacian sharpness, plus blinking/kinematic variability stats).
- Acoustic: ffmpeg 16 kHz mono float32 -> pure NumPy+ORT Silero VAD (speech ratio,
  pause rate & duration stats) + Praat-Parselmouth (F0 pitch stats/IQR, voiced
  frame ratio, HNR, jitter, shimmer) + RMS energy stats.

Each CPU worker process initializes FaceLandmarker ONCE via ProcessPoolExecutor
``initializer`` to eliminate C++ / XNNPACK thread contention.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from face_expression.core.feature_extraction.au_calculator import AUFeatureCalculator
from face_expression.models.blendshapes import BLENDSHAPE_NAMES
from voice_interaction.core.feature_extraction import silero_vad_wrapper

DEFAULT_DATASETS_ROOT = Path("/home/zgy/huihuibui/DATASETS")
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "mediapipe" / "face_landmarker.task"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"

CHALEARN_TARGETS: tuple[str, ...] = (
    "extraversion",
    "neuroticism",
    "agreeableness",
    "conscientiousness",
    "openness",
    "interview",
)

RECRUITVIEW_TARGETS: tuple[str, ...] = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
    "overall_personality",
    "interview_score",
    "answer_score",
    "speaking_skills",
    "confidence_score",
    "facial_expression",
    "overall_performance",
)

KEY_GEOM_FEATURES: tuple[str, ...] = (
    "au4_frown",
    "au6_cheek_raise",
    "au7_eye_squeeze",
    "au9_nose_wrinkle",
    "au10_upper_lip_raise",
    "au12_smile",
    "au14_dimpler",
    "au15_mouth_down",
    "au20_lip_stretcher",
    "au23_lip_compression",
    "au25_mouth_open",
    "au26_jaw_drop",
    "au1_inner_brow_raise",
    "au2_outer_brow_raise",
    "avg_ear",
    "gaze_dev",
    "symmetry_score",
    "head_yaw",
    "head_pitch",
    "face_scale",
)

BILATERAL_BLENDSHAPE_GROUPS: dict[str, tuple[str, ...]] = {
    "bs_mouthSmile": ("mouthSmileLeft", "mouthSmileRight"),
    "bs_jawOpen": ("jawOpen",),
    "bs_eyeBlink": ("eyeBlinkLeft", "eyeBlinkRight"),
    "bs_browDown": ("browDownLeft", "browDownRight"),
    "bs_browInnerUp": ("browInnerUp",),
    "bs_mouthPress": ("mouthPressLeft", "mouthPressRight"),
    "bs_cheekSquint": ("cheekSquintLeft", "cheekSquintRight"),
    "bs_eyeSquint": ("eyeSquintLeft", "eyeSquintRight"),
    "bs_mouthFrown": ("mouthFrownLeft", "mouthFrownRight"),
    "bs_mouthDimple": ("mouthDimpleLeft", "mouthDimpleRight"),
    "bs_mouthStretch": ("mouthStretchLeft", "mouthStretchRight"),
    "bs_noseSneer": ("noseSneerLeft", "noseSneerRight"),
}

_WORKER_LANDMARKER: Any = None


@dataclass(frozen=True)
class ClipJob:
    corpus: str
    sample_id: str
    group_id: str
    split: str
    video_path: str
    audio_path: str
    targets: dict[str, float]


def _init_worker(model_path: str) -> None:
    """Initialize MediaPipe FaceLandmarker once per worker process."""
    global _WORKER_LANDMARKER
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    opts = vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    _WORKER_LANDMARKER = vision.FaceLandmarker.create_from_options(opts)


def rotation_matrix_to_euler_deg(mat: np.ndarray) -> tuple[float, float, float]:
    """Decompose 4x4 or 3x3 facial transformation matrix into (yaw_deg, pitch_deg, roll_deg)."""
    arr = np.asarray(mat, dtype=np.float64)
    if arr.shape[0] >= 3 and arr.shape[1] >= 3:
        R = arr[:3, :3]
    else:
        return (float("nan"), float("nan"), float("nan"))
    # Re-orthogonalize via SVD so minor numerical drift never fails Rotation.from_matrix
    u, _, vt = np.linalg.svd(R)
    r_orth = u @ vt
    if np.linalg.det(r_orth) < 0:
        u[:, -1] *= -1.0
        r_orth = u @ vt
    pitch_deg, yaw_deg, roll_deg = Rotation.from_matrix(r_orth).as_euler("xyz", degrees=True)
    return float(yaw_deg), float(pitch_deg), float(roll_deg)


def extract_visual_features_from_frames(
    frames_rgb: list[np.ndarray],
    landmarker: Any,
    frame_dt_sec: float = 1.0,
) -> dict[str, float]:
    """Extract MediaPipe 478 geometry, 52 blendshape, 3D pose, and image quality stats."""
    import mediapipe as mp

    n_sampled = len(frames_rgb)
    feats: dict[str, float] = {
        "n_sampled_frames": float(n_sampled),
        "n_valid_face_frames": 0.0,
        "face_valid_ratio": 0.0,
        "frame_dt_sec": float(max(frame_dt_sec, 0.0)),
    }

    if n_sampled == 0:
        return feats

    brightness_vals: list[float] = []
    sharpness_vals: list[float] = []
    geom_records: dict[str, list[float]] = {k: [] for k in KEY_GEOM_FEATURES}
    bs_records: dict[str, list[float]] = {name: [] for name in BLENDSHAPE_NAMES}
    yaw_list: list[float] = []
    pitch_list: list[float] = []
    roll_list: list[float] = []

    au_calc = AUFeatureCalculator()
    n_valid = 0

    for rgb in frames_rgb:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        brightness_vals.append(float(np.mean(gray)))
        sharpness_vals.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

        if landmarker is None:
            continue
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect(mp_img)
        if not res.face_landmarks:
            continue

        n_valid += 1
        lm_raw = res.face_landmarks[0]
        lm = [(float(pt.x), float(pt.y)) for pt in lm_raw]
        face_width = float(dist.euclidean(lm[234], lm[455]))
        face_outline_height = float(dist.euclidean(lm[10], lm[152]))
        if face_width <= 1e-6 or face_outline_height <= 1e-6:
            continue

        au = au_calc.calculate(lm, face_width, face_outline_height)
        au_dict = au.__dict__
        for k in KEY_GEOM_FEATURES:
            src_key = "gaze_deviation" if k == "gaze_dev" else k
            val = au_dict.get(src_key)
            if val is not None and np.isfinite(val):
                geom_records[k].append(float(val))

        if res.face_blendshapes:
            for cat in res.face_blendshapes[0]:
                if cat.category_name in bs_records:
                    bs_records[cat.category_name].append(float(cat.score))

        if res.facial_transformation_matrixes:
            y_d, p_d, r_d = rotation_matrix_to_euler_deg(res.facial_transformation_matrixes[0])
            if np.isfinite(y_d) and np.isfinite(p_d) and np.isfinite(r_d):
                yaw_list.append(y_d)
                pitch_list.append(p_d)
                roll_list.append(r_d)

    feats["n_valid_face_frames"] = float(n_valid)
    feats["face_valid_ratio"] = float(n_valid) / float(n_sampled)
    feats["brightness_mean"] = float(np.mean(brightness_vals)) if brightness_vals else float("nan")
    feats["brightness_std"] = float(np.std(brightness_vals)) if brightness_vals else float("nan")
    feats["sharpness_mean"] = float(np.mean(sharpness_vals)) if sharpness_vals else float("nan")

    for k, vals in geom_records.items():
        if vals:
            arr = np.asarray(vals, dtype=float)
            feats[f"{k}_mean"] = float(np.mean(arr))
            feats[f"{k}_std"] = float(np.std(arr))
        else:
            feats[f"{k}_mean"] = float("nan")
            feats[f"{k}_std"] = float("nan")

    # Blinking & mouth/head kinematic variability
    ear_vals = geom_records["avg_ear"]
    if ear_vals:
        ear_arr = np.asarray(ear_vals, dtype=float)
        feats["ear_low_ratio"] = float(np.mean(ear_arr < 0.20))
        feats["ear_range"] = float(np.max(ear_arr) - np.min(ear_arr))
    else:
        feats["ear_low_ratio"] = float("nan")
        feats["ear_range"] = float("nan")

    mouth_vals = geom_records["au25_mouth_open"]
    if mouth_vals:
        m_arr = np.asarray(mouth_vals, dtype=float)
        feats["mouth_open_range"] = float(np.max(m_arr) - np.min(m_arr))
    else:
        feats["mouth_open_range"] = float("nan")

    hy_arr = np.asarray(geom_records["head_yaw"], dtype=float)
    hp_arr = np.asarray(geom_records["head_pitch"], dtype=float)
    if len(hy_arr) >= 2 and len(hp_arr) >= 2:
        step = 0.5 * (float(np.mean(np.abs(np.diff(hy_arr)))) + float(np.mean(np.abs(np.diff(hp_arr)))))
        feats["head_motion_step"] = step
        feats["head_motion_rate"] = step / max(float(frame_dt_sec), 0.25)
    else:
        feats["head_motion_step"] = float("nan")
        feats["head_motion_rate"] = float("nan")

    # All 52 blendshape means
    for name in BLENDSHAPE_NAMES:
        vals = bs_records[name]
        feats[f"bs_{name}_mean"] = float(np.mean(vals)) if vals else float("nan")

    # Bilateral grouped blendshape mean / std / max
    for group_name, members in BILATERAL_BLENDSHAPE_GROUPS.items():
        member_lists = [bs_records[m] for m in members if bs_records[m]]
        if member_lists and all(len(lst) == len(member_lists[0]) for lst in member_lists):
            avg_series = np.mean(np.vstack(member_lists), axis=0)
            feats[f"{group_name}_mean"] = float(np.mean(avg_series))
            feats[f"{group_name}_std"] = float(np.std(avg_series))
            feats[f"{group_name}_max"] = float(np.max(avg_series))
        else:
            feats[f"{group_name}_mean"] = float("nan")
            feats[f"{group_name}_std"] = float("nan")
            feats[f"{group_name}_max"] = float("nan")

    # 3D Euler angles
    for axis_name, vals in (("yaw_deg", yaw_list), ("pitch_deg", pitch_list), ("roll_deg", roll_list)):
        if vals:
            arr = np.asarray(vals, dtype=float)
            feats[f"{axis_name}_mean"] = float(np.mean(arr))
            feats[f"{axis_name}_std"] = float(np.std(arr))
            feats[f"{axis_name}_range"] = float(np.max(arr) - np.min(arr))
        else:
            feats[f"{axis_name}_mean"] = float("nan")
            feats[f"{axis_name}_std"] = float("nan")
            feats[f"{axis_name}_range"] = float("nan")

    return feats


def extract_acoustic_features_from_array(audio: np.ndarray, sr: int = 16000) -> dict[str, float | str]:
    """Extract Silero VAD + Parselmouth pitch/HNR/jitter/shimmer + RMS energy stats."""
    import parselmouth

    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    dur_sec = float(len(arr)) / float(sr) if sr > 0 else 0.0
    out: dict[str, float | str] = {
        "duration_sec": dur_sec,
        "vad_backend": "none",
        "speech_ratio": float("nan"),
        "pause_count": float("nan"),
        "pause_rate_per_sec": float("nan"),
        "pause_rate_per_min": float("nan"),
        "pause_duration_mean": float("nan"),
        "pause_duration_p50": float("nan"),
        "rms_mean": float("nan"),
        "rms_std": float("nan"),
        "rms_db_mean": float("nan"),
        "rms_db_std": float("nan"),
        "pitch_mean": float("nan"),
        "pitch_std": float("nan"),
        "pitch_p10": float("nan"),
        "pitch_p90": float("nan"),
        "pitch_iqr": float("nan"),
        "voiced_frame_ratio": float("nan"),
        "hnr_db": float("nan"),
        "jitter_local": float("nan"),
        "shimmer_local": float("nan"),
    }

    if len(arr) < int(0.1 * sr) or float(np.max(np.abs(arr))) < 1e-7:
        return out

    # Frame-level RMS over 32ms windows, 16ms hop
    win = int(0.032 * sr)
    hop = int(0.016 * sr)
    if len(arr) >= win:
        n_steps = 1 + (len(arr) - win) // hop
        rms_vals = np.array(
            [np.sqrt(np.mean(arr[i * hop : i * hop + win] ** 2)) for i in range(n_steps)],
            dtype=float,
        )
    else:
        rms_vals = np.array([float(np.sqrt(np.mean(arr ** 2)))], dtype=float)
    rms_db = 20.0 * np.log10(np.maximum(rms_vals, 1e-7))
    out["rms_mean"] = float(np.mean(rms_vals))
    out["rms_std"] = float(np.std(rms_vals))
    out["rms_db_mean"] = float(np.mean(rms_db))
    out["rms_db_std"] = float(np.std(rms_db))

    # Silero VAD
    vad = silero_vad_wrapper.detect_speech_segments(arr, sr=sr)
    out["vad_backend"] = str(vad.get("vad_backend", "unknown"))
    out["speech_ratio"] = float(vad.get("speech_ratio", 0.0))
    pauses = [float(p) for p in vad.get("pause_durations_sec", []) if np.isfinite(p)]
    out["pause_count"] = float(len(pauses))
    out["pause_rate_per_sec"] = float(len(pauses)) / max(dur_sec, 1e-6)
    out["pause_rate_per_min"] = float(len(pauses)) / max(dur_sec / 60.0, 1e-6)
    if pauses:
        p_arr = np.asarray(pauses, dtype=float)
        out["pause_duration_mean"] = float(np.mean(p_arr))
        out["pause_duration_p50"] = float(np.median(p_arr))
    else:
        out["pause_duration_mean"] = 0.0
        out["pause_duration_p50"] = 0.0

    # Praat Parselmouth pitch, HNR, jitter, shimmer
    # Cap Praat acoustic window at 120s for ultra-long recordings while keeping Silero full-length
    praat_arr = arr[: min(len(arr), 120 * sr)].astype(np.float64)
    try:
        snd = parselmouth.Sound(praat_arr, sampling_frequency=float(sr))
        pitch = snd.to_pitch(time_step=0.02, pitch_floor=65.0, pitch_ceiling=500.0)
        freq = np.asarray(pitch.selected_array["frequency"], dtype=float)
        voiced_mask = freq > 0.0
        out["voiced_frame_ratio"] = float(np.mean(voiced_mask)) if len(freq) > 0 else 0.0
        f0_voiced = freq[voiced_mask]
        if len(f0_voiced) >= 3:
            out["pitch_mean"] = float(np.mean(f0_voiced))
            out["pitch_std"] = float(np.std(f0_voiced))
            out["pitch_p10"] = float(np.percentile(f0_voiced, 10))
            out["pitch_p90"] = float(np.percentile(f0_voiced, 90))
            out["pitch_iqr"] = float(np.percentile(f0_voiced, 75) - np.percentile(f0_voiced, 25))

        try:
            harm = snd.to_harmonicity_cc()
            hnr_vals = harm.values[harm.values > -200.0]
            if hnr_vals.size > 0:
                out["hnr_db"] = float(np.mean(hnr_vals))
        except Exception:
            pass

        try:
            pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 65.0, 500.0)
            jit = float(parselmouth.praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
            shim = float(parselmouth.praat.call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
            if np.isfinite(jit):
                out["jitter_local"] = jit
            if np.isfinite(shim):
                out["shimmer_local"] = shim
        except Exception:
            pass
    except Exception:
        pass

    return out


def _sample_video_frames(video_path: str, n_frames: int) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 1.0 or not np.isfinite(fps):
        fps = 30.0
    if total_frames <= 0:
        cap.release()
        return [], 0.0

    # Sample evenly across [5%, 95%] of clip to avoid fade-in/out black frames
    lo = max(0, int(round(0.05 * (total_frames - 1))))
    hi = max(lo, int(round(0.95 * (total_frames - 1))))
    idxs = np.unique(np.linspace(lo, hi, num=max(1, n_frames), dtype=int))
    frames_rgb: list[np.ndarray] = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if ok and bgr is not None:
            frames_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    dur_sec = float(total_frames) / fps
    dt = (dur_sec * 0.90) / max(1, len(idxs) - 1)
    return frames_rgb, dt


def _decode_audio_ffmpeg(audio_path: str, sr: int = 16000) -> np.ndarray:
    cmd = [
        "ffmpeg", "-y", "-threads", "1", "-i", audio_path,
        "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-loglevel", "error", "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False, timeout=45)
        if proc.returncode != 0 or not proc.stdout:
            return np.zeros(0, dtype=np.float32)
        return np.frombuffer(proc.stdout, dtype=np.float32).copy()
    except Exception:
        return np.zeros(0, dtype=np.float32)


def _process_clip_job(args: tuple[ClipJob, int]) -> dict[str, Any]:
    job, n_frames = args
    frames_rgb, dt = _sample_video_frames(job.video_path, n_frames=n_frames)
    vis_feats = extract_visual_features_from_frames(frames_rgb, _WORKER_LANDMARKER, frame_dt_sec=dt)
    audio = _decode_audio_ffmpeg(job.audio_path, sr=16000)
    aud_feats = extract_acoustic_features_from_array(audio, sr=16000)

    row: dict[str, Any] = {
        "corpus": job.corpus,
        "sample_id": job.sample_id,
        "group_id": job.group_id,
        "split": job.split,
    }
    row.update(vis_feats)
    row.update(aud_feats)
    for t_name, t_val in job.targets.items():
        row[f"target_{t_name}"] = float(t_val)
    return row


def build_mit_jobs(datasets_root: Path, limit: int | None = None) -> list[ClipJob]:
    mit_root = datasets_root / "MIT_INTERVIEW_DATASET"
    ts_df = pd.read_csv(mit_root / "Labels" / "turker_scores_full_interview.csv")
    target_cols = [c for c in ts_df.columns if c not in ("Participant", "Worker", "Total")]
    ts_df["key"] = ts_df["Participant"].astype(str).str.lower()
    agg = ts_df.groupby("key")[target_cols].mean()

    vids = sorted((mit_root / "Videos").glob("*.avi"))
    jobs: list[ClipJob] = []
    for vpath in vids:
        key = vpath.stem.lower()
        if key not in agg.index:
            continue
        pid = str(int(re.search(r"\d+", key).group()))
        wav_path = mit_root / "Audio" / f"{vpath.stem}.wav"
        audio_path = str(wav_path if wav_path.exists() else vpath)
        targets = {c: float(agg.loc[key, c]) for c in target_cols}
        jobs.append(ClipJob(
            corpus="mit",
            sample_id=key,
            group_id=pid,
            split="all",
            video_path=str(vpath),
            audio_path=audio_path,
            targets=targets,
        ))
    if limit is not None and limit < len(jobs):
        # Keep paired pre/post sessions for sampled participants first
        by_pid: dict[str, list[ClipJob]] = {}
        for j in jobs:
            by_pid.setdefault(j.group_id, []).append(j)
        ordered: list[ClipJob] = []
        for pid in sorted(by_pid, key=lambda x: int(x)):
            ordered.extend(by_pid[pid])
        jobs = ordered[:limit]
    return jobs


def build_recruitview_jobs(
    datasets_root: Path,
    limit: int | None = 160,
    seed: int = 42,
) -> list[ClipJob]:
    rv_root = datasets_root / "RecruitView"
    rows = [json.loads(line) for line in open(rv_root / "metadata.jsonl", encoding="utf-8")]
    df = pd.DataFrame(rows)

    if limit is not None and limit < len(df):
        # Stratified deterministic selection across users sorted by overall_personality quantile
        # Aim for ~2 clips per user across many distinct users for clean 5-fold GroupKFold
        user_stats = (
            df.groupby("user_no", sort=True)
            .agg(n=("id", "count"), op=("overall_personality", "mean"))
            .reset_index()
        )
        multi = user_stats[user_stats["n"] >= 2].sort_values(["op", "user_no"]).reset_index(drop=True)
        n_users_needed = min(len(multi), max(1, (limit + 1) // 2))
        sel_user_idx = np.linspace(0, len(multi) - 1, num=n_users_needed, dtype=int)
        sel_users = list(dict.fromkeys(multi.iloc[sel_user_idx]["user_no"].tolist()))

        rng = np.random.RandomState(seed)
        selected_indices: list[int] = []
        for u in sel_users:
            u_df = df[df["user_no"] == u].sort_values("id")
            take = min(2, len(u_df))
            picks = rng.choice(u_df.index.to_numpy(), size=take, replace=False)
            selected_indices.extend(sorted(picks.tolist()))

        if len(selected_indices) < limit:
            remaining = [idx for idx in df.index if idx not in set(selected_indices)]
            extra = rng.choice(remaining, size=min(limit - len(selected_indices), len(remaining)), replace=False)
            selected_indices.extend(sorted(extra.tolist()))
        df = df.loc[selected_indices[:limit]].sort_values("id")

    jobs: list[ClipJob] = []
    for _, r in df.iterrows():
        vpath = rv_root / str(r["file_name"])
        if not vpath.exists():
            continue
        targets = {t: float(r[t]) for t in RECRUITVIEW_TARGETS}
        jobs.append(ClipJob(
            corpus="recruitview",
            sample_id=str(r["id"]),
            group_id=str(r["user_no"]),
            split="all",
            video_path=str(vpath),
            audio_path=str(vpath),
            targets=targets,
        ))
    return jobs


def build_chalearn_jobs(
    datasets_root: Path,
    split_limits: dict[str, int | None],
    seed: int = 42,
) -> list[ClipJob]:
    fi_root = datasets_root / "first-impressions"
    ann_paths = {
        "train": fi_root / "annotations" / "train-annotation" / "annotation_training.pkl",
        "val": fi_root / "annotations" / "annotation_validation.pkl",
        "test": fi_root / "annotations" / "annotation_test.pkl",
    }
    jobs: list[ClipJob] = []
    rng = np.random.RandomState(seed)

    for split in ("train", "val", "test"):
        with open(ann_paths[split], "rb") as f:
            ann = pickle.load(f, encoding="latin1")
        all_vids = sorted(ann["extraversion"].keys())
        lim = split_limits.get(split)
        if lim is not None and lim < len(all_vids):
            # Maximize speaker diversity by sampling distinct YouTube video IDs first
            by_yt: dict[str, list[str]] = {}
            for v in all_vids:
                yt_id = v.split(".")[0]
                by_yt.setdefault(yt_id, []).append(v)
            yt_keys = sorted(by_yt.keys())
            rng.shuffle(yt_keys)
            chosen: list[str] = []
            for yt in yt_keys:
                chosen.append(sorted(by_yt[yt])[0])
                if len(chosen) >= lim:
                    break
            if len(chosen) < lim:
                chosen_set = set(chosen)
                rest = [v for v in all_vids if v not in chosen_set]
                rng.shuffle(rest)
                chosen.extend(rest[: lim - len(chosen)])
            all_vids = sorted(chosen)

        for vid in all_vids:
            vpath = fi_root / split / vid
            if not vpath.exists():
                continue
            targets = {t: float(ann[t][vid]) for t in CHALEARN_TARGETS}
            yt_id = vid.split(".")[0]
            jobs.append(ClipJob(
                corpus="chalearn",
                sample_id=vid,
                group_id=yt_id,
                split=split,
                video_path=str(vpath),
                audio_path=str(vpath),
                targets=targets,
            ))
    return jobs


def _save_corpus_cache(df: pd.DataFrame, cache_dir: Path, corpus_name: str) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / f"{corpus_name}_multimodal_features.csv"
    npz_path = cache_dir / f"{corpus_name}_multimodal_features.npz"
    df.to_csv(csv_path, index=False)

    meta_cols = ["corpus", "sample_id", "group_id", "split", "vad_backend"]
    target_cols = [c for c in df.columns if c.startswith("target_")]
    feat_cols = [c for c in df.columns if c not in set(meta_cols) and c not in set(target_cols)]
    X = df[feat_cols].to_numpy(dtype=np.float32) if feat_cols else np.empty((len(df), 0), dtype=np.float32)
    Y = df[target_cols].to_numpy(dtype=np.float32) if target_cols else np.empty((len(df), 0), dtype=np.float32)
    np.savez_compressed(
        npz_path,
        X=X,
        Y=Y,
        feature_names=np.asarray(feat_cols, dtype=object),
        target_names=np.asarray(target_cols, dtype=object),
        sample_ids=df["sample_id"].astype(str).to_numpy(dtype=object),
        group_ids=df["group_id"].astype(str).to_numpy(dtype=object),
        splits=df["split"].astype(str).to_numpy(dtype=object),
    )
    return {
        "n_clips": int(len(df)),
        "n_features": int(len(feat_cols)),
        "n_targets": int(len(target_cols)),
        "csv_path": str(csv_path),
        "npz_path": str(npz_path),
        "mean_face_valid_ratio": float(df["face_valid_ratio"].mean()) if len(df) else 0.0,
        "silero_onnx_count": int((df["vad_backend"] == "silero_vad_onnx").sum()) if len(df) else 0,
    }


def run_extraction(
    *,
    datasets_root: Path = DEFAULT_DATASETS_ROOT,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    model_path: Path = DEFAULT_MODEL_PATH,
    recruitview_limit: int | None = 160,
    chalearn_split_limits: dict[str, int | None] | None = None,
    mit_limit: int | None = 138,
    frames_per_clip: int = 8,
    workers: int = 10,
    datasets: tuple[str, ...] = ("mit", "recruitview", "chalearn"),
) -> dict[str, Any]:
    if chalearn_split_limits is None:
        chalearn_split_limits = {"train": 180, "val": 60, "test": 60}

    jobs_by_corpus: dict[str, list[ClipJob]] = {}
    if "mit" in datasets:
        jobs_by_corpus["mit"] = build_mit_jobs(datasets_root, limit=mit_limit)
    if "recruitview" in datasets:
        jobs_by_corpus["recruitview"] = build_recruitview_jobs(datasets_root, limit=recruitview_limit)
    if "chalearn" in datasets:
        jobs_by_corpus["chalearn"] = build_chalearn_jobs(datasets_root, split_limits=chalearn_split_limits)

    all_jobs = [j for c in datasets for j in jobs_by_corpus.get(c, [])]
    if not all_jobs:
        raise RuntimeError("No clips scheduled for extraction.")

    print(
        f"Scheduled {len(all_jobs)} clips across {list(jobs_by_corpus.keys())} "
        f"(workers={workers}, frames_per_clip={frames_per_clip})",
        flush=True,
    )

    t0 = time.perf_counter()
    rows_by_corpus: dict[str, list[dict[str, Any]]] = {c: [] for c in jobs_by_corpus}
    failed = 0

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(model_path),),
    ) as ex:
        futs = [ex.submit(_process_clip_job, (job, frames_per_clip)) for job in all_jobs]
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                res = fut.result()
                rows_by_corpus[res["corpus"]].append(res)
            except Exception as exc:
                failed += 1
                print(f"[warn] clip job failed: {exc}", flush=True)
            if done % 50 == 0 or done == len(futs):
                el = time.perf_counter() - t0
                print(f"  progress: {done}/{len(futs)} ({done / max(el, 1e-6):.2f} clips/sec)", flush=True)

    elapsed = time.perf_counter() - t0
    summary: dict[str, Any] = {
        "elapsed_sec": round(elapsed, 2),
        "total_clips_scheduled": len(all_jobs),
        "total_clips_succeeded": sum(len(v) for v in rows_by_corpus.values()),
        "total_clips_failed": failed,
        "clips_per_sec": round(len(all_jobs) / max(elapsed, 1e-6), 3),
        "workers": int(workers),
        "frames_per_clip": int(frames_per_clip),
        "corpora": {},
    }

    for corpus_name, rows in rows_by_corpus.items():
        if not rows:
            continue
        df = pd.DataFrame(rows).sort_values(["split", "sample_id"]).reset_index(drop=True)
        summary["corpora"][corpus_name] = _save_corpus_cache(df, cache_dir, corpus_name)

    bench_path = cache_dir / "extraction_benchmark.json"
    bench_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Extraction finished in {elapsed:.1f}s ({summary['clips_per_sec']} clips/s) -> {cache_dir}", flush=True)
    return summary


def _parse_split_limits(spec: str) -> dict[str, int | None]:
    out: dict[str, int | None] = {"train": 180, "val": 60, "test": 60}
    for part in spec.split(","):
        if "=" not in part:
            continue
        k, v = part.strip().split("=", 1)
        out[k.strip()] = None if v.strip().lower() in ("all", "none", "-1") else int(v.strip())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-corpus multimodal feature extractor")
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--recruitview-limit", type=int, default=160)
    parser.add_argument("--chalearn-split-limits", type=str, default="train=180,val=60,test=60")
    parser.add_argument("--mit-limit", type=int, default=138)
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--datasets", type=str, default="mit,recruitview,chalearn")
    parser.add_argument("--full", action="store_true", help="Extract all clips in every corpus")
    args = parser.parse_args()

    if args.full:
        rv_lim = None
        ch_lims = {"train": None, "val": None, "test": None}
        mit_lim = None
    else:
        rv_lim = args.recruitview_limit
        ch_lims = _parse_split_limits(args.chalearn_split_limits)
        mit_lim = args.mit_limit

    ds_tuple = tuple(s.strip().lower() for s in args.datasets.split(",") if s.strip())
    run_extraction(
        datasets_root=args.datasets_root,
        cache_dir=args.cache_dir,
        model_path=args.model_path,
        recruitview_limit=rv_lim,
        chalearn_split_limits=ch_lims,
        mit_limit=mit_lim,
        frames_per_clip=args.frames_per_clip,
        workers=args.workers,
        datasets=ds_tuple,
    )


if __name__ == "__main__":
    main()
