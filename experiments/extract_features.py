#!/usr/bin/env python
"""
批量特征提取脚本 — CueCoT 实验阶段一
===========================================
对 RecruitView 数据集的 2011 个 mp4 视频，逐帧提取：
  - 面部特征（AU + gaze + head pose + 情绪 + 紧张度 + 时间统计）
  - 手势特征（Hand + Arm + Shoulder + UpperBody）
  - 语音特征（pitch / energy / speech ratio / pause）

用法：
  python extract_features.py                   # 处理全部视频（断点续跑）
  python extract_features.py --test 10         # 仅测试前 10 个视频
  python extract_features.py --workers 4       # 4 进程并行
  python extract_features.py --video vid_0042  # 单视频调试
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from collections import deque
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

# ---------- 将 jingxin 根目录加入 sys.path ----------
_JINGXIN_ROOT = Path(__file__).resolve().parent.parent
if str(_JINGXIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_JINGXIN_ROOT))

# ---------- 项目配置 ----------
VIDEO_DIR = Path(r"D:\RecruitView\RecruitView\videos")
METADATA_PATH = Path(r"D:\RecruitView\RecruitView\metadata.jsonl")
OUTPUT_BASE = Path(__file__).resolve().parent / "results" / "features"
PROGRESS_FILE = OUTPUT_BASE / "progress.json"
MAX_VIDEO_DURATION_SEC = 300  # 超过 5 分钟的视频只处理前 5 分钟
FRAME_SKIP = 3                # 每 3 帧取 1 帧（≈10 fps @ 30fps 源）
FACE_CONFIDENCE = 0.7         # MediaPipe FaceMesh 置信度阈值


def offline_timestamp_ms(k: int, frame_skip: int, src_fps: float) -> int:
    """离线路径的时间戳:第 k 个**提交**帧距视频开头的毫秒数。

    为什么是 `frame_skip / src_fps` 而不是 `1 / src_fps`:提交的是每 `frame_skip`
    帧里的第 1 帧,所以相邻两个提交帧在**视频时间**上相隔 `frame_skip / src_fps` 秒。
    旧代码把 `fps=30` 直接传给管线,等于声称每个提交帧相隔 33 ms,真实是 100 ms
    —— 时间量整体错 3 倍(M2.5 spec §3.2)。

    `src_fps` 为 0 / NaN 时回退 30:容器读不出帧率是常见情形(实测
    `cap.get(cv2.CAP_PROP_FPS)` 会返回 0),而除零或 NaN 会污染整列时间戳。
    """
    if not src_fps or src_fps != src_fps:      # 0 / NaN 都回退
        src_fps = 30.0
    return int(round(k * frame_skip * 1000.0 / src_fps))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════
# 音频提取
# ═══════════════════════════════════════════════════════

def extract_audio(video_path: Path, out_wav: Path) -> bool:
    """用 ffmpeg 从视频提取单声道 16kHz WAV。成功返回 True。"""
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                "-loglevel", "error",
                str(out_wav),
            ],
            check=True, timeout=120,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return out_wav.exists() and out_wav.stat().st_size > 0
    except Exception:
        return False


# ═══════════════════════════════════════════════════════
# 面部特征提取器（复用 VideoPipeline）
# ═══════════════════════════════════════════════════════

class FaceExtractor:
    """轻量封装：重用 face_expression 的 VideoPipeline，只关心逐帧输出。"""

    def __init__(self):
        from face_expression.pipeline.video_pipeline import VideoPipeline
        # M2.5:`fps` 已不是计时依据(时间由调用方按 offline_timestamp_ms 给)。
        self.pipeline = VideoPipeline(session_id="batch", save_landmarks=False)

    def process_frame(self, rgb: np.ndarray, timestamp_ms: int):
        """返回 AnalysisFrameResult 或 None。时间戳由主循环按帧序号算好传进来。"""
        result, _, _ = self.pipeline.process_frame(rgb, timestamp_ms)
        return result

    def frame_to_row(self, result) -> dict:
        """将 VideoPipeline 结果序列化为扁平字典（≈182 列）。"""
        return result.to_dict() if result else {}


# ═══════════════════════════════════════════════════════
# 手势特征提取器（复用 gesture_analysis 底层分析器）
# ═══════════════════════════════════════════════════════

class GestureExtractor:
    """每帧运行 MediaPipe Hands + Pose，把结果喂给分析器。"""

    def __init__(self):
        import mediapipe as mp
        from gesture_analysis.core.analysis.hand_analyzer import HandAnalyzer
        from gesture_analysis.core.analysis.arm_analyzer import ArmAnalyzer
        from gesture_analysis.core.analysis.shoulder_analyzer import ShoulderAnalyzer
        from gesture_analysis.core.analysis.upper_body_analyzer import UpperBodyAnalyzer

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=0.7, min_tracking_confidence=0.5,
        )
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.6, min_tracking_confidence=0.6,
        )
        self.left_hand = HandAnalyzer(hand_id=0)
        self.right_hand = HandAnalyzer(hand_id=1)
        self.left_arm = ArmAnalyzer(arm_id="left")
        self.right_arm = ArmAnalyzer(arm_id="right")
        self.shoulder = ShoulderAnalyzer()
        self.upper_body = UpperBodyAnalyzer()
        self._all_analyzers = [
            self.left_hand, self.right_hand,
            self.left_arm, self.right_arm,
            self.shoulder, self.upper_body,
        ]

    def process_frame(self, rgb: np.ndarray, timestamp_ms: int) -> dict:
        """返回当前帧的手势特征扁平字典（或空 dict）。

        ⚠️ `timestamp_ms` 目前**收下但用不上** —— 本类走的是 `mp.solutions.hands/pode`
        直连(`__init__` 里),而那套 API 在 mediapipe 1.0.0 已被整个删除
        (实测 `hasattr(mp,'solutions') == False`)。也就是说**这个类在本环境根本构造不起来**,
        离线手势线当前不可跑。M1.5 只迁了活服务(:8000/:8002),漏了这个离线文件。
        把签名补齐是为了让文件内部一致(调用方统一带时间戳),**不是**声称它能跑。
        迁移它属于 M1.5 的漏项,已记进账本 deferred,不在 M2.5 范围内。
        """
        rgb.flags.writeable = False
        hand_res = self.hands.process(rgb)
        pose_res = self.pose.process(rgb)
        rgb.flags.writeable = True

        # 更新手部分析器
        if hand_res and hand_res.multi_hand_landmarks and hand_res.multi_handedness:
            for idx, (lm, handedness) in enumerate(
                zip(hand_res.multi_hand_landmarks, hand_res.multi_handedness)
            ):
                if idx >= 2:
                    break
                label = handedness.classification[0].label
                if label == "Left":
                    self.left_hand.update(lm.landmark)
                else:
                    self.right_hand.update(lm.landmark)
        else:
            self.left_hand.update(None)
            self.right_hand.update(None)

        # 更新姿态分析器
        if pose_res and pose_res.pose_landmarks:
            pose_lm = pose_res.pose_landmarks.landmark
            self.left_arm.update(pose_lm)
            self.right_arm.update(pose_lm)
            self.shoulder.update(pose_lm)
            self.upper_body.update(pose_lm)
        else:
            for a in [self.left_arm, self.right_arm, self.shoulder, self.upper_body]:
                a.update(None)

        return self._collect_results()

    def _collect_results(self) -> dict:
        row = {}
        # 左手
        lh = self.left_hand.get_results()
        for k, v in lh.items():
            row[f"left_hand_{k}"] = v
        # 右手
        rh = self.right_hand.get_results()
        for k, v in rh.items():
            row[f"right_hand_{k}"] = v
        # 左臂
        la = self.left_arm.get_results()
        for k, v in la.items():
            row[f"left_arm_{k}"] = v
        # 右臂
        ra = self.right_arm.get_results()
        for k, v in ra.items():
            row[f"right_arm_{k}"] = v
        # 肩部
        sh = self.shoulder.get_results()
        for k, v in sh.items():
            row[f"shoulder_{k}"] = v
        # 上肢
        ub = self.upper_body.get_results()
        for k, v in ub.items():
            row[f"upper_body_{k}"] = v
        return row

    def reset(self):
        for a in self._all_analyzers:
            a.reset()

    def close(self):
        self.hands.close()
        self.pose.close()


# ═══════════════════════════════════════════════════════
# 语音特征提取器
# ═══════════════════════════════════════════════════════

class VoiceExtractor:
    """从 WAV 文件提取韵律特征。"""

    def __init__(self):
        from voice_interaction.core.feature_extraction.prosody_extractor import (
            ProsodyFeatureExtractor,
        )
        self.extractor = ProsodyFeatureExtractor(sample_rate=16000)

    def extract_from_wav(self, wav_path: Path) -> dict:
        import librosa
        try:
            audio, sr = librosa.load(str(wav_path), sr=16000, mono=True)
            if len(audio) == 0:
                return {}
            feats = self.extractor.extract_all_features(audio)
            return {k: v for k, v in feats.items() if isinstance(v, (int, float))}
        except Exception:
            return {}


# ═══════════════════════════════════════════════════════
# 单视频处理函数（每个 worker 进程独立调用）
# ═══════════════════════════════════════════════════════

def process_one_video(args: tuple) -> dict:
    """处理单个视频，返回统计信息字典。"""
    video_path_str, output_dir_str, metadata_entry = args
    video_path = Path(video_path_str)
    out_dir = Path(output_dir_str)
    vid = video_path.stem

    try:
        return _process_one_video_impl(video_path, out_dir, vid, metadata_entry)
    except Exception as e:
        return {"video_id": vid, "status": "crash", "error": str(e)}


def _process_one_video_impl(video_path, out_dir, vid, metadata_entry) -> dict:

    out_dir.mkdir(parents=True, exist_ok=True)

    # 子进程独立初始化 MediaPipe / 分析器
    try:
        face_ext = FaceExtractor()
        ges_ext = GestureExtractor()
        voice_ext = VoiceExtractor()
    except Exception as e:
        return {"video_id": vid, "status": "init_error", "error": str(e)}

    # --- 音频处理 ---
    wav_tmp = out_dir / f"{vid}_audio.wav"
    voice_rows = []
    audio_ok = extract_audio(video_path, wav_tmp)
    if audio_ok:
        voice_feats = voice_ext.extract_from_wav(wav_tmp)
        if voice_feats:
            voice_rows = [voice_feats]
        try:
            wav_tmp.unlink()
        except OSError:
            pass

    # --- 视频处理 ---
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        ges_ext.close()
        return {"video_id": vid, "status": "open_error"}

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(min(total_frames, MAX_VIDEO_DURATION_SEC * src_fps))

    face_rows: list[dict] = []
    gesture_rows: list[dict] = []
    skipped_face = 0
    skipped_gesture = 0
    frame_idx = 0

    try:
        k = 0                       # 已**提交**的帧序号(M2.5:离线时间戳按它算,不是 frame_idx)
        while frame_idx < max_frames:
            ret, bgr = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_SKIP == 0:
                ts_ms = offline_timestamp_ms(k, FRAME_SKIP, src_fps)
                k += 1
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                # 面部
                try:
                    fr = face_ext.process_frame(rgb, ts_ms)
                    if fr is not None:
                        face_rows.append(face_ext.frame_to_row(fr))
                    else:
                        skipped_face += 1
                except Exception:
                    skipped_face += 1

                # 手势
                try:
                    gs = ges_ext.process_frame(rgb, ts_ms)
                    if gs:
                        gesture_rows.append(gs)
                    else:
                        skipped_gesture += 1
                except Exception:
                    skipped_gesture += 1

            frame_idx += 1
    except Exception as e:
        # MediaPipe 偶发崩溃：保存已处理帧，标记为 partial
        log(f"  !! {vid} crashed at frame {frame_idx}/{max_frames}: {e}")

    cap.release()
    ges_ext.close()

    # --- 写入 CSV ---
    _write_csv(out_dir / "face.csv", face_rows)
    _write_csv(out_dir / "gesture.csv", gesture_rows)
    if voice_rows:
        _write_csv(out_dir / "voice.csv", voice_rows)

    # --- 写入 metadata.json ---
    meta_out = dict(metadata_entry)
    meta_out["_face_frames"] = len(face_rows)
    meta_out["_gesture_frames"] = len(gesture_rows)
    meta_out["_skipped_face"] = skipped_face
    meta_out["_skipped_gesture"] = skipped_gesture
    meta_out["_src_fps"] = src_fps
    meta_out["_frames_processed"] = frame_idx
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)

    return {
        "video_id": vid,
        "status": "ok",
        "face_frames": len(face_rows),
        "gesture_frames": len(gesture_rows),
        "voice_features": len(voice_rows),
        "skipped_face": skipped_face,
        "skipped_gesture": skipped_gesture,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    # 取所有行的键的并集（以防某些行缺列）
    all_keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ═══════════════════════════════════════════════════════
# 进度管理
# ═══════════════════════════════════════════════════════

def load_progress() -> set[str]:
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE) as f:
                data = json.load(f)
            return set(data.get("completed", []))
        except Exception:
            return set()
    return set()


def save_progress(completed: set[str]) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"completed": sorted(completed), "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, f)


def load_metadata_index() -> dict[str, dict]:
    """读取 metadata.jsonl，按 file_name stem 索引。"""
    index: dict[str, dict] = {}
    if not METADATA_PATH.exists():
        log(f"!! 元数据文件不存在: {METADATA_PATH}")
        return index
    with open(METADATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            stem = Path(entry["file_name"]).stem  # "vid_0001"
            index[stem] = entry
    return index


# ═══════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser(description="CueCoT 批量特征提取")
    ap.add_argument("--test", type=int, default=0, help="只处理前 N 个视频（验证用）")
    ap.add_argument("--workers", type=int, default=4, help="并行进程数")
    ap.add_argument("--video", type=str, default="", help="只处理指定视频 ID（如 vid_0042）")
    args = ap.parse_args()

    log("=== CueCoT 批量特征提取 ===")
    log(f"视频目录 : {VIDEO_DIR}")
    log(f"输出目录 : {OUTPUT_BASE}")
    log(f"并行进程 : {args.workers}")

    # 加载元数据索引
    meta_index = load_metadata_index()
    log(f"元数据条目: {len(meta_index)}")

    # 收集待处理视频
    if args.video:
        video_files = [VIDEO_DIR / f"{args.video}.mp4"]
    else:
        video_files = sorted(VIDEO_DIR.glob("vid_*.mp4"))
    if args.test > 0:
        video_files = video_files[: args.test]
    log(f"待处理视频: {len(video_files)}")

    # 断点续跑
    completed = load_progress()
    if completed:
        log(f"已跳过 {len(completed)} 个已完成视频")
    pending = [
        (str(vf), str(OUTPUT_BASE / vf.stem), meta_index.get(vf.stem, {}))
        for vf in video_files
        if vf.stem not in completed
    ]
    log(f"实际需处理: {len(pending)}")

    if not pending:
        log("没有需要处理的视频。")
        return

    # 并行处理
    t0 = time.time()
    results = []
    if args.workers <= 1 or len(pending) == 1:
        # 单进程（方便调试）
        for p in pending:
            r = process_one_video(p)
            results.append(r)
            if r["status"] == "ok":
                completed.add(r["video_id"])
                save_progress(completed)
                log(f"  {r['video_id']} done — face={r['face_frames']} gesture={r['gesture_frames']} "
                    f"voice={r['voice_features']} skip_f={r['skipped_face']} skip_g={r['skipped_gesture']}")
            else:
                log(f"  {r['video_id']} FAILED: {r.get('error', r['status'])}")
    else:
        with Pool(processes=args.workers) as pool:
            for r in pool.imap_unordered(process_one_video, pending):
                results.append(r)
                if r["status"] == "ok":
                    completed.add(r["video_id"])
                    save_progress(completed)
                    log(f"  {r['video_id']} done — face={r['face_frames']} gesture={r['gesture_frames']} "
                        f"voice={r['voice_features']} skip_f={r['skipped_face']} skip_g={r['skipped_gesture']}")
                else:
                    log(f"  {r['video_id']} FAILED: {r.get('error', r['status'])}")

    elapsed = time.time() - t0
    ok_count = sum(1 for r in results if r["status"] == "ok")
    fail_count = len(results) - ok_count

    log(f"=== 完成 ===")
    log(f"成功: {ok_count}  失败: {fail_count}  总耗时: {elapsed/60:.1f} min")
    log(f"已完成总计: {len(completed)}")

    # 验证报告
    if args.test > 0:
        log("\n--- 验证报告 ---")
        for r in results:
            if r["status"] == "ok":
                vid = r["video_id"]
                out_d = OUTPUT_BASE / vid
                for fname in ["face.csv", "gesture.csv", "voice.csv", "metadata.json"]:
                    fp = out_d / fname
                    if fp.exists():
                        if fname.endswith(".csv"):
                            with open(fp) as f:
                                nrows = sum(1 for _ in f) - 1  # minus header
                            log(f"  {vid}/{fname}: {nrows} 行")
                        else:
                            log(f"  {vid}/{fname}: OK ({fp.stat().st_size} bytes)")
                    else:
                        log(f"  {vid}/{fname}: MISSING")

        # 抽查一个 metadata.json 的 12 维分数
        sample_d = OUTPUT_BASE / results[0]["video_id"] / "metadata.json"
        if sample_d.exists():
            with open(sample_d, encoding="utf-8") as f:
                meta = json.load(f)
            dims = ["openness", "conscientiousness", "extraversion", "agreeableness",
                    "neuroticism", "overall_personality", "interview_score",
                    "answer_score", "speaking_skills", "confidence_score",
                    "facial_expression", "overall_performance"]
            scores = {d: meta.get(d) for d in dims}
            log(f"\n  抽查 {results[0]['video_id']} 的 12 维分数: {json.dumps(scores, indent=2)}")


if __name__ == "__main__":
    main()
