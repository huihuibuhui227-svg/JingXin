#!/usr/bin/env python
"""
时序特征聚合脚本 — CueCoT 实验阶段二
======================================
将阶段一产出的逐帧 CSV 聚合为固定长度特征向量。

输入：
  D:\jingxin\experiments\results\features\vid_XXXX\ 下的 face.csv, gesture.csv, voice.csv, metadata.json

输出：
  - 每个视频的 feature_vector.json
  - 全局 all_features.jsonl / all_features.npy / targets.npy / feature_names.txt
  - data_summary.json / quality_report.json

用法：
  python aggregate_features.py                    # 处理全部视频
  python aggregate_features.py --test 10          # 仅测试前 10 个视频
  python aggregate_features.py --video vid_0042   # 单视频调试
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

# ---------- 项目配置 ----------
FEATURES_DIR = Path(r"D:\jingxin\experiments\results\features")
OUTPUT_BASE = FEATURES_DIR
PROGRESS_FILE = OUTPUT_BASE / "aggregate_progress.json"

# 质量阈值
FACE_DETECTION_THRESHOLD = 0.5  # 面部检测率低于 50% 的视频标记为低质量
MAX_VIDEO_DURATION_SEC = 300

# 需要在聚合时跳过的列（标识符 / 时间戳 / 字符串标签）
FACE_SKIP_COLS = {"session_id", "timestamp", "tension_level", "dominant_emotion",
                  "emotion_state", "micro_exp_au_name"}
# 始终跳过（在所有模态中）
UNIVERSAL_SKIP_COLS = {}

# 12 维目标
TARGET_KEYS = [
    "openness", "conscientiousness", "extraversion", "agreeableness",
    "neuroticism", "overall_personality", "interview_score", "answer_score",
    "speaking_skills", "confidence_score", "facial_expression", "overall_performance",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════
# 聚合核心函数
# ═══════════════════════════════════════════════════════

def _safe_float(val):
    """尝试将值转换为 float，失败返回 NaN。"""
    if val is None:
        return np.nan
    if isinstance(val, (int, float, np.floating, np.integer)):
        return float(val)
    if isinstance(val, str):
        v = val.strip()
        if v.lower() in ("true", "yes"):
            return 1.0
        if v.lower() in ("false", "no"):
            return 0.0
        try:
            return float(v)
        except ValueError:
            return np.nan
    return np.nan


def is_numeric_series(series: pd.Series) -> bool:
    """判断 Series 是否为数值型（跳过纯字符串列）。"""
    if series.dtype in (np.float64, np.float32, np.int64, np.int32, bool):
        return True
    # 抽样检查前 20 个非空值
    sample = series.dropna().head(20)
    if len(sample) == 0:
        return False
    numeric_count = sum(1 for v in sample if _safe_float(v) == _safe_float(v))
    return numeric_count >= len(sample) * 0.8  # 80% 可转为数值


def _to_numeric_array(series: pd.Series) -> np.ndarray:
    """将 Series 转为 float 数组，非数值置 NaN。"""
    return np.array([_safe_float(v) for v in series], dtype=np.float64)


def aggregate_series_full(values: np.ndarray) -> dict:
    """面部特征全量聚合：mean, std, min, max, slope, volatility, 3-segment means。"""
    clean = values[~np.isnan(values)]
    n = len(clean)
    if n < 2:
        return {
            "mean": float(clean[0]) if n == 1 else 0.0,
            "std": 0.0,
            "min": float(clean[0]) if n == 1 else 0.0,
            "max": float(clean[0]) if n == 1 else 0.0,
            "slope": 0.0,
            "volatility": 0.0,
            "start_avg": 0.0,
            "mid_avg": 0.0,
            "end_avg": 0.0,
        }

    mean = float(np.mean(clean))
    std = float(np.std(clean))
    vmin = float(np.min(clean))
    vmax = float(np.max(clean))

    # slope via linear regression
    x = np.arange(n, dtype=np.float64)
    slope = float(linregress(x, clean).slope)

    # volatility: std/mean, with fallback when mean ≈ 0
    if abs(mean) > 1e-8:
        volatility = float(std / abs(mean))
    else:
        volatility = float(std * 10.0)

    # 3-segment means
    seg_size = max(1, n // 3)
    seg1 = clean[:seg_size]
    seg2 = clean[seg_size:2 * seg_size]
    seg3 = clean[2 * seg_size:]
    start_avg = float(np.mean(seg1)) if len(seg1) > 0 else 0.0
    mid_avg = float(np.mean(seg2)) if len(seg2) > 0 else 0.0
    end_avg = float(np.mean(seg3)) if len(seg3) > 0 else 0.0

    return {
        "mean": mean, "std": std, "min": vmin, "max": vmax,
        "slope": slope, "volatility": volatility,
        "start_avg": start_avg, "mid_avg": mid_avg, "end_avg": end_avg,
    }


def aggregate_series_compact(values: np.ndarray) -> dict:
    """手势特征精简聚合：mean, std, slope, volatility。"""
    clean = values[~np.isnan(values)]
    n = len(clean)
    if n < 2:
        return {"mean": float(clean[0]) if n == 1 else 0.0, "std": 0.0,
                "slope": 0.0, "volatility": 0.0}

    mean = float(np.mean(clean))
    std = float(np.std(clean))
    x = np.arange(n, dtype=np.float64)
    slope = float(linregress(x, clean).slope)
    if abs(mean) > 1e-8:
        volatility = float(std / abs(mean))
    else:
        volatility = float(std * 10.0)
    return {"mean": mean, "std": std, "slope": slope, "volatility": volatility}


def _flatten_stats(prefix: str, stats: dict, flat: dict):
    """将聚合字典展平到扁平字典中，如 face.au1_mean, face.au1_std, ..."""
    for k, v in stats.items():
        flat[f"{prefix}_{k}"] = float(v)


# ═══════════════════════════════════════════════════════
# 单视频处理
# ═══════════════════════════════════════════════════════

def process_one_video(vid_dir: Path) -> dict | None:
    """处理单个视频的特征聚合，返回扁平特征字典 + 元数据；失败返回 None。"""

    vid = vid_dir.name
    face_path = vid_dir / "face.csv"
    gesture_path = vid_dir / "gesture.csv"
    voice_path = vid_dir / "voice.csv"
    meta_path = vid_dir / "metadata.json"

    if not face_path.exists():
        return None

    # --- 读取元数据 ---
    metadata = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)

    targets = {k: metadata.get(k) for k in TARGET_KEYS}
    # 检查 targets 完整性
    if any(v is None for v in targets.values()):
        # 缺少 target 的视频跳过（不应该发生，但做防御）
        return None

    # --- 面部特征 ---
    face_aggregated = {}
    face_flat = {}
    total_frames = 0
    face_valid_frames = 0

    try:
        df_face = pd.read_csv(face_path)
        df_face.columns = [c.strip() for c in df_face.columns]
        total_frames = len(df_face)

        for col in df_face.columns:
            if col in FACE_SKIP_COLS:
                continue
            if not is_numeric_series(df_face[col]):
                continue
            arr = _to_numeric_array(df_face[col])
            # 统计有效帧（用核心 AU 列：au12_smile）
            if col == "au12_smile":
                face_valid_frames = int(np.sum(~np.isnan(arr)))

            stats = aggregate_series_full(arr)
            face_aggregated[col] = stats
            _flatten_stats(f"face.{col}", stats, face_flat)
    except Exception as e:
        log(f"  !! {vid} face.csv 处理失败: {e}")
        return None

    # --- 手势特征 ---
    gesture_aggregated = {}
    gesture_flat = {}
    gesture_frames = 0

    try:
        df_gesture = pd.read_csv(gesture_path) if gesture_path.exists() else None
        if df_gesture is not None and len(df_gesture) > 0:
            df_gesture.columns = [c.strip() for c in df_gesture.columns]
            gesture_frames = len(df_gesture)
            for col in df_gesture.columns:
                if not is_numeric_series(df_gesture[col]):
                    continue
                arr = _to_numeric_array(df_gesture[col])
                stats = aggregate_series_compact(arr)
                gesture_aggregated[col] = stats
                _flatten_stats(f"gesture.{col}", stats, gesture_flat)
    except Exception as e:
        log(f"  !! {vid} gesture.csv 处理失败: {e}")
        # 手势失败不致命，继续
        gesture_flat = {}

    # --- 语音特征 ---
    voice_features = {}
    voice_flat = {}
    try:
        if voice_path.exists():
            df_voice = pd.read_csv(voice_path)
            df_voice.columns = [c.strip() for c in df_voice.columns]
            if len(df_voice) == 1:
                for col in df_voice.columns:
                    val = _safe_float(df_voice[col].iloc[0])
                    if val == val:  # not NaN
                        voice_features[col] = float(val)
                        voice_flat[f"voice.{col}"] = float(val)
            elif len(df_voice) > 1:
                for col in df_voice.columns:
                    arr = _to_numeric_array(df_voice[col])
                    clean = arr[~np.isnan(arr)]
                    if len(clean) >= 2:
                        voice_features[f"{col}_mean"] = float(np.mean(clean))
                        voice_features[f"{col}_std"] = float(np.std(clean))
                        voice_flat[f"voice.{col}_mean"] = float(np.mean(clean))
                        voice_flat[f"voice.{col}_std"] = float(np.std(clean))
                    elif len(clean) == 1:
                        voice_features[col] = float(clean[0])
                        voice_flat[f"voice.{col}"] = float(clean[0])
    except Exception as e:
        log(f"  !! {vid} voice.csv 处理失败: {e}")

    # --- 计算面部检测率 ---
    face_detection_rate = face_valid_frames / total_frames if total_frames > 0 else 0.0

    # --- 构建输出 ---
    result = {
        "video_id": vid,
        "user_no": metadata.get("user_no", ""),
        "question_id": metadata.get("question_id", ""),
        "duration": metadata.get("duration", ""),
        "face_detection_rate": round(face_detection_rate, 4),
        "n_frames": total_frames,
        "n_gesture_frames": gesture_frames,
        "face": face_aggregated,
        "gesture": gesture_aggregated,
        "voice": voice_features,
        "targets": targets,
    }

    return {
        "nested": result,
        "flat": face_flat | gesture_flat | voice_flat,
        "targets": targets,
        "face_detection_rate": face_detection_rate,
    }


# ═══════════════════════════════════════════════════════
# 质量检查
# ═══════════════════════════════════════════════════════

def quality_check(all_features: np.ndarray, feature_names: list[str],
                  video_ids: list[str]) -> dict:
    """运行质量检查，返回 quality_report 字典。"""
    n_videos, n_dims = all_features.shape
    report = {
        "n_videos": n_videos,
        "n_features": n_dims,
        "nan_videos": [],
        "inf_videos": [],
        "constant_features": [],
        "nan_columns": [],
    }

    # 每个视频的 NaN/Inf 统计
    for i, vid in enumerate(video_ids):
        row = all_features[i]
        nan_count = int(np.sum(np.isnan(row)))
        inf_count = int(np.sum(np.isinf(row)))
        if nan_count > 0:
            report["nan_videos"].append({"video_id": vid, "nan_count": nan_count})
        if inf_count > 0:
            report["inf_videos"].append({"video_id": vid, "inf_count": inf_count})

    # 常数列检查
    for j, name in enumerate(feature_names):
        col = all_features[:, j]
        col_clean = col[~np.isnan(col)]
        if len(col_clean) == 0:
            report["constant_features"].append({"column": name, "reason": "all NaN"})
            report["nan_columns"].append(name)
        elif np.std(col_clean) < 1e-10:
            report["constant_features"].append({"column": name, "reason": "std ≈ 0", "value": float(col_clean[0])})

    report["n_nan_videos"] = len(report["nan_videos"])
    report["n_inf_videos"] = len(report["inf_videos"])
    report["n_constant_features"] = len(report["constant_features"])

    return report


# ═══════════════════════════════════════════════════════
# 进度管理
# ═══════════════════════════════════════════════════════

def load_progress() -> set[str]:
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, encoding="utf-8") as f:
                return set(json.load(f).get("completed", []))
        except Exception:
            return set()
    return set()


def save_progress(completed: set[str]) -> None:
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"completed": sorted(completed), "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, f)


# ═══════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser(description="CueCoT 特征聚合")
    ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--video", type=str, default="")
    args = ap.parse_args()

    log("=== CueCoT 特征聚合 ===")
    log(f"特征目录: {FEATURES_DIR}")

    # 收集目录
    if args.video:
        dirs = [FEATURES_DIR / args.video]
    else:
        dirs = sorted(FEATURES_DIR.glob("vid_*"))
    if args.test > 0:
        dirs = dirs[:args.test]
    log(f"待处理视频: {len(dirs)}")

    # 断点续跑
    completed = load_progress()
    if completed:
        log(f"已跳过 {len(completed)} 个已完成视频")
    pending = [d for d in dirs if d.name not in completed]
    log(f"实际需处理: {len(pending)}")

    if not pending:
        log("没有需要处理的视频，开始全局汇总...")
    else:
        # 处理
        nested_results = []
        flat_vectors = []
        targets_list = []
        video_ids_list = []
        excluded = []
        low_quality = []

        t0 = time.time()
        for i, d in enumerate(pending):
            try:
                res = process_one_video(d)
            except Exception as e:
                log(f"  {d.name} EXCEPTION: {e}")
                traceback.print_exc()
                res = None

            if res is None:
                excluded.append(d.name)
                continue

            # 质量过滤
            if res["face_detection_rate"] < FACE_DETECTION_THRESHOLD:
                low_quality.append({
                    "video_id": d.name,
                    "face_detection_rate": res["face_detection_rate"],
                })

            nested_results.append(res["nested"])
            flat_vectors.append(res["flat"])
            targets_list.append(res["targets"])
            video_ids_list.append(d.name)
            completed.add(d.name)

            if (i + 1) % 100 == 0 or i == len(pending) - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                log(f"  进度 {i + 1}/{len(pending)} ({rate:.1f} vid/s) — {elapsed:.0f}s")
                save_progress(completed)

        elapsed = time.time() - t0
        log(f"处理完成: {elapsed:.1f}s, {len(nested_results)} 成功, {len(excluded)} 排除")

        # --- 写入每个视频的 feature_vector.json ---
        log("写入 video-level feature_vector.json ...")
        for nr in nested_results:
            out_path = FEATURES_DIR / nr["video_id"] / "feature_vector.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(nr, f, ensure_ascii=False, indent=2)

        # --- 加载已完成的旧结果（断点续跑时）---
        if completed and len(nested_results) < len(completed):
            log("加载历史 feature_vector.json ...")
            for vid in sorted(completed - {nr["video_id"] for nr in nested_results}):
                fv_path = FEATURES_DIR / vid / "feature_vector.json"
                if fv_path.exists():
                    with open(fv_path, encoding="utf-8") as f:
                        old = json.load(f)
                    nested_results.append(old)
                    flat_vectors.append(
                        {k: v for k, v in old.items() if k not in ("video_id", "user_no",
                             "question_id", "duration", "face_detection_rate",
                             "n_frames", "n_gesture_frames", "face", "gesture", "voice", "targets")
                        and not isinstance(v, dict)}
                    )
                    # 重建 flat
                    new_flat = {}
                    for face_k, face_v in old.get("face", {}).items():
                        if isinstance(face_v, dict):
                            _flatten_stats(f"face.{face_k}", face_v, new_flat)
                    for ges_k, ges_v in old.get("gesture", {}).items():
                        if isinstance(ges_v, dict):
                            _flatten_stats(f"gesture.{ges_k}", ges_v, new_flat)
                    for voi_k, voi_v in old.get("voice", {}).items():
                        new_flat[f"voice.{voi_k}"] = voi_v
                    flat_vectors[-1] = new_flat
                    targets_list.append(old.get("targets", {}))
                    video_ids_list.append(vid)

        # --- 全局汇总 ---
        log("全局汇总...")
        _build_global_outputs(nested_results, flat_vectors, targets_list,
                              video_ids_list, low_quality, excluded)

    # 断点续跑时也要做全局汇总
    if not pending:
        log("从断点恢复，加载所有 feature_vector.json ...")
        all_dirs = sorted(FEATURES_DIR.glob("vid_*"))
        nested_results = []
        flat_vectors = []
        targets_list = []
        video_ids_list = []
        excluded = []
        low_quality = []
        for d in all_dirs:
            fv_path = d / "feature_vector.json"
            meta_path = d / "metadata.json"
            if not fv_path.exists():
                excluded.append(d.name)
                continue
            with open(fv_path, encoding="utf-8") as f:
                nr = json.load(f)
            nested_results.append(nr)
            targets_list.append(nr.get("targets", {}))
            video_ids_list.append(d.name)
            fdr = nr.get("face_detection_rate", 1.0)
            if fdr < FACE_DETECTION_THRESHOLD:
                low_quality.append({"video_id": d.name, "face_detection_rate": fdr})
            # 重建 flat
            new_flat = {}
            for face_k, face_v in nr.get("face", {}).items():
                if isinstance(face_v, dict):
                    _flatten_stats(f"face.{face_k}", face_v, new_flat)
            for ges_k, ges_v in nr.get("gesture", {}).items():
                if isinstance(ges_v, dict):
                    _flatten_stats(f"gesture.{ges_k}", ges_v, new_flat)
            for voi_k, voi_v in nr.get("voice", {}).items():
                new_flat[f"voice.{voi_k}"] = voi_v
            flat_vectors.append(new_flat)
        log(f"加载: {len(nested_results)} 个 feature_vector")
        _build_global_outputs(nested_results, flat_vectors, targets_list,
                              video_ids_list, low_quality, excluded)


def _build_global_outputs(nested_results, flat_vectors, targets_list,
                          video_ids_list, low_quality, excluded):
    """从内存数据构建全局输出文件。"""
    n_videos = len(flat_vectors)
    if n_videos == 0:
        log("无有效视频，跳过全局汇总。")
        return

    # 收集所有特征名（取并集，排序保证可复现）
    all_feature_names = sorted(set(k for fv in flat_vectors for k in fv))
    n_features = len(all_feature_names)
    log(f"特征维度: {n_features}")

    # 构建 numpy 矩阵
    all_features = np.full((n_videos, n_features), np.nan, dtype=np.float64)
    for i, fv in enumerate(flat_vectors):
        for j, name in enumerate(all_feature_names):
            val = fv.get(name)
            if val is not None:
                all_features[i, j] = float(val)

    targets = np.zeros((n_videos, len(TARGET_KEYS)), dtype=np.float64)
    for i, t in enumerate(targets_list):
        for j, k in enumerate(TARGET_KEYS):
            targets[i, j] = float(t.get(k, 0.0))

    # NaN 填充：用列中位数
    nan_mask = np.isnan(all_features)
    total_nan = int(np.sum(nan_mask))
    if total_nan > 0:
        log(f"  NaN 值: {total_nan}, 使用列中位数填充")
        col_medians = np.nanmedian(all_features, axis=0)
        for j in range(n_features):
            col_nan = nan_mask[:, j]
            if np.any(col_nan):
                all_features[col_nan, j] = col_medians[j]

    # Inf 填充
    inf_mask = np.isinf(all_features)
    if np.any(inf_mask):
        log(f"  Inf 值: {int(np.sum(inf_mask))}, 替换为列中位数")
        col_medians = np.nanmedian(all_features, axis=0)
        for j in range(n_features):
            col_inf = inf_mask[:, j]
            if np.any(col_inf):
                all_features[col_inf, j] = col_medians[j]

    # 写入 all_features.npy
    np.save(str(OUTPUT_BASE / "all_features.npy"), all_features.astype(np.float32))
    log(f"  all_features.npy: {all_features.shape}")

    # 写入 targets.npy
    np.save(str(OUTPUT_BASE / "targets.npy"), targets.astype(np.float32))
    log(f"  targets.npy: {targets.shape}")

    # 写入 feature_names.txt
    with open(OUTPUT_BASE / "feature_names.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(all_feature_names))
    log(f"  feature_names.txt: {len(all_feature_names)} 列")

    # 写入 all_features.jsonl
    with open(OUTPUT_BASE / "all_features.jsonl", "w", encoding="utf-8") as f:
        for i, nr in enumerate(nested_results):
            entry = {
                "video_id": video_ids_list[i] if i < len(video_ids_list) else nr.get("video_id"),
                "user_no": nr.get("user_no", ""),
                "question_id": nr.get("question_id", ""),
                "duration": nr.get("duration", ""),
                "face_detection_rate": nr.get("face_detection_rate", 1.0),
            }
            # 扁平特征
            for j, name in enumerate(all_feature_names):
                entry[name] = float(all_features[i, j])
            # targets
            for k in TARGET_KEYS:
                entry[f"target_{k}"] = float(targets[i, TARGET_KEYS.index(k)])
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log(f"  all_features.jsonl: {n_videos} 行")

    # 质量检查
    quality = quality_check(all_features, all_feature_names, video_ids_list)
    quality["low_quality_videos"] = low_quality
    quality["n_low_quality"] = len(low_quality)
    quality["excluded_videos"] = excluded
    quality["n_excluded"] = len(excluded)
    with open(OUTPUT_BASE / "quality_report.json", "w", encoding="utf-8") as f:
        json.dump(quality, f, ensure_ascii=False, indent=2)
    log(f"  quality_report.json: {quality['n_nan_videos']} NaN视频, "
        f"{quality['n_constant_features']} 常数列, {quality['n_low_quality']} 低质量")

    # data_summary.json
    face_dim = sum(1 for n in all_feature_names if n.startswith("face."))
    gesture_dim = sum(1 for n in all_feature_names if n.startswith("gesture."))
    voice_dim = sum(1 for n in all_feature_names if n.startswith("voice."))

    target_stats = {}
    for j, k in enumerate(TARGET_KEYS):
        tcol = targets[:, j]
        target_stats[k] = {
            "mean": float(np.mean(tcol)),
            "std": float(np.std(tcol)),
            "min": float(np.min(tcol)),
            "max": float(np.max(tcol)),
        }

    summary = {
        "n_videos": n_videos,
        "n_excluded": len(excluded),
        "n_low_quality": len(low_quality),
        "total_feature_dim": n_features,
        "face_features": face_dim,
        "gesture_features": gesture_dim,
        "voice_features": voice_dim,
        "target_dim": len(TARGET_KEYS),
        "target_stats": target_stats,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(OUTPUT_BASE / "data_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"  data_summary.json: face={face_dim} gesture={gesture_dim} voice={voice_dim}")

    log("=== 聚合完成 ===")


if __name__ == "__main__":
    main()
