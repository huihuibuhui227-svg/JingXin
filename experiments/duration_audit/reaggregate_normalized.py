#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
时长归一化重新聚合 —— 修「行为特征被视频时长污染」
================================================================================
背景: 见 ~/jingxin-下一步.md

原 aggregate_features.py 的问题
------------------------------
对每一列算 9 个统计量(mean/std/min/max/slope/volatility/start_avg/mid_avg/end_avg)。
其中 std / slope / volatility / min / max 随「观察时长」单调增长 ——
看的时间越长,越可能遇到极值、趋势越显著、波动越大。
于是 1410 维里很大一部分实际在测「这个视频有多长」,而不是「这个人怎么样」。

本脚本的做法
------------
每列只出 1 个统计量(mean),外加真正「事件列」的每秒率:
    A 组 ≈ 185 维   全部一阶列 + 84 列滚动二阶量(只取 mean)
    B 组 ≈ 100 维   同上,但去掉 84 列滚动二阶量

关键修正(结论来自对 extract_features.py / video_pipeline.py 的实测核对)
------------------------------------------------------------------------
1. 时长来源
   ✅ 用 metadata.json 的 `_frames_processed / _src_fps`
      —— 实测它与 `_gesture_frames*3/_src_fps`、`voice.duration_sec` 三者完全一致
   ❌ 绝不用 face.csv 的 `timestamp` 列:那是 `time.time()` 处理挂钟时间,
      中位是真时长的 2.32 倍(p10=0.45, p90=3.18),与真时长只相关 0.666

2. `micro_exp_duration_frames` 单位是「采样帧」(每 3 帧取 1),不是秒
   → 秒数 = 值 * 3 / _src_fps

3. `micro_exp_onset_frame` 是 15 帧滑窗内部的偏移量(micro_expression.py:45),
   恒在 [7,13] 之间,没有任何时间轴语义 → 丢弃,不进特征

4. 上游已知缺陷(本脚本改不了,需要重跑抽取 —— 详见输出的 upstream_issues)
   * VideoPipeline(fps=30) 但实际每 3 帧取 1 帧(≈10fps)
     → au_history 窗口 90 帧 = 9 秒(而非 3 秒)
     → micro 检测窗 15 帧 = 1.5 秒(而非 0.5 秒)
     → eye_closed_sec 每帧只加 1/30,真实间隔 1/10 → 该列被低估 3 倍
   * blink_rate_per_min 用 `time.time()` 的 60 秒窗口,基准是「处理时间」而非视频时间
     → 不可靠。本脚本用 is_blink 自行算「每秒眨眼数」作为替代

用法
----
    PY=~/miniconda3/envs/jingxin/bin/python
    cd ~/jingxin/experiments/duration_audit

    # 产出 A / B 两组特征 + 分类表 + 时长清单
    $PY reaggregate_normalized.py

    # 回归保护:用原 9 统计量重跑,应与现有 all_features.npy 完全一致
    $PY reaggregate_normalized.py --stats legacy --out-dir /tmp/legacy_probe
    $PY reaggregate_normalized.py --verify-legacy /tmp/legacy_probe
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# 路径与常量
# ──────────────────────────────────────────────────────────────────────────────
FEATURES_DIR = Path.home() / "jingxin" / "experiments" / "results" / "features"
DEFAULT_OUT = Path.home() / "jingxin" / "experiments" / "results"
AUDIT_DIR = Path(__file__).resolve().parent

TARGET_KEYS = [
    "openness", "conscientiousness", "extraversion", "agreeableness",
    "neuroticism", "overall_personality", "interview_score", "answer_score",
    "speaking_skills", "confidence_score", "facial_expression", "overall_performance",
]

# 与原脚本一致的跳过列(标识符 / 时间戳 / 字符串标签)
FACE_SKIP_COLS = {"session_id", "timestamp", "tension_level", "dominant_emotion",
                  "emotion_state", "micro_exp_au_name"}

# 滚动二阶量后缀 —— 决定该列进不进 B 组
ROLLING_SUFFIXES = ("_trend", "_volatility", "_change_rate")

# 真正的事件列(二值,有明确的「发生次数」语义)→ 额外产出 per_sec
EVENT_COLS = {
    ("face", "is_blink"),
    ("gesture", "left_hand_fist_status"),
    ("gesture", "right_hand_fist_status"),
}

# 帧单位列 → 先换算成秒再聚合
FRAME_UNIT_COLS = {
    ("face", "micro_exp_duration_frames"): 3.0,   # 采样帧 → 源帧
}

# 明确丢弃的列(单位/语义有缺陷,见模块 docstring)
DROP_COLS = {
    ("face", "micro_exp_onset_frame"),   # 滑窗内偏移量,无时间语义
}

# 时长分母下限:短于此时的视频,速率类特征置 NaN
MIN_DURATION_FOR_RATE = 2.0

# 生成「列分类表」时全局扫描多少个视频的取值分布
PROBE_SAMPLE = 120

# C / C+ 白名单(由四个模块的代码审查结论得出,见 whitelist_C.json)
WHITELIST_PATH = Path(__file__).resolve().parent / "whitelist_C.json"


def load_whitelist(which: str) -> set[tuple[str, str]]:
    """
    读 whitelist_C.json,返回 {(modality, col), ...}。
    which ∈ {"C", "Cplus"}
    """
    with open(WHITELIST_PATH, encoding="utf-8") as f:
        wl = json.load(f)
    cols: set[tuple[str, str]] = set()
    for mod, names in wl["C"].items():
        if mod.startswith("_"):
            continue
        for n in names:
            cols.add((mod, n))
    if which == "Cplus":
        for mod, names in wl["Cplus_extra"].items():
            if mod.startswith("_"):
                continue
            for n in names:
                cols.add((mod, n))
    return cols


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# 与原 aggregate_features.py 逐字保持一致的数值工具
#    —— legacy 模式的回归验证依赖这份一致性,不要"顺手优化"
# ──────────────────────────────────────────────────────────────────────────────
def _safe_float(val):
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
    if series.dtype in (np.float64, np.float32, np.int64, np.int32, bool):
        return True
    sample = series.dropna().head(20)
    if len(sample) == 0:
        return False
    numeric_count = sum(1 for v in sample if _safe_float(v) == _safe_float(v))
    return numeric_count >= len(sample) * 0.8


def _to_numeric_array(series: pd.Series) -> np.ndarray:
    return np.array([_safe_float(v) for v in series], dtype=np.float64)


def agg_legacy_face(values: np.ndarray) -> dict:
    """原 aggregate_series_full:9 个统计量(含 linregress slope)。"""
    from scipy.stats import linregress
    clean = values[~np.isnan(values)]
    n = len(clean)
    if n < 2:
        return {"mean": float(clean[0]) if n == 1 else 0.0, "std": 0.0,
                "min": float(clean[0]) if n == 1 else 0.0,
                "max": float(clean[0]) if n == 1 else 0.0,
                "slope": 0.0, "volatility": 0.0,
                "start_avg": 0.0, "mid_avg": 0.0, "end_avg": 0.0}
    mean, std = float(np.mean(clean)), float(np.std(clean))
    vmin, vmax = float(np.min(clean)), float(np.max(clean))
    slope = float(linregress(np.arange(n, dtype=np.float64), clean).slope)
    volatility = float(std / abs(mean)) if abs(mean) > 1e-8 else float(std * 10.0)
    # 三段均值。⚠️ 空段的 0.0 保护不能省:
    # n=2 时 seg_size=1,end_avg 取 clean[2:] 是空数组,np.mean([]) 会得到 NaN,
    # 再被全局中位数填充 —— 原脚本在这里给的是 0.0,差别会在回归验证里暴露。
    seg_size = max(1, n // 3)
    seg1, seg2, seg3 = clean[:seg_size], clean[seg_size:2 * seg_size], clean[2 * seg_size:]
    return {"mean": mean, "std": std, "min": vmin, "max": vmax,
            "slope": slope, "volatility": volatility,
            "start_avg": float(np.mean(seg1)) if len(seg1) > 0 else 0.0,
            "mid_avg": float(np.mean(seg2)) if len(seg2) > 0 else 0.0,
            "end_avg": float(np.mean(seg3)) if len(seg3) > 0 else 0.0}


def agg_legacy_gesture(values: np.ndarray) -> dict:
    """原 aggregate_series_compact:4 个统计量。"""
    from scipy.stats import linregress
    clean = values[~np.isnan(values)]
    n = len(clean)
    if n < 2:
        return {"mean": float(clean[0]) if n == 1 else 0.0, "std": 0.0,
                "slope": 0.0, "volatility": 0.0}
    mean, std = float(np.mean(clean)), float(np.std(clean))
    slope = float(linregress(np.arange(n, dtype=np.float64), clean).slope)
    volatility = float(std / abs(mean)) if abs(mean) > 1e-8 else float(std * 10.0)
    return {"mean": mean, "std": std, "slope": slope, "volatility": volatility}


# ──────────────────────────────────────────────────────────────────────────────
# 新增:归一化聚合
# ──────────────────────────────────────────────────────────────────────────────
def is_binary(arr: np.ndarray) -> bool:
    """非 NaN 取值是否 ⊆ {0, 1}。"""
    clean = arr[~np.isnan(arr)]
    if len(clean) == 0:
        return False
    return bool(np.all((clean == 0.0) | (clean == 1.0)))


def count_rising_edges(arr: np.ndarray) -> int:
    """0→1 跳变次数。NaN 视作保持前值(不产生跳变)。"""
    clean = arr[~np.isnan(arr)]
    if len(clean) < 2:
        return 0
    prev = clean[:-1]
    cur = clean[1:]
    return int(np.sum((prev < 0.5) & (cur >= 0.5)))


def duration_seconds(meta: dict, voice: dict | None) -> tuple[float, str]:
    """
    真视频时长(秒)+ 来源标记。
    优先级:`_frames_processed/_src_fps` → `voice.duration_sec`

    刻意不使用 face.csv 的 timestamp —— 那是处理挂钟时间。
    """
    n_frames = meta.get("_frames_processed")
    fps = meta.get("_src_fps")
    if isinstance(n_frames, (int, float)) and isinstance(fps, (int, float)) and fps > 0:
        return float(n_frames) / float(fps), "frames/fps"
    if voice and isinstance(voice.get("duration_sec"), (int, float)):
        return float(voice["duration_sec"]), "voice.duration_sec"
    return float("nan"), "missing"


def aggregate_normalized(values: np.ndarray, dur: float, emit_rate: bool) -> dict:
    """
    归一化聚合:每列只出 mean;事件列额外出 per_sec(每秒钟发生次数)。

    注意 kind 不影响 mean —— 对水平列和二值列,"全片均值"是同一件事。
    唯一的差别就是要不要多出一个每秒率,而这由列名(EVENT_COLS)决定,
    不靠数据探测(逐视频探测会被"整列恒 0"骗到)。
    """
    clean = values[~np.isnan(values)]
    out = {"mean": float(np.mean(clean)) if len(clean) else np.nan}

    if emit_rate:
        if np.isfinite(dur) and dur >= MIN_DURATION_FOR_RATE:
            out["per_sec"] = count_rising_edges(values) / dur
        else:
            out["per_sec"] = np.nan
    return out


def is_rolling(col: str) -> bool:
    """原始列名是否属于滚动二阶量(注意:传原始列名,不是聚合后的扁平键)。"""
    return any(col.endswith(s) for s in ROLLING_SUFFIXES)


def probe_columns(dirs, limit: int) -> dict:
    """
    全局扫描样本,统计每列的取值分布 —— 用于生成「列分类表」(人可复查的报告)。

    必须全局做:单个视频里"手没检测到 → 整列恒 0"会被误判成二值列。
    判定二值要求:取值 ⊆ {0,1} 且 0 和 1 都出现过。
    """
    inf = float("inf")
    stats: dict[tuple[str, str], dict] = {}
    for d in dirs[:limit]:
        for modality, fname in (("face", "face.csv"), ("gesture", "gesture.csv"),
                                ("voice", "voice.csv")):
            p = d / fname
            if not p.exists():
                continue
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            df.columns = [c.strip() for c in df.columns]
            skip = FACE_SKIP_COLS if modality == "face" else set()
            for col in df.columns:
                if col in skip or not is_numeric_series(df[col]):
                    continue
                arr = _to_numeric_array(df[col])
                c = arr[~np.isnan(arr)]
                if len(c) == 0:
                    continue
                s = stats.setdefault((modality, col),
                                     {"min": inf, "max": -inf, "n": 0,
                                      "ones": 0, "zeros": 0, "other": 0})
                s["min"] = min(s["min"], float(c.min()))
                s["max"] = max(s["max"], float(c.max()))
                s["n"] += int(len(c))
                s["ones"] += int(np.sum(c == 1.0))
                s["zeros"] += int(np.sum(c == 0.0))
                # 关键:必须统计"既不是 0 也不是 1"的取值。
                # 只查 min>=0 且 max<=1 会把所有归一化到 [0,1] 的连续列(AU 强度)
                # 全判成二值。
                s["other"] += int(np.sum((c != 0.0) & (c != 1.0)))
    return stats


def describe_column(modality: str, col: str, s: dict | None) -> tuple[str, str]:
    """按名字规则 + 全局分布,给出 (kind, note)。纯报告用途。"""
    if (modality, col) in DROP_COLS:
        return "drop", "15 帧滑窗内的偏移量,恒在 [7,13],无时间轴语义"
    if (modality, col) in FRAME_UNIT_COLS:
        return "frames", "单位为采样帧,聚合前先换算成秒"
    if (modality, col) in EVENT_COLS:
        return "binary_event", "事件列:占片比 + 每秒发生次数"
    # 滚动二阶量只存在于 face/gesture。
    # ⚠️ 不能对 voice 用 is_rolling:voice.pitch_trend 会因 _trend 后缀被误判成滚动量。
    if modality in ("face", "gesture") and is_rolling(col):
        return "rolling2nd", "滚动二阶量(3 秒窗);只进 A 组,只出 mean"
    if s and s["other"] == 0 and s["ones"] > 0 and s["zeros"] > 0:
        return "binary_flag", "二值标志位:只出占片比"
    return "level", "水平量:只出 mean"


# ──────────────────────────────────────────────────────────────────────────────
# 单视频处理
# ──────────────────────────────────────────────────────────────────────────────
def process_one_video(vid_dir: Path, stats: str, whitelist: set | None = None):
    """
    返回 dict:
      nested / flat_all(A) / flat_first(B) / targets / meta
    或 None(缺 face.csv / 缺 target)

    whitelist 不为 None 时,只保留 {(modality, 原始列名)} 里的列(C / C+ 组)。
    """
    vid = vid_dir.name
    face_path = vid_dir / "face.csv"
    gesture_path = vid_dir / "gesture.csv"
    voice_path = vid_dir / "voice.csv"
    meta_path = vid_dir / "metadata.json"

    if not face_path.exists():
        return None

    metadata = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)

    targets = {k: metadata.get(k) for k in TARGET_KEYS}
    if any(v is None for v in targets.values()):
        return None

    # ---- 语音(会话级汇总,原样保留)----
    voice_flat: dict[str, float] = {}
    voice_raw: dict[str, float] = {}
    if voice_path.exists():
        try:
            df_v = pd.read_csv(voice_path)
            df_v.columns = [c.strip() for c in df_v.columns]
            # voice.csv 实测 2011/2011 都是 1 行 —— 它已经是「会话级汇总」
            # (ProsodyFeatureExtractor 内部已做过聚合)。
            # 所以原样透传列名,不加 _mean 后缀:既如实说明"这里没做聚合",
            # 也让 voice.duration_sec 这个真时长列在 A/B 两组里都找得到。
            if len(df_v) == 1:
                for c in df_v.columns:
                    val = _safe_float(df_v[c].iloc[0])
                    if val != val:
                        continue
                    # voice_raw 保留全部列 —— 它还是时长的兜底来源;只过滤写出的特征
                    voice_raw[c] = float(val)
                    if whitelist is None or ("voice", c) in whitelist:
                        voice_flat[f"voice.{c}"] = float(val)
            else:
                # 防御:万一以后出现多行,退回逐列求均值
                for c in df_v.columns:
                    arr = _to_numeric_array(df_v[c])
                    clean = arr[~np.isnan(arr)]
                    if len(clean):
                        voice_flat[f"voice.{c}_mean"] = float(np.mean(clean))
        except Exception as e:
            log(f"  !! {vid} voice.csv 失败: {e}")

    dur, dur_src = duration_seconds(metadata, voice_raw)

    # ---- 面部 / 手势 ----
    # flat_a:全列;flat_b:去掉滚动二阶量。两者在循环里同时构建,
    # 避免事后对聚合后的扁平键做字符串手术(键尾是 _mean,认不出 _volatility)。
    flat_a: dict[str, float] = {}
    flat_b: dict[str, float] = {}
    nested = {"face": {}, "gesture": {}, "voice": voice_raw}
    n_face = n_gesture = 0
    kept_cols: set[tuple[str, str]] = set()
    face_detected_frames = 0

    for modality, path in (("face", face_path), ("gesture", gesture_path)):
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            log(f"  !! {vid} {modality}.csv 失败: {e}")
            continue
        df.columns = [c.strip() for c in df.columns]
        if modality == "face":
            n_face = len(df)
        else:
            n_gesture = len(df)

        skip = FACE_SKIP_COLS if modality == "face" else set()

        for col in df.columns:
            if col in skip:
                continue
            if whitelist is not None and (modality, col) not in whitelist:
                continue
            if not is_numeric_series(df[col]):
                continue
            arr = _to_numeric_array(df[col])

            if modality == "face" and col == "au12_smile":
                face_detected_frames = int(np.sum(~np.isnan(arr)))

            if stats == "legacy":
                agg = agg_legacy_face(arr) if modality == "face" else agg_legacy_gesture(arr)
                nested[modality][col] = agg
                for k, v in agg.items():
                    flat_a[f"{modality}.{col}_{k}"] = float(v)
                continue

            if (modality, col) in DROP_COLS:
                continue
            kept_cols.add((modality, col))

            vals = arr
            if (modality, col) in FRAME_UNIT_COLS:
                # 采样帧 → 秒:×FRAME_SKIP 还原成源帧,再 ÷ 源 fps
                vals = arr * FRAME_UNIT_COLS[(modality, col)] / float(metadata.get("_src_fps") or 30.0)

            agg = aggregate_normalized(vals, dur, emit_rate=(modality, col) in EVENT_COLS)
            nested[modality][col] = agg
            rolling = is_rolling(col)
            for k, v in agg.items():
                key = f"{modality}.{col}_{k}"
                flat_a[key] = float(v)
                if not rolling:
                    flat_b[key] = float(v)

    for k, v in voice_flat.items():
        flat_a[k] = v
        flat_b[k] = v
        # voice 列也进分类表(它不走上面的 face/gesture 循环)
        kept_cols.add(("voice", k.split(".", 1)[1]))

    if stats == "legacy":
        flat_b = flat_a

    return {
        "video_id": vid,
        "nested": nested,
        "flat_a": flat_a,
        "flat_b": flat_b,
        "targets": targets,
        "duration": dur,
        "duration_source": dur_src,
        "n_face_frames": n_face,
        "n_gesture_frames": n_gesture,
        "face_detected_frames": face_detected_frames,
        "kept_cols": kept_cols,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 全局汇总(与 aggregate_features._build_global_outputs 同 schema)
# ──────────────────────────────────────────────────────────────────────────────
def build_outputs(records, flat_key, out_dir: Path, label: str):
    if not records:
        log(f"[{label}] 无记录,跳过")
        return None

    names = sorted({k for r in records for k in r[flat_key]})
    n_v, n_f = len(records), len(names)
    X = np.full((n_v, n_f), np.nan, dtype=np.float64)
    for i, r in enumerate(records):
        fv = r[flat_key]
        for j, nm in enumerate(names):
            v = fv.get(nm)
            if v is not None:
                X[i, j] = float(v)

    y = np.zeros((n_v, len(TARGET_KEYS)), dtype=np.float64)
    for i, r in enumerate(records):
        for j, k in enumerate(TARGET_KEYS):
            y[i, j] = float(r["targets"].get(k, 0.0))

    # NaN / Inf 中位数填充 —— 逐字照搬 aggregate_features._build_global_outputs,
    # 分两趟、各自重算中位数。legacy 回归验证依赖这份一致性,不要合并或"优化"。
    #
    # ⚠️ 代价:被填充的格子从此看不出"这里原本没有数据"。
    # 例如时长 <2s 的视频,其 is_blink_per_sec 本该是 NaN(0.6 秒算速率是噪声),
    # 填充后会变成列中位数,看起来像个正常观测。
    # 所以下面把填充数量逐列记进 imputed_cells.json,并在 all_features.jsonl
    # 里保留 duration 与 too_short_for_rate 两列,供下游过滤。
    import warnings
    imputed = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # 全 NaN 列的中位数
        nan_mask = np.isnan(X)
        if nan_mask.any():
            col_medians = np.nanmedian(X, axis=0)
            for j in range(n_f):
                col_nan = nan_mask[:, j]
                if col_nan.any():
                    X[col_nan, j] = col_medians[j]
                    if names[j] not in ("duration",):
                        imputed[names[j]] = imputed.get(names[j], 0) + int(col_nan.sum())
        inf_mask = np.isinf(X)
        if inf_mask.any():
            col_medians = np.nanmedian(X, axis=0)
            for j in range(n_f):
                col_inf = inf_mask[:, j]
                if col_inf.any():
                    X[col_inf, j] = col_medians[j]
    n_remaining = int(np.isnan(X).sum())

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "all_features.npy", X.astype(np.float32))
    np.save(out_dir / "targets.npy", y.astype(np.float32))
    (out_dir / "feature_names.txt").write_text("\n".join(names), encoding="utf-8")

    with open(out_dir / "all_features.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(records):
            entry = {
                "video_id": r["video_id"],
                "duration": r["duration"],
                "duration_source": r["duration_source"],
                "n_face_frames": r["n_face_frames"],
                "n_gesture_frames": r["n_gesture_frames"],
                # 供下游过滤:速率特征的 NaN 已被中位数填充,靠这个标记才能识别
                "too_short_for_rate": bool(
                    not (np.isfinite(r["duration"]) and r["duration"] >= MIN_DURATION_FOR_RATE)),
            }
            entry.update({nm: float(X[i, j]) for j, nm in enumerate(names)})
            entry.update({f"target_{k}": float(y[i, j]) for j, k in enumerate(TARGET_KEYS)})
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 质量报告
    dead = [names[j] for j in range(n_f)
            if np.std(X[:, j]) < 1e-9]
    n_face = sum(1 for n in names if n.startswith("face."))
    n_gest = sum(1 for n in names if n.startswith("gesture."))
    n_voic = sum(1 for n in names if n.startswith("voice."))
    summary = {
        "label": label,
        "n_videos": n_v,
        "total_feature_dim": n_f,
        "face_features": n_face,
        "gesture_features": n_gest,
        "voice_features": n_voic,
        "n_dead_features": len(dead),
        "dead_features": dead[:50],
        "n_remaining_nan": n_remaining,
        "n_imputed_cells": sum(imputed.values()),
        "n_columns_with_imputation": len(imputed),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "data_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if imputed:
        top = sorted(imputed.items(), key=lambda kv: -kv[1])[:30]
        with open(out_dir / "imputed_cells.json", "w", encoding="utf-8") as f:
            json.dump({"total": sum(imputed.values()), "by_column": imputed,
                       "top_30": top}, f, ensure_ascii=False, indent=2)
        log(f"  中位数填充 {sum(imputed.values())} 格,涉及 {len(imputed)} 列"
            f"(明细 → imputed_cells.json)")

    log(f"[{label}] {n_v} 视频 × {n_f} 维 (face={n_face} gesture={n_gest} voice={n_voic}),"
        f" 死特征 {len(dead)}")
    return X, names


# ──────────────────────────────────────────────────────────────────────────────
# 回归保护:legacy 模式 vs 现有 all_features.npy
# ──────────────────────────────────────────────────────────────────────────────
def verify_legacy(probe_dir: Path) -> bool:
    log("=" * 74)
    log("回归验证:legacy 重跑 vs 现有 all_features.npy")
    log("=" * 74)
    ok = True

    ref_names = [l.strip() for l in (FEATURES_DIR / "feature_names.txt").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    new_names = [l.strip() for l in (probe_dir / "feature_names.txt").read_text(
        encoding="utf-8").splitlines() if l.strip()]

    ref_set, new_set = set(ref_names), set(new_names)
    only_ref, only_new = sorted(ref_set - new_set), sorted(new_set - ref_set)
    log(f"原特征名 {len(ref_names)}  新特征名 {len(new_names)}")
    if only_ref:
        ok = False
        log(f"❌ 只在原结果里的列 ({len(only_ref)}): {only_ref[:10]}")
    if only_new:
        ok = False
        log(f"❌ 只在新结果里的列 ({len(only_new)}): {only_new[:10]}")
    if not only_ref and not only_new:
        log("✅ 特征名集合完全一致")

    def read_rows(d):
        with open(d / "all_features.jsonl", encoding="utf-8") as f:
            ids = [json.loads(ln)["video_id"] for ln in f]
        return ids, np.load(d / "all_features.npy").astype(np.float64), \
            np.load(d / "targets.npy").astype(np.float64)

    ref_ids, ref_X, ref_y = read_rows(FEATURES_DIR)
    new_ids, new_X, new_y = read_rows(probe_dir)
    ref_name_list = (FEATURES_DIR / "feature_names.txt").read_text(
        encoding="utf-8").splitlines()

    # ⚠️ 必须按 video_id 对齐,不能按行号。
    # 原始 all_features.jsonl 的行序是 vid_0011…vid_2011 + vid_0001…vid_0010
    # —— 断点续跑把历史条目追加到了末尾,不是 sorted。
    if set(ref_ids) != set(new_ids):
        ok = False
        log(f"❌ 视频集合不同:原 {len(ref_ids)} 新 {len(new_ids)},"
            f"差集 {sorted(set(ref_ids) ^ set(new_ids))[:10]}")
        return ok
    log(f"视频集合一致 ({len(ref_ids)} 个);原行序是否 sorted: {ref_ids == sorted(ref_ids)}")
    order = [ref_ids.index(v) for v in new_ids]
    ref_X, ref_y = ref_X[order], ref_y[order]

    if ref_X.shape != new_X.shape:
        ok = False
        log(f"❌ 形状不一致:原 {ref_X.shape} 新 {new_X.shape}")
        return ok

    ref_col = [ref_name_list.index(n) for n in new_names]
    a, b = ref_X[:, ref_col], new_X

    # 先查「有限性」是否一致 —— 一边 NaN 一边有值,是填充逻辑不一致的信号,
    # 不能混在数值比较里被吞掉。
    fin_a, fin_b = np.isfinite(a), np.isfinite(b)
    n_fin_mismatch = int((fin_a != fin_b).sum())
    if n_fin_mismatch:
        ok = False
        log(f"❌ 有限性不一致的格子: {n_fin_mismatch}(NaN 填充逻辑与原脚本不同)")
    else:
        log("✅ 有限性完全一致(无 NaN/Inf 分歧)")

    finite = fin_a & fin_b
    diff = np.where(finite, np.abs(a - b), 0.0)
    scale = np.where(finite, np.abs(a), 1.0)
    rel = np.where(scale > 1e-9, diff / scale, diff)

    log(f"最大绝对差 {diff.max():.3e}   最大相对差 {rel.max():.3e}")
    log(f"相对差 > 1e-4 的格子: {int((rel > 1e-4).sum())} / {rel.size}")

    y_diff = np.abs(ref_y - new_y).max()
    log(f"targets 最大绝对差 {y_diff:.3e}")

    if rel.max() < 1e-4 and y_diff < 1e-5 and n_fin_mismatch == 0 and not only_ref and not only_new:
        log("✅ 回归验证通过:legacy 模式能逐格复现现有特征矩阵")
    else:
        ok = False
        log("❌ 回归验证失败 —— 差异超出 float32 舍入范围")
        bad_j = int(np.argmax(rel.max(axis=0))) if rel.ndim == 2 else 0
        log(f"   最差的列: {new_names[bad_j]}")
    return ok


# ──────────────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="时长归一化重新聚合")
    ap.add_argument("--stats", choices=["mean", "legacy"], default="mean")
    ap.add_argument("--set", dest="wl_set", choices=["none", "C", "Cplus"], default="none",
                    help="只保留白名单里的源列(C=严格 / Cplus=宽松);默认 none=全列")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个视频(调试)")
    ap.add_argument("--verify-legacy", type=Path, default=None,
                    help="对指定目录做 legacy 回归验证,不做聚合")
    args = ap.parse_args()

    if args.verify_legacy:
        sys.exit(0 if verify_legacy(args.verify_legacy) else 1)

    dirs = sorted(FEATURES_DIR.glob("vid_*"))
    if args.limit:
        dirs = dirs[:args.limit]
    whitelist = None if args.wl_set == "none" else load_whitelist(args.wl_set)
    log(f"=== 重新聚合 ({args.stats} 模式"
        + (f",白名单 {args.wl_set} = {len(whitelist)} 个源列" if whitelist else "") + ") ===")
    log(f"输入 {FEATURES_DIR}   共 {len(dirs)} 个视频目录")

    records, excluded = [], []
    t0 = time.time()
    for i, d in enumerate(dirs):
        try:
            r = process_one_video(d, args.stats, whitelist)
        except Exception as e:
            log(f"  {d.name} EXCEPTION: {e}")
            traceback.print_exc()
            r = None
        if r is None:
            excluded.append(d.name)
        else:
            records.append(r)
        if (i + 1) % 200 == 0 or i == len(dirs) - 1:
            el = time.time() - t0
            log(f"  {i + 1}/{len(dirs)}  ({el:.0f}s, 排除 {len(excluded)})")

    log(f"聚合完成 {time.time() - t0:.0f}s:成功 {len(records)},排除 {len(excluded)}")

    if args.stats == "legacy":
        build_outputs(records, "flat_a", args.out_dir, "legacy")
        log(f"legacy 探针已写入 {args.out_dir};接着跑 --verify-legacy {args.out_dir}")
        return

    # ---- 时长清单 ----
    import csv as _csv
    man_path = AUDIT_DIR / "duration_manifest.csv"
    with open(man_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["video_id", "duration_sec", "duration_source", "n_face_frames",
                    "n_gesture_frames", "face_detection_rate", "too_short_for_rate"])
        for r in records:
            nf = r["n_face_frames"] or 0
            w.writerow([r["video_id"], f"{r['duration']:.3f}", r["duration_source"],
                        nf, r["n_gesture_frames"],
                        f"{(r['face_detected_frames'] / nf) if nf else 0:.4f}",
                        int(not (np.isfinite(r["duration"]) and r["duration"] >= MIN_DURATION_FOR_RATE))])
    log(f"时长清单 → {man_path}")

    # ---- 列分类表(全局扫描样本,人可复查)----
    kept = sorted({c for r in records for c in r["kept_cols"]})
    probe = probe_columns(dirs, PROBE_SAMPLE)
    log(f"列分类:全局扫描 {min(PROBE_SAMPLE, len(dirs))} 个视频的取值分布")
    cls_path = AUDIT_DIR / f"column_classification_{args.wl_set if whitelist else 'AB'}.csv"
    kinds = {}
    with open(cls_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["modality", "column", "kind", "note", "in_group_b",
                    "obs_min", "obs_max", "observed_ones"])
        for mod, col in kept:
            s = probe.get((mod, col))
            kind, note = describe_column(mod, col, s)
            kinds[(mod, col)] = kind
            # voice 列不走 rolling 判定(见 describe_column 的说明),且始终在 B 组里
            in_b = 1 if mod == "voice" else int(not is_rolling(col))
            w.writerow([mod, col, kind, note, in_b,
                        f"{s['min']:.4g}" if s else "",
                        f"{s['max']:.4g}" if s else "",
                        s["ones"] if s else ""])
    from collections import Counter
    n_kept = sum(1 for k in kinds.values() if k != "drop")
    n_no_roll = sum(1 for (m, c), k in kinds.items() if k != "drop" and not is_rolling(c))
    log(f"列分类表 → {cls_path}")
    if whitelist is not None:
        log(f"  白名单 {args.wl_set}:{n_kept} 个源列;分类分布 {dict(Counter(kinds.values()))}")
    else:
        log(f"  A 保留 {n_kept} 列 / B 保留 {n_no_roll} 列;"
            f"分类分布 {dict(Counter(kinds.values()))}")

    # ---- 特征产出 ----
    if whitelist is not None:
        # 白名单模式:只出一套(白名单已排除全部滚动二阶列,flat_a == flat_b)
        label = f"{args.wl_set} 白名单 {len(whitelist)} 源列"
        build_outputs(records, "flat_b", args.out_dir / f"features_{args.wl_set}", label)
    else:
        build_outputs(records, "flat_a", args.out_dir / "features_norm_A", "A 全量 mean")
        build_outputs(records, "flat_b", args.out_dir / "features_norm_B", "B 仅一阶 mean")

    # ---- 上游缺陷备忘 ----
    issues = [
        {"file": "experiments/extract_features.py:45", "const": "FRAME_SKIP = 3",
         "issue": "每 3 帧取 1 帧(≈10fps),但 VideoPipeline 按 fps=30 构造",
         "effect": [
             "au_history 窗口 90 帧 = 9 秒(设计意图 3 秒)",
             "micro 检测窗 15 帧 = 1.5 秒(设计意图 0.5 秒)",
             "eye_closed_sec 每帧加 1/30,真实帧间隔 1/10 → 该列被低估 3 倍",
         ],
         "fix": "重跑抽取,把 VideoPipeline 的 fps 传成 _src_fps/FRAME_SKIP"},
        {"file": "face_expression/pipeline/video_pipeline.py:67", "const": "time.time()",
         "issue": "face.csv 的 timestamp 是处理挂钟时间,不是视频内时间",
         "effect": ["timestamp 跨度中位是真时长的 2.32 倍,与真时长只相关 0.666"],
         "fix": "改为 frame_idx / fps;本次聚合改用 metadata 的 _frames_processed/_src_fps"},
        {"file": "face_expression/pipeline/video_pipeline.py:67-85", "const": "blink_rate_per_min",
         "issue": "眨眼计数的 60 秒窗口走 time.time(),基准是处理时间而非视频时间",
         "effect": ["该列时间基准错误,不可靠;focus_score 也受其影响"],
         "fix": "本次已用 is_blink 自行算 face.is_blink_per_sec 作为替代"},
        {"file": "experiments/aggregate_features.py:282", "const": "face_detection_rate",
         "issue": "分母用 len(face.csv) 而非采样帧总数,而 face.csv 只含检测到脸的帧",
         "effect": ["该指标恒接近 1,失去质控意义"],
         "fix": "分母应为 metadata 的 _gesture_frames(每个采样帧一行)或 _frames_processed/FRAME_SKIP"},
    ]
    ip = AUDIT_DIR / "upstream_issues.json"
    with open(ip, "w", encoding="utf-8") as f:
        json.dump({"excluded_videos": excluded, "issues": issues}, f,
                  ensure_ascii=False, indent=2)
    log(f"上游缺陷备忘 → {ip}")
    log("=== 完成 ===")


if __name__ == "__main__":
    main()
