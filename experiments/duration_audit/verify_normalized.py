#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
归一化聚合产出的独立验证
================================================================================
不信任 reaggregate_normalized.py 自己的输出,从原始 CSV 手工复算若干量再比对。

检查项
------
  1. A 组里不应残留任何被砍掉的统计量后缀(std/slope/min/max/volatility/3段均值)
  2. face.is_blink_per_sec  ==  手数 is_blink 上升沿次数 / 真时长
  3. face.micro_exp_duration_frames_mean 的帧→秒换算正确且在物理范围内
  4. 水平列的 _mean == 该列朴素均值
  5. voice.duration_sec == metadata 的 _frames_processed/_src_fps
  6. B 组 == A 组去掉 84 个滚动二阶列
  7. 时长清单里没有 <2s 的视频仍产出速率特征

用法
    PY=~/miniconda3/envs/jingxin/bin/python
    $PY verify_normalized.py --probe /tmp/probe_norm
    $PY verify_normalized.py --probe ~/jingxin/experiments/results   # 跑真实产出
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def close(a, b, rtol=1e-6, atol=1e-7) -> bool:
    """
    all_features.npy 存的是 float32(精度 ~1e-7),比对必须用 float32 量级的容差。
    早期版本用 1e-9 会把所有正常结果都判成失败。
    """
    return bool(np.isclose(float(a), float(b), rtol=rtol, atol=atol, equal_nan=True))


def agg_stat_of(key: str) -> str:
    """从扁平键取尾部统计量名:'face.au12_smile_mean' -> 'mean'"""
    return key.rsplit("_", 1)[-1]


def base_col_of(key: str) -> str:
    """
    从扁平键还原原始列名:'face.head_pitch_volatility_mean' -> 'head_pitch_volatility'

    ⚠️ 不能用 str.rstrip("_mean") —— rstrip 按「字符集合」剥离,
    '...change_rate_mean'.rstrip('_mean') 会把 e 也吃掉变成 '...change_rat'。
    """
    body = key.split(".", 1)[1]
    return re.sub(r"_(mean|per_sec)$", "", body)

FEATURES_DIR = Path.home() / "jingxin" / "experiments" / "results" / "features"
ROLLING_SUFFIXES = ("_trend", "_volatility", "_change_rate")
BANNED_SUFFIXES = ("_std", "_slope", "_min", "_max", "_start_avg", "_mid_avg", "_end_avg")
VOICE_NATIVE = {"voice.pitch_std", "voice.energy_std"}   # 会话级原始列,不是聚合出来的

FAILS: list[str] = []


def check(cond: bool, msg: str):
    print(("  ✅ " if cond else "  ❌ ") + msg)
    if not cond:
        FAILS.append(msg)


def load_dir(root: Path, sub: str):
    d = root / sub
    names = [l.strip() for l in (d / "feature_names.txt").read_text(encoding="utf-8").splitlines()
             if l.strip()]
    X = np.load(d / "all_features.npy").astype(np.float64)
    rows = [json.loads(l) for l in (d / "all_features.jsonl").read_text(encoding="utf-8").splitlines()]
    return names, X, rows


def raw_duration(vid_dir: Path) -> float:
    md = json.loads((vid_dir / "metadata.json").read_text(encoding="utf-8"))
    return md["_frames_processed"] / md["_src_fps"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, default=Path("/tmp/probe_norm"),
                    help="含有 features_norm_A / features_norm_B 的目录")
    ap.add_argument("--videos", default="vid_0001,vid_0005,vid_0030")
    args = ap.parse_args()

    A_names, A, A_rows = load_dir(args.probe, "features_norm_A")
    B_names, B, B_rows = load_dir(args.probe, "features_norm_B")
    a_idx = {r["video_id"]: i for i, r in enumerate(A_rows)}
    b_idx = {r["video_id"]: i for i, r in enumerate(B_rows)}
    scope = "样本" if len(A_rows) < 200 else "全量"
    print(f"=== 验证 {args.probe}  ({scope} {len(A_rows)} 视频) ===")
    print(f"  A {len(A_names)} 维   B {len(B_names)} 维\n")

    # ---- 1 ----
    print("① A 组不含被砍掉的统计量后缀")
    hit = [n for n in A_names
           if any(n.endswith(s) for s in BANNED_SUFFIXES) and n not in VOICE_NATIVE]
    check(not hit, f"残留: {hit[:8] if hit else '无'}"
                   f"   (voice.pitch_std / voice.energy_std 是会话级原始列,已豁免)")

    # ---- 6 ----
    print("\n⑥ B 组 == A 组去掉滚动二阶列")
    b_from_a = sorted(n for n in A_names
                      if n.startswith("voice.") or not base_col_of(n).endswith(ROLLING_SUFFIXES))
    check(sorted(B_names) == b_from_a,
          f"A 去滚动后 {len(b_from_a)} 维 vs B {len(B_names)} 维,"
          f" 差集 {sorted(set(b_from_a) ^ set(B_names))[:6]}")
    n_removed = len(A_names) - len(b_from_a)
    check(n_removed > 0, f"被移除的滚动二阶列 {n_removed} 维(应约 84)")

    # ---- 逐视频复算 ----
    for vid in [v.strip() for v in args.videos.split(",") if v.strip()]:
        if vid not in a_idx:
            print(f"\n(跳过 {vid}:不在该产出里)")
            continue
        vd = FEATURES_DIR / vid
        if not (vd / "face.csv").exists():
            print(f"\n(跳过 {vid}:原始 CSV 不在)")
            continue
        print(f"\n─── {vid} ───")
        df = pd.read_csv(vd / "face.csv")
        df.columns = [c.strip() for c in df.columns]
        dur = raw_duration(vd)
        i = a_idx[vid]

        # ②	is_blink 每秒率
        print("② face.is_blink_per_sec 手工复算")
        b = df["is_blink"].astype(float).to_numpy()
        edges = int(np.sum((b[:-1] < 0.5) & (b[1:] >= 0.5)))
        exp = edges / dur
        got = A[i, A_names.index("face.is_blink_per_sec")]
        check(close(got, exp),
              f"{edges} 次上升沿 / {dur:.3f}s = {exp:.6f}   产出 {got:.6f}")
        got_m = A[i, A_names.index("face.is_blink_mean")]
        check(close(got_m, np.nanmean(b)),
              f"is_blink_mean 占片比 手算 {np.nanmean(b):.6f}   产出 {got_m:.6f}")

        # ③ 帧→秒
        print("③ face.micro_exp_duration_frames_mean 帧→秒换算")
        c = df["micro_exp_duration_frames"].dropna()
        got_f = A[i, A_names.index("face.micro_exp_duration_frames_mean")]
        if len(c):
            exp_f = c.mean() * 3.0 / 30.0          # 采样帧 → 源帧 → 秒
            check(close(got_f, exp_f),
                  f"原始帧均值 {c.mean():.4f} × 3/30 = {exp_f:.5f}s   产出 {got_f:.5f}s")
            check(0.15 <= got_f <= 0.85,
                  f"值 {got_f:.4f}s 落在微表情时长的物理范围(0.2–0.8s)内")
        else:
            print("     (该视频无微表情检出,跳过)")

        # ④ 水平列均值
        print("④ 水平列 _mean 是否等于朴素均值")
        for col in ["head_pitch", "head_yaw", "au12_smile", "tension_score", "focus_score"]:
            key = f"face.{col}_mean"
            if key not in A_names:
                check(False, f"{key} 缺失")
                continue
            v = df[col].astype(float).to_numpy()
            check(close(A[i, A_names.index(key)], np.nanmean(v)),
                  f"face.{col}_mean 手算 {np.nanmean(v):.6f}")

        # ⑤ 时长
        print("⑤ voice.duration_sec")
        key = "voice.duration_sec"
        if key in A_names:
            check(close(A[i, A_names.index(key)], dur),
                  f"手算 {dur:.4f}s   产出 {A[i, A_names.index(key)]:.4f}s")
        else:
            check(False, f"A 组缺少 {key}")

    # ---- 7 ----
    print("\n⑦ 短于 2s 的视频:速率特征应为 NaN(分母下限)")
    man = Path(__file__).resolve().parent / "duration_manifest.csv"
    if man.exists():
        m = pd.read_csv(man)
        short = m[m.too_short_for_rate == 1].video_id.tolist()
        print(f"     清单里标为过短的视频: {len(short)} 个 {short[:5]}")
        if short and "face.is_blink_per_sec" in A_names:
            j = A_names.index("face.is_blink_per_sec")
            present = [v for v in short if v in a_idx]
            if present:
                # 聚合阶段这些格子确实是 NaN,但 build_outputs 会用列中位数填充,
                # 所以矩阵里看到的是中位数。这里断言填充确实发生了 ——
                # 并明确记录:这些值不是观测,是填充。
                med = np.median(A[:, j])
                n_med = sum(1 for v in present if close(A[a_idx[v], j], med))
                check(n_med == len(present),
                      f"这 {len(present)} 个视频的 is_blink_per_sec 已被中位数填充 "
                      f"({med:.6f}),实际匹配 {n_med}/{len(present)}")
                print(f"     ⚠️ 注意:聚合时这些格子是 NaN(时长 <2s,速率无意义),"
                      f"矩阵里看到的中位数是填充值,不是观测。")
            else:
                print("     (清单里的短视频不在本次产出中)")
    else:
        print(f"     (没有 {man},跳过)")

    print("\n" + "=" * 70)
    if FAILS:
        print(f"❌ {len(FAILS)} 项失败:")
        for f in FAILS:
            print(f"   - {f}")
        sys.exit(1)
    print("✅ 全部通过")


if __name__ == "__main__":
    main()
