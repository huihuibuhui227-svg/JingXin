#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
First Impressions 时长检验（对照组）
====================================

前两个数据集被"时长混淆"污染，FI 很可能是**天然的对照组**：
它是 thin-slice 设计，片段长度被人为固定，所以时长在它身上不应该是变量。

这个脚本回答两件事：
  1. FI 的片段时长到底有多一致？如果全部约 15 秒，那它没有时长混淆
  2. 时长（残余的那点差异）与 6 个标注的相关有多强

顺带记录一个已知事实：FI 只有原始视频 + 标注，**没有预计算特征**，
所以它没法像 RecruitView 那样直接做 1410 维的审计——
要用它必须先用 jingxin 的模块抽一遍特征。

只读，不改任何文件。

用法：
    ~/miniconda3/envs/jingxin/bin/python ~/jingxin/experiments/duration_audit/audit_first_impressions.py
    # 想抽更多样本：加 --n 300
"""
import argparse
import json
import os
import pickle
import random
import subprocess
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

FI = os.path.expanduser("~/first-impressions")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "results_first_impressions.json")

DIMS = ["openness", "conscientiousness", "extraversion",
        "agreeableness", "neuroticism", "interview"]

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=150, help="抽样视频数（默认 150）")
ap.add_argument("--split", default="train", choices=["train", "val", "test"])
args = ap.parse_args()


def die(msg):
    print(f"\n❌ {msg}")
    sys.exit(1)


print("=" * 74)
print("First Impressions 时长检验（对照组）")
print("=" * 74)

if not os.path.isdir(FI):
    die(f"找不到 {FI}")

split_dir = os.path.join(FI, args.split)
vids = sorted(f for f in os.listdir(split_dir) if f.endswith(".mp4"))
if not vids:
    die(f"{split_dir} 里没有 mp4")

print(f"\n{args.split}/ 共 {len(vids)} 个 mp4，随机抽 {min(args.n, len(vids))} 个测时长")


def probe(path):
    """ffprobe 读容器头拿时长（不读内容）"""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=20)
        return float(out.stdout.strip())
    except Exception:
        return np.nan


random.seed(0)
sample = random.sample(vids, min(args.n, len(vids)))
durs = np.array([probe(os.path.join(split_dir, v)) for v in sample])
ok = ~np.isnan(durs)
durs = durs[ok]
sample = [v for v, o in zip(sample, ok) if o]

print("\n" + "─" * 74)
print("① 时长分布")
print("─" * 74)
print(f"   n = {len(durs)}")
print(f"   min {durs.min():.3f}s   中位 {np.median(durs):.3f}s   max {durs.max():.3f}s")
print(f"   均值 {durs.mean():.3f}s   标准差 {durs.std():.3f}s"
      f"   变异系数 {durs.std()/durs.mean()*100:.2f}%")
print(f"   极差 {durs.max()-durs.min():.3f}s  （max/min = {durs.max()/durs.min():.3f}）")

# 长的对比：RecruitView 的跨度
print(f"\n   对照 RecruitView：short≈13s / medium≈26s / long≈52s，跨度约 4 倍")
print(f"   对照 MIT：抽样到 197s ~ 619s，跨度约 3 倍")

res = {"split": args.split, "n_sampled": int(len(durs)),
       "dur_min": round(float(durs.min()), 3), "dur_median": round(float(np.median(durs)), 3),
       "dur_max": round(float(durs.max()), 3), "dur_std": round(float(durs.std()), 4),
       "cv_pct": round(float(durs.std() / durs.mean() * 100), 3)}

# ---------------------------------------------------------------- 时长 vs 标注
print("\n" + "─" * 74)
print("② 时长（残余差异）与 6 个标注的相关")
print("─" * 74)
pkl = os.path.join(FI, "annotations", f"annotation_{args.split}.pkl")
if args.split == "train":
    pkl = os.path.join(FI, "annotations", "train-annotation", "annotation_training.pkl")

if not os.path.exists(pkl):
    print(f"   ⚠️ 找不到标注 {pkl}，跳过")
    res["annotation_check"] = "skipped"
else:
    class U(pickle.Unpickler):
        OK = {("__builtin__", n) for n in
              ("dict", "list", "tuple", "str", "int", "float", "bool", "set", "frozenset", "unicode")}

        def find_class(self, m, n):
            if (m, n) in self.OK or m.startswith("numpy"):
                return super().find_class(m, n)
            raise pickle.UnpicklingError(f"拒绝 {m}.{n}")

    with open(pkl, "rb") as fh:
        ann = U(fh, encoding="latin1").load()

    # 标注的 key 带 .mp4，和视频文件名直接对得上
    dmap = {v: d for v, d in zip(sample, durs)}
    corr_log = {}
    for dim in DIMS:
        if dim not in ann:
            continue
        xs, ys = [], []
        for v in sample:
            if v in ann[dim]:
                xs.append(dmap[v])
                ys.append(float(ann[dim][v]))
        xs, ys = np.array(xs), np.array(ys)
        c = 0.0
        if len(xs) > 3 and xs.std() > 1e-12 and ys.std() > 1e-12:
            c = float(np.corrcoef(xs, ys)[0, 1])
        corr_log[dim] = round(c, 4)
        print(f"   {dim:18s} r = {c:+.3f}   (n={len(xs)})")
    res["duration_vs_annotation"] = corr_log
    if corr_log:
        print(f"\n   平均 |r| = {np.mean(np.abs(list(corr_log.values()))):.3f}")

# ---------------------------------------------------------------- 结论
print("\n" + "=" * 74)
print("结论")
print("=" * 74)
cv = durs.std() / durs.mean() * 100
if cv < 2:
    print(f"   ✅ FI 片段长度高度一致（变异系数 {cv:.2f}%），**没有时长混淆**。")
    print("      它是三个数据集里唯一能排除这条混淆的——适合当干净对照。")
else:
    print(f"   ⚠️ FI 片段仍有 {cv:.2f}% 的长度差异，虽然远小于另两家，")
    print("      但做实验时仍建议把时长当协变量控制。")

print("\n   注意：FI **只有原始视频 + 标注，没有预计算特征**。")
print("   要用它，得先用 jingxin 的模块跑一遍特征抽取（10,000 个 15 秒片段）。")

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(f"\n结果已写入 {OUT_JSON}")
