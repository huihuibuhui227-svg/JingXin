#!/usr/bin/env python
"""
MIT INTERVIEW DATASET — 特征构建(每场一个向量)
=============================================
主键: interview key = 视频小写名 {p<k>, pp<k>}; 138 场 = 69 人 × (pre/post)
产出:
  行为特征:  人脸 Facial_Features(37 列 → mean/std/min/max 4 统计量)
             + 韵律 Prosody(60 列 → 按 Q 求 mean/std)
             + 微笑 SmileData 前 3 列(mean/std/min/max/pct>0)
  文本:      transcript 仅保留 Interviewee 发言(候选人的回答内容)
  真值:      turker 10 人评分的均值, 取下列目标维
输出到 out/: keys.json, X_behavior.npy, texts.json, y.npy, y_names.json
"""
from __future__ import annotations
import json, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path.home() / "MIT_INTERVIEW_DATASET"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

# 目标评分维(均为越高越好): 综合质量 + 结构化表达/沉稳(偏内容)
TARGETS = ["Overall", "RecommendHiring", "Colleague",
           "StructuredAnswers", "Calm", "EngagingTone"]


def all_keys():
    keys = sorted(p.name[:-4].lower() for p in (ROOT / "Videos").glob("*.avi"))
    return keys


def person_of(key):
    return int(re.search(r"\d+", key).group())


# ── 文本: 仅候选人发言 ─────────────────────────────────────────────
def load_text(key):
    fp = ROOT / "Labels" / "interview_transcripts_by_turkers.csv"
    # 快速字典: 缓存
    global _TRANS
    if _TRANS is None:
        _TRANS = {}
        df = pd.read_csv(fp)
        for _, row in df.iterrows():
            _TRANS[str(row.iloc[0]).lower()] = row.iloc[1]
    raw = _TRANS.get(key)
    if not raw or not isinstance(raw, str):
        return ""
    turns = []
    for seg in raw.split("|"):
        seg = seg.strip()
        if seg.lower().startswith("interviewee"):
            seg = seg.split(":", 1)[-1].strip()
            if seg and "[???]" not in seg:
                turns.append(seg)
    return " ".join(turns)


# ── 人脸: 每帧 → 统计 ──────────────────────────────────────────────
def agg_frame_stats(df):
    """df: (帧, 列). 对每列给 mean/std/min/max, 列名加后缀; 仅数值列。"""
    feats = {}
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 5:
            continue
        base = re.sub(r"\W+", "_", str(col)).strip("_") or "f"
        feats[f"{base}_mean"] = float(s.mean())
        feats[f"{base}_std"] = float(s.std(ddof=0))
        feats[f"{base}_min"] = float(s.min())
        feats[f"{base}_max"] = float(s.max())
    return feats


def load_facial(key):
    # 文件名是大写 (P10.csv / PP8.csv), 与主键(小写)对应
    fp = ROOT / "Facial_Features" / f"{key.upper()}.csv"
    if not fp.exists():
        return {}
    df = pd.read_csv(fp)
    df = df.dropna(how="all")
    if df.empty:
        return {}
    return agg_frame_stats(df)


# ── 韵律: 每 Q 一行 → 该场所有 Q 聚合 ─────────────────────────────
def load_prosody(key):
    df = pd.read_csv(ROOT / "Prosody" / "prosodic_features.csv")
    df = df[df["participant&question"].astype(str).str.lower().str.startswith(key)]
    if df.empty:
        return {}
    df = df.drop(columns=["participant&question"])
    feats = {}
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 2:
            continue
        base = re.sub(r"\W+", "_", str(col)).strip("_")
        feats[f"pros_{base}_mean"] = float(s.mean())
        feats[f"pros_{base}_std"] = float(s.std(ddof=0))
    return feats


# ── 微笑/点头/摇头: SHORE 时序, 取前 3 列 ────────────────────────
def load_smile(key):
    # 前测(p<k>) → SmileData/pre; 后测(pp<k>) → SmileData/post
    sub = "pre" if key.startswith("p") and not key.startswith("pp") else "post"
    num = re.search(r"\d+", key).group()
    fp = ROOT / "SmileData" / sub / f"Smoothed-features-P{num}.txt"
    if not fp.exists():
        return {}
    df = pd.read_csv(fp, sep=r"\s+", header=None, engine="python").iloc[:, :3]
    feats = {}
    for j, name in enumerate(["smile", "nod", "shake"]):
        s = pd.to_numeric(df[j], errors="coerce").dropna()
        if len(s) < 5:
            continue
        feats[f"{name}_mean"] = float(s.mean())
        feats[f"{name}_std"] = float(s.std(ddof=0))
        feats[f"{name}_max"] = float(s.max())
        feats[f"{name}_pct_pos"] = float((s > 0).mean())
    return feats


# ── 真值: turker 均值 ─────────────────────────────────────────────
def load_y(keys):
    df = pd.read_csv(ROOT / "Labels" / "turker_scores_full_interview.csv")
    df["key"] = df["Participant"].astype(str).str.lower()
    agg = df.groupby("key")[TARGETS].mean()
    Y = np.stack([agg.loc[k].values for k in keys])
    return Y


_TRANS = None


def main():
    keys = all_keys()
    print(f"{len(keys)} interviews")
    beh_feats, texts = [], []
    n_no_beh = 0
    for k in keys:
        f = {}
        f.update(load_facial(k))
        f.update(load_prosody(k))
        f.update(load_smile(k))
        if not f:
            n_no_beh += 1
        beh_feats.append(f)
        texts.append(load_text(k))
    if n_no_beh:
        print(f"警告: {n_no_beh} 场无任何行为特征")
    # 统一特征列
    cols = sorted({c for f in beh_feats for c in f})
    X = np.zeros((len(keys), len(cols)))
    for i, f in enumerate(beh_feats):
        for c, v in f.items():
            X[i, cols.index(c)] = v
    Y = load_y(keys)
    # 文本缺失(无候选发言)
    n_empty_text = sum(1 for t in texts if not t.strip())
    print(f"行为维度 {len(cols)} | 空文本 {n_empty_text} | 真值 {Y.shape}")
    json.dump({"keys": keys, "cols": cols}, open(OUT / "keys.json", "w"))
    np.save(OUT / "X_behavior.npy", X)
    np.save(OUT / "y.npy", Y)
    json.dump({"targets": TARGETS, "texts": texts}, open(OUT / "texts.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=0)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
