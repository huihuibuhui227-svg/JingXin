#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MIT Interview 时长混淆审计
==========================

和 audit_recruitview.py 做同一件事，但针对 MIT 那条线：
MIT 的 beh_rf = 0.479 比 RecruitView 的 0.383 还高 —— 而 MIT 是整场面试，
时长差异极大（抽样看到 pre 197 秒 vs post 619 秒，差 3 倍）。
所以"行为 ≫ 文本"这个结论需要做同样的时长控制检验。

长度变量用 Facial_Features/*.csv 的行数（= 视频帧数）。
这个 CSV 一行一帧，所以行数就是长度——也正是让 volatility/std 类特征膨胀的东西。

只读，不改任何现有文件。结果写到本目录 results_mit.json。

用法：
    ~/miniconda3/envs/jingxin/bin/python ~/jingxin/experiments/duration_audit/audit_mit.py
"""
import json
import os
import re
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

MIT_DIR = os.path.expanduser("~/MIT_INTERVIEW_DATASET")
OUT_DIR = os.path.expanduser("~/jingxin/experiments/mit/out")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "results_mit.json")

TARGETS = ["Overall", "RecommendHiring", "Colleague",
           "StructuredAnswers", "Calm", "EngagingTone"]
SHORT = ["Overall", "Recomm", "Colleag", "Struct", "Calm", "EngagT"]


def die(msg):
    print(f"\n❌ {msg}")
    sys.exit(1)


print("=" * 74)
print("MIT Interview 时长混淆审计")
print("=" * 74)

for p in (f"{OUT_DIR}/X_behavior.npy", f"{OUT_DIR}/y.npy", f"{OUT_DIR}/keys.json"):
    if not os.path.exists(p):
        die(f"缺少文件：{p}\n   先跑 mit/build_mit_features.py 生成")

X = np.load(f"{OUT_DIR}/X_behavior.npy").astype(np.float64)
y = np.load(f"{OUT_DIR}/y.npy").astype(np.float64)
kj = json.load(open(f"{OUT_DIR}/keys.json", encoding="utf-8"))
keys = kj["keys"]
cols = kj["cols"]

print(f"\n特征矩阵 {X.shape}   标签 {y.shape}   特征名 {len(cols)}")
if X.shape[1] != len(cols):
    die(f"特征名数量 {len(cols)} 与矩阵列数 {X.shape[1]} 不符")

# ---------------------------------------------------------------- 长度变量
print("\n" + "─" * 74)
print("① 长度变量：从 Facial_Features 的帧数取")
print("─" * 74)
frames = []
missing = 0
for k in keys:
    # keys 是小写（p10 / pp10），文件名是大写（P10.csv）
    cand = [os.path.join(MIT_DIR, "Facial_Features", f"{k.upper()}.csv"),
            os.path.join(MIT_DIR, "Facial_Features", f"{k}.csv")]
    path = next((c for c in cand if os.path.exists(c)), None)
    if path is None:
        missing += 1
        frames.append(np.nan)
        continue
    with open(path, "rb") as fh:
        frames.append(sum(1 for _ in fh) - 1)  # 减表头
frames = np.array(frames, dtype=float)
if missing:
    print(f"   ⚠️ {missing} 个视频没找到 Facial_Features CSV")
ok = ~np.isnan(frames)
FPS = 29.0
print(f"   帧数：min {np.nanmin(frames):.0f}  中位 {np.nanmedian(frames):.0f}  max {np.nanmax(frames):.0f}")
print(f"   折算时长(29fps)：{np.nanmin(frames)/FPS:.0f}s ~ {np.nanmax(frames)/FPS:.0f}s"
      f"   —— 跨度 {np.nanmax(frames)/max(np.nanmin(frames),1):.1f} 倍")
print(f"   前 5 个：{[(keys[i], int(frames[i])) for i in np.argsort(-frames)[:5]]}")

res = {"n_samples": int(X.shape[0]), "n_features": int(X.shape[1]),
       "frames_min": float(np.nanmin(frames)), "frames_max": float(np.nanmax(frames))}


def corr(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3 or a[m].std() < 1e-12 or b[m].std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a[m], b[m])[0, 1])


# ---------------------------------------------------------------- 死特征 / 有效维度
print("\n" + "─" * 74)
print("② 死特征与有效维度")
print("─" * 74)
std = X.std(0)
dead = std < 1e-9
print(f"   死特征 {int(dead.sum())} / {len(cols)}  ({dead.sum()/len(cols)*100:.1f}%)")
Xa = X[:, ~dead]
Za = (Xa - Xa.mean(0)) / (Xa.std(0) + 1e-12)
ev = np.linalg.eigvalsh(np.corrcoef(Za.T))[::-1]
ev = ev / ev.sum()
cum = np.cumsum(ev)
eff = {}
for th in (0.5, 0.8, 0.9, 0.95):
    k = int(np.searchsorted(cum, th)) + 1
    eff[f"{int(th*100)}%"] = k
    print(f"   解释 {th*100:4.0f}% 方差 → {k:4d} 个主成分  (占 {k/len(cols)*100:5.1f}%)")
res["dead_features"] = int(dead.sum())
res["effective_dims"] = eff

# ---------------------------------------------------------------- 长度 vs 目标
print("\n" + "─" * 74)
print("③ 时长（帧数）与各目标的相关")
print("─" * 74)
len_c = []
for j, t in enumerate(TARGETS):
    c = corr(frames, y[:, j])
    len_c.append(c)
    print(f"   {t:18s}{c:+.3f}")
print(f"\n   平均 |r| = {np.mean(np.abs(len_c)):.3f}")
res["length_vs_targets"] = {t: round(c, 4) for t, c in zip(SHORT, len_c)}

# ---------------------------------------------------------------- 只靠时长建模
print("\n" + "─" * 74)
print("④ 只用「时长」建模 vs 用全部特征")
print("─" * 74)
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold
    from scipy.stats import pearsonr
except ImportError as e:
    die(f"需要 scikit-learn / scipy：{e}")

# 同一人的 pre/post 必须同折
people = np.array([int(re.search(r"\d+", k).group()) for k in keys])
gkf = GroupKFold(n_splits=5)
RF = dict(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=0)


def cv_score(Xm):
    per = np.zeros(y.shape[1])
    for tr, te in gkf.split(Xm, y, groups=people):
        m = RandomForestRegressor(**RF)
        m.fit(Xm[tr], y[tr])
        p = m.predict(Xm[te])
        for j in range(y.shape[1]):
            if y[te, j].std() > 1e-12 and p[:, j].std() > 1e-12:
                per[j] += pearsonr(y[te, j], p[:, j])[0] / gkf.n_splits
    return per


print("   （5 折 person-level 分组交叉验证，pre/post 同折）")
only_len = cv_score(frames.reshape(-1, 1))
full = cv_score(X)
print(f"\n   {'目标':18s}{'仅时长':>10s}{'全部特征':>11s}{'差值':>9s}")
for j, t in enumerate(TARGETS):
    print(f"   {t:18s}{only_len[j]:+10.3f}{full[j]:+11.3f}{full[j]-only_len[j]:+9.3f}")
print(f"\n   macro r ── 仅时长 {only_len.mean():.3f}   全部 {full.mean():.3f}"
      f"   净增益 {full.mean()-only_len.mean():+.3f}")
res["cv_macro_r_length_only"] = round(float(only_len.mean()), 4)
res["cv_macro_r_full"] = round(float(full.mean()), 4)

# ---------------------------------------------------------------- 偏相关
print("\n" + "─" * 74)
print("⑤ 每个目标最强的 5 个特征：控制长度后还剩多少")
print("─" * 74)
partial_log = {}
A = np.c_[np.ones_like(frames[ok]), frames[ok]]
for j, t in enumerate(TARGETS):
    rs = np.array([abs(corr(X[ok, i], y[ok, j])) if std[i] > 1e-9 else 0
                   for i in range(X.shape[1])])
    top = np.argsort(rs)[::-1][:5]
    print(f"\n   {t}")
    partial_log[t] = []
    for i in top:
        f = X[ok, i]
        raw = corr(f, y[ok, j])
        fr = f - A @ np.linalg.lstsq(A, f, rcond=None)[0]
        yr = y[ok, j] - A @ np.linalg.lstsq(A, y[ok, j], rcond=None)[0]
        pr = corr(fr, yr)
        flag = "  ⚠️符号翻转" if (abs(raw) > 0.05 and np.sign(raw) != np.sign(pr) and abs(pr) > 0.05) else ""
        print(f"     {raw:+.3f} → {pr:+.3f}   {cols[i]}{flag}")
        partial_log[t].append({"feature": str(cols[i]), "raw_r": round(raw, 4),
                               "partial_r": round(pr, 4)})
res["partial_correlations"] = partial_log

# ---------------------------------------------------------------- 汇总
raw_top = np.array([abs(v[0]["raw_r"]) for v in partial_log.values()])
par_top = np.array([abs(v[0]["partial_r"]) for v in partial_log.values()])
print("\n" + "=" * 74)
print("汇总")
print("=" * 74)
print(f"   时长跨度           {np.nanmin(frames)/FPS:.0f}s ~ {np.nanmax(frames)/FPS:.0f}s"
      f"   （{np.nanmax(frames)/max(np.nanmin(frames),1):.1f} 倍）")
print(f"   死特征             {res['dead_features']}/{len(cols)}")
print(f"   有效维度           90% 方差需 {eff['90%']} 维")
print(f"   时长平均 |r|       {np.mean(np.abs(len_c)):.3f}")
print(f"   最强特征 原始 r    {raw_top.mean():.3f}")
print(f"   最强特征 控制后    {par_top.mean():.3f}   (缩水 {(1-par_top.mean()/raw_top.mean())*100:.0f}%)")
print(f"   模型 macro r       仅时长 {only_len.mean():.3f}  →  全部特征 {full.mean():.3f}"
      f"  (净增益 {full.mean()-only_len.mean():+.3f})")
print(f"\n   对照：论文里 MIT 的 beh_rf = 0.479")

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(f"\n结果已写入 {OUT_JSON}")
