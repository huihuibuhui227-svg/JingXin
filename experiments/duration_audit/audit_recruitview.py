#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RecruitView 时长混淆审计
========================

回答一个问题：现有 1410 维特征的表现，有多少是"视频时长"伪装的？

检验四件事：
  1. 1410 维里有多少是死特征（零方差）
  2. 1410 维的有效维度是多少（PCA）
  3. 时长单独一个变量，能预测到什么程度
  4. 最强的那些特征，控制时长之后还剩多少

只读，不改任何现有文件。结果写到本目录 results_recruitview.json。

用法：
    ~/miniconda3/envs/jingxin/bin/python ~/jingxin/experiments/duration_audit/audit_recruitview.py
"""
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

FEAT_DIR = os.path.expanduser("~/jingxin/experiments/results/features")
RV_META = os.path.expanduser("~/RecruitView/metadata.jsonl")
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_recruitview.json")

TARGETS = ["openness", "conscientiousness", "extraversion", "agreeableness",
           "neuroticism", "overall_personality", "interview_score", "answer_score",
           "speaking_skills", "confidence_score", "facial_expression", "overall_performance"]
SHORT = ["open", "consc", "extra", "agree", "neuro", "ov_pers",
         "interv", "answer", "speak", "confid", "facial", "ov_perf"]


def die(msg):
    print(f"\n❌ {msg}")
    sys.exit(1)


# ---------------------------------------------------------------- 载入
print("=" * 74)
print("RecruitView 时长混淆审计")
print("=" * 74)

for p in (f"{FEAT_DIR}/all_features.npy", f"{FEAT_DIR}/targets.npy",
          f"{FEAT_DIR}/feature_names.txt", f"{FEAT_DIR}/all_features.jsonl"):
    if not os.path.exists(p):
        die(f"缺少文件：{p}")

X = np.load(f"{FEAT_DIR}/all_features.npy").astype(np.float64)
y = np.load(f"{FEAT_DIR}/targets.npy").astype(np.float64)
names = np.array([l.strip() for l in open(f"{FEAT_DIR}/feature_names.txt", encoding="utf-8")])
vids = [json.loads(l)["video_id"] for l in open(f"{FEAT_DIR}/all_features.jsonl", encoding="utf-8")]

if X.shape[1] != len(names):
    die(f"特征名数量 {len(names)} 与矩阵列数 {X.shape[1]} 不符")

print(f"\n特征矩阵 {X.shape}   标签 {y.shape}")
mods = np.array([n.split(".")[0] for n in names])
for m in ("face", "gesture", "voice"):
    print(f"   {m:8s} {int((mods == m).sum()):5d} 维")

# 画质 / 时长大类（对照用）
meta = {}
if os.path.exists(RV_META):
    for line in open(RV_META, encoding="utf-8"):
        d = json.loads(line)
        meta["vid_" + d["id"]] = d
    print(f"   已载入 RecruitView metadata（{len(meta)} 条）")

QUAL = np.array([1.0 if meta.get(v, {}).get("video_quality") == "High" else 0.0 for v in vids])
DURCAT = np.array([{"short": 0, "medium": 1, "long": 2}.get(meta.get(v, {}).get("duration"), 1)
                   for v in vids])

# 时长：优先用实测的 voice.duration_sec
if "voice.duration_sec" not in names:
    die("找不到 voice.duration_sec，无法做时长检验")
DUR = X[:, int(np.where(names == "voice.duration_sec")[0][0])]

res = {"n_samples": int(X.shape[0]), "n_features": int(X.shape[1])}


def corr(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3 or a[m].std() < 1e-12 or b[m].std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a[m], b[m])[0, 1])


# ---------------------------------------------------------------- 1. 死特征
print("\n" + "─" * 74)
print("① 死特征（零方差，模型学不到任何东西）")
print("─" * 74)
std = X.std(0)
dead = std < 1e-9
print(f"   {int(dead.sum())} / {len(names)} 个   ({dead.sum() / len(names) * 100:.1f}%)")
res["dead_features"] = int(dead.sum())
res["dead_examples"] = [str(n) for n in names[dead][:8]]

# ---------------------------------------------------------------- 2. 有效维度
print("\n" + "─" * 74)
print("② 有效维度（1410 维里真正有多少信息）")
print("─" * 74)
Xa = X[:, ~dead]
Za = (Xa - Xa.mean(0)) / (Xa.std(0) + 1e-12)
ev = np.linalg.eigvalsh(np.corrcoef(Za.T))[::-1]
ev = ev / ev.sum()
cum = np.cumsum(ev)
eff = {}
for th in (0.5, 0.8, 0.9, 0.95):
    k = int(np.searchsorted(cum, th)) + 1
    eff[f"{int(th*100)}%"] = k
    print(f"   解释 {th*100:4.0f}% 方差 → {k:5d} 个主成分  (占全部 {k/len(names)*100:5.1f}%)")
res["effective_dims"] = eff

# ---------------------------------------------------------------- 3. 时长的力量
print("\n" + "─" * 74)
print("③ 时长 / 画质 与各目标的相关")
print("─" * 74)
print(f"   {'目标':10s}{'时长':>10s}{'时长大类':>11s}{'画质High':>11s}")
dur_c, cat_c, qual_c = [], [], []
for j, t in enumerate(TARGETS):
    a, b, c = corr(DUR, y[:, j]), corr(DURCAT, y[:, j]), corr(QUAL, y[:, j])
    dur_c.append(a)
    cat_c.append(b)
    qual_c.append(c)
    print(f"   {SHORT[j]:10s}{a:+10.3f}{b:+11.3f}{c:+11.3f}")
print(f"\n   时长平均 |r| = {np.mean(np.abs(dur_c)):.3f}"
      f"    画质平均 |r| = {np.mean(np.abs(qual_c)):.3f}")
res["duration_vs_targets"] = {t: round(c, 4) for t, c in zip(SHORT, dur_c)}
res["quality_vs_targets"] = {t: round(c, 4) for t, c in zip(SHORT, qual_c)}

# ---------------------------------------------------------------- 4. 只靠时长建模拟
print("\n" + "─" * 74)
print("④ 只用「时长」一个变量建模 vs 用全部 1410 维")
print("─" * 74)
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold
    from scipy.stats import pearsonr
except ImportError as e:
    die(f"需要 scikit-learn / scipy：{e}")

# 按 user 分组切分，防止同一人的多个视频跨集（RecruitView 的硬性要求）
users = np.array([meta.get(v, {}).get("user_no", v) for v in vids])
n_groups = len(set(users))
gkf = GroupKFold(n_splits=5)

RF = dict(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=0)


def cv_score(Xm):
    """5 折 user-level CV，返回每个目标的 macro Pearson r"""
    per_target = np.zeros(y.shape[1])
    for tr, te in gkf.split(Xm, y, groups=users):
        m = RandomForestRegressor(**RF)
        m.fit(Xm[tr], y[tr])
        p = m.predict(Xm[te])
        for j in range(y.shape[1]):
            if y[te, j].std() > 1e-12 and p[:, j].std() > 1e-12:
                per_target[j] += pearsonr(y[te, j], p[:, j])[0] / gkf.n_splits
    return per_target


print("   （5 折 user-level 分组交叉验证，每折训练一个 RF）")
print(f"\n   {'目标':10s}{'仅时长':>10s}{'全部1410维':>13s}{'差值':>9s}")
dur_only = cv_score(DUR.reshape(-1, 1))
full = cv_score(X)
for j, t in enumerate(SHORT):
    print(f"   {t:10s}{dur_only[j]:+10.3f}{full[j]:+13.3f}{full[j]-dur_only[j]:+9.3f}")
print(f"\n   macro r ── 仅时长 {dur_only.mean():.3f}   全部 {full.mean():.3f}"
      f"   净增益 {full.mean()-dur_only.mean():+.3f}")
res["cv_macro_r_duration_only"] = round(float(dur_only.mean()), 4)
res["cv_macro_r_full"] = round(float(full.mean()), 4)
res["cv_per_target"] = {"duration_only": [round(float(v), 4) for v in dur_only],
                        "full": [round(float(v), 4) for v in full]}

# ---------------------------------------------------------------- 5. 偏相关
print("\n" + "─" * 74)
print("⑤ 每个目标最强的 5 个特征：控制时长后还剩多少")
print("─" * 74)
partial_log = {}
for j, t in enumerate(TARGETS):
    rs = np.array([abs(corr(X[:, i], y[:, j])) if std[i] > 1e-9 else 0
                   for i in range(X.shape[1])])
    top = np.argsort(rs)[::-1][:5]
    print(f"\n   {t}")
    partial_log[t] = []
    A = np.c_[np.ones_like(DUR), DUR]
    for i in top:
        f = X[:, i]
        fr = f - A @ np.linalg.lstsq(A, f, rcond=None)[0]
        yr = y[:, j] - A @ np.linalg.lstsq(A, y[:, j], rcond=None)[0]
        pr = corr(fr, yr)
        flag = "  ⚠️符号翻转" if (rs[i] > 0.05 and np.sign(corr(f, y[:, j])) != np.sign(pr) and abs(pr) > 0.05) else ""
        print(f"     原始 {corr(f,y[:,j]):+.3f} → 控制时长 {pr:+.3f}   {names[i]}{flag}")
        partial_log[t].append({"feature": str(names[i]), "raw_r": round(corr(f, y[:, j]), 4),
                               "partial_r": round(pr, 4)})
res["partial_correlations"] = partial_log

# ---------------------------------------------------------------- 汇总
raw_top = np.array([abs(v[0]["raw_r"]) for v in partial_log.values()])
par_top = np.array([abs(v[0]["partial_r"]) for v in partial_log.values()])
print("\n" + "=" * 74)
print("汇总")
print("=" * 74)
print(f"   死特征            {res['dead_features']}/{len(names)}  ({res['dead_features']/len(names)*100:.1f}%)")
print(f"   有效维度          90% 方差只需 {eff['90%']} 维")
print(f"   时长平均 |r|      {np.mean(np.abs(dur_c)):.3f}")
print(f"   最强特征 原始 r   {raw_top.mean():.3f}")
print(f"   最强特征 控制后   {par_top.mean():.3f}   (缩水 {(1-par_top.mean()/raw_top.mean())*100:.0f}%)")
print(f"   模型 macro r      仅时长 {dur_only.mean():.3f}  →  全部特征 {full.mean():.3f}"
      f"  (净增益 {full.mean()-dur_only.mean():+.3f})")

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print(f"\n结果已写入 {OUT_JSON}")
