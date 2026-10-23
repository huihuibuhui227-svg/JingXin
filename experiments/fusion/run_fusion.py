#!/usr/bin/env python
"""
行为(1410d) + 文本(TF-IDF) 融合验证 — RecruitView
=================================================
目的:检验「融合显著超越各自单模态」的假设(供参赛论文的「互补」叙事定调)。
协议(与 baselines.py 主线对齐,避免信息泄漏):
  * user-level 7:1:2 划分(同一 user 的视频只进一个 split),5 seeds
  * TF-IDF 只在 TRAIN 文本上 fit,val/test 仅 transform
  * 行为特征标准化只在其所在折的 TRAIN 上 fit(仅对线性模型;RF 不做缩放)
  * 模型:RandomForest(多输出);LateFusion = 对 (text_pred, beh_pred)
    用 val 拟合每维权重融合

行:文本单模态 / 行为单模态 / 行为+文本 early-concat / late-fusion
输出:每 seed 每维 pearson_r/mse + macro;json 存档;paired bootstrap p
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import sparse, stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

# ── 路径(WSL)────────────────────────────────────────────────────────────
FEATURES_DIR = Path.home() / "jingxin" / "experiments" / "results" / "features"
METADATA_PATH = Path.home() / "RecruitView" / "metadata.jsonl"
OUTDIR = Path(__file__).resolve().parent / "results"
OUTDIR.mkdir(parents=True, exist_ok=True)

# 时长代理列 —— 用于「时长控制」评估。见 ~/jingxin-下一步.md
DURATION_COL = "voice.duration_sec"

SEEDS = [42, 123, 456, 789, 1024]
TARGET_KEYS = [
    "openness", "conscientiousness", "extraversion", "agreeableness",
    "neuroticism", "overall_personality",
    "interview_score", "answer_score", "speaking_skills",
    "confidence_score", "facial_expression", "overall_performance",
]
PERSONALITY = TARGET_KEYS[:6]
INTERVIEW = TARGET_KEYS[6:]
RF_PARAMS = dict(n_estimators=200, min_samples_leaf=5, max_depth=None, n_jobs=-1, random_state=0)


def load(features_dir=FEATURES_DIR, drop_duration=False):
    X = np.load(features_dir / "all_features.npy").astype(np.float64)
    y = np.load(features_dir / "targets.npy").astype(np.float64)
    with open(features_dir / "feature_names.txt") as f:
        names = [ln.strip() for ln in f]
    vids, jrows = [], []
    with open(features_dir / "all_features.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            vids.append(d["video_id"])
            jrows.append(d)
    assert len(vids) == X.shape[0] == y.shape[0]

    # 时长来源(两级回退):
    #   1) voice.duration_sec 列 —— 基线 1410 与 A/B 组有
    #   2) jsonl 的 duration 字段 —— C/C+ 组刻意剔除了 duration_sec,只能从这里取
    if DURATION_COL in names:
        dur = X[:, names.index(DURATION_COL)].copy()
        dur_src = f"特征列 {DURATION_COL}"
    elif all(isinstance(r.get("duration"), (int, float)) for r in jrows):
        dur = np.array([float(r["duration"]) for r in jrows])
        dur_src = "jsonl 的 duration 字段(秒)"
    else:
        sys.exit(f"❌ {features_dir} 里既没有 {DURATION_COL} 列,jsonl 也没有数值 duration")
    if drop_duration and DURATION_COL in names:
        keep = [i for i, n in enumerate(names) if n != DURATION_COL]
        X = X[:, keep]
        names = [names[i] for i in keep]
        print(f"[drop-duration] 已剔除 {DURATION_COL},行为维度 {len(names)}", file=sys.stderr)
    print(f"[features] {features_dir}  X={X.shape}  时长来源={dur_src}  "
          f"中位={np.median(dur):.1f}s min={dur.min():.1f}s max={dur.max():.1f}s", file=sys.stderr)

    meta = {}
    with open(METADATA_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            vid = Path(d["file_name"]).stem
            meta[vid] = {"user": d["user_no"],
                         "text": (d.get("transcript", "") or "") + " " + (d.get("gemini_summary", "") or "")}
    # user → videos
    u2v = defaultdict(list)
    texts, users = [], []
    missing = 0
    for vid in vids:
        m = meta.get(vid)
        if m is None:
            missing += 1
            texts.append(""); users.append(f"__missing_{missing}")
            continue
        texts.append(m["text"]); users.append(m["user"])
    print(f"行 {len(vids)} | 无文本 {sum(1 for t in texts if not t.strip())} | meta 缺失 {missing}",
          file=sys.stderr)
    return X, y, names, vids, np.array(texts), users, dur


def user_splits(users, seed):
    """User-level 7:1:2, 每个 seed 真随机重新分组(修正:原版只 shuffle 顺序, 组员固定)。"""
    ucount = defaultdict(list)
    for i, u in enumerate(users):
        ucount[u].append(i)  # video indices per user
    ordered = sorted(ucount.keys(), key=lambda u: len(ucount[u]))
    rng = np.random.RandomState(seed)
    rng.shuffle(ordered)
    n = len(ordered)
    n_tr = int(round(0.7 * n)); n_va = int(round(0.1 * n))
    tr_u, va_u, te_u = ordered[:n_tr], ordered[n_tr:n_tr + n_va], ordered[n_tr + n_va:]
    def idxs(ul):
        return [i for u in ul for i in ucount[u]]
    return idxs(tr_u), idxs(va_u), idxs(te_u)


def pearson(a, b):
    return stats.pearsonr(a, b)[0]


def eval_dims(y_true, y_pred):
    rows = {}
    for d, k in enumerate(TARGET_KEYS):
        rows[k] = {"pearson_r": float(pearson(y_true[:, d], y_pred[:, d])),
                   "mse": float(np.mean((y_true[:, d] - y_pred[:, d]) ** 2))}
    rows["macro_r"] = float(np.mean([rows[k]["pearson_r"] for k in TARGET_KEYS]))
    rows["personality_r"] = float(np.mean([rows[k]["pearson_r"] for k in PERSONALITY]))
    rows["interview_r"] = float(np.mean([rows[k]["pearson_r"] for k in INTERVIEW]))
    return rows


def residualize(a, Z):
    """把 a 对 Z(已含截距列)做最小二乘残差化。"""
    beta, *_ = np.linalg.lstsq(Z, a, rcond=None)
    return a - Z @ beta


def dur_controlled_r(y_true, y_pred, dur):
    """
    控制时长后的 macro Pearson r。

    把预测和真值都对 [1, duration] 残差化,再算相关 —— 回答
    「排除掉时长能解释的那部分之后,模型还剩多少预测力」。
    与 duration_audit 里的偏相关口径一致,便于横向比较。
    """
    Z = np.column_stack([np.ones_like(dur), dur])
    rows = {}
    for d, k in enumerate(TARGET_KEYS):
        yt = residualize(y_true[:, d].astype(np.float64), Z)
        yp = residualize(y_pred[:, d].astype(np.float64), Z)
        rows[k] = {"pearson_r": float(pearson(yt, yp))}
    rows["macro_r"] = float(np.mean([rows[k]["pearson_r"] for k in TARGET_KEYS]))
    rows["personality_r"] = float(np.mean([rows[k]["pearson_r"] for k in PERSONALITY]))
    rows["interview_r"] = float(np.mean([rows[k]["pearson_r"] for k in INTERVIEW]))
    return rows


def fit_text(tr_i, va_i, te_i, texts, max_features=800):
    """TF-IDF 只在训练文本上 fit(ngram 二元组曾致 RF 稀疏分裂过慢, 故用 unigram)。"""
    vec = TfidfVectorizer(max_features=max_features, stop_words="english",
                          sublinear_tf=True, min_df=3)
    Xtr = vec.fit_transform(texts[tr_i])
    Xva = vec.transform(texts[va_i])
    Xte = vec.transform(texts[te_i])
    return Xtr, Xva, Xte, vec


def csr(a):
    return a if sparse.issparse(a) else sparse.csr_matrix(a)


def run_seed(seed, X, y, tr, va, te, texts, dur):
    Xtr, Xva, Xte, vec = fit_text(tr, va, te, texts)
    beh_tr, beh_va, beh_te = X[tr], X[va], X[te]
    # early concat (行为+文本)
    fu_tr = sparse.hstack([csr(beh_tr), Xtr], format="csr")
    fu_va = sparse.hstack([csr(beh_va), Xva], format="csr")
    fu_te = sparse.hstack([csr(beh_te), Xte], format="csr")

    res = {}
    # 文本单模态
    mt = RandomForestRegressor(**RF_PARAMS).fit(Xtr, y[tr])
    pt, pva_t = mt.predict(Xte), mt.predict(Xva)
    res["text_rf"] = eval_dims(y[te], pt)
    # 行为单模态
    mb = RandomForestRegressor(**RF_PARAMS).fit(beh_tr, y[tr])
    pb, pva_b = mb.predict(beh_te), mb.predict(beh_va)
    res["beh_rf"] = eval_dims(y[te], pb)
    # early 融合
    mf = RandomForestRegressor(**RF_PARAMS).fit(fu_tr, y[tr])
    pf = mf.predict(fu_te)
    res["fus_rf"] = eval_dims(y[te], pf)
    # late fusion: val 上学每维权重  (text_pred, beh_pred) → y
    pred_stack_tr = np.column_stack([pt, pb])
    pred_stack_va = np.column_stack([pva_t, pva_b])
    lf_test = np.zeros_like(pf)
    lf_val = np.zeros_like(mt.predict(Xva))
    for d in range(y.shape[1]):
        w = Ridge(alpha=1.0).fit(pred_stack_va, y[va][:, d])
        lf_test[:, d] = w.predict(pred_stack_tr)
    res["late_fusion"] = eval_dims(y[te], lf_test)

    # ── 时长控制(新增)────────────────────────────────────────────────
    dur_te = dur[te]
    # (a) 行为单模态,控制时长后的残差相关
    res["beh_rf_durctrl"] = dur_controlled_r(y[te], pb, dur_te)
    # (b) 时长直接当额外特征喂进去,看它还有没有边际价值
    beh_dur_tr = np.column_stack([beh_tr, dur[tr]])
    beh_dur_te = np.column_stack([beh_te, dur_te])
    md = RandomForestRegressor(**RF_PARAMS).fit(beh_dur_tr, y[tr])
    res["beh_plus_dur"] = eval_dims(y[te], md.predict(beh_dur_te))
    # (c) 只用时长建模拟
    mdu = RandomForestRegressor(**RF_PARAMS).fit(dur[tr].reshape(-1, 1), y[tr])
    res["duration_only"] = eval_dims(y[te], mdu.predict(dur_te.reshape(-1, 1)))

    res["_meta"] = {"n_tr": len(tr), "n_va": len(va), "n_te": len(te),
                    "text_dim": Xtr.shape[1],
                    "n_beh_dims": int(X.shape[1])}
    return res, pt, pb, pf, y[te]


def paired_bootstrap_p(err_a, err_b, n_boot=2000, seed=0):
    """err_a/err_b: (n, 12) 逐样本逐维误差;返回 per-dim p 的平均。"""
    d = err_a - err_b   # >0 表示 b 更好?我们用 err_a - err_b;若 a<b(a 更好) 差值<0
    n = d.shape[0]
    rng = np.random.RandomState(seed)
    per_dim_p = np.zeros(d.shape[1])
    for j in range(d.shape[1]):
        dj = d[:, j]
        m = np.mean(dj)
        bs = np.array([np.mean(dj[rng.randint(0, n, n)]) - m for _ in range(n_boot)])
        # p = 双侧经验 p:m=0 在 bootstrap 分布中的位置
        p = (np.sum(bs >= 0) + 1) / (n_boot + 1)
        per_dim_p[j] = p
    return float(np.mean(per_dim_p))


def main():
    ap = argparse.ArgumentParser(description="行为+文本融合验证(带时长控制)")
    ap.add_argument("--features-dir", type=Path, default=FEATURES_DIR,
                    help="特征目录,默认 = 现有 1410 维基线")
    ap.add_argument("--tag", default="",
                    help="结果文件名后缀,避免覆盖基线 fusion_result.json")
    ap.add_argument("--drop-duration", action="store_true",
                    help="把 voice.duration_sec 从行为特征里剔除")
    args = ap.parse_args()

    t0 = time.time()
    X, y, names, vids, texts, users, dur = load(args.features_dir, args.drop_duration)
    per_seed = {}
    for seed in SEEDS:
        tr, va, te = user_splits(users, seed)
        r, pt, pb, pf, yte = run_seed(seed, X, y, tr, va, te, texts, dur)
        # paired bootstrap: 逐样本误差 = (y - pred)^2
        errT = (yte - pt) ** 2
        errB = (yte - pb) ** 2
        errF = (yte - pf) ** 2
        r["_p_F_vs_text"] = {"mean_p": paired_bootstrap_p(errT, errF, seed=seed)}
        r["_p_F_vs_beh"] = {"mean_p": paired_bootstrap_p(errB, errF, seed=seed)}
        per_seed[str(seed)] = r
        print(f"[seed {seed}] text={r['text_rf']['macro_r']:.3f} "
              f"beh={r['beh_rf']['macro_r']:.3f} "
              f"beh|dur={r['beh_rf_durctrl']['macro_r']:.3f} "
              f"fus={r['fus_rf']['macro_r']:.3f} "
              f"late={r['late_fusion']['macro_r']:.3f} "
              f"(beh_dim={r['_meta']['n_beh_dims']})  "
              f"p(F>T)={r['_p_F_vs_text']['mean_p']:.3f} p(F>B)={r['_p_F_vs_beh']['mean_p']:.3f}")
    # 汇总
    def macro_series(key):
        return np.array([per_seed[str(s)][key]["macro_r"] for s in SEEDS])
    summ = {}
    for key in ["text_rf", "beh_rf", "beh_rf_durctrl", "fus_rf", "late_fusion",
                "beh_plus_dur", "duration_only"]:
        mm = macro_series(key)
        summ[key] = {"macro_mean": float(mm.mean()), "macro_std": float(mm.std()),
                     "per_seed": [float(x) for x in mm]}
    summ["delta_fus_vs_text"] = float(per_seed and macro_series("fus_rf").mean() - macro_series("text_rf").mean())
    summ["delta_fus_vs_beh"] = float(macro_series("fus_rf").mean() - macro_series("beh_rf").mean())
    summ["seed_better"] = {
        "fus>text": int(np.sum(macro_series("fus_rf") > macro_series("text_rf"))),
        "fus>beh": int(np.sum(macro_series("fus_rf") > macro_series("beh_rf"))),
    }
    summ["p_F_vs_text_mean"] = float(np.mean([per_seed[str(s)]["_p_F_vs_text"]["mean_p"] for s in SEEDS]))
    summ["p_F_vs_beh_mean"] = float(np.mean([per_seed[str(s)]["_p_F_vs_beh"]["mean_p"] for s in SEEDS]))
    summ["reference_beh_rf_all"] = 0.383  # 已有已调优 user-split 结果
    summ["_run"] = {"features_dir": str(args.features_dir),
                    "drop_duration": bool(args.drop_duration),
                    "n_beh_dims": int(X.shape[1]),
                    "tag": args.tag}
    out_path = OUTDIR / (f"fusion_result{args.tag}.json" if args.tag else "fusion_result.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summ, "per_seed": per_seed}, f, indent=1, ensure_ascii=False)
    print("\n=== 汇总(5 seeds mean±std) ===")
    print(f"  特征目录 {args.features_dir}   行为维度 {X.shape[1]}"
          f"{'   [已剔除时长列]' if args.drop_duration else ''}")
    for key, nm in [("text_rf", "text_rf     "), ("beh_rf", "beh_rf      "),
                    ("beh_rf_durctrl", "beh_rf|dur  "), ("beh_plus_dur", "beh+dur     "),
                    ("duration_only", "duration_only"), ("fus_rf", "fus_rf      "),
                    ("late_fusion", "late_fusion ")]:
        print(f"  {nm} macro = {summ[key]['macro_mean']:.4f}±{summ[key]['macro_std']:.4f}")
    print(f"  Δ fus-text = {summ['delta_fus_vs_text']:+.4f}  Δ fus-beh = {summ['delta_fus_vs_beh']:+.4f}")
    print(f"  fus>text 种子数 {summ['seed_better']['fus>text']}/5  fus>beh {summ['seed_better']['fus>beh']}/5")
    print(f"  mean p(F>T)={summ['p_F_vs_text_mean']:.3f}  mean p(F>B)={summ['p_F_vs_beh_mean']:.3f}")
    print(f"  参考: 既有已调优行为 RF_all = 0.383  (用户级 7:1:2)")
    print(f"结果 → {out_path}")
    print(f"耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
