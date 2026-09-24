#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
时长控制评估驱动器
================================================================================
对「多个特征集 × 含/不含时长代理列」依次跑 fusion/run_fusion.py,
再把结果汇总成一张对照表 —— 用来回答:

    去掉时长伪信号后,行为单模态和「行为+文本」融合各变成多少?

配置矩阵
--------
  基线 1410    results/features          原 9 统计量聚合(对照锚点)
  A 全量 mean  results/features_norm_A   一阶 + 84 列滚动二阶,每列只出 mean
  B 仅一阶     results/features_norm_B   A 去掉 84 列滚动二阶
  各组的 --drop-duration 版本:把 voice.duration_sec 从行为特征里剔除

每次 fusion 跑 5 seeds,单次约 7–10 分钟。

用法
----
    PY=~/miniconda3/envs/jingxin/bin/python
    cd ~/jingxin/experiments/duration_audit

    $PY run_duration_eval.py                   # 核心 3 组(约 30 分钟)
    $PY run_duration_eval.py --full            # 全部 5 组(约 50 分钟)
    $PY run_duration_eval.py --summarize-only  # 只汇总已有结果,不跑模型

**不会覆盖** fusion/results/fusion_result.json —— 基线跑用 _baseline1410 后缀。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

AUDIT_DIR = Path(__file__).resolve().parent
FUSION_DIR = AUDIT_DIR.parent / "fusion"
RESULTS_DIR = Path.home() / "jingxin" / "experiments" / "results"
FUSION_RESULTS = FUSION_DIR / "results"

PY = sys.executable
RUNNER = FUSION_DIR / "run_fusion.py"

# (标签, 特征目录, 是否剔除时长列)
#
# 决定性的一组是 C / Cplus —— 审查判定「方向正确、量测真实」的源列子集。
# 它们回答:A) 去掉所有已知伪影后还剩多少信号;B) 若为 0,是"没信号"还是"维度太少"。
DECISIVE = [
    ("_C", RESULTS_DIR / "features_C", False),
    ("_Cplus", RESULTS_DIR / "features_Cplus", False),
]
AGG = [
    ("_baseline1410", RESULTS_DIR / "features", False),
    ("_normA", RESULTS_DIR / "features_norm_A", False),
    ("_normB", RESULTS_DIR / "features_norm_B", False),
]
CORE = DECISIVE
EXTRA = AGG

# 汇总表的行顺序 + 中文名
LABELS = {
    "_C": "C 白名单 22 维(只留可信测量)",
    "_Cplus": "C+ 白名单 53 维(含带缺陷的真实量)",
    "_baseline1410": "基线 1410 维(原 9 统计量)",
    "_normA": "A 全量 mean(187 维)",
    "_normB": "B 仅一阶 mean(103 维)",
}

# 表格列:(结果键, 表头)
COLS = [
    ("beh_rf", "beh_rf"),
    ("beh_rf_durctrl", "beh|dur"),
    ("beh_plus_dur", "beh+dur"),
    ("duration_only", "dur_only"),
    ("text_rf", "text_rf"),
    ("fus_rf", "fus_rf"),
]


def run_one(tag: str, feats: Path, drop: bool, logdir: Path) -> bool:
    cmd = [PY, str(RUNNER), "--features-dir", str(feats), "--tag", tag]
    if drop:
        cmd.append("--drop-duration")
    log_path = logdir / f"fusion{tag}.log"
    print(f"\n{'=' * 78}\n▶ {LABELS.get(tag, tag)}\n  {' '.join(cmd)}\n  日志 → {log_path}\n{'=' * 78}",
          flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(FUSION_DIR))
    print(f"  退出码 {p.returncode},耗时 {time.time() - t0:.0f}s", flush=True)
    return p.returncode == 0


def summarize() -> list[dict]:
    rows = []
    for tag in LABELS:
        fp = FUSION_RESULTS / f"fusion_result{tag}.json"
        if not fp.exists():
            rows.append({"tag": tag, "missing": True})
            continue
        d = json.loads(fp.read_text(encoding="utf-8"))
        s = d["summary"]
        row = {"tag": tag, "missing": False,
               "n_dims": s.get("_run", {}).get("n_beh_dims"),
               "drop_duration": s.get("_run", {}).get("drop_duration")}
        for k, _ in COLS:
            row[k] = s.get(k, {}).get("macro_mean")
            row[k + "_std"] = s.get(k, {}).get("macro_std")
        row["delta_fus_vs_beh"] = s.get("delta_fus_vs_beh")
        row["delta_fus_vs_text"] = s.get("delta_fus_vs_text")
        row["beh_minus_durctrl"] = (
            row["beh_rf"] - row["beh_rf_durctrl"]
            if row["beh_rf"] is not None and row.get("beh_rf_durctrl") is not None else None)
        rows.append(row)
    return rows


def print_table(rows: list[dict]):
    print("\n" + "=" * 118)
    print("时长控制评估汇总  (5 seeds 的 macro Pearson r 均值)")
    print("=" * 118)
    hdr = f"{'配置':<32}{'维度':>6}"
    for _, h in COLS:
        hdr += f"{h:>12}"
    hdr += f"{'Δfus-beh':>11}{'beh缩水':>10}"
    print(hdr)
    print("-" * 118)
    for r in rows:
        if r["missing"]:
            print(f"{LABELS.get(r['tag'], r['tag']):<32}{'—':>6}   (还没跑)")
            continue
        line = f"{LABELS.get(r['tag'], r['tag']):<32}{r['n_dims']:>6}"
        for k, _ in COLS:
            v = r.get(k)
            line += f"{v:>12.4f}" if v is not None else f"{'—':>12}"
        d = r.get("delta_fus_vs_beh")
        line += f"{d:>+11.4f}" if d is not None else f"{'—':>11}"
        s = r.get("beh_minus_durctrl")
        line += f"{s:>+10.4f}" if s is not None else f"{'—':>10}"
        print(line)
    print("-" * 118)
    print("列说明:")
    print("  beh_rf        行为单模态 RF 的 macro r")
    print("  beh|dur       同上,但预测与真值都先对时长残差化 —— 扣除时长能解释的部分")
    print("  beh+dur       把时长当额外特征喂进去(时长是否还有边际价值)")
    print("  dur_only      只用时长一个变量")
    print("  text_rf       文本单模态 TF-IDF")
    print("  fus_rf        行为+文本 early concat")
    print("  Δfus-beh      融合相对行为单模态的增益(文档里关注的 +0.034 是 Δfus-text)")
    print("  beh缩水       beh_rf − beh|dur,越大说明越依赖时长")


def main():
    ap = argparse.ArgumentParser(description="时长控制评估")
    ap.add_argument("--full", action="store_true", help="跑全部 5 个配置")
    ap.add_argument("--summarize-only", action="store_true", help="只汇总,不跑模型")
    ap.add_argument("--only", default="", help="只跑指定 tag,逗号分隔")
    args = ap.parse_args()

    logdir = AUDIT_DIR / "logs"
    logdir.mkdir(exist_ok=True)

    if not args.summarize_only:
        cfgs = CORE + EXTRA if args.full else CORE
        if args.only:
            want = {t.strip() for t in args.only.split(",")}
            cfgs = [c for c in (CORE + EXTRA) if c[0] in want]
        for tag, feats, drop in cfgs:
            if not feats.exists():
                print(f"⚠️  跳过 {tag}:特征目录不存在 {feats}(先跑 reaggregate_normalized.py)")
                continue
            run_one(tag, feats, drop, logdir)

    rows = summarize()
    print_table(rows)
    out = AUDIT_DIR / "duration_eval_summary.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n汇总 → {out}")


if __name__ == "__main__":
    main()
