#!/usr/bin/env python
"""一条命令跑完本仓的**元检查**(引用 / 登记 / 口径 / 变异 / 范围),并给人看的结论。

## 为什么要有这个文件

本仓的元守卫已经很厚(引用路径、行号、断言覆盖、死机制、懒加载、封停理由…),
但它们的**触发方式是"跑全量 pytest"**,而全量要 70+ 秒、输出几千行。
于是实际发生的是:**人(和 AI)靠记忆手工重建一小撮检查,而且会漏** ——
2026-10-03 一天里就漏了两次(一次"发帧验证"因 glob 空而根本没跑,一次 JSON 改动
没落到目标条目还提交了)。

这个脚本把那撮检查**固化成一条命令**,并分三层给结论:

1. **引用与登记**(秒级)—— 散文里的引用指不指得着、登记表与实际是否一致;
2. **口径纪律**(秒级)—— 报告层那些"不该说的话"的钉子;
3. **范围**(即时)—— `git status` 里哪些改动是**使用者的未提交工作**,别混进提交。

`--all` 追加全量套件;`--mutations` 追加**真跑变异**(每条一个 pytest 子进程,慢)。

## 用法

    ~/anaconda3/envs/jingxin/bin/python tools/jx_selfcheck.py
    ~/anaconda3/envs/jingxin/bin/python tools/jx_selfcheck.py --all
    ~/anaconda3/envs/jingxin/bin/python tools/jx_selfcheck.py --mutations

退出码:0 = 全绿;1 = 有组失败;2 = 环境/路径问题(不是代码问题)。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: (组名, 该组负责什么, pytest 目标)
#: ⚠️ 加组请连"它在防什么"一起写 —— 一个说不出防什么的组迟早会被当噪音删掉。
_FAST_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("引用完整性",
     "散文里的条目标号 / 行号 / 测试名 / 路径指不指得着",
     ("tests/test_prose_references.py",)),
    ("登记一致性",
     "维度的死权重、白名单的 why/planned、封停名单与表的交叉引用",
     ("tests/test_dimension_live_weight.py",
      "tests/test_mutation_registry.py",
      "tests/test_l0_column_table.py",
      "tests/test_l0_reference_paths.py")),
    ("口径纪律",
     "报告层「把没测到说成测到了」那一类话的钉子",
     ("tests/test_report_layer.py",
      "tests/test_evidence_gaps_tell_the_truth.py",
      "tests/test_evidence_gate.py",
      "tests/test_sample_size_honesty.py")),
)

#: `--all` 追加的:全量套件(慢)。
_ALL_GROUP = ("全量套件", "所有测试(含三模态、媒体留存、面板)", ("tests/",))

#: `--mutations` 追加的:真跑变异(很慢)。
_MUTATION_GROUP = ("变异实测",
                   "把登记表里每条变异真的施加到临时副本上,确认测试真的变红",
                   ("tests/test_mutation_registry.py",))


def _env_for_pytest(tmp: Path) -> dict:
    """给 pytest 子进程的环境。

    这三项是本仓在此沙箱里跑测试的前提:`numba` 的缓存目录、`joblib` 的临时目录、
    `matplotlib` 的配置目录 —— 默认位置在工作区外,写不进去会让部分模块**收集期**就报错
    (`RuntimeError: cannot cache function`),而那不是代码问题。
    """
    env = dict(os.environ)
    env.setdefault("NUMBA_CACHE_DIR", str(tmp / "numba"))
    env.setdefault("JOBLIB_TEMP_FOLDER", str(tmp / "joblib"))
    env.setdefault("MPLCONFIGDIR", str(tmp / "mpl"))
    return env


def _run_group(name: str, targets: tuple[str, ...], tmp: Path, *, mutations: bool) -> tuple:
    """跑一组,返回 `(名字, 通过?, 耗时秒, 结论行)`。"""
    env = _env_for_pytest(tmp)
    if mutations:
        env["JX_RUN_MUTATIONS"] = "1"
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *targets, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=REPO, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    out = (proc.stdout or "") + (proc.stderr or "")
    # 取 pytest 那行 `N passed, M failed ...`(最后一条非空行里通常就是它)
    tail = [l.strip() for l in out.splitlines() if l.strip()]
    summary = tail[-1] if tail else "(没有输出)"
    if proc.returncode not in (0, 1):        # 2+ = 收集/内部错误
        summary = f"★ 跑不起来(返回码 {proc.returncode}):{summary}"
    return name, proc.returncode == 0, dt, summary, out


def _git_scope() -> tuple[int, str]:
    """`git status` 的范围报告 —— 哪些改动是**使用者的未提交工作**。

    本仓纪律:「提交前先点名"哪些文件里装着使用者的未提交工作" —— git 按文件暂存,
    **绕得开不等于可以不说**」。所以这一节不判对错,只**如实列出来**并要求你看着它提交。
    """
    proc = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return 2, "  取不到 git status(不是 git 仓库?)"
    lines = [l for l in proc.stdout.splitlines() if l.strip()]
    if not lines:
        return 0, "  (工作区干净)"
    modified = [l for l in lines if not l.startswith("??")]
    untracked = [l for l in lines if l.startswith("??")]
    out = []
    if modified:
        out.append(f"  已跟踪、有改动({len(modified)} 个) —— 提交时**逐个**点名:")
        out += [f"    {l}" for l in modified]
    if untracked:
        out.append(f"  未跟踪({len(untracked)} 个) —— **默认不进提交**;")
        out.append("  若其中混着使用者的工作(如 `experiments/`、文档草稿),别 `git add -A`:")
        out += [f"    {l}" for l in untracked[:12]]
        if len(untracked) > 12:
            out.append(f"    … 另有 {len(untracked) - 12} 个")
    return 0, "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="本仓的元检查(引用/登记/口径/变异/范围)")
    ap.add_argument("--all", action="store_true", help="追加全量套件(慢,70s+)")
    ap.add_argument("--mutations", action="store_true",
                    help="追加真跑变异(很慢:每条一个 pytest 子进程)")
    ap.add_argument("--quiet", action="store_true", help="只打结论,不打在失败组的输出尾部")
    args = ap.parse_args(argv)

    if not (REPO / "tests").is_dir():
        print(f"✗ 这里不是 JingXin 仓库根:{REPO}", file=sys.stderr)
        return 2

    groups = list(_FAST_GROUPS)
    if args.all:
        groups.append(_ALL_GROUP)
    if args.mutations:
        groups.append(_MUTATION_GROUP)

    # ★ 预检:**先验每个目标存在**,再开跑。
    #   理由:本脚本自己的清单里就写错过文件名(`gaps` 写成 `gates`),
    #   而 pytest 会把它报成 "file or directory not found"、混在一堆测试输出里 ——
    #   那是**工具配置错**,不是被测代码错,必须一眼能分开。
    missing = [t for _n, _w, targets in groups for t in targets if not (REPO / t).exists()]
    if missing:
        print("✗ 本脚本的检查清单里有不存在的目标(工具配置错,不是代码问题):", file=sys.stderr)
        for t in missing:
            print(f"    {t}", file=sys.stderr)
        return 2

    print(f"仓库: {REPO}")
    print(f"解释器: {sys.executable}")
    print()

    tmp = Path(tempfile.mkdtemp(prefix="jx_selfcheck_"))
    failed = []
    outputs: dict[str, str] = {}
    try:
        for name, what, targets in groups:
            print(f"── {name} ──  {what}")
            gname, ok, dt, summary, out = _run_group(name, targets, tmp,
                                                     mutations=args.mutations)
            outputs[gname] = out
            print(f"   {'✓' if ok else '✗'} {summary}   ({dt:.1f}s)")
            print()
            if not ok:
                failed.append(gname)

        print("── 提交范围 ──")
        code, scope = _git_scope()
        print(scope)
        print()

        if failed:
            print(f"══ 有 {len(failed)} 组失败:{', '.join(failed)} ══")
            if not args.quiet:
                for gname in failed:
                    print(f"\n── {gname} 的输出尾部 ──")
                    tail = [l for l in outputs.get(gname, "").splitlines() if l.strip()][-25:]
                    print("\n".join("  " + l for l in tail))
            return 1
        print("══ 全绿 ══")
        if not args.all:
            print("  (只跑了元检查;加 --all 跑全量套件)")
        if not args.mutations:
            print("  (没跑变异实测;加 --mutations 真跑登记表里那 7 条)")
        return 0
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
