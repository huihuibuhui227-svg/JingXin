#!/usr/bin/env python3
r"""公开仓库的 PII 守卫 —— 素材目录名里的**真实姓名 + 学号**不许进仓库。

## 背景

本仓的 GitHub 远端是**公开仓库**。素材目录名的形状是

    <姓名>-<学号>-<专业>__<session_id>
    某某甲-<9位像真学号>-示例学院__20260928_212130_5b10

`tools/jx_inventory.py` 会把目录名整个写进生成的文档,测试 fixture 也常直接
抄一场真实目录名当例子。2026-10-25 实测:历史里有 **9 个 commit** 带着真实
姓名+学号(最早 2026-10-03)。当天的决定是**不改写已推上去的历史**
(改写要强推、所有克隆失效,且 GitHub 仍留 dangling commit),
**从现在起守住** ⟹ 闸必须装在**提交前**。

## 判据:形状,不是"出现中文"

    [一-龥]{2,4} - \d{9} - \S+? __ \d{8}_\d{6}_\w{4}

假值一律 9 个 0 打头(`000000000` / `000000001` / `987654321`),
**真实学号不会全 0** ⟹ 判据自带豁免,不需要维护白名单 —— 白名单会腐。

## 三种入口

* `scan_text(text)`         —— 任意字符串
* `scan_patch(patch)`       —— `git diff --cached` 的输出(**只看 `+` 行**:
                               删掉一行真实姓名是清理动作,不该被拦)
* `scan_tracked_files(repo)` —— 仓库已跟踪文件的当前内容(回归地板用)

## 用法

    python3 tools/jx_pii_guard.py            # 扫已跟踪文件,有命中则退出码 1
    python3 tools/jx_pii_guard.py --patch f  # 扫补丁文件
    bash tools/install_pii_hook.sh           # 装成 pre-commit 钩子

退出码:`0` = 干净,`1` = 有命中,`2` = 用法错误。
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

#: 素材目录名的形状。`{2,4}` 是中文姓名常见长度;9 位学号是本院的学号位数。
#: ⚠️ 别放宽到 `\d+` —— 那会把 session_id 里的 `20260928` 之类也吞进来。
_NAME_RE = re.compile(r"[一-龥]{2,4}-\d{9}-[^\s/\\]+?__\d{8}_\d{6}_\w{4}")

#: 假值的判据:**学号里最多一个非零数字**。
#:
#: 现有假值长这样:`000000000` / `000000001` / `987654321` / `123456789` ——
#: 共同点是**没有真实学号的样子**(真实学号是 9 位不规则的院系编号)。
#: 前两个是"几乎全 0",后两个是连号。
#:
#: ⚠️ 宁可放过一个可疑的假值,也不许误报一个真值 —— 一个到处误报的闸会被绕过,
#: 而绕过一次之后就不会有人再看它。
_FAKE_ID_RE = re.compile(r"^0{8}[\d]$|^(\d)\1{8}$|^0*123456789$|^0*987654321$")


def _is_fake(name: str) -> bool:
    """`<姓名>-<学号>-<专业>__<sid>` → 学号是不是明显编的。"""
    parts = name.split("-")
    if len(parts) < 2:
        return False
    return bool(_FAKE_ID_RE.match(parts[1]))


def scan_text(text: str) -> list[tuple[int, str]]:
    """扫一段文本 → `[(行号, 命中的目录名)]`。假值不算命中。"""
    hits: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for m in _NAME_RE.finditer(line):
            name = m.group(0)
            if not _is_fake(name):
                hits.append((lineno, name))
    return hits


def scan_patch(patch: str) -> list[tuple[int, str]]:
    """扫 `git diff` 输出 —— **只看新增行**(`+`),不看删除行。

    为什么:删掉一行真实姓名正是清理动作,拦住它等于让人没法修。
    为什么不能整段扫:`+++ b/path` 这类行会被误判。
    """
    hits: list[tuple[int, str]] = []
    for lineno, line in enumerate(patch.splitlines(), start=1):
        if not line.startswith("+") or line.startswith("+++"):
            continue
        for m in _NAME_RE.finditer(line):
            name = m.group(0)
            if not _is_fake(name):
                hits.append((lineno, name))
    return hits


#: 扫哪些后缀。二进制与素材目录一律跳过 —— 它们不进仓库,扫了只是浪费时间。
_TEXT_SUFFIXES = {
    ".py", ".md", ".json", ".jsonl", ".ts", ".tsx", ".mjs", ".js", ".sh",
    ".txt", ".csv", ".yml", ".yaml", ".toml", ".cfg", ".ini", ".html",
}


def scan_tracked_files(repo: Path | str) -> list[tuple[str, str]]:
    """扫仓库**已跟踪**文件的当前内容 → `[(相对路径, 命中的目录名)]`。"""
    repo = Path(repo)
    out = subprocess.run(["git", "ls-files"], cwd=repo,
                         capture_output=True, text=True, check=True).stdout
    hits: list[tuple[str, str]] = []
    for rel in out.split("\n"):
        if not rel or Path(rel).suffix.lower() not in _TEXT_SUFFIXES:
            continue
        try:
            text = (repo / rel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for _lineno, name in scan_text(text):
            hits.append((rel, name))
    return hits


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--patch":
        if len(args) < 2:
            print("用法:jx_pii_guard.py --patch <补丁文件>", file=sys.stderr)
            return 2
        patch = Path(args[1]).read_text(encoding="utf-8", errors="replace")
        hits = scan_patch(patch)
        for lineno, name in hits:
            print(f"  补丁第 {lineno} 行:{name}")
        return 1 if hits else 0

    if args and args[0] not in ("--check",):
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        print("用法:jx_pii_guard.py [--check]", file=sys.stderr)
        return 2

    repo = Path(__file__).resolve().parents[1]
    hits = scan_tracked_files(repo)
    if not hits:
        return 0
    print("✗ 已跟踪文件里有真实姓名+学号(本仓远端是公开仓库):", file=sys.stderr)
    for rel, name in hits:
        print(f"    {rel}: {name}", file=sys.stderr)
    print("\n改用假值:`示例甲-000000001-示例学院__<sid>`(学号全 0 即自动豁免)。",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
