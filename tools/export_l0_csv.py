#!/usr/bin/env python3
"""把 `l0_columns.json` 按需导出成 CSV —— **取代原先存放在 ~/shared 的那两份导出**。

为什么要有它:那两份 `L0列清单_M3.0.csv` / `L0白名单_M3.0.csv` 是 2026-09-26 的手工导出,
到 2026-09-27 就已经**和真源不一致了**(实测:清单 85 行 vs 真值 86;白名单 119 条 vs 真值 108)。
它们是本仓那个老病的又一个实例 —— **同一份数据存两份,就会分叉**。

所以:不存文件,改成要用时现导。CSV 只是**视图**,真源永远是 `l0_columns.json`。

用法:
    ~/miniconda3/envs/jingxin/bin/python tools/export_l0_csv.py [--out DIR]
    # 默认写到 stdout(清单),白名单写到 <out>/legacy_allowlist.csv 或 stdout 后半段
"""
from __future__ import annotations

import argparse
import csv
import importlib
import sys
from pathlib import Path

# 本脚本在 <repo>/tools/ 下运行 ⟹ `l0_columns` 在**上一级**,需把仓库根加进 sys.path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
l0 = importlib.import_module("l0_columns")

COLUMN_FIELDS = ["column", "modality", "maturity", "status", "unit",
                 "definition", "source", "normalization", "l0_output",
                 "quarantine_ref", "acceptance"]
ALLOWLIST_FIELDS = ["column", "modality", "why", "planned"]


def write(rows: list[dict], fields: list[str], fh) -> None:
    w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fields})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None,
                    help="输出目录;不给就写 stdout(清单 + 空行 + 白名单)")
    args = ap.parse_args()

    doc = l0.load()
    cols, allow = doc["columns"], doc["legacy_allowlist"]

    if args.out:
        d = Path(args.out); d.mkdir(parents=True, exist_ok=True)
        with (d / "l0_columns.csv").open("w", newline="", encoding="utf-8-sig") as fh:
            write(cols, COLUMN_FIELDS, fh)
        with (d / "legacy_allowlist.csv").open("w", newline="", encoding="utf-8-sig") as fh:
            write(allow, ALLOWLIST_FIELDS, fh)
        print(f"已导出 {len(cols)} 列 + {len(allow)} 条白名单 → {d}/", file=sys.stderr)
    else:
        write(cols, COLUMN_FIELDS, sys.stdout)
        print(file=sys.stdout)
        write(allow, ALLOWLIST_FIELDS, sys.stdout)
        print(f"# 共 {len(cols)} 列 / {len(allow)} 条白名单(真源:l0_columns.json)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
