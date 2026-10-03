#!/usr/bin/env python
"""给清单里已有的条目补上 `ledger_sig`（台账指纹），让 `--pending` 立刻变便宜。

背景：`--uniform` 那一轮跑的时候，工具里还没有 `ledger_sig` 这个字段，所以
清单里的条目都缺它。而 `--pending` 对"缺指纹"的场次一律**保守地**判为"变过"⟹
补指纹之前跑 `--pending`，它会把全语料再重跑一遍（约半小时）。

这些条目本来就是**读着那份台账**得出的结论，所以事后把当前台账的指纹写进去
是如实的，不是编造。补完立刻生效：`--pending` 变成几毫秒的比对。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bf", REPO / "tools" / "jx_backfill_face.py")
bf = importlib.util.module_from_spec(spec)
sys.modules["bf"] = bf
spec.loader.exec_module(bf)


def main() -> int:
    man_path = bf.MANIFEST
    if not man_path.is_file():
        print(f"✗ 没有清单: {man_path}")
        return 2
    man = json.loads(man_path.read_text(encoding="utf-8"))

    # session_id -> 会话目录
    by_sid: dict[str, Path] = {}
    for d in bf.DEFAULT_RECORDINGS.iterdir():
        if d.is_dir() and (d / "retention.face.jsonl").is_file():
            by_sid[d.name.split("__")[-1]] = d

    added, missing, already = 0, [], 0
    for sid, entry in man.items():
        if "ledger_sig" in entry:
            already += 1
            continue
        d = by_sid.get(sid)
        if d is None:
            missing.append(sid)
            continue
        sig = bf.ledger_sig(d)
        if sig is None:
            missing.append(sid)
            continue
        entry["ledger_sig"] = sig
        added += 1

    man_path.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✓ 补指纹 {added} 条；本来就有 {already} 条")
    if missing:
        print(f"⚠ {len(missing)} 条找不到对应会话目录（保持无指纹，--pending 会重跑它们）:")
        for s in missing:
            print(f"    {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
