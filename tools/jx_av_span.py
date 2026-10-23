#!/usr/bin/env python
"""`camera.webm` 音轨与视轨的真实时长对账 —— 兼查"短场次是录断了还是本来就短"。

为什么需要这个工具(戊-14):

`docs/素材体检_2026-10-03.md` §6 记着「部分素材时长异常短(15~26 秒),是录到一半
断了还是本来就短,尚未查证」。查这件事要先能量出"文件里到底有多少秒",而
**流式 webm 量不出时长** —— Chrome `MediaRecorder` 产的文件没有 `format.duration`
头:

    $ ffprobe -show_entries format=duration camera.webm
    Duration: N/A, start: 0.000000, bitrate: N/A

⚠️ 这一步踩过两次,都值得记:

1. **`ffprobe` 返回 `N/A` 时 exit code 仍是 0** —— 脚本里 `if r.returncode != 0`
   判不出"没量到",于是整张表打印一排 `None`,看着像"这些场次没有文件"。
   必须显式判 `N/A` / 空串。
2. **`r_frame_rate` 在流式 webm 里是垃圾值**(实测 `1000/1`)。用它换算视频时长会
   得到「视/音 恒为 3.00」—— 那不是缺陷,是**算错了对象**。真实时长只能从
   **packet 的 pts 跨度**读。

所以本工具只信 `pts_time`:音轨末减首 = 音频秒,视轨末减首 = 视频秒。

判据(不引入任何拟合参数):

| 现象 | 判读 |
|---|---|
| 音/视 ≈ 1.0 | 正常 —— 两条轨同步 |
| 音 ≪ 视 | **音轨没录全**(文件里视频比音频长) |
| 管线跨度 ≫ 文件时长 | **录断了** —— 采集管线还在跑,但文件没收全 |
| 管线跨度 ≈ 文件时长 | 本来就短 —— 人录了几十秒就停了 |

用法:
    tools/jx_av_span.py                 # 全语料对账
    tools/jx_av_span.py --short 120     # 只看 <120 秒的场次,并给判读
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _recordings_root() -> Path:
    """录制根:环境变量优先(与 `media_retention.root()` 同一个名字)。"""
    env = os.environ.get("JINGXIN_RECORDINGS_DIR") or os.environ.get("JX_RECORDINGS")
    if env:
        return Path(env).expanduser()
    for c in (Path.home() / "JingXin/recordings",
              Path.home() / "shared/jingxin_recordings"):
        if c.exists():
            return c
    return Path.home() / "JingXin/recordings"


def pts_span(path: Path, stream: str) -> tuple[float, float] | None:
    """(首 pts, 末 pts)。量不到返回 None。

    ⚠️ 不要用 `format=duration`(流式 webm 是 N/A),也不要用 `r_frame_rate`
    (是垃圾值,见模块 docstring)。
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", stream,
             "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.SubprocessError):
        return None
    vals = []
    for x in r.stdout.split():
        if x and x != "N/A":
            try:
                vals.append(float(x))
            except ValueError:
                pass
    return (min(vals), max(vals)) if len(vals) >= 2 else None


def ledger_span(path: Path) -> float | None:
    """留存账本里 `declared_ts` 的跨度(秒)。`declared_ts` 是**毫秒**。"""
    if not path.exists():
        return None
    ts = []
    try:
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                x = json.loads(line).get("declared_ts")
            except ValueError:
                continue
            if x is not None:
                ts.append(float(x))
    except OSError:
        return None
    return (max(ts) - min(ts)) / 1000.0 if len(ts) >= 2 else None


def sessions(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and not p.name.startswith("_") and p.name != "NONE")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--short", type=float, default=None,
                    metavar="SEC", help="只看音频短于此值的场次,并给判读")
    args = ap.parse_args()

    root = args.root or _recordings_root()
    if not root.exists():
        print(f"❌ 录制根不存在:{root}", file=sys.stderr)
        return 2

    rows = []
    for d in sessions(root):
        webm = d / "media" / "camera.webm"
        if not webm.exists():
            continue
        a = pts_span(webm, "a:0")
        v = pts_span(webm, "v:0")
        rows.append((d, a, v))

    if not rows:
        print(f"❌ {root} 下没有任何 media/camera.webm", file=sys.stderr)
        return 2

    hdr = f"{'session':44s} {'音频秒':>8s} {'视频秒':>8s} {'音/视':>6s}"
    if args.short is not None:
        hdr += f" {'管线跨度':>9s}  判读"
    print(hdr)
    print("-" * len(hdr))

    bad = []
    for d, a, v in sorted(rows, key=lambda r: (r[1][1] - r[1][0]) if r[1] else 0):
        ad = (a[1] - a[0]) if a else None
        vd = (v[1] - v[0]) if v else None
        if args.short is not None and (ad is None or ad >= args.short):
            continue
        line = f"{d.name[:44]:44s}"
        line += f" {ad:8.2f}" if ad is not None else f" {'?':>8s}"
        line += f" {vd:8.2f}" if vd is not None else f" {'?':>8s}"
        if ad and vd:
            ratio = ad / vd
            line += f" {ratio:6.2f}"
            if not 0.8 <= ratio <= 1.2:
                bad.append((d.name, ad, vd, ratio))
        else:
            line += f" {'?':>6s}"
        if args.short is not None:
            pipe = max([x for x in (ledger_span(d / "retention.face.jsonl"),
                                    ledger_span(d / "retention.gesture.jsonl"))
                        if x is not None], default=None)
            line += f" {pipe:9.1f}" if pipe is not None else f" {'—':>9s}"
            if pipe is None:
                line += "  无法判读(无留存账本)"
            elif ad is None:
                line += "  无法判读(无音轨)"
            elif pipe <= ad * 1.5:
                line += "  本来就短"
            else:
                line += f"  ⚠️ 录断了?管线跑了 {pipe:.0f}s"
        print(line)

    if bad:
        print(f"\n音轨与视轨时长偏差 >20%:{len(bad)} 场")
        for name, ad, vd, ratio in bad:
            print(f"  {name}  音频 {ad:.1f}s / 视频 {vd:.1f}s = {ratio:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
