#!/usr/bin/env python
"""离线标边界(裁定②)的**可行性核算** —— 全部数字当场算,不手打。

## 它回答什么

「投入离线标边界(重跑转写补绝对原点 + 人工点边界)值不值?」

三件事必须分开算,混在一起就会得出错误结论:

1. **音频覆盖**:一场的 `media/camera.webm` 音轨覆盖了观测时间轴的多少。
   <90% 的场次,边界标了也只能覆盖那一段 —— 别把它算进"可补"。
2. **题数从哪来**:正式场没有 `questions.jsonl`,是老师**自由问答**,
   不是固定 8 题(`voice_interaction/pipeline/assessment_pipeline.py:101` 那份
   题库只用于机器提问流程)。所以"208 个边界"这个数**前提就不成立** ——
   要标的是实际发生的每一次问答,只能从转写里数。
3. **标什么**:标的是**面试官每次问完**的时刻(问-答交界),不是"第 k 题"。
   没有题库就没有 `qid`,只能取转写原文。

## 用法

    ~/anaconda3/envs/jingxin/bin/python tools/jx_boundary_feasibility.py
    ~/anaconda3/envs/jingxin/bin/python tools/jx_boundary_feasibility.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RECORDINGS = Path.home() / "JingXin" / "recordings"
TRANSCRIPTS = Path(__file__).resolve().parents[1] / "data" / "transcripts"

#: 音轨覆盖低于这个比例 = 截断场,边界只能标到覆盖到的那一段
FULL_COVER = 0.90


def _log_span_sec(d: Path) -> float:
    """观测时间轴跨度(秒),取面部留存账本的 `declared_ts`。

    ⚠️ `declared_ts` 是**毫秒**(`session_clock.py`),按秒读会得到 17843 分钟这种数。
    没取到面部账本时退回手势,再没有就 0(并让调用方看见)。
    """
    for name in ("retention.face.jsonl", "retention.gesture.jsonl"):
        f = d / name
        if not f.is_file():
            continue
        ts = []
        for line in f.open(encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            v = rec.get("declared_ts")
            if isinstance(v, (int, float)):
                ts.append(v)
        if len(ts) > 1:
            return (max(ts) - min(ts)) / 1000.0
    return 0.0


def survey(recordings: Path = RECORDINGS, transcripts: Path = TRANSCRIPTS) -> dict:
    """逐场核算。返回 {"sessions": [...], "totals": {...}}。"""
    from report_frontend.corpus import FORMAL_PATTERN

    rows: list[dict] = []
    if not recordings.is_dir():
        return {"sessions": [], "totals": {}, "error": f"素材根不存在:{recordings}"}

    for d in sorted(p for p in recordings.iterdir() if p.is_dir()):
        if d.name.startswith("_") or d.name == "NONE" or not FORMAL_PATTERN.search(d.name):
            continue
        sid = d.name.split("__")[-1]
        span = _log_span_sec(d)
        tf = transcripts / f"{sid}.json"
        row = {"session": d.name.split("__")[0], "sid": sid,
               "log_span_min": round(span / 60, 1),
               "audio_min": None, "cover": None, "segments": None,
               "transcript": tf.is_file()}
        if tf.is_file():
            try:
                j = json.loads(tf.read_text(encoding="utf-8"))
            except Exception:
                j = {}
            a = j.get("audio_seconds") or 0.0
            row["audio_min"] = round(a / 60, 1)
            row["cover"] = round(a / span, 3) if span else None
            row["segments"] = len(j.get("segments") or [])
        rows.append(row)

    # ⚠️ 三分类,不是两分类。`cover is None` 有两种来路,必须分开:
    #   · 没有转写 ⟹ 没转过,不是"截断";
    #   · 有转写但没有观测账本(实测 `孙偲齐`/`吴思诚` 两场)⟹ **无法核算**,
    #     不是"覆盖 0%"。把这两种并进"截断"会把 2 场误报成音频损坏。
    full = [r for r in rows if r["cover"] is not None and r["cover"] >= FULL_COVER]
    trunc = [r for r in rows if r["cover"] is not None and r["cover"] < FULL_COVER]
    unknown = [r for r in rows if r["cover"] is None and r["transcript"]]
    missing = [r for r in rows if not r["transcript"]]
    totals = {
        "n_formal": len(rows),
        "n_full_cover": len(full),
        "n_truncated": len(trunc),
        "n_unknown_span": len(unknown),
        "n_no_transcript": len(missing),
        # 可标边界数的**上界**:转写段数。实际问答次数 ≤ 段数(VAD 会把长回答切段,
        # 也会把一句提问切两段),所以这是"最多要标多少",不是"要标多少"。
        "segments_full_cover": sum(r["segments"] or 0 for r in full),
        "audio_min_full_cover": round(sum(r["audio_min"] or 0 for r in full), 1),
        # 截断场的音频覆盖比例,用来判断"标了也只覆盖一小段"的代价
        "truncated": [{"session": r["session"], "cover": r["cover"],
                       "log_span_min": r["log_span_min"], "audio_min": r["audio_min"]}
                      for r in trunc],
        "unknown_span": [{"session": r["session"], "audio_min": r["audio_min"]}
                         for r in unknown],
        "no_transcript": [r["session"] for r in missing],
    }
    return {"sessions": rows, "totals": totals}


def main() -> int:
    ap = argparse.ArgumentParser(description="离线标边界的可行性核算")
    ap.add_argument("--recordings", default=str(RECORDINGS))
    ap.add_argument("--transcripts", default=str(TRANSCRIPTS))
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    r = survey(Path(args.recordings).expanduser(), Path(args.transcripts).expanduser())
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return 0
    if r.get("error"):
        print(f"✗ {r['error']}", file=sys.stderr)
        return 2

    print(f"{'session':44s} {'log':>7s} {'audio':>7s} {'cover':>6s} {'seg':>4s}")
    for s in r["sessions"]:
        cov = f"{s['cover']*100:.0f}%" if s["cover"] is not None else "—"
        seg = s["segments"] if s["segments"] is not None else "—"
        aud = f"{s['audio_min']:.1f}m" if s["audio_min"] is not None else "—"
        print(f"{s['session'][:44]:44s} {s['log_span_min']:6.1f}m {aud:>7s} {cov:>6s} {str(seg):>4s}")

    t = r["totals"]
    print(f"\n正式场 {t['n_formal']}:音轨覆盖 ≥{FULL_COVER:.0%} 的 {t['n_full_cover']} 场,"
          f"截断 {t['n_truncated']} 场,观测账本缺失无法核算 {t['n_unknown_span']} 场,"
          f"无转写 {t['n_no_transcript']} 场")
    print(f"可标边界上界 = {t['segments_full_cover']} 段"
          f"(覆盖完整的 {t['n_full_cover']} 场,共 {t['audio_min_full_cover']:.0f} 分钟音频)")
    if t["truncated"]:
        print("\n截断场(标了也只覆盖音频那一段):")
        for x in t["truncated"]:
            cov = f"{x['cover']*100:.0f}%" if x["cover"] is not None else "0%"
            print(f"  {x['session'][:44]:44s} 观测 {x['log_span_min']:6.1f}m / 音频 {x['audio_min']:.1f}m ({cov})")
    if t["unknown_span"]:
        print("\n观测账本缺失(音频在,但算不出覆盖率 —— **不是**截断):")
        for x in t["unknown_span"]:
            print(f"  {x['session'][:44]:44s} 音频 {x['audio_min']:.1f}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
