#!/usr/bin/env python3
"""一场新会话开始时铺 meta.json 模板,并把**机器能测到的字段**先填进去。

用法:
    P=~/miniconda3/envs/jingxin/bin/python
    $P tools/jx_new_session.py <session_id>
    $P tools/jx_new_session.py --latest      # 用盘上最新的那个会话目录

只填**实测到的机器事实**(主机/摄像头/分辨率/采样率/录音时间)。
人的属性(性别年龄母语方言)、取景距离、面试官评分**一律留空等人填** ——
往协变量里塞估计值,正是本项目一直在杀的那种东西。
"""
from __future__ import annotations
import os

import datetime
import json
import pathlib
import subprocess
import sys

REC_ROOT = pathlib.Path.home() / "shared" / "jingxin_recordings"
REPO = pathlib.Path.home() / "jingxin"
# 解释器:可用 JX_PY 覆盖(换机器时必改的一项)
PY = pathlib.Path(os.environ.get("JX_PY", pathlib.Path.home() / "miniconda3" / "envs" / "jingxin" / "bin" / "python"))

# 2026-09-26 经浏览器 getSettings + Windows PnP 实测所得(非估计)
MEASURED = {
    "device": "Integrated Camera (04f2:b7b8) / LENOVO 83JJ 内置",
    "resolution": "1280x720 @30fps",
    "audio_sample_rate": 48000,
}
# 候选人自报的协变量 —— 2026-09-26 使用者口述。**同一台机器上的同一个人**才适用;
# 换人录必须改。脚本会把它填了什么明确打出来,不静默沿用。
CANDIDATE = {
    "candidate.sex": "男",
    "candidate.age": 21,
    "candidate.native_language": "中文",
    "candidate.dialect_region": "普通话",
    "capture.camera_distance_cm": 100,
    "capture.lighting": "略偏暗",
    "consent.archived": True,
}
CONSENT_NOTE = ("知情同意:候选人当场同意已给出,archived=true。"
                "⚠️ 暂无纸质签署件/扫描件(2026-09-26 确认),consent.path 留空;"
                "有归档件后填 path 即可,archived 无需改动。")

NOTE = (
    "机器实测(WSL 侧经浏览器 getSettings 读出,非估计): "
    "主机 LENOVO 83JJ;摄像头 Integrated Camera (04f2:b7b8),1280x720@30fps 已是该头上限;"
    "麦克风 AB17X USB Audio 耳机式麦克风,48000Hz 单声道。"
    "⚠️ 本机另有一路 ToDesk Camera(虚拟摄像头),实测应用未选中它。"
    "⚠️ camera_distance_cm / mic_gain_db 无法机器测量,待人填。"
)


def newest_sid() -> str:
    dirs = [p for p in REC_ROOT.iterdir() if p.is_dir() and p.name[0].isdigit()]
    return max(dirs, key=lambda p: p.stat().st_mtime).name


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("sid", nargs="?", default="")
    ap.add_argument("--blank", action="store_true",
                    help="候选人不沿用上次的值(换了人录时用)")
    args = ap.parse_args()
    sid = args.sid
    if sid in ("--latest", ""):
        sid = newest_sid()
    d = REC_ROOT / sid
    if not d.is_dir():
        print(f"✗ 没有这个会话目录: {d}")
        return 2

    mp = d / "meta.json"
    if not mp.exists():
        r = subprocess.run([str(PY), "-m", "session_meta", "--write-template", sid],
                           cwd=str(REPO), capture_output=True, text=True)
        print((r.stdout + r.stderr).strip())

    m = json.loads(mp.read_text(encoding="utf-8"))
    m["recorded_at"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    m["capture"].update(MEASURED)
    if not args.blank:
        for dotted, val in CANDIDATE.items():
            cur, *rest = dotted.split(".")
            path_, key = (cur, rest[0]) if rest else (None, cur)
            (m if path_ is None else m.setdefault(path_, {}))[key] = val
        if CONSENT_NOTE not in (m.get("notes") or ""):
            m["notes"] = ((m.get("notes") or "").rstrip() + "\n" + CONSENT_NOTE).strip()
    m["notes"] = NOTE if args.blank else (NOTE + "\n" + CONSENT_NOTE)

    # 题目块(§3.2 必填):从 questions.jsonl 取题目原文,难度题库没有 → 留空
    qf = d / "questions.jsonl"
    if qf.exists():
        try:
            rows = [json.loads(l) for l in qf.read_text(encoding="utf-8").strip().splitlines()]
            if rows:
                m["questions"] = [{"index": r["index"], "qid": r["qid"], "difficulty": None}
                                  for r in sorted(rows, key=lambda x: x["index"])]
        except Exception:  # noqa: BLE001 - 会话刚开始时台账可能只有半行
            pass
    mp.write_text(json.dumps(m, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n✅ {sid}")
    print(f"   Windows: D:\\Shared\\jingxin_recordings\\{sid}\\meta.json")
    print("   机器实测:", "、".join(MEASURED) + "、recorded_at")
    if args.blank:
        print("   候选人  : **未填**(--blank)")
    else:
        print("   候选人  :", "、".join(f"{k.split('.')[-1]}={v!r}" for k, v in CANDIDATE.items()))
        print("             ⚠️ 这是**沿用 2026-09-26 你口述的那套**;换人录必须改。")
    print("   题目块  :", f"{len(m.get('questions') or [])} 题(难度留空 —— 题库里没有)")
    print("   拿不到  : capture.mic_gain_db(浏览器不暴露)、interviewer_ratings.*"
          "(需独立观察者;自评是红线)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
