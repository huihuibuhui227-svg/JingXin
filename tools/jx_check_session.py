#!/usr/bin/env python3
"""按 `docs/下一步.md` §8.4 的判据,逐条核一场录制能不能进 M3。

用法:
    P=~/miniconda3/envs/jingxin/bin/python
    $P tools/jx_check_session.py <session_id 或 录制目录>
    $P tools/jx_check_session.py --all      # 把盘上所有会话都过一遍

判据原样照 §8.4 那六条,外加它「每场操作要求」里能从盘上验的两条
(答满题库 / meta.json 当场填了)。**不改判据本身** —— 要改先改文档。
"""
from __future__ import annotations
import os

import csv
import json
import subprocess
import sys
from pathlib import Path

REC_ROOT = Path.home() / "shared" / "jingxin_recordings"
LOG_DIR = Path.home() / "jingxin"
# 解释器:可用 JX_PY 覆盖(换机器时必改的一项)
PY = Path(os.environ.get("JX_PY", Path.home() / "miniconda3" / "envs" / "jingxin" / "bin" / "python"))

EXPECTED_QUESTIONS = 8   # §8.4 第 1 条:题库上限
MIN_ANSWERS = 5          # §8.4 第 1 条:语音族/密度族门槛 5
MIN_FRAMES = 300         # §8.4 判据:面部/手势各 ~400 帧
MIN_DETECT = 0.80        # §8.4 判据:检出率 ≥80%

OK, BAD, WARN = "✓", "✗", "!"


class Row:
    def __init__(self, name: str, ok: bool, detail: str, *, warn: bool = False):
        self.name, self.ok, self.detail, self.warn = name, ok, detail, warn

    def render(self) -> str:
        mark = WARN if (self.warn and not self.ok) else (OK if self.ok else BAD)
        return f"  {mark} {self.name:22s} {self.detail}"


def _load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _detect_rate(csv_path: Path) -> tuple[int, int, float]:
    """返回 (总行数, 检出行数, 检出率)。

    一行算「检出」= 除 session_id/timestamp/timestamp_iso 外**还有非空值**。
    没检出时服务端写的是空格(§8.1「没数据的槽写 None,logger 写空格」)。
    """
    if not csv_path.exists():
        return 0, 0, 0.0
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0, 0, 0.0
    skip = {"session_id", "timestamp", "timestamp_iso"}
    hit = 0
    for r in rows:
        if any((v or "").strip() for k, v in r.items() if k not in skip):
            hit += 1
    return len(rows), hit, hit / len(rows)


def _ffprobe_codecs(path: Path) -> list[str]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=60, check=True).stdout
        return [c.strip() for c in out.splitlines() if c.strip()]
    except Exception as e:  # noqa: BLE001 - 探针失败就是判据不过,原因照实写
        return [f"<ffprobe 失败: {e}>"]


def check(sid: str) -> tuple[list[Row], dict]:
    d = REC_ROOT / sid
    rows: list[Row] = []
    facts: dict = {"sid": sid, "dir": str(d)}

    if not d.is_dir():
        return [Row("会话目录", False, f"不存在: {d}")], facts

    # ── 1. questions.jsonl:index 连续无漏报 ───────────────────────────
    qs = _load_jsonl(d / "questions.jsonl")
    idx = [q.get("index") for q in qs]
    facts["n_questions"] = len(qs)
    if not qs:
        rows.append(Row("题目台账", False, "questions.jsonl 空/缺失 —— 题号与提问窗口全无"))
    else:
        want = list(range(len(qs)))
        good = idx == want
        rows.append(Row("题目台账 index", good,
                        f"{len(qs)} 题, index={idx}" + ("" if good else f" —— 应为 {want}(漏报/重复)")))

    # ── 2. transcript.json:段数、有无重复 ────────────────────────────
    tp = d / "transcript.json"
    segs = []
    if tp.exists():
        try:
            segs = json.loads(tp.read_text(encoding="utf-8")).get("segments") or []
        except json.JSONDecodeError:
            segs = []
    facts["n_segments"] = len(segs)
    if not segs:
        rows.append(Row("转写", False, "transcript.json 无 segments"))
    else:
        dup = sum(1 for a, b in zip(segs, segs[1:])
                  if (a.get("text") or "").strip() == (b.get("text") or "").strip())
        facts["n_dup_segments"] = dup
        rows.append(Row("转写段数", dup == 0,
                        f"{len(segs)} 段" + (f",其中 {dup} 处相邻重复(§0.6 那个双提交的形态)" if dup else ",无相邻重复"),
                        warn=True))

    # ── 3. media/camera.webm + vp8/opus 双流 ─────────────────────────
    cam = d / "media" / "camera.webm"
    if not cam.exists():
        rows.append(Row("原生视频", False, "media/camera.webm **不存在** —— §8.4 第 2 条:没走到结束页"))
    else:
        codes = _ffprobe_codecs(cam)
        mb = cam.stat().st_size / 1e6
        ok = "vp8" in codes and "opus" in codes
        rows.append(Row("原生视频", ok, f"camera.webm {mb:.1f} MB, 流={codes}"))

    # ── 4. 面部 / 手势:帧数 + 检出率 ─────────────────────────────────
    for mod, csv_name in (("face", f"face_au_log_{sid}.csv"),
                          ("gesture", f"gesture_emotion_log_{sid}.csv")):
        n_frames = len(_load_jsonl(d / f"retention.{mod}.jsonl"))
        n_rows, n_hit, rate = _detect_rate(LOG_DIR / "data" / "logs" / csv_name)
        facts[f"{mod}_frames"], facts[f"{mod}_detect"] = n_frames, rate
        if n_frames == 0 and n_rows == 0:
            rows.append(Row(f"{mod} 帧", False, "留存与日志都没有 —— 这一模态整场没上来"))
            continue
        ok = n_frames >= MIN_FRAMES and rate >= MIN_DETECT
        rows.append(Row(f"{mod} 帧/检出", ok,
                        f"留存 {n_frames} 帧, 日志 {n_rows} 行, 检出 {n_hit} ({rate:.0%})"
                        + ("" if ok else f" —— 要 ≥{MIN_FRAMES} 帧且 ≥{MIN_DETECT:.0%}")))

    # ── 5. 语音:行数 = 答题数,三个新列逐行有值 ───────────────────────
    # ⚠️ 两个端点各写各的日志:`/interview/answer_audio` → interview_*,
    # `/research/answer_audio` → research_*。两边的**列完全一样**(都 31 列),
    # 只是文件名不同 —— 只认前者会把走科研端点的一整场误判成"语音特征一个都没有"。
    vp, vkind = None, None
    for kind in ("interview", "research"):
        cand = LOG_DIR / "data" / "logs" / f"{kind}_emotion_log_{sid}.csv"
        if cand.exists():
            vp, vkind = cand, kind
            break
    if vp is None:
        rows.append(Row("语音行", False,
                        f"interview/research_emotion_log_{sid}.csv 都不存在 —— 语音特征一个都没有"))
    else:
        with vp.open(encoding="utf-8-sig", newline="") as f:
            vr = list(csv.DictReader(f))
        ans = facts.get("n_questions", len(qs))
        facts["n_voice_rows"] = len(vr)
        cols = ("n_chars", "chars_per_sec", "reaction_time")
        blank = {c: sum(1 for r in vr if not (r.get(c) or "").strip()) for c in cols}
        # reaction_time 允许为空(那一题没有可配的提问窗口);n_chars/chars_per_sec 不该空
        bad = [f"{c} 空 {blank[c]}/{len(vr)}" for c in ("n_chars", "chars_per_sec") if blank[c]]
        rows.append(Row("语音行数", len(vr) == ans,
                        f"{len(vr)} 行 vs 答题 {ans} 题 ({vkind} 端点)"
                        + ("" if len(vr) == ans else " —— 对不上")))
        rows.append(Row("语音新列", not bad,
                        ("; ".join(bad) if bad else
                         f"n_chars/chars_per_sec 逐行有值; reaction_time 共 {len(vr) - blank['reaction_time']}/{len(vr)} 行有值"),
                        warn=True))

    # ── 6. meta.json:写了没 / 填了没 ────────────────────────────────
    mp = d / "meta.json"
    if not mp.exists():
        rows.append(Row("meta.json", False, "**没写** —— §8.4 第 7 条:录完就补不回来"))
    else:
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        def filled(v):
            # ⚠️ False 不算「填了」—— `consent.archived: false` 是**没同意**,
            # 把它算成填过会放过一场伦理上不能用的素材。
            return v not in (None, "", [], {}, False)
        parts = {
            "性别年龄": filled(meta.get("candidate", {}).get("sex")) or filled(meta.get("candidate", {}).get("age")),
            "设备": filled(meta.get("capture", {}).get("device")),
            "距离/光照": filled(meta.get("capture", {}).get("camera_distance_cm")) or filled(meta.get("capture", {}).get("lighting")),
            "面试官评分": any(filled(v) for v in (meta.get("interviewer_ratings") or {}).values()),
            "知情同意": filled(meta.get("consent", {}).get("archived")) or filled(meta.get("consent", {}).get("path")),
        }
        facts["meta_filled"] = [k for k, v in parts.items() if v]
        n_fill = len(facts["meta_filled"])
        rows.append(Row("meta.json", n_fill > 0,
                        f"存在,填了 {n_fill}/{len(parts)} 项" + (f":{facts['meta_filled']}" if n_fill else " —— 是空模板"),
                        warn=True))

    # ── 7. 收尾对账(session_meta --check-session)───────────────────
    try:
        p = subprocess.run(
            [str(PY), "-m", "session_meta", "--check-session", sid,
             "--expected-questions", str(EXPECTED_QUESTIONS)],
            cwd=str(LOG_DIR), capture_output=True, text=True, timeout=120)
        out = (p.stdout + p.stderr).strip()
        facts["check_session_rc"] = p.returncode
        # 只把「缺项」行拎出来展示
        # ⚠️ 要解析 `meta.json 缺项:[...]` 那一行的**完整列表** ——
        #    早先这里只 grep 关键字再截断,把 5 条缺项显示成"共 3 条",
        #    **少报的恰好是最后那两条**。缺项清单少报 = 让人以为快齐了。
        miss = None
        for ln in out.splitlines():
            s = ln.strip()
            if s.startswith("meta.json 缺项:"):
                miss = s.split(":", 1)[1].strip()
        if miss:
            try:
                facts["meta_missing"] = eval(miss)  # noqa: S307 - 自己服务端打的 list 字面量
            except Exception:  # noqa: BLE001
                facts["meta_missing"] = []
            n = len(facts["meta_missing"])
            rows.append(Row("收尾对账", n == 0, f"meta.json 缺 {n} 项: {miss[:220]}"))
        else:
            rows.append(Row("收尾对账", p.returncode == 0, f"rc={p.returncode}"))
    except Exception as e:  # noqa: BLE001
        rows.append(Row("收尾对账", False, f"跑不起来: {e}"))

    return rows, facts


# 这两类**结构性拿不到**,不是漏填,补也只能靠外部:
#   · mic_gain_db —— 浏览器不向页面暴露麦克风增益(实测 enumerateDevices/getSettings 都没有);
#   · interviewer_ratings.* —— 需要一个**独立观察者**打分。你自己填会变成自评量表,
#     而「M3 之前不采自评量表」是强制项(录制需求 §2 前提 3)。
UNOBTAINABLE = {"capture.mic_gain_db",
                "interviewer_ratings.logical_thinking",
                "interviewer_ratings.communication",
                "interviewer_ratings.confidence"}


def verdict(rows: list[Row], facts: dict) -> str:
    """一句话:这场够不够格进 M3。"""
    hard = [r for r in rows if not r.ok and not r.warn]
    soft = [r for r in rows if not r.ok and r.warn]
    n_q = facts.get("n_questions", 0)
    if any(x.name == "原生视频" for x in hard):
        return "不够格 —— 没有原生视频(没走到结束页),这一场只能当试跑"
    if n_q and n_q < MIN_ANSWERS:
        return f"不够格 —— 只答了 {n_q} 题,低于门槛 {MIN_ANSWERS}(正式要 {EXPECTED_QUESTIONS})"
    if hard:
        # 只剩收尾对账、且缺的全是"结构性拿不到"的那几项 ⟹ 素材本身没问题,
        # 别用"不够格"吓人;但也**不许说成齐了** —— 缺什么照实列。
        if all(r.name == "收尾对账" for r in hard):
            left = [k for k in facts.get("meta_missing", []) if k not in UNOBTAINABLE]
            if not left:
                return ("素材够格进 M3 —— 判据全过。meta 还缺 %d 项,但全是结构性拿不到的"
                        "(mic_gain_db 浏览器不暴露;面试官评分需独立观察者,自评是红线)"
                        % len(facts.get("meta_missing", [])))
            return "不够格 —— meta 还缺可填却没填的:" + "、".join(left)
        return "不够格 —— 硬项没过:" + "、".join(r.name for r in hard)
    if soft:
        return "可作为素材,但有该补的:" + "、".join(r.name for r in soft)
    return f"够格进 M3(答满 {n_q} 题,判据全过)"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    if sys.argv[1] == "--all":
        sids = sorted(p.name for p in REC_ROOT.iterdir() if p.is_dir())
    else:
        arg = sys.argv[1].rstrip("/")
        sids = [Path(arg).name if os.sep in arg else arg]

    bad = 0
    for sid in sids:
        rows, facts = check(sid)
        v = verdict(rows, facts)
        if not v.startswith("够格"):
            bad += 1
        print(f"\n═══ {sid} ═══")
        for r in rows:
            print(r.render())
        print(f"  → {v}")
    if len(sids) > 1:
        print(f"\n── {len(sids)} 场里 {len(sids) - bad} 场够格 ──")
    return 0


if __name__ == "__main__":
    sys.exit(main())
