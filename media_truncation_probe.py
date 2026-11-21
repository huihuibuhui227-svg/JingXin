#!/usr/bin/env python
"""媒体完整性探针 —— 把 `questions.jsonl` 的跨度与盘上媒体时长对上,调 `assess` 判定。

分工:
  * **判定**在 `media_truncation.assess`(纯函数,不读盘,可离线单测);
  * **量具**在 `capture_health._span_sec`(流式 webm 只能从 packet 的 pts 求跨度,
    见那个函数的注释:不用 `format=duration`,也不用 `r_frame_rate`);
  * 本文件只做**读盘 + 拼装**,不自己发明判据、不自己发明时长算法。

为什么必须复用 `capture_health._span_sec`:同一件事量两遍必然分叉(换编码器时一处认、
一处不认)。本仓对这件事栽过 —— 量具收进 `capture_health` 之后,`tools/jx_check_session.py`
就只许把它取回来用。

⚠️ **只读**。本文件绝不写 `session.json`、绝不改任何素材。要把结果落到
`session.json`,由调用方走 `capture_health.record` 那套"读-改-写 + os.replace"。

⚠️ **ffprobe 不在就优雅降级**:不猜、不估、不拿文件大小反推 —— 所有时长记 `None`,
`judgeable=False`,`gap_detail` 写明"没装 ffprobe,这一场没量",**不伪造通过**。

CLI:
    python media_truncation_probe.py <session_dir> [...]          # 文件级(默认)
    python media_truncation_probe.py --track audio <session_dir>  # 只按音轨算
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _c in (_HERE, _HERE.parent):  # 本文件被复制进仓库根之后也能自洽 import
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

import media_truncation as mt  # noqa: E402


# ── 量具:优先复用 capture_health,取不到才用等价实现 ──────────────────────

_CH_CACHE: list = []          # 用 list 当"已解析"标记(None 也要能缓存)
_CH_FALLBACK_ROOTS = (
    Path.home() / "JingXin/jingxin",
    Path.home() / "jingxin",
    _HERE.parent,
)


def _capture_health():
    """→ `capture_health` 模块;拿不到 → `None`。

    本补丁包要在**系统仓库只读**的条件下也能跑(& `python tools/...` 之外的地方),
    所以先普通 import;失败再按候选根目录找 `capture_health.py` 按文件加载。
    """
    if _CH_CACHE:
        return _CH_CACHE[0]
    mod = None
    try:
        import capture_health as mod  # noqa: F811 - 就是它
    except Exception:  # noqa: BLE001 - 任何 import 失败都退到按路径加载
        mod = None
    if mod is None:
        import os
        cands = []
        env = os.environ.get("JINGXIN_REPO") or os.environ.get("JX_REPO")
        if env:
            cands.append(Path(env).expanduser())
        cands.extend(_CH_FALLBACK_ROOTS)
        for root in cands:
            f = root / "capture_health.py"
            if not f.is_file():
                continue
            try:
                spec = importlib.util.spec_from_file_location("capture_health", f)
                assert spec and spec.loader
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)
            except Exception:  # noqa: BLE001
                continue
            mod = m
            break
    _CH_CACHE.append(mod)
    return mod


def _ffprobe() -> str | None:
    return shutil.which("ffprobe")


_PROBE_TIMEOUT = 180


def _span_sec(ffprobe: str, path: Path, stream: str) -> float | None:
    """这条轨的时长(秒)。没有这条轨 / 量不到 → `None`。

    **首选** `capture_health._span_sec`(系统里的唯一量具);它不在时用下面
    `_span_sec_local` —— 逻辑与它逐字等价(只读 packet 的 `pts_time` 求末减首),
    这是本补丁包能在仓库外跑的代价,不是第二套判据。
    """
    ch = _capture_health()
    if ch is not None and hasattr(ch, "_span_sec"):
        return ch._span_sec(ffprobe, path, stream)
    return _span_sec_local(ffprobe, path, stream)


def _span_sec_local(ffprobe: str, path: Path, stream: str) -> float | None:
    """`capture_health._span_sec` 的等价实现(兜底用)。

    ⚠️ 只信 packet 的 `pts_time`:
      * `format=duration` 对 Chrome MediaRecorder 的流式 webm 是 `N/A` 而
        **exit code 仍为 0** —— 用 returncode 判不出"没量到";
      * `r_frame_rate` 在这种文件里是垃圾值(实测恒 `1000/1`)。
    """
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", stream,
             "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=_PROBE_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None
    pts = []
    for x in (r.stdout or "").split():
        if x and x != "N/A":
            try:
                pts.append(float(x))
            except ValueError:
                pass
    if len(pts) < 2:
        return None
    return max(pts) - min(pts)


# ── 读盘 ────────────────────────────────────────────────────────────────

#: `track` 取值。`auto` = 视轨优先(录像的时长就是画面的时长),没有视轨才退音轨。
TRACK_AUTO, TRACK_VIDEO, TRACK_AUDIO = "auto", "video", "audio"
_STREAMS_FOR_TRACK = {
    TRACK_AUTO: ("v:0", "a:0"),
    TRACK_VIDEO: ("v:0",),
    TRACK_AUDIO: ("a:0",),
}

CAMERA_FILE = "camera.webm"
AUDIO_SUBDIR = "audio"
AUDIO_SUFFIXES = (".webm", ".wav")


def read_questions(session_dir: Path) -> list[dict] | None:
    """→ 题目行;`questions.jsonl` 不存在 → `None`;坏行跳过(不让一行毁一场)。"""
    p = Path(session_dir) / "questions.jsonl"
    if not p.is_file():
        return None
    rows: list[dict] = []
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def media_paths(session_dir: Path) -> dict[str, Path]:
    """本场的媒体文件 → `{相对标签: 路径}`。标签是给人看的,`probe` 用它当 key。"""
    d = Path(session_dir)
    out: dict[str, Path] = {}
    cam = d / "media" / CAMERA_FILE
    if cam.is_file():
        out[CAMERA_FILE] = cam
    audio_dir = d / "media" / AUDIO_SUBDIR
    if audio_dir.is_dir():
        try:
            for f in sorted(audio_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in AUDIO_SUFFIXES:
                    out[f"{AUDIO_SUBDIR}/{f.name}"] = f
        except OSError:
            pass
    return out


def _measure(path: Path, ffprobe: str, track: str) -> tuple[float | None, dict]:
    """量一份媒体的时长(按 `track` 选轨)。→ `(秒 | None, 明细)`。"""
    detail: dict = {}
    if path.name == CAMERA_FILE:
        # camera.webm 同时有视轨与音轨,**两个都量并都记下来**:
        # 实测 `20260926_153854_1592` 视轨 550.51s / 音轨 57.00s —— 只记一个数
        # 就会让人以为"整场录像只有 57 秒"(任务书里那条就是这个误读)。
        spans = {}
        for st in ("v:0", "a:0"):
            spans[st] = _round(_span_sec(ffprobe, path, st))
        detail["tracks"] = spans
        if spans.get("v:0") and spans.get("a:0"):
            detail["av_ratio"] = _round(spans["a:0"] / spans["v:0"], 3)
        for st in _STREAMS_FOR_TRACK[track]:
            if spans.get(st) is not None:
                return spans[st], detail
        return None, detail
    # 音频块:只有 a:0
    v = _span_sec(ffprobe, path, "a:0")
    detail["tracks"] = {"a:0": _round(v)}
    return v, detail


def probe_session(
    session_dir: Path,
    *,
    track: str = TRACK_AUTO,
    tol: float = mt.DEFAULT_TOL,
    ceiling: float = mt.DEFAULT_CEILING,
) -> dict:
    """量一场并判定。→ `assess()` 的结果 + 本场取证。**只读,永不抛异常。**

    `track`:
      * `"auto"`(默认)—— `camera.webm` 取**视轨**时长,没有视轨才退音轨。
        这是"整场录像有多长"的读数,`assess` 的 `media_sec` 就是它。
      * `"audio"` —— `camera.webm` 只取音轨。用它可以复现"按音轨算"的读数
        (`20260926_153854_1592`:520.7 / 57.0 = 9.13 → `truncated`)。
      * `"video"` —— 只取视轨;没有视轨就记 `None`(音频-only 场次会落到
        `judgeable=False`,而不是被判成"0 秒录像")。
    """
    if track not in _STREAMS_FOR_TRACK:
        raise ValueError(f"track 只能是 {sorted(_STREAMS_FOR_TRACK)},收到 {track!r}")
    d = Path(session_dir)
    session = d.name

    questions = read_questions(d)
    if not questions:
        # `None` = 文件不在(或读不了);`[]` = 文件在、但一道可解析的题都没有。
        # 两者都**算不出跨度**,所以都落到 `no_ledger`;原因分开写 ——
        # 「没交台账」与「台账是空的」补救动作不一样。
        why = ("没有 questions.jsonl" if questions is None
               else "questions.jsonl 里一道题都没有(空文件 / 全是坏行)")
        return {
            "session": session, "verdict": "no_ledger", "span_sec": None,
            "media_sec": None, "ratio": None, "gap": None, "judgeable": False,
            "gap_detail": f"{why} —— 算不出任务跨度,"
                          f"**(这既不是媒体坏了,也不是通过)**;先补台账再判",
            "evidence": {"session_dir": str(d), "track": track,
                         "media_sec_by_file": {}},
        }

    paths = media_paths(d)
    ffprobe = _ffprobe()
    media: dict[str, float | None] = {}
    per_file: dict[str, dict] = {}
    note = None
    if ffprobe is None:
        note = ("ffprobe 不在 PATH —— 本场一个时长都没量(不是媒体坏了);"
                "装上 ffmpeg/ffprobe 再跑,别用文件大小反推时长")
        media = {name: None for name in paths}
    else:
        for name, p in paths.items():
            sec, detail = _measure(p, ffprobe, track)
            media[name] = sec
            per_file[name] = detail
    if not paths:
        note = (note + ";" if note else "") + "本场没有任何 media 文件 —— 没有可比的媒体"
    has_camera = CAMERA_FILE in paths
    if paths and not has_camera:
        # ⚠️ 没有 `camera.webm` 的场次(音频-only / 只落了分题音频块)里,
        #    `media_sec` 取到的是**最长的那一块**,不是整场的连续录制时长。
        #    实测 `..._0d89`(跨度 288.9s)与 `..._302f`(125.4s)就是这种:它们真的
        #    没有整段录像,但这是**设计如此**(与 `capture_health` 的 `audio_only`
        #    同一件事),不是"录像被截断"。判读时必须带上这一句,否则又是一条假 warn。
        note = (note + ";" if note else "") + (
            "本场没有 camera.webm,只有分题音频块 —— media_sec 取的是最长**单块**"
            "(不是整场连续录制);音频-only 场次请按 audio_only 口径解读,别当截断")

    res = mt.assess(questions, media, tol=tol, ceiling=ceiling)
    ev = dict(res.get("evidence") or {})
    ev.update({
        "session_dir": str(d),
        "track": track,
        "ffprobe": ffprobe,
        "has_camera": has_camera,
        "span_source": ("capture_health._span_sec"
                        if (_capture_health() is not None
                            and hasattr(_capture_health(), "_span_sec"))
                        else "local fallback(_span_sec_local)"),
        "per_file": per_file,
        "fps_note": "时长只取 packet 的 pts 跨度;format=duration 对流式 webm 是 N/A",
    })
    if note:
        ev["probe_note"] = note
    out = dict(res)
    out["session"] = session
    out["evidence"] = ev
    return out


def _round(x: float | None, nd: int = 3) -> float | None:
    return None if x is None else round(x, nd)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="媒体完整性探针:题目跨度 vs 媒体时长(只读,输出 JSON 行)")
    ap.add_argument("sessions", nargs="+", type=Path, help="会话目录(可多个)")
    ap.add_argument("--track", choices=sorted(_STREAMS_FOR_TRACK), default=TRACK_AUTO,
                    help="camera.webm 取哪条轨(默认 auto=视轨优先)")
    ap.add_argument("--tol", type=float, default=mt.DEFAULT_TOL,
                    help=f"低于 tol*span 判截断(默认 {mt.DEFAULT_TOL})")
    ap.add_argument("--ceiling", type=float, default=mt.DEFAULT_CEILING,
                    help=f"高于 ceiling*span 判超长(默认 {mt.DEFAULT_CEILING})")
    args = ap.parse_args(argv)

    rc = 0
    for sd in args.sessions:
        if not sd.is_dir():
            print(json.dumps({"session": str(sd), "verdict": "no_session",
                              "gap_detail": f"目录不存在:{sd}"}, ensure_ascii=False))
            rc = 2
            continue
        try:
            res = probe_session(sd, track=args.track, tol=args.tol,
                                ceiling=args.ceiling)
        except Exception as exc:  # noqa: BLE001 - 一场跑挂不许带崩整批
            res = {"session": sd.name, "verdict": "probe_error",
                   "gap_detail": f"探针异常:{type(exc).__name__}: {exc}"}
            rc = 1
        print(json.dumps(res, ensure_ascii=False))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
