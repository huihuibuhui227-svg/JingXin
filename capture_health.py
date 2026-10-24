"""录制体检 —— 录完当场问一句「这一场真的录到了吗」。

为什么要有它(`docs/素材体检_2026-10-03.md` §7.2 那条建议,2026-10-24 落地):

盘上有 37 场录像,其中**2 场根本没有视频轨**、**1 场音轨只有视轨的十分之一**、
**23 场面部检出率是 0%**。这些**在录的时候就已经废了**,而当时没有任何提示 ——
白录一场,事后只能重录(原始素材不可再生)。

判据只用**物理下限**,不引入任何拟合参数(本仓硬约束):

| 检查 | 判据 | 说明 |
|---|---|---|
| 有视频轨 | 流表里有 `codec_type=video` | 不是"脸检不到",是**文件里就没有视频** |
| 有音轨 | 流表里有 `codec_type=audio` | 没有音轨 ⟹ 转写与语音指标全不可出 |
| 音视同步 | 两者时长比在 [0.8, 1.2] | 只报事实,不猜原因 |
| 有画面 | 抽帧的平均亮度 > 5 | 全黑 = 摄像头被挡/断电 |

⚠️ **本模块绝不阻塞留存**。它只在文件**已经落盘之后**读一遍,结果写进
`session.json` 的 `capture_health` 块。探针失败(`ffprobe` 不存在、文件坏)
只记 `status="unknown"`,**不许**让上传返回错误 —— 留存优先。

⚠️ 也**不许**把"没检出人脸"当成体检项:那是**分析层**的结论(而且要先知道
取景协议),不是"录制本身坏了"。这里只判"文件里有没有这条轨、有没有画面"。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

#: 平均亮度下限。全黑帧的物理下限 —— 不是拟合出来的阈值。
#: 实测(2026-10-24,44 场):正常场次抽帧平均亮度 73~176,没有一场低于 5。
MIN_MEAN_LUMA = 5.0

#: 音视时长比的容忍区间。**只报事实**,超区间不判定谁对谁错。
AV_RATIO_LO, AV_RATIO_HI = 0.8, 1.2

#: 抽几帧测亮度。3 帧够判"全黑",而且几乎不花时间。
LUMA_SAMPLES = 3

#: ffprobe/ffmpeg 的超时。最大的一份素材 311 MB 实测 1.77 s;留足余量,
#: 但**必须**有上限 —— 挂住的探针会把一次上传的响应永远吊着。
PROBE_TIMEOUT = 180

STATUS_OK, STATUS_WARN, STATUS_UNKNOWN = "ok", "warn", "unknown"


def _ffprobe() -> str | None:
    return shutil.which("ffprobe")


def _streams(ffprobe: str, path: Path) -> list[dict] | None:
    """→ `[{"type": "video", "codec": "vp8"}, ...]`;`None` = ffprobe 读不懂这个文件。

    用 `codec_type` 而不是去猜 codec 名:猜名字(`vp8`/`opus`)会把「换了编码器」
    误判成「没有视频轨」—— 而这里要判的是**有没有这条轨**,不是它用什么编的。
    """
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "stream=codec_type,codec_name",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=PROBE_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    try:
        obj = json.loads(r.stdout or "{}")
    except json.JSONDecodeError:
        return None
    return [{"type": s.get("codec_type") or "?",
             "codec": s.get("codec_name") or "?"}
            for s in (obj.get("streams") or [])]


def _span_sec(ffprobe: str, path: Path, stream: str) -> float | None:
    """这条轨的时长(秒)。没有这条轨 / 量不到 → None。

    ⚠️ 不用 `format=duration`:Chrome MediaRecorder 产的**流式 webm 没有时长头**,
    `ffprobe` 会给 `N/A` 而 **exit code 仍是 0**(实测两次踩过)。所以只能读 packet
    的 pts 求跨度。
    ⚠️ 也不用 `r_frame_rate` 去反推:流式 webm 里它是垃圾值(实测恒为 `1000/1`),
    拿它算比值会得到"每场都恰好 3.00"这种看着像发现、其实是量具坏了的数。
    """
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", stream,
             "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=PROBE_TIMEOUT)
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


def _mean_luma(path: Path) -> float | None:
    """抽几帧算平均亮度。量不到返回 None(**不判**"全黑")。

    灰度 rawvideo 的每字节就是一个像素的亮度(0~255),取算术平均即可。
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    try:
        r = subprocess.run(
            [ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0",
             "-vf", "select='not(mod(n\\,200))',scale=32:32,format=gray",
             "-frames:v", str(LUMA_SAMPLES), "-f", "rawvideo", "-"],
            capture_output=True, timeout=PROBE_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None
    raw = r.stdout or b""
    if not raw:
        return None
    return sum(raw) / len(raw)


def probe(path: str | Path, *, audio_only: bool = False) -> dict:
    """量一份录像。→ 一个**只含事实**的 dict,**永远不抛异常**。

    字段:
      status      "ok" / "warn" / "unknown"
      has_video   有视频轨?(None = 没量成)
      has_audio   有音轨?
      video_sec   视频时长(秒,None = 量不到)
      audio_sec   音频时长(秒)
      av_ratio    音频/视频,None = 缺一条轨或量不到
      mean_luma   抽帧平均亮度,None = 量不到
      codecs      流表里探到的编码名(排查用)
      findings    人话列表 —— **每条都能直接照着修**

    `audio_only`:本场受试者只同意音频,前端**根本没打开摄像头** ⟹ 没有视频轨是
    **设计如此**,不是录坏了。不区分的话,每一场 audio-only 的 session.json 都会
    带一条假的 warn,而假 warn 会淹掉真的录坏。默认 `False` —— 老调用方
    (`tools/jx_check_session.py`)的行为一个字都不变。
    """
    p = Path(path)
    out: dict = {
        "status": STATUS_UNKNOWN, "has_video": None, "has_audio": None,
        "video_sec": None, "audio_sec": None, "av_ratio": None,
        "mean_luma": None, "codecs": [], "findings": [],
    }
    ffprobe = _ffprobe()
    if not ffprobe:
        out["findings"].append("机器上没有 ffprobe —— 这一场没做体检(不是录坏了)")
        return out
    if not p.is_file():
        out["findings"].append(f"文件不存在:{p.name} —— 录制没走完?")
        return out

    streams = _streams(ffprobe, p)
    if streams is None:
        out["findings"].append(
            "ffprobe 读不懂这个文件 —— 可能没写完、或者根本不是 webm。"
            "**这不等于「没有视频轨」**,先看文件大小与来源。")
        return out
    if not streams:
        out["findings"].append(
            "文件里一条音视频流都没有 —— 录制中途坏了(有字节、没有流)。")
        return out

    out["has_video"] = any(s["type"] == "video" for s in streams)
    out["has_audio"] = any(s["type"] == "audio" for s in streams)
    out["codecs"] = [s["codec"] for s in streams]

    if out["has_video"]:
        out["video_sec"] = _round(_span_sec(ffprobe, p, "v:0"))
    if out["has_audio"]:
        out["audio_sec"] = _round(_span_sec(ffprobe, p, "a:0"))

    if not out["has_video"] and not audio_only:
        out["findings"].append(
            "**文件里没有视频轨** —— 视觉指标与面部/手势全不可出。"
            "这不是「脸检不到」,是压根没有画面;音频可能完整,别急着整场重录。")
    if not out["has_audio"]:
        out["findings"].append(
            "**文件里没有音轨** —— 转写与语音指标全不可出。")

    if out["video_sec"] and out["audio_sec"]:
        ratio = out["audio_sec"] / out["video_sec"]
        out["av_ratio"] = round(ratio, 3)
        if not (AV_RATIO_LO <= ratio <= AV_RATIO_HI):
            out["findings"].append(
                f"音轨与视轨时长对不上(音频 {out['audio_sec']}s / 视频 "
                f"{out['video_sec']}s = {ratio:.2f})—— 只报事实:两条轨没录成一样长。")

    if out["has_video"]:
        luma = _mean_luma(p)
        out["mean_luma"] = _round(luma, 1)
        if luma is not None and luma < MIN_MEAN_LUMA:
            out["findings"].append(
                f"抽帧平均亮度只有 {luma:.1f}(低于物理下限 {MIN_MEAN_LUMA})—— "
                f"**画面可能全黑**:摄像头被挡、断电,或者镜头盖没开。")

    if out["findings"]:
        out["status"] = STATUS_WARN
    elif out["has_audio"] and (out["has_video"] or audio_only):
        # ⚠️ `or audio_only`:只采声音的场次**按设计**没有视频轨,有一条音轨就是
        #    这一场该有的全部。少了这半句,每一场 audio-only 都会落到 warn。
        out["status"] = STATUS_OK
    # 只剩一条轨且量不到时长 → 仍然是 warn(findings 非空),不会落到这里。
    return out


def _round(x: float | None, nd: int = 2) -> float | None:
    return None if x is None else round(x, nd)


def summarize(result: dict) -> str:
    """一句话结论,给日志/面板用。"""
    st = result.get("status")
    if st == STATUS_OK:
        return (f"录制体检通过(视频 {result.get('video_sec')}s / 音频 "
                f"{result.get('audio_sec')}s,亮度 {result.get('mean_luma')})")
    if st == STATUS_WARN:
        return "录制体检有告警:" + "；".join(result.get("findings") or [])
    return ("录制体检未完成(不是录坏了):"
            + "；".join(result.get("findings") or ["原因未知"]))


def record(session_json: str | Path, result: dict) -> bool:
    """把体检结果写进 `session.json` 的 `capture_health` 块。→ 写成了吗。

    ⚠️ **永远不抛异常**,也**不许**成为上传失败的理由:它是体检,不是留存。
    写不进去(清单不存在 / 盘只读)就返回 False,由调用方记日志。

    读-改-写用临时文件 + `os.replace`(同盘原子):并发写清单的
    `ensure_manifest`/`refresh_manifest` 不会读到半个 JSON。
    """
    p = Path(session_json)
    tmp = None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        payload["capture_health"] = result
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".session.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)
        return True
    except Exception:  # noqa: BLE001 - 体检写不进去不许影响任何别的事
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return False
