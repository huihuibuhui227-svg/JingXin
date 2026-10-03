"""拿留存**视频的音轨**走本机 ASR,把语言文本采出来。

## 为什么需要它(2026-10-03 实测定性)

09-28/09-29 的 **26 场**「老师面试学生」素材,每一场的 `session.json` 都声明
`logs.voice.expected=True`,而语音日志 `interview_emotion_log_<sid>.csv`
是**空表头(0 数据行)** —— 实测 26/26 全空,且没有任何关卡报警。

这批素材**没有 `questions.jsonl`**,因为**不是机器提问**:是老师面对面面试学生的
完整自然过程。⟹ 三条结论:

1. **`reaction_time` 定义上不存在** —— 它 = 首次开口墙钟 − 该题 `ask_end`;
   机器没提问就没有 `ask_end`。按本仓"不许编"的规矩,**不得伪造**,该槽对这批素材永久不适用。
2. **不能按「第几题」切分** —— 没有题目台账。本工具因此**只产出会话级文本**,
   不产出逐题列(`n_chars` / `connective_density` 那族的"每回答一个样本"口径在此不成立)。
3. **音轨是唯一的语音来源** —— `media/camera.webm` 里带 opus 音轨,
   实测 26/26 都有;而留存音频段(`media/audio/`)这 26 场**一场都没有**。

## 它**不**做什么(免得被误用)

- 不重算 `pitch`/`energy` 等韵律特征 —— 那要 `prosody_features_from_pcm`,是另一件事。
- 不区分话者 —— 音轨里老师和学生都在。转写是**两人混合**的,谁说的要靠人工或
  额外的话者分离,本工具**不假装**能分。
- 不写进报告层 —— 产出只落在 `data/transcripts/`(已在 `.gitignore` 里)。

## 用法

    PY=~/anaconda3/envs/jingxin/bin/python
    $PY tools/jx_transcribe_recordings.py --limit 1     # 先试一场
    $PY tools/jx_transcribe_recordings.py               # 全部(幂等,已转的跳过)
    $PY tools/jx_transcribe_recordings.py --force       # 重转
    $PY tools/jx_transcribe_recordings.py --sessions-dir ~/JingXin/recordings
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RECORDINGS = Path.home() / "JingXin" / "recordings"
OUT_DIR = REPO / "data" / "transcripts"

# 带真实姓名的场次目录形如 `20260929_195403-2-姬欣怡-261640123-投资学__20260929_195403_040b`
NAMED_RE = re.compile(r"^\d{8}_\d{6}-\d+-")


def _sessions(root: Path) -> list[Path]:
    """待转场次 = 带真实姓名 **且** 有 camera.webm 的目录。

    为什么要求带姓名:匿名场次(`20260925_225936_a971`)是早期的受控流程,
    它们要么已有留存音频段、要么已有语音日志,不属于本工具要救的那一批。
    用 `camera.webm` 存在与否做硬门槛 —— 没视频就没音轨,转了也是空。
    """
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if not NAMED_RE.match(d.name):
            continue
        if (d / "media" / "camera.webm").is_file():
            out.append(d)
    return out


def _extract_pcm(video: Path, wav_path: Path) -> float:
    """视频 → 16 kHz / 16 bit / 单声道 wav。返回秒数。

    ⚠️ 用解码实测时长,不读容器头:MediaRecorder 产的 webm **头里没有 duration**
    (`ffprobe -show_entries format=duration` 返回 `N/A`),照它算会得到 0。
    """
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg 抽音轨失败(退出 {r.returncode}):{r.stderr.strip()[:300]}")
    with wave.open(str(wav_path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def _label(d: Path) -> dict:
    p = d / "label.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def transcribe_one(d: Path, force: bool = False) -> dict:
    """转一场。返回结果字典(`status` 为 `ok` / `skipped` 时含落盘路径)。"""
    sid = d.name.split("__")[-1]
    out_json = OUT_DIR / f"{sid}.json"
    out_txt = OUT_DIR / f"{sid}.txt"

    if out_json.is_file() and not force:
        return {"session": d.name, "sid": sid, "status": "skipped",
                "json": str(out_json), "txt": str(out_txt)}

    # 延迟导入:没装依赖时也不该在 --help 上炸
    from voice_interaction.asr.transcribe import transcribe

    video = d / "media" / "camera.webm"
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "a.wav"
        dur = _extract_pcm(video, wav)
        with wave.open(str(wav), "rb") as w:
            pcm = w.readframes(w.getnframes())
    t_extract = time.time() - t0

    t1 = time.time()
    utt = transcribe(pcm)
    t_asr = time.time() - t1

    lab = _label(d)
    payload = {
        "session_dir": d.name,
        "session_id": sid,
        "name": lab.get("name"),
        "student_id": lab.get("student_id"),
        "department": lab.get("department"),
        "recorded_at_wall": lab.get("recorded_at_wall"),
        "consent": lab.get("consent"),
        "audio_seconds": round(dur, 2),
        "text": utt.text,
        "n_chars": utt.n_chars,
        "n_segments": utt.n_segments,
        "vad_split": utt.vad_split,
        "segments": utt.segments,
        "extract_seconds": round(t_extract, 1),
        "asr_seconds": round(t_asr, 1),
        "source": "media/camera.webm (opus 音轨)",
        # ⚠️ 显式声明这条产物的边界,免得后来人拿它当"逐题语音特征"用
        "notes": [
            "会话级转写,非逐题;本场无 questions.jsonl(非机器提问)",
            "reaction_time 定义上不适用,未产出且不得伪造",
            "音轨含老师与学生两人,本文本未做话者分离",
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_txt.write_text(utt.text or "", encoding="utf-8")
    return {"session": d.name, "sid": sid, "status": "ok", "audio_s": round(dur, 1),
            "n_chars": utt.n_chars, "asr_s": round(t_asr, 1),
            "json": str(out_json), "txt": str(out_txt)}


def main() -> int:
    ap = argparse.ArgumentParser(description="拿留存视频的音轨走本机 ASR,采出语言文本")
    ap.add_argument("--sessions-dir", default=str(DEFAULT_RECORDINGS))
    ap.add_argument("--limit", type=int, default=0, help="只转前 N 场(试跑用)")
    ap.add_argument("--force", action="store_true", help="已存在也重转")
    args = ap.parse_args()

    root = Path(args.sessions_dir).expanduser()
    if not root.is_dir():
        print(f"✗ 素材目录不存在:{root}", file=sys.stderr)
        return 2

    todo = _sessions(root)
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("没有待转场次(带姓名且有 camera.webm 的目录)")
        return 0

    print(f"待转 {len(todo)} 场 → {OUT_DIR}\n")
    ok = skipped = 0
    failed: list[tuple[str, str]] = []
    for i, d in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {d.name[:56]}", flush=True)
        try:
            r = transcribe_one(d, force=args.force)
        except Exception as e:  # 失败要说出来,不吞
            failed.append((d.name, f"{type(e).__name__}: {e}"))
            print(f"    ✗ 失败:{type(e).__name__}: {e}", file=sys.stderr, flush=True)
            continue
        if r["status"] == "skipped":
            skipped += 1
            print("    · 已转过,跳过(要重转加 --force)")
        else:
            ok += 1
            print(f"    ✓ {r['audio_s']}s 音频 / {r['n_chars']} 字 / ASR {r['asr_s']}s")

    print(f"\n{'─'*60}")
    print(f"成功 {ok}   跳过 {skipped}   失败 {len(failed)}")
    if failed:
        print("\n失败清单(需要处理,不要当成成功):")
        for name, why in failed:
            print(f"  ✗ {name[:56]}\n      {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
