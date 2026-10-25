# report_frontend/transcript_chars.py
"""**读批量转写的字数** —— 给 V 族的 `speech_rate` 供 `n_chars`。

## 为什么需要这一层

正式语料 24/24 场的**语音日志是 0 行**(`interview_emotion_log_*.csv` 只有表头),
所以 `speech_rate` 原本拿不到 `n_chars`(它来自语音日志的文本列)。

但 `n_chars` 其实**在盘上**:`tools/jx_transcribe_recordings.py` 早就把整场录音
(`media/camera.webm` 的 opus 音轨)跑过 ASR,产出落在 `data/transcripts/<sid>.json`
(该目录在 `.gitignore` 里,是缓存不是源码)。实测 **13/13 场 A 档全都有**,
`n_chars` 320~3899,`n_segments` 到 44 段,还带**逐字时间戳**。

⟹ 这一层只做一件事:把那个 `n_chars` 读出来。**不重跑 ASR**(缓存已在),
也不解析 `segments`(当前没有消费方;逐字时间戳留给将来的 T 族)。

## 一条必须写进公式的保留

音频是**全场混音**(面试官 + 考生,没有说话人分离),所以

    speech_rate = 全场字数 ÷ 全场有声秒

量的是"这一场里说话有多密",**不是"考生语速"**。两者在有面试官插话时不等。
公式里必须写明这一点,否则读者会把它当成考生语速 —— 那是无声地改构念。

## 缺了怎么办

文件不在 / 读不出 / `n_chars` 缺失 ⟹ 返回 `None`。调用方据此报 `not_stored`,
**不许**用 0 顶替(0 字是一场没人说话,和"没转写过"是两回事)。

## 两处都要看(2026-10-25)

上面的缓存目录是 `tools/jx_transcribe_recordings.py` 的**批量**产物。而**活的那条路**
把转写写在**录制目录**里:`voice_interaction/asr/transcript_store.py` 每次
`/interview/answer_audio` 都累积进 `<录制目录>/transcript.json`。

2026-10-25 实测(WSL 本机 `~/shared/jingxin_recordings` 下 21 场):
**每场都有 `transcript.json`**(`merged.n_chars` 47~673),而 `data/transcripts/`
**是空的**。只读缓存的后果不是报错,是报告静默地说错话 —— T 族 3 条 + `speech_rate`
全报「没有转写」,还建议去跑一个根本不需要跑的工具。

⟹ 读取顺序:**缓存 > 录制目录**。缓存是显式转写产物,回退只补空缺,不覆盖已有结果
(反过来会让"我特意转的那一份"被录制目录里更早的版本悄悄盖掉)。
两种形状都认:活路径的 `merged.*`,以及批量工具的扁平 `text` / `n_chars`
(它被 `--sessions-dir` 指到录制根时,产物就跟着落在录制目录里)。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

#: 缓存目录。`JINGXIN_TRANSCRIPTS_DIR` 可覆盖(测试用)。
_REPO = Path(__file__).resolve().parent.parent


def transcript_dir() -> Path:
    env = os.environ.get("JINGXIN_TRANSCRIPTS_DIR")
    if env:
        return Path(env)
    return _REPO / "data" / "transcripts"


def path(session_id: str) -> Path:
    return transcript_dir() / f"{session_id}.json"


def _recording_transcript_path(session_id: str) -> Path | None:
    """活路径写的 `transcript.json`。**目录名不是死的**(`<标签>__<sid>`),
    所以必须走 `media_retention.resolve_recording_dir`,不能自己拼。

    延迟导入:`media_retention` 会牵进整条留存链,而本模块在报告侧被反复导入;
    读不到录制根(没配、无权限)时静默返回 `None` —— 那是"没有回退源",不是故障。
    """
    try:
        import media_retention
        return media_retention.resolve_recording_dir(session_id) / "transcript.json"
    except Exception:
        return None


def _read(path_: Path | None) -> dict | None:
    """读一份 JSON。读不出(不在 / 损坏 / 正在写)一律 `None`,不抛。"""
    if path_ is None or not path_.is_file():
        return None
    try:
        got = json.loads(path_.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return got if isinstance(got, dict) else None


def _norm(raw: dict | None) -> dict | None:
    """把两种形状归一成 `{text, n_chars, audio_seconds}`。

    * 活路径(`transcript_store._merge`):字段在 `merged` 下;
    * 批量工具(`jx_transcribe_recordings`):字段在顶层。
    """
    if not raw:
        return None
    merged = raw.get("merged")
    body = merged if isinstance(merged, dict) else raw
    return {
        "text": body.get("text"),
        "n_chars": body.get("n_chars"),
        "audio_seconds": raw.get("audio_seconds", body.get("audio_seconds")),
    }


def meta(session_id: str) -> dict | None:
    """整场转写记录(**已归一**);两处都没有就 `None`。

    顺序:缓存 > 录制目录。见模块 docstring 的「两处都要看」。
    """
    return _norm(_read(path(session_id))) or _norm(
        _read(_recording_transcript_path(session_id)))


def n_chars(session_id: str) -> int | None:
    """整场转写的字数。**读不出就是 `None`,不是 0。**"""
    m = meta(session_id)
    if not m:
        return None
    v = m.get("n_chars")
    if not isinstance(v, int) or v <= 0:
        # 0 或负数 ⟹ 视同"没有可用转写",不当作"一场没人说话"。
        return None
    return v


def text(session_id: str) -> str | None:
    """整场转写的正文。**读不出就是 `None`**(不是空串 —— 空串会被算成"0 字",
    与"没转写过"混为一谈)。"""
    m = meta(session_id)
    if not m:
        return None
    t = m.get("text")
    if not isinstance(t, str) or not t.strip():
        return None
    return t


def audio_seconds(session_id: str) -> float | None:
    """转写时记录的音频时长(秒)。仅用于交叉核对,不参与指标计算。"""
    m = meta(session_id)
    if not m:
        return None
    v = m.get("audio_seconds")
    return float(v) if isinstance(v, (int, float)) and v > 0 else None
