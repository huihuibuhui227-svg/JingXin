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


def _body(session_id: str) -> dict | None:
    """原始转写的**内层**字典(活路径在 `merged` 下,批量工具在顶层)。

    与 `meta()` 的区别:这里**不裁剪**键 —— `segments` 只在原始形状里,
    `meta()` 归一掉的正是它。所以逐段读取必须走这一条,而不是 `meta()`。
    """
    raw = _read(path(session_id)) or _read(_recording_transcript_path(session_id))
    if not raw:
        return None
    merged = raw.get("merged")
    return merged if isinstance(merged, dict) else raw


def segments(session_id: str) -> list[dict] | None:
    """逐段转写(`[{index, text, n_chars, ...}]`);**读不出就是 `None`**。

    ## 为什么加这个访问器(2026-11-21)

    `answer_detail_chars_p50` 要的是**每段回答的字数**,而 `meta()` / `n_chars()`
    只给整场合并后的一个数 —— 逐段信息会被归一掉。逐段结构在原始 JSON 的
    `segments` 里(每段带 `text` 与 `n_chars`),所以必须有一个只读它的入口。

    ## 为什么放在本模块,而不是让调用方自己读 JSON

    "两处都要看"(缓存 > 录制目录、目录名可能是 `<标签>__<短id>`)这条读法
    只该有**一处实现**。调用方各读一遍,迟早有一处漏掉回退、或者顺序反过来
    (反过来会让显式转写的那一份被更早的版本盖掉)。

    ## 两件必须说清的事

    * **段 ≠ 题**:正式场没有题目台账,这些段是 VAD 切出来的**连续语音段**
      (`evidence_thresholds.json` 里 `answer` 单位的既有定义:录制语料上的
      `answer` 就是 VAD 段)。所以逐段中位数是"每段话讲了多长",不是"每题答了多长";
    * **`n_chars` 那一列不要用**:它是 `count_cjk_chars` 算的,只数汉字,
      英文/数字段恒为 0。要用字数请对 `text` 重新数(`count_non_space_chars`)。
    """
    # ⚠️ 2026-11-22 修:`_body()` 在**活路径**形状下返回的是 `raw["merged"]`,
    #   而 `merged` 里**没有** `segments`(逐段结构在**顶层**)——于是本函数对
    #   `recordings/<sid>/transcript.json` 一律返回 `None`,`answer_detail_chars_p50`
    #   白白报 `not_stored`(实测 `95a8` 明明有 10 段)。⟹ 这里**先看顶层**,再看内层。
    raw = _read(path(session_id)) or _read(_recording_transcript_path(session_id))
    if not raw:
        return None
    segs = raw.get("segments")
    if not isinstance(segs, list):
        merged = raw.get("merged")
        body = merged if isinstance(merged, dict) else raw
        segs = body.get("segments")
    if not isinstance(segs, list):
        return None
    return [s for s in segs if isinstance(s, dict)]


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
