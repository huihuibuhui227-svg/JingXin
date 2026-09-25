# session_meta.py
"""阶段 A 元数据层:题目时刻台账 + `meta.json` 的模板与校验(spec §5.5 / §5.6)。

与 `media_retention.py` **分开**的理由:那个管的是**媒体字节**,这里管的是
**元数据**(题目时刻、人填的协变量)。两者落点也不同 —— 媒体在 `media/` 下面,
这里两样都在**会话根**。

它为什么必须在录制之前就位(spec §3.9 / §5.5):`response_latency` 的分子是
「首次开口墙钟 − `ask_end`」,而**录完就再也补不回来** —— 没有 `questions.jsonl`
的场次永久没有这个量。同理,该场的光照/增益/面试官评分都是"当时那一刻的状态",
不是稳定属性。

## `qid` 是什么(本题库没有题目 id)

实测:题库是**纯字符串列表**(`voice_interaction/pipeline/assessment_pipeline.py:101`、`:315`),
**题目没有 id**。而 spec §5.6 的请求体同时要 `qid` 与 `index`。所以:

- `qid` = **题目文本原文** —— 唯一、稳定、跨会话可归组;
- `index` = 该题在**本次会话**里的 0 基序号 —— 它标的是**顺序**,不是身份。

存文本**不违反** M1 的"原句不进仓库"边界:那条管的是**候选人说的话**;题目是这个
系统的固定输入,而且 `questions.jsonl` 落在 `~/shared`(**仓库外**)。
"""
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from media_retention import root as _recordings_root, validate_session_id

QUESTIONS_FILENAME = "questions.jsonl"
META_FILENAME = "meta.json"

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _session_lock(session_id: str) -> threading.Lock:
    """逐会话锁。语义与 `transcript_store._session_lock` 相同:读-改-写必须串行,
    否则并发重试会拿旧快照把新写入整个盖掉,而且丢得无声无息。
    """
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(session_id, threading.Lock())


def _reset_for_tests() -> None:
    with _LOCKS_GUARD:
        _LOCKS.clear()


def _session_dir(session_id: str, *, create: bool) -> Path:
    """会话根目录。`create=False` 时**不建** —— 读侧不该有副作用。

    为什么不用 `media_retention.recording_dir`:那个函数**总是建目录**,
    于是"读一场根本不存在/没报过题的会话"会在盘上留下一个空会话目录。
    """
    sid = validate_session_id(session_id)
    d = _recordings_root() / sid
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def questions_path(session_id: str) -> Path:
    return _session_dir(session_id, create=False) / QUESTIONS_FILENAME


def read_questions(session_id: str) -> list[dict]:
    """本会话已上报的题目窗口。文件不存在 → `[]`(不是错误)。"""
    p = questions_path(session_id)
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _validate_window(qid: Any, index: Any, ask_start: Any, ask_end: Any) -> None:
    if not isinstance(qid, str) or not qid.strip():
        raise ValueError(f"qid 必须是非空字符串(题目文本原文),收到 {qid!r}")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"index 必须是非负整数,收到 {index!r}")
    for name, v in (("ask_start", ask_start), ("ask_end", ask_end)):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{name} 必须是墙钟秒数(数字),收到 {v!r}")
        if not v > 0:
            raise ValueError(f"{name} 必须是正数,收到 {v!r}")
    if ask_end <= ask_start:
        raise ValueError(
            f"ask_end({ask_end}) 必须晚于 ask_start({ask_start})—— "
            f"零长或倒挂的提问窗口会让 response_latency 变成无意义的数")


def upsert_question(session_id: str, *, qid: str, index: int,
                    ask_start: float, ask_end: float,
                    source: str = "") -> dict:
    """记一道题的提问窗口。同一 `qid` **覆盖**前一次(spec §5.6 的"以后来的为准"、
    §7.6.2 的"落盘的只有后一个")。

    为什么是 upsert 而不是纯追加:下游(M3 的 `response_latency`)要的就是
    **一题一个时间窗**;留重复行等于把判重推给下游。
    读-改-写在**本会话的锁**内完成。
    """
    _validate_window(qid, index, ask_start, ask_end)
    sid = validate_session_id(session_id)
    rec = {"qid": qid, "index": index,
           "ask_start": float(ask_start), "ask_end": float(ask_end),
           "source": source, "reported_at_wall": time.time(),
           "session_id": sid}
    with _session_lock(sid):
        p = _session_dir(sid, create=True) / QUESTIONS_FILENAME
        rows = [r for r in read_questions(sid) if r.get("qid") != qid]
        rows.append(rec)
        rows.sort(key=lambda r: (r["index"], r["qid"]))
        # 写临时文件再 replace:进程被杀时不会留下半行(单文件、整份重写,
        # 所以不需要 media_retention 那条「追加必须单次 os.write」的讲究)
        tmp = p.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                       encoding="utf-8")
        os.replace(tmp, p)
    return rec
