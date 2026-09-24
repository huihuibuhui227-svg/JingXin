"""转写与会话清单落盘 —— 一律在仓库外(spec D2)。

仓库内只允许出现数字;原句只进 ~/shared/jingxin_recordings/{session_id}/。
"""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from .session import NONE_SESSION

DEFAULT_ROOT = Path.home() / "shared" / "jingxin_recordings"          # D:\Shared\jingxin_recordings
TRANSCRIPT_FILENAME = "transcript.json"        # spec §6.4:一个会话一个文件,累积写
LOG_PREFIXES = {"face": "face_au_log", "gesture": "gesture_emotion_log",
                "voice": "interview_emotion_log"}

# 会话 id 直接当目录名用,而它是【客户端可控】的(query/表单参数),所以必须限量。
SESSION_ID_PAT = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def validate_session_id(session_id: str) -> str:
    """守卫:会话 id 只允许 `[A-Za-z0-9_-]{1,128}`(minted 形状与 `NONE` 都在内)。

    为什么必须在 store 这一层拦:端点把 `session_id` 当 query/表单参数收下(客户端可控),
    而这个值会被拼进 `recording_dir` 的路径再 `mkdir(parents=True)` ——
    `../../jingxin/x` 这类值能在录制根**之外**建目录并写入含原句的 transcript.json,
    把"原句绝不进仓库"(spec D2)这条不变量整个打穿。
    放在这里而不是端点里:任何调用方(含以后的 face/gesture)都自动受保护。
    公开出来是给端点用的:端点在跑识别之前先校验,好回 400 而不是等落盘时炸成 500。
    """
    if not isinstance(session_id, str) or not SESSION_ID_PAT.fullmatch(session_id):
        raise ValueError(
            f"非法 session_id: {session_id!r} —— 只允许字母/数字/下划线/连字符,1–128 位"
            f"(它是目录名,不接受路径分隔符与 '..')")
    return session_id


def normalize_session_id(raw: str | None) -> str:
    """把「客户端给的原始 id」规范化:**只有"参数没给"算没给**。

    与 `validate_session_id`(什么算非法)是**分工**,不是两层各判一次:
      * `None`(参数不存在 / 表单里没这个键)→ `NONE`,这是无会话客户端的正常路径;
      * 给了但内容是空的(`""` / `"  "`)→ **原样交出,由守卫判非法 → 400**。

    为什么空串不"顺手归 NONE":空串与"没给"在客户端那里是两件事 —— 后者是没接会话,
    前者是**参数拼错了**(例如 `?session_id=${sid}` 而 sid 为空)。静默归进 `NONE` 之后
    客户端拿到的是 200 和一份看着正常的响应,问题只在报告里以"数据对不上"的形式浮出来。
    三份副本(voice/face/gesture)由 tests/test_session_id_normalization.py 压着逐字相同。
    """
    return NONE_SESSION if raw is None else raw


# 每个 session_id 一把锁,保护 append_utterance 的读-改-写(见 _session_lock)。
# 放进程内:这个服务是单进程的 uvicorn;若以后多进程/多机部署,这里要换成文件锁。
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _session_lock(session_id: str) -> threading.Lock:
    """取(或建)本会话的写锁。

    `append_utterance` 是「读 json → 追加段 → 原子替换」:两个并发请求读到同一份旧
    segments 时,后写的会把先写的整个盖掉 —— 丢掉的那次回答在数字报告里看不出来
    (总段数只是少了一段,不报错),所以必须在写侧按会话串行。加锁粒度是会话:不同
    会话之间没有共享状态,不必互相等。

    `ensure_manifest` / `refresh_manifest` 不加锁:前者只在会话开始写一次且已存在即
    返回(幂等),后者只在会话结束写一次,都不在"同一会话并发写"的路径上。
    """
    with _LOCKS_GUARD:
        lock = _LOCKS.get(session_id)
        if lock is None:
            lock = _LOCKS[session_id] = threading.Lock()
        return lock


def _root(root: str | Path | None) -> Path:
    return Path(root) if root else Path(os.getenv("JINGXIN_RECORDINGS_DIR", DEFAULT_ROOT))


def recording_dir(session_id: str, root: str | Path | None = None) -> Path:
    d = _root(root) / validate_session_id(session_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest_path(session_id: str, root=None) -> Path:
    return recording_dir(session_id, root) / "session.json"


def _transcript_path(session_id: str, root=None) -> Path:
    return recording_dir(session_id, root) / TRANSCRIPT_FILENAME


def _asr_meta() -> dict[str, Any]:
    """provenance 块(spec §6.4)。

    engine/endpoint/models 从 `asr_config.json` 现取(经公开访问器 `load_config`)——
    不在这里再存一份 host/port。
    `asr_confidence` 恒为 null 且 `asr_confidence_source` 恒为 "unavailable":
    该部署的 raw 里没有置信度字段(实测,spec §3),必须显式落盘而非省略(spec §9.1)。
    """
    from .funasr_engine import load_config

    cfg = load_config()
    return {
        "engine": "funasr",
        "endpoint": f"ws://{cfg['funasr_host']}:{cfg['funasr_port']}",
        "models": dict(cfg.get("models") or {}),
        "asr_confidence": None,
        "asr_confidence_source": "unavailable",
    }


def _segment_records(utt, base_index: int) -> list[dict[str, Any]]:
    """按 spec §6.4 的段形状落盘(只留契约里的 6 个键,引擎以后加字段也不会漏进契约)。"""
    segs = list(getattr(utt, "segments", None) or [])
    return [{
        "index": base_index + i,
        "text": seg.get("text", "") or "",
        "n_chars": seg.get("n_chars", 0),
        "timestamps_ms": seg.get("timestamps_ms") or [],
        "punc_array": seg.get("punc_array") or [],
        "ts_origin": "segment_relative",     # spec §6.4:段内相对,不假装绝对
    } for i, seg in enumerate(segs)]


def _merge(segments: list[dict[str, Any]], vad_split: bool) -> dict[str, Any]:
    """merged = 全段重算:文本拼接、字数求和、段数、是否裂过(spec §6.4)。"""
    return {
        "text": "".join(s["text"] for s in segments),
        "n_chars": sum(s["n_chars"] for s in segments),
        "n_segments": len(segments),
        "vad_split": vad_split,
    }


def append_utterance(session_id: str, utt, recorded_at: str | None = None,
                     root: str | Path | None = None) -> Path:
    """把一次识别结果累积进仓库外的 `transcript.json`(spec §6.4 / D2)。

    一个会话一个文件:本次的段**追加**到既有 segments 之后(`index` 连续、不清空既有段),
    `merged` 按全部段重算;`recorded_at` 只由**第一次写入**决定,后续追加不刷新;
    `asr` provenance 块也保留第一次写入的那份。
    任一次识别被 VAD 裂过,整场 `merged.vad_split` 即为 true(累积 OR)。

    整段读-改-写在**本会话的锁**内完成(见 `_session_lock`):同一会话的并发调用
    (客户端重试 / 重复提交 / `/asr` 与 `/answer_audio` 撞车)必须串行,否则后写的
    会拿旧快照把先写的那次回答整个盖掉,而且丢得无声无息。
    """
    with _session_lock(session_id):
        p = _transcript_path(session_id, root)
        existing = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

        if existing is None:
            segments: list[dict[str, Any]] = []
            vad_split = bool(utt.vad_split)
            first_recorded_at = recorded_at or datetime.now().astimezone().isoformat(timespec="seconds")
            asr_meta = _asr_meta()
        else:
            segments = existing["segments"]
            vad_split = bool(existing["merged"]["vad_split"]) or bool(utt.vad_split)
            first_recorded_at = existing["recorded_at"]
            asr_meta = existing["asr"]

        segments.extend(_segment_records(utt, len(segments)))
        payload = {
            "session_id": session_id,
            "recorded_at": first_recorded_at,
            "asr": asr_meta,
            "merged": _merge(segments, vad_split),
            "segments": segments,
        }
        # 先写临时文件再原子替换:read-modify-write 中途崩掉不会毁掉整场既有段
        tmp = p.parent / (p.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, p)
        return p


def ensure_manifest(session_id: str, asr_meta: dict[str, Any], root=None) -> Path:
    """会话开始时写一次;已存在则不覆盖(保留 started_at 与既有 logs 状态)。"""
    p = _manifest_path(session_id, root)
    if p.exists():
        return p
    payload = {
        "session_id": session_id,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "asr": asr_meta,
        "logs": {mod: {"expected_file": f"{pre}_{session_id}.csv", "expected": True}
                 for mod, pre in LOG_PREFIXES.items()},
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def refresh_manifest(session_id: str, log_dir: str | Path, root=None) -> dict:
    """会话结束时按磁盘实况把 expected 换成 present/missing。"""
    p = _manifest_path(session_id, root)
    payload = json.loads(p.read_text(encoding="utf-8"))
    for mod, entry in payload["logs"].items():
        entry["present"] = (Path(log_dir) / entry["expected_file"]).exists()
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
