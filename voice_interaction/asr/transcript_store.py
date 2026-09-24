"""转写与会话清单落盘 —— 一律在仓库外(spec D2)。

仓库内只允许出现数字;原句只进 ~/shared/jingxin_recordings/{session_id}/。
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path.home() / "shared" / "jingxin_recordings"          # D:\Shared\jingxin_recordings
TRANSCRIPT_FILENAME = "transcript.json"        # spec §6.4:一个会话一个文件,累积写
LOG_PREFIXES = {"face": "face_au_log", "gesture": "gesture_emotion_log",
                "voice": "interview_emotion_log"}


def _root(root: str | Path | None) -> Path:
    return Path(root) if root else Path(os.getenv("JINGXIN_RECORDINGS_DIR", DEFAULT_ROOT))


def recording_dir(session_id: str, root: str | Path | None = None) -> Path:
    d = _root(root) / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest_path(session_id: str, root=None) -> Path:
    return recording_dir(session_id, root) / "session.json"


def _transcript_path(session_id: str, root=None) -> Path:
    return recording_dir(session_id, root) / TRANSCRIPT_FILENAME


def _asr_meta() -> dict[str, Any]:
    """provenance 块(spec §6.4)。

    engine/endpoint/models 从 `asr_config.json` 现取 —— 不在这里再存一份 host/port。
    `asr_confidence` 恒为 null 且 `asr_confidence_source` 恒为 "unavailable":
    该部署的 raw 里没有置信度字段(实测,spec §3),必须显式落盘而非省略(spec §9.1)。
    """
    from .funasr_engine import _config

    cfg = _config()
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
    """
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
