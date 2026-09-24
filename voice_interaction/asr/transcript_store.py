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
    return recording_dir(session_id, root) / f"transcript_{session_id}.jsonl"


def append_utterance(session_id: str, utt, recorded_at: str | None = None,
                     root: str | Path | None = None) -> Path:
    """把一段识别结果【追加】到仓库外的 transcript jsonl,返回该文件路径。

    追加而非重写:会话中途崩溃也留得住已经识别出来的内容。
    每行一条 JSON:recorded_at / text / n_chars / n_segments / vad_split / segments。
    """
    p = _transcript_path(session_id, root)
    record = {
        "recorded_at": recorded_at or datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "text": utt.text,
        "n_chars": utt.n_chars,
        "n_segments": utt.n_segments,
        "vad_split": utt.vad_split,
        "segments": utt.segments,
    }
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
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
