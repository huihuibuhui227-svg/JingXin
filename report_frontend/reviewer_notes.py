# report_frontend/reviewer_notes.py
"""《衡才智析》导师与复核人员人工注记存储（严禁打分、强制红线复检）。

设计定位：
- 承接第三层 `Split Conformal` 保形门自动升级人工复核的闭环终点；
- 仅允许记录核验状态、重点关注标签与客观追问备注，严禁写入分值、划档或非客观推断词；
- 默认存储于 `data/reviewer_notes/<session_id>.json`（已 gitignore），支持环境变量覆盖。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import media_retention
import redline_guard

_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_NOTES_DIR = _REPO / "data" / "reviewer_notes"

VERIFICATION_STATUSES: dict[str, str] = {
    "pending_review": "待人工复核",
    "verified_consistent": "复核与证据一致",
    "needs_recollect": "建议补充采录",
    "signal_artifact_noted": "已注明采样干扰",
}


def notes_dir() -> Path:
    env = os.getenv("JINGXIN_REVIEW_NOTES_DIR")
    return Path(env) if env else _DEFAULT_NOTES_DIR


def _note_path(session_id: str) -> Path:
    sid = media_retention.validate_session_id(session_id)
    return notes_dir() / f"{sid}.json"


def read_reviewer_note(session_id: str) -> dict[str, Any]:
    """读取单场会话的人工复核注记；尚未填写时返回默认待复核空结构。"""
    sid = media_retention.validate_session_id(session_id)
    p = _note_path(sid)
    if not p.is_file():
        return {
            "session_id": sid,
            "exists": False,
            "verification_status": "pending_review",
            "verification_label": VERIFICATION_STATUSES["pending_review"],
            "focus_tags": [],
            "note_text": "",
            "updated_at": None,
        }
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        st = str(raw.get("verification_status") or "pending_review")
        if st not in VERIFICATION_STATUSES:
            st = "pending_review"
        return {
            "session_id": sid,
            "exists": True,
            "verification_status": st,
            "verification_label": VERIFICATION_STATUSES[st],
            "focus_tags": list(raw.get("focus_tags") or []),
            "note_text": str(raw.get("note_text") or ""),
            "updated_at": raw.get("updated_at"),
        }
    except Exception:
        return {
            "session_id": sid,
            "exists": False,
            "verification_status": "pending_review",
            "verification_label": VERIFICATION_STATUSES["pending_review"],
            "focus_tags": [],
            "note_text": "",
            "updated_at": None,
        }


def write_reviewer_note(
    session_id: str,
    *,
    verification_status: str = "verified_consistent",
    focus_tags: list[str] | None = None,
    note_text: str = "",
) -> dict[str, Any]:
    """写入结构化人工复核注记；若包含任何分值、等级或受限词条则直接拒收。"""
    sid = media_retention.validate_session_id(session_id)
    st = str(verification_status or "pending_review").strip()
    if st not in VERIFICATION_STATUSES:
        raise ValueError(f"不支持的复核状态 {st!r}（可选 {list(VERIFICATION_STATUSES)}）")

    text = str(note_text or "").strip()
    if len(text) > 1000:
        raise ValueError("复核注记正文不得超过 1000 字。")

    tags = [str(t).strip() for t in (focus_tags or []) if str(t).strip()][:10]
    combined = " ".join([text, *tags])
    hits = redline_guard.find_assistant_violations(combined)
    if hits:
        raise ValueError(f"复核注记含禁止打分或红线受限表述：{', '.join(hits)}")

    payload = {
        "session_id": sid,
        "exists": True,
        "verification_status": st,
        "verification_label": VERIFICATION_STATUSES[st],
        "focus_tags": tags,
        "note_text": text,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    target = _note_path(sid)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    return payload
