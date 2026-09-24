from datetime import datetime
from pathlib import Path

from voice_interaction.asr import session, transcript_store


def test_new_session_id_shape():
    sid = session.new_session_id(now=datetime(2026, 9, 24, 15, 30, 12))
    assert sid.startswith("20260924_153012_")
    assert len(sid) == len("20260924_153012_") + 4
    int(sid[-4:], 16)


def test_new_session_id_is_unique():
    ids = {session.new_session_id() for _ in range(50)}
    assert len(ids) == 50


def test_recording_dir_is_outside_repo(tmp_path):
    d = transcript_store.recording_dir("20260924_153012_9f3c", root=tmp_path)
    assert d == tmp_path / "20260924_153012_9f3c"
    assert d.exists()


def test_manifest_records_expected_log_paths(tmp_path):
    p = transcript_store.ensure_manifest("20260924_153012_9f3c",
                                         {"engine": "funasr", "models": {}}, root=tmp_path)
    payload = __import__("json").loads(p.read_text(encoding="utf-8"))
    assert payload["session_id"] == "20260924_153012_9f3c"
    assert set(payload["logs"]) == {"face", "gesture", "voice"}
    assert all(v["expected"] is True for v in payload["logs"].values())
