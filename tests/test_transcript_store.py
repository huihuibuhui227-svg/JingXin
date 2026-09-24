import json
from datetime import datetime
from pathlib import Path

from voice_interaction.asr import session, transcript_store
from voice_interaction.asr.funasr_engine import AsrUtterance


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


def test_append_utterance_writes_jsonl_outside_repo(tmp_path):
    """原句只能落在仓库外的 jsonl 里(计划 Produces 承诺的接口)。"""
    sid = "20260924_153012_9f3c"
    utt = AsrUtterance(text="欢迎", n_chars=2, n_segments=1, vad_split=False,
                       segments=[{"index": 0, "text": "欢迎", "n_chars": 2,
                                  "timestamps_ms": [[0, 100]], "punc_array": [],
                                  "ts_origin": "segment_relative"}])
    p = transcript_store.append_utterance(sid, utt, recorded_at="2026-09-24T15:31:00", root=tmp_path)
    assert p == tmp_path / sid / f"transcript_{sid}.jsonl"
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["text"] == "欢迎" and rec["recorded_at"] == "2026-09-24T15:31:00"
    assert rec["n_chars"] == 2 and rec["vad_split"] is False
    assert rec["segments"][0]["timestamps_ms"] == [[0, 100]]


def test_append_utterance_appends_newline_per_call(tmp_path):
    """追加而非重写:中途崩溃也要留住已识别的内容。"""
    sid = "20260924_153012_9f3c"
    for text in ("第一句", "第二句"):
        p = transcript_store.append_utterance(sid, AsrUtterance(text=text, n_chars=3), root=tmp_path)
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").strip().splitlines()]
    assert [r["text"] for r in recs] == ["第一句", "第二句"]


def test_refresh_manifest_flags_present_and_missing(tmp_path):
    """会话结束按磁盘实况回填 present,并真的写回 session.json。"""
    sid = "20260924_153012_9f3c"
    transcript_store.ensure_manifest(sid, {"engine": "funasr", "models": {}}, root=tmp_path)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / f"face_au_log_{sid}.csv").write_text("x", encoding="utf-8")

    payload = transcript_store.refresh_manifest(sid, log_dir, root=tmp_path)
    assert payload["logs"]["face"]["present"] is True
    assert payload["logs"]["gesture"]["present"] is False
    assert payload["logs"]["voice"]["present"] is False

    on_disk = json.loads((tmp_path / sid / "session.json").read_text(encoding="utf-8"))
    assert on_disk["logs"]["face"]["present"] is True
