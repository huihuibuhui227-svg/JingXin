import json
import re
from datetime import datetime
from pathlib import Path

from voice_interaction.asr import funasr_engine, session, transcript_store
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


def _one_segment_utt(text: str) -> AsrUtterance:
    """一段回答只被 VAD 切成一个段(最常见形状);n_chars 用引擎的算法(去标点)。"""
    n = funasr_engine.count_cjk_chars(text)
    return AsrUtterance(text=text, n_chars=n, n_segments=1, vad_split=False,
                        segments=[{"index": 0, "text": text, "n_chars": n,
                                   "timestamps_ms": [[0, 100]], "punc_array": [1],
                                   "ts_origin": "segment_relative"}])


def _split_utt() -> AsrUtterance:
    """一次回答被 VAD 切成两段(段内时间戳各自从 0 重计)。"""
    return AsrUtterance(text="前半段后半段", n_chars=6, n_segments=2, vad_split=True,
                        segments=[
                            {"index": 0, "text": "前半段", "n_chars": 3,
                             "timestamps_ms": [[0, 100]], "punc_array": [1],
                             "ts_origin": "segment_relative"},
                            {"index": 1, "text": "后半段", "n_chars": 3,
                             "timestamps_ms": [[0, 50]], "punc_array": [2],
                             "ts_origin": "segment_relative"},
                        ])


def test_append_utterance_writes_single_transcript_json(tmp_path):
    """spec §6.4:一个会话一个 transcript.json(不是 jsonl),形状固定。"""
    sid = "20260924_153012_9f3c"
    utt = AsrUtterance(text="欢迎", n_chars=2, n_segments=1, vad_split=False,
                       segments=[{"index": 0, "text": "欢迎", "n_chars": 2,
                                  "timestamps_ms": [[210, 450], [450, 690]],
                                  "punc_array": [1, 1], "ts_origin": "segment_relative",
                                  "engine_future_field": "不该漏进契约"}])

    p = transcript_store.append_utterance(sid, utt, recorded_at="2026-09-24T15:30:12+08:00",
                                          root=tmp_path)
    assert p == tmp_path / sid / "transcript.json"
    assert sorted(x.name for x in (tmp_path / sid).iterdir()) == ["transcript.json"]

    payload = json.loads(p.read_text(encoding="utf-8"))
    assert set(payload) == {"session_id", "recorded_at", "asr", "merged", "segments"}
    assert payload["session_id"] == sid
    assert payload["recorded_at"] == "2026-09-24T15:30:12+08:00"

    assert set(payload["asr"]) == {"engine", "endpoint", "models",
                                   "asr_confidence", "asr_confidence_source"}
    assert payload["asr"]["engine"] == "funasr"
    assert payload["asr"]["endpoint"].startswith("ws://")
    assert isinstance(payload["asr"]["models"], dict)
    # 实测事实:本部署不返回置信度 —— 必须显式落盘,不能省略这两个键(§9.1)
    assert payload["asr"]["asr_confidence"] is None
    assert payload["asr"]["asr_confidence_source"] == "unavailable"

    assert payload["merged"] == {"text": "欢迎", "n_chars": 2, "n_segments": 1,
                                 "vad_split": False}
    assert payload["segments"] == [{"index": 0, "text": "欢迎", "n_chars": 2,
                                    "timestamps_ms": [[210, 450], [450, 690]],
                                    "punc_array": [1, 1],
                                    "ts_origin": "segment_relative"}]


def test_asr_block_follows_asr_config(tmp_path, monkeypatch):
    """engine/endpoint/models 只能来自 asr_config.json(不在代码里再存一份 host/port)。"""
    cfg = tmp_path / "asr_config.json"
    cfg.write_text(json.dumps({"funasr_host": "10.1.2.3", "funasr_port": 4321,
                               "timeout_s": 60.0,
                               "models": {"asr_offline": "…2pass-offline"}}),
                   encoding="utf-8")
    monkeypatch.setattr(funasr_engine, "_CONFIG_PATH", cfg)

    p = transcript_store.append_utterance("20260924_153012_9f3c",
                                         _one_segment_utt("你好"), root=tmp_path)
    asr = json.loads(p.read_text(encoding="utf-8"))["asr"]
    assert asr["endpoint"] == "ws://10.1.2.3:4321"
    assert asr["models"] == {"asr_offline": "…2pass-offline"}


def test_append_utterance_accumulates_across_calls(tmp_path):
    """spec §6.4:第二次调用【追加】段(index 连续)、重算 merged,recorded_at 不刷新。

    文本刻意带标点:merged.n_chars 是【各段 n_chars 之和】(去标点),不等于 len(merged.text)
    —— 用纯中文文本时两者相等,这条断言就区分不出"求和"与"数字符"。
    """
    sid = "20260924_153012_9f3c"
    transcript_store.append_utterance(sid, _one_segment_utt("第一句。"),
                                      recorded_at="2026-09-24T15:30:12+08:00", root=tmp_path)
    p = transcript_store.append_utterance(sid, _one_segment_utt("第二句!"),
                                          recorded_at="2026-09-24T15:59:59+08:00", root=tmp_path)

    payload = json.loads(p.read_text(encoding="utf-8"))
    assert [s["index"] for s in payload["segments"]] == [0, 1]
    assert [s["text"] for s in payload["segments"]] == ["第一句。", "第二句!"]   # 不覆盖既有段
    assert payload["merged"] == {"text": "第一句。第二句!", "n_chars": 6,
                                 "n_segments": 2, "vad_split": False}
    assert payload["merged"]["n_chars"] != len(payload["merged"]["text"])   # 求和 ≠ 数字符
    # 两次都只有一段 => 整场没有被 VAD 裂过(不是"总段数 > 1")
    assert payload["recorded_at"] == "2026-09-24T15:30:12+08:00"          # 第一次写入的时间


def test_merged_vad_split_is_true_once_any_call_was_split(tmp_path):
    """任一次识别被 VAD 裂过,整场 merged.vad_split 即为 true(累积 OR)。"""
    sid = "20260924_153012_9f3c"
    transcript_store.append_utterance(sid, _one_segment_utt("你好"), root=tmp_path)
    p = transcript_store.append_utterance(sid, _split_utt(), root=tmp_path)

    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["merged"] == {"text": "你好前半段后半段", "n_chars": 8,
                                 "n_segments": 3, "vad_split": True}
    assert [s["index"] for s in payload["segments"]] == [0, 1, 2]         # 跨调用仍连续
    # 第二段的段内相对时间戳原样保留,不假装绝对、也不被累加
    assert payload["segments"][2]["timestamps_ms"] == [[0, 50]]
    assert [s["ts_origin"] for s in payload["segments"]] == ["segment_relative"] * 3


def test_recorded_at_default_is_iso_with_offset(tmp_path):
    """spec §6.4 示例带时区偏移:默认值不能是裸 isoformat()。"""
    p = transcript_store.append_utterance("20260924_153012_9f3c",
                                         _one_segment_utt("你好"), root=tmp_path)
    recorded_at = json.loads(p.read_text(encoding="utf-8"))["recorded_at"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}", recorded_at), \
        recorded_at


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
