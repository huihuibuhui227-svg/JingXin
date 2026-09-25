# tests/test_session_closeout.py
"""会话收尾对账(spec §6 的两条出口 + §7.6.3/4)。

它是"当场逐条打勾"那份清单的机器版 —— 靠流程不靠代码(spec §8.6),
但**至少要能响**。
"""
import importlib
import json
import time
from pathlib import Path

import pytest

import media_retention
session_meta = importlib.import_module("session_meta")

SID = "20260925_203826_2449"
T0 = time.time()


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.delenv("JINGXIN_RETAIN_MEDIA", raising=False)
    media_retention._reset_for_tests()
    session_meta._reset_for_tests()
    return tmp_path


def _fill_meta(sid=SID):
    """把模板**全部填满**(包括 `questions`)。

    ⚠️ `questions` 那一项是 Task 5 的裁定补进 `META_REQUIRED` 的(录制需求 §3.2
    明列「题目 ID + 难度」必填)。不填它,`test_everything_present_reports_nothing_missing`
    会因为「questions 空列表 = 缺」而红 —— 那条断言本身是对的。
    """
    session_meta.write_template(sid)
    p = session_meta.meta_path(sid)
    m = json.loads(p.read_text(encoding="utf-8"))
    m["candidate"].update(sex="M", age=30, native_language="zh", dialect_region="吴语")
    m["capture"].update(device="Logitech C920", resolution="1280x720",
                        camera_distance_cm=60, lighting="室内顶灯", mic_gain_db=12)
    m["interviewer_ratings"].update(logical_thinking=4, communication=3, confidence=5)
    m["consent"]["archived"] = True
    m["questions"] = [{"qid": "Q0", "difficulty": "中"}, {"qid": "Q1", "difficulty": "难"}]
    p.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")


def test_everything_present_reports_nothing_missing(_isolated):
    _fill_meta()
    media_retention.retain_frame(SID, "face", b"\xff\xd8\xff\xe0jpeg", source="/analyze")
    media_retention.retain_uploaded_video(SID, b"\x1a\x45\xdf\xa3webm")
    session_meta.upsert_question(SID, qid="Q0", index=0, ask_start=T0, ask_end=T0 + 1)
    session_meta.upsert_question(SID, qid="Q1", index=1, ask_start=T0 + 10, ask_end=T0 + 11)

    got = session_meta.check_session(SID, expected_questions=2)
    assert got["media"]["face"] == 1
    assert got["video"] is True
    assert got["degraded"] == []
    assert got["missing_meta"] == []
    assert got["missing_questions"] == []


def test_missing_interviewer_rating_is_reported_but_does_not_raise(_isolated):
    """spec §7.6.3:删掉 interviewer_ratings 的**某一项** → 收尾报出来。
    注意是**单项**:整块删是"没填",删一项是"填漏了",两种都要报得上。"""
    _fill_meta()
    p = session_meta.meta_path(SID)
    m = json.loads(p.read_text(encoding="utf-8"))
    del m["interviewer_ratings"]["communication"]
    p.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")

    got = session_meta.check_session(SID)          # 不抛
    assert "interviewer_ratings.communication" in got["missing_meta"]


def test_unreported_question_is_named_by_index(_isolated):
    """spec §7.6.4:两题只报一题 → 点名缺的那题(按 index)。"""
    _fill_meta()
    session_meta.upsert_question(SID, qid="Q0", index=0, ask_start=T0, ask_end=T0 + 1)
    got = session_meta.check_session(SID, expected_questions=2)
    assert got["missing_questions"] == [1], got["missing_questions"]


def test_degraded_reasons_surface_in_the_report(_isolated):
    """留存中途失败 → 收尾必须看得到(spec §6:绝不允许静默)。

    注入方式同 Task 1:先成功写一件过预检,再用 `MonkeyPatch.context()`
    (**不是** `monkeypatch.undo()` —— 那会把 `JINGXIN_RECORDINGS_DIR` 一起撤掉,
    于是真的写进 `~/shared`)。
    """
    _fill_meta()
    media_retention.retain_frame(SID, "face", b"\xff\xd8\xff\xe0ok", source="/analyze")
    calls = {"n": 0}
    real = Path.write_bytes

    def flaky(self, data):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(28, "No space left on device")
        return real(self, data)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Path, "write_bytes", flaky)
        media_retention.retain_frame(SID, "face", b"\xff\xd8\xff\xe0boo", source="/analyze")

    got = session_meta.check_session(SID)
    assert any("No space left" in d for d in got["degraded"]), got["degraded"]
    # 降级行**不许**被当成材料数进去(复核实测:把 kind 过滤改成 `if True:` 时
    # 没有任何测试变红 —— 于是一场"每次写都失败"的会话会报出并不存在的素材)
    assert got["media"] == {"face": 1}, got["media"]   # 只有成功的那一件


def test_response_latency_is_computable_from_the_stored_ask_end(_isolated):
    """spec §7.6.1:拿 `questions.jsonl` + 一份带逐字时间戳的 transcript,
    `response_latency` **算得出**,且**用的是 `ask_end` 自己的值**(改成错值 → 结果跟着变)。

    ⚠️ 这里**只做算术演示,不落地成生产函数** —— spec §5.6 明确写"本条不在 M2.6 实现
    (M2.6 只负责把 `ask_end` 留下来)"。它存在的理由是**证明留下的那个数接得上**:
    `response_latency = 首次开口墙钟 − ask_end`,而 ASR 的逐字时间戳是**段内相对值**
    (从 0 起),所以"首次开口墙钟"必须由**段起始墙钟 + 段内偏移**合成。
    不把这个示范钉住,M3 真去算时才会发现两个基不是一回事。
    """
    seg_wall, first_char_offset = T0 + 11.5, 0.75         # 段起始墙钟 / 段内偏移
    first_speech_wall = seg_wall + first_char_offset

    session_meta.upsert_question(SID, qid="Q0", index=0,
                                 ask_start=T0, ask_end=T0 + 6.0)
    ask_end = session_meta.read_questions(SID)[0]["ask_end"]
    assert round(first_speech_wall - ask_end, 3) == 6.25

    session_meta.upsert_question(SID, qid="Q0", index=0,        # 换成错值
                                 ask_start=T0, ask_end=T0 + 9.5)
    ask_end2 = session_meta.read_questions(SID)[0]["ask_end"]
    assert round(first_speech_wall - ask_end2, 3) == 2.75, "结果没跟着 ask_end 变 ⟹ 它没读那个字段"


def test_missing_meta_file_reports_all_required_and_does_not_raise(_isolated):
    got = session_meta.check_session(SID)
    assert set(got["missing_meta"]) == set(session_meta.META_REQUIRED)
    assert got["media"] == {}
    assert got["video"] is False
