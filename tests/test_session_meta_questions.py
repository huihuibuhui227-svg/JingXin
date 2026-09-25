# tests/test_session_meta_questions.py
"""题目时刻台账(spec §5.6)。它是 `response_latency` 的唯一来源。"""
import importlib

import pytest

session_meta = importlib.import_module("session_meta")

SID = "20260925_203826_2449"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    session_meta._reset_for_tests()
    return tmp_path


def test_reported_window_lands_on_disk(_isolated):
    rec = session_meta.upsert_question(SID, qid="请简单介绍一下你自己", index=0,
                                       ask_start=1758824000.0, ask_end=1758824006.5,
                                       source="/session/question")
    p = _isolated / SID / "questions.jsonl"
    assert p.exists()
    rows = session_meta.read_questions(SID)
    assert len(rows) == 1
    assert rows[0]["qid"] == "请简单介绍一下你自己"
    assert rows[0]["index"] == 0
    assert rows[0]["ask_end"] == 1758824006.5
    assert rec["session_id"] == SID


def test_same_qid_reported_twice_keeps_only_the_later(_isolated):
    """spec §7.6.2:同一 (sid, qid) 报两次 → **落盘的只有后一个**。

    红法:把 upsert 改成纯追加(`rows.append(rec)`)→ 本测试红在 len(rows)。
    """
    session_meta.upsert_question(SID, qid="Q", index=0, ask_start=1.0, ask_end=2.0)
    session_meta.upsert_question(SID, qid="Q", index=0, ask_start=1.0, ask_end=9.0)
    rows = session_meta.read_questions(SID)
    assert len(rows) == 1, f"同一题留了 {len(rows)} 行"
    assert rows[0]["ask_end"] == 9.0


def test_two_different_questions_both_stay(_isolated):
    session_meta.upsert_question(SID, qid="Q1", index=0, ask_start=1.0, ask_end=2.0)
    session_meta.upsert_question(SID, qid="Q2", index=1, ask_start=3.0, ask_end=4.0)
    assert [r["qid"] for r in session_meta.read_questions(SID)] == ["Q1", "Q2"]


@pytest.mark.parametrize("kw,why", [
    (dict(qid="  ", index=0, ask_start=1.0, ask_end=2.0), "空 qid"),
    (dict(qid="Q", index=-1, ask_start=1.0, ask_end=2.0), "负 index"),
    (dict(qid="Q", index=0, ask_start=0.0, ask_end=2.0), "ask_start 非正"),
    (dict(qid="Q", index=0, ask_start=1.0, ask_end=0.0), "ask_end 非正"),
    (dict(qid="Q", index=0, ask_start=5.0, ask_end=2.0), "ask_end 早于 ask_start"),
    (dict(qid="Q", index=0, ask_start=5.0, ask_end=5.0), "零长窗口"),
    (dict(qid="Q", index=0, ask_start="1.0", ask_end=2.0), "字符串当时间"),
])
def test_bad_windows_are_rejected_loudly(_isolated, kw, why):
    """倒挂 / 零长 / 非数字的窗口必须**拒绝**。

    零长的提问窗口会让 `response_latency = 首次开口 − ask_end` 变成无意义的数,
    而且它看起来像个正常值 —— 比显然的错更危险。
    """
    with pytest.raises(ValueError):
        session_meta.upsert_question(SID, **kw)
    assert not (_isolated / SID / "questions.jsonl").exists(), f"{why}:拒绝得不够干净"


def test_reading_a_session_with_no_questions_is_empty_not_an_error(_isolated):
    """读一场没报过题的会话:返回空列表,而且**不建目录**(读侧不该有副作用)。"""
    assert session_meta.read_questions("20260101_000000_aaaa") == []
    assert not (_isolated / "20260101_000000_aaaa").exists()


def test_path_traversal_is_rejected(_isolated):
    with pytest.raises(ValueError):
        session_meta.upsert_question("../escape", qid="Q", index=0,
                                     ask_start=1.0, ask_end=2.0)
