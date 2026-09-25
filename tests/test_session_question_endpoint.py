# tests/test_session_question_endpoint.py
"""`POST /session/{sid}/question`:前端推题时刻的落点(spec §5.6)。"""
import asyncio
import importlib

import pytest

session_meta = importlib.import_module("session_meta")
voice_app = importlib.import_module("voice_interaction.api.app")

SID = "20260925_203826_2449"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    session_meta._reset_for_tests()
    return tmp_path


def _post(**kw):
    """调端点并把协程跑完。

    ⚠️ 端点是 `async def` —— 不 `asyncio.run` 的话这里返回的是**协程对象**,
    函数体根本不会执行:于是 `r["status"]` 抛 TypeError,而
    `pytest.raises(HTTPException)` 也永远等不到那个 raise。
    **红得不对 = 那条测试什么也没验到。**
    """
    body = {"qid": "请简单介绍一下你自己", "index": 0,
            "ask_start": 1758824000.0, "ask_end": 1758824006.5}
    body.update(kw)
    return asyncio.run(voice_app.submit_session_question(
        session_id=SID, body=voice_app.QuestionWindow(**body)))


def test_reported_window_is_written(_isolated):
    r = _post()
    assert r["status"] == "success" and r["qid"] == "请简单介绍一下你自己"
    rows = session_meta.read_questions(SID)
    assert len(rows) == 1 and rows[0]["ask_end"] == 1758824006.5


def test_illegal_session_id_is_400():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as ei:
        asyncio.run(voice_app.submit_session_question(
            session_id="../escape",
            body=voice_app.QuestionWindow(qid="Q", index=0,
                                          ask_start=1.0, ask_end=2.0)))
    assert ei.value.status_code == 400


def test_inverted_window_is_400_not_500(_isolated):
    """倒挂窗口是**请求本身**不合法 → 400。

    红法:去掉端点里那个 `except ValueError` → `ValueError` 未捕获上抛
    (经 ASGI 即 500),客户端拿到的是"服务器出错",而真相是它自己发错了
    —— 排查方向全偏。
    """
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as ei:
        _post(ask_start=5.0, ask_end=2.0)
    assert ei.value.status_code == 400
    assert not (_isolated / SID / "questions.jsonl").exists()
