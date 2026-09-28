# tests/test_session_label_endpoint.py
"""`POST /session/{sid}/label`:前端弹窗填的本场标注的落点。

端点存在的理由:标注要**存在服务器上**(换台电脑打开也查得到),而前端只能通过
HTTP 落盘 —— 这不是纯前端 localStorage 能顶的事。
"""
import asyncio
import importlib

import pytest
from fastapi import HTTPException

session_meta = importlib.import_module("session_meta")
voice_app = importlib.import_module("voice_interaction.api.app")

SID = "20260927_203359_19c3"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    session_meta._reset_for_tests()
    return tmp_path


def _post(**kw):
    """调端点并把协程跑完(与 `test_session_question_endpoint` 同一理由:端点是
    `async def`,不 `asyncio.run` 就只拿到协程对象 —— 函数体根本不执行,
    于是"该抛的没抛",**红得不对 = 什么也没验到**)。"""
    body = {"serial": "01", "name": "张三", "student_id": "2021001",
            "department": "计算机与人工智能学院"}
    body.update(kw)
    return asyncio.run(voice_app.submit_session_label(
        session_id=SID, body=voice_app.SessionLabel(**body)))


def test_label_is_written_and_readable_back(_isolated):
    r = _post()
    assert r["status"] == "success"
    assert r["session_id"] == SID
    # 端点把**拼好的**标签回给前端:界面显示的是服务端算出来的那一个,
    # 不是前端自己拼的 —— 拼法只允许有一处定义。
    assert r["label"] == "20260927_203359-01-张三-2021001-计算机与人工智能学院"
    assert session_meta.read_label(SID)["label"] == r["label"]


def test_illegal_session_id_is_400_not_500():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as ei:
        asyncio.run(voice_app.submit_session_label(
            session_id="../escape",
            body=voice_app.SessionLabel(serial="01", name="张三",
                                        student_id="", department="")))
    assert ei.value.status_code == 400


def test_all_blank_label_is_400_not_500(_isolated):
    """四个字段全空 ⟹ 没有可标的信息,是**请求本身**的毛病(400)。"""
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as ei:
        asyncio.run(voice_app.submit_session_label(
            session_id=SID,
            body=voice_app.SessionLabel(serial="", name="", student_id="",
                                        department="")))
    assert ei.value.status_code == 400
    assert not (_isolated / SID / "label.json").exists()


def test_consent_round_trips_through_the_endpoint(_isolated):
    """★ 红法:把端点里 `consent=body.consent or None` 那个参数去掉。

    征询结果**必须能通过这个端点落盘** —— 它是前端唯一的上报口。少了这一句,
    前端选「只同意声音」也照样存不进去,而界面上什么都不会报错。
    """
    r = _post(consent="audio_only")
    assert r["status"] == "success", r
    assert session_meta.read_label(SID)["consent"] == "audio_only"


def test_an_unknown_consent_is_a_400_not_a_silent_default(_isolated):
    """★ 红法:把 `session_meta._consent` 的拒绝改成回落 `full`。

    回落 `full` 是**最坏的方向** —— 把"没同意"记成"全同意"。所以这里要 400。
    """
    with pytest.raises(HTTPException) as e:
        _post(consent="yes")
    assert e.value.status_code == 400
