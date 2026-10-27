"""题库：**多套** + 当前生效 + 内置默认 + 参考答案（可选）。

2026-10-26 使用者原话：「题库为什么是按题目修改？这样不就只能用一份题库了吗」。
判据因此盯的是：**新建一套不许覆盖已有那套**，切换当前只改"用哪套"，删掉当前那套
要回退到内置默认而不是空题库。
"""
import json

import pytest

import question_bank as qb


@pytest.fixture
def bank(tmp_path, monkeypatch):
    """把题库文件指到 tmp —— 不碰真实 `data/question_bank.json`。"""
    monkeypatch.setattr(qb, "BANK_PATH", tmp_path / "question_bank.json")
    return qb


def test_default_bank_before_any_custom(bank):
    d = bank.load("interview")
    assert d["source"] == "default"
    assert d["active_id"] is None
    assert d["banks"] == []
    assert len(d["active"]["questions"]) == 8
    assert len(d["defaults"]) == 8


def test_create_does_not_overwrite_the_previous_bank(bank):
    a = bank.create_bank("interview", "秋招", ["A1", "A2"])
    assert a["source"] == "custom" and len(a["banks"]) == 1
    first_id = a["banks"][0]["id"]
    assert a["active_id"] == first_id          # 第一套自动生效

    b = bank.create_bank("interview", "春招", [
        {"question": "B1", "reference_answer": "参考答案 B1"}])
    assert len(b["banks"]) == 2
    assert b["active_id"] == first_id          # **不**自动把第一套换掉
    names = {x["name"]: x for x in b["banks"]}
    assert names["秋招"]["questions"][0]["question"] == "A1"   # 旧套原样还在
    assert names["春招"]["questions"][0]["reference_answer"] == "参考答案 B1"


def test_update_only_touches_that_bank(bank):
    a = bank.create_bank("interview", "套一", ["一"])
    b = bank.create_bank("interview", "套二", ["二"])
    id1 = a["banks"][0]["id"]
    id2 = b["banks"][1]["id"]

    bank.update_bank("interview", id1, name="套一（改）", items=["一改"])
    d = bank.load("interview")
    names = {x["id"]: x["name"] for x in d["banks"]}
    assert names[id1] == "套一（改）"
    assert names[id2] == "套二"
    one = next(x for x in d["banks"] if x["id"] == id1)
    two = next(x for x in d["banks"] if x["id"] == id2)
    assert one["questions"][0]["question"] == "一改"
    assert two["questions"][0]["question"] == "二"


def test_activate_switches_and_null_goes_back_to_default(bank):
    a = bank.create_bank("research", "A", ["A题"])
    b = bank.create_bank("research", "B", ["B题"])
    id_b = b["banks"][1]["id"]

    assert bank.get_question_texts("research") == ["A题"]     # active 还是 A
    bank.activate("research", id_b)
    assert bank.get_question_texts("research") == ["B题"]

    bank.activate("research", None)
    d = bank.load("research")
    assert d["source"] == "default" and d["active_id"] is None
    assert len(bank.get_question_texts("research")) == 8


def test_delete_active_falls_back_to_default_not_empty(bank):
    a = bank.create_bank("interview", "唯一", ["题"])
    bid = a["banks"][0]["id"]
    d = bank.delete_bank("interview", bid)
    assert d["banks"] == []
    assert d["active_id"] is None
    assert d["source"] == "default"
    assert len(bank.get_question_texts("interview")) == 8


def test_bad_inputs_are_rejected(bank):
    with pytest.raises(ValueError):
        bank.create_bank("interview", "空", [])
    with pytest.raises(ValueError):
        bank.create_bank("nope", "x", ["题"])
    with pytest.raises(ValueError):
        bank.create_bank("interview", "空白", [{"question": "   "}])
    with pytest.raises(ValueError):
        bank.create_bank("interview", "超长", ["超" * (qb.MAX_QUESTION_CHARS + 1)])
    with pytest.raises(ValueError):
        bank.activate("interview", "不存在的id")
    with pytest.raises(ValueError):
        bank.update_bank("interview", "不存在的id", items=["题"])


def test_file_shape_is_stable(bank):
    bank.create_bank("interview", "秋招", ["题一"])
    data = json.loads(bank.BANK_PATH.read_text(encoding="utf-8"))
    mod = data["interview"]
    assert set(mod) == {"active", "banks"}
    assert mod["banks"][0]["name"] == "秋招"
    assert mod["banks"][0]["questions"] == [
        {"question": "题一", "reference_answer": ""}]
    assert mod["banks"][0]["updated_at"]


def test_legacy_single_bank_file_is_migrated(bank):
    """旧格式（模块下直接挂 questions）读进来要变成一套题库，不是消失。"""
    bank.BANK_PATH.write_text(json.dumps({
        "interview": {"updated_at": "2026-10-26 10:00:00",
                      "questions": [{"question": "旧题", "reference_answer": ""}]}
    }, ensure_ascii=False), encoding="utf-8")
    d = bank.load("interview")
    assert d["source"] == "custom"
    assert len(d["banks"]) == 1
    assert d["banks"][0]["name"] == "导入的题库"
    assert d["banks"][0]["questions"][0]["question"] == "旧题"
    assert bank.get_question_texts("interview") == ["旧题"]


def test_corrupt_file_is_loud(bank):
    bank.BANK_PATH.write_text("{ 这不是 JSON", encoding="utf-8")
    with pytest.raises(ValueError, match="题库文件损坏"):
        bank.load("interview")
