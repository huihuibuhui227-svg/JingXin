# tests/test_session_meta_label.py
"""`label.json`:这一场「是谁的、第几场」的人可读标注。

为什么它**不叫** `session_id`:报告侧的文件名契约是
`^(face|gesture|interview|research)_(.+?)_log_(\\d{8})_(\\d{6})(?:_([0-9a-f]{4}))?\\.csv$`
(`report_frontend/data_loader.py`),把姓名/学号塞进 id 会让**每份日志**都匹配不上
⟹ 报告一份都加载不到、覆盖恒 0/20;而写侧 `validate_session_id` 只收
`[A-Za-z0-9_-]`,中文姓名直接 400。所以标注**另存一份**,id 一个字节不动。
"""
import importlib

import pytest

session_meta = importlib.import_module("session_meta")

SID = "20260927_203359_19c3"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    session_meta._reset_for_tests()
    return tmp_path


def _write(**kw):
    body = {"serial": "01", "name": "张三", "student_id": "2021001",
            "department": "计算机与人工智能学院"}
    body.update(kw)
    return session_meta.upsert_label(SID, **body)


def test_label_lands_on_disk_next_to_the_session(_isolated):
    rec = _write()
    p = _isolated / SID / "label.json"
    assert p.exists(), "label.json 该落在会话根(与 questions.jsonl / meta.json 同级)"
    got = session_meta.read_label(SID)
    assert got["name"] == "张三"
    assert got["student_id"] == "2021001"
    assert got["department"] == "计算机与人工智能学院"
    assert got["session_id"] == SID
    assert rec["label"] == got["label"]


def test_timestamp_comes_from_the_session_id_not_the_wall_clock():
    """标签里的「时间」取自 `session_id` 自己那一段。

    红法:把 `compose_label` 改成用 `time.strftime` 现取当前时刻 → 本测试红。
    为什么必须来自 id:报告端、日志文件名、录制目录名全都用 id 那一串;标签是人拿来
    **对号**的,如果它自己另取一个时刻,就会出现"标签说 20:35、文件名说 20:33"这种
    对不上的场面 —— 而标签存在的唯一理由就是能对上。
    """
    assert session_meta.compose_label(
        SID, serial="01", name="张三", student_id="2021001", department="计算机与人工智能学院"
    ) == "20260927_203359-01-张三-2021001-计算机与人工智能学院"


def test_empty_parts_are_dropped_not_left_as_dangling_dashes():
    """留空的字段**不**留下一个孤零零的分隔符 —— `a--b` 看着像"那里有个空字段",
    而实际是"没填"。空的部分整段不出现。"""
    assert session_meta.compose_label(
        SID, serial="01", name="张三", student_id="", department=""
    ) == "20260927_203359-01-张三"


def test_all_blank_is_rejected(_isolated):
    """四个字段全空 ⟹ 标签就是那串 id,**等于没标** —— 而这一场的标注正是为了
    事后能把它与别的场次分开。所以拒掉,而不是存一份没有信息量的标注。

    红法:去掉 `upsert_label` 里那句 `if not any(...)` → 本测试红在没抛 ValueError。
    """
    with pytest.raises(ValueError):
        _write(serial="  ", name="", student_id="", department="")


def test_second_write_overwrites_the_first(_isolated):
    """同一会话再报一次 = **覆盖**(与 `upsert_question` 同一语义)。

    为什么:标注是"这一场是谁的",一个会话只有一个答案;打错一个字必须能改。
    红法:改成追加(把两条都写进文件)→ 本测试红在 name 读到第一条。
    """
    _write(name="张三")
    _write(name="李四")
    assert session_meta.read_label(SID)["name"] == "李四"


def test_field_that_is_not_a_string_is_rejected():
    """类型错是**请求本身**的毛病 → 抛 ValueError(端点转 400),不是 500。"""
    with pytest.raises(ValueError):
        _write(serial=1)


def test_overlong_field_is_rejected():
    """长度闸:这是个会被写进 JSON、并在界面上原样渲染的值,不设上限等于让一个
    粘贴事故把界面撑爆。**上限是 120 字符**(远超真名/学号/院系的长度)。"""
    with pytest.raises(ValueError):
        _write(name="张" * 121)
    assert _write(name="张" * 120)["name"] == "张" * 120


def test_illegal_session_id_is_rejected():
    """会话 id 是目录名 —— 与 `questions.jsonl` 走同一道守卫(`../escape` 会建到录制根之外)。"""
    with pytest.raises(ValueError):
        session_meta.upsert_label("../escape", serial="01", name="张三",
                                  student_id="", department="")


def test_reading_a_session_that_never_reported_has_no_side_effect(_isolated):
    """读侧**不建目录**:读一场没报过标注的会话不该在盘上留下空会话目录。"""
    assert session_meta.read_label("20260101_000000_aaaa") is None
    assert not (_isolated / "20260101_000000_aaaa").exists()
