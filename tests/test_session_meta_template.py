# tests/test_session_meta_template.py
"""`meta.json`:模板能生成、缺项能报出来、且**不阻断分析**(spec §5.5 / §6)。

覆盖面来自录制需求 §3.2 —— 那一整张表就是"必填"的定义。
"""
import importlib
import json

import pytest

session_meta = importlib.import_module("session_meta")

SID = "20260925_203826_2449"


def _set(obj: dict, dotted: str, value) -> None:
    """按点分路径写进嵌套 dict(路径中间的层不存在就建)。"""
    parts = dotted.split(".")
    for part in parts[:-1]:
        obj = obj.setdefault(part, {})
    obj[parts[-1]] = value


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    session_meta._reset_for_tests()
    return tmp_path


def test_template_is_written_with_the_session_id_filled_in(_isolated):
    p = session_meta.write_template(SID)
    assert p == _isolated / SID / "meta.json"
    got = json.loads(p.read_text(encoding="utf-8"))
    assert got["session_id"] == SID
    assert got["recorded_at"], "生成时该把录制时刻填上"


def test_fresh_template_reports_every_required_field_as_missing(_isolated):
    """刚生成的空模板:所有必填项都该被点出来(不是"校验通过")。"""
    session_meta.write_template(SID)
    missing = session_meta.missing_meta_fields(SID)
    assert set(missing) == set(session_meta.META_REQUIRED), missing


def test_blank_and_null_count_as_missing_not_as_filled(_isolated):
    """空串 / None / `False` **同样是缺** —— 只判键在不在会让全空模板假绿。"""
    session_meta.write_template(SID)
    p = _isolated / SID / "meta.json"
    filled = json.loads(p.read_text(encoding="utf-8"))
    for path in session_meta.META_REQUIRED:
        _set(filled, path, "x")
    _set(filled, "consent.archived", True)     # 布尔项填成真才算填了
    p.write_text(json.dumps(filled, ensure_ascii=False), encoding="utf-8")
    assert session_meta.missing_meta_fields(SID) == []

    for bad in ("", None):
        _set(filled, "candidate.sex", bad)
        p.write_text(json.dumps(filled, ensure_ascii=False), encoding="utf-8")
        assert "candidate.sex" in session_meta.missing_meta_fields(SID), repr(bad)

    # `consent.archived: false` = "知情同意还没归档" ⟹ **必须算缺**
    # (模板刚生成时它就是 false;不把 False 当缺的话,没做知情同意的场次会过校验)
    _set(filled, "candidate.sex", "M")
    _set(filled, "consent.archived", False)
    p.write_text(json.dumps(filled, ensure_ascii=False), encoding="utf-8")
    assert "consent.archived" in session_meta.missing_meta_fields(SID)


def test_missing_meta_file_reports_all_required(_isolated):
    """连 meta.json 都没有 → 全部必填项都缺(不是抛异常)。"""
    assert set(session_meta.missing_meta_fields(SID)) == set(session_meta.META_REQUIRED)


def test_required_list_covers_the_recording_requirements_table():
    """把录制需求 §3.2 那张表的**每一项**都钉住,防止有人删项。"""
    for path in ["candidate.sex", "candidate.age", "candidate.native_language",
                 "candidate.dialect_region", "capture.device", "capture.resolution",
                 "capture.camera_distance_cm", "capture.lighting", "capture.mic_gain_db",
                 "interviewer_ratings.logical_thinking",
                 "interviewer_ratings.communication",
                 "interviewer_ratings.confidence", "consent.archived"]:
        assert path in session_meta.META_REQUIRED, f"漏了必填项 {path}"
