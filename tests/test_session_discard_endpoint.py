# tests/test_session_discard_endpoint.py
"""`POST /session/{sid}/discard`:把一场素材从盘上**彻底**抹掉。

它存在的理由:录制页在「不留存 / 这是测试」那条路上要当场删掉本场,而
**录制的时候没有管理员 token**(那是素材页的东西,且 token 在 sessionStorage 里
按标签页隔离)。所以这个入口在语音服务上,与它同组的 `/label`、`/question`
同一姿态 —— 那些本来就能改身份数据。

它**没有密码**是使用者确认过的取舍,不是遗漏。所以这里要钉的是**删得对不对**:
两处都删(场次目录 + 日志 CSV)、不碰别人、拒绝 `NONE`。
"""
import asyncio
import importlib

import pytest

import media_retention as mr
import session_purge
from fastapi import HTTPException

voice_app = importlib.import_module("voice_interaction.api.app")

SID = "20260928_212130_5b10"
OTHER = "20260928_180131_8102"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    root = tmp_path / "recordings"
    root.mkdir()
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(root))
    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(session_purge, "default_logs_dir", lambda: logs)
    mr._reset_for_tests()
    return root, logs


def _session(root, sid):
    d = root / f"18-赵六-2021004-物理学院__{sid}"
    (d / "media" / "face").mkdir(parents=True)
    (d / "media" / "face" / "000001.jpg").write_bytes(b"\xff\xd8\xff")
    (d / "label.json").write_text("{}", encoding="utf-8")
    return d


def _discard(sid=SID):
    return asyncio.run(voice_app.discard_session(session_id=sid))


def test_discard_removes_the_session_dir_and_the_log_csvs(_isolated):
    """★ 红法:把 `session_purge.purge_session` 里的 `log_csvs` 那一段去掉。

    少删日志 ⟹ 报告层看到一场**有日志没素材**的会话。它看起来只是"这一场没采到",
    不是"这一场被作废了" —— 两种解释指向完全不同的处置。
    """
    root, logs = _isolated
    d = _session(root, SID)
    csvs = [logs / f"{p}_{SID}.csv" for p in ("face_au_log", "gesture_emotion_log")]
    for c in csvs:
        c.write_text("x", encoding="utf-8")

    r = _discard()
    assert r["status"] == "success" and r["session_id"] == SID, r
    assert len(r["logs"]) == 2, r
    assert not d.exists() and not any(c.exists() for c in csvs)


def test_discard_leaves_the_neighbouring_session_alone(_isolated):
    """★ 红法:把 `mr.resolve_recording_dir` 换成 `mr.root() / sid`(少了后缀解析)。"""
    root, logs = _isolated
    keep = _session(root, OTHER)
    _session(root, SID)
    other_csv = logs / f"face_au_log_{OTHER}.csv"
    other_csv.write_text("x", encoding="utf-8")

    _discard()
    assert keep.is_dir() and other_csv.is_file(), "把隔壁那一场也删了"


def test_discard_refuses_the_none_bucket(_isolated):
    """★ 红法:去掉 `session_purge` 里的 NONE 守卫。"""
    root, _ = _isolated
    (root / "NONE").mkdir()
    with pytest.raises(HTTPException) as e:
        _discard("NONE")
    assert e.value.status_code == 400
    assert (root / "NONE").is_dir()


def test_discard_refuses_a_malformed_sid(_isolated):
    """★ 红法:去掉 `SID_PAT.fullmatch` 那一句。"""
    for bad in ("..", "../../etc", "a/b", "my_notes"):
        with pytest.raises(HTTPException) as e:
            _discard(bad)
        assert e.value.status_code in (400, 404), f"{bad} → {e.value.status_code}"


def test_discard_reports_404_when_there_is_nothing_to_delete(_isolated):
    """★ 红法:把 `FileNotFoundError` 也当成 success 回。

    "什么都没删"与"删掉了"必须分得开 —— 前者会让前端显示"已删除"而盘上本来就没东西。
    """
    with pytest.raises(HTTPException) as e:
        _discard()
    assert e.value.status_code == 404
