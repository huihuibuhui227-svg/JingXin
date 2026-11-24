# tests/test_anonymous_privacy_mode.py
"""`anonymous_kinematics_only` 零原始图像/音频落盘隐私模式回归测试。"""
from __future__ import annotations

import media_retention
import session_meta


def test_anonymous_kinematics_privacy_scope_blocks_raw_disk_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    media_retention._reset_for_tests()

    sid = "20261124_230000_priv"

    # 1. 上下文头触发 anonymous_kinematics_only：不落 jpg/wav/webm
    with media_retention.privacy_scope("anonymous_kinematics_only"):
        assert media_retention.is_anonymous_kinematics_only(sid) is True
        assert media_retention.retain_frame(sid, "face", b"\xff\xd8\xff\xe0abc", declared_ts=100) is None
        assert media_retention.retain_frame(sid, "gesture", b"\xff\xd8\xff\xe0def", declared_ts=100) is None
        assert media_retention.retain_audio(sid, "raw", b"RIFF" + b"\x00" * 60) is None
        assert media_retention.retain_uploaded_video(sid, b"\x1a\x45\xdf\xa3" + b"\x00" * 30) is None

    # 盘上零原始影像或音频文件
    assert list(tmp_path.rglob("*.jpg")) == []
    assert list(tmp_path.rglob("*.wav")) == []
    assert list(tmp_path.rglob("*.webm")) == []

    # 2. 持久化 label consent="anonymous_kinematics_only" 即使无请求头也拦截
    rec = session_meta.upsert_label(
        sid,
        serial="01",
        name="匿名档案A",
        student_id="SUBJ-101",
        department="行为随访",
        consent="anonymous_kinematics_only",
    )
    assert rec["consent"] == "anonymous_kinematics_only"
    assert media_retention.is_anonymous_kinematics_only(sid) is True
    assert media_retention.retain_frame(sid, "face", b"\xff\xd8\xff\xe0xyz", declared_ts=200) is None
    assert list(tmp_path.rglob("*.jpg")) == []
