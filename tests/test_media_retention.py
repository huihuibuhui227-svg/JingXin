# tests/test_media_retention.py
"""M2.6 服务端留存:路径/守卫/开关/预检这一层。

这一层是"素材会不会被静默丢掉"的唯一防线,所以每条测试都要能说出
**撤掉哪个生产改动会让它变红**。
"""
import hashlib
import json
import threading
from pathlib import Path

import pytest

import media_retention


@pytest.fixture(autouse=True)
def _isolated_root(tmp_path, monkeypatch):
    """每个测试都落在自己的临时目录里 —— 绝不碰真的 ~/shared。"""
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.delenv("JINGXIN_RETAIN_MEDIA", raising=False)
    media_retention._reset_for_tests()
    return tmp_path


def test_default_root_is_the_shared_volume():
    """默认落点必须是 ~/shared/jingxin_recordings(→ /mnt/d)。

    红法:把 DEFAULT_ROOT 改成仓库内的路径 —— 那会让原始媒体进 git,
    也会撑 WSL 的 ext4.vhdx(下一步 §4.7)。
    """
    assert media_retention.DEFAULT_ROOT == Path.home() / "shared" / "jingxin_recordings"


def test_enabled_by_default_and_can_be_switched_off(monkeypatch):
    """默认开 —— "忘了开"正是丢素材的那条路径(spec §6)。"""
    assert media_retention.enabled() is True
    for off in ("0", "false", "FALSE", "no", "off", " off "):
        monkeypatch.setenv("JINGXIN_RETAIN_MEDIA", off)
        assert media_retention.enabled() is False, off


def test_path_traversal_is_rejected():
    """会话 id 是客户端可控的,而它会被拼进路径 —— 必须拦在拼路径之前。

    红法:去掉 validate_session_id 里的正则。`../../jingxin/x` 就能在录制根
    **之外**建目录并写入原始媒体。
    """
    for bad in ("../x", "a/b", "..", "", "  ", "a" * 129, "a.b"):
        with pytest.raises(ValueError):
            media_retention.validate_session_id(bad)
    assert media_retention.validate_session_id("20260925_124346_6e4a") == "20260925_124346_6e4a"
    assert media_retention.validate_session_id("NONE") == "NONE"


def test_recording_dir_honors_the_env_override(_isolated_root):
    d = media_retention.recording_dir("s1")
    assert d == _isolated_root / "s1"
    assert d.is_dir()


def test_preflight_is_loud_when_the_root_is_not_writable(tmp_path, monkeypatch):
    """首件素材预检不过 → **抛**,不是 warning。

    红法:把 _prepare 里的 raise 改成 return False —— 于是磁盘坏了也一路静默,
    跑完一场才发现一张图都没存(而这一场是人重跑不回来的)。
    """
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    blocked.chmod(0o500)                       # 只读目录
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(blocked / "sub"))
    media_retention._reset_for_tests()
    if media_retention._can_write_here(blocked):   # root 用户跑测试时 chmod 拦不住
        pytest.skip("本环境以 root 运行,chmod 拦不住写")
    with pytest.raises(RuntimeError, match="预检"):
        media_retention.retain_frame("s1", "face", b"\xff\xd8jpeg", declared_ts=0)


def test_preflight_failure_is_only_loud_once(tmp_path, monkeypatch):
    """首件大声报过之后,后续件不再抛 —— 中断也换不回已丢的字节,还白耗人的时间。

    但要**留痕**(见 Task 4)。红法:把 _LOUD_DONE 判断去掉 → 每一帧都 500,
    整场用不了。
    """
    blocked = tmp_path / "blocked2"
    blocked.mkdir()
    blocked.chmod(0o500)
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(blocked / "sub"))
    media_retention._reset_for_tests()
    if media_retention._can_write_here(blocked):
        pytest.skip("本环境以 root 运行,chmod 拦不住写")
    with pytest.raises(RuntimeError):
        media_retention.retain_frame("s1", "face", b"x", declared_ts=0)
    assert media_retention.retain_frame("s1", "face", b"x", declared_ts=1) is None
    assert media_retention.degraded_reasons("s1")
