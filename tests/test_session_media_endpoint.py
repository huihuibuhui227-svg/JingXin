# tests/test_session_media_endpoint.py
"""`POST /session/{sid}/media`:前端原生视频的落点(spec §5.3)。

它**不分析任何东西** —— 只落盘 + 记账。所以这里的断言全是"字节与台账"。
"""
import asyncio
import importlib

import pytest

import media_retention
voice_app = importlib.import_module("voice_interaction.api.app")

WEBM = b"\x1a\x45\xdf\xa3" + b"native-camera-payload" * 40
SID = "20260925_203826_2449"


class _FakeUpload:
    def __init__(self, data, name="camera.webm"):
        self._data, self.filename, self.content_type = data, name, "video/webm"

    async def read(self):
        return self._data


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.delenv("JINGXIN_RETAIN_MEDIA", raising=False)
    media_retention._reset_for_tests()
    return tmp_path


def test_uploaded_video_is_retained(_isolated):
    """200 + 文件逐字节在盘上 + 台账一行。"""
    r = asyncio.run(voice_app.submit_session_media(session_id=SID, file=_FakeUpload(WEBM)))
    assert r["status"] == "success" and r["stored"] is True
    p = _isolated / SID / "media" / "camera.webm"
    assert p.read_bytes() == WEBM
    assert r["sha256"] == media_retention.hashlib.sha256(WEBM).hexdigest()


def test_retention_off_says_so_instead_of_pretending(_isolated, monkeypatch):
    """留存关了 → 200 但**明说没存**。不许静默成功让人以为存下了。"""
    monkeypatch.setenv("JINGXIN_RETAIN_MEDIA", "0")
    r = asyncio.run(voice_app.submit_session_media(session_id=SID, file=_FakeUpload(WEBM)))
    assert r["status"] == "success"
    assert r["stored"] is False
    assert r["reason"], "说了没存,却没给原因"
    assert not (_isolated / SID).exists()


@pytest.mark.parametrize("bad", ["../escape", "a/b", "..", ""])
def test_illegal_session_id_is_400_and_writes_nothing(_isolated, bad):
    """路径穿越守卫:非法 id → 400,盘上不得出现越界目录。"""
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as ei:
        asyncio.run(voice_app.submit_session_media(session_id=bad, file=_FakeUpload(WEBM)))
    assert ei.value.status_code == 400
    assert not (_isolated.parent / "escape").exists()
