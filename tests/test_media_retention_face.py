# tests/test_media_retention_face.py
"""M2.6:face 端点真的把收到的字节交给留存了(接线正确)。

为什么单独一个文件:这是一个**接线**测试 —— 模块本身在 test_media_retention.py 里
已经测透,这里只回答"端点在正确的时机、用正确的参数调了它吗"。
"""
import asyncio
import importlib
import json
import types

import pytest

# ⚠️ 这里**不要**照抄 test_analyze_session_fallback.py 的 python_multipart 替身:
# 那条注释("本环境连 python-multipart 都没装")已经过期 —— 实测真包就装在
# site-packages/python_multipart/。装残缺替身会让本文件**单独跑就崩**
# (starlette 要从它 import `__all__`),只在全量套件里靠"别的测试先 import 过真包"
# 侥幸通过 —— 那就是顺序依赖的假绿。
import media_retention                                              # noqa: E402
face_app = importlib.import_module("face_expression.api.app")       # noqa: E402


class _FakeUpload:
    def __init__(self, data: bytes):
        self._data, self.filename, self.content_type = data, "frame.jpg", "image/jpeg"

    async def read(self) -> bytes:
        return self._data


class _FakeRequest:
    async def form(self):
        raise RuntimeError('Form data requires "python-multipart" to be installed.')


class _FakeImage:
    """替 `cv2.imread` 的返回:端点会读 `.shape` 去写日志,所以替身得带上它。"""
    shape = (720, 1280, 3)


class _FakePipeline:
    def __init__(self, session_id=None):
        self.session_id = session_id

    def process_frame(self, _image_rgb, timestamp_ms):
        return object(), None, {"timestamp": timestamp_ms / 1000.0, "focus_score": 0.5}

    def measured_fps(self):
        return 1.0

    def close(self):
        pass


@pytest.fixture
def wired(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.delenv("JINGXIN_RETAIN_MEDIA", raising=False)
    media_retention._reset_for_tests()
    monkeypatch.setattr(face_app, "get_or_create_pipeline", lambda sid: _FakePipeline(sid))
    monkeypatch.setattr(face_app, "cv2", types.SimpleNamespace(
        imread=lambda p: _FakeImage(), cvtColor=lambda i, c: i, COLOR_BGR2RGB=1))
    monkeypatch.setattr(face_app, "session_loggers", {})
    return tmp_path


def _post(payload: bytes, sid: str):
    return asyncio.run(face_app.analyze_frame(
        request=_FakeRequest(), file=_FakeUpload(payload), session_id=sid, fps=30))


def test_endpoint_hands_the_bytes_to_retention(wired):
    """端点收到的字节,必须原样出现在盘上。

    红法:删掉端点里那一行 retain_frame —— 本测试立刻红(盘上没有文件),
    而"特征照样算得出来"这个假象会让问题一路滑到重抽那天才暴露。
    """
    payload = b"\xff\xd8\xff\xe0" + b"A" * 100
    _post(payload, "20260925_120000_aaaa")
    f = wired / "20260925_120000_aaaa" / "media" / "face" / "000001.jpg"
    assert f.read_bytes() == payload


def test_declared_ts_matches_the_session_clock(wired):
    """账本里的 `declared_ts` 必须是**服务端给这一帧的那个时间戳**(M2.5 的会话相对时钟)。

    为什么重要:重抽时必须喂当时那个值,不能用墙钟 —— 否则重抽出来的数不等于当场
    算的数,而"逐格相等"正是留存的验收判据(spec §7.1)。
    红法:把 declared_ts 换成 time.time()。
    """
    _post(b"\xff\xd8jpeg", "20260925_120000_bbbb")
    line = json.loads(media_retention.ledger_path("20260925_120000_bbbb", "face")
                      .read_text(encoding="utf-8").splitlines()[0])
    # 首帧 → 会话相对时钟约等于 0(不写死 == 0:机器慢时可能已经过了 1 ms)
    assert 0 <= line["declared_ts"] < 1000
    assert line["source_endpoint"] == "/analyze"
    assert line["declared_ts"] != int(line["received_at_wall"])   # 不是墙钟


def test_retention_off_leaves_no_files_and_still_analyzes(wired, monkeypatch):
    """关掉留存 → 盘上什么都没有,但**请求照常成功**(它是旁路)。

    红法:把端点里的调用写在 if enabled() 之外、或让留存失败冒泡成 500 ——
    关掉留存就整场用不了。
    """
    monkeypatch.setenv("JINGXIN_RETAIN_MEDIA", "0")
    r = _post(b"\xff\xd8jpeg", "20260925_120000_cccc")
    assert not (wired / "20260925_120000_cccc" / "media").exists()
    assert r is not None
