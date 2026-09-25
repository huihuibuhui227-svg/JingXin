# tests/test_media_retention_gesture.py
"""M2.6:gesture 端点真的把收到的字节交给留存了(接线正确)。"""
import asyncio
import importlib
import json
import types

import pytest

# ⚠️ 不装 python_multipart 替身:真包装着(见 test_media_retention_face.py 的注释)。
import media_retention                                                # noqa: E402
gesture_app = importlib.import_module("gesture_analysis.api.app")     # noqa: E402


class _FakeUpload:
    def __init__(self, data: bytes):
        self._data, self.filename, self.content_type = data, "f.jpg", "image/jpeg"

    async def read(self) -> bytes:
        return self._data


class _FakeRequest:
    async def form(self):
        raise RuntimeError("no multipart")


class _FakeImage:
    """替 `cv2.imdecode` 的返回:端点会读 `.shape` 写日志。"""
    shape = (720, 1280, 3)


class _FakeAnalyzer:
    """手套/肩/臂分析器的最小替身:端点只用到 `update` 与 `get_results`。"""
    def update(self, _landmarks): pass
    def get_results(self): return {}


class _FakeEmotionAnalyzer:
    def infer_emotion(self, *_args):
        # 端点 `:427-433` 会读这五个键(少一个就是 KeyError,而不是我要的那条断言红)
        return {"emotion_state": "neutral", "overall_score": 50.0, "emoji": "😐",
                "feedback": "", "used_features": []}


@pytest.fixture
def wired(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path))
    monkeypatch.delenv("JINGXIN_RETAIN_MEDIA", raising=False)
    media_retention._reset_for_tests()
    monkeypatch.setattr(gesture_app, "cv2", types.SimpleNamespace(
        imdecode=lambda buf, flag: _FakeImage(), cvtColor=lambda i, c: i,
        IMREAD_COLOR=1, COLOR_BGR2RGB=1))
    monkeypatch.setattr(gesture_app, "np", types.SimpleNamespace(
        frombuffer=lambda b, t: b, uint8="u8"))
    analyzers = {k: _FakeAnalyzer() for k in
                 ("left_hand", "right_hand", "shoulder", "left_arm", "right_arm")}
    analyzers["emotion"] = _FakeEmotionAnalyzer()
    monkeypatch.setattr(gesture_app, "get_or_create_analyzers", lambda sid: analyzers)
    monkeypatch.setattr(gesture_app, "get_or_create_detectors", lambda sid: {
        "hands": types.SimpleNamespace(detect=lambda img, ts: []),
        "pose": types.SimpleNamespace(detect=lambda img, ts: []),
    })
    monkeypatch.setattr(gesture_app, "session_loggers", {})
    return tmp_path


def _post(payload: bytes, sid: str):
    return asyncio.run(gesture_app.analyze_image(
        request=_FakeRequest(), file=_FakeUpload(payload), session_id=sid))


def test_gesture_endpoint_hands_the_bytes_to_retention(wired):
    """红法:删掉端点里那一行 retain_frame —— 手势素材从此静默不留。"""
    payload = b"\xff\xd8\xff\xe0" + b"G" * 77
    _post(payload, "20260925_120000_dddd")
    f = wired / "20260925_120000_dddd" / "media" / "gesture" / "000001.jpg"
    assert f.read_bytes() == payload


def test_gesture_declared_ts_comes_from_its_own_clock(wired):
    """gesture 有自己的会话时钟(M2.5 Task 6 加的),不是复用 face 的。"""
    _post(b"\xff\xd8jpeg", "20260925_120000_eeee")
    line = json.loads((wired / "20260925_120000_eeee" / media_retention.RETENTION_FILENAME)
                      .read_text(encoding="utf-8").splitlines()[0])
    assert 0 <= line["declared_ts"] < 1000          # 首帧 ≈ 0(不写死 == 0,机器慢时可能过了 1 ms)
    assert line["source"] == "/analyze"
    assert line["modality"] == "gesture"
