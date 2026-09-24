"""Task 5:face / gesture 的 `/analyze` 收 `session_id` —— 无 id 写 `NONE`、文件名带会话 id、非法 id 回 400。

两级断言一起上:

1. **文本级**(brief 给的那条):两个 app 里不再有 `uuid4()`、且用到 `NONE_SESSION`。
2. **行为级**(本任务加的):把模块的落盘目录 / 时钟 / 视觉管线换成替身,真的调一次端点,
   断言落盘**文件名**与**文件个数**。文本级断言挡不住"常量写对了却没接到路径上" —— 本项目
   已经反复栽在这类假绿上,而"每帧一个新文件"这个缺陷恰恰只在文件名上现形。

为什么要冻住时钟:`get_or_create_pipeline` 的旧实现用 `time.time()` 派生的墙上时间戳命名文件,
同一秒内的两帧会撞进同一个文件、跨秒的两帧则各写一个。把模块时钟每次推进 1000 秒,
"跨秒"这件事在测试里就**必然**发生 —— 旧实现因此必然给出两个文件名(见各条测试的 docstring)。
"""

from __future__ import annotations

import asyncio
import csv
import itertools
import json
import re
import sys
import types
from pathlib import Path

import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parent.parent
SID = "20260924_153012_9f3c"
BAD_SID = "../../x"


# ---------------------------------------------------------------------------
# 导入两个 app 模块(本环境有两个与本任务无关的缝,补最小替身)
# ---------------------------------------------------------------------------

def _install_env_shims() -> None:
    """补两个缝,否则两个 app 模块在本环境根本 import 不进来:

    - `face_expression/api/app.py` 用了 `File(...)`:FastAPI 在**定义路由时**就要
      `python_multipart`(未装;本任务不许装新包)。替身只要一个够大的 `__version__`。
    - `gesture_analysis/api/app.py` **模块级**构造 `mp.solutions.hands.Hands(...)`,
      而本环境的 mediapipe 1.0 已删掉 `solutions` 命名空间。替身只需满足 import 期的形状:
      `process()` 返回"没有手也没有姿态"的空结果,端点后半段的真实逻辑照常跑。

    两处都与"会话 id → 日志文件名"无关:被测逻辑一行都不在替身里。
    """
    if "python_multipart" not in sys.modules:
        stub = types.ModuleType("python_multipart")
        stub.__version__ = "0.0.20"          # FastAPI 只做字符串大小比较
        sys.modules["python_multipart"] = stub

    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        def _process(self, _image):
            return types.SimpleNamespace(multi_hand_landmarks=None, pose_landmarks=None)

        def _ctor(self, **_kwargs):
            pass

        mp.solutions = types.SimpleNamespace(
            hands=types.SimpleNamespace(
                Hands=type("Hands", (), {"__init__": _ctor, "process": _process})),
            pose=types.SimpleNamespace(
                Pose=type("Pose", (), {"__init__": _ctor, "process": _process})),
        )


_install_env_shims()

# 必须用 `import_module` 而不是 `import face_expression.api.app as face_app`:
# 两个包的 `api/__init__.py` 都写了 `from .app import app`,把包属性 `app` 覆盖成了
# **FastAPI 实例**;`import ... as` 走属性查找,于是拿到的是实例而不是模块(改它的
# `LOGS_DIR` 会 AttributeError)。`import_module` 回的是 `sys.modules` 里那份模块。
import importlib  # noqa: E402

import face_expression.utils.logger as face_logger_module          # noqa: E402
import gesture_analysis.utils.logger as gesture_logger_module      # noqa: E402

face_app = importlib.import_module("face_expression.api.app")
gesture_app = importlib.import_module("gesture_analysis.api.app")


# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------

class _FakeUpload:
    """替 `starlette.UploadFile`:端点只用到 `.content_type` / `.filename` / `await .read()`。"""

    def __init__(self, data: bytes, filename: str = "frame.jpg",
                 content_type: str = "image/jpeg"):
        self._data = data
        self.filename = filename
        self.content_type = content_type

    async def read(self) -> bytes:
        return self._data


class _FakeFacePipeline:
    """真的 `VideoPipeline` 要 mediapipe FaceMesh(本环境没有),而本测试测的是
    "会话 → 日志文件"的接线,不是视觉。替身返回一帧非空结果,好让端点真的走到
    `face_logger.log()` —— 否则"文件建了但一行没写"这种情况测不出来。
    """

    def __init__(self, fps: int = 30, session_id: str | None = None):
        self.fps = fps
        self.session_id = session_id

    def process_frame(self, _image_rgb):
        return object(), None, {"timestamp": 0.0, "focus_score": 0.5,
                                "dominant_emotion": "neutral", "confidence": 0.5}


_JPEG: list[bytes] = []


def _jpeg() -> bytes:
    if not _JPEG:
        import cv2
        import numpy as np
        ok, buf = cv2.imencode(".jpg", np.zeros((16, 16, 3), np.uint8))
        if not ok:
            raise RuntimeError("cv2 无法编码测试用 JPEG")
        _JPEG.append(buf.tobytes())
    return _JPEG[0]


def _rows(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _freeze_clock(app_module, monkeypatch) -> None:
    """把 app 模块的 `time.time()` 换成 1000, 2000, 3000, … 的假时钟。

    步长 1000 秒 > `SESSION_TIMEOUT`(300):旧实现每次调用都落在**不同**的墙上时间戳上,
    文件名的差别不再是"两次调用恰好跨没跨秒"的运气,而是必然。
    """
    ticks = itertools.count(1000.0, 1000.0)
    monkeypatch.setattr(app_module, "time",
                        types.SimpleNamespace(time=lambda: next(ticks)))


def _drive_face(session_id=None):
    return asyncio.run(face_app.analyze_frame(file=_FakeUpload(_jpeg()), session_id=session_id))


def _drive_gesture(session_id=None):
    return asyncio.run(gesture_app.analyze_image(file=_FakeUpload(_jpeg()), session_id=session_id))


@pytest.fixture
def face_env(tmp_path, monkeypatch):
    """落盘目录指到 tmp_path,视觉管线换成替身,并清空模块级的会话表(跨测试会残留)。

    `LOGS_DIR` 要改**两处**:app 里 `from face_expression.config import LOGS_DIR` 拿了一份,
    `face_expression/utils/logger.py` 里也拿了一份(DataLogger 的构造会用它建头文件)。
    只改一处的话,另一半仍然写进仓库的 `data/logs/`。
    """
    monkeypatch.setattr(face_app, "LOGS_DIR", str(tmp_path))
    monkeypatch.setattr(face_logger_module, "LOGS_DIR", str(tmp_path))
    monkeypatch.setattr(face_app, "VideoPipeline", _FakeFacePipeline)
    face_app.session_pipelines.clear()
    face_app.session_loggers.clear()
    return tmp_path


@pytest.fixture
def gesture_env(tmp_path, monkeypatch):
    """同上。gesture 这边 app 传的是 `log_file_path`,logger 自己的 LOGS_DIR 用不上;
    仍然一起改掉,免得哪次接线写歪就往仓库里落文件(测试要能响亮地失败在 tmp_path 上)。
    """
    monkeypatch.setattr(gesture_app, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(gesture_logger_module, "LOGS_DIR", tmp_path)
    gesture_app.session_analyzers.clear()
    gesture_app.session_loggers.clear()
    return tmp_path


# ---------------------------------------------------------------------------
# 文本级:不再 mint uuid、用 NONE_SESSION
# ---------------------------------------------------------------------------

def test_no_endpoint_mints_a_uuid_session():
    """无 id 时必须是 NONE:旧行为是每请求 uuid4 → 每帧一个新会话、日志文件爆炸。"""
    for rel in ("face_expression/api/app.py", "gesture_analysis/api/app.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "uuid4()" not in text, f"{rel} 仍在 mint uuid 会话"
        assert "NONE_SESSION" in text, f"{rel} 未使用 NONE_SESSION"


# ---------------------------------------------------------------------------
# face_expression/api/app.py
# ---------------------------------------------------------------------------

def test_face_no_id_writes_none_and_reuses_one_file(face_env, monkeypatch):
    """无 id:文件名带 `NONE`,且**连续两帧落进同一个文件**。

    「同一个文件」是这条的要点:旧实现 = 每请求一个新 uuid 会话(于是同一个客户端的
    每一帧都是新会话)+ 墙上时间戳文件名(于是跨秒就一个新文件),两者叠加 = 日志爆炸。
    这里时钟必然跨秒(见 `_freeze_clock`),所以旧实现必然给出多个文件、文件名里也没有 NONE。
    """
    _freeze_clock(face_app, monkeypatch)

    first = json.loads(_drive_face().body)
    second = json.loads(_drive_face().body)
    assert first["session_id"] == "NONE"
    assert second["session_id"] == "NONE"

    files = sorted(p.name for p in face_env.iterdir())
    assert files == ["face_au_log_NONE.csv"], files
    assert len(_rows(face_env / files[0])) == 2, "两帧没有都落进同一个文件"
    # app 是"构造后再覆盖 log_file";覆盖后仍必须是合法 CSV(表头在场)——靠 Task 3
    # 给 log() 加的那段"新路径缺表头就补写"
    header = (face_env / files[0]).read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("session_id,"), header


def test_face_with_id_names_the_file_after_the_session(face_env, monkeypatch):
    """有 id:日志文件名必须带它 —— 这是"三种日志能按会话归堆"的全部意义。

    旧实现把墙上时间戳写进文件名(app 构造后覆盖 `log_file`),id 只进 CSV 首列:
    两次不同秒的调用会各写一个文件,报告侧无法只看文件名就把它们归到一个会话。
    """
    _freeze_clock(face_app, monkeypatch)

    body = json.loads(_drive_face(session_id=SID).body)
    assert body["session_id"] == SID

    files = sorted(p.name for p in face_env.iterdir())
    assert files == [f"face_au_log_{SID}.csv"], files
    rows = _rows(face_env / files[0])
    assert [r["session_id"] for r in rows] == [SID]


def test_face_rejects_malformed_session_id_with_400(face_env):
    """非法 id(`../../x`)必须 400,且**不许**悄悄改成 NONE(本项目一贯:坏输入不吞)。

    这条同时是路径逃逸的闸:id 会被拼进 `face_au_log_{id}.csv`,
    `../../x` 能让日志文件写到日志目录**之外**(录制根之外)。
    """
    with pytest.raises(HTTPException) as ei:
        _drive_face(session_id=BAD_SID)
    assert ei.value.status_code == 400
    assert BAD_SID in ei.value.detail and "session_id" in ei.value.detail, ei.value.detail
    assert list(face_env.iterdir()) == [], "被拒的请求不该落任何文件"


# ---------------------------------------------------------------------------
# gesture_analysis/api/app.py
# ---------------------------------------------------------------------------

def test_gesture_no_id_writes_none_and_reuses_one_file(gesture_env, monkeypatch):
    """同 face:无 id → 文件名带 `NONE`,且连续两帧落同一个文件。"""
    _freeze_clock(gesture_app, monkeypatch)

    first = _drive_gesture()
    second = _drive_gesture()
    assert first["session_id"] == "NONE"
    assert second["session_id"] == "NONE"

    files = sorted(p.name for p in gesture_env.iterdir())
    assert files == ["gesture_emotion_log_NONE.csv"], files
    assert len(_rows(gesture_env / files[0])) == 2, "两帧没有都落进同一个文件"


def test_gesture_with_id_names_the_file_after_the_session(gesture_env, monkeypatch):
    """同 face:文件名带会话 id(旧实现是墙上时间戳)。"""
    _freeze_clock(gesture_app, monkeypatch)

    result = _drive_gesture(session_id=SID)
    assert result["session_id"] == SID

    files = sorted(p.name for p in gesture_env.iterdir())
    assert files == [f"gesture_emotion_log_{SID}.csv"], files
    rows = _rows(gesture_env / files[0])
    assert [r["session_id"] for r in rows] == [SID]


def test_gesture_rejects_malformed_session_id_with_400(gesture_env):
    """非法 id 回 400,且不落文件(同 face)。"""
    with pytest.raises(HTTPException) as ei:
        _drive_gesture(session_id=BAD_SID)
    assert ei.value.status_code == 400
    assert BAD_SID in ei.value.detail and "session_id" in ei.value.detail, ei.value.detail
    assert list(gesture_env.iterdir()) == [], "被拒的请求不该落任何文件"


# ---------------------------------------------------------------------------
# 守卫的模式本身(两个模块各持一份,值必须一致)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", [face_app, gesture_app], ids=["face", "gesture"])
def test_session_id_guard_accepts_only_safe_ids(module):
    """守卫按模式收:`NONE` 与 128 位合法;129 位 / 路径形状 / 空 / 非字符串一律 400。

    直接测助手(端点级已由上面几条覆盖):这里钉的是**模式本身** —— 128/129 这条边界
    最容易在"顺手放宽"时被改掉,而两个模块各持一份,值必须一样。
    """
    assert module.SESSION_ID_PAT.pattern == r"^[A-Za-z0-9_-]{1,128}$"
    assert module.validate_session_id("NONE") == "NONE"
    assert module.validate_session_id("a" * 128) == "a" * 128
    for bad in ("a" * 129, BAD_SID, "a/b", "", None):
        with pytest.raises(HTTPException) as ei:
            module.validate_session_id(bad)
        assert ei.value.status_code == 400, bad


# ---------------------------------------------------------------------------
# requirements.txt 与现实一致(本任务的第 4 项修正)
# ---------------------------------------------------------------------------

def test_requirements_drops_vosk_and_adds_websockets():
    """vosk 已随 2 GB 模型删除;ASR 现在走仓库内的 FunASR 客户端,它的传输层是 websockets。

    要求文件"描述现实":留着 vosk 会让照文件装环境的人拿到一个用不上的 Kaldi 包,
    缺 websockets 则会让 ASR 一 import 就崩。
    """
    lines = [ln.strip() for ln in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()]
    deps = [re.split(r"[<>=!~\[\s]", ln)[0].lower() for ln in lines
            if ln and not ln.startswith("#")]
    assert "vosk" not in deps, deps
    assert "websockets" in deps, deps
