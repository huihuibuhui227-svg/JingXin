"""把「帧取景」接进实时采集路径（`face_expression/api/app.py`）的钉子。

这套测试盯的是**接入正确性**，不是取景算法本身（那在 `test_face_framing.py`）。
每条都对着一个**会真的出事**的点：

- ★ **留存拿到的是原始上传字节**：裁切作用在**解码后的图**上，而留存收的是
  `contents`（原始字节）—— 这两条路是分开的。所以**不管取景怎么接，全画幅都留在盘上、
  随时可离线重提取**。这是整个改动的安全垫。
  （⚠️ 更正：本条最初写成「顺序写反就永久丢数据」，**那是错的** —— 实测把取景挪到
  留存之前，这条测试仍然全绿，因为两个变量互不影响。真正值得钉的「先留存后处理」
  在下面那条。）
- ★ **先留存、后处理**：处理抛异常时素材不能丢（这是 `retain_frame` 那句注释的本意）。
- **取景默认关闭**（2026-10-03 定的协议）：环境变量没设时活跑**不裁切**，
  取景交给离线探针统一决定；`JX_FACE_FRAMING=on` 能把活跑裁切装回去。
  两条路都要能真的走通（"默认关"不能靠蒙混，要能真的关掉、也能真的开起来）。
- **一场只决定一次**：逐帧探测既慢又会污染 `*_trend`（滚动窗口）。
- **探测失败不能拖垮采集**：取景是优化，不该让端点 500。
- **`/reset` 要清掉取景状态**：否则下一次同 id 的会话会继承上一场的取景。
"""
from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from starlette.requests import Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import face_expression.utils.logger as face_logger_module  # noqa: E402

face_app = importlib.import_module("face_expression.api.app")

CENTER50 = (0.25, 0.00, 0.75, 0.75)


# ──────────────────────────────────────────────────────────────
# 替身与钩子
# ──────────────────────────────────────────────────────────────

class _FakeUpload:
    def __init__(self, data: bytes, filename: str = "frame.jpg",
                 content_type: str = "image/jpeg"):
        self._data, self.filename, self.content_type = data, filename, content_type

    async def read(self) -> bytes:
        return self._data


class _FakeRequest:
    def __init__(self, form):
        self._form = form

    async def form(self):
        if self._form is None:
            raise RuntimeError("表单解析不了")
        return self._form


class _RecordingPipeline:
    """记录**它收到了什么形状的图** —— 取景有没有生效全看这个。

    返回非空特征字典，好让端点真的走到 `face_logger.log()`。
    """

    received: list = []

    def __init__(self, session_id=None):
        self.session_id = session_id

    def process_frame(self, image_rgb, timestamp_ms):
        _RecordingPipeline.received.append(tuple(image_rgb.shape))
        return object(), None, {"timestamp": timestamp_ms / 1000.0, "focus_score": 0.5,
                                "dominant_emotion": "neutral", "confidence": 0.5}

    def measured_fps(self):
        return 1.0

    def close(self):
        pass


_JPEG: list = []


def _jpeg(w: int = 200, h: int = 100) -> bytes:
    """固定尺寸的测试 JPEG。取景断言要拿尺寸比，所以尺寸必须可控。"""
    import cv2
    ok, buf = cv2.imencode(".jpg", np.zeros((h, w, 3), np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(face_app, "LOGS_DIR", str(tmp_path))
    monkeypatch.setattr(face_logger_module, "LOGS_DIR", str(tmp_path))
    monkeypatch.setattr(face_app, "VideoPipeline", _RecordingPipeline)
    monkeypatch.setattr(face_app, "FACE_FRAMING_MANIFEST", str(tmp_path / ".framing.json"))
    monkeypatch.setattr(face_app, "_FACE_FRAMING_ENABLED", True)

    _RecordingPipeline.received = []
    face_app.session_pipelines.clear()
    face_app.session_loggers.clear()
    face_app.session_clocks.clear()
    face_app.session_framing.clear()

    # 冻住时钟：TTL 回收那套逻辑与本文件无关，别让它干扰
    import itertools, types
    ticks = itertools.count(1000.0, 1.0)
    monkeypatch.setattr(face_app, "time", types.SimpleNamespace(time=lambda: next(ticks)))
    return tmp_path


def _probe_spy(name="center50", crop=CENTER50):
    """把 `probe_crop` 换成可控替身，并记录被调用了几次。"""
    calls: list = []

    def spy(frames, timestamps, **kw):
        calls.append({"frames": len(frames), "probe_n": kw.get("probe_n")})
        if name == "raise":
            raise RuntimeError("探测炸了")
        return name, crop, {"probed": len(frames)}

    spy.calls = calls
    return spy


def _drive(session_id="S1", data=None):
    return asyncio.run(face_app.analyze_frame(
        request=_FakeRequest({}), file=_FakeUpload(data or _jpeg()),
        session_id=session_id))


# ──────────────────────────────────────────────────────────────
# ★ 最重要的一条：留存必须是原始帧
# ──────────────────────────────────────────────────────────────

def test_retained_bytes_are_the_original_frame_not_the_crop(env, monkeypatch):
    """`retain_frame` 收到的必须是**上传的原始字节**，不是裁切后的图。

    生产改动会让它变红：把 `retain_frame` 的第三个实参从 `contents` 换成裁切后的图
    （例如 `apply_crop(image, crop)`）。

    ⚠️ **这条抓不住"顺序"** —— 实测把 `_framing_for` 挪到 `retain_frame` 之前，
    它照样绿（`contents` 与 `image` 是两个变量）。顺序那条由下面
    `test_frame_is_retained_even_when_processing_raises` 钉。
    """
    original = _jpeg(200, 100)
    seen = {}

    def spy_retain(session_id, modality, contents, **kw):
        seen["contents"] = contents
        seen["modality"] = modality

    monkeypatch.setattr(face_app.media_retention, "retain_frame", spy_retain)
    monkeypatch.setattr(face_app, "probe_crop", _probe_spy("center50"))

    _drive(data=original)

    assert seen["contents"] == original, "留存存到的不是原始上传字节 —— 全画幅丢了"


def test_frame_is_retained_even_when_processing_raises(env, monkeypatch):
    """★ 「先留存、后处理」：处理抛异常时素材不能丢。

    这是 `retain_frame` 那句注释（"先把这一帧的原始字节存一份,再算 —— 算的过程中
    抛异常也不丢素材"）的本意，也是这套取景改动必须保住的性质。

    生产改动会让它变红：把 `retain_frame` 挪到 `process_frame` **之后**
    （或挪到 `_framing_for` 之后而 `_framing_for` 改成不吞异常）。
    """
    seen = {}

    def spy_retain(session_id, modality, contents, **kw):
        seen["retained"] = True

    class _Exploding(_RecordingPipeline):
        def process_frame(self, image_rgb, timestamp_ms):
            raise RuntimeError("处理炸了")

    monkeypatch.setattr(face_app.media_retention, "retain_frame", spy_retain)
    monkeypatch.setattr(face_app, "VideoPipeline", _Exploding)
    monkeypatch.setattr(face_app, "probe_crop", _probe_spy("center50"))

    # 端点会把内部异常转成 HTTPException(500)，所以这里接的是它，不是 RuntimeError
    with pytest.raises(HTTPException):
        _drive(data=_jpeg(200, 100))

    assert seen.get("retained") is True, "处理抛异常前没留存 —— 这一帧素材丢了"


def test_pipeline_receives_the_cropped_image_when_a_crop_is_decided(env, monkeypatch):
    """决定了取景，就得**真的裁了再喂给管线** —— 否则取景只是个摆设。"""
    monkeypatch.setattr(face_app, "probe_crop", _probe_spy("center50"))
    _drive(data=_jpeg(200, 100))

    # 上传的是 100 高 × 200 宽；center50 = 宽取 50%、高取 75% ⟹ 75 × 100
    assert _RecordingPipeline.received, "管线一次都没被调用"
    h, w, _ = _RecordingPipeline.received[-1]
    assert (h, w) == (75, 100), f"喂给管线的不是裁切后的图，实得 {h}×{w}"


def test_full_frame_is_untouched_when_probe_says_full(env, monkeypatch):
    """全画幅就可用时，图必须**原样**进管线（这条路径零行为变化）。"""
    monkeypatch.setattr(face_app, "probe_crop", _probe_spy("full", None))
    _drive(data=_jpeg(200, 100))

    h, w, _ = _RecordingPipeline.received[-1]
    assert (h, w) == (100, 200)


# ──────────────────────────────────────────────────────────────
# 开关与决策次数
# ──────────────────────────────────────────────────────────────

def test_framing_is_off_by_default(monkeypatch):
    """★ 环境变量**没设**时，取景是关的 —— 这是 2026-10-03 定的取景协议。

    协议：活跑不裁切（原始全画幅进留存），取景由离线
    `tools/jx_backfill_face.py --uniform` 用同一套探针统一决定 ⟹ 全语料同一套协议。

    这条红了 ⟹ 默认值被改回 `on`，"直接跑 python -m face_expression.api.app"
    这条启动路径就会偷偷裁切，与经 `start_all.sh` / systemd 启动的行为分叉，
    "所有场次同一套取景协议"当场不成立。
    """
    monkeypatch.delenv("JX_FACE_FRAMING", raising=False)
    assert face_app.FACE_FRAMING_DEFAULT == "off", "默认值不是 off"
    # 直接调生产代码里那段解析逻辑（导入时的 `_FACE_FRAMING_ENABLED` 已经固定，改不了）
    assert face_app._env_flag("JX_FACE_FRAMING", face_app.FACE_FRAMING_DEFAULT) is False


def test_framing_can_be_turned_back_on(monkeypatch):
    """`JX_FACE_FRAMING=on` 能把活跑裁切装回去（保留这个逃生口）。

    这条红了 ⟹ 开启路径坏了（想恢复活跑裁切时会发现开关失效）。
    """
    monkeypatch.setenv("JX_FACE_FRAMING", "on")
    assert face_app._env_flag("JX_FACE_FRAMING", face_app.FACE_FRAMING_DEFAULT) is True


@pytest.mark.parametrize("val", ["off", "OFF", "0", "false", "False", "no", " no "])
def test_framing_env_falsey_spellings(monkeypatch, val):
    """`off` 的几种写法都得认（含大小写与首尾空格）。

    这条红了 ⟹ 有人写成 `JX_FACE_FRAMING=False` 却仍然在裁切。
    """
    monkeypatch.setenv("JX_FACE_FRAMING", val)
    assert face_app._env_flag("JX_FACE_FRAMING", "on") is False


@pytest.mark.parametrize("val", ["on", "ON", "1", "true", "yes"])
def test_framing_env_truthy_spellings(monkeypatch, val):
    """`on` 的几种写法都得认。"""
    monkeypatch.setenv("JX_FACE_FRAMING", val)
    assert face_app._env_flag("JX_FACE_FRAMING", "off") is True


def test_framing_off_disables_cropping_entirely(env, monkeypatch):
    """`JX_FACE_FRAMING=off` 必须真的关掉 —— 而且**一次探测都不做**。"""
    monkeypatch.setattr(face_app, "_FACE_FRAMING_ENABLED", False)
    spy = _probe_spy("center50")
    monkeypatch.setattr(face_app, "probe_crop", spy)

    _drive(data=_jpeg(200, 100))

    assert spy.calls == [], "关掉了却还在探测"
    h, w, _ = _RecordingPipeline.received[-1]
    assert (h, w) == (100, 200), "关掉了却还是裁了"
    assert face_app.session_framing == {}, "关掉了却写了取景状态"


def test_framing_is_decided_once_per_session(env, monkeypatch):
    """一场只探测一次 —— 逐帧探测既慢、又会污染 `*_trend`（滚动窗口）。"""
    spy = _probe_spy("center50")
    monkeypatch.setattr(face_app, "probe_crop", spy)

    for _ in range(3):
        _drive(session_id="S-multi")

    assert len(spy.calls) == 1, f"探测了 {len(spy.calls)} 次，应当只 1 次"
    assert face_app.session_framing["S-multi"] == "center50"


def test_probe_failure_keeps_full_frame_and_does_not_break_capture(env, monkeypatch):
    """探测炸了 ⟹ 回退全画幅、端点照常出结果。取景是优化，不该让采集挂掉。"""
    monkeypatch.setattr(face_app, "probe_crop", _probe_spy("raise"))

    resp = _drive(data=_jpeg(200, 100))

    assert face_app.session_framing["S1"] == "full"
    h, w, _ = _RecordingPipeline.received[-1]
    assert (h, w) == (100, 200)
    assert resp is not None


def test_all_rungs_failing_keeps_full_frame(env, monkeypatch):
    """整条阶梯都检不到 ⟹ 记 `full`（不是 `none`），图原样进管线。"""
    monkeypatch.setattr(face_app, "probe_crop",
                        lambda *a, **k: ("none", None, {"probed": 1}))
    _drive(data=_jpeg(200, 100))

    assert face_app.session_framing["S1"] == "full"
    h, w, _ = _RecordingPipeline.received[-1]
    assert (h, w) == (100, 200)


# ──────────────────────────────────────────────────────────────
# 生命周期
# ──────────────────────────────────────────────────────────────

def test_reset_session_clears_framing_state(env, monkeypatch):
    """`/reset` 要清掉取景状态，否则同 id 的新会话会继承上一场的取景。

    生产改动会让它变红：删掉 `_reset_session` 里那句 `session_framing.pop(...)`。
    """
    monkeypatch.setattr(face_app, "probe_crop", _probe_spy("center50"))
    _drive(session_id="S-reset")
    assert face_app.session_framing.get("S-reset") == "center50"

    face_app._reset_session("S-reset")

    assert "S-reset" not in face_app.session_framing, "reset 后取景状态还在"


def test_decision_is_written_to_an_audit_manifest(env, monkeypatch):
    """取景决策要落盘可审计 —— CSV 不许加列，所以记旁挂清单。"""
    import json
    monkeypatch.setattr(face_app, "probe_crop", _probe_spy("center40"))
    _drive(session_id="S-manifest")

    path = Path(face_app.FACE_FRAMING_MANIFEST)
    assert path.is_file(), "清单没写出来"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["S-manifest"]["crop"] == "center40"
    assert data["S-manifest"]["source"] == "live"


def test_manifest_failure_does_not_break_capture(env, monkeypatch):
    """清单写不出去也不许影响采集（它只是审计）。"""
    monkeypatch.setattr(face_app, "probe_crop", _probe_spy("center50"))
    monkeypatch.setattr(face_app, "FACE_FRAMING_MANIFEST",
                        "/proc/definitely/not/writable/x.json")
    _drive(data=_jpeg(200, 100))          # 不抛异常即通过
    assert face_app.session_framing["S1"] == "center50"
