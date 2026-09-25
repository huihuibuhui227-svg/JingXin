# tests/test_detector_contract.py
"""M1.5 契约测:探测器封装。spec §6.3 / §6.4 / §8。

本文件**不 import mediapipe** —— 封装的全部行为都能用一个注入的假工厂验完。
这既是"模块级不 import mediapipe"这条设计的证明,也让这些测试跑得飞快、不依赖那 21 MB 模型。
"""

import logging
import time
from pathlib import Path

import numpy as np
import pytest

from face_expression.pipeline.detector import FaceDetector, close_detached, verify_models
from gesture_analysis.core.detectors import HandDetector, PoseDetector
from gesture_analysis.core.detectors import close_detached as gesture_close_detached
from gesture_analysis.core.detectors import verify_models as gesture_verify_models

_FRAME = np.zeros((48, 48, 3), dtype=np.uint8)


class _FakeLandmarker:
    """假的 tasks 探测器:记下收到的每个 timestamp_ms,返回可配置的结果。"""

    def __init__(self, points_per_group=0, groups=1):
        self.timestamps = []
        self.closed = False
        self._points = points_per_group
        self._groups = groups

    def _result(self):
        # 刻意造成**带 .x/.y 属性的对象**(而不是元组)—— tasks 的 NormalizedLandmark
        # 就是这个形状,gesture 的分析器依赖它(见 test_gesture_detectors_keep_x_and_y_attributes)
        pt = type("P", (), {"x": 0.5, "y": 0.5})
        return [pt() for _ in range(self._points)]

    def detect_for_video(self, image, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        groups = [self._result() for _ in range(self._groups)] if self._points else []
        return type("R", (), {
            "face_landmarks": groups, "hand_landmarks": groups,
            "pose_landmarks": groups})()

    def close(self):
        self.closed = True


def _factory(holder, **kwargs):
    """造一个假工厂:holder['l'] 拿到被造出来的假探测器。

    ⚠️ `build` 必须收 `*args` —— 真工厂的签名是 `(model_path, **opts)`,探测器封装
    会把 model_path **位置传参**。写成 `def build(**opts)` 的话,`FaceDetector.__init__`
    那一行会 `TypeError: build() takes 0 positional arguments but 1 was given`,
    测试变成 ERROR 而不是"按预期原因红"。
    """
    def build(*args, **opts):
        holder["args"] = args
        holder["opts"] = opts
        holder["l"] = _FakeLandmarker(**kwargs)
        return holder["l"]
    return build


def test_face_detector_flattens_to_xy_pairs():
    """face 的契约:`.detect()` 交出 `[(x, y)]` —— 它的下游 `au_calculator` 吃元组
    (`video_pipeline.py:52` 现在就摊平)。

    红法:让它返回点对象(`au_calculator` 的下标运算会 `TypeError`)。
    """
    h = {}
    d = FaceDetector(Path("/nonexistent.task"), factory=_factory(h, points_per_group=3))
    out = d.detect(_FRAME, 0)

    assert out == [(0.5, 0.5)] * 3, out


def test_gesture_detectors_keep_x_and_y_attributes():
    """★ gesture 的契约**与 face 相反**:交出 landmark 对象,不摊平。

    这是 2026-09-24 实测出来的静默失效,不是口味问题:分析器收到 `(x, y)` 元组
    **不抛异常**,只是 `is_valid=False`、分数回落到默认的 `50.0` —— 报告里于是印着
    一个数,而它什么都不代表。实测对照:

        HandAnalyzer <- 元组 (x,y)    : is_valid=False  resilience=50.0
        HandAnalyzer <- 带 .x/.y 对象 : is_valid=True   resilience=51.466…
        ArmAnalyzer  <- 元组          : is_valid=False  arm_score=50.0
        ArmAnalyzer  <- 对象          : is_valid=True   arm_score=90.0

    红法:把 `_Base._groups` 改成 `[[(p.x, p.y) for p in g] for g in groups]`。
    """
    h = {}
    d = HandDetector(Path("/nonexistent.task"), factory=_factory(h, points_per_group=21))
    groups = d.detect(_FRAME, 0)

    assert groups, "前提不成立:假探测器应当返回一组点"
    assert hasattr(groups[0][0], "x") and hasattr(groups[0][0], "y"), (
        "landmarks 被摊平成了元组 —— 分析器会静默回落到默认分"
        "(实测 is_valid=False, score=50.0,且不抛异常、不留日志)")

    h2 = {}
    p = PoseDetector(Path("/nonexistent.task"), factory=_factory(h2, points_per_group=33))
    pose = p.detect(_FRAME, 0)
    assert pose and hasattr(pose[0], "x"), "姿态的 landmarks 也被摊平了"


def test_detector_passes_the_caller_timestamp_straight_through():
    """M2.5 契约:时间戳由调用方给,探测器**原样**转交,不再自己算。

    红法:把 `detect()` 改回 `int(self._frame_index * 1000 / self.fps)` ——
    下面三个值立刻变成 [0, 100, 200],红。
    """
    h = {}
    d = FaceDetector(Path("/nonexistent.task"), factory=_factory(h, points_per_group=1))
    for ts in (0, 7, 1234):
        d.detect(_FRAME, ts)

    assert h["l"].timestamps == [0, 7, 1234], h["l"].timestamps


def test_detector_rejects_a_timestamp_that_goes_backwards():
    """★ 时间戳回退必须**抛**,不许静默转交 —— mediapipe 会拿着乱序时间戳继续算。

    红法:去掉 `if timestamp_ms < self._last_ts: raise`。
    """
    h = {}
    d = FaceDetector(Path("/nonexistent.task"), factory=_factory(h, points_per_group=1))
    d.detect(_FRAME, 100)

    with pytest.raises(ValueError) as ei:
        d.detect(_FRAME, 99)
    assert "100" in str(ei.value) and "99" in str(ei.value), (
        f"错误信息要同时带上一次和本次的值,否则排障时不知道谁回退了:{ei.value}")


def test_equal_timestamps_are_allowed_but_only_because_the_clock_prevents_them():
    """相等**不**抛(严格递增由 `SessionClock` 保证,见 Task 1),但也不许被改写成别的值。

    这条刻意把责任划清:探测器只管"不许回退",「严格递增」是时钟的契约。
    红法:在探测器里自作主张 `timestamp_ms = self._last_ts + 1` —— 转交的值就变了。
    """
    h = {}
    d = FaceDetector(Path("/nonexistent.task"), factory=_factory(h, points_per_group=1))
    d.detect(_FRAME, 50)
    d.detect(_FRAME, 50)

    assert h["l"].timestamps == [50, 50], h["l"].timestamps


def test_reset_allows_the_next_session_to_start_from_zero_again():
    """`/reset` 之后新一段可以从 0 起(不抛),但**不做任何改写**。

    红法:把 `reset()` 写成空函数 —— 第二段的 `d.detect(_FRAME, 0)` 会抛 ValueError。
    """
    h = {}
    d = FaceDetector(Path("/nonexistent.task"), factory=_factory(h, points_per_group=1))
    d.detect(_FRAME, 5000)
    d.reset()
    d.detect(_FRAME, 0)

    assert h["l"].timestamps == [5000, 0], h["l"].timestamps


def test_close_releases_the_underlying_landmarker():
    """spec §6.3:`close()` 必须真的转发到探测器(native 资源)。

    红法:把 `close()` 写成空函数 —— TTL 回收时泄漏 native 句柄。
    """
    h = {}
    d = FaceDetector(Path("/nonexistent.task"), factory=_factory(h, points_per_group=1))
    d.detect(_FRAME, 0)
    d.close()

    assert h["l"].closed is True


def test_two_sessions_get_different_detectors():
    """spec D3:按会话 —— 两个实例必须各自持有探测器。

    红法:把探测器做成模块级单例(这正是 gesture 迁移前的形态)。
    """
    h1, h2 = {}, {}
    a = HandDetector(Path("/nonexistent.task"), factory=_factory(h1, points_per_group=1))
    b = HandDetector(Path("/nonexistent.task"), factory=_factory(h2, points_per_group=1))
    a.detect(_FRAME, 0)
    b.detect(_FRAME, 0)

    assert h1["l"] is not h2["l"], "两个会话共用了同一个探测器实例"


def test_close_detached_surfaces_failures_instead_of_swallowing_them(caplog):
    """★ N1(复审抓到的、**修复 I1 时引入的回归**):后台 close 抛异常必须留痕。

    失效形态(修复 I1 之后实测):`concurrent.futures` 不会报告"没人取回的异常",所以
    `_CLOSER.submit(detector.close)` 里抛的东西**完全静默** —— 无 traceback、无 warning、
    退出码 0。而 `.close()` 只在成功后才把 `_landmarker` 置 None,于是失败时
    **native 句柄静默泄漏** —— 正是 spec §6.3 加 `close()` 要防的那件事。
    改之前(同步 close)失败是**看得见**的:端点会 500、`/reset` 会回 500。

    红法:去掉 `add_done_callback` 里的异常记录(或整个回调)。
    """
    class Boom:
        def close(self):
            raise RuntimeError("close 炸了")

    for name, fn in (("face", close_detached), ("gesture", gesture_close_detached)):
        with caplog.at_level(logging.ERROR):
            caplog.clear()
            fn(Boom())            # 不许把异常抛回调用方(那就白挪线程了)
            time.sleep(0.4)       # 等后台线程跑完

        assert caplog.records, f"{name}: 后台 close 的异常被静默吞掉了(无任何日志)"
        assert any("close" in r.getMessage().lower() or "炸了" in r.getMessage()
                   for r in caplog.records), \
            f"{name}: 有日志但没说是 close 失败:{[r.getMessage() for r in caplog.records]}"


def test_verify_models_fails_loudly_when_the_file_is_missing(tmp_path):
    """spec §8:模型缺失 → **启动即失败**,不是推迟到第一次请求。

    红法:把 `verify_models()` 写成空函数(或只打日志)—— 那就回到 face 现在
    "启动正常、每帧 500"的哑死法。
    """
    missing = tmp_path / "not_here.task"
    with pytest.raises(RuntimeError) as ei:
        verify_models(missing, factory=_factory({}, points_per_group=1))

    assert str(missing) in str(ei.value), "错误信息里必须含期望路径,否则排障时不知道去哪找"


def test_verify_models_accepts_a_present_file(tmp_path):
    """另一侧:`verify_models` 能通过时不许抛(否则服务永远起不来)。

    红法:让 `verify_models` 无条件抛。
    """
    real = tmp_path / "ok.task"
    real.write_bytes(b"stub")
    verify_models(real, factory=_factory({}, points_per_group=1))    # 不抛即通过


def test_verify_models_accepts_the_config_default_as_a_string(tmp_path, monkeypatch):
    """★ 回归:`face_expression.config.FACE_MODEL` 是 **str**(`os.path.join` 出来的),不是 Path。

    红在(2026-09-24 实跑实测,契约测当时全绿):`verify_models()` 不带参数时走
    `path = FACE_MODEL` → 直接 `path.exists()` →
    `AttributeError: 'str' object has no attribute 'exists'`。
    上一条测试看不见它,因为它显式传了 `Path` —— **只有"不带参数真的调一次"才暴露**。
    这正是"测试通过 ≠ 有约束力"的又一例:契约测把参数的形状定死了,而生产路径给的不是那个形状。

    红法:把 `path = Path(model_path) if ... else FACE_MODEL` 改回不转换。
    """
    real = tmp_path / "ok.task"
    real.write_bytes(b"stub")

    import face_expression.config as cfg
    monkeypatch.setattr(cfg, "FACE_MODEL", str(real))        # 刻意给 str,模拟真实 config 的形状

    verify_models(factory=_factory({}, points_per_group=1))  # 不抛即通过


@pytest.mark.parametrize("close_detached", [close_detached, gesture_close_detached],
                         ids=["face", "gesture"])
def test_close_detached_does_not_block_the_caller(close_detached):
    """★ I1:`close()` **实测恒 5.0 秒**(构造只要 0.08–0.34s),所以它不许在调用者线程上跑。

    为什么这条值得钉:两个 app 都在 `async def` 端点里**同步**调 close() —— TTL 回收一个
    gesture 会话 = 关 2 个探测器 = **10.12s**、无 id 的 `/reset`(2 会话)= **20.04s**,
    期间整个单 worker 事件循环被冻住(并发 `/health` 实测 **19.73s**,基线 0.0019s)。

    所以这条断言有两半,缺一不可:
      1. `close_detached(d)` **立刻返回**(否则请求路径照样被冻);
      2. 底层的 `close()` **最终真的被调到**(否则 native 句柄泄漏 —— 那就是 spec §6.3
         要防的东西,只是换了个方向)。

    红法:把实现改回 `d.close()`(直接同步调)—— 第 1 半立刻量到 ~1.0s,红。
    两个模块**刻意各持一份封装**,所以两个 helper 都要压住。
    """
    import time

    class _SlowDetector:
        def __init__(self):
            self.closed = False

        def close(self):
            time.sleep(1.0)        # 真 close() 的 5.0s 的缩影,但测试跑得起
            self.closed = True

    d = _SlowDetector()
    t0 = time.monotonic()
    close_detached(d)
    elapsed = time.monotonic() - t0

    assert elapsed < 0.2, (
        f"close_detached 在调用者线程上等了 {elapsed:.2f}s —— 请求路径会被冻住"
        f"(真 close() 是 5.0s,TTL 回收/`/reset` 会把它放大到 10–20s)")

    # 后半:后台线程里那个 close() 必须真的跑完,否则句柄泄漏(只是晚几秒)
    deadline = time.monotonic() + 5.0
    while not d.closed and time.monotonic() < deadline:
        time.sleep(0.02)
    assert d.closed is True, "close() 最终没被调到 —— native 句柄不会被释放"


def test_gesture_verify_models_fails_loudly_when_a_file_is_missing(tmp_path):
    """★ I3(a):gesture 的 `verify_models` 也要**启动即失败**(spec §8 / Review Focus ⑤)。

    为什么单列:face 那份有测,gesture 这份**一条都没有** —— 而两者的签名不同
    (`(model_path, factory)` vs `(hand_path, pose_path, factory)`),正因签名不同才更容易漏。
    两半都验:hand 缺、pose 缺(缺哪个就在消息里点名哪个,否则排障时不知道去哪找)。

    红法:让 gesture 的 verify_models 不检查 `.exists()`(把故障推迟到第一个请求)。
    """
    ok = tmp_path / "ok.task"
    ok.write_bytes(b"stub")
    missing = tmp_path / "not_here.task"

    with pytest.raises(RuntimeError) as ei_hand:
        gesture_verify_models(hand_path=missing, pose_path=ok,
                              factory=_factory({}, points_per_group=1))
    assert str(missing) in str(ei_hand.value), "错误信息里必须含缺的那个路径"

    with pytest.raises(RuntimeError) as ei_pose:
        gesture_verify_models(hand_path=ok, pose_path=missing,
                              factory=_factory({}, points_per_group=1))
    assert str(missing) in str(ei_pose.value), "错误信息里必须含缺的那个路径"


def test_gesture_verify_models_accepts_present_files(tmp_path):
    """另一侧:两个文件都在时不许抛(否则服务永远起不来)。

    红法:让 gesture 的 verify_models 无条件抛。
    """
    hand = tmp_path / "hand.task"
    pose = tmp_path / "pose.task"
    hand.write_bytes(b"stub")
    pose.write_bytes(b"stub")

    gesture_verify_models(hand_path=hand, pose_path=pose,
                          factory=_factory({}, points_per_group=1))    # 不抛即通过
