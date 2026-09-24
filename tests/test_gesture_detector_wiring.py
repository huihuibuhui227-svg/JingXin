# tests/test_gesture_detector_wiring.py
"""M1.5 T3:gesture 侧的接线。Review Focus 3/4 + D3 的全部意义。

注意本文件**能 import 真 app 模块且不需要 mediapipe** —— 这正是 D3 的副产品:
探测器从模块级单例改成按会话之后,import 期不再构造任何探测器。
如果哪天这条 import 又开始要 mediapipe,说明有人把探测器挪回模块级了。
"""

import importlib

import numpy as np
import pytest


@pytest.fixture
def gapp():
    return importlib.import_module("gesture_analysis.api.app")


@pytest.fixture
def fake_detectors(monkeypatch):
    """把真探测器换成假的。

    为什么必须有这个 fixture:`get_or_create_detectors` 会构造真的 HandDetector /
    PoseDetector —— 那要加载 21 MB 的 `.task` 模型(慢),而且会让**没下模型的环境整片红**。
    测试不该依赖那 21 MB 的存在。
    """
    import gesture_analysis.core.detectors as det

    class _Fake:
        def __init__(self, *a, **k):
            self.closed = False

        def detect(self, image_rgb):
            return []

        def reset(self):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setattr(det, "HandDetector", _Fake)
    monkeypatch.setattr(det, "PoseDetector", _Fake)
    return _Fake


def test_importing_the_app_does_not_touch_mediapipe(gapp):
    """D3 的副产品也是它的证明:import 期不许构造探测器。

    红法:把 `hands = HandLandmarker(...)` 挪回模块级 —— 在没装 mediapipe(或
    mediapipe 1.0 无 solutions)的环境里,这个 import 会直接炸,本测试先红。
    """
    assert not hasattr(gapp, "hands"), "模块级探测器回来了 —— 那正是并发污染的形态"
    assert not hasattr(gapp, "pose"), "模块级探测器回来了"


def test_two_sessions_get_different_detectors(gapp, fake_detectors):
    """spec D3 / Review Focus 4:两会话必须是不同探测器对象。

    红法:探测器做成模块级单例(迁移前的形态)—— VIDEO 模式会让两会话互相污染跟踪。
    """
    gapp.session_analyzers.clear()
    gapp.detectors.clear()
    a = gapp.get_or_create_detectors("t_a")
    b = gapp.get_or_create_detectors("t_b")

    assert a["hands"] is not b["hands"]
    assert a["pose"] is not b["pose"]


def test_expired_sessions_close_their_detectors(gapp, fake_detectors):
    """spec §6.3 / Review Focus 3:TTL 回收时必须 close 探测器,不然泄漏 native 句柄。

    红法:回收循环只 `del session_analyzers[sid]`(迁移前的样子),不碰探测器
    → `closed` 为空 → 断言失败。

    注意:`fake_detectors` 造出来的假件自带 `closed` 标志,所以这里**不用**再 monkeypatch
    `.close` —— 直接读标志更接近真实(也避免测到"我替换掉的那个方法")。
    """
    import time

    gapp.session_analyzers.clear()
    gapp.detectors.clear()
    dets = gapp.get_or_create_detectors("t_old")
    hands, pose = dets["hands"], dets["pose"]

    # 先给它建一条分析器记录,再把最后使用时间推到 TTL 之外(回收是挂在分析器表上的)
    gapp.get_or_create_analyzers("t_old")
    analyzers, _ = gapp.session_analyzers["t_old"]
    gapp.session_analyzers["t_old"] = (analyzers, time.time() - gapp.SESSION_TIMEOUT - 1)

    gapp.get_or_create_detectors("t_new")

    assert hands.closed and pose.closed, "过期会话的探测器没被 close —— native 句柄泄漏"
    assert "t_old" not in gapp.detectors, "过期会话没从探测器表里移除"


def test_analyzer_actually_consumes_what_the_detector_produces():
    """★ 消费侧钉子:封装交出的东西必须让分析器给出 `is_valid=True`。

    T1 那条契约测钉的是"封装不摊平";这条钉的是"**摊平之后真的会坏**"——
    两条合起来才能防住"有人觉得元组更干净就改回去"。

    红法:把封装的 `_groups` 改成摊平元组 → `is_valid` 变 False、分数回落 50.0
    → 断言失败。(实测:元组不抛异常,所以**没有这条断言就没有任何东西会红**。)
    """
    from gesture_analysis.core.analysis.hand_analyzer import HandAnalyzer

    class Pt:
        def __init__(self, x, y):
            self.x, self.y = x, y

    a = HandAnalyzer(hand_id=0)
    a.update([Pt(0.3 + 0.02 * i, 0.4 + 0.01 * i) for i in range(21)])
    r = a.get_results()

    assert r["is_valid"] is True, (
        "分析器没吃下探测器给的 landmarks —— 分数会静默回落成默认的 50.0")
    assert r["resilience_score"] != 50.0, "分数恰好是默认值,很可能就是静默回落"
