# tests/test_face_session_clock.py
"""M2.5:face 服务的时钟接线。spec §5.4 / Review Focus ②。"""

import importlib
import inspect

# 必须用 `import_module`:本包的 `__init__.py` 做了 `from .app import app`,
# 于是 `from face_expression.api import app` 拿到的是那个 **FastAPI 实例**,不是模块。
# 这条坑仓里已有先例,见 `tests/test_analyze_session_fallback.py:69` 的同一段说明。
face_app = importlib.import_module("face_expression.api.app")


def test_query_fps_is_no_longer_accepted_by_the_pipeline_factory():
    """★ spec §5.4:`?fps=` 不再进管线 —— 它是错的 30 倍的那个数(spec §3.1)。

    红法:把 `fps` 参数加回 `get_or_create_pipeline`。
    """
    params = inspect.signature(face_app.get_or_create_pipeline).parameters
    assert "fps" not in params, (
        f"get_or_create_pipeline 又收 fps 了 —— 那会重新变成计时依据:{list(params)}")


def test_fps_warning_fires_once_per_session_and_only_for_a_real_discrepancy(caplog):
    """★ 审查 F5:那条 `?fps=` 警告**永远不会触发**对于真实客户端 —— 前端恒定发 `?fps=30`
    (`~/JingXin-frontend/src/services/api.ts:53`),而判据是 `fps != 30`;
    同时它写在每帧都走的端点里,且没有"只记一次"的标志 —— 注释却写着"记一次就够"。

    于是 spec §5.4 要的那点遥测**是死代码**,而注释在说谎。

    修法:抽出一个每会话只记一次的 helper,并让它对"申报值确实是 30"保持沉默
    (那没有不一致可言)。这条钉住三件:同会话第二次不再记、不同会话会记、
    申报 30 不记。

    红法:去掉 `session_id in _fps_warned` 那道判据(退回每帧刷屏)。
    """
    import logging

    face_app._fps_warned.clear()
    with caplog.at_level(logging.WARNING):
        caplog.clear()
        assert face_app._warn_once_about_fps("s1", 15) is True
        assert face_app._warn_once_about_fps("s1", 15) is False, "同一会话记了第二次"
        assert face_app._warn_once_about_fps("s2", 15) is True, "不同会话应当各记一次"
        assert face_app._warn_once_about_fps("s3", 30) is False, "申报 30 没有不一致,不该记"

        msgs = [r.getMessage() for r in caplog.records]
        assert sum("fps=15" in m for m in msgs) == 2, msgs


def test_each_session_gets_its_own_clock():
    """★ Review Focus ②:时钟必须**按会话**,不能是模块级单例。

    否则两个会话共享一个起点,各自的 `timestamp_ms` 会互相错位。
    红法:把 `session_clocks` 改成模块级一个 `SessionClock()`。
    """
    face_app.session_clocks.clear()
    a = face_app._clock_for("20260925_120000_aaaa")
    b = face_app._clock_for("20260925_120000_bbbb")

    assert a is not b
    assert face_app._clock_for("20260925_120000_aaaa") is a, "同一会话应复用同一个时钟"


def test_reset_drops_the_clock_so_the_next_session_starts_at_zero():
    """★ Review Focus ②(核心):`/reset` 的语义是**删掉整个会话**,所以时钟也要一起丢 ——
    下一次请求重建出来的是全新时钟,时间戳从 0 起。

    为什么**不是**"把老时钟 reset() 一下":本端点根本不保留会话(见它的 docstring),
    管线和探测器都是 pop 掉重建的。留着一个老时钟对象反而与"全新会话"的语义不一致 ——
    它会成为唯一一个跨会话存活的状态。

    红法:`_reset_session()` 里不 pop `session_clocks` —— 第二段会话会接着第一段涨。
    """
    face_app.session_clocks.clear()
    face_app.session_pipelines.clear()
    sid = "20260925_120000_cccc"

    clock = face_app._clock_for(sid)
    clock.stamp_ms()
    clock.stamp_ms()
    assert clock.stamp_ms() > 0, "前提不成立:时钟应当已经走过一段"

    face_app._reset_session(sid)

    fresh = face_app._clock_for(sid)
    assert fresh is not clock, "reset 之后仍是同一个时钟对象 —— 旧状态跨会话活下来了"
    assert fresh.stamp_ms() == 0, "新会话的第一个时间戳不是 0"
