# tests/test_gesture_session_clock.py
"""M2.5:gesture 服务的时钟接线。gesture 连 `?fps=` 参数都没有(spec §3.5),所以这里只验时钟。

⚠️ 「时间戳真的传进了探测器」这一条**刻意不写在这里** —— 它由既有的
`tests/test_analyze_session_fallback.py` 用假探测器真跑端点来压(见 Task 6 Step 4),
比字符串比对源码结实得多:那种断言在换行/改变量名时就会碎,而它想守的东西其实一点没变
(账本 §4.1「测试通过 ≠ 有约束力」的同一类毛病)。
"""

import importlib

# 同 face:本包 `__init__.py` 也做了 `from .app import app`,所以直接 import 拿到的是
# FastAPI 实例而不是模块(见 Task 5 的账本行)。
gesture_app = importlib.import_module("gesture_analysis.api.app")


def test_each_session_gets_its_own_clock():
    """按会话,不是单例。红法:改成模块级一个 `SessionClock()`。"""
    gesture_app.session_clocks.clear()
    a = gesture_app._clock_for("20260925_120000_aaaa")
    b = gesture_app._clock_for("20260925_120000_bbbb")

    assert a is not b
    assert gesture_app._clock_for("20260925_120000_aaaa") is a, "同一会话应复用同一个时钟"


def test_reset_drops_the_clock_so_the_next_session_starts_at_zero():
    """`/reset` 与 face 同语义:删掉会话,所以时钟一起丢,下一段从 0 起。

    红法:`_reset_session()` 里不 pop `session_clocks` —— 第二段会话接着第一段涨。
    """
    gesture_app.session_clocks.clear()
    gesture_app.session_analyzers.clear()
    sid = "20260925_120000_cccc"

    clock = gesture_app._clock_for(sid)
    clock.stamp_ms()
    clock.stamp_ms()
    assert clock.stamp_ms() > 0, "前提不成立:时钟应当已经走过一段"

    gesture_app._reset_session(sid)

    fresh = gesture_app._clock_for(sid)
    assert fresh is not clock, "reset 之后仍是同一个时钟对象 —— 旧状态跨会话活下来了"
    assert fresh.stamp_ms() == 0, "新会话的第一个时间戳不是 0"
