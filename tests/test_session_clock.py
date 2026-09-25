# tests/test_session_clock.py
"""M2.5:会话时钟。spec §4 / §5.1 / Review Focus ①。

本文件不 import mediapipe、不起服务 —— 时钟的全部行为都能用一个假 `now` 验完。
"""

from session_clock import SessionClock


def test_stamp_is_relative_to_construction_in_milliseconds():
    """会话开始 = 0 ms,之后按真实流逝走。

    红法:把 `stamp_ms()` 写成 `int(time.time() * 1000)`(绝对墙钟)——
    第一条断言立刻红(那是 1.7e12 量级,不是 0)。
    """
    t = [100.0]
    c = SessionClock(now=lambda: t[0])
    assert c.stamp_ms() == 0           # 建好即 0

    t[0] = 100.5
    assert c.stamp_ms() == 500         # 500 ms 后

    t[0] = 101.25
    assert c.stamp_ms() == 1250


def test_stamp_is_strictly_increasing_even_within_one_millisecond():
    """★ Review Focus ①:`now()` 不动(同一毫秒内两次调用)时也必须**严格**递增。

    红法:直接 `return int((self._now() - self._start) * 1000)` ——
    第二帧与第一帧同值,mediapipe 的 VIDEO 模式要求时间戳单调递增,旧契约测
    `ts == sorted(set(ts))` 断言的也正是「严格」(不是「非递减」)。
    """
    t = [100.0]
    c = SessionClock(now=lambda: t[0])
    a = c.stamp_ms()
    b = c.stamp_ms()                   # now() 没动
    c_ms = c.stamp_ms()

    assert a == 0, a
    assert b == 1 and c_ms == 2, (a, b, c_ms)
    assert a < b < c_ms, "必须严格递增,否则 mediapipe 会抛"


def test_reset_rewinds_to_zero():
    """`/reset` 之后新一段从 0 起(Review Focus ② 的一半:时钟这一半)。

    红法:把 `reset()` 写成空函数 —— 第二段的时间戳接着第一段涨,
    `/reset` 那条路径(管线归零)就只归零了一半。
    """
    t = [100.0]
    c = SessionClock(now=lambda: t[0])
    t[0] = 105.0
    assert c.stamp_ms() == 5000

    c.reset()
    assert c.stamp_ms() == 0

    t[0] = 106.0
    assert c.stamp_ms() == 1000
