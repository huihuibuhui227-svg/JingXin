# tests/test_micro_expression_window.py
"""M2.5:微表情窗按**时间**而不是帧数(spec §5.3 —— 本设计对 spec 措辞的唯一偏离)。"""

from types import SimpleNamespace

from face_expression.core.analysis.micro_expression import MicroExpressionDetector


def _au(**kw):
    """`detect` 只从对象上按名字取那几个 AU,`SimpleNamespace` 就够。"""
    base = dict(au4_frown=0.0, au7_eye_squeeze=0.0, au15_mouth_down=0.0)
    base.update(kw)
    return SimpleNamespace(**base)


def test_window_is_pruned_by_time_not_by_frame_count():
    """★ 窗长是 1.5 **秒**,不是 15 帧。

    造法:先喂 20 帧密集帧(每 10 ms),再跳到 5 秒后喂 1 帧 ——
    如果窗是按帧数(15)而不是按时间,那 5 秒前那些帧还会留在窗里。

    红法:把 `maxlen=15` 的旧实现拿回来 —— 断言 `len(history)` 会 >= 15 而不是 1。
    """
    d = MicroExpressionDetector()
    for i in range(20):
        d.detect(_au(au4_frown=1.0), timestamp_ms=i * 10)
    assert len(d._window('au4_frown')) > 1, "前提不成立:密集帧应当留在窗里"

    d.detect(_au(au4_frown=1.0), timestamp_ms=5000)
    assert len(d._window('au4_frown')) == 1, (
        f"5 秒前的帧还在窗里 —— 窗是按帧数而不是按时间的:{len(d._window('au4_frown'))}")


def test_same_frame_count_different_spacing_gives_different_windows():
    """同样的**帧数**、不同的**间隔**,窗里剩的帧数必须不同。

    这是「帧数窗」和「时间窗」的分水岭:旧实现在两种情形下窗里都是 15 帧。

    红法:实现里不读 `timestamp_ms`。
    """
    dense = MicroExpressionDetector()
    for i in range(20):
        dense.detect(_au(au4_frown=1.0), timestamp_ms=i * 10)      # 200 ms 内 20 帧
    assert len(dense._window('au4_frown')) == 20

    sparse = MicroExpressionDetector()
    for i in range(20):
        sparse.detect(_au(au4_frown=1.0), timestamp_ms=i * 1000)   # 20 秒内 20 帧
    assert len(sparse._window('au4_frown')) == 2, (
        f"1.5 秒窗里应当只剩最后一帧和它 1 秒前那帧:{len(sparse._window('au4_frown'))}")
