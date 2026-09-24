# tests/test_face_detector_wiring.py
"""M1.5 T2:face 侧的接线。Review Focus 2/3/4。

不 import mediapipe:探测器通过 `detector=` 注入假件。
"""

import numpy as np
import pytest

from face_expression.pipeline.video_pipeline import VideoPipeline


class _FakeDetector:
    def __init__(self):
        self.calls = 0
        self.resets = 0
        self.closed = False

    def detect(self, image_rgb):
        self.calls += 1
        return None            # "没检出" —— 走 process_frame 的 no_face 分支,不碰几何层

    def reset(self):
        self.resets += 1

    def close(self):
        self.closed = True


_FRAME = np.zeros((48, 48, 3), dtype=np.uint8)


def test_pipeline_uses_the_injected_detector():
    """接线的底线:process_frame 走注入的探测器,而不是自己造一个。

    红法:保留旧的惰性 `face_mesh` 属性(它会去 import mediapipe.solutions 而炸)。
    """
    d = _FakeDetector()
    p = VideoPipeline(fps=30, session_id="s", detector=d)
    p.process_frame(_FRAME)

    assert d.calls == 1


def test_reset_forwards_to_the_detector():
    """spec §6.4 / Review Focus 2:`/session/{sid}/reset` 必须把帧计数一起归零。

    红法:`VideoPipeline.reset()` 只清自己的历史、不转发给探测器 —— 之后同一会话
    再来一帧,时间戳回退,mediapipe 抛错。
    """
    d = _FakeDetector()
    p = VideoPipeline(fps=30, session_id="s", detector=d)
    p.process_frame(_FRAME)
    p.reset()

    assert d.resets == 1


def test_close_releases_the_detector():
    """spec §6.3 / Review Focus 3:TTL 回收时必须 close(否则泄漏 native 句柄)。

    红法:不给 VideoPipeline 提供 close(),或写了但不转发。
    """
    d = _FakeDetector()
    p = VideoPipeline(fps=30, session_id="s", detector=d)
    p.close()

    assert d.closed is True
