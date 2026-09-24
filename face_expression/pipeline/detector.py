"""人脸探测器封装:mediapipe `tasks` 的 FaceLandmarker(spec §4/§6.3/§6.4)。

**模块级不 import mediapipe** —— 延迟到工厂函数里。两个理由:
  1. `face_expression/api/app.py` 与 `gesture_analysis/api/app.py` 在 import 期会被测试
     加载,而测试不该因为机器上没装 mediapipe 就整片死掉(spec §10.2);
  2. 让"探测器怎么造"成为**可注入的一等参数** —— 契约测因此既不需要 mediapipe、
     也不需要那 21 MB 模型文件。

为什么按会话实例(而不是像 gesture 迁移前那样做模块级单例):`tasks` 的 VIDEO 模式把
跟踪状态挂在探测器实例上 —— 单例会让两个并发会话互相污染跟踪(spec §3.5 / D3)。
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Callable

import numpy as np

# close() 的后台执行器。**一个模块一份**(与 `_default_factory` 同理:这两个封装本来是
# 刻意不共享的,合并会反转依赖方向)。max_workers=1 是刻意的:close() 是 native 释放,
# 串行更安全,而且它本来就慢(5.0s),开并行只会同时占住更多句柄。
_CLOSER = concurrent.futures.ThreadPoolExecutor(max_workers=1,
                                                thread_name_prefix="detector-close")


def close_detached(detector) -> None:
    """把 close() 挪到后台线程。

    为什么:close() 实测恒 5.0s(构造只要 0.1–0.3s),而它是在**请求路径上同步**调的
    —— TTL 回收 2 个探测器 = +10s、无 id 的 /reset = +20s,期间整个事件循环被冻住
    (并发 /health 实测 19.73s,基线 0.0019s)。挪到线程后请求立刻返回,native 句柄
    仍会被释放(晚几秒)。**别"顺手"把它改回同步** —— 那会把这个停顿带回来。
    """
    _CLOSER.submit(detector.close)


def _default_factory(model_path: Path, *, num_faces: int,
                     min_detection_confidence: float, min_tracking_confidence: float):
    """真的造一个 FaceLandmarker。**mediapipe 只在这里被 import。**"""
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=num_faces,
        min_face_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence))


class FaceDetector:
    """一个会话一个人脸探测器。`detect()` 交出 `[(x, y)]`,没检出则 `None`。

    **交出元组是刻意的**:face 的下游 `au_calculator` 吃 `(x, y)`
    (`video_pipeline.py:52` 现在就摊平)。gesture 那两个封装**刻意相反** —— 见
    `gesture_analysis/core/detectors.py` 的说明,以及那条实测出来的静默失效。
    """

    def __init__(self, model_path: Path, *, fps: int = 30, num_faces: int = 1,
                 min_detection_confidence: float = 0.8,
                 min_tracking_confidence: float = 0.8,
                 factory: Callable | None = None):
        self.model_path = Path(model_path)
        self.fps = fps
        self._frame_index = 0
        self._factory = factory or _default_factory
        self._landmarker = self._factory(
            self.model_path, num_faces=num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)

    def detect(self, image_rgb: np.ndarray):
        """VIDEO 模式:时间戳由本会话的帧计数导出,天然单调(spec §6.4)。"""
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(image_rgb))
        result = self._landmarker.detect_for_video(
            image, int(self._frame_index * 1000 / self.fps))
        self._frame_index += 1
        if not result.face_landmarks:
            return None
        return [(p.x, p.y) for p in result.face_landmarks[0]]

    def reset(self) -> None:
        """`/session/{sid}/reset` 要调:帧计数归零,否则下一帧时间戳会回退。"""
        self._frame_index = 0

    def close(self) -> None:
        """释放 native 句柄。TTL 回收时必须调,否则泄漏(spec §6.3)。"""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


def verify_models(model_path: Path | None = None,
                  factory: Callable | None = None) -> None:
    """启动自检:模型文件在不在、能不能加载。**缺失即抛**(spec §8)。

    为什么必须是"启动即失败":face 现在的哑死法正是"启动看着正常、每帧 500"——
    故障被推迟到第一次请求,而那时客户端只看到一个 500,排障的人不知道去哪找。
    """
    from ..config import FACE_MODEL

    # ⚠️ 两边都要过 `Path()`:config 里的 `FACE_MODEL` 是 `os.path.join` 出来的 **str**
    # (本文件的 config 用 os.path 风格,gesture 那个用 pathlib —— 两个模块风格本来就不同)。
    # 只写 `else FACE_MODEL` 的话,不带参数调用会 `AttributeError: 'str' object has no
    # attribute 'exists'` —— 2026-09-24 实跑抓到的,契约测当时全绿(它们都显式传了 Path)。
    path = Path(model_path) if model_path is not None else Path(FACE_MODEL)
    if not path.exists():
        raise RuntimeError(
            f"人脸模型文件不存在:{path}\n"
            f"  期望位置是 <repo>/models/mediapipe/,下载方式见 spec §6.1。")

    build = factory or _default_factory
    landmarker = build(path, num_faces=1,
                       min_detection_confidence=0.8, min_tracking_confidence=0.8)
    landmarker.close()
