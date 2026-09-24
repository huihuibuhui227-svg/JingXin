"""手势/姿态探测器封装:mediapipe `tasks` 的 HandLandmarker / PoseLandmarker。

与 `face_expression/pipeline/detector.py` **同形但各持一份**,刻意不共享 —— 理由与
Ruling M1-2 同:三个包各自持有守卫副本是有意为之,跨包 import 会把整条依赖链拉起来并让
依赖方向反转。这里重复的代价(~15 行)小于耦合的代价(spec §4)。

**模块级不 import mediapipe** —— 同 face 那份的理由(测试要能在没装 mediapipe、
没下 21 MB 模型的机器上跑)。

⚠️ **本模块的 `detect()` 交出 landmark 对象本身,不是 `(x, y)` 元组** —— 与 face 那份
刻意相反。2026-09-24 实测:分析器收到元组**不抛异常**,只是 `is_valid=False`、分数回落
到默认的 `50.0`,于是报告里印着一个什么都不代表的数。见 `_groups` 的说明。
"""

from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

# close() 的后台执行器。**一个模块一份**(与 `_default_hand_factory` 同理:这两个封装
# 本来是刻意不共享的,合并会反转依赖方向)。max_workers=1 是刻意的:close() 是 native
# 释放,串行更安全,而且它本来就慢(5.0s),开并行只会同时占住更多句柄。
_CLOSER = concurrent.futures.ThreadPoolExecutor(max_workers=1,
                                                thread_name_prefix="detector-close")


def _log_close_failure(fut) -> None:
    """后台 close 失败了**必须留痕**。

    为什么不能不管:`concurrent.futures` 不会报告"没人取回的异常" —— 不取 `.exception()`
    的话,`detector.close()` 里抛的东西**完全静默**(无 traceback、无 warning、退出码 0)。
    而 `.close()` 只在成功后才把 `_landmarker` 置 None,于是失败时 **native 句柄静默泄漏**
    —— 正是 spec §6.3 加 close() 要防的那件事。改回同步 close 时失败是看得见的(端点会 500);
    挪到线程之后,这份可见性必须由这里补回来。
    """
    exc = fut.exception()
    if exc is not None:
        logger.error("探测器后台 close 失败,native 句柄可能未释放:%r", exc,
                     exc_info=exc)


def close_detached(detector) -> None:
    """把 close() 挪到后台线程。

    为什么:close() 实测恒 5.0s(构造只要 0.1–0.3s),而它是在**请求路径上同步**调的
    —— TTL 回收 2 个探测器 = +10s、无 id 的 /reset = +20s,期间整个事件循环被冻住
    (并发 /health 实测 19.73s,基线 0.0019s)。挪到线程后请求立刻返回,native 句柄
    仍会被释放(晚几秒)。**别"顺手"把它改回同步** —— 那会把这个停顿带回来。

    ⚠️ `add_done_callback` 不是装饰:去掉它,close 的失败就变成静默的(见上面那条)。
    """
    _CLOSER.submit(detector.close).add_done_callback(_log_close_failure)


def _default_hand_factory(model_path: Path, *, num_hands: int,
                          min_detection_confidence: float, min_tracking_confidence: float):
    """真的造一个 HandLandmarker。**mediapipe 只在这里被 import。**"""
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence))


def _default_pose_factory(model_path: Path, *, num_poses: int,
                          min_detection_confidence: float, min_tracking_confidence: float):
    """真的造一个 PoseLandmarker。**mediapipe 只在这里被 import。**"""
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    return vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence))


class _Base:
    """两个封装的共同部分(fps / 帧计数 / close)。子类给 `_default_factory` 与 `_result_attr`。"""

    _result_attr = ""
    _default_factory = None

    def __init__(self, model_path: Path, *, fps: int = 30, factory: Callable | None = None,
                 **build_kwargs):
        self.model_path = Path(model_path)
        self.fps = fps
        self._frame_index = 0
        self._landmarker = (factory or self._default_factory)(self.model_path, **build_kwargs)

    def _timestamp_ms(self) -> int:
        return int(self._frame_index * 1000 / self.fps)

    def _bump(self) -> None:
        self._frame_index += 1

    def _groups(self, result):
        """**原样交出 landmark 对象,不做任何转换。**

        ⚠️ 别"顺手"摊平成 `(x, y)` 元组。2026-09-24 实测:分析器收到元组**不抛异常**,
        只是 `is_valid=False`、分数回落到默认的 `50.0` —— 报告里于是印着一个数,而它
        什么都不代表。这是本项目一路在杀的那种静默失效,连日志都不会留。

        `tasks.NormalizedLandmark` 与 `solutions` 的点对象一样暴露 `.x/.y/.z/.visibility`,
        所以下游四个分析器 + 四个特征抽取器**一行都不用改**(spec §3.3)。
        """
        return getattr(result, self._result_attr) or []

    def detect(self, image_rgb: np.ndarray):
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(image_rgb))
        result = self._landmarker.detect_for_video(image, self._timestamp_ms())
        self._bump()
        return self._groups(result)

    def reset(self) -> None:
        """`/reset` 要调:帧计数归零,否则下一帧时间戳会回退(spec §6.4)。"""
        self._frame_index = 0

    def close(self) -> None:
        """释放 native 句柄。TTL 回收时必须调,否则泄漏(spec §6.3)。"""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


class HandDetector(_Base):
    """手部探测器。`detect()` 交出 landmark **对象**分组(0–2 只手,每只 21 个点)。

    为什么不是 `[(x, y)]`:分析器要 `.x/.y` 属性,元组会静默失效 —— 见 `_groups` 的说明。
    """

    _result_attr = "hand_landmarks"
    _default_factory = staticmethod(_default_hand_factory)

    def __init__(self, model_path: Path, *, fps: int = 30, num_hands: int = 2,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 factory: Callable | None = None):
        super().__init__(model_path, fps=fps, factory=factory, num_hands=num_hands,
                         min_detection_confidence=min_detection_confidence,
                         min_tracking_confidence=min_tracking_confidence)


class PoseDetector(_Base):
    """姿态探测器。`detect()` 交出 landmark **对象**列表(33 个点)或 `None`。"""

    _result_attr = "pose_landmarks"
    _default_factory = staticmethod(_default_pose_factory)

    def __init__(self, model_path: Path, *, fps: int = 30, num_poses: int = 1,
                 min_detection_confidence: float = 0.6,
                 min_tracking_confidence: float = 0.6,
                 factory: Callable | None = None):
        super().__init__(model_path, fps=fps, factory=factory, num_poses=num_poses,
                         min_detection_confidence=min_detection_confidence,
                         min_tracking_confidence=min_tracking_confidence)

    def detect(self, image_rgb: np.ndarray):
        groups = super().detect(image_rgb)
        return groups[0] if groups else None


def verify_models(hand_path: Path | None = None, pose_path: Path | None = None,
                  factory: Callable | None = None) -> None:
    """启动自检(spec §8)。两个模型都要能加载,否则抛。

    为什么必须是"启动即失败":gesture 现在的死法是进程在 import 期就没了,而
    迁移后如果改成"按会话懒建",故障会推迟到第一个请求 —— 那正是 face 那个哑死法的形状。
    """
    from ..config import HAND_MODEL, POSE_MODEL

    # 两边都过 `Path()`:本模块的 config 用 pathlib(已经是 Path),但显式转换让这条
    # 不受 config 风格影响 —— face 那边的 config 用 os.path,曾经因此炸过(见那边的注释)。
    hp = Path(hand_path) if hand_path is not None else Path(HAND_MODEL)
    pp = Path(pose_path) if pose_path is not None else Path(POSE_MODEL)
    for path in (hp, pp):
        if not path.exists():
            raise RuntimeError(
                f"手势/姿态模型文件不存在:{path}\n"
                f"  期望位置是 <repo>/models/mediapipe/,下载方式见 spec §6.1。")

    if factory is not None:
        factory(hp, num_hands=2, min_detection_confidence=0.7,
                min_tracking_confidence=0.5).close()
        factory(pp, num_poses=1, min_detection_confidence=0.6,
                min_tracking_confidence=0.6).close()
        return

    _default_hand_factory(hp, num_hands=2, min_detection_confidence=0.7,
                          min_tracking_confidence=0.5).close()
    _default_pose_factory(pp, num_poses=1, min_detection_confidence=0.6,
                          min_tracking_confidence=0.6).close()
