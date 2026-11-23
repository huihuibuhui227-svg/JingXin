"""可选 RTMW / RTMPose 全身与手部关键点后端（COCO-WholeBody 133 点 -> MediaPipe 映射）。

本模块提供基于 ``rtmlib.Wholebody`` 的纯 CPU ONNXRuntime 推理封装与适配器，
可在不改动上游分析器（HandAnalyzer / ShoulderAnalyzer / ArmAnalyzer /
UpperBodyAnalyzer / angles.py）的前提下，作为 MediaPipe 的可选后端。

关键设计约定：
1. **模块级不 import rtmlib**：与 ``detectors.py`` 同理，测试与无 ONNX 权重环境
   必须能无痛导入本模块；``rtmlib`` 仅在首次推理或显式 ``ensure_ready()`` 时加载。
2. **单帧共享缓存**：``RTMWPoseAdapter`` 与 ``RTMWHandAdapter`` 共享同一个
   ``RTMWWholeBodyEngine``。同一帧先调姿态、再调手部时只执行一次 RTMW 推理。
3. **左右手标签约定（重要）**：
   COCO-WholeBody 的 ``91..111`` 为受试者解剖学左手（``"Left"``）、``112..132``
   为受试者解剖学右手（``"Right"``）。然而下游 ``gesture_analysis/api/app.py``
   的 ``process_frame()`` 按 MediaPipe 镜像自拍假定对 ``detect_with_handedness()``
   的返回标签做了翻转（``side = 'right' if label == 'Left' else 'left'``）。
   因此 ``mediapipe_handedness=True``（默认）时：
     - COCO 左手（91..111）输出 ``"Right"`` → 经 ``app.py`` 翻转后落入 ``left_hand`` 槽；
     - COCO 右手（112..132）输出 ``"Left"`` → 经 ``app.py`` 翻转后落入 ``right_hand`` 槽。
   若直接调用 ``coco_wholebody133_to_mediapipe(..., mediapipe_handedness=False)``，
   则交出原始解剖学标签 ``"Left"`` / ``"Right"``。
"""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, NamedTuple, Optional, Tuple
from urllib.parse import urlparse

import numpy as np


@dataclass(slots=True)
class SimpleLandmark:
    """轻量级关键点对象，暴露与 MediaPipe NormalizedLandmark 一致的属性。"""

    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0


class WholeBodyMappingResult(NamedTuple):
    """COCO-WholeBody 133 点映射后的单人姿态与手部结果。"""

    pose_landmarks: Optional[List[SimpleLandmark]]
    hand_groups: List[List[SimpleLandmark]]
    handedness: List[Tuple[str, float]]


# MediaPipe Pose 33 点索引 <- COCO-WholeBody 身体 17 点索引（仅映射上肢与躯干核心点）
_COCO_TO_MP_POSE: Tuple[Tuple[int, int], ...] = (
    (0, 0),    # Nose
    (2, 1),    # Left Eye
    (5, 2),    # Right Eye
    (7, 3),    # Left Ear
    (8, 4),    # Right Ear
    (11, 5),   # Left Shoulder
    (12, 6),   # Right Shoulder
    (13, 7),   # Left Elbow
    (14, 8),   # Right Elbow
    (15, 9),   # Left Wrist
    (16, 10),  # Right Wrist
    (23, 11),  # Left Hip
    (24, 12),  # Right Hip
)

_NUM_MP_POSE = 33
_NUM_MP_HAND = 21
_LEFT_HAND_START = 91
_LEFT_HAND_END = 112
_RIGHT_HAND_START = 112
_RIGHT_HAND_END = 133

_VALID_RTMW_MODES = ("lightweight", "balanced", "performance")


def _rgb_to_bgr_contiguous(image_rgb: np.ndarray) -> np.ndarray:
    """将 RGB 图像转换为 rtmlib/OpenCV 所需的连续 BGR numpy 数组。"""
    arr = np.asarray(image_rgb)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[:, :, :3]
    else:
        raise ValueError(f"期望 2D/3D 图像数组，实际 shape={arr.shape}")
    return np.ascontiguousarray(arr[:, :, ::-1])


def _frame_digest(image_rgb: np.ndarray) -> bytes:
    """计算图像字节的轻量内容摘要，避免复用同一缓冲区时命中陈旧缓存。"""
    contig = np.ascontiguousarray(image_rgb)
    return hashlib.blake2b(memoryview(contig.view(np.uint8)), digest_size=16).digest()


def coco_wholebody133_to_mediapipe(
    kpts_133: Any,
    scores_133: Any,
    img_w: int,
    img_h: int,
    min_conf: float = 0.3,
    *,
    mediapipe_handedness: bool = True,
) -> WholeBodyMappingResult:
    """将单人 COCO-WholeBody 133 关键点映射到 MediaPipe Pose(33) 与 Hand(21)。

    参数:
        kpts_133: 形状为 ``(133, 2)``（或 ``(1, 133, 2)``）的像素坐标数组。
        scores_133: 形状为 ``(133,)``（或 ``(1, 133)``）的置信度数组；为 ``None`` 时按 1.0 处理。
        img_w: 原图宽度（像素，必须 > 0）。
        img_h: 原图高度（像素，必须 > 0）。
        min_conf: 关键点/手部保留门限（默认 0.3）。
        mediapipe_handedness: 是否按 MediaPipe 非镜像输入的镜像假定转换左右手标签
            （默认 ``True``，使 ``gesture_analysis/api/app.py`` 翻转后准确落入对应左右手槽）。

    返回:
        ``WholeBodyMappingResult(pose_landmarks, hand_groups, handedness)``。
    """
    w = float(img_w)
    h = float(img_h)
    if w <= 0 or h <= 0:
        raise ValueError(f"图像宽高必须大于 0，实际: img_w={img_w}, img_h={img_h}")

    kpts_arr = np.asarray(kpts_133, dtype=float)
    if kpts_arr.ndim == 3:
        if kpts_arr.shape[0] == 0:
            return WholeBodyMappingResult(None, [], [])
        kpts_arr = kpts_arr[0]

    if kpts_arr.ndim != 2 or kpts_arr.shape[0] < 133 or kpts_arr.shape[1] < 2:
        return WholeBodyMappingResult(None, [], [])

    if scores_133 is None:
        scores_arr = np.ones((kpts_arr.shape[0],), dtype=float)
    else:
        scores_arr = np.asarray(scores_133, dtype=float)
        if scores_arr.ndim == 2:
            if scores_arr.shape[0] == 0:
                return WholeBodyMappingResult(None, [], [])
            scores_arr = scores_arr[0]
        if scores_arr.ndim != 1 or scores_arr.shape[0] < 133:
            return WholeBodyMappingResult(None, [], [])

    # 1. 映射 Pose 33 点（未映射槽位保持 0.0 可见度，供 angles.py VISIBILITY_FLOOR 过滤）
    pose_landmarks: Optional[List[SimpleLandmark]] = [
        SimpleLandmark(0.0, 0.0, 0.0, 0.0) for _ in range(_NUM_MP_POSE)
    ]
    valid_body_points = 0
    for mp_idx, coco_idx in _COCO_TO_MP_POSE:
        px = float(kpts_arr[coco_idx, 0])
        py = float(kpts_arr[coco_idx, 1])
        sc = float(scores_arr[coco_idx])
        if not (np.isfinite(px) and np.isfinite(py) and np.isfinite(sc)):
            continue
        vis = max(0.0, sc)
        pose_landmarks[mp_idx] = SimpleLandmark(
            x=px / w,
            y=py / h,
            z=0.0,
            visibility=vis,
        )
        if vis >= min_conf:
            valid_body_points += 1

    if valid_body_points == 0:
        pose_landmarks = None

    # 2. 映射左右手各 21 点（91..111 为解剖左手，112..132 为解剖右手）
    hand_groups: List[List[SimpleLandmark]] = []
    handedness: List[Tuple[str, float]] = []

    hand_slices = (
        ("Left", _LEFT_HAND_START, _LEFT_HAND_END),
        ("Right", _RIGHT_HAND_START, _RIGHT_HAND_END),
    )
    for anatomical_side, start, end in hand_slices:
        hand_xy = kpts_arr[start:end, :2]
        hand_sc = scores_arr[start:end]
        if hand_xy.shape[0] != _NUM_MP_HAND or hand_sc.shape[0] != _NUM_MP_HAND:
            continue

        wrist_finite = bool(np.isfinite(hand_xy[0, 0]) and np.isfinite(hand_xy[0, 1]))
        all_finite = bool(np.all(np.isfinite(hand_xy)) and np.all(np.isfinite(hand_sc)))
        if not (wrist_finite and all_finite):
            continue

        wrist_score = max(0.0, float(hand_sc[0]))
        clipped_scores = np.maximum(0.0, hand_sc)
        mean_score = float(np.mean(clipped_scores))
        if wrist_score < min_conf or mean_score < min_conf:
            continue

        landmarks_21 = [
            SimpleLandmark(
                x=float(hand_xy[i, 0]) / w,
                y=float(hand_xy[i, 1]) / h,
                z=0.0,
                visibility=float(clipped_scores[i]),
            )
            for i in range(_NUM_MP_HAND)
        ]
        hand_groups.append(landmarks_21)

        if mediapipe_handedness:
            # 匹配 MediaPipe 对非镜像帧的标签约定，避免 api/app.py 二次错翻
            out_label = "Right" if anatomical_side == "Left" else "Left"
        else:
            out_label = anatomical_side
        conf = float(min(1.0, max(0.0, mean_score)))
        handedness.append((out_label, conf))

    return WholeBodyMappingResult(
        pose_landmarks=pose_landmarks,
        hand_groups=hand_groups,
        handedness=handedness,
    )


def has_cached_rtmw_weights(mode: str | None = None) -> bool:
    """检查 rtmlib 默认缓存目录下是否已存在对应 mode 的 ONNX 权重。"""
    resolved_mode = (mode or os.getenv("JX_RTMW_MODE") or "lightweight").strip().lower()
    if resolved_mode not in _VALID_RTMW_MODES:
        return False
    try:
        from rtmlib.tools.file import _get_rtmhub_dir
        from rtmlib.tools.solution.wholebody import Wholebody
    except Exception:
        return False

    cfg = Wholebody.MODE.get(resolved_mode)
    if not cfg:
        return False
    ckpt_dir = Path(_get_rtmhub_dir()) / "checkpoints"
    for key in ("det", "pose"):
        url = cfg[key]
        filename = os.path.basename(urlparse(url).path)
        stem = filename.split(".")[0]
        onnx_path = ckpt_dir / f"{stem}.onnx"
        if not onnx_path.is_file():
            return False
    return True


class RTMWWholeBodyEngine:
    """共享单帧推理缓存的 RTMW Wholebody 引擎（纯 CPU ONNXRuntime）。"""

    def __init__(
        self,
        mode: Optional[str] = None,
        *,
        backend: str = "onnxruntime",
        device: str = "cpu",
        min_conf: float = 0.3,
        mediapipe_handedness: bool = True,
        wholebody_factory: Optional[Callable[..., Any]] = None,
    ):
        resolved_mode = (mode or os.getenv("JX_RTMW_MODE") or "lightweight").strip().lower()
        if resolved_mode not in _VALID_RTMW_MODES:
            raise ValueError(
                f"不支持的 RTMW mode={resolved_mode!r}，可选值: {_VALID_RTMW_MODES}"
            )
        self.mode = resolved_mode
        self.backend = backend
        self.device = device
        self.min_conf = float(min_conf)
        self.mediapipe_handedness = bool(mediapipe_handedness)
        self._wholebody_factory = wholebody_factory

        self._lock = threading.RLock()
        self._model: Any = None
        self._cached_key: Optional[Tuple[Optional[int], Tuple[int, ...], bytes]] = None
        self._cached_result: Optional[WholeBodyMappingResult] = None
        self._ref_count = 0
        self.inference_count = 0

    def _ensure_loaded(self) -> Any:
        if self._model is None:
            if self._wholebody_factory is not None:
                self._model = self._wholebody_factory(
                    mode=self.mode, backend=self.backend, device=self.device
                )
            else:
                from rtmlib import Wholebody

                self._model = Wholebody(
                    mode=self.mode, backend=self.backend, device=self.device
                )
        return self._model

    def ensure_ready(self) -> None:
        """显式触发模型加载，用于启动检查与不可用时的快速回退。"""
        with self._lock:
            self._ensure_loaded()

    def _retain(self) -> None:
        with self._lock:
            self._ref_count += 1

    def _release(self) -> None:
        with self._lock:
            if self._ref_count > 0:
                self._ref_count -= 1
            if self._ref_count == 0:
                self.close()

    def infer(
        self, image_rgb: np.ndarray, timestamp_ms: Optional[int] = None
    ) -> WholeBodyMappingResult:
        """执行单帧推理（命中同一 ``(timestamp_ms, shape, digest)`` 缓存时直接复用）。"""
        arr = np.asarray(image_rgb)
        if arr.ndim < 2 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
            raise ValueError(f"无效图像输入 shape={arr.shape}")

        ts_key = None if timestamp_ms is None else int(timestamp_ms)
        cache_key = (ts_key, tuple(arr.shape), _frame_digest(arr))

        with self._lock:
            if self._cached_key == cache_key and self._cached_result is not None:
                return self._cached_result

            model = self._ensure_loaded()
            image_bgr = _rgb_to_bgr_contiguous(arr)
            keypoints, scores = model(image_bgr)
            self.inference_count += 1

            kpts_arr = np.asarray(keypoints, dtype=float)
            scores_arr = np.asarray(scores, dtype=float)
            if kpts_arr.ndim == 2:
                kpts_arr = kpts_arr[None, ...]
            if scores_arr.ndim == 1:
                scores_arr = scores_arr[None, ...]

            if (
                kpts_arr.size == 0
                or kpts_arr.ndim != 3
                or kpts_arr.shape[0] == 0
                or kpts_arr.shape[1] < 133
            ):
                mapped = WholeBodyMappingResult(None, [], [])
            else:
                if kpts_arr.shape[0] > 1 and scores_arr.ndim == 2 and scores_arr.shape[0] == kpts_arr.shape[0]:
                    # 多人画面下优先取上肢身体关键点平均置信度最高的主体
                    best_idx = int(np.argmax(np.mean(scores_arr[:, :17], axis=1)))
                else:
                    best_idx = 0
                person_scores = (
                    scores_arr[best_idx]
                    if scores_arr.ndim == 2 and scores_arr.shape[0] > best_idx
                    else None
                )
                img_h, img_w = arr.shape[0], arr.shape[1]
                mapped = coco_wholebody133_to_mediapipe(
                    kpts_arr[best_idx],
                    person_scores,
                    img_w=img_w,
                    img_h=img_h,
                    min_conf=self.min_conf,
                    mediapipe_handedness=self.mediapipe_handedness,
                )

            self._cached_key = cache_key
            self._cached_result = mapped
            return mapped

    def reset(self) -> None:
        """清空单帧推理缓存。"""
        with self._lock:
            self._cached_key = None
            self._cached_result = None

    def close(self) -> None:
        """释放底层模型引用与缓存。"""
        with self._lock:
            self._cached_key = None
            self._cached_result = None
            model = self._model
            self._model = None
            if model is not None and hasattr(model, "close"):
                model.close()


class _RTMWAdapterBase:
    """Hand / Pose 适配器公共逻辑（时间戳单调守卫 / reset / 引用计数 close）。"""

    def __init__(self, engine: RTMWWholeBodyEngine):
        self.engine = engine
        self._frame_index = 0
        self._last_ts: Optional[int] = None
        self._closed = False
        self.engine._retain()

    def _check_timestamp(self, timestamp_ms: int) -> int:
        ts = int(timestamp_ms)
        if self._last_ts is not None and ts < self._last_ts:
            raise ValueError(
                f"timestamp_ms 回退:上一次 {self._last_ts},本次 {ts} —— "
                f"探测器接口要求时间戳非递减"
            )
        self._last_ts = ts
        self._frame_index += 1
        return ts

    def reset(self) -> None:
        self._frame_index = 0
        self._last_ts = None
        self.engine.reset()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.engine._release()


class RTMWHandAdapter(_RTMWAdapterBase):
    """兼容 ``HandDetector`` 接口的 RTMW 手部适配器。"""

    def detect(self, image_rgb: np.ndarray, timestamp_ms: int) -> List[List[SimpleLandmark]]:
        ts = self._check_timestamp(timestamp_ms)
        return self.engine.infer(image_rgb, ts).hand_groups

    def detect_with_handedness(
        self, image_rgb: np.ndarray, timestamp_ms: int
    ) -> Tuple[List[List[SimpleLandmark]], List[Optional[Tuple[str, float]]]]:
        """返回 ``(hand_groups, per_hand)``，约定与 ``HandDetector.detect_with_handedness`` 一致。"""
        ts = self._check_timestamp(timestamp_ms)
        res = self.engine.infer(image_rgb, ts)
        per_hand: List[Optional[Tuple[str, float]]] = list(res.handedness)
        return res.hand_groups, per_hand


class RTMWPoseAdapter(_RTMWAdapterBase):
    """兼容 ``PoseDetector`` 接口的 RTMW 姿态适配器。"""

    def detect(self, image_rgb: np.ndarray, timestamp_ms: int) -> Optional[List[SimpleLandmark]]:
        ts = self._check_timestamp(timestamp_ms)
        return self.engine.infer(image_rgb, ts).pose_landmarks

    def detect_with_world(
        self, image_rgb: np.ndarray, timestamp_ms: int
    ) -> Tuple[Optional[List[SimpleLandmark]], Optional[List[SimpleLandmark]]]:
        """返回 ``(pose_landmarks, None)``（2D RTMW 不提供米制 3D world 关键点）。"""
        ts = self._check_timestamp(timestamp_ms)
        res = self.engine.infer(image_rgb, ts)
        return res.pose_landmarks, None
