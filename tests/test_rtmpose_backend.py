"""RTMW / RTMPose 上肢与手部可选后端契约测试。

覆盖：
1. 模块级不 import rtmlib；
2. COCO-WholeBody 133 点 -> MediaPipe Pose(33) & Hand(21) 精确映射与下游四大分析器 / angles 消费；
3. 低置信度 / 非有限坐标过滤与可见度门限；
4. Pose 与 Hand 适配器共享单帧推理缓存（同帧仅跑一次推理）；
5. 时间戳非递减守卫、reset 与共享 close；
6. 与 gesture_analysis.api.app.process_frame 的左右手标签约定对齐（不产生二次错翻）；
7. create_gesture_detectors 的后端路由（mediapipe / rtmw / auto）与权重不可用时平滑回退。
"""

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gesture_analysis.core.analysis.arm_analyzer import ArmAnalyzer
from gesture_analysis.core.analysis.emotion_inferencer import EmotionInferencer
from gesture_analysis.core.analysis.hand_analyzer import HandAnalyzer
from gesture_analysis.core.analysis.shoulder_analyzer import ShoulderAnalyzer
from gesture_analysis.core.analysis.upper_body_analyzer import UpperBodyAnalyzer
from gesture_analysis.core.detectors import (
    HandDetector,
    PoseDetector,
    create_gesture_detectors,
)
from gesture_analysis.core.feature_extraction.angles import (
    finger_angles,
    pose_angles,
    shoulder_width,
)
from gesture_analysis.core.rtmpose_backend import (
    RTMWHandAdapter,
    RTMWPoseAdapter,
    RTMWWholeBodyEngine,
    SimpleLandmark,
    coco_wholebody133_to_mediapipe,
)

REPO = Path(__file__).resolve().parent.parent


def _make_synthetic_coco133(
    img_w: int = 1280,
    img_h: int = 720,
    *,
    dx: float = 0.0,
    left_hand_x: float = 820.0,
    right_hand_x: float = 420.0,
    hip_score: float = 0.88,
    left_hand_score: float = 0.90,
    right_hand_score: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    """构造带解剖几何形态的合成 COCO-WholeBody 133 点坐标与置信度。"""
    kpts = np.zeros((133, 2), dtype=float)
    scores = np.full((133,), 0.92, dtype=float)

    # 身体关键点 0..16
    body = {
        0: (640.0 + dx, 180.0),   # nose
        1: (670.0 + dx, 160.0),   # left_eye
        2: (610.0 + dx, 160.0),   # right_eye
        3: (710.0 + dx, 175.0),   # left_ear
        4: (570.0 + dx, 175.0),   # right_ear
        5: (760.0 + dx, 300.0),   # left_shoulder
        6: (520.0 + dx, 300.0),   # right_shoulder
        7: (820.0 + dx, 430.0),   # left_elbow
        8: (460.0 + dx, 430.0),   # right_elbow
        9: (left_hand_x + dx, 370.0),   # left_wrist
        10: (right_hand_x + dx, 370.0),  # right_wrist
        11: (730.0 + dx, 540.0),  # left_hip
        12: (550.0 + dx, 540.0),  # right_hip
    }
    for idx, (x, y) in body.items():
        kpts[idx] = [x, y]
    scores[11] = hip_score
    scores[12] = hip_score

    def _fill_hand(start: int, base_x: float, base_y: float, sc: float) -> None:
        # 21 点张开手掌骨架：0=wrist, 1..4=thumb, 5..8=index, 9..12=middle, 13..16=ring, 17..20=pinky
        kpts[start + 0] = [base_x, base_y]
        finger_offsets = [
            (-40, -20, -65, -45, -80, -70, -90, -95),   # thumb
            (-22, -55, -24, -85, -26, -110, -28, -135),  # index
            (0, -60, 0, -95, 0, -125, 0, -150),          # middle
            (22, -55, 24, -85, 26, -110, 28, -132),      # ring
            (42, -45, 48, -70, 52, -92, 56, -112),       # pinky
        ]
        idx = 1
        for ox1, oy1, ox2, oy2, ox3, oy3, ox4, oy4 in finger_offsets:
            kpts[start + idx + 0] = [base_x + ox1, base_y + oy1]
            kpts[start + idx + 1] = [base_x + ox2, base_y + oy2]
            kpts[start + idx + 2] = [base_x + ox3, base_y + oy3]
            kpts[start + idx + 3] = [base_x + ox4, base_y + oy4]
            idx += 4
        scores[start : start + 21] = sc

    _fill_hand(91, left_hand_x + dx, 370.0, left_hand_score)
    _fill_hand(112, right_hand_x + dx, 370.0, right_hand_score)
    return kpts, scores


class _StubFakeLandmarker:
    def __init__(self):
        self.closed = False

    def detect_for_video(self, image, timestamp_ms):
        return type("R", (), {"hand_landmarks": [], "pose_landmarks": []})()

    def close(self):
        self.closed = True


def _fake_mp_factory(*_args, **_kwargs):
    return _StubFakeLandmarker()


def test_module_import_does_not_import_rtmlib():
    code = (
        "import sys; "
        "import gesture_analysis.core.detectors; "
        "import gesture_analysis.core.rtmpose_backend; "
        "assert 'rtmlib' not in sys.modules, 'rtmlib leaked at import time'; "
        "assert 'mediapipe' not in sys.modules, 'mediapipe leaked at import time'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_coco_wholebody133_to_mediapipe_pose_and_hand_mapping_with_analyzers():
    kpts1, scores1 = _make_synthetic_coco133(dx=0.0)
    kpts2, scores2 = _make_synthetic_coco133(dx=12.0)

    res1 = coco_wholebody133_to_mediapipe(kpts1, scores1, img_w=1280, img_h=720, min_conf=0.3)
    res2 = coco_wholebody133_to_mediapipe(kpts2, scores2, img_w=1280, img_h=720, min_conf=0.3)

    assert res1.pose_landmarks is not None and len(res1.pose_landmarks) == 33
    assert isinstance(res1.pose_landmarks[0], SimpleLandmark)
    assert res1.pose_landmarks[0].x == pytest.approx(640.0 / 1280.0)
    assert res1.pose_landmarks[0].y == pytest.approx(180.0 / 720.0)
    assert res1.pose_landmarks[0].visibility == pytest.approx(0.92)
    assert res1.pose_landmarks[11].x == pytest.approx(760.0 / 1280.0)
    assert res1.pose_landmarks[12].x == pytest.approx(520.0 / 1280.0)
    assert res1.pose_landmarks[23].x == pytest.approx(730.0 / 1280.0)
    assert res1.pose_landmarks[24].x == pytest.approx(550.0 / 1280.0)

    # 未映射槽位保持 0.0 可见度
    for unmapped in (1, 3, 4, 6, 9, 10, 17, 25, 32):
        assert res1.pose_landmarks[unmapped].visibility == 0.0

    # 两只手均为 21 点，默认 mediapipe_handedness=True 匹配 api/app.py 翻转约定
    assert len(res1.hand_groups) == 2
    assert len(res1.hand_groups[0]) == 21
    assert len(res1.hand_groups[1]) == 21
    assert res1.handedness[0][0] == "Right"  # COCO 91..111 (解剖左手) -> MediaPipe "Right"
    assert res1.handedness[0][1] == pytest.approx(0.90)
    assert res1.handedness[1][0] == "Left"   # COCO 112..132 (解剖右手) -> MediaPipe "Left"
    assert res1.handedness[1][1] == pytest.approx(0.85)

    # mediapipe_handedness=False 时直接返回解剖学标签
    anat = coco_wholebody133_to_mediapipe(
        kpts1, scores1, img_w=1280, img_h=720, min_conf=0.3, mediapipe_handedness=False
    )
    assert [lbl for lbl, _ in anat.handedness] == ["Left", "Right"]

    # 下游四大分析器 + angles 端到端消费验证
    hand_l = HandAnalyzer(hand_id=0)
    hand_r = HandAnalyzer(hand_id=1)
    shoulder = ShoulderAnalyzer()
    arm_l = ArmAnalyzer(arm_id="left")
    arm_r = ArmAnalyzer(arm_id="right")
    upper = UpperBodyAnalyzer()

    for ts, res in ((0, res1), (100, res2)):
        sw = shoulder_width(res.pose_landmarks)
        assert sw == pytest.approx((760.0 - 520.0) / 1280.0)
        hand_l.update(res.hand_groups[0], timestamp_ms=ts, shoulder_width=sw)
        hand_r.update(res.hand_groups[1], timestamp_ms=ts, shoulder_width=sw)
        shoulder.update(res.pose_landmarks, timestamp_ms=ts, shoulder_width=sw)
        arm_l.update(res.pose_landmarks, timestamp_ms=ts, shoulder_width=sw)
        arm_r.update(res.pose_landmarks, timestamp_ms=ts, shoulder_width=sw)
        upper.update(res.pose_landmarks, timestamp_ms=ts, shoulder_width=sw)

    assert hand_l.is_valid() is True
    assert hand_r.is_valid() is True
    assert shoulder.is_valid() is True
    assert arm_l.is_valid() is True
    assert arm_r.is_valid() is True
    assert upper.is_valid() is True

    pa = pose_angles(res1.pose_landmarks)
    assert set(pa.keys()) == {
        "left_elbow_angle",
        "right_elbow_angle",
        "left_shoulder_angle",
        "right_shoulder_angle",
        "shoulder_angle",
        "head_tilt_angle",
        "head_pitch_angle",
        "torso_angle",
    }
    fa = finger_angles(res1.hand_groups[0])
    assert set(fa.keys()) == {"thumb", "index", "middle", "ring", "pinky"}


def test_low_confidence_and_non_finite_hand_filtering():
    # 左手腕部 NaN 且右手置信度正常 -> 只保留右手
    kpts, scores = _make_synthetic_coco133(hip_score=0.4)
    kpts[91] = [np.nan, 370.0]

    res = coco_wholebody133_to_mediapipe(kpts, scores, img_w=1280, img_h=720, min_conf=0.3)
    assert len(res.hand_groups) == 1
    assert res.handedness == [("Left", pytest.approx(0.85))]

    # 髋部可见度 0.4 <= VISIBILITY_FLOOR(0.6) -> 依赖髋部的 3 个角度自动缺失，肘角保留
    pa = pose_angles(res.pose_landmarks)
    assert "left_elbow_angle" in pa
    assert "left_shoulder_angle" not in pa
    assert "right_shoulder_angle" not in pa
    assert "torso_angle" not in pa

    # 全部关键点低置信度 -> pose_landmarks=None 且 hand_groups=[]
    low_scores = np.full((133,), 0.1, dtype=float)
    empty_res = coco_wholebody133_to_mediapipe(kpts, low_scores, img_w=1280, img_h=720, min_conf=0.3)
    assert empty_res.pose_landmarks is None
    assert empty_res.hand_groups == []
    assert empty_res.handedness == []


def test_single_frame_inference_cache_shared_between_pose_and_hand_adapters():
    kpts, scores = _make_synthetic_coco133()
    seen_bgr: list[np.ndarray] = []

    class _FakeWholebody:
        def __init__(self, **_kw):
            self.calls = 0
            self.closed = False

        def __call__(self, image_bgr: np.ndarray):
            self.calls += 1
            seen_bgr.append(image_bgr.copy())
            return kpts[None, ...], scores[None, ...]

        def close(self):
            self.closed = True

    holder: dict[str, _FakeWholebody] = {}

    def _factory(**kw):
        holder["wb"] = _FakeWholebody(**kw)
        return holder["wb"]

    engine = RTMWWholeBodyEngine(mode="lightweight", wholebody_factory=_factory)
    pose_adapter = RTMWPoseAdapter(engine)
    hand_adapter = RTMWHandAdapter(engine)

    frame1 = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame1[..., 0] = 200  # R
    frame1[..., 1] = 100  # G
    frame1[..., 2] = 50   # B

    # 同一帧先调 pose 再调 hand：底层 RTMW 只执行一次推理，且收到 BGR
    pose_lms, world_lms = pose_adapter.detect_with_world(frame1, 100)
    hand_groups, per_hand = hand_adapter.detect_with_handedness(frame1, 100)
    assert pose_lms is not None and world_lms is None
    assert len(hand_groups) == 2 and len(per_hand) == 2
    assert engine.inference_count == 1
    assert holder["wb"].calls == 1
    assert tuple(seen_bgr[0][0, 0]) == (50, 100, 200)  # RGB -> BGR

    # 下一帧先调 hand 再调 pose：同样共享缓存，总推理次数只增 1
    frame2 = frame1.copy()
    frame2[0, 0, 0] = 199
    hand_adapter.detect_with_handedness(frame2, 200)
    pose_adapter.detect_with_world(frame2, 200)
    assert engine.inference_count == 2
    assert holder["wb"].calls == 2

    # 原地修改同一 buffer 但时间戳不变，也必须命中新推理而非脏缓存
    frame2[10, 10, 1] = 250
    pose_adapter.detect_with_world(frame2, 200)
    assert engine.inference_count == 3

    # 仅关闭一个适配器不应提前销毁共享模型，两个都关才释放
    hand_adapter.close()
    assert holder["wb"].closed is False
    pose_adapter.close()
    assert holder["wb"].closed is True
    # 重复 close 幂等
    pose_adapter.close()
    hand_adapter.close()


def test_adapter_timestamp_guard_and_reset():
    kpts, scores = _make_synthetic_coco133()

    class _FakeWholebody:
        def __call__(self, _bgr):
            return kpts[None, ...], scores[None, ...]

    engine = RTMWWholeBodyEngine(wholebody_factory=lambda **_kw: _FakeWholebody())
    pose_adapter = RTMWPoseAdapter(engine)
    hand_adapter = RTMWHandAdapter(engine)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    pose_adapter.detect(frame, 50)
    pose_adapter.detect(frame, 50)  # 相等允许
    with pytest.raises(ValueError, match="timestamp_ms"):
        pose_adapter.detect(frame, 49)

    pose_adapter.reset()
    hand_adapter.reset()
    pose_adapter.detect(frame, 0)


def test_process_frame_routes_coco_left_and_right_hands_into_correct_slots():
    gesture_app = importlib.import_module("gesture_analysis.api.app")
    # 令 COCO 左手 (91..111) 在 x=900, COCO 右手 (112..132) 在 x=300
    kpts, scores = _make_synthetic_coco133(left_hand_x=900.0, right_hand_x=300.0)

    class _FakeWholebody:
        def __call__(self, _bgr):
            return kpts[None, ...], scores[None, ...]

    engine = RTMWWholeBodyEngine(wholebody_factory=lambda **_kw: _FakeWholebody())
    dets = {
        "hands": RTMWHandAdapter(engine),
        "pose": RTMWPoseAdapter(engine),
    }

    recorded_left: list[float] = []
    recorded_right: list[float] = []

    class _SpyHandAnalyzer(HandAnalyzer):
        def __init__(self, hand_id: int, bucket: list[float]):
            super().__init__(hand_id=hand_id)
            self._bucket = bucket

        def update(self, landmarks, timestamp_ms=None, shoulder_width=None):
            self._bucket.append(float(landmarks[0].x))
            super().update(landmarks, timestamp_ms=timestamp_ms, shoulder_width=shoulder_width)

    analyzers = {
        "left_hand": _SpyHandAnalyzer(0, recorded_left),
        "right_hand": _SpyHandAnalyzer(1, recorded_right),
        "shoulder": ShoulderAnalyzer(),
        "left_arm": ArmAnalyzer(arm_id="left"),
        "right_arm": ArmAnalyzer(arm_id="right"),
        "upper_body": UpperBodyAnalyzer(),
        "emotion": EmotionInferencer(),
        "world": {
            "shoulder": ShoulderAnalyzer(metric=True),
            "left_arm": ArmAnalyzer(arm_id="left", metric=True),
            "right_arm": ArmAnalyzer(arm_id="right", metric=True),
        },
    }

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    out = gesture_app.process_frame(frame, 100, analyzers, dets)
    assert out["detected_hands"] == 2
    assert recorded_left == [pytest.approx(900.0 / 1280.0)]
    assert recorded_right == [pytest.approx(300.0 / 1280.0)]
    assert engine.inference_count == 1


def test_create_gesture_detectors_routing_and_graceful_fallback(monkeypatch, tmp_path, caplog):
    hand_task = tmp_path / "hand.task"
    pose_task = tmp_path / "pose.task"
    hand_task.write_bytes(b"stub")
    pose_task.write_bytes(b"stub")

    # 1. 默认走 MediaPipe
    monkeypatch.delenv("JX_GESTURE_BACKEND", raising=False)
    h_mp, p_mp = create_gesture_detectors(hand_path=hand_task, pose_path=pose_task, factory=_fake_mp_factory)
    assert isinstance(h_mp, HandDetector)
    assert isinstance(p_mp, PoseDetector)

    # 2. JX_GESTURE_BACKEND=rtmw 且工厂可用 -> 返回共享 engine 的 RTMW 适配器
    kpts, scores = _make_synthetic_coco133()

    class _OkWholebody:
        def __call__(self, _bgr):
            return kpts[None, ...], scores[None, ...]

    monkeypatch.setenv("JX_GESTURE_BACKEND", "rtmw")
    h_rt, p_rt = create_gesture_detectors(
        hand_path=hand_task,
        pose_path=pose_task,
        wholebody_factory=lambda **_kw: _OkWholebody(),
    )
    assert isinstance(h_rt, RTMWHandAdapter)
    assert isinstance(p_rt, RTMWPoseAdapter)
    assert h_rt.engine is p_rt.engine

    # 3. create_gesture_detectors("rtmw") 在权重不可用时平滑回退到 MediaPipe
    def _broken_wholebody(**_kw):
        raise RuntimeError("weights download blocked")

    with caplog.at_level(logging.WARNING):
        caplog.clear()
        h_fb, p_fb = create_gesture_detectors(
            "rtmw",
            hand_kwargs={},
            pose_kwargs={},
            factory=_fake_mp_factory,
            wholebody_factory=_broken_wholebody,
        )
    # 注意传了 "rtmw" 位置参数且 hand_path 使用配置默认，或显式传 backend="rtmw" + hand_path/pose_path
    assert isinstance(h_fb, HandDetector)
    assert isinstance(p_fb, PoseDetector)
    assert any("回退" in rec.getMessage() for rec in caplog.records)

    # 4. auto 模式在无缓存权重时不卡网络直接回退 MediaPipe
    import gesture_analysis.core.rtmpose_backend as rtb

    monkeypatch.setattr(rtb, "has_cached_rtmw_weights", lambda _m=None: False)
    h_auto, p_auto = create_gesture_detectors(
        hand_path=hand_task,
        pose_path=pose_task,
        backend="auto",
        factory=_fake_mp_factory,
    )
    assert isinstance(h_auto, HandDetector)
    assert isinstance(p_auto, PoseDetector)

    # 5. 非法 backend 响亮报错
    with pytest.raises(ValueError, match="不支持的手势检测后端"):
        create_gesture_detectors(backend="invalid_backend")
