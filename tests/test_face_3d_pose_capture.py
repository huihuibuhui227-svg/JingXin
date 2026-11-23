# tests/test_face_3d_pose_capture.py
"""Wave 1: 3D 面部姿态矩阵欧拉角解析、采集画面质量审计与接口增量透出测试。"""

from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import face_expression.utils.logger as face_logger_module
from face_expression.models.blendshapes import BLENDSHAPE_COLUMNS
from face_expression.pipeline.detector import FaceDetector, extract_euler_3d_from_matrix
from face_expression.pipeline.video_pipeline import VideoPipeline, assess_frame_quality

face_app = importlib.import_module("face_expression.api.app")


def _rot_x(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _mat4(rot3: np.ndarray, tx: float = 0.0, ty: float = 0.0, tz: float = -45.0) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot3
    mat[:3, 3] = [tx, ty, tz]
    return mat


def _valid_478_landmarks() -> list[tuple[float, float]]:
    pts = [(0.40 + (i % 20) * 0.01, 0.25 + (i // 20) * 0.02) for i in range(478)]
    pts[10] = (0.50, 0.15)
    pts[152] = (0.50, 0.85)
    pts[234] = (0.30, 0.50)
    pts[455] = (0.70, 0.50)
    pts[1] = (0.50, 0.45)
    pts[33] = (0.38, 0.40)
    pts[133] = (0.45, 0.40)
    pts[160] = (0.40, 0.39)
    pts[153] = (0.40, 0.41)
    pts[159] = (0.42, 0.39)
    pts[144] = (0.42, 0.41)
    pts[362] = (0.62, 0.40)
    pts[263] = (0.55, 0.40)
    pts[387] = (0.60, 0.39)
    pts[380] = (0.60, 0.41)
    pts[386] = (0.58, 0.39)
    pts[373] = (0.58, 0.41)
    pts[130] = (0.36, 0.40)
    pts[359] = (0.64, 0.40)
    pts[107] = (0.46, 0.34)
    pts[336] = (0.54, 0.34)
    return pts


def _textured_image(mean_val: int = 128, h: int = 120, w: int = 160) -> np.ndarray:
    img = np.full((h, w, 3), mean_val, dtype=np.uint8)
    # 棋盘纹理保证拉普拉斯清晰度远大于 15
    yy, xx = np.indices((h, w))
    delta = np.where(((yy // 4 + xx // 4) % 2) == 0, 30, -30)
    img = np.clip(img.astype(np.int16) + delta[:, :, None], 0, 255).astype(np.uint8)
    return img


# ── 1. extract_euler_3d_from_matrix ───────────────────────────────────────────

def test_euler_identity_gives_zeros():
    got = extract_euler_3d_from_matrix(np.eye(4, dtype=np.float32))
    assert got is not None
    assert got == {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}
    json.dumps(got, allow_nan=False)


@pytest.mark.parametrize("angle", [-65.0, -30.0, 18.5, 45.0, 75.0])
def test_euler_pure_rotations(angle):
    yaw_only = extract_euler_3d_from_matrix(_mat4(_rot_y(angle)))
    assert yaw_only["yaw_deg"] == pytest.approx(angle, abs=1e-6)
    assert yaw_only["pitch_deg"] == pytest.approx(0.0, abs=1e-6)
    assert yaw_only["roll_deg"] == pytest.approx(0.0, abs=1e-6)

    pitch_only = extract_euler_3d_from_matrix(_mat4(_rot_x(angle)))
    assert pitch_only["yaw_deg"] == pytest.approx(0.0, abs=1e-6)
    assert pitch_only["pitch_deg"] == pytest.approx(angle, abs=1e-6)
    assert pitch_only["roll_deg"] == pytest.approx(0.0, abs=1e-6)

    roll_only = extract_euler_3d_from_matrix(_mat4(_rot_z(angle)))
    assert roll_only["yaw_deg"] == pytest.approx(0.0, abs=1e-6)
    assert roll_only["pitch_deg"] == pytest.approx(0.0, abs=1e-6)
    assert roll_only["roll_deg"] == pytest.approx(angle, abs=1e-6)


def test_euler_combined_yxz_and_nested_list_and_arcsin_clamp():
    rot = _rot_y(-22.5) @ _rot_x(14.0) @ _rot_z(9.5)
    mat_list = _mat4(rot, tx=-10.0, ty=3.0, tz=-50.0).tolist()
    got = extract_euler_3d_from_matrix(mat_list)
    assert got["yaw_deg"] == pytest.approx(-22.5, abs=1e-6)
    assert got["pitch_deg"] == pytest.approx(14.0, abs=1e-6)
    assert got["roll_deg"] == pytest.approx(9.5, abs=1e-6)

    # float32 微小超界(-R[1,2] > 1.0)应被夹到 [-1, 1],不产生 NaN
    overshoot = _mat4(_rot_x(90.0))
    overshoot[1, 2] = -1.000001
    clamped = extract_euler_3d_from_matrix(overshoot)
    assert clamped is not None
    assert clamped["pitch_deg"] == pytest.approx(90.0, abs=1e-4)
    json.dumps(clamped, allow_nan=False)


@pytest.mark.parametrize("pitch_deg", [90.0, -90.0])
@pytest.mark.parametrize("yaw_deg", [-40.0, 0.0, 35.0])
def test_euler_gimbal_lock_plus_minus_90_pitch(pitch_deg, yaw_deg):
    rot = _rot_y(yaw_deg) @ _rot_x(pitch_deg) @ _rot_z(25.0)
    got = extract_euler_3d_from_matrix(_mat4(rot))
    assert got is not None
    assert got["pitch_deg"] == pytest.approx(pitch_deg, abs=1e-4)
    assert got["roll_deg"] == 0.0
    # 重建旋转矩阵应与原矩阵精确一致
    rec = _rot_y(got["yaw_deg"]) @ _rot_x(got["pitch_deg"]) @ _rot_z(got["roll_deg"])
    assert np.allclose(rec, rot, atol=1e-6)


@pytest.mark.parametrize("bad", [
    None,
    [],
    np.eye(3),
    np.ones((4, 3)),
    np.zeros((4, 4)),
    [[1, 0], [0]],
    "not_a_matrix",
    np.where(np.eye(4) == 1, np.nan, 0.0),
    np.where(np.eye(4) == 1, np.inf, 0.0),
])
def test_euler_malformed_inputs_return_none(bad):
    assert extract_euler_3d_from_matrix(bad) is None


# ── 2. detector.py: _default_factory + detect_full ────────────────────────────

def test_default_factory_enables_3d_transformation_matrixes(monkeypatch):
    from mediapipe.tasks.python import vision
    from face_expression.pipeline.detector import _default_factory

    captured = {}

    def fake_create(options, *args, **kwargs):
        captured["options"] = options
        return type("LM", (), {"close": lambda self: None})()

    monkeypatch.setattr(vision.FaceLandmarker, "create_from_options", staticmethod(fake_create))
    _default_factory(Path("unused.task"), num_faces=1,
                     min_detection_confidence=0.8, min_tracking_confidence=0.8)

    opts = captured["options"]
    assert getattr(opts, "output_face_blendshapes", None) is True
    assert getattr(opts, "output_facial_transformation_matrixes", None) is True


def test_detect_full_returns_landmarks_blendshapes_and_3d_pose():
    pt = type("P", (), {"x": 0.4, "y": 0.6})
    cat = type("C", (), {"category_name": "jawOpen", "score": 0.25})
    mat = _mat4(_rot_y(15.0) @ _rot_x(-8.0) @ _rot_z(5.0))

    class _FakeLandmarker:
        def detect_for_video(self, image, ts):
            return type("R", (), {
                "face_landmarks": [[pt(), pt()]],
                "face_blendshapes": [[cat()]],
                "facial_transformation_matrixes": [mat],
            })()

        def close(self):
            pass

    d = FaceDetector(Path("unused.task"), factory=lambda *a, **k: _FakeLandmarker())
    lm, bs, pose = d.detect_full(np.zeros((32, 32, 3), dtype=np.uint8), 100)
    assert lm == [(0.4, 0.6), (0.4, 0.6)]
    assert bs == {"jawOpen": 0.25}
    assert pose["yaw_deg"] == pytest.approx(15.0, abs=1e-6)
    assert pose["pitch_deg"] == pytest.approx(-8.0, abs=1e-6)
    assert pose["roll_deg"] == pytest.approx(5.0, abs=1e-6)

    # detect_with_blendshapes 与 detect 契约保持不变
    assert d.detect_with_blendshapes(np.zeros((32, 32, 3), dtype=np.uint8), 200) == (lm, bs)
    assert d.detect(np.zeros((32, 32, 3), dtype=np.uint8), 300) == lm


def test_detect_full_handles_no_face_and_missing_matrix():
    class _NoFaceLandmarker:
        def detect_for_video(self, image, ts):
            return type("R", (), {
                "face_landmarks": [],
                "face_blendshapes": [],
                "facial_transformation_matrixes": [],
            })()

        def close(self):
            pass

    d_empty = FaceDetector(Path("unused.task"), factory=lambda *a, **k: _NoFaceLandmarker())
    assert d_empty.detect_full(np.zeros((16, 16, 3), dtype=np.uint8), 0) == (None, {}, None)

    pt = type("P", (), {"x": 0.5, "y": 0.5})

    class _NoMatrixLandmarker:
        def detect_for_video(self, image, ts):
            return type("R", (), {"face_landmarks": [[pt()]]})()

        def close(self):
            pass

    d_nomat = FaceDetector(Path("unused.task"), factory=lambda *a, **k: _NoMatrixLandmarker())
    lm, bs, pose = d_nomat.detect_full(np.zeros((16, 16, 3), dtype=np.uint8), 0)
    assert lm == [(0.5, 0.5)]
    assert bs == {}
    assert pose is None


# ── 3. assess_frame_quality ───────────────────────────────────────────────────

def test_assess_frame_quality_good_frame():
    img = _textured_image(mean_val=128)
    lms = [(0.2, 0.2), (0.7, 0.8)]
    q = assess_frame_quality(img, lms)
    assert set(q.keys()) == {
        "brightness_mean", "blur_laplacian", "face_area_ratio", "quality_ok", "warnings"
    }
    assert q["brightness_mean"] == pytest.approx(128.0, abs=2.0)
    assert q["blur_laplacian"] >= 15.0
    assert q["face_area_ratio"] == pytest.approx(0.30, abs=1e-5)
    assert q["quality_ok"] is True
    assert q["warnings"] == []
    json.dumps(q, allow_nan=False)


def test_assess_frame_quality_flags_dark_overexposed_blur_and_no_face():
    dark = _textured_image(mean_val=20)
    q_dark = assess_frame_quality(dark, [(0.3, 0.3), (0.6, 0.6)])
    assert q_dark["brightness_mean"] < 35.0
    assert q_dark["quality_ok"] is False
    assert "光线偏暗" in q_dark["warnings"]

    bright = _textured_image(mean_val=240)
    q_bright = assess_frame_quality(bright, [(0.3, 0.3), (0.6, 0.6)])
    assert q_bright["brightness_mean"] > 230.0
    assert q_bright["quality_ok"] is False
    assert any("过亮" in w for w in q_bright["warnings"])

    blurry = np.full((64, 64, 3), 128, dtype=np.uint8)
    q_blur = assess_frame_quality(blurry, [(0.3, 0.3), (0.6, 0.6)])
    assert q_blur["blur_laplacian"] < 15.0
    assert q_blur["quality_ok"] is False
    assert "画面模糊" in q_blur["warnings"]

    good = _textured_image(mean_val=128)
    q_noface = assess_frame_quality(good, None)
    assert q_noface["face_area_ratio"] is None
    assert q_noface["quality_ok"] is False
    assert "未检测到人脸" in q_noface["warnings"]


# ── 4. VideoPipeline backward-compat & CSV schema unchanged ───────────────────

def test_video_pipeline_supports_all_detector_tiers_and_preserves_csv(tmp_path, monkeypatch):
    import face_expression.utils.logger as lg
    monkeypatch.setattr(lg, "LOGS_DIR", str(tmp_path))

    lms = _valid_478_landmarks()
    pose = {"yaw_deg": 12.0, "pitch_deg": -3.0, "roll_deg": 1.5}
    img = _textured_image(128)

    class _FullDetector:
        def detect_full(self, image_rgb, ts):
            return lms, {"eyeBlinkLeft": 0.2}, pose

        def reset(self):
            pass

        def close(self):
            pass

    class _BlendshapeOnlyDetector:
        def detect_with_blendshapes(self, image_rgb, ts):
            return lms, {"jawOpen": 0.1}

        def reset(self):
            pass

        def close(self):
            pass

    class _LegacyDetectOnly:
        def detect(self, image_rgb, ts):
            return lms

        def reset(self):
            pass

        def close(self):
            pass

    class _NoFaceFullDetector:
        def detect_full(self, image_rgb, ts):
            return None, {}, None

    p_full = VideoPipeline(session_id="s_full", detector=_FullDetector())
    res_obj, _, d_full = p_full.process_frame(img, 100)
    assert res_obj is not None
    assert d_full["head_pose_3d"] == pose
    assert d_full["frame_quality"]["quality_ok"] is True

    # DataLogger 列头与 52 blendshape 协议保持完全不变
    logger_before = lg.DataLogger("video", session_id="20261124_000001_aaaa")
    expected_cols = list(logger_before.fieldnames)
    assert "head_pose_3d" not in expected_cols and "frame_quality" not in expected_cols
    assert len([c for c in expected_cols if c.startswith("bs_")]) == len(BLENDSHAPE_COLUMNS)
    assert logger_before.log(d_full) is True

    p_bs = VideoPipeline(session_id="s_bs", detector=_BlendshapeOnlyDetector())
    _, _, d_bs = p_bs.process_frame(img, 100)
    assert d_bs["head_pose_3d"] is None
    assert d_bs["bs_jawOpen"] == 0.1
    assert d_bs["frame_quality"]["quality_ok"] is True

    p_leg = VideoPipeline(session_id="s_leg", detector=_LegacyDetectOnly())
    _, _, d_leg = p_leg.process_frame(img, 100)
    assert d_leg["head_pose_3d"] is None
    assert d_leg["frame_quality"]["quality_ok"] is True

    p_nf = VideoPipeline(session_id="s_nf", detector=_NoFaceFullDetector())
    res_nf, lms_nf, d_nf = p_nf.process_frame(img, 250)
    assert res_nf is None and lms_nf is None
    assert d_nf["emotion"] == "no_face"
    assert d_nf["timestamp"] == 0.25
    assert d_nf["head_pose_3d"] is None
    assert "未检测到人脸" in d_nf["frame_quality"]["warnings"]


# ── 5. face_expression.api.app /health & /analyze ─────────────────────────────

class _FakeUpload:
    def __init__(self, data: bytes):
        self._data = data
        self.filename = "frame.jpg"
        self.content_type = "image/jpeg"

    async def read(self) -> bytes:
        return self._data


class _FakeRequest:
    async def form(self):
        return {}


def _encode_jpg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_api_health_and_analyze_incremental_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path / "rec"))
    monkeypatch.setenv("JINGXIN_RETAIN_MEDIA", "0")
    monkeypatch.setattr(face_app, "LOGS_DIR", str(tmp_path))
    monkeypatch.setattr(face_logger_module, "LOGS_DIR", str(tmp_path))
    face_app.session_pipelines.clear()
    face_app.session_loggers.clear()
    face_app.session_clocks.clear()
    face_app.session_framing.clear()

    health = asyncio.run(face_app.health_check())
    assert health["status"] == "healthy"
    assert health["landmarker_3d_matrix"] is True
    assert health["blendshapes_count"] == 52
    assert health["live_framing"] in ("on", "off")

    pose = {"yaw_deg": 5.5, "pitch_deg": -2.0, "roll_deg": 0.5}
    qual = {
        "brightness_mean": 120.0,
        "blur_laplacian": 45.0,
        "face_area_ratio": 0.25,
        "quality_ok": True,
        "warnings": [],
    }

    class _RichPipeline:
        def __init__(self, session_id=None):
            self.session_id = session_id

        def process_frame(self, image_rgb, ts):
            return object(), [(0.5, 0.5)], {
                "timestamp": ts / 1000.0,
                "focus_score": 0.8,
                "face_scale": 0.5,
                "dominant_emotion": "neutral",
                "confidence": 0.9,
                "head_pose_3d": pose,
                "frame_quality": qual,
            }

        def measured_fps(self):
            return 1.0

        def close(self):
            pass

    monkeypatch.setattr(face_app, "VideoPipeline", _RichPipeline)
    resp = asyncio.run(face_app.analyze_frame(
        request=_FakeRequest(),
        file=_FakeUpload(_encode_jpg(_textured_image(120))),
        session_id="20261124_000002_cccc",
        probe=True,
    ))
    body = json.loads(resp.body)
    assert body["status"] == "success"
    assert body["result"]["head_pose_3d"] == pose
    assert body["result"]["frame_quality"] == qual

    # 没脸路径同样安全透出 frame_quality
    class _NoFacePipeline(_RichPipeline):
        def process_frame(self, image_rgb, ts):
            return None, None, {
                "emotion": "no_face",
                "timestamp": ts / 1000.0,
                "head_pose_3d": None,
                "frame_quality": {**qual, "face_area_ratio": None, "quality_ok": False, "warnings": ["未检测到人脸"]},
            }

    face_app.session_pipelines.clear()
    monkeypatch.setattr(face_app, "VideoPipeline", _NoFacePipeline)
    resp_nf = asyncio.run(face_app.analyze_frame(
        request=_FakeRequest(),
        file=_FakeUpload(_encode_jpg(_textured_image(120))),
        session_id="20261124_000003_dddd",
        probe=True,
    ))
    body_nf = json.loads(resp_nf.body)
    assert body_nf["status"] == "no_face"
    assert "未检测到人脸" in body_nf["frame_quality"]["warnings"]
