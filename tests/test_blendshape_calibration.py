# tests/test_blendshape_calibration.py
"""Wave 2: 3D 头姿/画面质量 CSV 持久化与 Blendshape 时序姿态校准测试。"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import face_expression.utils.logger as face_logger_module
from face_expression.models.blendshapes import BLENDSHAPE_COLUMNS
from face_expression.pipeline.video_pipeline import VideoPipeline
from report_frontend.blendshape_calibration import (
    _smooth_series,
    calibrate_session_blendshapes,
)
from tools import jx_blendshape_calibration_bench as bench
from tools.jx_pii_guard import scan_text


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
    yy, xx = np.indices((h, w))
    delta = np.where(((yy // 4 + xx // 4) % 2) == 0, 30, -30)
    return np.clip(img.astype(np.int16) + delta[:, :, None], 0, 255).astype(np.uint8)


# ── 1. 3D 头姿与画面质量标量列持久化 ──────────────────────────────────────────

def test_3d_pose_and_quality_persist_to_csv_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(face_logger_module, "LOGS_DIR", str(tmp_path))

    lms = _valid_478_landmarks()
    pose = {"yaw_deg": 18.25, "pitch_deg": -7.5, "roll_deg": 3.125}

    class _SequenceDetector:
        def __init__(self):
            self.step = 0

        def detect_full(self, image_rgb, ts):
            self.step += 1
            if self.step == 3:
                return None, {}, None
            return lms, {"eyeSquintLeft": 0.35, "jawOpen": 0.15}, pose

        def reset(self):
            pass

        def close(self):
            pass

    pipeline = VideoPipeline(session_id="20261124_112233_cafe", detector=_SequenceDetector())
    csv_logger = face_logger_module.DataLogger("video", session_id="20261124_112233_cafe")

    good_img = _textured_image(128)
    blurry_img = np.full((120, 160, 3), 128, dtype=np.uint8)

    _, _, d_good = pipeline.process_frame(good_img, 100)
    _, _, d_blur = pipeline.process_frame(blurry_img, 200)
    _, _, d_noface = pipeline.process_frame(good_img, 300)

    assert d_good["head_pose_3d"] == pose
    assert d_good["frame_quality"]["quality_ok"] is True
    assert d_blur["frame_quality"]["quality_ok"] is False
    assert d_noface["head_pose_3d"] is None

    assert csv_logger.log(d_good) is True
    assert csv_logger.log(d_blur) is True
    assert csv_logger.log(d_noface) is True

    fieldnames = list(csv_logger.fieldnames)
    anchor_bs = fieldnames.index("micro_exp_onset_frame")
    assert fieldnames[anchor_bs + 1:anchor_bs + 1 + len(BLENDSHAPE_COLUMNS)] == list(BLENDSHAPE_COLUMNS)
    scale_idx = fieldnames.index("face_scale")
    assert fieldnames[scale_idx + 1:] == [
        "head_yaw_3d",
        "head_pitch_3d",
        "head_roll_3d",
        "frame_brightness",
        "frame_blur_laplacian",
        "frame_quality_ok",
    ]
    assert "head_pose_3d" not in fieldnames
    assert "frame_quality" not in fieldnames

    df = pd.read_csv(csv_logger.get_log_path())
    assert len(df) == 3
    assert df.loc[0, "head_yaw_3d"] == pytest.approx(18.25)
    assert df.loc[0, "head_pitch_3d"] == pytest.approx(-7.5)
    assert df.loc[0, "head_roll_3d"] == pytest.approx(3.125)
    assert df.loc[0, "frame_quality_ok"] == 1
    assert df.loc[1, "frame_quality_ok"] == 0
    assert df.loc[1, "frame_blur_laplacian"] < 15.0
    # no_face 帧不得伪造 0 度头姿
    assert pd.isna(df.loc[2, "head_yaw_3d"])
    assert pd.isna(df.loc[2, "head_pitch_3d"])
    assert pd.isna(df.loc[2, "head_roll_3d"])
    assert df.loc[2, "frame_quality_ok"] == 0


# ── 2. 合成头姿混杂数据校准与平滑验证 ─────────────────────────────────────────

def _synthetic_pose_confounded_df(n_clean: int = 60, n_bad: int = 10, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_total = n_clean + n_bad
    t = np.arange(n_total, dtype=float) * 0.2

    # 五个真实面部动作信号（多周期自然起伏，与慢速头部漂移正交）
    s_eye = np.clip(0.35 + 0.30 * np.sin(1.25 * t) + 0.10 * np.cos(2.1 * t), 0.05, 0.95)
    s_smile = np.clip(0.30 + 0.28 * np.sin(1.10 * t + 0.7), 0.05, 0.95)
    s_mouth = np.clip(0.32 + 0.26 * np.cos(1.40 * t), 0.05, 0.95)
    s_press = np.clip(0.30 + 0.25 * np.sin(0.95 * t + 1.2), 0.05, 0.95)
    s_brow = np.clip(0.30 + 0.25 * np.cos(1.15 * t + 0.4), 0.05, 0.95)

    # 剧烈 3D 头部转动在几何代理和 Blendshape 上制造反向混杂漂移（信号保持在有效量程内）
    yaw_3d = 22.0 * np.sin(0.13 * t)
    pitch_3d = 14.0 * np.cos(0.09 * t)
    roll_3d = 8.0 * np.sin(0.07 * t)

    def _pair(sig: np.ndarray, *, polarity: int = 1) -> tuple[np.ndarray, np.ndarray]:
        geo_pos = 0.55 * sig + 0.012 * yaw_3d - 0.010 * pitch_3d + rng.normal(0.0, 0.025, n_total)
        geo = (0.75 - geo_pos) if polarity == -1 else (0.15 + geo_pos)
        bs = np.clip(
            0.35 + 0.50 * sig - 0.010 * yaw_3d + 0.008 * pitch_3d + rng.normal(0.0, 0.025, n_total),
            0.0,
            1.0,
        )
        return geo, bs

    au7, bs_eye = _pair(s_eye, polarity=1)
    au12, bs_smile = _pair(s_smile, polarity=1)
    au25, bs_mouth = _pair(s_mouth, polarity=1)
    au23, bs_press = _pair(s_press, polarity=-1)
    au4, bs_brow = _pair(s_brow, polarity=-1)

    q_ok = np.ones(n_total, dtype=int)
    blur = np.full(n_total, 85.0, dtype=float)
    scale = 0.35 + 0.03 * np.sin(0.05 * t)

    # 后 n_bad 帧注入未通过画面质量门控的严重模糊/异常数据
    q_ok[n_clean:] = 0
    blur[n_clean:] = 4.0
    au7[n_clean:] = rng.uniform(0.0, 1.0, n_bad)
    bs_eye[n_clean:] = rng.uniform(0.0, 1.0, n_bad)

    return pd.DataFrame({
        "timestamp": t,
        "face_scale": scale,
        "head_yaw_3d": yaw_3d,
        "head_pitch_3d": pitch_3d,
        "head_roll_3d": roll_3d,
        "frame_brightness": np.full(n_total, 128.0),
        "frame_blur_laplacian": blur,
        "frame_quality_ok": q_ok,
        "au7_eye_squeeze": au7,
        "au12_smile": au12,
        "au25_mouth_open": au25,
        "au23_lip_compression": au23,
        "au4_frown": au4,
        "bs_eyeSquintLeft": bs_eye,
        "bs_eyeSquintRight": bs_eye,
        "bs_mouthSmileLeft": bs_smile,
        "bs_mouthSmileRight": bs_smile,
        "bs_jawOpen": bs_mouth,
        "bs_mouthPressLeft": bs_press,
        "bs_mouthPressRight": bs_press,
        "bs_browDownLeft": bs_brow,
        "bs_browDownRight": bs_brow,
    })


def test_calibrate_session_removes_3d_pose_confounding_and_gates_quality():
    df = _synthetic_pose_confounded_df(n_clean=60, n_bad=10)
    res = calibrate_session_blendshapes({"face": df})

    assert res["available"] is True
    assert res["pose_mode"] == "3d_matrix"
    assert res["n_raw_frames"] == 70
    assert res["n_clean_frames"] == 60
    assert res["quality_pass_rate"] == pytest.approx(60 / 70, abs=1e-5)
    assert res["eligible_construct_count"] == 5
    assert isinstance(res["inference_ms"], float) and res["inference_ms"] >= 0.0

    by_key = {c["key"]: c for c in res["constructs"]}
    assert set(by_key.keys()) == {"eye_squeeze", "smile", "mouth_open", "lip_compression", "brow_furrow"}

    for key, item in by_key.items():
        assert item["n_valid"] == 60
        assert item["raw_rho"] is not None
        assert item["calibrated_rho"] is not None
        assert item["calibrated_rho"] > item["raw_rho"]
        assert item["calibrated_rho"] >= 0.75, f"{key} 校准后相关度未恢复: {item}"
        assert item["rho_delta"] > 0.2
        assert item["pose_r2"] is not None and item["pose_r2"] > 0.3
        assert item["eligibility"] == "已校准可融合"
        assert "(" not in item["note"] and "（" not in item["note"]

    assert "(" not in res["summary_plain"] and "（" not in res["summary_plain"]
    assert "(" not in res["disclaimer"] and "（" not in res["disclaimer"]
    json.dumps(res, allow_nan=False)


def test_calibrate_legacy_2d_and_inactive_channel_and_short_streams():
    # 短序列与中等序列平滑（含 len < 5 滚动中位数回退与 len >= 5 savgol_filter）
    for n in (2, 3, 4, 5, 6, 15, 30):
        arr = np.sin(np.linspace(0, 2 * np.pi, n)) + (0.15 * np.array([(-1) ** i for i in range(n)]))
        sm = _smooth_series(arr)
        assert sm.shape == (n,)
        assert np.all(np.isfinite(sm))
        if n >= 5:
            assert np.std(np.diff(sm)) < np.std(np.diff(arr))

    # 2d_legacy 模式 + brow_furrow 通道贴底未激活
    df_leg = _synthetic_pose_confounded_df(n_clean=30, n_bad=0, seed=99).drop(
        columns=["head_yaw_3d", "head_pitch_3d", "head_roll_3d", "frame_brightness", "frame_blur_laplacian", "frame_quality_ok"]
    )
    df_leg["head_yaw"] = np.linspace(-0.2, 0.2, len(df_leg))
    df_leg["head_pitch"] = np.linspace(-0.3, 0.1, len(df_leg))
    df_leg["bs_browDownLeft"] = 0.0
    df_leg["bs_browDownRight"] = 0.0
    df_leg.loc[0, "bs_browDownLeft"] = 0.02
    df_leg.loc[0, "bs_browDownRight"] = 0.02

    res_leg = calibrate_session_blendshapes({"face": df_leg})
    assert res_leg["available"] is True
    assert res_leg["pose_mode"] == "2d_legacy"
    by_key = {c["key"]: c for c in res_leg["constructs"]}
    assert by_key["brow_furrow"]["eligibility"] == "待3D重采验证"

    # 短流 (< 10 帧) 仍应给出数值但不允许进入已校准可融合
    df_short = _synthetic_pose_confounded_df(n_clean=6, n_bad=0, seed=5)
    res_short = calibrate_session_blendshapes({"face": df_short})
    assert res_short["available"] is True
    assert res_short["eligible_construct_count"] == 0
    for c in res_short["constructs"]:
        assert c["eligibility"] == "待3D重采验证"


# ── 3. 空日志与异常输入的严格确定性 ───────────────────────────────────────────

@pytest.mark.parametrize(
    "bad_logs",
    [
        None,
        {},
        {"face": None},
        {"face": pd.DataFrame()},
        {"face": pd.DataFrame({"timestamp": [0.0, 1.0], "au12_smile": [0.1, 0.2]})},
        {"face": pd.DataFrame({"bs_jawOpen": [float("nan"), float("nan")]})},
        {
            "face": pd.DataFrame({
                "timestamp": [0.0, 1.0],
                "frame_quality_ok": [0, 0],
                "frame_blur_laplacian": [5.0, 8.0],
                "bs_jawOpen": [0.2, 0.3],
            })
        },
    ],
)
def test_empty_or_unavailable_logs_are_strictly_deterministic(bad_logs):
    r1 = calibrate_session_blendshapes(bad_logs)
    r2 = calibrate_session_blendshapes(bad_logs)
    assert r1 == r2
    assert r1["available"] is False
    assert r1["inference_ms"] == 0.0
    assert r1["eligible_construct_count"] == 0
    json.dumps(r1, allow_nan=False)


# ── 4. 禁止词 AST 守卫 ────────────────────────────────────────────────────────

def test_no_banned_words_in_blendshape_calibration():
    banned_all = (
        "焦虑", "紧张", "压力", "抗压", "情绪稳定", "说谎", "诚信",
        "录用", "人格", "心理画像", "常模", "心理", "情绪", "得分", "评级", "评分",
    )
    path = REPO / "report_frontend" / "blendshape_calibration.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for word in banned_all:
                assert word not in node.value, f"{path}:{node.lineno} 命中红线词 {word}"


# ── 5. 匿名基准脚本与零 PII 校验 ──────────────────────────────────────────────

def test_benchmark_script_anonymizes_and_writes_valid_json(tmp_path):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    df1 = _synthetic_pose_confounded_df(n_clean=30, n_bad=5, seed=1)
    df2 = _synthetic_pose_confounded_df(n_clean=25, n_bad=0, seed=2)
    df1.to_csv(logs_dir / "face_au_log_20261124_010101_aaaa.csv", index=False)
    df2.to_csv(logs_dir / "face_au_log_20261124_020202_bbbb.csv", index=False)

    out_json = tmp_path / "out" / "bench.json"
    rc = bench.main(["--logs-dir", str(logs_dir), "--output", str(out_json), "--quiet"])
    assert rc == 0
    assert out_json.is_file()

    text = out_json.read_text(encoding="utf-8")
    assert "20261124_" not in text
    assert "aaaa" not in text and "bbbb" not in text
    assert not scan_text(text)

    data = json.loads(text)
    assert data["anonymized"] is True
    assert data["corpus_summary"]["total_sessions"] == 2
    assert data["corpus_summary"]["available_sessions"] == 2
    assert [s["session_index"] for s in data["sessions"]] == ["S01", "S02"]
