# tests/test_face_dual_instrument.py
"""面部双仪器对照、第二仪器内存指标修复、AU15 俯仰角残差诊断与批量核验脚本测试。"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from report_frontend import blendshape_second_instrument as bsi
from report_frontend import face_dual_instrument_audit as fda
from report_frontend.l1_report import build_l1_report
from tools import jx_face_au_calibration as calib


def _synthetic_face_df(n: int = 40, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float) * 0.5
    signal_eye = np.linspace(0.0, 1.0, n) + rng.normal(0.0, 0.03, n)
    signal_smile = np.sin(np.linspace(0.0, 3.0 * np.pi, n)) * 0.3 + 0.35
    signal_mouth = np.abs(np.cos(np.linspace(0.0, 2.0 * np.pi, n))) * 0.4
    signal_press = np.linspace(0.02, 0.45, n)
    signal_brow = np.linspace(0.01, 0.25, n)
    pitch = np.linspace(-0.6, -0.2, n)

    # au4_frown 与 au23_lip_compression 为物理距离比值，越小越收缩/越压紧
    au4 = 0.34 - 0.15 * signal_brow + rng.normal(0.0, 0.005, n)
    au23 = 0.07 - 0.06 * signal_press + rng.normal(0.0, 0.002, n)

    # au12 为中度相关，加入少量扰动
    bs_smile = np.clip(0.55 * signal_smile + rng.normal(0.0, 0.12, n), 0.0, 1.0)
    # au15 受 head_pitch 强混杂
    au15_latent = 0.65 * (pitch + 0.6) + 0.05 * rng.normal(0.0, 0.2, n)
    au15 = np.clip(au15_latent, 0.0, 1.0)

    blink = np.full(n, 0.1, dtype=float)
    blink[5:7] = 0.85
    blink[20:22] = 0.85

    return pd.DataFrame({
        "timestamp": t,
        "au7_eye_squeeze": np.clip(signal_eye, 0.0, 1.0),
        "au12_smile": np.clip(signal_smile, 0.0, 1.0),
        "au25_mouth_open": np.clip(signal_mouth, 0.0, 1.0),
        "au23_lip_compression": au23,
        "au4_frown": au4,
        "au15_mouth_down": au15,
        "head_pitch": pitch,
        "bs__neutral": np.full(n, 0.8),
        "bs_eyeSquintLeft": np.clip(signal_eye, 0.0, 1.0),
        "bs_eyeSquintRight": np.clip(signal_eye, 0.0, 1.0),
        "bs_mouthSmileLeft": bs_smile,
        "bs_mouthSmileRight": bs_smile,
        "bs_jawOpen": np.clip(signal_mouth * 0.9, 0.0, 1.0),
        "bs_mouthPressLeft": signal_press,
        "bs_mouthPressRight": signal_press,
        "bs_browDownLeft": signal_brow,
        "bs_browDownRight": signal_brow,
        "bs_eyeBlinkLeft": blink,
        "bs_eyeBlinkRight": blink,
        "bs_mouthFrownLeft": np.clip(rng.uniform(0.0, 0.05, n), 0.0, 1.0),
        "bs_mouthFrownRight": np.clip(rng.uniform(0.0, 0.05, n), 0.0, 1.0),
    })


# ── 1. blendshape_second_instrument.compute_blendshape_indicators ─────────────

def test_compute_blendshape_indicators_matches_csv_api_and_includes_extra_channels(tmp_path):
    df = _synthetic_face_df(30)
    csv_path = tmp_path / "face_au_log_20261124_000000_abcd.csv"
    df.to_csv(csv_path, index=False)

    indicators = bsi.compute_blendshape_indicators(df)
    assert len(indicators) == 6
    expected_keys = {
        "name", "label", "value", "unit", "n_valid",
        "threshold_used", "geometric_counterpart", "comparable_note",
    }
    by_name = {}
    for item in indicators:
        assert set(item.keys()) == expected_keys
        assert "(" not in item["comparable_note"] and "（" not in item["comparable_note"]
        by_name[item["name"]] = item

    # 与既有 compute_second_instrument 逐项精确对齐
    csv_results = bsi.compute_all(csv_path)
    for c in bsi.CONSTRUCTS:
        assert by_name[c]["value"] == pytest.approx(csv_results[c]["blendshape_value"])
        assert by_name[c]["n_valid"] == csv_results[c]["n_frames"]
        assert by_name[c]["threshold_used"] == csv_results[c]["threshold_used"]

    # 新增 2 条高活跃通道：jaw_open_p50 与 mouth_press_ratio
    assert by_name["jaw_open_p50"]["value"] == pytest.approx(float(np.median(df["bs_jawOpen"])))
    assert by_name["jaw_open_p50"]["threshold_used"] is None
    assert by_name["mouth_press_ratio"]["threshold_used"] == pytest.approx(0.25)
    expected_press = float((((df["bs_mouthPressLeft"] + df["bs_mouthPressRight"]) / 2.0) > 0.25).mean())
    assert by_name["mouth_press_ratio"]["value"] == pytest.approx(expected_press)

    # 严格合法 JSON（无 NaN）
    json.dumps(indicators, allow_nan=False)


def test_compute_blendshape_indicators_handles_partial_missing_and_bilateral_nan():
    df = pd.DataFrame({
        "bs_jawOpen": [0.1, 0.2, float("nan"), 0.5],
        "bs_mouthPressLeft": [0.4, 0.4, 0.1, 0.1],
        "bs_mouthPressRight": [0.4, float("nan"), 0.1, 0.1],  # 第 1 行单侧缺失应从分母剔除
    })
    indicators = bsi.compute_blendshape_indicators(df)
    by_name = {d["name"]: d for d in indicators}

    assert by_name["jaw_open_p50"]["value"] == pytest.approx(0.2)
    assert by_name["jaw_open_p50"]["n_valid"] == 3
    # 有效配对仅 3 帧（0.4, 0.1, 0.1），>0.25 占 1/3
    assert by_name["mouth_press_ratio"]["n_valid"] == 3
    assert by_name["mouth_press_ratio"]["value"] == pytest.approx(1.0 / 3.0)
    # 缺失时间戳和其它列时，value 必须为 None 而非 NaN
    assert by_name["blink_rate_per_min"]["value"] is None
    assert by_name["smile_ratio"]["value"] is None
    json.dumps(indicators, allow_nan=False)


def test_compute_blendshape_indicators_empty_or_all_nan_returns_empty():
    assert bsi.compute_blendshape_indicators(pd.DataFrame()) == []
    df_nan = pd.DataFrame({"bs_jawOpen": [float("nan"), float("nan")]})
    assert bsi.compute_blendshape_indicators(df_nan) == []


def test_l1_report_second_instrument_appendix_is_now_wired():
    df = _synthetic_face_df(30)
    payload = build_l1_report("20261124_000000_abcd", logs={"face": df})
    sec = payload["l1"]["second_instrument_appendix"]
    assert sec["available"] is True
    assert len(sec["rows"]) == 6
    names = [r["name"] for r in sec["rows"]]
    assert "jaw_open_p50" in names and "mouth_press_ratio" in names
    json.dumps(sec, allow_nan=False)


# ── 2. face_dual_instrument_audit.audit_face_dual_instrument ─────────────────

@pytest.mark.parametrize("bad_logs", [
    None,
    {},
    {"face": None},
    {"face": pd.DataFrame()},
    {"face": pd.DataFrame({"timestamp": [0.0, 1.0], "au12_smile": [0.1, 0.2]})},
])
def test_audit_missing_or_no_bs_returns_exact_unavailable_shape(bad_logs):
    got = fda.audit_face_dual_instrument(bad_logs)
    assert got == {
        "available": False,
        "summary_plain": "当前场次未包含面部模型系数列，未开展双仪器一致性对照。",
        "pairs": [],
        "au15_pitch_diagnostic": None,
    }


def test_audit_all_nan_bs_returns_unavailable():
    df = pd.DataFrame({
        "au7_eye_squeeze": [0.2, 0.3],
        "bs_eyeSquintLeft": [float("nan"), float("nan")],
    })
    got = fda.audit_face_dual_instrument({"face": df})
    assert got["available"] is False
    assert got["pairs"] == []
    assert got["au15_pitch_diagnostic"] is None


def test_audit_agreement_levels_polarity_and_au15_diagnostic():
    n = 30
    idx = np.arange(n, dtype=float)
    # 1) eye_squeeze: 高度一致 (rho >= 0.7)，非线性序列以使帧间差分同样有方差
    eye_g = 0.5 + 0.35 * np.sin(idx * 0.45) + 0.005 * idx
    eye_b = eye_g.copy()
    # 2) smile: 中度一致 (0.4 <= rho < 0.7)
    rng = np.random.default_rng(7)
    smile_g = np.linspace(0.0, 1.0, n)
    smile_b = np.clip(0.55 * smile_g + rng.normal(0.0, 0.22, n), 0.01, 1.0)
    r_smile = float(spearmanr(smile_g, smile_b).statistic)
    assert 0.4 <= r_smile < 0.7

    # 3) mouth_open: 分歧较大 (rho < 0.4，反向构造)
    mouth_g = np.linspace(0.1, 0.9, n)
    mouth_b = np.linspace(0.9, 0.1, n)

    # 4) lip_compression: 物理方向反转 (au23 越小越压紧, bs_mouthPress 越大越压紧 -> 高度一致)
    press_b = np.linspace(0.05, 0.45, n)
    press_g = 0.08 - 0.05 * press_b

    # 5) brow_furrow: 通道未激活 (>95% 贴底)
    brow_b = np.zeros(n, dtype=float)
    brow_b[-1] = 0.2  # 29/30 = 96.7% 贴底
    brow_g = np.linspace(0.28, 0.33, n)

    # AU15: 受 pitch 线性混杂 + 贴底
    pitch = np.linspace(-0.7, -0.3, n)
    au15 = np.clip(0.8 * (pitch + 0.55), 0.0, 1.0)
    bs_frown = np.clip(0.02 + 0.01 * np.sin(idx), 0.0, 1.0)

    df = pd.DataFrame({
        "au7_eye_squeeze": eye_g,
        "bs_eyeSquintLeft": eye_b,
        "bs_eyeSquintRight": eye_b,
        "au12_smile": smile_g,
        "bs_mouthSmileLeft": smile_b,
        "bs_mouthSmileRight": smile_b,
        "au25_mouth_open": mouth_g,
        "bs_jawOpen": mouth_b,
        "au23_lip_compression": press_g,
        "bs_mouthPressLeft": press_b,
        "bs_mouthPressRight": press_b,
        "au4_frown": brow_g,
        "bs_browDownLeft": brow_b,
        "bs_browDownRight": brow_b,
        "au15_mouth_down": au15,
        "head_pitch": pitch,
        "bs_mouthFrownLeft": bs_frown,
        "bs_mouthFrownRight": bs_frown,
    })

    got = fda.audit_face_dual_instrument({"face": df})
    assert got["available"] is True
    assert "(" not in got["summary_plain"] and "（" not in got["summary_plain"]

    by_c = {p["construct"]: p for p in got["pairs"]}
    assert by_c["eye_squeeze"]["agreement_level"] == "高度一致"
    assert by_c["eye_squeeze"]["spearman_rho"] == pytest.approx(1.0)
    assert by_c["eye_squeeze"]["delta_spearman_rho"] == pytest.approx(1.0)

    assert by_c["smile"]["agreement_level"] == "中度一致"
    assert by_c["smile"]["spearman_rho"] == pytest.approx(r_smile, abs=1e-5)

    assert by_c["mouth_open"]["agreement_level"] == "分歧较大"
    assert by_c["mouth_open"]["spearman_rho"] == pytest.approx(-1.0)

    # au23_lip_compression：raw 为 -1.0，方向对齐后为 +1.0 -> 高度一致
    assert by_c["lip_compression"]["raw_spearman_rho"] == pytest.approx(-1.0)
    assert by_c["lip_compression"]["spearman_rho"] == pytest.approx(1.0)
    assert by_c["lip_compression"]["agreement_level"] == "高度一致"

    # au4_frown：模型通道 96.7% 贴底 -> 通道未激活
    assert by_c["brow_furrow"]["agreement_level"] == "通道未激活"
    assert by_c["brow_furrow"]["bs_floor_rate"] > 0.95

    diag = got["au15_pitch_diagnostic"]
    assert diag is not None
    assert diag["available"] is True
    assert diag["diagnostic_only"] is True
    assert diag["status"] == "diagnostic_only"
    assert diag["raw_floor_rate"] > 0.3
    assert diag["au15_pitch_spearman_rho"] > 0.8
    assert diag["pitch_r_squared"] > 0.7
    assert "诊断参考项" in diag["diagnostic_note"]
    assert "(" not in diag["diagnostic_note"] and "（" not in diag["diagnostic_note"]

    json.dumps(got, allow_nan=False)


def test_audit_short_sequence_is_inactive_with_null_rho():
    df = _synthetic_face_df(8)
    got = fda.audit_face_dual_instrument({"face": df})
    assert got["available"] is True
    for p in got["pairs"]:
        assert p["agreement_level"] == "通道未激活"
        assert p["spearman_rho"] is None
        assert p["delta_spearman_rho"] is None
    assert got["au15_pitch_diagnostic"]["available"] is False


# ── 3. 外部 FACS CSV 对接适配器 ───────────────────────────────────────────────

def test_compare_external_facs_csv_openface_and_pyfeat_formats(tmp_path):
    df = _synthetic_face_df(25)

    # OpenFace 风格列名（带前导空格和 _r 后缀）
    ext_openface = pd.DataFrame({
        " AU04_r": df["bs_browDownLeft"],
        " AU12_r": df["bs_mouthSmileLeft"],
        " AU25_r": df["bs_jawOpen"],
    })
    ext_csv = tmp_path / "openface_out.csv"
    ext_openface.to_csv(ext_csv, index=False)

    res = fda.compare_external_facs_csv(df, ext_csv)
    assert res["available"] is True
    assert res["alignment_mode"] == "row_index"
    assert set(res["matched_columns"]) == {"AU04_r", "AU12_r", "AU25_r"}
    by_au = {c["canonical_au"]: c for c in res["comparisons"]}
    assert by_au["AU12_r"]["blendshape_comparison"]["spearman_rho"] == pytest.approx(1.0)
    assert by_au["AU25_r"]["blendshape_comparison"]["spearman_rho"] > 0.99
    assert by_au["AU04_r"]["blendshape_comparison"]["spearman_rho"] == pytest.approx(1.0)

    # Py-Feat 风格 + 不同长度按 timestamp 对齐
    ext_pyfeat = pd.DataFrame({
        "timestamp": df["timestamp"].iloc[::2].to_numpy() + 1000.0,  # 偏移测试相对时间对齐
        "AU04": df["bs_browDownLeft"].iloc[::2].to_numpy(),
        "AU12": df["bs_mouthSmileLeft"].iloc[::2].to_numpy(),
        "AU25": df["bs_jawOpen"].iloc[::2].to_numpy(),
    })
    res_pf = fda.compare_external_facs_csv(df, ext_pyfeat)
    assert res_pf["available"] is True
    assert res_pf["alignment_mode"] == "timestamp_asof"
    json.dumps(res_pf, allow_nan=False)

    assert fda.compare_external_facs_csv(df, None)["available"] is False
    with pytest.raises(FileNotFoundError):
        fda.compare_external_facs_csv(df, tmp_path / "missing.csv")


# ── 4. tools/jx_face_au_calibration.py ──────────────────────────────────────

def test_batch_calibration_tool_anonymized_and_external_csv(tmp_path):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    df1 = _synthetic_face_df(30, seed=1)
    df2 = _synthetic_face_df(25, seed=2)
    df3 = pd.DataFrame({"timestamp": [0.0, 1.0], "au12_smile": [0.1, 0.2]})  # 无 bs_*

    # 刻意写入含时间戳和随机段的真实命名形态，验证输出绝不泄露原文件名/会话号
    df1.to_csv(logs_dir / "face_au_log_20261124_010101_aaaa.csv", index=False)
    df2.to_csv(logs_dir / "face_au_log_20261124_020202_bbbb.csv", index=False)
    df3.to_csv(logs_dir / "face_au_log_20261124_030303_cccc.csv", index=False)

    ext_csv = tmp_path / "ext_facs.csv"
    pd.DataFrame({
        "AU12_r": df1["bs_mouthSmileLeft"],
        "AU25_r": df1["bs_jawOpen"],
    }).to_csv(ext_csv, index=False)

    out_json = tmp_path / "out" / "audit.json"
    rc = calib.main([
        "--logs-dir", str(logs_dir),
        "--output", str(out_json),
        "--external-csv", str(ext_csv),
        "--target-index", "S01",
        "--quiet",
    ])
    assert rc == 0
    assert out_json.is_file()

    raw_text = out_json.read_text(encoding="utf-8")
    assert "20261124_" not in raw_text
    assert "aaaa" not in raw_text and "bbbb" not in raw_text and "cccc" not in raw_text

    data = json.loads(raw_text)
    assert data["anonymized"] is True
    assert data["corpus_summary"]["total_sessions"] == 3
    assert data["corpus_summary"]["available_audit_sessions"] == 2
    assert data["corpus_summary"]["unavailable_audit_sessions"] == 1
    assert [s["session_index"] for s in data["sessions"]] == ["S01", "S02", "S03"]
    assert data["external_facs_validation"]["target_index"] == "S01"
    assert data["external_facs_validation"]["result"]["available"] is True


# ── 5. 严格红线禁用词 AST 守卫 ───────────────────────────────────────────────

def test_no_banned_words_in_new_and_edited_modules():
    banned_all = (
        "焦虑", "紧张", "压力", "抗压", "情绪稳定", "说谎", "诚信",
        "录用", "人格", "心理画像", "常模", "心理", "情绪", "得分", "评级", "评分",
    )
    for rel in (
        "report_frontend/blendshape_second_instrument.py",
        "report_frontend/face_dual_instrument_audit.py",
    ):
        path = REPO / rel
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for word in banned_all:
                    assert word not in node.value, f"{rel}:{node.lineno} 命中红线词 {word}"
