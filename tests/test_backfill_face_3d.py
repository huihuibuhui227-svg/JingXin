"""Tests for tools/jx_backfill_face_3d_telemetry.py."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from face_expression.utils.logger import DataLogger
from tools.jx_pii_guard import scan_text


def _load_tool():
    path = ROOT / "tools" / "jx_backfill_face_3d_telemetry.py"
    spec = importlib.util.spec_from_file_location("jx_backfill_face_3d_telemetry", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


bf3d = _load_tool()

EXPECTED_98_COLS = list(DataLogger(log_type="video", session_id="x").fieldnames)
COLS_92 = EXPECTED_98_COLS[:-6]


def _textured_bgr(mean_val: int = 128, h: int = 64, w: int = 80) -> np.ndarray:
    img = np.full((h, w, 3), mean_val, dtype=np.uint8)
    yy, xx = np.indices((h, w))
    delta = np.where(((yy // 4 + xx // 4) % 2) == 0, 25, -25)
    return np.clip(img.astype(np.int16) + delta[:, :, None], 0, 255).astype(np.uint8)


def _valid_landmarks() -> list[tuple[float, float]]:
    pts = [(0.5, 0.5)] * 478
    pts[10] = (0.50, 0.20)
    pts[152] = (0.50, 0.70)
    return pts


def _make_92_row(sid: str, ts_sec: str, face_scale: str = "0.5000", au12: str = "0.123") -> dict[str, str]:
    row = {c: "" for c in COLS_92}
    row["session_id"] = sid
    row["timestamp"] = ts_sec
    row["au12_smile"] = au12
    row["bs_mouthSmileLeft"] = "0.250"
    row["face_scale"] = face_scale
    return row


def _write_92_csv(csv_path: Path, rows: list[dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLS_92)
        writer.writeheader()
        writer.writerows(rows)


def _write_mock_recording(
    rec_dir: Path,
    sid: str,
    timestamps_ms: list[int],
    *,
    solid_bgr: tuple[int, int, int] | None = None,
) -> None:
    face_dir = rec_dir / "media" / "face"
    face_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for idx, ts_ms in enumerate(timestamps_ms, start=1):
        rel = f"media/face/{idx:06d}.jpg"
        if solid_bgr is not None:
            img = np.full((64, 80, 3), solid_bgr, dtype=np.uint8)
        else:
            img = _textured_bgr(120)
        ok = cv2.imwrite(str(rec_dir / rel), img)
        assert ok
        lines.append(
            json.dumps(
                {
                    "kind": "frame",
                    "modality": "face",
                    "seq": idx,
                    "file": rel,
                    "declared_ts": ts_ms,
                    "session_id": sid,
                }
            )
        )
    (rec_dir / "retention.face.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


class _FakeDetector:
    """Lightweight deterministic detector stub verifying RGB input & missing-pose rows."""

    def __init__(self, *, yaw_base: float = 12.5, skip_ts: set[int] | None = None, no_matrix_ts: set[int] | None = None):
        self.yaw_base = yaw_base
        self.skip_ts = skip_ts or set()
        self.no_matrix_ts = no_matrix_ts or set()
        self.seen_rgb_means: list[tuple[float, float, float]] = []
        self.closed = False

    def detect_full(self, image_rgb: np.ndarray, timestamp_ms: int):
        self.seen_rgb_means.append(tuple(float(x) for x in image_rgb.mean(axis=(0, 1))))
        if timestamp_ms in self.skip_ts:
            return None, {}, None
        lms = _valid_landmarks()
        if timestamp_ms in self.no_matrix_ts:
            return lms, {"jawOpen": 0.1}, None
        pose = {
            "yaw_deg": self.yaw_base + timestamp_ms / 1000.0,
            "pitch_deg": -4.25,
            "roll_deg": 2.75,
        }
        return lms, {"jawOpen": 0.1}, pose

    def close(self) -> None:
        self.closed = True


def test_backfill_92_to_98_preserves_verbatim_and_creates_clean_backup(tmp_path: Path):
    sid = "20261124_010203_abcd"
    logs_dir = tmp_path / "logs"
    rec_root = tmp_path / "recordings"
    backup_dir = tmp_path / "backup"

    # Preserve non-normalized string formatting like "0.0", "0.5000", "0.1230" verbatim.
    orig_rows = [
        _make_92_row(sid, "0.0", face_scale="0.5000", au12="0.1230"),
        _make_92_row(sid, "1.000", face_scale="", au12=""),
        _make_92_row(sid, "2.000", face_scale="0.5000", au12="0.4500"),
    ]
    csv_path = logs_dir / f"face_au_log_{sid}.csv"
    _write_92_csv(csv_path, orig_rows)
    orig_bytes = csv_path.read_bytes()

    # Use fake PII-safe student dir name `<姓名>-000000001-<专业>__<sid>`
    rec_dir = rec_root / f"示例甲-000000001-示例学院__{sid}"
    _write_mock_recording(rec_dir, sid, [0, 1000, 2000])

    # Row 1 (ts=1000) simulates no face; Row 2 (ts=2000) simulates landmarks without 3D matrix.
    created_detectors: list[_FakeDetector] = []

    def _factory():
        d = _FakeDetector(skip_ts={1000}, no_matrix_ts={2000})
        created_detectors.append(d)
        return d

    res = bf3d.backfill_session(
        sid,
        logs_dir=logs_dir,
        recordings_dir=rec_root,
        backup_dir=backup_dir,
        detector_factory=_factory,
    )
    assert res["status"] == "backfilled"
    assert res["n_rows"] == 3
    assert res["n_pose_valid"] == 1
    assert res["n_quality_ok"] == 2
    assert all(d.closed for d in created_detectors)

    # Backup saved ONLY in backup_dir; zero extra files in logs_dir.
    assert sorted(p.name for p in logs_dir.iterdir()) == [f"face_au_log_{sid}.csv"]
    backup_file = backup_dir / f"face_au_log_{sid}.csv"
    assert backup_file.is_file()
    assert backup_file.read_bytes() == orig_bytes

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        got_header = list(reader.fieldnames or [])
        got_rows = list(reader)

    assert len(got_header) == 98
    assert got_header == EXPECTED_98_COLS
    assert len(got_rows) == 3

    # All original 92 columns are preserved verbatim.
    for orig, got in zip(orig_rows, got_rows):
        for col in COLS_92:
            assert got[col] == orig[col], f"Column {col} mutated: {orig[col]!r} -> {got[col]!r}"

    # Row 0 has valid 3D pose & quality_ok=1.
    assert float(got_rows[0]["head_yaw_3d"]) == pytest.approx(12.5)
    assert float(got_rows[0]["head_pitch_3d"]) == pytest.approx(-4.25)
    assert float(got_rows[0]["head_roll_3d"]) == pytest.approx(2.75)
    assert float(got_rows[0]["frame_brightness"]) > 35.0
    assert float(got_rows[0]["frame_blur_laplacian"]) > 15.0
    assert got_rows[0]["frame_quality_ok"] == "1"

    # Row 1 (no face) & Row 2 (missing 3D matrix) must keep empty string "", NEVER fake 0!
    for idx in (1, 2):
        assert got_rows[idx]["head_yaw_3d"] == ""
        assert got_rows[idx]["head_pitch_3d"] == ""
        assert got_rows[idx]["head_roll_3d"] == ""
    assert got_rows[1]["frame_quality_ok"] == "0"
    assert got_rows[2]["frame_quality_ok"] == "1"


def test_backfill_idempotent_skip_and_force_and_preserves_original_backup(tmp_path: Path):
    sid = "20261124_020304_beef"
    logs_dir = tmp_path / "logs"
    rec_root = tmp_path / "recordings"
    backup_dir = tmp_path / "backup"

    _write_92_csv(logs_dir / f"face_au_log_{sid}.csv", [_make_92_row(sid, "0.0")])
    _write_mock_recording(rec_root / sid, sid, [0])

    orig_bytes = (logs_dir / f"face_au_log_{sid}.csv").read_bytes()

    r1 = bf3d.backfill_session(
        sid,
        logs_dir=logs_dir,
        recordings_dir=rec_root,
        backup_dir=backup_dir,
        detector_factory=lambda: _FakeDetector(yaw_base=10.0),
    )
    assert r1["status"] == "backfilled"

    # Second run without --force must skip cleanly without invoking detector.
    called = False

    def _never_called():
        nonlocal called
        called = True
        return _FakeDetector(yaw_base=99.0)

    r2 = bf3d.backfill_session(
        sid,
        logs_dir=logs_dir,
        recordings_dir=rec_root,
        backup_dir=backup_dir,
        detector_factory=_never_called,
    )
    assert r2["status"] == "skipped"
    assert not called

    # With --force, recomputes telemetry while preserving the pre-3D original backup!
    r3 = bf3d.backfill_session(
        sid,
        logs_dir=logs_dir,
        recordings_dir=rec_root,
        backup_dir=backup_dir,
        force=True,
        detector_factory=lambda: _FakeDetector(yaw_base=25.0),
    )
    assert r3["status"] == "backfilled"
    assert (backup_dir / f"face_au_log_{sid}.csv").read_bytes() == orig_bytes

    with open(logs_dir / f"face_au_log_{sid}.csv", "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert float(rows[0]["head_yaw_3d"]) == pytest.approx(25.0)


def test_backfill_converts_bgr_jpeg_to_rgb_and_supports_dry_run_and_webm_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sid = "20261124_030405_cafe"
    logs_dir = tmp_path / "logs"
    rec_root = tmp_path / "recordings"
    backup_dir = tmp_path / "backup"

    _write_92_csv(logs_dir / f"face_au_log_{sid}.csv", [_make_92_row(sid, "0.0")])
    # Solid BGR=(10, 20, 240) -> RGB should be ~(240, 20, 10)
    _write_mock_recording(rec_root / sid, sid, [0], solid_bgr=(10, 20, 240))

    seen_detectors: list[_FakeDetector] = []

    def _factory():
        d = _FakeDetector()
        seen_detectors.append(d)
        return d

    dry_res = bf3d.backfill_session(
        sid,
        logs_dir=logs_dir,
        recordings_dir=rec_root,
        backup_dir=backup_dir,
        dry_run=True,
        detector_factory=_factory,
    )
    assert dry_res["status"] == "dry_run"
    assert not backup_dir.exists()
    with open(logs_dir / f"face_au_log_{sid}.csv", "r", newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f).fieldnames or [])) == 92

    # Verify RGB conversion happened (R channel >> B channel).
    mean_r, mean_g, mean_b = seen_detectors[-1].seen_rgb_means[0]
    assert mean_r > 200.0 and mean_b < 30.0

    # Also test nearest-timestamp alignment + out-of-tolerance gap & webm fallback alignment.
    sid_webm = "20261124_040506_dade"
    _write_92_csv(
        logs_dir / f"face_au_log_{sid_webm}.csv",
        [_make_92_row(sid_webm, "0.2"), _make_92_row(sid_webm, "5.0")],
    )
    webm_dir = rec_root / sid_webm / "media"
    webm_dir.mkdir(parents=True, exist_ok=True)
    (webm_dir / "camera.webm").write_bytes(b"mock-webm")

    class _FakeCap:
        def __init__(self, _path: str):
            self.pos_ms = 0.0

        def isOpened(self) -> bool:
            return True

        def set(self, prop_id: int, value: float) -> bool:
            if prop_id == cv2.CAP_PROP_POS_MSEC:
                self.pos_ms = value
            return True

        def read(self):
            return True, _textured_bgr(110)

        def release(self) -> None:
            pass

    monkeypatch.setattr(bf3d.cv2, "VideoCapture", _FakeCap)
    res_webm = bf3d.backfill_session(
        sid_webm,
        logs_dir=logs_dir,
        recordings_dir=rec_root,
        backup_dir=backup_dir,
        detector_factory=lambda: _FakeDetector(yaw_base=7.0),
    )
    assert res_webm["status"] == "backfilled"
    assert res_webm["source"] == "camera_webm"
    assert res_webm["n_pose_valid"] == 2


def test_nearest_ledger_alignment_and_zero_pii_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    sid = "20261124_050607_eeee"
    logs_dir = tmp_path / "logs"
    rec_root = tmp_path / "recordings"
    backup_dir = tmp_path / "backup"

    # 3 CSV rows vs 2 ledger frames (row at 10.0s exceeds 1.5s tolerance -> empty telemetry)
    rows = [
        _make_92_row(sid, "0.1"),
        _make_92_row(sid, "1.1"),
        _make_92_row(sid, "10.0"),
    ]
    _write_92_csv(logs_dir / f"face_au_log_{sid}.csv", rows)
    rec_dir = rec_root / f"示例乙-000000002-计算机__{sid}"
    _write_mock_recording(rec_dir, sid, [0, 1000])

    res = bf3d.backfill_session(
        sid,
        logs_dir=logs_dir,
        recordings_dir=rec_root,
        backup_dir=backup_dir,
        detector_factory=lambda: _FakeDetector(yaw_base=5.0),
    )
    assert res["status"] == "backfilled"
    assert res["n_aligned"] == 2
    assert res["n_pose_valid"] == 2

    with open(logs_dir / f"face_au_log_{sid}.csv", "r", newline="", encoding="utf-8") as f:
        got = list(csv.DictReader(f))
    assert float(got[0]["head_yaw_3d"]) == pytest.approx(5.0)
    assert float(got[1]["head_yaw_3d"]) == pytest.approx(6.0)
    assert got[2]["head_yaw_3d"] == ""
    assert got[2]["frame_quality_ok"] == ""

    # CLI run on --all-matched prints zero PII
    rc = bf3d.main([
        "--all-matched",
        "--logs-dir", str(logs_dir),
        "--recordings-dir", str(rec_root),
        "--backup-dir", str(backup_dir),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "000000002" not in out
    assert "示例乙" not in out
    assert sid in out

    # PII guard clean on both newly created files.
    for rel in ("tools/jx_backfill_face_3d_telemetry.py", "tests/test_backfill_face_3d.py"):
        assert scan_text((ROOT / rel).read_text(encoding="utf-8")) == []


def test_resolve_crop_reads_manifest_and_probes_ladder_by_face_scale(tmp_path: Path):
    sid_man = "20261124_060708_aaaa"
    logs_dir = tmp_path / "logs"
    rec_root = tmp_path / "recordings"
    backup_dir = tmp_path / "backup"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / ".face_backfill_manifest.json").write_text(
        json.dumps({
            sid_man: {
                "status": "rebuilt",
                "crop": "center50",
                "crop_box": [0.25, 0.0, 0.75, 0.75],
            }
        }),
        encoding="utf-8",
    )

    # Image is 160x120; center50 crops to 80x90 (w=80). Detector only triggers on cropped width <= 100.
    _write_92_csv(logs_dir / f"face_au_log_{sid_man}.csv", [_make_92_row(sid_man, "0.0", face_scale="0.5000")])
    _write_mock_recording(rec_root / sid_man, sid_man, [0])
    # Re-write frame as 160x120 so full width=160 and center50 width=80
    cv2.imwrite(str(rec_root / sid_man / "media" / "face" / "000001.jpg"), _textured_bgr(120, h=120, w=160))

    class _WidthSensitiveDetector:
        def detect_full(self, image_rgb: np.ndarray, timestamp_ms: int):
            if image_rgb.shape[1] > 100:
                return None, {}, None
            return _valid_landmarks(), {}, {"yaw_deg": 4.0, "pitch_deg": 1.0, "roll_deg": -1.0}

        def close(self) -> None:
            pass

    res_man = bf3d.backfill_session(
        sid_man,
        logs_dir=logs_dir,
        recordings_dir=rec_root,
        backup_dir=backup_dir,
        detector_factory=_WidthSensitiveDetector,
    )
    assert res_man["status"] == "backfilled"
    assert res_man["crop"] == "center50"
    assert res_man["n_pose_valid"] == 1

    # Without manifest (or status=failed), ladder probing discovers center50 via face_scale=0.5000!
    sid_probe = "20261124_060709_bbbb"
    _write_92_csv(logs_dir / f"face_au_log_{sid_probe}.csv", [_make_92_row(sid_probe, "0.0", face_scale="0.5000")])
    _write_mock_recording(rec_root / sid_probe, sid_probe, [0])
    cv2.imwrite(str(rec_root / sid_probe / "media" / "face" / "000001.jpg"), _textured_bgr(120, h=120, w=160))

    res_probe = bf3d.backfill_session(
        sid_probe,
        logs_dir=logs_dir,
        recordings_dir=rec_root,
        backup_dir=backup_dir,
        detector_factory=_WidthSensitiveDetector,
    )
    assert res_probe["status"] == "backfilled"
    assert res_probe["crop"] == "center50"
    assert res_probe["n_pose_valid"] == 1

