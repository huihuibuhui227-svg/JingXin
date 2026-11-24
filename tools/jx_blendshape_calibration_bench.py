#!/usr/bin/env python3
"""《衡才智析》面部动作模型系数时序与头姿残差校准语料基准脚本。

遍历历史面部日志（复用 `LogDataLoader` 安全发现各会话文件，仅输出匿名序号 `S01..S55`，
零受试者 PII、零原始文件名或会话编号落盘），批量运行 `calibrate_session_blendshapes`，
汇总校准前后秩相关、头部姿态解释度与可用性分级分布并输出 JSON 基准报告。
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from report_frontend.blendshape_calibration import calibrate_session_blendshapes
from report_frontend.data_loader import LogDataLoader
from report_frontend.face_dual_instrument_audit import CONSTRUCT_PAIRS

DEFAULT_LOGS_DIR = REPO / "data" / "logs"
DEFAULT_OUTPUT_PATH = (
    REPO.parent / "07_三方向与评审筹备" / "系统补强" / "blendshape_calibration_benchmark_20261124.json"
)


def _read_face_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _stats_dict(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "min": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "mean": round(float(np.mean(arr)), 6),
        "median": round(float(np.median(arr)), 6),
        "p25": round(float(np.quantile(arr, 0.25)), 6),
        "p75": round(float(np.quantile(arr, 0.75)), 6),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
    }


def _discover_face_csvs(logs_dir: Path) -> list[Path]:
    with contextlib.redirect_stdout(io.StringIO()):
        loader = LogDataLoader(log_dir=str(logs_dir))
    groups = loader._scan_and_group_files()
    matched = sorted(
        (item["path"] for item in groups.get("face", [])),
        key=lambda p: p.name,
    )
    if matched:
        return matched
    return sorted(Path(logs_dir).glob("face_au_log_*.csv"))


def _build_corpus_summary(session_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(session_rows)
    available_rows = [r["calibration"] for r in session_rows if r["calibration"]["available"]]
    pose_mode_counts = {"3d_matrix": 0, "2d_legacy": 0, "unavailable": 0}
    for row in session_rows:
        pm = row["calibration"].get("pose_mode", "unavailable")
        pose_mode_counts[pm] = pose_mode_counts.get(pm, 0) + 1

    constructs_summary: dict[str, dict[str, Any]] = {}
    overall_elig = {"已校准可融合": 0, "辅助对照": 0, "待3D重采验证": 0}

    for spec in CONSTRUCT_PAIRS:
        raw_rhos: list[float] = []
        cal_rhos: list[float] = []
        deltas: list[float] = []
        pose_r2s: list[float] = []
        elig_counts = {"已校准可融合": 0, "辅助对照": 0, "待3D重采验证": 0}

        for cal in available_rows:
            item = next((c for c in cal["constructs"] if c["key"] == spec.construct), None)
            if item is None:
                continue
            elig = item["eligibility"]
            elig_counts[elig] = elig_counts.get(elig, 0) + 1
            overall_elig[elig] = overall_elig.get(elig, 0) + 1
            if item["n_valid"] < 10:
                continue
            if item["raw_rho"] is not None:
                raw_rhos.append(float(item["raw_rho"]))
            if item["calibrated_rho"] is not None:
                cal_rhos.append(float(item["calibrated_rho"]))
            if item["rho_delta"] is not None:
                deltas.append(float(item["rho_delta"]))
            if item["pose_r2"] is not None:
                pose_r2s.append(float(item["pose_r2"]))

        constructs_summary[spec.construct] = {
            "label": spec.label,
            "n_valid_sessions": len(cal_rhos),
            "raw_rho": _stats_dict(raw_rhos),
            "calibrated_rho": _stats_dict(cal_rhos),
            "rho_delta": _stats_dict(deltas),
            "pose_r2": _stats_dict(pose_r2s),
            "counts_by_eligibility": elig_counts,
        }

    return {
        "total_sessions": total,
        "available_sessions": len(available_rows),
        "unavailable_sessions": total - len(available_rows),
        "pose_mode_counts": pose_mode_counts,
        "constructs_summary": constructs_summary,
        "overall_eligibility_counts": overall_elig,
    }


def run_benchmark(
    logs_dir: Path = DEFAULT_LOGS_DIR,
    output_path: Path | None = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    """批量运行跨会话校准基准测试，并可选写入匿名 JSON 报告。"""
    csv_files = _discover_face_csvs(Path(logs_dir))
    session_rows: list[dict[str, Any]] = []

    for idx, csv_path in enumerate(csv_files, start=1):
        s_idx = f"S{idx:02d}"
        df = _read_face_csv(csv_path)
        cal = calibrate_session_blendshapes({"face": df})
        session_rows.append({
            "session_index": s_idx,
            "n_rows": int(len(df)),
            "calibration": cal,
        })

    payload: dict[str, Any] = {
        "schema_version": "blendshape_calibration_benchmark_v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "anonymized": True,
        "corpus_summary": _build_corpus_summary(session_rows),
        "sessions": session_rows,
    }

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="面部动作模型系数时序与姿态去偏校准语料基准")
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR, help="面部日志目录")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="输出 JSON 路径")
    parser.add_argument("--quiet", action="store_true", help="仅写文件不打印汇总")
    args = parser.parse_args(argv)

    payload = run_benchmark(logs_dir=args.logs_dir, output_path=args.output)
    if not args.quiet:
        cs = payload["corpus_summary"]
        print(
            f"[blendshape_calibration_bench] total={cs['total_sessions']} "
            f"available={cs['available_sessions']} "
            f"unavailable={cs['unavailable_sessions']} "
            f"pose_modes={cs['pose_mode_counts']} "
            f"output={args.output}"
        )
        for key, info in cs["constructs_summary"].items():
            print(
                f"  - {key:16s} ({info['label']}) "
                f"n={info['n_valid_sessions']:2d} "
                f"raw_med={info['raw_rho']['median']} raw_mean={info['raw_rho']['mean']} "
                f"cal_med={info['calibrated_rho']['median']} cal_mean={info['calibrated_rho']['mean']} "
                f"delta_med={info['rho_delta']['median']} "
                f"pose_r2_med={info['pose_r2']['median']} "
                f"elig={info['counts_by_eligibility']}"
            )
        print(f"  overall_eligibility={cs['overall_eligibility_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
