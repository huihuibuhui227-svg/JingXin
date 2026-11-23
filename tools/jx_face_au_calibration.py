#!/usr/bin/env python3
"""《衡才智析》面部双仪器一致性批量核验与标定诊断脚本。

遍历 `data/logs/face_au_log_*.csv`（仅使用匿名序号 `S01..S55`，零受试者 PII、零原始会话号落盘），
批量调用 `audit_face_dual_instrument` 与 `compute_blendshape_indicators`，并支持通过
`--external-csv` 对接外部 OpenFace / Py-Feat FACS AU CSV 进行交叉校验。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from report_frontend.blendshape_second_instrument import compute_blendshape_indicators
from report_frontend.face_dual_instrument_audit import (
    CONSTRUCT_PAIRS,
    audit_face_dual_instrument,
    compare_external_facs_csv,
)

DEFAULT_LOGS_DIR = REPO / "data" / "logs"
DEFAULT_OUTPUT_PATH = (
    REPO.parent / "07_三方向与评审筹备" / "系统补强" / "face_dual_instrument_audit_20261124.json"
)


def _read_face_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _stats_dict(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "median": None,
            "mean": None,
            "p25": None,
            "p75": None,
            "min": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "median": round(float(np.median(arr)), 6),
        "mean": round(float(np.mean(arr)), 6),
        "p25": round(float(np.quantile(arr, 0.25)), 6),
        "p75": round(float(np.quantile(arr, 0.75)), 6),
        "min": round(float(np.min(arr)), 6),
        "max": round(float(np.max(arr)), 6),
    }


def _build_corpus_summary(session_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(session_rows)
    available_audits = [r["dual_instrument_audit"] for r in session_rows if r["dual_instrument_audit"]["available"]]

    pairs_summary: dict[str, dict[str, Any]] = {}
    for spec in CONSTRUCT_PAIRS:
        levels = {"高度一致": 0, "中度一致": 0, "分歧较大": 0, "通道未激活": 0}
        rhos: list[float] = []
        raw_rhos: list[float] = []
        drhos: list[float] = []
        geo_floors: list[float] = []
        bs_floors: list[float] = []

        for audit in available_audits:
            pair = next((p for p in audit["pairs"] if p["construct"] == spec.construct), None)
            if pair is None:
                continue
            lvl = pair["agreement_level"]
            levels[lvl] = levels.get(lvl, 0) + 1
            if pair["spearman_rho"] is not None:
                rhos.append(float(pair["spearman_rho"]))
            if pair["raw_spearman_rho"] is not None:
                raw_rhos.append(float(pair["raw_spearman_rho"]))
            if pair["delta_spearman_rho"] is not None:
                drhos.append(float(pair["delta_spearman_rho"]))
            if pair["geo_floor_rate"] is not None:
                geo_floors.append(float(pair["geo_floor_rate"]))
            if pair["bs_floor_rate"] is not None:
                bs_floors.append(float(pair["bs_floor_rate"]))

        rho_stats = _stats_dict(rhos)
        raw_rho_stats = _stats_dict(raw_rhos)
        drho_stats = _stats_dict(drhos)
        pairs_summary[spec.construct] = {
            "label": spec.label,
            "geometric_col": spec.geometric_col,
            "blendshape_cols": list(spec.blendshape_cols),
            "geo_polarity": spec.geo_polarity,
            "counts_by_level": levels,
            "n_valid_sessions_for_rho": len(rhos),
            "spearman_rho": rho_stats,
            "raw_spearman_rho": {
                "median": raw_rho_stats["median"],
                "mean": raw_rho_stats["mean"],
            },
            "delta_spearman_rho": {
                "median": drho_stats["median"],
                "mean": drho_stats["mean"],
                "p25": drho_stats["p25"],
                "p75": drho_stats["p75"],
            },
            "geo_floor_rate_median": _stats_dict(geo_floors)["median"],
            "bs_floor_rate_median": _stats_dict(bs_floors)["median"],
        }

    # 第二仪器 6 项指标跨场统计
    second_inst_acc: dict[str, dict[str, Any]] = {}
    for row in session_rows:
        for ind in row["second_instrument_indicators"]:
            name = ind["name"]
            if name not in second_inst_acc:
                second_inst_acc[name] = {
                    "label": ind["label"],
                    "unit": ind["unit"],
                    "geometric_counterpart": ind["geometric_counterpart"],
                    "thresholds": [],
                    "values": [],
                }
            if ind["value"] is not None:
                second_inst_acc[name]["values"].append(float(ind["value"]))
            if ind["threshold_used"] is not None:
                second_inst_acc[name]["thresholds"].append(float(ind["threshold_used"]))

    second_instrument_summary: dict[str, dict[str, Any]] = {}
    for name, info in second_inst_acc.items():
        vals = info["values"]
        stats = _stats_dict(vals)
        second_instrument_summary[name] = {
            "label": info["label"],
            "unit": info["unit"],
            "geometric_counterpart": info["geometric_counterpart"],
            "threshold_used": info["thresholds"][0] if info["thresholds"] else None,
            "n_valid_sessions": len(vals),
            "n_nonzero_sessions": sum(1 for v in vals if abs(v) > 1e-12),
            **stats,
        }

    # AU15 俯仰角回归残差诊断汇总
    au15_diags = [
        audit["au15_pitch_diagnostic"]
        for audit in available_audits
        if isinstance(audit.get("au15_pitch_diagnostic"), dict)
        and audit["au15_pitch_diagnostic"].get("available")
    ]
    raw_floors = [float(d["raw_floor_rate"]) for d in au15_diags if d.get("raw_floor_rate") is not None]
    pitch_rhos = [float(d["au15_pitch_spearman_rho"]) for d in au15_diags if d.get("au15_pitch_spearman_rho") is not None]
    r2s = [float(d["pitch_r_squared"]) for d in au15_diags if d.get("pitch_r_squared") is not None]
    bs_frown_floors = [float(d["bs_frown_floor_rate"]) for d in au15_diags if d.get("bs_frown_floor_rate") is not None]
    raw_bs_rhos = [float(d["raw_bs_frown_spearman_rho"]) for d in au15_diags if d.get("raw_bs_frown_spearman_rho") is not None]
    res_bs_rhos = [float(d["residual_bs_frown_spearman_rho"]) for d in au15_diags if d.get("residual_bs_frown_spearman_rho") is not None]

    raw_floor_stats = _stats_dict(raw_floors)
    pitch_rho_stats = _stats_dict(pitch_rhos)
    r2_stats = _stats_dict(r2s)
    res_bs_stats = _stats_dict(res_bs_rhos)
    raw_bs_stats = _stats_dict(raw_bs_rhos)

    au15_pitch_diagnostic_summary = {
        "n_sessions_available": len(au15_diags),
        "raw_floor_rate_median": raw_floor_stats["median"],
        "raw_floor_rate_mean": raw_floor_stats["mean"],
        "au15_pitch_spearman_rho_median": pitch_rho_stats["median"],
        "au15_pitch_spearman_rho_max": pitch_rho_stats["max"],
        "pitch_r_squared_median": r2_stats["median"],
        "pitch_r_squared_max": r2_stats["max"],
        "bs_frown_floor_rate_median": _stats_dict(bs_frown_floors)["median"],
        "raw_bs_frown_spearman_rho_median": raw_bs_stats["median"],
        "residual_bs_frown_spearman_rho_median": res_bs_stats["median"],
        "conclusion_plain": (
            f"在 {len(au15_diags)} 场有效诊断会话中，原始嘴角下压几何列零值贴底中位占比 "
            f"{(raw_floor_stats['median'] or 0.0):.1%}，与俯仰角代理相关中位数 "
            f"{pitch_rho_stats['median']}；回归扣除俯仰角分量后，残差与模型皱嘴角系数相关中位数 "
            f"{res_bs_stats['median']}，仍维持阻塞状态。"
        ),
    }

    return {
        "total_sessions": total,
        "available_audit_sessions": len(available_audits),
        "unavailable_audit_sessions": total - len(available_audits),
        "pairs_summary": pairs_summary,
        "second_instrument_summary": second_instrument_summary,
        "au15_pitch_diagnostic_summary": au15_pitch_diagnostic_summary,
    }


def run_calibration(
    logs_dir: Path = DEFAULT_LOGS_DIR,
    output_path: Path | None = DEFAULT_OUTPUT_PATH,
    external_csv: Path | None = None,
    target_index: str | None = None,
) -> dict[str, Any]:
    """批量运行面部双仪器一致性审计，并可选写入匿名 JSON 报告。"""
    csv_files = sorted(Path(logs_dir).glob("face_au_log_*.csv"))

    session_rows: list[dict[str, Any]] = []
    dfs_by_index: dict[str, pd.DataFrame] = {}

    for idx, csv_path in enumerate(csv_files, start=1):
        s_idx = f"S{idx:02d}"
        df = _read_face_csv(csv_path)
        dfs_by_index[s_idx] = df
        bs_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("bs_")]
        audit = audit_face_dual_instrument({"face": df})
        second_inds = compute_blendshape_indicators(df)
        session_rows.append({
            "session_index": s_idx,
            "n_rows": int(len(df)),
            "has_bs_columns": bool(bs_cols),
            "dual_instrument_audit": audit,
            "second_instrument_indicators": second_inds,
        })

    external_validation = None
    if external_csv is not None:
        ext_path = Path(external_csv)
        chosen_idx = target_index
        if chosen_idx is None:
            chosen_idx = next(
                (r["session_index"] for r in session_rows if r["dual_instrument_audit"]["available"]),
                session_rows[0]["session_index"] if session_rows else None,
            )
        if chosen_idx is not None and chosen_idx in dfs_by_index:
            external_validation = {
                "target_index": chosen_idx,
                "result": compare_external_facs_csv(dfs_by_index[chosen_idx], ext_path),
            }
        else:
            external_validation = {
                "target_index": chosen_idx,
                "result": {
                    "available": False,
                    "summary_plain": "未找到对应匿名场次用于外部动作单元校验。",
                },
            }

    payload: dict[str, Any] = {
        "schema_version": "face_dual_instrument_audit_v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "anonymized": True,
        "corpus_summary": _build_corpus_summary(session_rows),
        "external_facs_validation": external_validation,
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
    parser = argparse.ArgumentParser(description="面部双仪器一致性批量核验与 AU15 头姿残差诊断")
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR, help="面部日志目录")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="输出 JSON 路径")
    parser.add_argument("--external-csv", type=Path, default=None, help="可选外部 OpenFace / Py-Feat CSV 路径")
    parser.add_argument("--target-index", type=str, default=None, help="与 --external-csv 配对的匿名场次编号如 S01")
    parser.add_argument("--quiet", action="store_true", help="仅写文件不打印汇总表")
    args = parser.parse_args(argv)

    payload = run_calibration(
        logs_dir=args.logs_dir,
        output_path=args.output,
        external_csv=args.external_csv,
        target_index=args.target_index,
    )
    if not args.quiet:
        cs = payload["corpus_summary"]
        print(
            f"[face_dual_instrument] total={cs['total_sessions']} "
            f"available={cs['available_audit_sessions']} "
            f"unavailable={cs['unavailable_audit_sessions']} "
            f"output={args.output}"
        )
        for construct, info in cs["pairs_summary"].items():
            print(
                f"  - {construct:18s} ({info['label']}) "
                f"med_rho={info['spearman_rho']['median']} "
                f"raw_med_rho={info['raw_spearman_rho']['median']} "
                f"med_drho={info['delta_spearman_rho']['median']} "
                f"levels={info['counts_by_level']}"
            )
        au15 = cs["au15_pitch_diagnostic_summary"]
        print(
            f"  - au15_pitch_diag: n={au15['n_sessions_available']} "
            f"floor_med={au15['raw_floor_rate_median']} "
            f"pitch_rho_med={au15['au15_pitch_spearman_rho_median']} "
            f"r2_med={au15['pitch_r_squared_median']} "
            f"res_bs_rho_med={au15['residual_bs_frown_spearman_rho_median']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
