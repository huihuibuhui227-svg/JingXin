# report_frontend/parameter_provenance.py
"""参数来源三级登记加载器与汇总视图（激活 `provisional_flags.py`）。

从单一事实源 `parameter_registry.json` 读取全链路参数，给每条 T3 临时参数
调用 `provisional_flags.mark_provisional()` 打上标准未标定标签，生成顶层
`l1.parameter_provenance` 载荷供 HTML 报告与前端同源消费。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import provisional_flags

REGISTRY_PATH = Path(__file__).resolve().parent / "parameter_registry.json"


@lru_cache(maxsize=1)
def load_registry() -> dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def build_parameter_provenance(indicator_names: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """构建顶层 `l1.parameter_provenance` 块。

    - 诚实统计 T1 / T2a / T2b / T3a / T3b 各档参数数量（T2b 自采标定当前实报 0 条）；
    - 为所有 `provisional=True` 的数值参数通过 `provisional_flags.mark_provisional` 校验依据；
    - 生成 `by_indicator` 映射供前端在指标明细行挂来源徽章。
    """
    raw = load_registry()
    params = raw.get("parameters") or []
    counts = {"T1": 0, "T2a": 0, "T2b": 0, "T3a": 0, "T3b": 0}
    provisional_marks: list[dict[str, Any]] = []
    registry_out: list[dict[str, Any]] = []
    by_indicator: dict[str, list[str]] = {n: [] for n in indicator_names}

    for p in params:
        sub = str(p.get("sub_tier") or p.get("tier") or "T3a")
        if sub in counts:
            counts[sub] += 1
        entry = {
            "param_id": p["param_id"],
            "label": p.get("label", p["param_id"]),
            "value": p["value"],
            "unit": p.get("unit", ""),
            "tier": p.get("tier", "T3"),
            "sub_tier": sub,
            "provisional": bool(p.get("provisional", False)),
            "code_locations": list(p.get("code_locations") or []),
            "affects_indicators": list(p.get("affects_indicators") or []),
            "provenance": p.get("provenance", ""),
            "upgrade_condition": p.get("upgrade_condition", ""),
        }
        if entry["provisional"] and isinstance(entry["value"], (int, float)) and not isinstance(entry["value"], bool):
            mark = provisional_flags.mark_provisional(
                entry["param_id"], entry["value"], entry["provenance"]
            )
            entry["provisional_tag"] = provisional_flags.line(mark)
            provisional_marks.append(mark)
        registry_out.append(entry)

        affects = entry["affects_indicators"]
        if "*" in affects:
            for n in indicator_names:
                by_indicator.setdefault(n, []).append(entry["param_id"])
        else:
            for n in affects:
                if n in by_indicator:
                    by_indicator[n].append(entry["param_id"])

    n_total = len(registry_out)
    n_prov = len(provisional_marks)
    summary_plain = (
        f"共登记 {n_total} 项核心阈值与参数：物理/数学定义（T1）{counts['T1']} 项、"
        f"文献基准（T2a）{counts['T2a']} 项、本校自采标定（T2b）{counts['T2b']} 项、"
        f"经验临时值（T3a）{counts['T3a']} 项、公开集研究参考（T3b）{counts['T3b']} 项；"
        f"其中 {n_prov} 项带有未标定标记，不得当作已标定常模解读。"
    )
    return {
        "schema_version": raw.get("_schema_version", "1.0.0"),
        "tier_descriptions": dict(raw.get("_tiers") or {}),
        "counts_by_tier": counts,
        "n_total": n_total,
        "n_provisional": n_prov,
        "by_indicator": by_indicator,
        "registry": registry_out,
        "summary_plain": summary_plain,
    }
