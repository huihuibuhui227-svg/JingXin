# report_frontend/scene_profiles.py
"""《衡才智析》声明式多场景档案加载与绑定层。

核心原则：
1. 三大应用基地共享同一套 41 项核心物理指标与不打分统计保形裁判底座；
2. 切换场景仅切换声明式档案中的场景名称、基线模式、隐私要求、主动封停列表、
   场景扩展项与合规提示条，绝不改动核心指标计数（41 项）或输出任何复合总分；
3. 与 `redline_guard` 对齐：场景档案中的用户可见文案严禁包含禁用词汇。
"""
from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

SCENE_PROFILES_PATH = Path(__file__).resolve().with_name("scene_profiles.json")
DEFAULT_SCENE_PROFILE_ID = "research_interview"

_SCENARIO_ALIAS_MAP: dict[str, str] = {
    "research": "research_interview",
    "interview": "research_interview",
    "research_interview": "research_interview",
    "metacognition": "ai_metacognition",
    "ai_metacognition": "ai_metacognition",
    "career": "ai_metacognition",
    "career_mcp": "ai_metacognition",
    "followup": "longitudinal_followup",
    "longitudinal": "longitudinal_followup",
    "longitudinal_followup": "longitudinal_followup",
}


@lru_cache(maxsize=1)
def _load_registry() -> dict[str, Any]:
    with open(SCENE_PROFILES_PATH, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or not isinstance(data.get("profiles"), list):
        raise ValueError("scene_profiles.json 结构无效：缺少 profiles 列表")
    return data


def normalize_scene_profile_id(scene_id: str | None) -> str:
    """规范化场景标识；未知或空值时按明确规则回落或报错。"""
    if scene_id is None or str(scene_id).strip() == "":
        return DEFAULT_SCENE_PROFILE_ID
    raw = str(scene_id).strip().lower()
    if raw in _SCENARIO_ALIAS_MAP:
        return _SCENARIO_ALIAS_MAP[raw]
    valid_ids = {p["id"] for p in _load_registry()["profiles"]}
    if raw in valid_ids:
        return raw
    raise ValueError(f"未知场景档案：{scene_id!r}（支持 {sorted(valid_ids)}）")


def list_scene_profiles() -> list[dict[str, Any]]:
    """返回全部声明式场景档案（深拷贝，避免调用方篡改缓存）。"""
    return [copy.deepcopy(p) for p in _load_registry()["profiles"]]


def get_scene_profile(scene_id: str | None = None) -> dict[str, Any]:
    """按标识获取单一场景档案详情；缺省返回默认科研过程场景。"""
    target = normalize_scene_profile_id(scene_id)
    for prof in _load_registry()["profiles"]:
        if prof.get("id") == target:
            return copy.deepcopy(prof)
    raise ValueError(f"未找到场景档案：{target!r}")


def build_scene_profile_payload(
    scene_id: str | None = None,
    *,
    extensions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造注入报告载荷 `payload['scene_profile']` 的公开摘要结构。"""
    prof = get_scene_profile(scene_id)
    return {
        "schema_version": _load_registry().get("schema_version", "scene_profiles_v1"),
        "id": prof["id"],
        "name": prof["name"],
        "short_label": prof["short_label"],
        "weight_label": prof["weight_label"],
        "route_base": prof.get("route_base", "/research"),
        "baseline_mode": prof["baseline_mode"],
        "default_privacy_mode": prof["default_privacy_mode"],
        "privacy_modes_supported": list(prof.get("privacy_modes_supported") or []),
        "core_indicator_count": int(prof.get("core_indicator_count", 41)),
        "active_indicator_families": list(prof.get("active_indicator_families") or []),
        "extension_indicator_keys": list(prof.get("extension_indicators") or []),
        "stage_windows": list(prof.get("stage_windows") or []),
        "explicit_blocked_overrides": list(prof.get("explicit_blocked_overrides") or []),
        "threshold_overrides": dict(prof.get("threshold_overrides") or {}),
        "compliance_banner": prof["compliance_banner"],
        "description": prof["description"],
        "mcp_server": prof.get("mcp_server"),
        "extensions": copy.deepcopy(extensions or {}),
    }
