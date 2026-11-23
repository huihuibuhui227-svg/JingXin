# redline_guard.py
"""《衡才智析》对外文本红线合规扫描器（独立于 report_frontend 输出模块）。

将禁用词表与合规检查函数放在仓库根部独立模块中，避免与 `report_frontend/`
面向用户输出字符串的白盒扫描守卫（`test_no_banned_words_in_output_strings`）冲突。
"""
from __future__ import annotations

import re

# 1. 停用旧名与虚假/夸大技术词
FORBIDDEN_BRAND_AND_HYPE_TERMS: tuple[str, ...] = (
    "镜心",
    "JingXin",
    "科研心流态智能判推系统",
    "数字化仪表盘",
    "综合得分",
    "量子融合",
    "ST-GNN",
    "临床级",
    "2000例",
    "60Hz 眼动仪",
    "4K 自适应",
    "边缘即时销毁",
)

# 2. 违规情绪/临床/心理诊断词（欧盟 AI Act + 中国 NMPA 红线）
FORBIDDEN_CLINICAL_EMOTION_TERMS: tuple[str, ...] = (
    "高度焦虑",
    "焦虑",
    "紧张度",
    "轻微紧张",
    "高紧张",
    "处于紧张",
    "抑郁",
    "心理风险",
    "压力韧性",
    "抗压",
    "微表情",
    "语音情感",
    "科研潜质评估",
    "候选人得分",
)

FORBIDDEN_USER_TEXT_RE = re.compile(
    "|".join(
        re.escape(w)
        for w in (*FORBIDDEN_BRAND_AND_HYPE_TERMS, *FORBIDDEN_CLINICAL_EMOTION_TERMS)
    )
)


def find_redline_violations(text: str) -> list[str]:
    """扫描面向用户文本中的红线违规词，返回去重后的命中列表。"""
    if not text:
        return []
    seen: list[str] = []
    for m in FORBIDDEN_USER_TEXT_RE.finditer(text):
        hit = m.group(0)
        if hit not in seen:
            seen.append(hit)
    return seen
