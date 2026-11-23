# report_frontend/brand.py
"""《衡才智析》后端单一品牌与红线口径常量源。

全仓所有对外输出（HTML 报告、导出文件头、Flask 面板、CLI 摘要、合规守卫测试）
统一从本模块读取正式品牌名与红线禁用词，杜绝多处硬编码导致旧名残留或口径漂移。
"""
from __future__ import annotations

import re

# ── 对外正式品牌口径（见 24_定稿名称与夸大的边界_20261121.md） ─────────────
BRAND_NAME = "衡才智析"
BRAND_SUBTITLE = "多模态行为证据披露系统"
BRAND_FULL_TITLE = f"《{BRAND_NAME}——{BRAND_SUBTITLE}》"
BRAND_ENGLISH = "Hengcai Zhixi: A Multimodal Behavioral Evidence Disclosure System"
COPYRIGHT_SOFTWARE_NAME = "衡才智析——科研行为证据门控评估系统 V1.0"
BRAND_SLOGAN = "不给分，只给证据与缺口"
ARCHITECTURE_SLOGAN = "仪器可以是黑盒，裁判必须是白盒"

# 报告与导出物默认标题 / 页脚 / 文件名前缀
REPORT_HTML_TITLE = f"{BRAND_NAME} · 多模态行为证据披露报告"
REPORT_HTML_FOOTER = (
    f"{BRAND_NAME} · {BRAND_SUBTITLE} · 报告由系统自动生成 · 不含评分与评级"
)
TRANSCRIPT_EXPORT_HEADER = f"{BRAND_NAME} · 语音转写原件导出"
REPORT_FILENAME_PREFIX = "HengCai_Evidence_Report"

# 同时识别新前缀（HengCai_Evidence_Report_*）与历史遗留前缀（Research_Assessment_Report_*）
REPORT_FILENAME_RE = re.compile(
    r"^(?:HengCai_Evidence_Report|Research_Assessment_Report)_.*\.html$",
    re.IGNORECASE,
)

# ── 绝对红线禁用词（面向用户界面/报告/导出正文与实时 API 文案） ──────────
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
