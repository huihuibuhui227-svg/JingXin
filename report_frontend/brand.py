# report_frontend/brand.py
"""《衡才智析》后端单一品牌常量源。

全仓所有对外输出（HTML 报告、导出文件头、Flask 面板、CLI 摘要、合规守卫测试）
统一从本模块读取正式品牌名与文件命名规则；红线词表与扫描器由根模块 `redline_guard`
提供并在此统一转出。
"""
from __future__ import annotations

import re

from redline_guard import (
    FORBIDDEN_BRAND_AND_HYPE_TERMS,
    FORBIDDEN_CLINICAL_EMOTION_TERMS,
    FORBIDDEN_USER_TEXT_RE,
    find_redline_violations,
)

# ── 对外正式品牌口径（定稿：衡才智析——多模态行为证据分析系统） ─────────────
BRAND_NAME = "衡才智析"
BRAND_SUBTITLE = "多模态行为证据分析系统"
BRAND_FULL_TITLE = f"《{BRAND_NAME}——{BRAND_SUBTITLE}》"
BRAND_ENGLISH = "Hengcai Zhixi: A Multimodal Behavioral Evidence Analysis System"
COPYRIGHT_SOFTWARE_NAME = "衡才智析——科研行为证据门控评估系统 V1.0"
BRAND_SLOGAN = "不给分，只给证据与缺口"
ARCHITECTURE_SLOGAN = "仪器可以是黑盒，裁判必须是白盒"

# 报告与导出物默认标题 / 页脚 / 文件名前缀
REPORT_HTML_TITLE = f"{BRAND_NAME} · 多模态行为证据分析报告"
REPORT_HTML_FOOTER = (
    f"{BRAND_NAME} · {BRAND_SUBTITLE} · 报告由系统自动生成 · 仅披露实测证据与缺口"
)
TRANSCRIPT_EXPORT_HEADER = f"{BRAND_NAME} · 语音转写原件导出"
REPORT_FILENAME_PREFIX = "HengCai_Evidence_Report"

# 同时识别新前缀（HengCai_Evidence_Report_*）与历史遗留前缀（Research_Assessment_Report_*）
REPORT_FILENAME_RE = re.compile(
    r"^(?:HengCai_Evidence_Report|Research_Assessment_Report)_.*\.html$",
    re.IGNORECASE,
)

__all__ = [
    "BRAND_NAME",
    "BRAND_SUBTITLE",
    "BRAND_FULL_TITLE",
    "BRAND_ENGLISH",
    "COPYRIGHT_SOFTWARE_NAME",
    "BRAND_SLOGAN",
    "ARCHITECTURE_SLOGAN",
    "REPORT_HTML_TITLE",
    "REPORT_HTML_FOOTER",
    "TRANSCRIPT_EXPORT_HEADER",
    "REPORT_FILENAME_PREFIX",
    "REPORT_FILENAME_RE",
    "FORBIDDEN_BRAND_AND_HYPE_TERMS",
    "FORBIDDEN_CLINICAL_EMOTION_TERMS",
    "FORBIDDEN_USER_TEXT_RE",
    "find_redline_violations",
]
