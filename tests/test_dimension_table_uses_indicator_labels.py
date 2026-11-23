# tests/test_dimension_table_uses_indicator_labels.py
"""Task 25 回归钉子：维度表/审计链/缺口行必须优先使用 L1 指标中文名，只在真删槽时退回槽名。"""
from __future__ import annotations

from report_frontend import brand, l1_mapper, l1_quality, l1_report, report_generator


def test_name_of_prefers_indicator_label_over_legacy_slot_label():
    assert l1_report._name_of({"slot": "eye_contact", "l1": "gaze_dwell_p50"}) == "眼神接触时长"
    assert l1_report._name_of({"slot": "symmetry_score", "l1": "facial_asymmetry_p90"}) == "面部不对称度"
    assert l1_report._name_of({"slot": "pause_duration", "l1": "pause_duration_p50"}) == "停顿中位时长"
    assert l1_report._name_of({"slot": "left_wrist_jitter_world", "l1": "gesture_activity_per_sec"}) == "上肢活动抖动度"
    assert l1_report._name_of({"slot": "fluency_score", "l1": "pause_structure_entropy"}) == "停顿结构均匀度"


def test_name_of_falls_back_to_slot_label_only_when_no_l1_indicator():
    assert l1_report._name_of({"slot": "eye_contact", "l1": None}) == l1_mapper.slot_label("eye_contact")
    assert l1_report._name_of({"slot": "symmetry_score"}) == l1_mapper.slot_label("symmetry_score")


def test_full_coverage_lead_sentence_never_calls_two_of_two_weakest():
    payload = {
        "dimensions": {
            "logical_thinking": {"display_name": "话语结构特征", "matched_indicators": "3/3"},
            "cognitive_efficiency": {"display_name": "作答过程特征", "matched_indicators": "2/2"},
        },
        "coverage": {
            "n_passed": 5, "n_slots": 5, "n_dropped": 0,
            "n_measurable": 5, "n_not_measured": 0, "n_no_indicator": 0,
            "confidence_cap": "高",
        },
    }
    lead = l1_report.build_lead_summary(payload["dimensions"], payload["coverage"])
    assert "最薄" not in lead, f"满覆盖时不应出现「最薄」: {lead}"
    assert "完整覆盖" in lead
    html_lead = report_generator.ReportGenerator()._lead_sentence(payload)
    assert "最薄" not in html_lead, f"HTML 满覆盖首段不应出现「最薄」: {html_lead}"
    assert "完整覆盖" in html_lead


def test_html_report_uses_hengcai_brand_and_zero_redline_words():
    payload = l1_report.build_l1_report("20260927_212143_95a8", logs={})
    html = report_generator.ReportGenerator()._build_html_report(payload, {}, {}, {}, [])
    assert brand.BRAND_NAME in html
    assert "JingXin" not in html
    assert "⟹" not in html
    assert not brand.find_redline_violations(html), brand.find_redline_violations(html)
