# tests/test_report_layer.py
from report_frontend.research_mapper import ResearchCapabilityMapper


def test_zero_input_produces_no_scores():
    """spec §6.1:零输入必须不再产出分数。基线是 47.08。"""
    r = ResearchCapabilityMapper().map_features_to_scores({})

    assert r["total_score"] is None
    for key, dim in r["dimensions"].items():
        assert dim["score"] is None, f"{key} 零输入却仍出分"
        assert dim["confidence"] == "无", f"{key} 置信度应为无"
    assert r["evidence_gaps"], "必须给出 evidence_gaps"


def test_no_fake_resume_fallback():
    """spec §5.3:假简历兜底必须删除。"""
    r = ResearchCapabilityMapper().map_features_to_scores({})
    for dim in r["dimensions"].values():
        for ev in dim["evidence_chain"]:
            assert ev["feature"] != "logic_keyword_density", "假简历兜底仍在"
            assert "BASELINE" not in ev["feature"], "BASELINE_FILL 仍可达"
            assert "代理" not in ev["feature"], "硬编码代理仍在"


def test_quarantined_columns_are_rejected():
    """spec §5.2:封停列即使有值也不得进入证据链。"""
    feats = {"face": {"face_focus_score_mean": 0.3,
                      "face_symmetry_score_mean": 0.98}}
    r = ResearchCapabilityMapper().map_features_to_scores(feats)
    for dim in r["dimensions"].values():
        for ev in dim["evidence_chain"]:
            assert "focus_score" not in ev["feature"]
            assert "symmetry_score" not in ev["feature"]


def test_confidence_can_be_none_and_low():
    """spec §6.5:置信度必须能取到'无'与'低'。"""
    seen = set()
    seen.add(ResearchCapabilityMapper().map_features_to_scores({})
             ["dimensions"]["logical_thinking"]["confidence"])
    feats = {"voice_research": {"interview_pause_duration_mean": 0.8}}
    r = ResearchCapabilityMapper().map_features_to_scores(feats)
    for dim in r["dimensions"].values():
        seen.add(dim["confidence"])
    assert "无" in seen


def test_dimension_weights_sum_to_one():
    """spec §5.3:comm 维度权重和曾为 1.4。"""
    m = ResearchCapabilityMapper()
    for dim_key, rule in m.mapping_rules.items():
        total = sum(w for _, w, _, _, _ in rule["indicators"])
        assert abs(total - 1.0) < 1e-9, f"{dim_key} 权重和为 {total}"
