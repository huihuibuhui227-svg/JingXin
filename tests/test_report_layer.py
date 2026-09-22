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


def test_slot_level_quarantine_covers_spec_5_3():
    """spec §5.3 的槽位级封停必须反映到 QUARANTINE,否则这些指标绕过 G4 直接进打分。"""
    from report_frontend.evidence_gate import is_quarantined

    for key in ("face_gaze_stability_mean", "face_micro_exp_au_name_au4_freq",
                "face_micro_exp_au_name_au7_freq", "gesture_left_hand_jitter_mean",
                "voice_research_research_speech_ratio_mean", "face_eye_contact_ratio"):
        assert is_quarantined(key) is not None, f"{key} 未被封停"


def test_zero_evidence_report_makes_no_claims():
    """零证据时不得出现任何才能/心理素质断言(spec §5.4)。"""
    from report_frontend.report_generator import ReportGenerator

    result = ResearchCapabilityMapper().map_features_to_scores({})
    html = ReportGenerator(output_dir="/tmp")._generate_deep_text_analysis({}, result)
    for claim in ("科研天赋", "心理素质", "最为突出", "表现最为"):
        assert claim not in html, f"零证据下仍出现断言:{claim}"
    assert "证据不足" in html or "未采集到" in html


def test_all_slot_level_quarantine_keys_are_pinned():
    """7 个槽位级封停键必须逐个被钉住,不能只测真实键。"""
    from report_frontend.evidence_gate import is_quarantined

    for entry in ("gaze_stability", "au4_freq", "au7_freq", "jitter",
                  "speech_ratio", "eye_contact", "fluency_score"):
        assert is_quarantined(entry) is not None, f"{entry} 未被封停"


def test_question_index_is_not_dropped():
    """spec §8-3:question_index 曾被 'index' 跳过规则静默丢弃。"""
    import pandas as pd

    from report_frontend.feature_engine import PsychologicalFeatureEngine

    df = pd.DataFrame({"question_index": [0, 1, 2], "pitch_mean": [100.0, 120.0, 110.0]})
    feats = PsychologicalFeatureEngine({"voice_research": df}).extract_all_features()
    keys = " ".join(feats.get("voice_research", {}).keys())
    assert "question_index" in keys, "question_index 仍被丢弃"


def test_dead_fluency_branch_removed():
    """spec §5.3:该分支要求列名同时含 speech_ratio 与 mean,故为死代码。

    真实日志列名为 speech_ratio,永远不含 mean,故该分支从未触发;
    这里额外用 speech_ratio_mean 这个「能触发该分支」的列名钉死它已被删除,
    否则仅用 speech_ratio 时本用例在改动前后都会通过(空断言)。
    """
    import pandas as pd

    from report_frontend.feature_engine import PsychologicalFeatureEngine

    for col in ("speech_ratio", "speech_ratio_mean"):
        df = pd.DataFrame({col: [0.9, 0.8, 0.95]})
        feats = PsychologicalFeatureEngine({"voice_research": df}).extract_all_features()
        keys = " ".join(feats.get("voice_research", {}).keys())
        assert "fluency_score" not in keys, f"{col} 触发了已删除的 fluency_score 死分支"
        assert "fluency_proxy" not in keys, f"{col} 触发了已删除的 fluency_proxy 死分支"


def test_no_fabricated_percentile():
    """spec §5.5:百分位只能来自真实常模。

    ⚠️ 必须喂**能过证据门**的输入。否则 evidence_chain 全为空、嵌套循环零断言,
    本测试对**任何**实现都通过 —— 包括把百分位加回来的实现。
    Task 5 复审实测:原输入 `interview_pause_duration_mean` 让 5 个维度全部
    score=None / chain=[],是彻底的空断言("回归守卫"的标签夸大了它)。
    """
    feats = {"voice_research": {"logic_keyword_density": 0.05,
                                "logic_keyword_density_std": 0.01,
                                "_n_rows": 100.0}}
    r = ResearchCapabilityMapper().map_features_to_scores(feats)

    total = sum(len(d["evidence_chain"]) for d in r["dimensions"].values())
    assert total > 0, "输入未过证据门,本测试退化为空断言"
    for dim in r["dimensions"].values():
        for ev in dim["evidence_chain"]:
            assert "percentile" not in ev or ev["percentile"] is None


def test_radar_has_no_norm_baseline():
    """spec §5.5:[60]*5 与 '常模基准' 图例必须删除。"""
    from report_frontend.visualizer import ReportVisualizer

    result = ResearchCapabilityMapper().map_features_to_scores({})
    fig = ReportVisualizer(output_dir="/tmp")._build_radar_figure(result)
    names = [t.name for t in fig.data]
    assert "常模基准" not in names
    # 正向断言:否则"一张轨迹都没有的图"也会通过,候选分轨迹的存续无人钉住
    assert names == ["候选人得分"]
