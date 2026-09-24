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
    """spec §5.3:假简历兜底与 BASELINE_FILL 必须不可达。

    ⚠️ 原版用零输入 fixture,证据链为空 → 循环零断言,对任何实现都通过。
    改为喂一个只让 1 个槽过门的输入,断言 matched 恰为 1/4 且缺口为 3。
    若 BASELINE_FILL 回归,缺失的 3 个槽会被填空 → matched 变 4/4 → 本测试变红。
    控制器已实测该 fixture 输出:score=100.0 / conf=低 / matched=1/4 / 缺口 3。
    """
    feats = {"voice_research": {"logic_keyword_density": 0.05,
                                "logic_keyword_density_std": 0.01,
                                "_n_rows": 100.0}}
    dim = ResearchCapabilityMapper().map_features_to_scores(feats)["dimensions"]["logical_thinking"]

    assert dim["matched_indicators"] == "1/4", "缺失槽被填充(BASELINE_FILL 回归?)"
    assert len(dim["evidence_gaps"]) == 3
    for ev in dim["evidence_chain"]:
        # 裸键名是假简历兜底的签名;带模态前缀才是真测量
        assert ev["feature"] != "logic_keyword_density", "裸键名 = 假简历兜底的签名"
        assert "BASELINE" not in ev["feature"]
        assert "代理" not in ev["feature"]


def test_quarantined_columns_are_rejected():
    """spec §5.2:封停列即使有值也不得进入证据链。

    ⚠️ 原版只喂封停列 → 全部被拒 → 链为空 → 循环零断言。
    改为同时喂一个干净列让链非空,再断言封停列不在其中。
    若 G4 被移除,focus_score 会进链 → 本测试变红。

    symmetry_score 属 stress_resilience 维度,而该维四个槽全部封停、链必为空,
    故不能靠"链里没有它"来钉(那恒真)。改为断它**出现在证据缺口里** ——
    若删掉它的封停条目且喂入该列,它会进链、缺口里便不再有它 → 本测试变红。
    控制器实测:stress_resilience 链=0、缺口=4、缺口含「面部对称性」。
    """
    feats = {"voice_research": {"logic_keyword_density": 0.05,
                                "logic_keyword_density_std": 0.01,
                                "_n_rows": 100.0},
             "face": {"face_focus_score_mean": 0.3,
                      "face_focus_score_std": 0.05,
                      "face_symmetry_score_mean": 0.98,
                      "face_symmetry_score_std": 0.02,
                      "_n_rows": 100.0}}
    dims = ResearchCapabilityMapper().map_features_to_scores(feats)["dimensions"]

    lt = dims["logical_thinking"]
    assert lt["evidence_chain"], "证据链为空 —— 本测试退化为空断言"
    for ev in lt["evidence_chain"]:
        assert "focus_score" not in ev["feature"]

    sr = dims["stress_resilience"]
    assert sr["evidence_chain"] == [], "封停列进了链"
    assert any("对称" in g for g in sr["evidence_gaps"]), "对称性槽未被处理"


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


BANNED = ["焦虑", "紧张", "压力", "抗压", "情绪稳定", "说谎", "诚信",
          "录用", "人格", "心理画像", "常模"]

# 混合 fixture:让 logical_thinking 真的出分(1/4 槽过门),其余 4 维仍为 score=None。
# 叙事层里禁止词与硬编码句原本住在**出分路径**上,零证据输入会在组合那几句之前
# 就短接掉 —— 用零证据 fixture 的测试因此对改动前后都通过(本计划已栽过八次)。
# 本 fixture 让三条叙事层测试同时走到出分路径与无证据路径。
_MIXED = {"voice_research": {"logic_keyword_density": 0.05,
                             "logic_keyword_density_std": 0.01,
                             "_n_rows": 100.0}}


def test_deep_analysis_has_no_banned_words():
    """spec §5.4 + §5.6:叙事层不得含情绪/心理/诚信构念。

    ⚠️ 只覆盖**叙事层自己写的句子**。报告里出现的禁止词有另一个来源:
    `research_mapper` 提供的两个标签 —— `display_name`「抗压与情绪稳定性」
    (含 抗压、情绪稳定)与 `human_name`「面部紧张度」(含 紧张)。
    实测确认全 mapper 只有这两处命中。

    它们属 **Task 7** 的改名范围(Task 7 改完 `display_name` 与 `human_name`
    后,**须恢复全量扫描** —— 见计划 Task 7 Step 3a)。
    本测试用剔除这两个标签的方式,把叙事层自己的输出隔离出来测。
    剔除是精确字符串替换,所以叙事层**自己在别处**写出的禁止词仍会被抓到。

    ⚠️ fixture 必须是 _MIXED:禁止词原本住在出分路径上,零证据输入只扫"无证据"那几句。
    """
    from report_frontend.report_generator import ReportGenerator

    result = ResearchCapabilityMapper().map_features_to_scores(_MIXED)
    # 断言 fixture 真的走出了分路径 —— 否则本测试退化为只扫无证据文案
    assert result["total_score"] is not None, "fixture 未出分,本测试退化为空断言"
    html = ReportGenerator()._generate_deep_text_analysis(_MIXED, result)

    # 剔除 mapper 提供的标签(Task 7 修复后本段移除,恢复全量)
    for label in ("抗压与情绪稳定性", "面部紧张度"):
        html = html.replace(label, "")

    for word in BANNED:
        assert word not in html, f"叙事层出现禁止词:{word}"


def test_deep_analysis_handles_none_scores():
    """score 为 None 的维度不得参与排序/取最大值,且仍要渲染成"证据不足"。

    ⚠️ fixture 必须是 _MIXED:零证据输入会让 total_score 直接为 None,
    根本走不到 max(..., key=_CONF_ORDER.get) 那一行,本测试就守不住它。
    _MIXED 下 logical_thinking 出分、其余 4 维为 None,两条路径同时被走到。
    """
    from report_frontend.report_generator import ReportGenerator

    result = ResearchCapabilityMapper().map_features_to_scores(_MIXED)
    html = ReportGenerator()._generate_deep_text_analysis(_MIXED, result)

    assert result["total_score"] is not None, "fixture 未出分,本测试守不住 conf_cap 那一行"
    assert any(d["score"] is None for d in result["dimensions"].values()), \
        "fixture 无 None 维度,本测试守不住'证据不足'渲染"
    assert "证据不足" in html


def test_no_hardcoded_gaze_claim():
    """spec §5.4:那句'未出现异常的回避行为'是纯硬编码。

    ⚠️ fixture 必须是 _MIXED:零证据输入会在组合那三句之前短接,
    本测试在改动前后都会通过(实测原版即如此)。
    """
    from report_frontend.report_generator import ReportGenerator

    result = ResearchCapabilityMapper().map_features_to_scores(_MIXED)
    assert result["total_score"] is not None, "fixture 未出分,本测试退化为空断言"
    html = ReportGenerator()._generate_deep_text_analysis(_MIXED, result)
    assert "未出现异常的回避行为" not in html
    assert "如外科医生般" not in html
