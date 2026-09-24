# tests/test_report_layer.py
import ast
from pathlib import Path

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
    Task 6 换键(connective_density)后实测该 fixture 输出:score=0.5 / conf=低 /
    matched=1/4 / 缺口 3 —— 0.05(每百字)在新量程 10 下归一为 0.005,故分数不再是
    旧量程下的 100.0;本测试断言的是 matched 与缺口,两者不受换键影响。
    """
    feats = {"voice_research": {"connective_density_mean": 0.05,
                                "connective_density_std": 0.01,
                                "_n_rows": 100.0}}
    dim = ResearchCapabilityMapper().map_features_to_scores(feats)["dimensions"]["logical_thinking"]

    assert dim["matched_indicators"] == "1/4", "缺失槽被填充(BASELINE_FILL 回归?)"
    assert len(dim["evidence_gaps"]) == 3
    for ev in dim["evidence_chain"]:
        # 裸键名是假简历兜底的签名;带模态前缀才是真测量
        assert ev["feature"] != "connective_density", "裸键名 = 假简历兜底的签名"
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
    feats = {"voice_research": {"connective_density_mean": 0.05,
                                "connective_density_std": 0.01,
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
    """spec §6.5:置信度必须能取到'无'**与**'低'。

    ⚠️ 原版只断言 `"无" in seen`,而它的第二个 fixture(`interview_pause_duration_mean`,
    已被 G4 封停、又无 `_n_rows`)只能产出「无」——「低」这一半的声明**从未被任何断言守住**。
    把 `confidence_from` 改成永不返回「低」,原版照样全绿。这是本计划空断言家族的第 9 个成员,
    且 `sys.settrace` 扫描抓不到它 —— 它是**缺断言**,不是断言没执行。

    「低」的可达性由 `_MIXED`(logical_thinking 1/4 槽过门)证明,该 fixture 定义在文件下方。
    """
    mapper = ResearchCapabilityMapper()

    zero = mapper.map_features_to_scores({})["dimensions"]["logical_thinking"]["confidence"]
    assert zero == "无", "0 个槽过门必须落在「无」"

    low = mapper.map_features_to_scores(_MIXED)["dimensions"]["logical_thinking"]["confidence"]
    assert low == "低", (
        "1/4 个槽过门必须落在「低」—— 若 confidence_from 不再返回「低」,本断言变红"
    )

    from report_frontend.evidence_gate import confidence_from

    # 三值化的边界本身(与 mapper 解耦的一组定点)
    assert tuple(confidence_from(n, 4) for n in range(5)) == ("无", "低", "中", "中", "中")


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
    feats = {"voice_research": {"connective_density_mean": 0.05,
                                "connective_density_std": 0.01,
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
    assert names == ["通过证据门的指标槽数"]


BANNED = ["焦虑", "紧张", "压力", "抗压", "情绪稳定", "说谎", "诚信",
          "录用", "人格", "心理画像", "常模"]


# --- I2:折算因子必须在 evidence_thresholds.json 登记(spec §5.1) ---

# (slot 关键词, JSON 内的补丁路径, 补丁值, 输入原始值, 期望归一值)
# 期望值按**打补丁后的登记值**手工推算:若某族又被写回代码里的裸常量,
# 该行的期望值不会出现(例如 density 会恒为 1.0 而不是 0.05)。
_SCALE_CASES = [
    ("connective_density", ("scale_factors", "density", "full_scale"), 1.0, 0.05, 0.05),
    ("jitter", ("scale_factors", "jitter", "full_scale"), 0.4, 0.1, 0.25),
    ("gaze_deviation", ("scale_factors", "deviation", "full_scale"), 0.4, 0.1, 0.25),
    ("text_avg_length", ("scale_factors", "length", "full_scale"), 60.0, 30.0, 0.5),
    ("energy", ("scale_factors", "energy", "full_scale"), 0.002, 0.001, 0.5),
    ("pitch_variation", ("scale_factors", "pitch_variation", "full_scale"), 40.0, 20.0, 0.5),
    ("fluency_score", ("scale_factors", "score", "percent_divisor"), 50.0, 20.0, 0.4),
    # pause 分箱用合成键 pause_seconds:真实词表里唯一含 pause 的键是 pause_duration,
    # 而它同时含 ratio,按 legacy 归因走恒等族(见下方 shadowing 测试)。
    ("pause_seconds", ("scale_factors", "pause", "above_span"), 1.0, 4.0, 1.0),
    ("pause_seconds", ("scale_factors", "pause", "below_value"), 0.25, 0.3, 0.25),
    ("blink_rate", ("_default_scale_factor", "full_scale"), 100.0, 50.0, 0.5),
    # 恒等族(ratio/freq/stability/contact)也必须来自登记表:整个族换成 full_scale 后
    # 半径语义变了,输出必须跟着变(1.0 → 0.5),否则说明该族是代码里的分支常量。
    ("speech_ratio", ("scale_factors", "ratio"),
     {"kind": "full_scale", "full_scale": 2.0,
      "basis_kind": "legacy_arbitrary", "basis": "测试探针"}, 1.0, 0.5),
]


def test_scale_factors_are_registered_not_hardcoded(tmp_path, monkeypatch):
    """spec §5.1:临时值不得写成代码里的裸常量。

    逐族把 evidence_thresholds.json 里的登记值改掉,`_dynamic_normalize` 的输出必须随之
    改变。任何一族被硬编码回代码里,该族那一行就会拿到打补丁前的值 → 本测试变红。
    """
    import json

    from report_frontend import evidence_gate

    data = json.loads(evidence_gate._THRESHOLD_FILE.read_text(encoding="utf-8"))
    probe = tmp_path / "evidence_thresholds.json"

    for keyword, path, value, raw, expected in _SCALE_CASES:
        node = data
        for part in path[:-1]:
            node = node[part]
        node[path[-1]] = value
        probe.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setattr(evidence_gate, "_THRESHOLD_FILE", probe)

        got = ResearchCapabilityMapper()._dynamic_normalize(raw, keyword)
        assert abs(got - expected) < 1e-9, (
            f"{keyword} 的归一化没跟着登记值变:期望 {expected},实际 {got}"
            f" —— 该族的因子可能被写回了代码里的裸常量"
        )


def test_pause_duration_is_shadowed_by_the_ratio_family():
    """pause_duration 撞名留档 —— spec §8.2 #2 把这条标为「待查」,本轮实测结案。

    实测(旧实现的 if 链 与 本实现的最长匹配,两者一致):`'ratio' in 'pause_duration'`
    为真,而恒等族在旧 if 链里排在 pause 之前 → pause_duration 走的是**恒等**分支,
    0.5–3.0s 的分箱从未执行过。spec §8.2 说"不含 ratio、走 pause 分支"是**记错了**。

    本测试把 legacy 行为钉住(改它要动登记表,且应带 M5 依据):
    若有人把 pause 族调到 ratio 之前,pause_duration 的归一化随之改变 → 本测试变红。
    """
    from report_frontend.evidence_gate import scale_factor_for

    assert "ratio" in "pause_duration" and "pause" in "pause_duration", "撞名的前提变了"
    assert scale_factor_for("pause_duration")["kind"] == "identity", \
        "pause_duration 的归因变了 —— legacy 走的是恒等族(0.5–3.0s 分箱从未执行)"
    # 合成键 pause_seconds 不含 ratio,才能走到登记的分箱(真实词表里没有这样的键)
    assert scale_factor_for("pause_seconds")["kind"] == "pause_bins"


def test_every_registered_scale_factor_declares_a_basis():
    """spec §5.1:登记项必须声明依据;没有依据的必须显式写 legacy_arbitrary,不得留空。"""
    from report_frontend.evidence_gate import load_thresholds

    data = load_thresholds()
    assert data["scale_factors"], "折算因子未登记"
    kinds = {"definitional", "physical", "legacy_arbitrary"}
    for name, spec in data["scale_factors"].items():
        assert spec.get("basis"), f"{name} 未声明依据"
        assert spec["basis_kind"] in kinds, f"{name} 的 basis_kind 未登记:{spec['basis_kind']}"
    default = data["_default_scale_factor"]
    assert default.get("basis") and default["basis_kind"] in kinds

# 混合 fixture:让 logical_thinking 真的出分(1/4 槽过门),其余 4 维仍为 score=None。
# 叙事层里禁止词与硬编码句原本住在**出分路径**上,零证据输入会在组合那几句之前
# 就短接掉 —— 用零证据 fixture 的测试因此对改动前后都通过(本计划已栽过八次)。
# 本 fixture 让三条叙事层测试同时走到出分路径与无证据路径。
# 键名与量程同源:真实日志列 connective_density 经 feature_engine 的通用数值路径
# 产出 _mean/_std,量程是「每百字 10 个」—— 10.0 正好落在满量程上(分数饱和端)。
_MIXED = {"voice_research": {"connective_density_mean": 10.0,
                             "connective_density_std": 1.0,
                             "_n_rows": 100.0}}

# 低分混合 fixture:同一槽过门,但归一值为 0 → 旧实现给出 0.0 分 + 档位「待提升」。
# 审查者在真实 ASR 转写上复现的正是这条路径(报告头渲染「0.0 / 综合行为观测评分 / 待提升」)。
_LOW = {"voice_research": {"connective_density_mean": 0.0,
                           "connective_density_std": 0.01,
                           "_n_rows": 100.0}}

# 五档评语(spec §5.4 :157-158 的处置对象:它们是对人的评级,不得再出现在报告里)
_LEVEL_WORDS = ("卓越", "优秀", "良好", "合格", "待提升")


def _rendered_html(result, features):
    """走完整的报告渲染路径(不落盘、不画图)。

    `_build_html_report` 的 chart_paths/static_images 传空即可 —— 缺图会渲染成占位符,
    本轮要断言的是文字面。
    """
    from report_frontend.report_generator import ReportGenerator

    return ReportGenerator(output_dir="/tmp")._build_html_report(result, {}, features, {}, [])


def test_report_renders_no_candidate_rating():
    """spec §5.4 :157-158 / §5.6:点分与五档评语不得作为对人的评定出现在报告里。

    fixture 必须是 `_MIXED`:零证据输入下 `total_level` 是「证据不足」,五个档位词
    一个都到不了渲染层 —— 用零证据 fixture 的断言在改动前后都通过(空断言)。
    """
    result = ResearchCapabilityMapper().map_features_to_scores(_MIXED)
    assert result["total_level"] in _LEVEL_WORDS, "fixture 没走进五档评语,本测试退化为空断言"

    html = _rendered_html(result, _MIXED)

    for word in _LEVEL_WORDS:
        assert word not in html, f"报告仍把档位评语渲染成对人的评级:{word}"
    assert "综合行为观测评分" not in html, "报告仍把未标定标尺上的点分当评分渲染"
    assert str(result["total_score"]) not in html, "报告仍渲染点分数值"

    # 正向断言:否则"把整段都删掉"也能让本测试通过
    assert "观测区间" in html, "聚合呈现缺区间(spec §5.6:区间 + 置信度 + 依据)"
    assert "置信度上限" in html, "聚合呈现缺置信度"
    assert "连接词密度" in html, "聚合呈现缺依据"


def test_zero_value_scored_session_renders_no_verdict():
    """审查者在 live 路径上的第二个复现:真实 ASR 转写 → 0.0 / 待提升。

    与 `_MIXED`(100.0 / 卓越)合起来覆盖点分的两端;只测高分端会漏掉低分端。
    """
    result = ResearchCapabilityMapper().map_features_to_scores(_LOW)
    assert result["total_level"] == "待提升", "fixture 没走进「待提升」档,本测试退化为空断言"

    html = _rendered_html(result, _LOW)

    for word in _LEVEL_WORDS:
        assert word not in html, f"报告仍把档位评语渲染成对人的评级:{word}"
    assert "综合行为观测评分" not in html
    assert "观测区间" in html, "低分端同样要给区间,不能只在高分端给"


def test_scored_dimension_follows_spec_5_4_shape_and_carries_5_6_caveat():
    """spec §5.4「整节的正确形态」= 值 + 有效样本量 + 置信度 + evidence_gaps;
    spec §5.6 要求出分维度带「该维度目前无独立效标,仅供行为描述」。
    """
    result = ResearchCapabilityMapper().map_features_to_scores(_MIXED)
    dim = result["dimensions"]["logical_thinking"]
    assert dim["score"] is not None, "fixture 未出分,本测试退化为空断言"

    html = _rendered_html(result, _MIXED)

    for token in ("有效样本量", "观测区间", "置信度：", "无独立效标", "仅供行为描述"):
        assert token in html, f"出分维度缺 {token}(spec §5.4 / §5.6)"
    assert "归一值" not in html, "维度表仍在渲染未标定标尺上的归一化点分"
    # 缺口仍要逐维列出(spec §5.4 的 evidence_gaps)
    assert len(dim["evidence_gaps"]) == 3 and "未过门的指标" in html


def test_deep_analysis_summary_has_no_point_score_or_level():
    """`_generate_deep_text_analysis` 那段聚合摘要不得再印档位/点分(spec §5.4 :157-158)。"""
    from report_frontend.report_generator import ReportGenerator

    result = ResearchCapabilityMapper().map_features_to_scores(_MIXED)
    assert result["total_level"] in _LEVEL_WORDS, "fixture 没走进五档评语,本测试退化为空断言"

    html = ReportGenerator()._generate_deep_text_analysis(_MIXED, result)

    for word in _LEVEL_WORDS:
        assert word not in html, f"深度分析摘要仍印档位:{word}"
    assert "综合行为观测评分" not in html


def test_radar_plots_coverage_not_scores():
    """spec §5.4/§5.6:雷达图的半径不得再是维度点分。

    `_MIXED` 下 logical_thinking 的维度分是 100.0(未标定标尺上的复合点分),
    而它的覆盖数是 1/4 —— 半径若回到 `dim['score']`,r 里会出现 100.0 → 本测试变红。
    """
    from report_frontend.visualizer import ReportVisualizer

    result = ResearchCapabilityMapper().map_features_to_scores(_MIXED)
    assert result["dimensions"]["logical_thinking"]["score"] == 100.0, \
        "fixture 未出分,本测试退化为空断言"

    fig = ReportVisualizer(output_dir="/tmp")._build_radar_figure(result)
    raw = list(fig.data[0].r)

    assert None not in raw, f"半径里出现空值 —— 半径已回到 dim['score'](无证据维度即 None):{raw}"
    r = [float(v) for v in raw]

    assert 100.0 not in r, "雷达图仍在画维度点分"
    assert r[:5] == [1.0, 0.0, 0.0, 0.0, 0.0], f"半径应为过门的槽数,实际 {r}"


def test_observed_interval_is_session_derived():
    """spec §5.6:聚合/维度呈现的区间只能来自本场会话自己的测量。

    `_MIXED` 里 `connective_density_mean` 的会话内标准差是 1.0、均值 10.0 →
    区间 9.0–11.0;`_n_rows` 100。

    这条同时钉住 `_std` 的取法 —— 两条分支都要活着:
    - 键以 `_mean` 结尾 → 查 `<基名>_std`(connective_density 走这条);
    - 键不带 `_mean`(如 `face_energy`)→ 只能查 `<键>_std`。
      删掉后一条分支,face_energy 会**静默**拿不到标准差:G2 退回 fail-open、
      区间变成"未采集到会话内变异信息" → 下面的断言变红。Task 6 把密度键从
      「无 _mean 后缀」改成了「带 _mean」,故第二条分支改由 face_energy 单独钉住。
    """
    result = ResearchCapabilityMapper().map_features_to_scores(_MIXED)
    slot = result["coverage"]["passed_slots"][0]

    assert slot["observed_interval"] == [9.0, 11.0], \
        f"区间未由本场会话的均值/标准差构成:{slot['observed_interval']}"
    assert slot["n_valid"] == 100

    html = _rendered_html(result, _MIXED)
    assert "9.0 – 11.0" in html, "会话内实测区间没有渲染出来"
    # 反向:不得出现任何暗示人群位置的表述
    for claim in ("百分位", "优于", "Top", "分位"):
        assert claim not in html, f"报告出现人群位置表述:{claim}"

    # 无 _mean 后缀的键:同一个 _std 取法,`<键>_std` 那条分支
    bare = {"face": {"face_energy": 0.5, "face_energy_std": 0.1, "_n_rows": 100.0}}
    dim = ResearchCapabilityMapper().map_features_to_scores(bare)["dimensions"]["confidence_level"]
    ev = next(e for e in dim["evidence_chain"] if e["human_name"] == "语音能量")
    assert ev["feature"] == "face_energy", f"钉的键变了:{ev['feature']}"
    assert ev["observed_interval"] == [0.4, 0.6], (
        f"不带 _mean 的键没取到伴随标准差:{ev['observed_interval']}"
    )


def test_deep_analysis_has_no_banned_words():
    """spec §5.4 + §5.6:叙事层不得含情绪/心理/诚信构念。

    ⚠️ 只覆盖**渲染出来的那一段 HTML**。全仓字符串字面量由
    `test_no_banned_words_in_output_strings` 守(那个覆盖面更大)。

    报告里出现的禁止词有两个来源,本测试两者都扫:
    1. 叙事层自己写的句子(report_generator);
    2. `research_mapper` 提供的标签 —— `display_name` 与 `human_name`,
       它们经 `_render_dimension_block` 进指标表与缺口清单。

    来源 2 的两个旧标签「抗压与情绪稳定性」「面部紧张度」曾由 Task 6
    以精确字符串替换剔除(否则本测试结构性不可满足);Task 7 改完
    `display_name` 与 `human_name` 后,剔除逻辑已删除,恢复全量扫描。
    **不要再把标签剔除加回来** —— 那会让本测试对 mapper 标签回归失明。

    ⚠️ fixture 必须是 _MIXED:禁止词原本住在出分路径上,零证据输入只扫"无证据"那几句。
    """
    from report_frontend.report_generator import ReportGenerator

    result = ResearchCapabilityMapper().map_features_to_scores(_MIXED)
    # 断言 fixture 真的走出了分路径 —— 否则本测试退化为只扫无证据文案
    assert result["total_score"] is not None, "fixture 未出分,本测试退化为空断言"
    html = ReportGenerator()._generate_deep_text_analysis(_MIXED, result)

    for word in BANNED:
        assert word not in html, f"叙事层出现禁止词:{word}"


def test_deep_analysis_handles_none_scores():
    """score 为 None 的维度不得参与排序/取最大值,且仍要渲染成"证据不足"。

    ⚠️ fixture 必须是 _MIXED:零证据输入会让 total_score 直接为 None,
    根本走不到 max(..., key=CONF_ORDER.get) 那一行,本测试就守不住它。
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


# 全仓扫描范围。**以本文件位置为准**,不用裸相对路径 —— 后者会让扫描
# 静默依赖当前工作目录:在仓库外跑 pytest 时 SCOPE 全部不存在,
# rglob 返回空,offenders 恒空,本测试对任何实现都通过。
_REPO_ROOT = Path(__file__).resolve().parents[1]
SCOPE = [_REPO_ROOT / "report_frontend", _REPO_ROOT / "templates"]

# 扫描的文件类型。原来只有 *.py —— 而「生成综合判推报告」这类过度承诺恰恰住在
# `templates/*.html`,折算因子住在 `evidence_thresholds.json`,两者都在扫描盲区里。
# .py 走 AST(注释豁免),其余按行读文本(HTML/JSON 没有注释豁免的概念)。
_SCAN_PATTERNS = ("*.py", "*.html", "*.json")

# 每个根目录各自的文件数下限(实测 report_frontend 9 = 8 py + 1 json,
# templates 2 = dashboard.html + __init__.py)。**逐根检查**:只看总数时,
# 整个根目录消失仍可能被另一个根目录的数量掩盖过去。
_MIN_FILES_PER_ROOT = {"report_frontend": 5, "templates": 2}

# 运行时生成的产物目录(如 report_frontend/data/output 下 66 MB 的 plotly 图表)不扫:
# 它们的字符串来自本扫描已覆盖的 visualizer/report_generator,扫它们不增加覆盖面,
# 只会把测试绑在生成物上。目录名 "data" 是生成产物的约定(仓库根同样如此)。
_SKIP_DIR_NAMES = {"data"}


def _scan_targets(root: Path):
    for pattern in _SCAN_PATTERNS:
        for path in sorted(root.rglob(pattern)):
            if _SKIP_DIR_NAMES & set(path.relative_to(root).parts[:-1]):
                continue
            yield path


def _string_literals(path: Path):
    """返回 [(文本, 行号)]。.py 只取字符串字面量(注释不算),其余整行取。"""
    if path.suffix == ".py":
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return [(node.value, node.lineno) for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    return [(line, i) for i, line in
            enumerate(path.read_text(encoding="utf-8").splitlines(), 1)]


def test_dashboard_report_card_describes_what_it_produces():
    """spec §5.4/§5.6:总控台不得再承诺「判推」「深度评估」—— 本分支已把这两样删掉。

    报告按钮是**用户可见的过度承诺**:它写的正是报告层刚被删掉的那两样东西
    (判推 = 解读推断,深度评估 = 对人的评定)。而它现在产出的只是过门指标与证据缺口。
    """
    text = (_REPO_ROOT / "templates" / "dashboard.html").read_text(encoding="utf-8")

    for claim in ("判推", "深度评估", "评估文书"):
        assert claim not in text, f"总控台仍宣称产出「{claim}」"
    assert "行为观测报告" in text, "按钮文案未改成它真正产出的东西"


def test_no_banned_words_in_output_strings():
    """spec §5.6:禁止词不得出现在**任何输出字符串**里(含 docstring、HTML、JSON)。

    .py 用 AST:`ast.Constant` 只覆盖字面量,所以说明性注释不受本测试约束 ——
    维护者解释"为什么删掉某个词"的注释是允许的。.html/.json 按行读文本。

    这是 Task 7 的全仓底线:维度名、描述、提示语、控制台 print、模板文案、阈值登记
    全在内,而 `test_deep_analysis_has_no_banned_words` 只守渲染出来的那一段 HTML。

    禁止词表复用模块级 BANNED(Task 6 引入),不另立同名常量。
    """
    offenders = []
    scanned = {}
    for root in SCOPE:
        count = 0
        for path in _scan_targets(root):
            count += 1
            for text, lineno in _string_literals(path):
                for word in BANNED:
                    if word in text:
                        offenders.append(f"{path}:{lineno} {word}")
        scanned[root.name] = count

    for root_name, floor in _MIN_FILES_PER_ROOT.items():
        assert scanned.get(root_name, 0) >= floor, (
            f"{root_name} 只扫到 {scanned.get(root_name, 0)} 个文件(<{floor}),本测试退化为空断言"
        )
    assert not offenders, "输出字符串含禁止词:\n" + "\n".join(offenders)
