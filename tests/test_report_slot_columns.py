# tests/test_report_slot_columns.py
"""报告层槽位的**列对账**钉子(2026-09-27)。

判据只有一条:**新列是同一个构念的更好测法** —— 不是"它能过门"。
每颗钉子钉住三件事,缺一条就算没钉:

  1. **槽接到了哪一列**(结构级:keyword/权重/方向)+ 行为级:喂该列 ⇒ 它进证据链;
  2. **方向**(`normalized_score` 随原始值往哪边走)—— 换了列就换了量纲,方向错了会
     **静默**把「更糟」算成「更好」,而分数照出;
  3. **旧列不再被这个槽命中** —— 喂新旧两列,**同一份输入**里只能进新的那一列。

⚠️ 第 3 条必须两边都喂。`_fuzzy_match` 是**返回第一个命中的键**的纯子串匹配,只喂
新列时,旧 keyword(`jitter`)照样能命中新列(`left_wrist_jitter_world`)⟹ 钉子
新旧都绿,等于没钉(本文件第一版就是这个毛病,已修)。

⚠️ 为什么方向必须单独钉:本组有 3 个槽的量纲从「越大越糟」翻成了「越大越好」
(皱眉=眉内距**越小**越皱;唇压=唇红**越薄**越压;耸肩=幅度越大越耸)。
`mapping_rules` 的第三元 `is_positive` 是**语义**位,不是装饰位。
"""

from report_frontend.research_mapper import ResearchCapabilityMapper


def _run(feats):
    return ResearchCapabilityMapper().map_features_to_scores(feats)["dimensions"]


def _chain(dim):
    return {ev["feature"]: ev for ev in dim["evidence_chain"]}


def _slot(dim_key, keyword):
    """这一维里那个槽的三元组(keyword, weight, is_positive)。"""
    for kw, weight, pos, _name, _imp in ResearchCapabilityMapper().mapping_rules[dim_key]["indicators"]:
        if kw == keyword:
            return weight, pos
    raise AssertionError(f"{dim_key} 里没有槽 {keyword}")


# ---------------------------------------------------------------- 皱眉 / au4
def test_frown_slot_measures_brow_furrow_geometrically():
    """`au4_freq` → `au4_frown`(双眉内侧点距 ÷ 眼距),权重 0.1 不变。

    `au4_freq` 想量的构念就是**皱眉**,而它是「从 micro_exp 字符串解析出的垃圾匹配」;
    `au4_frown` 是同一构念的真几何量。
    """
    assert _slot("logical_thinking", "au4_frown") == (0.1, True)
    dim = _run({"face": {"au4_frown_mean": 0.32, "au4_frown_std": 0.01,
                         "_n_rows": 300.0}})["logical_thinking"]
    assert "face_au4_frown_mean" in _chain(dim), "皱眉槽没有指到几何量"


def test_frown_slot_points_the_right_way_for_a_distance_measure():
    """方向钉子:au4_frown 是**距离比** —— 值越小眉越皱(越糟)。

    沿用旧槽的 `is_positive=False`(越大越糟)会让眉越皱得分越高。
    """
    many = _run({"face": {"au4_frown_mean": 0.45, "au4_frown_std": 0.01,
                          "_n_rows": 300.0}})["logical_thinking"]
    few = _run({"face": {"au4_frown_mean": 0.05, "au4_frown_std": 0.01,
                         "_n_rows": 300.0}})["logical_thinking"]
    assert _chain(many)["face_au4_frown_mean"]["normalized_score"] > \
        _chain(few)["face_au4_frown_mean"]["normalized_score"], \
        "眉内距越大(=越不皱)必须得分越高;反了就是把「更糟」算成「更好」"


# ------------------------------------------------- 张力 / 唇部压缩(真几何量)
def test_tension_slot_uses_real_lip_compression():
    """`tension_score`(伪合成)→ `au23_lip_compression`(唇红厚 ÷ 面部轮廓高),0.3 不变。

    槽的构念原文就是「眉间收缩与**唇部压缩**」;旧列是硬编码权重的伪合成,
    5 个分量里 eye_closure 恒 0。au23 是唇部压缩的真几何量。
    """
    assert _slot("stress_resilience", "au23_lip_compression") == (0.3, True)
    dim = _run({"face": {"au23_lip_compression_mean": 0.05,
                         "au23_lip_compression_std": 0.01,
                         "_n_rows": 300.0}})["stress_resilience"]
    assert "face_au23_lip_compression_mean" in _chain(dim)


def test_tension_slot_points_the_right_way_for_lip_red_thickness():
    """方向钉子:唇红**越薄** = 压得越紧 = 越糟 ⟹ 值越大得分越高。"""
    thick = _run({"face": {"au23_lip_compression_mean": 0.12,
                           "au23_lip_compression_std": 0.01,
                           "_n_rows": 300.0}})["stress_resilience"]
    thin = _run({"face": {"au23_lip_compression_mean": 0.02,
                          "au23_lip_compression_std": 0.01,
                          "_n_rows": 300.0}})["stress_resilience"]
    assert _chain(thick)["face_au23_lip_compression_mean"]["normalized_score"] > \
        _chain(thin)["face_au23_lip_compression_mean"]["normalized_score"], \
        "唇红越厚(=越没压)必须得分越高"


# ---------------------------------------------------------- 视线稳定性 / 抖动
def test_gaze_stability_slot_uses_variation_of_the_real_gaze_column():
    """`gaze_stability`(1/(1+std(gaze_deviation)),派生自被封停的列)→
    `gaze_direction_x_std`(视线水平偏移在会话内的标准差),0.2、反向。

    ⚠️ 那条封停自己写的解封条件就是「M3 修 gaze_direction_x」。
    稳定性 = 视线的**不变程度**,会话内标准差是它的直接测法;旧列量的是被 y 分量
    (解剖常量,gaze_direction_y 自身亦被封停)主导的那个向量的抖动。
    """
    assert _slot("logical_thinking", "gaze_direction_x_std") == (0.2, False)
    dim = _run({"face": {"gaze_stability_mean": 0.96,
                         "gaze_direction_x_mean": 0.0, "gaze_direction_x_std": 0.07,
                         "_n_rows": 300.0}})["logical_thinking"]
    chain = _chain(dim)
    assert "face_gaze_direction_x_std" in chain
    assert "face_gaze_stability_mean" not in chain, "槽还在用派生列"


def test_gaze_stability_slot_is_inverse_in_the_spread():
    """方向钉子:视线越散(std 越大)= 越不稳 = 越糟。"""
    steady = _run({"face": {"gaze_direction_x_std": 0.02,
                            "_n_rows": 300.0}})["logical_thinking"]
    wandering = _run({"face": {"gaze_direction_x_std": 0.8,
                               "_n_rows": 300.0}})["logical_thinking"]
    assert _chain(steady)["face_gaze_direction_x_std"]["normalized_score"] > \
        _chain(wandering)["face_gaze_direction_x_std"]["normalized_score"], \
        "视线越散必须得分越低"


def test_jitter_slot_uses_a_covered_framing_free_limb_jitter():
    """`jitter` → `left_wrist_jitter_world`,0.3、反向。

    理由不是"它能过门",是两条**测量学**理由:
      ① 手在画面里缺席 77–95% 的帧(三场实测 null=286/430/441),而腕的世界坐标
         三场全覆盖(null=0/3/1);
      ② 世界坐标是米制,取景无关 —— 旧列的封停理由正是「经 x5 归一化恒饱和」。
    两边同时喂:命中的必须是腕那一列(纯子串匹配下旧 keyword 也能命中新列,
    只喂新列的钉子是假的)。
    """
    assert _slot("stress_resilience", "left_wrist_jitter_world") == (0.3, False)
    dim = _run({"gesture": {"left_hand_jitter_mean": 0.0009, "left_hand_jitter_std": 0.001,
                            "left_wrist_jitter_world_mean": 0.0011,
                            "left_wrist_jitter_world_std": 0.0012,
                            "_n_rows": 300.0}})["stress_resilience"]
    chain = _chain(dim)
    assert "gesture_left_wrist_jitter_world_mean" in chain
    assert "gesture_left_hand_jitter_mean" not in chain, "槽还停在缺席 77–95% 的那只手上"


# --------------------------------------------- 手 / 肩(旧列已从落盘契约删掉)
def test_hand_slot_uses_the_normalised_palm_opening():
    """`hand_score` → `left_hand_spread`(指尖到掌根均值 ÷ 掌长),0.3 不变。

    旧列已从 `GestureLogger` 的落盘契约里删掉(可由同 block 的 jitter 精确重构);
    那个合成(jitter − fist 罚 + spread 奖)里唯一未被封停、且 B3 已补上归一化
    (÷ 掌长)的分量就是张开度。
    """
    assert _slot("confidence_level", "left_hand_spread") == (0.3, True)
    dim = _run({"gesture": {"left_hand_spread_mean": 1.37,
                            "left_hand_spread_std": 0.42,
                            "left_hand_spread_sample_size": 14,
                            "_n_rows": 300.0}})["confidence_level"]
    assert "gesture_left_hand_spread_mean" in _chain(dim)


def test_shoulder_slot_uses_shrug_level_with_relaxation_polarity():
    """`shoulder_score` → `shrug_level`(÷ 肩宽、基线=因果中位数),0.2、反向。

    槽的构念是**肩部放松度**,耸肩程度是同一部位的反向量;旧列是
    jitter + shrug 的确定性函数,分量各自已入槽。
    """
    assert _slot("confidence_level", "shrug_level") == (0.2, False)
    relaxed = _run({"gesture": {"shrug_level_mean": 0.0, "shrug_level_std": 0.05,
                                "_n_rows": 300.0}})["confidence_level"]
    shrugged = _run({"gesture": {"shrug_level_mean": 0.9, "shrug_level_std": 0.05,
                                 "_n_rows": 300.0}})["confidence_level"]
    assert "gesture_shrug_level_mean" in _chain(relaxed)
    assert _chain(relaxed)["gesture_shrug_level_mean"]["normalized_score"] > \
        _chain(shrugged)["gesture_shrug_level_mean"]["normalized_score"], \
        "耸肩越厉害(=越不放松)必须得分越低"


# ------------------------------------------------------------------ 不变量
def test_no_column_keyword_is_used_by_two_slots():
    """一个列被两个槽用 = 同一份测量数了两遍(封停理由里就点过这种病)。"""
    m = ResearchCapabilityMapper()
    seen = {}
    for dim_key, rule in m.mapping_rules.items():
        for kw, _w, _pos, _name, _imp in rule["indicators"]:
            assert kw not in seen, f"{kw} 同时被 {seen[kw]} 与 {dim_key} 用"
            seen[kw] = dim_key
