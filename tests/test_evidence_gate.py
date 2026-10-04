# tests/test_evidence_gate.py
import pytest

from report_frontend.evidence_gate import (
    Check,
    _threshold_for,
    confidence_from,
    gate,
    is_quarantined,
    load_thresholds,
)


class TestG1HasValue:
    def test_none_fails(self):
        assert gate("pitch_median", None, 100).ok is False

    def test_nan_fails(self):
        assert gate("pitch_median", float("nan"), 100).ok is False

    def test_number_passes(self):
        assert gate("pitch_median", 120.0, 100).ok is True


class TestG2NotConstant:
    def test_single_value_is_constant(self):
        c = gate("pitch_median", 1.0, 100, values=[1.0] * 50)
        assert c.ok is False
        assert c.gate_name == "G2"

    def test_all_zero_is_constant(self):
        assert gate("pitch_median", 0.0, 100, values=[0.0] * 50).ok is False

    def test_varying_passes(self):
        assert gate("pitch_median", 1.0, 100, values=[1.0, 2.0, 3.0]).ok is True


class TestG2FromStd:
    """报告层只有标量,用伴随的 _std 判定常量性。"""

    def test_zero_std_is_constant(self):
        c = gate("pitch_median", 100.0, 500, std=0.0)
        assert c.ok is False
        assert c.gate_name == "G2"

    def test_nonzero_std_passes(self):
        assert gate("pitch_median", 100.0, 500, std=0.3).ok is True

    def test_missing_std_does_not_block(self):
        assert gate("pitch_median", 100.0, 500, std=None).ok is True


class TestG3EnoughSamples:
    # ⚠️ 不能用 blink_* 系列:阈值 20 只由 "blink" 产出,而所有含 blink 的键都在封停
    # 名单里,会被 G4 拦下 —— 该边界用任何未被封停的键都测不到。改用 pitch
    # (`pitch_median` 未被封停)。
    #
    # ⚠️ 2026-10-03:这两个数**原来是写死的 9 / 10**(`pitch` 当时的阈值是 10)——
    #    甲-2 把 `pitch` 改成 5 之后它们就红了。红得好:它俩测的是「**门槛两侧**的行为」,
    #    与那个具体数字无关。现在从登记表推导,门槛再改也不会让这条测试失效,
    #    而它要钉的东西(差一个就拦、刚好就放)一个字都没少。
    _T = _threshold_for("pitch_median")

    def test_below_threshold_fails(self):
        c = gate("pitch_median", 120.0, self._T - 1, values=[1.0, 2.0])
        assert c.ok is False
        assert c.gate_name == "G3"

    def test_at_threshold_passes(self):
        assert gate("pitch_median", 120.0, self._T, values=[1.0, 2.0]).ok is True

    def test_the_boundary_is_not_vacuous(self):
        """门槛必须 >1,否则「差一个就拦」这条边界测的是 0 —— 与 G1「无值」撞车。"""
        assert self._T > 1, f"pitch 族门槛退化成了 {self._T}"


class TestG4NotProxy:
    def test_quarantined_column_fails(self):
        c = gate("face_focus_score_mean", 0.3, 500, values=[0.3, 0.4])
        assert c.ok is False
        assert c.gate_name == "G4"

    def test_permanent_quarantine_has_no_unblock(self):
        q = is_quarantined("overall_score")
        assert q is not None and q.permanent is True

    def test_clean_column_passes(self):
        # 不能用 interview_pause_duration_mean —— "pause_duration" 是它的子串,会被 G4 拦。
        assert gate("interview_duration_mean", 0.8, 500,
                    values=[0.7, 0.9]).ok is True

    def test_permanent_entries_hit_the_real_flattened_keys(self):
        """★ E7:两条永久封停此前打不中**目标**,于是一个本该被拦下的量看着正常。

        真键形如 `<模态>_<基名>_<统计量>`(实测:一场会话 579 个扁平键里 573 个带
        `_mean/_std/_min/_max/_sum/_trend` 后缀)。这两条写的是旧代码里的名字
        (`upper_body_head_tilt` / `shoulder_is_calibrated`),而真键是
        `gesture_head_tilt_*` / `gesture_is_calibrated_*` ⟹ 子串匹配恒不命中,
        `is_quarantined` 恒返回 `None`。

        红法:把键名改回 `upper_body_head_tilt` / `shoulder_is_calibrated` ⟹ 立刻红。
        """
        for key in ("gesture_head_tilt_mean", "gesture_is_calibrated_mean"):
            q = is_quarantined(key)
            assert q is not None, f"{key} 未被封停(E7:封停键与扁平化真键对不上)"
            assert q.permanent is True, f"{key} 该是永久封停"

    def test_head_tilt_does_not_swallow_head_tilt_angle(self):
        """★ 天真的改法会**误封一个要留的列**:`head_tilt` 是 `head_tilt_angle` 的子串。

        两者是**不同的列**,不是一个列的两个名字:
          · `head_tilt`       —— 参考系错位 180 度、分支命中率 0%,永久封停;
          · `head_tilt_angle` —— L0 表里 `maturity=A` / `status=implemented` 的**保留列**。

        子串匹配分不开它们 ⟹ 该条目必须带 `not_substrings` 豁免。
        红法:去掉豁免(即只改名、不加 `not_substrings`)⟹ 本测试立刻红,
        且是一个**静默**的错误结果:保留列被封停、报告里少一个指标,不报错。
        """
        for key in ("gesture_head_tilt_angle_mean", "gesture_head_tilt_angle_trend"):
            assert is_quarantined(key) is None, (
                f"{key} 是 maturity=A 的保留列,不该被封停 —— "
                f"被 `head_tilt` 的子串匹配误伤了")

    def test_not_substrings_only_shields_the_named_keys(self):
        """豁免必须**只**挡它点名的那个子串,不能顺手把整条封停废掉。"""
        assert is_quarantined("gesture_head_tilt_mean") is not None, "目标列该被拦"
        assert is_quarantined("gesture_is_calibrated_mean") is not None, "目标列该被拦"
        # 同条目族里不含 `head_tilt_angle` 的键,照旧被拦
        assert is_quarantined("gesture_head_tilt_max") is not None, "同族未被豁免的该被拦"


class TestConfidence:
    def test_zero_is_wu(self):
        assert confidence_from(0, 4) == "无"

    def test_minority_is_di(self):
        assert confidence_from(1, 4) == "低"

    def test_majority_is_zhong(self):
        assert confidence_from(3, 4) == "中"

    def test_high_is_unreachable_this_round(self):
        """① 阶段无外部效标验证,'高' 不可达(spec §5.1)。"""
        for n_ok in range(0, 10):
            assert confidence_from(n_ok, 10) != "高"


class TestUserMessage:
    def test_user_message_hides_maintainer_text(self):
        """报告层只能用 user_message;封停理由等内部文案不得外泄(spec §5.6)。"""
        from report_frontend.evidence_gate import user_message

        c = gate("face_focus_score_mean", 0.3, 500, values=[0.3, 0.4])
        assert c.ok is False
        msg = user_message(c)
        assert msg, "缺口文案不得为空"
        for leak in ("封停", "jitter", "M3", "精确重构"):
            assert leak not in msg, f"内部文案外泄:{leak}"

    def test_user_message_per_gate(self):
        from report_frontend.evidence_gate import user_message

        assert user_message(gate("pitch_median", None, 100)) == "未采集到对应数据"
        assert user_message(gate("pitch_median", 1.0, 100, std=0.0)) == "本次会话内无变化"
        assert user_message(gate("pitch_median", 1.0, 3, values=[1.0, 2.0])) == "有效样本不足"
        assert user_message(
            gate("face_focus_score_mean", 0.3, 500, values=[0.3, 0.4])
        ) == "该指标本轮停用"


class TestThresholds:
    def test_thresholds_are_marked_provisional(self):
        t = load_thresholds()
        assert t["_provisional"] is True
        assert t["_version"]

    def test_default_threshold_lives_in_json(self):
        """未登记指标的默认阈值也必须是登记过的临时值,不得是代码里的裸常量。"""
        from report_frontend.evidence_gate import _threshold_for

        data = load_thresholds()
        assert "_default_n_valid" in data
        assert _threshold_for("some_unregistered_metric") == data["_default_n_valid"]

    def test_longest_match_wins(self):
        """阈值表里 "au" 早于 "pause" 且是其子串(p-au-se)。

        若按"注册表第一个匹配"实现,pause_* 会静默拿到 au 的阈值 30 而不是 2。
        必须与 is_quarantined 一样按最长匹配。
        """
        from report_frontend.evidence_gate import _threshold_for

        assert _threshold_for("pause_duration_mean") == 2
        assert _threshold_for("au12_smile_mean") == 30

    def test_threshold_family_lookup_is_the_single_implementation(self):
        """取族名与取门槛值必须是**同一套**匹配 —— 否则两处会漂。

        `threshold_family_for` 是 2026-10-03 抽出来的:死权重那条守卫要判
        「这个槽拿到的门槛是不是帧级的」,而"单位"是按**族**登记的,所以它必须拿到
        **族名**。抽的时候刻意让 `_threshold_for` 也走它 —— 一个匹配规则一处实现。
        """
        from report_frontend.evidence_gate import _threshold_for, threshold_family_for

        for key in ("face_au4_frown_mean", "pause_duration_mean", "some_unregistered"):
            fam = threshold_family_for(key)
            data = load_thresholds()
            expected = (data["thresholds"][fam] if fam else data["_default_n_valid"])
            assert _threshold_for(key) == expected, f"{key} 的族名与门槛值不是同一套匹配算的"


def test_answer_level_families_are_not_gated_by_the_frame_level_default():
    """★ 门槛的**单位**必须与指标的**单位**同量纲 —— 甲-2 那一类缺陷的守卫。

    (⚠️ 刻意写成**模块级函数**而不是 `TestThresholds` 的方法:`tests/test_mutation_registry.py`
    用 `nodeid.partition("::")[2]` 去 `def test_...` 里找定义,类方法那个 `Class::method`
    形式它认不出 —— 而这条必须有变异实测。)

    为什么需要它:`_default_n_valid` = 10 是**帧级**列设的(一场几百帧,10 很容易过),
    而语音日志 **一行 = 一段回答**(题库每场 **8 题** ⟹ 答满全场也才 8 段)。
    帧级门槛被 answer 级指标继承 ⟹ `8 < 10` **结构性不可达** ⟹ 该指标永远显示
    「有效样本不足」。一个永不达标的门槛**不是**"更严的诚实门",它等价于该指标
    永久不可用 —— 而它长得跟"数据不够"一模一样,所以不测就看不出来。

    实证(`density`/`n_chars`/`reaction_time` 三族 2026-10-03 已修,`pitch`/`energy`
    当时**漏改**):`communication_fluency` 唯一那个看起来活的槽 `pitch_variation`
    实测被 G3 永久拦下 —— 场 `20260927_212143_95a8` 有效样本 **8 < 阈值 10**,
    整个维度 **0/3**。

    红法(**已实测**,两条变异都登记在 `tests/test_mutation_registry.py`):
    - 把 `thresholds.pitch` 改回 10 ⟹ 第一条断言红(10 不小于帧级默认),
      且**另一条独立判据**也红 —— `tests/test_dimension_live_weight.py` 的死权重
      5/7 → 7/7;
    - 往 `n_valid_units` 里新增一个 `answer` 族但不给门槛/依据 ⟹ 也红。
    """
    from report_frontend.evidence_gate import load_thresholds

    data = load_thresholds()
    thresholds, units = data["thresholds"], data["n_valid_units"]
    default = data["_default_n_valid"]

    # ① 登记在 thresholds 里的族必须都有单位 —— 单位只能显式写,不许靠"没写就当 frame"
    #    (那条兜底是给**未登记**的键用的,不是给已登记的族用的)
    missing_units = [f for f in thresholds if f not in units]
    assert not missing_units, (
        f"这些阈值族没写 n_valid 单位:{missing_units} —— "
        f"`n_valid_units` 是「门槛单位」这件事的唯一登记处,漏写等于让下一条断言失去对象")

    answer_fams = [f for f, u in units.items() if u == "answer"]
    assert answer_fams, "一个 answer 族都没有 —— 那这条测试退化成空断言"
    bad = [f"{f}: 门槛 {thresholds[f]} ≥ 帧级默认 {default}"
           for f in answer_fams if thresholds[f] >= default]
    assert not bad, (
        "这些族是**每段回答一个样本**,却拿着帧级门槛 —— 结构性不可达"
        "(题库每场 8 题,答满也才 8 段):\n  " + "\n  ".join(bad))

    # ② answer 族必须有依据登记:门槛是**改过**的数,改动必须留下理由
    unanswered = [f for f in answer_fams if f not in data["_thresholds_basis"]]
    assert not unanswered, (
        f"这些 answer 族的门槛没有依据登记:{unanswered} —— "
        f"门槛是临时值,依据必须逐族写在 `_thresholds_basis` 里(spec §5.1)。"
        f"⚠️ frame 族那 4 个(blink/gaze/au/head_pose)是本仓**已知欠账**,"
        f"只对 answer 族强制 —— 因为只有 answer 族有「单位错配」这一类缺陷。")

    # ③ 把 甲-2 的那次修复本身钉住:数改了没有依据不行,依据写错也不行
    assert thresholds["pitch"] == 5, (
        "`pitch` 族是**每段回答一个样本**,门槛必须是 5(≈ 半场答题量);"
        "退回帧级默认 10 ⟹ `pitch_variation` 被 G3 永久拦下")
    assert "_provisional" in data["_thresholds_basis"]["pitch"], (
        "`pitch` 的门槛是临时值,依据条目必须带 `_provisional`")


class TestGateOrder:
    def test_first_failure_wins(self):
        """G1 先于 G3。"""
        c = gate("blink_rate", None, 0)
        assert c.gate_name == "G1"
