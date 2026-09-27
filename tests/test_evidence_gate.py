# tests/test_evidence_gate.py
import pytest

from report_frontend.evidence_gate import (
    Check,
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
    # 名单里,会被 G4 拦下 —— 该边界用任何未被封停的键都测不到。改用 pitch(阈值 10,
    # pitch_median 未被封停)。

    def test_below_threshold_fails(self):
        c = gate("pitch_median", 120.0, 9, values=[1.0, 2.0])
        assert c.ok is False
        assert c.gate_name == "G3"

    def test_at_threshold_passes(self):
        assert gate("pitch_median", 120.0, 10, values=[1.0, 2.0]).ok is True


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


class TestGateOrder:
    def test_first_failure_wins(self):
        """G1 先于 G3。"""
        c = gate("blink_rate", None, 0)
        assert c.gate_name == "G1"
