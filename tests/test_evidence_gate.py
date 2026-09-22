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


class TestThresholds:
    def test_thresholds_are_marked_provisional(self):
        t = load_thresholds()
        assert t["_provisional"] is True
        assert t["_version"]

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
