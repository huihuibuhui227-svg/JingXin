# tests/test_sample_size_honesty.py
"""I1(最终全分支审查):「有效样本量」必须是**真正带值**的行数,不是日志文件的行数。

**失效形态(修复前,已复现):** `research_mapper.map_features_to_scores()` 从
`n_rows_by_modality`(`feature_engine` 产出的 `<模态>__n_rows` = **CSV 行数**)取
`n_valid`,对**每一个**指标都用同一个数。而语音日志里「回答太短」的那一行,密度列是
**空格**(`connective_density` 为 None 时不写 0,spec §8),特征引擎按 `dropna()` 把它
丢掉 —— 但模态行数仍然把它算进去。后果两件:

1. 报告在同一行上印「有效样本量 20」,而那个均值只由 3 行构成;
2. **一行都不带的日志也能过 G3 门槛**(默认 5 / density 族 5)—— 门槛是 M1 存在的
   理由(诚实轴),在错的单位上它就成了一枚橡皮图章。

修法:优先取该指标**自己**的 `<基名>_sample_size`(特征引擎按 dropna 后的行数产出),
取不到才退回模态行数。
"""

from pathlib import Path

import pytest

from report_frontend.data_loader import LogDataLoader
from report_frontend.feature_engine import PsychologicalFeatureEngine
from report_frontend.research_mapper import ResearchCapabilityMapper

_SID = "20260924_153012_9f3c"
_FLOOR = 5          # evidence_thresholds.json 里 density 族的 n_valid 门槛(Ruling M1-28)


def _voice_log(tmp_path: Path, values: list) -> Path:
    """用真的 VoiceLogger 写一份语音日志:`None` 的行就是「回答太短、密度不出值」。

    走生产写法(而不是手搓 CSV),这样"空格 vs 0"的口径由 logger 决定,
    不会与本测试的假设悄悄漂移。
    """
    from voice_interaction.utils.logger import VoiceLogger

    lg = VoiceLogger(log_type="interview", log_dir=str(tmp_path), session_id=_SID)
    for i, v in enumerate(values):
        lg.log_prosody({}, question_index=i, emotion="", feedback="",
                       connective_density=v, connective_density_std=0.0, n_rows=1)
    return lg.csv_file


def _density_slot(tmp_path: Path, values: list):
    """把日志一路推到映射层,返回 (logical_thinking 维度, 特征字典)。"""
    _voice_log(tmp_path, values)
    data = LogDataLoader(str(tmp_path)).get_fused_latest_data()
    assert "voice_interview" in data, "语音日志没被装载,本测试的前提不成立"
    feats = PsychologicalFeatureEngine(data).extract_all_features()
    dim = ResearchCapabilityMapper().map_features_to_scores(feats)["dimensions"]["logical_thinking"]
    return dim, feats


def _slot(dim):
    return next((ev for ev in dim["evidence_chain"] if ev["human_name"] == "连接词密度"), None)


def test_sample_size_is_the_rows_that_carry_a_value(tmp_path):
    """20 行日志、只有 6 行带值 → 印出来的样本量必须是 **6**,不是 20。

    红在(修复前实测):`n_valid` 取模态行数(20)→ 断言 `6 == 20` 失败。
    这条只钉数字,不碰门槛(6 ≥ 5,槽照样过门)。
    """
    values = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0] + [None] * 14
    dim, _ = _density_slot(tmp_path, values)

    ev = _slot(dim)
    assert ev is not None, f"槽没过门,本测试退化为空断言。缺口:{dim['evidence_gaps']}"
    assert ev["n_valid"] == 6, (
        f"有效样本量报成了日志行数:{ev['n_valid']} —— 这个均值只由 6 行构成"
    )


def test_rows_that_carry_nothing_cannot_clear_the_sample_floor(tmp_path):
    """20 行日志、只有 3 行带值 → 3 < 门槛 5 → 槽**必须**被 G3 拦下。

    红在(修复前实测):模态行数 20 ≥ 5 → 门槛被 17 行"什么都没带"的行放行
    → 证据链里出现「连接词密度」→ 断言 `ev is None` 失败。
    这正是"把橡皮图章盖在空行上":门槛是 M1 存在的理由,单位错了它就不起作用。
    """
    values = [3.0, 4.0, 5.0] + [None] * 17
    assert sum(1 for v in values if v is not None) < _FLOOR, "本测试的前提是「带值的行 < 门槛」"

    dim, _ = _density_slot(tmp_path, values)

    assert _slot(dim) is None, (
        "只有 3 行带值却过了样本量门槛 —— 门槛被什么都不带的行放行了"
    )
    assert any("连接词密度" in g and "样本" in g for g in dim["evidence_gaps"]), (
        f"槽被拦下了,但缺口里没说清是样本量不够:{dim['evidence_gaps']}"
    )


def test_sample_size_falls_back_to_modality_rows_when_the_indicator_has_none(tmp_path):
    """没有 `<基名>_sample_size` 的键(如 `face_eye_contact_ratio`)仍要用模态行数。

    这是修复的**另一侧**:优先取指标自己的样本量,但不能因此把退回路径弄丢
    —— 退回路径断了的话,那些没有 `_sample_size` 的指标会一律拿到 0、被 G3 无差别拦下。
    """
    feats = {"voice_research": {"connective_density_mean": 3.5,
                                "connective_density_std": 0.4,
                                "_n_rows": 8.0}}
    dim = ResearchCapabilityMapper().map_features_to_scores(feats)["dimensions"]["logical_thinking"]
    ev = _slot(dim)
    assert ev is not None, f"没有 _sample_size 时退回路径断了:{dim['evidence_gaps']}"
    assert ev["n_valid"] == 8, f"退回路径没有用模态行数:{ev['n_valid']}"


def test_indicator_sample_size_does_not_leak_across_indicators(tmp_path):
    """同一模态里两个指标各有各的样本量 —— 不能一个指标的数字给另一个用。

    face 的两个键给不同的 `_sample_size`(而模态行数是 30),断言各自拿到自己的那个。
    红法:把取法换成"每个模态只算一次"(例如在模态循环外算好再复用)。
    """
    feats = {"face": {
        "face_focus_score_mean": 0.5, "face_focus_score_std": 0.1,
        "face_focus_score_sample_size": 30.0,
        "_n_rows": 30.0,
    }, "voice_research": {"connective_density_mean": 3.5,
                          "connective_density_std": 0.4,
                          "connective_density_sample_size": 7.0,
                          "_n_rows": 20.0}}
    res = ResearchCapabilityMapper().map_features_to_scores(feats)
    got = {ev["feature"]: ev["n_valid"]
           for d in res["dimensions"].values() for ev in d["evidence_chain"]}
    density_key = next((k for k in got if "connective_density" in k), None)
    assert density_key is not None, f"密度槽没过门:{res['evidence_gaps']}"
    assert got[density_key] == 7, f"密度拿了别人的样本量:{got}"
