# tests/test_instrument_provenance.py
"""仪器来源披露的单元测试。

覆盖任务要求的四件事:**前缀映射 / 多仪器指标 / 空输入 / 未知列**,
外加三条本模块自己立的规矩(不改既有字段、不猜、描述计数)。

## 这个测试**不需要**仓库可导入

`patch/instrument_provenance.py` 与测试一起拷进 `~/JingXin/jingxin/tests/` 之前,
它只是一个裸文件。所以这里按文件路径加载,并支持三种落位:

1. `系统补强/patch/…`(原地跑,`tests/` 的兄弟目录);
2. 与测试同目录(拷成 `tests/instrument_provenance.py`);
3. `report_frontend/…`(正式应用后的位置,`import report_frontend.instrument_provenance`);
4. 环境变量 `JX_PATCH_DIR` 指定的目录。

哪种都没有 ⟹ **明确报错**,不 skip —— skip 会让"补丁文件丢了"变成一片绿。
"""

from __future__ import annotations

import copy
import importlib.util
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent


def _candidates(name: str) -> list[Path]:
    out = []
    env = os.environ.get("JX_PATCH_DIR")
    if env:
        out.append(Path(env) / f"{name}.py")
    out.append(_HERE.parent / "patch" / f"{name}.py")     # 原地:系统补强/patch/
    out.append(_HERE / f"{name}.py")                       # 拷进 tests/
    out.append(_HERE.parent / "report_frontend" / f"{name}.py")  # 应用后
    return out


def _load(name: str):
    for path in _candidates(name):
        if path.is_file():
            spec = importlib.util.spec_from_file_location(f"_jxp_{name}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    tried = "\n".join(f"  - {p}" for p in _candidates(name))
    raise ImportError(
        f"找不到 {name}.py。这个测试**必须**有它才能证明补丁逻辑,不 skip。试过:\n{tried}")


ip = _load("instrument_provenance")


# ── ① 前缀映射 ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("column,expected", [
    # blendshape:模型回归出来的 52 个系数
    ("bs_mouthSmileLeft", ip.INSTRUMENT_BLENDSHAPE),
    ("bs_browDownRight", ip.INSTRUMENT_BLENDSHAPE),
    ("bs_eyeBlinkLeft", ip.INSTRUMENT_BLENDSHAPE),
    ("bs__neutral", ip.INSTRUMENT_BLENDSHAPE),
    # 几何代理:视线 / 头姿 / 眨眼 / 手写 AU 比率
    ("gaze_direction_x", ip.INSTRUMENT_GEOMETRIC),
    ("gaze_deviation", ip.INSTRUMENT_GEOMETRIC),
    ("head_yaw", ip.INSTRUMENT_GEOMETRIC),
    ("head_pitch", ip.INSTRUMENT_GEOMETRIC),
    ("is_blink", ip.INSTRUMENT_GEOMETRIC),        # 后缀式名字,靠精确名登记
    ("au4_frown", ip.INSTRUMENT_GEOMETRIC),
    ("au12_smile", ip.INSTRUMENT_GEOMETRIC),
    ("avg_ear", ip.INSTRUMENT_GEOMETRIC),
    # 音频
    ("pitch_series", ip.INSTRUMENT_AUDIO),
    ("pitch", ip.INSTRUMENT_AUDIO),               # 裸列名,前缀 `pitch_` 匹不到
    ("energy_median", ip.INSTRUMENT_AUDIO),
    ("energy", ip.INSTRUMENT_AUDIO),
    ("voiced_flag", ip.INSTRUMENT_AUDIO),
    ("pause_duration", ip.INSTRUMENT_AUDIO),
    ("speech_duration_sec", ip.INSTRUMENT_AUDIO),
    ("reaction_time", ip.INSTRUMENT_AUDIO),
    # 文本
    ("transcript_text", ip.INSTRUMENT_TEXT),
    ("n_chars", ip.INSTRUMENT_TEXT),
])
def test_column_maps_to_expected_instrument(column, expected):
    assert ip.instrument_of_column(column) == expected


def test_prefix_table_covers_the_four_required_groups():
    """任务给定的最小集合必须在表里,而且**不许**被后来的补充登记覆盖掉。"""
    table = ip.INSTRUMENT_OF_PREFIX
    assert table["bs_"] == ip.INSTRUMENT_BLENDSHAPE
    for p in ("gaze_", "head_", "blink", "au_"):
        assert table[p] == ip.INSTRUMENT_GEOMETRIC, p
    for p in ("pitch_", "energy_", "voiced_", "pause_"):
        assert table[p] == ip.INSTRUMENT_AUDIO, p
    assert table["transcript"] == ip.INSTRUMENT_TEXT


def test_longest_prefix_wins():
    """`bs_` 与更长前缀同时存在时取长的 —— 加前缀不该悄悄改写既有映射。"""
    saved = dict(ip.INSTRUMENT_OF_PREFIX)
    try:
        ip.INSTRUMENT_OF_PREFIX["bs_mouth"] = ip.INSTRUMENT_GEOMETRIC
        assert ip.instrument_of_column("bs_mouthSmileLeft") == ip.INSTRUMENT_GEOMETRIC
        assert ip.instrument_of_column("bs_jawOpen") == ip.INSTRUMENT_BLENDSHAPE
    finally:
        ip.INSTRUMENT_OF_PREFIX.clear()
        ip.INSTRUMENT_OF_PREFIX.update(saved)


# ── ② 多仪器指标 ──────────────────────────────────────────────────────────────

def test_column_set_unions_instruments():
    """`speech_rate = n_chars ÷ speech_duration_sec`:一头转写、一头音频。"""
    assert ip.instrument_of_columns(["n_chars", "speech_duration_sec"]) == {
        ip.INSTRUMENT_TEXT, ip.INSTRUMENT_AUDIO}


def test_multi_instrument_indicator_gets_list_not_scalar():
    ind = {"name": "speech_rate", "family": "V1_amount",
           "required_l0": ["n_chars", "speech_duration_sec"], "value": 3.1}
    (row,) = ip.annotate_indicators([ind])
    assert row["instruments"] == [ip.INSTRUMENT_AUDIO, ip.INSTRUMENT_TEXT]  # 定序
    assert "instrument" not in row            # 单值键不该同时出现
    assert ip.instruments_of(row) == [ip.INSTRUMENT_AUDIO, ip.INSTRUMENT_TEXT]


def test_single_instrument_indicator_gets_scalar():
    ind = {"name": "blink_rate_per_min", "required_l0": ["is_blink", "timestamp"]}
    (row,) = ip.annotate_indicators([ind])
    assert row["instrument"] == ip.INSTRUMENT_GEOMETRIC
    assert "instruments" not in row
    # 时钟列不该把 `is_blink` 的归属搅成多仪器
    assert ip.instruments_of(row) == [ip.INSTRUMENT_GEOMETRIC]


def test_both_face_instruments_stay_distinguishable():
    """两套面部仪器**不能**被合并成同一个名字 —— 这正是本补丁要防的事。"""
    a = ip.instrument_of_columns(["au12_smile"])
    b = ip.instrument_of_columns(["bs_mouthSmileLeft", "bs_mouthSmileRight"])
    assert a == {ip.INSTRUMENT_GEOMETRIC}
    assert b == {ip.INSTRUMENT_BLENDSHAPE}
    assert not (a & b)


# ── ③ 空输入 ──────────────────────────────────────────────────────────────────

def test_empty_indicator_list():
    assert ip.annotate_indicators([]) == []
    assert ip.describe([]) == {name: 0 for name in ip.INSTRUMENTS}


def test_none_indicator_list_is_tolerated():
    """`annotate_indicators(None)` 不该炸:报告在"本场一条指标都没有"时会传空。"""
    assert ip.annotate_indicators(None) == []
    assert ip.describe(None) == {name: 0 for name in ip.INSTRUMENTS}


def test_empty_columns():
    assert ip.instrument_of_columns([]) == set()
    assert ip.instrument_of_columns(None) == set()
    assert ip.instrument_of_columns(["", None]) == set()


def test_indicator_without_dependencies():
    (row,) = ip.annotate_indicators([{"name": "x"}])
    assert row["instrument"] == ip.INSTRUMENT_UNKNOWN


def test_required_l0_as_bare_string_is_not_split_into_characters():
    """`"au12_smile"` 若被当可迭代对象,会逐字符拆成 10 个单字母列名 —— 静默错。"""
    (row,) = ip.annotate_indicators([{"name": "smile_ratio",
                                     "required_l0": "au12_smile"}])
    assert row["instrument"] == ip.INSTRUMENT_GEOMETRIC


# ── ④ 未知列 ──────────────────────────────────────────────────────────────────

def test_unknown_column_maps_to_none_not_guessed():
    assert ip.instrument_of_column("dominant_emotion") is None
    assert ip.instrument_of_column("confidence") is None
    assert ip.instrument_of_column("完全是编出来的列名") is None


def test_unknown_columns_do_not_claim_an_instrument():
    assert ip.instrument_of_columns(["dominant_emotion", "confidence"]) == set()


def test_indicator_with_only_unknown_columns_is_marked_unknown():
    (row,) = ip.annotate_indicators([{"name": "emotion_label",
                                     "required_l0": ["dominant_emotion"]}])
    assert row["instrument"] == ip.INSTRUMENT_UNKNOWN
    assert ip.instruments_of(row) == [ip.INSTRUMENT_UNKNOWN]


def test_unmapped_columns_reports_the_gap_but_not_the_clock():
    """时钟/标识列不是仪器,也不算披露缺口 —— 否则每条指标都背一条假缺口。"""
    assert ip.unmapped_columns(["dominant_emotion", "timestamp", "session_id"]) == {
        "dominant_emotion"}
    assert ip.unmapped_columns(["au12_smile", "timestamp"]) == set()


# ── 规矩:不改既有字段 / 不原地改输入 ─────────────────────────────────────────

def test_existing_fields_are_untouched_and_input_is_not_mutated():
    ind = {
        "name": "smile_ratio", "family": "F1_smile",
        "formula": "au12 > 0.263 帧占比", "required_l0": ["au12_smile"],
        "window": "session", "value": 0.09, "unit": "占比",
        "n_valid": 900, "n_total": 989, "confidence": "高", "gap": None,
    }
    before = copy.deepcopy(ind)
    rows = ip.annotate_indicators([ind])
    assert ind == before, "输入字典被改了 —— 报告既有消费方会跟着变"
    assert "instrument" not in ind
    assert rows[0]["instrument"] == ip.INSTRUMENT_GEOMETRIC
    # 既有字段逐键不变
    for key, value in before.items():
        assert rows[0][key] == value


def test_annotate_returns_fresh_dicts():
    """返回的是**字典副本**(浅拷贝)。

    `required_l0` 这类嵌套值仍是同一个对象 —— 所以报告层只读它,不要就地 append。
    这里只钉"原字典没被塞进 instrument",不钉拷贝深度。
    """
    ind = {"name": "a", "required_l0": ["bs_jawOpen"]}
    rows = ip.annotate_indicators([ind])
    assert rows[0] is not ind
    assert rows[0]["required_l0"] == ind["required_l0"]
    assert "instrument" not in ind


# ── 描述计数(报告页脚) ───────────────────────────────────────────────────────

def test_describe_counts_fixed_keys_and_multi_twice():
    inds = [
        {"name": "a", "required_l0": ["au12_smile"]},
        {"name": "b", "required_l0": ["bs_jawOpen"]},
        {"name": "c", "required_l0": ["n_chars", "speech_duration_sec"]},
    ]
    counts = ip.describe(inds)
    assert list(counts) == list(ip.INSTRUMENTS)     # 键序固定
    assert counts[ip.INSTRUMENT_GEOMETRIC] == 1
    assert counts[ip.INSTRUMENT_BLENDSHAPE] == 1
    assert counts[ip.INSTRUMENT_AUDIO] == 1
    assert counts[ip.INSTRUMENT_TEXT] == 1
    assert counts[ip.INSTRUMENT_UNKNOWN] == 0
    # 多仪器那条在两处各计 1 ⟹ 相加 4 > 3 条指标。页脚不许拿这个求和当总数。
    assert sum(counts.values()) == 4


def test_describe_works_without_prior_annotation():
    """页脚允许在任何阶段调用 —— 未标注的指标当场算,不抛错。"""
    counts = ip.describe([{"name": "a", "required_l0": ["gaze_direction_x"]}])
    assert counts[ip.INSTRUMENT_GEOMETRIC] == 1


def test_annotated_and_raw_agree():
    raw = {"name": "a", "required_l0": ["bs_jawOpen"]}
    (ann,) = ip.annotate_indicators([raw])
    assert ip.describe([raw]) == ip.describe([ann])


def test_footer_line_mentions_blendshape_zero():
    """`blendshape 0` 本身就是要披露的事实 —— 页脚必须在场,哪怕计数为 0。"""
    line = ip.format_footer([{"name": "a", "required_l0": ["au12_smile"]}])
    assert "blendshape 0" not in line       # 人话版用中文标签
    assert "模型系数 0" in line
    assert "共 1 条指标" in line


def test_indicator_object_with_to_dict_is_accepted():
    """报告链路上 `L1Result.indicators` 是对象、`QualityReport.indicators` 是 dict。"""
    class Fake:
        required_l0 = ("bs_mouthSmileLeft",)

        def to_dict(self):
            return {"name": "smile_bs", "required_l0": list(self.required_l0)}

    (row,) = ip.annotate_indicators([Fake()])
    assert row["instrument"] == ip.INSTRUMENT_BLENDSHAPE


def test_rejects_objects_it_cannot_read():
    with pytest.raises(TypeError):
        ip.annotate_indicators([object()])
