import json

from voice_interaction.asr import funasr_engine
from voice_interaction.asr.connective_density import connective_density, load_markers


def test_counts_markers_per_hundred_chars():
    text = "然后" + "字" * 98                       # 100 字,命中 1
    assert connective_density(text, min_chars=10) == 1.0


def test_punctuation_and_space_do_not_count_as_chars():
    a = connective_density("然后" + "字" * 98 + "。。。,,,   ", min_chars=10)
    assert a == 1.0


def test_below_min_chars_returns_none_not_zero():
    assert connective_density("然后好", min_chars=10) is None


def test_empty_text_returns_none():
    assert connective_density("", min_chars=10) is None


def test_no_marker_returns_zero_not_none():
    assert connective_density("字" * 50, min_chars=10) == 0.0


def test_marker_table_has_version_and_list():
    m = load_markers()
    assert m["version"]
    assert len(m["markers"]) >= 10
    assert "然后" in m["markers"] and "但是" in m["markers"]


def test_marker_table_carries_provenance():
    """标记表是带出处的数据文件:没有 _provisional/basis 的表等于裸常量(全局约束)。"""
    m = load_markers()
    assert m["_provisional"] is True
    assert m["basis"].strip()


def test_floor_is_read_from_config_not_hardcoded(tmp_path, monkeypatch):
    """密度下限只能来自 asr_config.json:代码里写死 10 的话,换配置也不动。"""
    p = tmp_path / "asr_config.json"
    p.write_text(json.dumps({"min_chars_for_density": {"value": 200}}), encoding="utf-8")
    monkeypatch.setattr(funasr_engine, "_CONFIG_PATH", p)
    assert connective_density("然后" + "字" * 98) is None      # 100 字 < 200
