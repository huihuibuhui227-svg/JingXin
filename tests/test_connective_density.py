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


def test_empty_text_returns_none_even_with_zero_floor():
    """空文本 → None 必须无条件成立,不能只是"下限恰好大于 0"的副产品。

    这是公开纯函数的契约:守卫写成 `chars < floor` 的话,min_chars=0 时放行到
    `hits / chars`,空文本直接 ZeroDivisionError —— 而 M3 之后的实验很可能传别的下限。
    """
    assert connective_density("", min_chars=0) is None


def test_no_marker_is_a_substring_of_another():
    """守卫的是**标记表这份数据**,不是生产代码:计数按"是否出现"做子串匹配,
    表里一旦出现包含关系(例如加一个 `为`),`因为` 会被两个标记各计一次,
    密度静默膨胀而没有任何东西会红。所以这条测试故意在被改坏的表上红。
    """
    ms = load_markers()["markers"]
    pairs = [(a, b) for a in ms for b in ms if a != b and a in b]
    assert pairs == [], f"标记表出现子串包含,会计数膨胀: {pairs}"
    # 字面重复的标记同样双计,而上面的 a != b 过滤按构造放过了"完全相同的串"这一对,
    # 所以重复必须单独断言 —— 否则同一轴上的静默膨胀照样没人拦(审查 Important/Minor 3)。
    assert len(ms) == len(set(ms)), f"标记表有字面重复项: {ms}"


def test_denominator_is_char_count_not_hit_count():
    """每百字的**分母**必须是被除数(字数),不是常数 100。

    这条用例的长度刻意**不是** 100 字:100 字的 fixture 会让 `hits / chars * 100`
    退化成 `hits`,于是"把 ÷ chars 整个删掉"也能全绿(审查实测确认)。50 字、命中 1
    → 2.0;删掉分母归一化会得 1.0,红在断言不等而不是异常。真实语音回答永远不会
    正好 100 字,而跨句长可比正是这个指标存在的理由(Task 4 要写进日志的就是它)。
    """
    assert connective_density("然后" + "字" * 48, min_chars=10) == 2.0


def test_chars_equal_to_floor_still_yields_a_value():
    """下限是 `<` 不是 `<=`:spec §6.5 写的是 `n_chars < min_chars_for_density`,
    所以**恰好等于下限要出值**。改成 `chars <= floor` 会得 None,红在断言不等。"""
    assert connective_density("然后" + "字" * 8, min_chars=10) == 10.0   # 正好 10 字


def test_repeated_marker_is_counted_once():
    """口径是"出现即计一次"(presence),不是出现次数(occurrence)——
    docstring 写了但此前只有手工验证。改成 `text.count(m)` 会得 6.0,红在断言不等。

    长度算清:`然后` × 3 = 6 字,加 `字` × 44,共 50 字;命中 1 → 2.0。
    """
    assert connective_density("然后" * 3 + "字" * 44, min_chars=10) == 2.0
