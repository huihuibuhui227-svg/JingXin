# tests/test_m4_text.py
"""**T 族(文本)3 条**的守卫 —— 2026-10-18 起从"占位"变为"可出"。

## 背景

规格 §5.5 原本只有一句「三个名字先占位,标注到位前不实现」。卡的是**定义**,
不是数据:`data/transcripts/` 里 13/13 场 A 档都有批量 ASR 转写。

## 本文件钉的五件事

1. **不许用原始 TTR** —— 它随文本变长单调下降(实测 320 字 0.4125 / 3899 字 0.1411),
   那是长度伪影,不是用词丰富度。改用 MATTR(固定窗口)。
2. `discourse_marker_rate` 数的是**出现次数**,不是"几个不同标记出现过" ——
   后者是既有的 `connective_density`,两条不是一个量。
3. `self_repair_rate` 只数**紧邻重复的二字组**;中文的合法单字叠词(`看看`/`慢慢`)
   不许算进去。
4. 没有转写 ⟹ `not_stored`;**不是 0**。
5. 三条的公式串都必须写明"全场混音(含面试官)" —— 转写没有说话人分离,
   不写清楚读者会把它当考生的话语。
"""

from __future__ import annotations

import pytest

from report_frontend import l1_text
from report_frontend.l1_indicators import GAP_NOT_STORED, GAP_NO_SAMPLES


def _by(text, thr=None):
    return {i.name: i for i in l1_text.compute_text_indicators(text, thr)}


# ── 1. MATTR 而不是原始 TTR ──────────────────────────────────────────────

def test_mattr_is_length_independent_where_raw_ttr_is_not():
    """★ 同一套二元组重复两次,MATTR 不变;原始 TTR 会掉一半。

    `["a","b","c","d"]` 重复:MATTR(窗口 4)恒为 1.0,而 TTR 从 1.0 掉到 0.5。

    红法:把 `mattr()` 换成 `len(set(tokens))/len(tokens)` ⟹ 本条红。
    """
    once = list("abcd")
    twice = once * 2
    assert l1_text.mattr(once, 4) == pytest.approx(1.0)
    assert l1_text.mattr(twice, 4) == pytest.approx(1.0)
    assert len(set(twice)) / len(twice) == pytest.approx(0.5)


def test_mattr_refuses_a_window_longer_than_the_text():
    """文本短于窗口 ⟹ `None`(算不出来),不许硬算。

    红法:把 `len(tokens) < window` 那个判断删掉 ⟹ 除零,本条红。
    """
    assert l1_text.mattr(list("abc"), 10) is None


def test_lexical_diversity_is_a_ratio_in_the_unit_interval():
    text = "我们今天讨论一下这个项目的可行性和它的落地路径" * 30
    ind = _by(text)["lexical_diversity"]
    assert ind.has_value
    assert 0.0 < ind.value <= 1.0


# ── 2. 话语标记:次数,不是"几个不同标记" ───────────────────────────────

def test_discourse_marker_rate_counts_occurrences_not_distinct_markers():
    """★ `然后然后然后` ⟹ 出现 3 次(不是"1 个不同标记")。

    这正是与 `connective_density` 的区别:那条数的是"哪些标记出现过",
    本条数的是"总共说了多少次"。6 个字 3 次 ⟹ 50.0 次/百字。

    红法:把 `body.count(m)` 换成 `sum(1 for m in ... if m in body)` ⟹ 得 16.67,本条红。
    """
    ind = _by("然后然后然后")["discourse_marker_rate"]
    assert ind.value == pytest.approx(3 / 6 * 100.0)


# ── 3. 自我修正:紧邻二字组重复 ─────────────────────────────────────────

def test_self_repair_counts_immediate_bigram_repeats():
    """`就是就是` ⟹ 1 次紧邻二字组重复;4 个字 ⟹ 25.0 次/百字。

    红法:把 `text[i:i+2] == text[i+2:i+4]` 改成单字比较 ⟹ 得 0,本条红。
    """
    ind = _by("就是就是")["self_repair_rate"]
    assert ind.value == pytest.approx(1 / 4 * 100.0)


def test_self_repair_ignores_legitimate_single_character_reduplication():
    """★ `看看慢慢试试` 是**合法的单字叠词**,不算自我修正。

    中文里单字重叠极其常见(看看/慢慢/试试/说说),数进来会把正常口语算成口吃。
    所以只认**二字组**重复。

    红法:改成单字重复检测 ⟹ 本条得 >0,红。
    """
    ind = _by("看看慢慢试试")["self_repair_rate"]
    assert ind.value == pytest.approx(0.0)


def test_zero_repairs_is_a_measured_zero():
    """一次修正都没有 ⟹ **有值的 0.0**,不是缺失。

    红法:在 `hits == 0` 时返回 `unavailable(..., GAP_NO_SAMPLES)` ⟹ 本条红。
    """
    ind = _by("我们今天讨论一下这个项目的可行性和它的落地路径")["self_repair_rate"]
    assert ind.has_value and ind.value == pytest.approx(0.0)
    assert ind.confidence != "无"


def test_repair_marker_table_has_no_marker_containing_another():
    """★ 表守卫:子串匹配下,标记之间不许有包含关系。

    否则 `我是说` 与 `我是说的` 会把同一处命中各计一次(与
    `connective_markers.json` 那条不变量同一条纪律)。

    红法:往表里加一个包含既有标记的新标记 ⟹ 本条红。
    """
    markers = l1_text.load_repair_table()["markers"]
    for a in markers:
        for b in markers:
            if a is b or a == b:
                continue
            assert a not in b, f"{a!r} 是 {b!r} 的子串 —— 会重复计数"


def test_repair_table_is_versioned_and_provisional():
    """标记表必须带 version 与 `_provisional`(与连接词表同格式)。"""
    table = l1_text.load_repair_table()
    assert table["version"] and table["_provisional"] is True
    assert table["basis"]


# ── 4. 缺转写 / 太短 ────────────────────────────────────────────────────

def test_no_transcript_means_not_stored_not_zero():
    """★ 没有转写 ⟹ 三条全 `not_stored`。

    红法:把 `if text is None` 那支改成用 `""` 继续算 ⟹ 得 0 而不是缺失,本条红。
    """
    got = _by(None)
    for name in l1_text.ORDER:
        assert got[name].gap == GAP_NOT_STORED, name
        assert not got[name].has_value


def test_short_text_has_no_lexical_diversity_but_still_rates_the_other_two():
    """文本短于 MATTR 窗口 ⟹ 只有 `lexical_diversity` 报 `no_samples`。

    另外两条不需要窗口,照常出值(置信度会因字数不足而偏低)。

    红法:把 `mattr(...) is None` 那支改成写 0.0 ⟹ 本条红。
    """
    got = _by("然后然后然后")
    assert got["lexical_diversity"].gap == GAP_NO_SAMPLES
    assert got["discourse_marker_rate"].has_value
    assert got["self_repair_rate"].has_value


# ── 4b. 中英混说的口径 ──────────────────────────────────────────────────

def test_tokens_split_chinese_into_bigrams_and_english_into_words():
    """★ 词元流按脚本切:汉字切二元组,英文切**整词**。

    实测 13 场 A 档里拉丁字母占正文的**中位 8.3%、最高 49.2%**(一场基本全英文)。
    若英文也按字符切,会得到 `he`/`el`/`ll`/`lo` 这类**字母**二元组 ——
    量的是拼写冗余度,与中文的字二元组不是同一个构念,两边的值不可比。

    红法:把 `tokens()` 改成对整串按字符切二元组 ⟹ 英文段变成字母二元组,本条红。
    """
    assert l1_text.tokens("我们hello world讨论") == ["我们", "hello", "world", "讨论"]
    # 汉字单字独立出现时不能被静默丢掉(两侧的拉丁单字母也是词元)
    assert l1_text.tokens("a我b") == ["a", "我", "b"]


def test_marker_rate_divides_by_chinese_chars_only():
    """★ 两条率量的分母**只数汉字**,不数英文。

    标记表是中文的,英文词不构成可命中的材料。拿"汉字+英文字母"当分母,
    英文多的场次会被无端稀释(49.2% 那场会腰斩一半)。

    `然后` + 20 个 `abc`:汉字 2 个、标记 1 次 ⟹ 50.0 次/百汉字。
    若分母混入 60 个字母则得 1.61 —— 差 31 倍。

    红法:把 `occ / chars` 的 `chars` 换成"汉字+拉丁字母" ⟹ 本条红。
    """
    ind = _by("然后" + "abc" * 20)["discourse_marker_rate"]
    assert ind.value == pytest.approx(1 / 2 * 100.0)


def test_pure_english_transcript_has_no_marker_rate_at_all():
    """★ 整场没有汉字 ⟹ 两条率量 `no_samples`,**不是 0**。

    报 0 等于替一场英文面试捏一个中文结论("这场一次连接词都没用")。
    词汇多样性照常出值 —— 它用的是英文词元。

    红法:去掉 `chars == 0` 那个分支 ⟹ 除零崩溃(或报 0),本条红。
    """
    # 词元数要过 MATTR 窗口(100),否则词汇多样性也报 no_samples ——
    # 那是另一件事(文本太短),不是本条要钉的口径问题。
    got = _by("i think we should discuss the project feasibility and its path "
              "to land in the next quarter because the team is ready now " * 12)
    assert got["lexical_diversity"].has_value
    for name in ("discourse_marker_rate", "self_repair_rate"):
        assert got[name].gap == GAP_NO_SAMPLES, name
        assert not got[name].has_value


# ── 5. 三条都必须声明"全场混音" ─────────────────────────────────────────

def test_all_three_formulas_admit_the_transcript_is_mixed_speakers():
    """★ 公式串必须写明"全场混音(含面试官)"。

    转写没有说话人分离 ⟹ 三条量的都是"这一场的话语",不是"考生的话语"。
    不写清楚,`self_repair_rate` 会被当成考生的口头禅频率,而里面混着面试官的。

    红法:把任一条公式里的"全场混音"去掉 ⟹ 本条红。
    """
    got = _by("我们今天讨论一下这个项目的可行性和它的落地路径")
    for name in l1_text.ORDER:
        assert "全场混音" in got[name].formula, name
        assert "含面试官" in got[name].formula, name


# ── 6. 两处计数/单位修正的守卫 ──────────────────────────────────────────

def test_bigram_repeats_are_counted_without_overlap():
    """★ `就是`×10 计 **9** 次(接缝数),不是 17。

    ## 这条测试的由来(一级 bug)

    原实现逐个位置比较 `text[i:i+2] == text[i+2:i+4]`,而周期 2 的长串上
    **每个 i 都成立**(连奇数位也成立,因为 `是就` 也等于 `是就`),于是 k 次连说
    被数成 2k−3。真实语料因此虚高 1.13 倍(214 vs 190,候选T 37 vs 30)。

    正确口径:数**接缝**,命中一处后跳过被重复的单元 ⟹ k 次连说计 k−1 次。

    红法:把 `i += 2` 改成 `i += 1`(即恢复逐位比较)⟹ 本条红。
    """
    for k, expect in ((2, 1), (3, 2), (4, 3), (10, 9)):
        got = l1_text.immediate_bigram_repeats("就是" * k)
        assert got == expect, f"k={k} 计得 {got},应为 {expect}"


def test_lexical_diversity_reports_and_thresholds_in_tokens():
    """★ `lexical_diversity` 的 `n_valid` 与门槛必须**同单位**(词元)。

    ## 这条测试钉的是"单位语义",不是数值行为 —— 说清楚免得后人误解

    审计指出 `n_valid=len(toks)`(词元)却用 `TEXT_MIN_CHARS`(字)当门槛。
    追下去发现:两个常量的**数值都是 300**(`TEXT_MIN_TOKENS = 3 × MATTR_WINDOW = 300`,
    `TEXT_MIN_CHARS = 300`),所以那条"修正"在今天的语料上**不改变任何输出** ——
    我实测把 `threshold=TEXT_MIN_TOKENS` 换回 `TEXT_MIN_CHARS`,本条测试**照样绿**。
    既然改不动数值,就不能声称修了一个数值 bug。

    真正需要钉住的是**单位**:`n_valid` 是词元数、门槛也是词元数,
    中文里词元 ≈ 字数所以看不出差别,英文里差别很大(300 词元 ≈ 300 个英文词)。
    将来谁把 `n_valid` 改成字数、或把门槛改成别的单位,本条会红。

    红法:把 `n_valid=len(toks)` 改成 `n_valid=chars` ⟹ 本条红。
    """
    assert l1_text.TEXT_MIN_TOKENS == 3 * l1_text.MATTR_WINDOW
    text = "我们今天讨论这个项目" * 25 + " " + "alpha beta gamma delta " * 100
    ind = _by(text)["lexical_diversity"]
    assert ind.has_value
    assert ind.n_valid == len(l1_text.tokens(text))
    assert ind.n_valid >= l1_text.TEXT_MIN_TOKENS
    # 中文 250 字 + 英文 400 词:词元数远大于汉字数 —— 若 n_valid 误用字数,这里会露出来。
    assert ind.n_valid > len(l1_text._cjk(text))
