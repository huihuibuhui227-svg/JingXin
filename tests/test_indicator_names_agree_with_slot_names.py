# tests/test_indicator_names_agree_with_slot_names.py
"""同一个量在两个出口必须同名 —— 旧槽名与 L1 指标名逐条对账。

## 为什么

旧引擎的 14 个槽(`l1_mapper.SLOT_LABEL`)与新引擎的 38 条 L1 指标
(`l1_quality.INDICATOR_LABEL`)**各自登记了一份中文名**,而其中 8 个量是
同一件事的两个叫法(`SLOT_MAP` 里逐条写着"换:同构念")。

两份名字一旦分叉,读者就会以为它们是两个不同的量。这个形态本仓记过:
2026-10-25 深夜实测,交付物正文印着 `speech_rate:` 这类审计键,而同一份报告
的维度表里同一个指标印的是「语调变化」—— 读者以为是两个东西。

`l1_quality.INDICATOR_LABEL` 的 docstring 把这条规矩写在里面,并点名本文件。
**不点名就没人守** —— 那句话自己不会红。

## 判据从哪来

不从任何一张手抄的对照表来:`SLOT_MAP` 是"旧槽 → L1 指标"的**唯一**登记处,
这里遍历它,凡是映射到某条 L1 的,两边中文名必须逐字相同。
手抄一份对照表等于多出第三个登记处,那正是要防的东西。

## 这个文件守什么

1. `SLOT_MAP` 里每一个"换"过去的槽,两边中文名逐字相同;
2. 反过来:L1 那一侧**不许漏登记**(漏了 `indicator_label` 会原样吐英文键)。
"""
from __future__ import annotations

import pytest

from report_frontend import l1_quality
from report_frontend.l1_mapper import SLOT_LABEL, SLOT_MAP, slot_label


def _mapped() -> list[tuple[str, str]]:
    """`[(旧槽键, L1 指标名), ...]` —— 凡 `SLOT_MAP` 判为"换"的都算。"""
    return [(slot, l1) for slot, (l1, _reason) in SLOT_MAP.items() if l1]


def test_there_are_mapped_slots_to_check():
    """本文件的前提:至少有一对同构念的槽。为 0 时下面全是空断言。"""
    assert _mapped(), "SLOT_MAP 里没有任何「换」过去的槽 —— 本测试退化为空断言"


#: **唯一允许两个出口不同名**的槽 —— 理由是**方向相反**,不是措辞偏好。
#:
#: ## 为什么必须有这份例外,而且只有一条(2026-11-22)
#:
#: `symmetry_score` 槽量的是"面部对称性"(列本身:越大越对称),
#: 而 `facial_asymmetry_p90` 数的是 `p90(1 − symmetry_score)`(越大越**不**对称)。
#: 两边同名会让读者把 0.062 读成"对称度 0.062"—— 含义**恰好说反**。
#: 这与本仓"换构念必须换名"是同一条规矩的另一面:名字必须与它数的方向一致。
#:
#: ⚠️ 例外只有这一条,而且**必须带方向证据**:下面除了比对两个名字不同,
#: 还要确认 `SLOT_MAP` 的理由里写明了方向(「越大越不对称」),以及
#: `NOT_COMPARABLE` 里登记了这处不可比 —— 否则"不同名"就成了无据的随意改名。
_DIRECTION_FLIP = {
    "symmetry_score": ("facial_asymmetry_p90", "越大越不对称"),
}


#: 例外以外的槽 —— 它们的判据是"两边必须同名",一个 assert 恒执行。
_SAME_NAMED = [(slot, l1) for slot, l1 in _mapped() if slot not in _DIRECTION_FLIP]


def test_the_direction_flip_exception_is_not_empty():
    """例外表非空 —— 否则下面那条分支就是死代码,而它守的是最要紧的一处。

    (`test_assert_coverage` 会抓"声明了却没跑到的 assert";把例外拆成独立用例
    就是为了让它**每个参数都真的执行**,而不是埋在 `if slot in _DIRECTION_FLIP:`
    里只跑一次。2026-11-21 被 meta 检查抓过一次,这是修法。)
    """
    assert _DIRECTION_FLIP, "例外表空了 —— 那段「不同名」的判据再也不会执行"


@pytest.mark.parametrize("slot", sorted(_DIRECTION_FLIP))
def test_a_direction_flip_slot_is_registered_and_never_same_named(slot):
    """★ 例外槽(目前只有面部对称性):两边**必须**不同名,且理由要写明方向。

    红法:把 `_DIRECTION_FLIP` 里的 L1 名换成现在映射的那个同名版本 ⟹ 这里红;
    或把 `SLOT_MAP` 的理由里的方向词删掉 ⟹ 这里红。
    """
    from report_frontend.l1_mapper import NOT_COMPARABLE

    want_l1, direction = _DIRECTION_FLIP[slot]
    l1_name = SLOT_MAP[slot][0]
    old = slot_label(slot)
    new = l1_quality.indicator_label(l1_name)
    assert l1_name == want_l1, (
        f"例外登记的是 {want_l1!r},而 SLOT_MAP 现在映射到 {l1_name!r} —— 登记过期了")
    assert new != old, (
        f"{slot!r} 两边同名(「{new}」),但它数的方向与槽相反 —— 同名会说反意思")
    assert direction in SLOT_MAP[slot][1], (
        f"{slot!r} 的映射理由没写清方向(缺「{direction}」):{SLOT_MAP[slot][1]!r}")
    assert slot in NOT_COMPARABLE, (
        f"{slot!r} 换了方向却没登记进 NOT_COMPARABLE —— 读者会把两边的数相减")


@pytest.mark.parametrize("slot,l1_name", _SAME_NAMED, ids=[s for s, _ in _SAME_NAMED])
def test_the_two_outlets_call_the_same_quantity_by_the_same_name(slot, l1_name):
    """★ 红法:把 `INDICATOR_LABEL` 里任一条改名 ⟹ 这里红。

    同一个量,旧报告叫「语调变化」、新报告叫「音高波动」—— 读者会以为是两个指标,
    复核的人会去查一个不存在的对应关系。

    ⚠️ 这条**只覆盖"应该同名"的槽**;方向相反那一处(面部对称性)由
    `test_a_direction_flip_slot_is_registered_and_never_same_named` 单独守。
    拆开的原因:`test_assert_coverage` 只认"真的被执行过的断言",把两种相反的
    判据塞进一个函数、靠 `if` 分岔,会让其中一半断言永远跑不到。
    """
    old = slot_label(slot)
    new = l1_quality.indicator_label(l1_name)
    assert new == old, (
        f"{slot!r} 与 {l1_name!r} 是同一个量(SLOT_MAP 判为同构念),"
        f"但旧出口叫「{old}」、新出口叫「{new}」—— 同一份报告里会出现两个名字")


@pytest.mark.parametrize("slot,l1_name", _mapped(),
                         ids=[s for s, _ in _mapped()])
def test_the_l1_name_is_actually_registered(slot, l1_name):
    """★ 红法:从 `INDICATOR_LABEL` 删掉某一条 ⟹ 这里红。

    `indicator_label` 未登记时**原样返回键名**。那条规矩是对的(宁可露英文,
    也不许吞掉),但它意味着"漏登记"不会报错 —— 只有这里能抓住。
    """
    assert l1_name in l1_quality.INDICATOR_LABEL, (
        f"{l1_name!r} 没登记中文名 —— 报告里会印出英文键")


def test_no_slot_label_is_left_blank():
    """空名字比英文键更糟:读者会以为这一条没有名字。"""
    blank = sorted(s for s, v in SLOT_LABEL.items() if not v.strip())
    assert not blank, f"槽名表里有空名字:{blank}"
