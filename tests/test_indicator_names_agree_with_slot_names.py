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


@pytest.mark.parametrize("slot,l1_name", _mapped(),
                         ids=[s for s, _ in _mapped()])
def test_the_two_outlets_call_the_same_quantity_by_the_same_name(slot, l1_name):
    """★ 红法:把 `INDICATOR_LABEL` 里任一条改名 ⟹ 这里红。

    同一个量,旧报告叫「语调变化」、新报告叫「音高波动」—— 读者会以为是两个指标,
    复核的人会去查一个不存在的对应关系。
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
