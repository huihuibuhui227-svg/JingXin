# tests/test_marker_table_immutable.py
"""D1(最终全分支审查,triage 标「验收前」):`load_markers()` 不得把缓存本身交出去。

**失效形态(修复前,已复现):** `connective_density.load_markers()` 带
`@lru_cache(maxsize=1)`,返回的是**那个被缓存的 dict 本体**。任何调用方对它做一次
就地修改(`table["markers"].append(...)` 或 `table["version"] = ...`),改动会留在
缓存里,**此后整个进程**算出的连接词密度都按被改过的标记表来 —— 而密度是要进报告的
数字。也就是说:报告里那个值取决于"谁先改过缓存",不取决于 `connective_markers.json`。
这正是 M1 要杀的诚实问题的一种(值不再由声明的依据决定)。

修法:缓存解析结果,但**交出副本**,让就地修改无法穿透到其他调用方。
"""

from voice_interaction.asr.connective_density import load_markers

_INTRUDER = "这个词不该出现在任何标记表里"


def test_appending_to_the_returned_marker_list_cannot_poison_later_callers():
    """改返回值的 `markers` 列表,下一次 `load_markers()` 必须看不到这个改动。

    红在(修复前):`lru_cache` 交出缓存本体 → `append` 改的就是缓存 →
    下一次调用仍带着 `_INTRUDER` → 断言失败。
    """
    table = load_markers()
    table["markers"].append(_INTRUDER)

    assert _INTRUDER not in load_markers()["markers"], (
        "标记表被调用方就地改掉了 —— 此后本进程算出的密度都按被污染的表来,"
        "而那个值要进报告"
    )


def test_overwriting_a_top_level_key_cannot_poison_later_callers():
    """顶层键同样不能被穿透(版本号是要随报告一起记账的元数据)。

    红法:把 `load_markers` 改回返回缓存本体(或只对 `markers` 做浅层保护而漏掉顶层)。
    """
    before = load_markers()["version"]

    load_markers()["version"] = "9999.0.0"

    assert load_markers()["version"] == before, (
        "标记表版本号被调用方改掉了 —— 报告记账的 marked version 会与实际用的表脱节"
    )
