# tests/test_prose_numbers_match_the_code.py
"""散文里的**架构数**必须与代码当场数出来的一致。

## 为什么需要这个文件

`test_l0_column_table.py` 已经钉住了 `l0_columns.json` 里那批数,但它**只扫表**。
`docs/开发者文档.md`(「接手先读这个」)与 `docs/下一步.md` 的正文里另有一批数,
没有任何守卫 —— 2026-10-25 凌晨的文档审计实测**一次性抓到 10 处过期**,
其中最贵的三处:

| 文档写 | 当场算出 | 后果 |
|---|---|---|
| `research_mapper` **20 个槽** | **14** | 读的人按 20 个槽理解证据门 |
| evidence_gate 有**折算层**(`normalize_value`) | 该层 2026-10-04 **整层删除** | 按不存在的层设计改动 |
| 套件 **74 个模块** | **111** | 低估覆盖面 |

这三条的共同形态是本仓记过多次的「散文里的数会静默过期」(2026-10-25 已是第 8 次现形):
**旧小节里的数被当成现状引用,而新小节已经改对了。**

## 这个文件守什么

判据刻意做得**窄**:只钉"改一次代码就必须同步改文档"的那几个数,
且每个数都从**代码当场数出来**,不在测试里手抄。这样:
- 加一个槽 / 一个指标 / 一个封停键 ⟹ 这里红,提醒同步散文;
- 只改散文里的数(改错)⟹ 这里也红。

**不**钉容易正常变动的量(测试条数、场次数)—— 那些每加一条测试就红一次,
会退化成噪声,反而让人习惯性忽略红。
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "开发者文档.md"


def _doc() -> str:
    assert DOC.is_file(), f"找不到 {DOC}"
    return DOC.read_text(encoding="utf-8")


# ── 1. 槽数:research_mapper 与 l1_mapper 必须同为 14,且文档里写的也是 14 ──

def test_slot_counts_match_the_code():
    """★ 红法:给任一维度加一个槽 ⟹ 这里红,提醒同步文档。

    ⚠️ 为什么两个引擎都数:它们**同名不同命**(实测五维里四维的数不一样),
    只数一个会让另一个悄悄漂。文档里那句「14 个槽」说的是**两个都是 14**。
    """
    from report_frontend import l1_mapper
    from report_frontend.research_mapper import ResearchCapabilityMapper

    legacy = sum(len(r["indicators"])
                 for r in ResearchCapabilityMapper().mapping_rules.values())
    l1 = len(l1_mapper.SLOT_MAP)
    assert legacy == l1, (
        f"两个引擎的槽数不一致:research_mapper={legacy}, l1_mapper={l1} —— "
        f"文档里「14 个槽」这句在说它们同一个数")
    assert f"research_mapper({legacy} 个槽" in _doc(), (
        f"文档里的槽数与代码对不上(代码是 {legacy})")
    assert f"{l1} 个槽" in _doc(), f"文档里没有「{l1} 个槽」这个数"


# ── 2. 折算层已删:文档不许再把它写成现役 ─────────────────────────────────

def test_normalize_value_is_gone_and_the_doc_says_so():
    """★ 红法:把折算层加回 `evidence_gate` ⟹ 这里红(文档得同步说清楚)。

    2026-10-04 删的(`v12-NoComposite` 那轮)。代码里只剩注释提到它,
    文档却还把它写成 `evidence_gate` 的一项能力 —— 读的人会去找一个不存在的函数。
    """
    import report_frontend.evidence_gate as eg

    assert not hasattr(eg, "normalize_value"), (
        "evidence_gate 又有 normalize_value 了 —— 那文档里「折算层已删」那段要跟着改")
    assert not hasattr(eg, "scale_factor_for"), "scale_factor_for 又回来了"
    # 阈值文件里也不许再有 scale_factors
    import json
    th = json.loads((REPO / "report_frontend" / "evidence_thresholds.json")
                    .read_text(encoding="utf-8"))
    assert "scale_factors" not in th, f"阈值文件又有 scale_factors:{sorted(th)}"

    doc = _doc()
    # 文档必须**说明它已删**,而不是把它列成 evidence_gate 的一项
    assert "整层删掉" in doc or "整层删除" in doc, (
        "文档没有说明折算层已删 —— 读者会去找 normalize_value")
    # 并且不许再在表格里把它当成 evidence_gate 的能力
    assert "折算(`normalize_value`)" not in doc, (
        "文档表格里仍把 `normalize_value` 写成 evidence_gate 的能力")


# ── 3. L1 指标总数:文档里的「38 条」与代码一致 ────────────────────────────

def test_l1_indicator_total_matches_the_code():
    """★ 红法:加/删一个 L1 指标 ⟹ 这里红。

    ⚠️ 第一版扫的是 `开发者文档.md`,那里**根本没写指标总数**(实测 grep 无命中),
    于是这条红的是测试而不是代码。指标总数写在 `下一步.md`。
    """
    from report_frontend import l1_face, l1_gesture, l1_text, l1_voice

    total = (len(l1_face.indicator_names()) + len(l1_gesture.ORDER)
             + len(l1_voice.ORDER) + len(l1_text.ORDER))
    plan = (REPO / "docs" / "下一步.md").read_text(encoding="utf-8")
    assert f"{total} 条" in plan, (
        f"`下一步.md` 里找不到「{total} 条」这个指标总数(代码当场数出 {total})")


# ── 4. 封停名单:文档里点名的数与 `QUARANTINE` 一致 ────────────────────────

def test_quarantine_count_matches_the_code():
    """★ 红法:加/删一个封停键 ⟹ 这里红。

    2026-10-25 实测:文档写「其余 20 条」,而 `QUARANTINE` 是 **30** 个键。
    任何口径都得不到 20 —— 那个数写下的当天就是错的。
    """
    from report_frontend.evidence_gate import QUARANTINE

    n = len(QUARANTINE)
    plan = (REPO / "docs" / "下一步.md").read_text(encoding="utf-8")
    m = re.search(r"其余\s*(\d+)\s*条封停照同一模式推", plan)
    assert m, "「其余 N 条封停照同一模式推」这句被删/改写了 —— 本测试要跟着改"
    stated = int(m.group(1))
    assert stated == n - 1, (
        f"文档写「其余 {stated} 条封停」,而 QUARANTINE 有 {n} 个键 "
        f"(已按其中 1 条的模式推,其余应为 {n - 1})")
