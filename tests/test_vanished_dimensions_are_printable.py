"""`vanished_dimensions` 的载荷要给前端**能直接印**的名字与理由。

## 为什么

`l1_mapper.map_l1_to_dimensions` 产出的 `vanished_dimensions` 长这样:

    {"key": "confidence_level", "display_name": "行为表现活跃度",
     "dropped": [{"slot": "left_hand_spread",
                  "reason": "hand_visible_ratio_l 量的是**可见性**,不是手掌张开度"}, ...]}

`slot` 是内部槽键,`reason` 是**审计细节**(它必须点名真实 L1 指标,
`tests/test_m4_l1_mapper.py` 要求如此)。后端 HTML 报告两样都不印 —— 它走
`l1_mapper.slot_label` / `drop_reason` 那层人话(见 `_render_vanished_dimensions`)。

但**前端 `ReportViewer` 直接印了原始字段**:

    {v.dropped.map((d) => `${d.slot}：${d.reason}`).join('；')}

⟹ 报告页上读者看到的是 `left_hand_spread：hand_visible_ratio_l 量的是**可见性**,
不是手掌张开度` —— 内部键 + 内部指标名 + markdown 星号,一次三样。

前端**没有**槽名对照表(它不该有:两张表迟早分叉),所以人话只能由载荷给。
⟹ 每个 `dropped` 条目补 `human_name`(中文槽名)与 `plain_reason`(一句话结论),
审计字段 `slot` / `reason` **原样保留**(核对口径的人仍要查得到)。

## 这个文件守什么

1. 每个消失维度的每个删槽条目都带 `human_name` 与 `plain_reason`。
2. 那两个字段里**没有**内部槽键、内部 L1 指标名、markdown 记号。
3. `slot` / `reason` 仍在(审计细节分层保留,不是删掉)。
4. 前端真的改用它 —— 由 `JingXin-frontend` 仓的
   `src/components/report/__tests__/vanishedDimensions.test.mjs` 守。
"""
from __future__ import annotations

import re

import pytest

from report_frontend import l1_face, l1_gesture, l1_text, l1_voice
from report_frontend.l1_mapper import DROP_REASON, SLOT_LABEL, SLOT_MAP
from report_frontend.l1_report import build_l1_report

_SID = "20260101_000000_0000"


def _all_l1_names() -> set[str]:
    return (set(l1_face.indicator_names()) | set(l1_gesture.ORDER)
            | set(l1_voice.ORDER) | set(l1_text.ORDER))


@pytest.fixture
def vanished(monkeypatch, tmp_path):
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path / "no_rec"))
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path / "no_series"))
    got = build_l1_report(_SID, logs={})["vanished_dimensions"]
    assert got, "载荷里没有消失的维度 —— 本测试退化为空断言"
    return got


def test_every_dropped_slot_carries_a_human_name_and_plain_reason(vanished):
    """★ 红法:去掉 `l1_report` 里那步补字段 ⟹ 这里红。

    地板:消失的维度当前是 1 个、里面 3 个删槽。写死数字是为了槽表被清空时
    下面的循环不会一次都不进而静默全绿。
    """
    entries = [d for v in vanished for d in v["dropped"]]
    assert len(entries) == 3, f"删槽条目数变了:{entries}"
    for d in entries:
        assert d.get("human_name"), f"{d['slot']} 没有中文名,前端只能印内部键"
        assert d.get("plain_reason"), f"{d['slot']} 没有给人看的理由"


def test_the_human_fields_name_no_internal_symbols(vanished):
    """★ 红法:把 `plain_reason` 换成 `reason` 的值 ⟹ 这里红。

    这一条是上一条的**内容**判据:有条目 ≠ 条目能给人读。
    ⚠️ 先断言字段**在**:第一版只扫 `d.get(field) or ""`,字段缺席时扫的是空串
    ⟹ 一写就绿(实测),又是空断言。
    """
    known = _all_l1_names()
    bad = []
    for v in vanished:
        for d in v["dropped"]:
            for field in ("human_name", "plain_reason"):
                assert field in d, f"{d['slot']} 没有 {field} —— 本测试会退化为空扫"
                text = d[field]
                hits = [n for n in known if n in text]
                if hits or "**" in text or "`" in text or d["slot"] in text:
                    bad.append(f"{d['slot']}.{field} = {text!r}(命中 {hits})")
    assert not bad, "给前端印的字段里还带着内部记号:\n  " + "\n  ".join(bad)


def test_the_audit_fields_are_still_there(vanished):
    """★ 红法:把 `slot` / `reason` 从载荷里删掉 ⟹ 这里红。

    分层的意思是**两处各有各的话**,不是把审计细节删掉。
    """
    for v in vanished:
        for d in v["dropped"]:
            assert d.get("slot") in SLOT_MAP
            assert len(d.get("reason") or "") > 20, (
                f"{d['slot']} 的审计理由被削短了:{d.get('reason')!r}")


def test_the_human_fields_come_from_the_one_shared_table(vanished):
    """★ 红法:在 `l1_report` 里另抄一张中文表 ⟹ 这里红。

    抄第二张表的后果可预见:新增一个删槽时改一处漏一处,界面上就冒出一个英文键。
    """
    for v in vanished:
        for d in v["dropped"]:
            assert d["human_name"] == SLOT_LABEL[d["slot"]], (
                f"{d['slot']} 的中文名与 `SLOT_LABEL` 不一致")
            assert d["plain_reason"] == DROP_REASON[d["slot"]], (
                f"{d['slot']} 的人话理由与 `DROP_REASON` 不一致")
