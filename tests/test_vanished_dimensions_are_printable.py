"""`vanished_dimensions` 的载荷要给前端**能直接印**的名字与理由。

## ⚠️ 2026-11-22:生产数据里**已经没有整维消失的维度了**

补进三条新指标(`facial_asymmetry_p90` / `hand_openness_p50` / `shrug_elevation_p90`)之后,
`行为表现活跃度` 从"3 槽全删"变成"2 槽有指标 + 1 槽如实登记不测" ⟹
`build_l1_report(...)["vanished_dimensions"]` 现在是**空列表**。
(这不是渲染层不印了,是事实变了 —— 载荷里加一条
`test_the_real_payload_no_longer_vanishes_any_dimension` 把这个新事实钉住。)

而"消失时必须能印人话"这条规矩**仍然必须有人守** —— 下一个人删掉
`_printable_vanished` 那一步时不该有任何测试变绿。所以本文件改成:
**用一个构造出来的"整维没有对应指标"维度**跑真装配,再验那四个判据。

⚠️ **2026-11-22(同日第二次)**:补进 `gaze_dwell_p50` 之后,`SLOT_MAP` 里
**一个删槽都不剩**(`fluency_score` / `eye_contact` / `n_chars` 现在全部有对应指标),
原来"借三个真删槽"的构造方式随之失效。现在改用**三个合成槽键**
(确定不在 `SLOT_MAP` / `SLOT_LABEL` / `DROP_REASON` 里,由 fixture 临时登记),
验的仍然是真装配链(`map_l1_to_dimensions` → `_printable_vanished`),
参数化的仍是"整维消失要能印人话"这条判据本身。

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

from report_frontend import l1_face, l1_gesture, l1_mapper, l1_text, l1_voice
from report_frontend.l1_mapper import DROP_REASON, SLOT_LABEL, SLOT_MAP
from report_frontend.l1_report import build_l1_report

_SID = "20260101_000000_0000"

#: 构造用的维度键(只在 `vanished` fixture 里存在于 `DIMENSIONS`)。
_SYNTHETIC_KEY = "synthetic_dead_dimension"


def _all_l1_names() -> set[str]:
    return (set(l1_face.indicator_names()) | set(l1_gesture.ORDER)
            | set(l1_voice.ORDER) | set(l1_text.ORDER))


#: 构造用的三个删槽 —— 键、中文名、人话理由**都**由 fixture 临时登记进三张真表。
#: 为什么三个都要登记:载荷里补 `human_name` / `plain_reason` 那一步读的就是
#: `SLOT_LABEL` / `DROP_REASON`;只往 `SLOT_MAP` 塞键的话,补出来的是英文键
#: (退化成"没登记"那条路,验不到"人话来自共享表"这条判据)。
_SYN_SLOTS = {
    "syn_vanished_alpha": ("构造槽甲", "这套采集里没有与它对应的量,不拿相邻的量顶替"),
    "syn_vanished_beta": ("构造槽乙", "它要的那一路信号本套采集不含,再采也不会有"),
    "syn_vanished_gamma": ("构造槽丙", "这个构念没有独立测量,报出来只是把别的量数两遍"),
}

#: **导入时**(任何 monkeypatch 生效之前)核对合成键真的不在真表里 ——
#: 否则构造的对象就是真删槽,断言会随槽表演进(本文件为此栽过一次)。
_COLLISIONS = {k for k in _SYN_SLOTS
               if k in SLOT_MAP or k in SLOT_LABEL or k in DROP_REASON}


@pytest.fixture
def vanished(monkeypatch, tmp_path):
    """一个**真的整维没有对应指标**的维度 —— 走完整装配,不是手拼 dict。

    ## 2026-11-22(同日第二次):构造方式从"借真删槽"改成"合成槽键"

    真表上已经没有删槽了(`eye_contact` 也换成了 `gaze_dwell_p50`),
    借真删槽这条路走不通。合成的只是**槽键与名字**,装配链一条没换:
    `map_l1_to_dimensions` 仍按 `SLOT_MAP.get(slot)[0] is None` 判删槽,
    `_printable_vanished` 仍从 `SLOT_LABEL` / `DROP_REASON` 取人话。
    """
    assert not _COLLISIONS, (
        f"合成槽键撞上了真表:{sorted(_COLLISIONS)} —— 请换键,"
        "构造必须验的是「没登记/已删」这条路,不能借用真槽")
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path / "no_rec"))
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path / "no_series"))
    weights = (0.4, 0.35, 0.25)
    for (slot, (label, plain)), weight in zip(_SYN_SLOTS.items(), weights):
        monkeypatch.setitem(SLOT_MAP, slot, (
            None, f"合成理由:不拿任何现有指标顶替({slot}),只为走通渲染层与审计分层"))
        monkeypatch.setitem(SLOT_LABEL, slot, label)
        monkeypatch.setitem(DROP_REASON, slot, plain)
    monkeypatch.setitem(l1_mapper.DIMENSIONS, _SYNTHETIC_KEY, {
        "name": "构造用整维",
        "slots": dict(zip(_SYN_SLOTS, weights)),
    })
    got = build_l1_report(_SID, logs={})["vanished_dimensions"]
    assert [v["key"] for v in got] == [_SYNTHETIC_KEY], (
        f"消失的维度不是构造进来的那一个:{[v['key'] for v in got]}")
    return got


def test_the_real_payload_no_longer_vanishes_any_dimension(monkeypatch, tmp_path):
    """★ 2026-11-22 的新事实:**没有任何维度整体消失**了。

    `行为表现活跃度` 3/3(`shrug_elevation_p90` + `hand_openness_p50` +
    `gaze_dwell_p50`)、`情境行为稳定性` 3/3(`facial_asymmetry_p90`)、
    `沟通表达流畅度` 3/3、`作答过程特征` 2/2 ⟹ 五个维度全在 `dimensions` 里,
    **14 个槽全部有对应指标**。

    为什么这条要单独存在:上面四条测试用**构造**的维度跑(因为生产数据里没有了)。
    若哪天有人把三条新指标撤掉,那四条**照样绿**(它们不读真表),而报告会悄悄
    退回"四维"。这条就是那个缺口的地板。

    红法:把 `left_hand_spread` / `shrug_level` / `eye_contact` 任一改回 `None` ⟹ 本条红。
    """
    monkeypatch.setenv("JINGXIN_RECORDINGS_DIR", str(tmp_path / "no_rec"))
    monkeypatch.setenv("JINGXIN_SERIES_DIR", str(tmp_path / "no_series"))
    rec = build_l1_report(_SID, logs={})
    assert rec["vanished_dimensions"] == [], (
        f"又有维度消失了:{[v['key'] for v in rec['vanished_dimensions']]}")
    assert len(rec["dimensions"]) == 5, sorted(rec["dimensions"])
    assert rec["coverage"]["n_no_indicator"] == 0
    # ⚠️ 2026-11-22(同日第二次):`n_not_measured` 3 → **0**(最后一个删槽
    #    `eye_contact` 也换成 `gaze_dwell_p50`)、`n_measurable` 11 → **14**。
    #    槽总数仍是 14。注意:这是**空数据**载荷,"可测 14"不等于"过门 14"
    #    (这条用 `logs={}` ⟹ 一条值都没有 ⟹ `n_passed == 0`)。
    assert rec["coverage"]["n_not_measured"] == 0
    assert rec["coverage"]["n_measurable"] == 14
    assert rec["coverage"]["n_passed"] == 0, "空数据下过门数必须是 0,不许按可测数报"
    assert rec["coverage"]["n_slots"] == 14


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
