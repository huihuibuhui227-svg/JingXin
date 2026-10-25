# tests/test_low_confidence_reasons_are_readable.py
"""「置信度为「低」的原因」那一段,读者不认识里面的指标名。

## 实测缺陷(2026-10-25 深夜,服务器上真跑的一场 `20260928_180131_8102`)

交付物正文里印着:

    置信度为「低」的原因（素材限制，不是算错）
      · speech_rate:整场只有一次测量(转写字数 ÷ 有声秒数),n_valid=1 ⟹ …
      · pause_rate_per_min:整场只有一次测量(停顿段数 ÷ 有声分钟数),n_valid=1。…
      · pitch_range_semitone:整场音频没有题目边界,…
      · pitch_variability_per_sec:同上:…
      · energy_dynamics_per_sec:同上:…

5 个内部指标名进了读者眼前。理由写得很清楚,但**读者不知道说的是哪个指标** ——
而同一份报告里,同一个指标在「沟通表达流畅度」那一维印的是**中文槽名**
(「语调变化」「平均停顿时长」)。

⟹ 同一个东西在一份报告里有两个名字,读者会以为是两个东西。

## 与前面两次是同一个形态(第 15/16 条)

`comparability_notes` 与「删槽理由」都走过这条:`explain_confidence()` 拼的是
`{指标名}:{理由}`,而指标名是**审计键**。前两次的修法是「载荷发两版」,
这次照办:

* `low_confidence_notes`       —— 审计版(带指标名),核对口径的人查;
* `low_confidence_notes_plain` —— 人话版(带中文名),交付物与界面印这一版。

## 这个文件守什么

1. 载荷真的发人话版,且**条数、顺序、理由文字**与审计版逐条对齐(只换名字);
2. 没有中文名的指标**原样露出英文**,不吞掉 —— 宁可露英文,不许把这一条藏起来;
3. 交付物印的是人话版;
4. 前端类型里登记了这个字段。
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from report_frontend import l1_quality

REPO = Path(__file__).resolve().parents[1]


# ── 名字表本身 ───────────────────────────────────────────────────────────

def test_every_low_confidence_indicator_has_a_chinese_name():
    """★ 红法:从 `INDICATOR_LABEL` 删掉任意一条 ⟹ 这里红。

    这张表是**唯一**的指标名 → 中文名登记处。漏一条 ⟹ 那条会露出英文键,
    而读者刚刚在维度表里见过它的中文名,会以为是两个东西。
    """
    missing = [n for n in l1_quality.LOW_CONFIDENCE_REASON
               if n not in l1_quality.INDICATOR_LABEL]
    assert not missing, (
        f"这些指标没有中文名,交付物会露出英文键:{missing}")


def test_the_plain_names_are_chinese_not_keys():
    """★ 红法:把 `INDICATOR_LABEL` 的某个值改成指标键 ⟹ 红。"""
    for name, label in l1_quality.INDICATOR_LABEL.items():
        assert re.search(r"[一-龥]", label), f"{name} 的中文名里没有汉字:{label!r}"
        assert label != name, f"{name} 的'中文名'就是它自己"


# ── 两版逐条对齐 ─────────────────────────────────────────────────────────

def _fake_report(names_and_reasons):
    """只造 `explain_confidence` 需要的形状(`indicators` 里有值且置信「低」)。"""
    class _R:
        indicators = [{"name": n, "confidence": "低", "value": 1.0}
                      for n, _ in names_and_reasons]
    return _R()


def test_the_plain_version_keeps_every_entry_and_only_swaps_the_name():
    """★ 红法:人话版按"有中文名的才发"过滤 ⟹ 这里红。

    过滤是最容易犯的错:少一条读者就少知道一条限制,而他无从发现。
    两版必须**条数相同、顺序相同、理由逐字相同**,差别只在名字。
    """
    names = list(l1_quality.LOW_CONFIDENCE_REASON)
    rep = _fake_report([(n, None) for n in names])

    audit = l1_quality.explain_confidence(rep)
    plain = l1_quality.explain_confidence(rep, plain=True)

    assert len(plain) == len(audit) == len(names)
    for a, p in zip(audit, plain):
        name_a, _, reason_a = a.partition(":")
        name_p, _, reason_p = p.partition(":")
        assert reason_a == reason_p, (
            f"理由被改了 —— 两版必须逐字相同:\n审计:{reason_a}\n人话:{reason_p}")
        # ⚠️ 名字必须**恰好**等于登记的中文名,不能只是"没有冒号"。
        #    第一版写成 `name_a == name_p or name_a in LOW_CONFIDENCE_REASON` ——
        #    那个 `or` 让整条恒真(右边永远是 True),把 `plain` 参数删掉都照样绿。
        assert name_p == l1_quality.INDICATOR_LABEL[name_a], (
            f"人话版的名字不对:{name_p!r} ≠ {l1_quality.INDICATOR_LABEL[name_a]!r}")
        assert name_p != name_a, f"人话版没有换名字:{name_p!r}"


def test_an_unregistered_indicator_keeps_its_key_visible():
    """★ 红法:未登记时返回 `None` 或空串(把这一条吞掉)⟹ 红。

    宁可露出英文,也不许把「这一条为什么只有这么信」藏起来 ——
    读者会以为它没有限制。
    """
    l1_quality.LOW_CONFIDENCE_REASON["_未登记示例"] = "示例理由"
    try:
        rep = _fake_report([("_未登记示例", None)])
        plain = l1_quality.explain_confidence(rep, plain=True)
        assert len(plain) == 1, "未登记的指标被吞掉了"
        assert plain[0].startswith("_未登记示例:"), (
            f"未登记的指标应当原样露出键名:{plain[0]!r}")
    finally:
        del l1_quality.LOW_CONFIDENCE_REASON["_未登记示例"]


# ── 载荷 ─────────────────────────────────────────────────────────────────

def test_the_payload_carries_both_versions():
    """★ 红法:删掉 `l1_report` 里 `low_confidence_notes_plain` 那一行 ⟹ 红。

    **两个版本都发**:审计版给核对口径的人,人话版给读者。
    只发人话版 ⟹ 复核时查不到是哪条指标;只发审计版 ⟹ 两个出口都漏英文。
    分层的决定权在载荷,不在渲染层。
    """
    src = (REPO / "report_frontend" / "l1_report.py").read_text(encoding="utf-8")
    assert '"low_confidence_notes"' in src
    assert '"low_confidence_notes_plain"' in src, (
        "载荷只发了审计版 —— 交付物与界面会印出指标键")
    assert "plain=True" in src, "人话版不是用 `plain=True` 生成的"


# ── 交付物 ───────────────────────────────────────────────────────────────

def test_the_html_report_prints_the_plain_version():
    """★ 红法:让 `report_generator` 退回读 `low_confidence_notes` ⟹ 红。

    走渲染层:**渲染层不许自己有一张名字表**(两张表迟早分叉),
    它只能照抄载荷给的那一版。
    """
    from report_frontend.report_generator import ReportGenerator

    seen = {}

    gen = ReportGenerator(output_dir="/tmp")
    result = {
        "dimensions": {}, "vanished_dimensions": [],
        "coverage": {"n_passed": 0, "n_slots": 14, "confidence_cap": "无",
                     "passed_slots": []},
        "summary_narrative": "", "model_metadata": {"version": "t"},
        "l1": {
            "low_confidence_notes": ["speech_rate:审计版理由"],
            "low_confidence_notes_plain": ["语音字数率:人话版理由"],
            "gaps": [], "comparability_notes": [], "comparability_notes_plain": [],
            "windows": {},
        },
    }
    html = gen._render_l1_audit_text(result)
    assert "人话版理由" in html, f"交付物没印人话版:{html}"
    assert "审计版理由" not in html, f"交付物印了审计版:{html}"
    assert "speech_rate" not in html, f"交付物露出了指标键:{html}"
