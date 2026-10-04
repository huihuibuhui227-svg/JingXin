# tests/test_dimension_live_weight.py
"""每个维度**还有多少声明权重是活的** —— 登记 + 守卫。

## 为什么需要

维度表里每个槽都写着权重,但一个槽可能**结构性用不上**:

* **封停**:`evidence_gate.QUARANTINE` 命中 ⟹ G4 永远拦(如 `focus_score`);
* **无生产者**:在 `research_mapper.NO_PRODUCER` 里 ⟹ G1 永远过不了
  (如 `fluency_score`,全系统没有任何代码在算它);
* **指向待删列**:`l0_columns.json` 白名单里标了「删列」的列(如 `speech_ratio`)。

于是"这个维度声明的证据"与"它实际能拿到的证据"是**两个数**,而后者此前**没有任何地方记着**。
2026-10-03 实测(四场高质量素材的同一套表):

| 维度(显示名) | 维度权重 | **死权重** | 可达 |
|---|---|---|---|
| 话语结构特征 | 0.25 | **0.0** | 1.0 |
| 情境行为稳定性 | 0.25 | **0.0** | 1.0 |
| **沟通表达流畅度** | **0.20** | **5/7 ≈ 0.714** | **2/7 ≈ 0.286** |
| 行为表现活跃度 | 0.15 | **0.0** | 1.0 |
| 言语流畅特征 | 0.15 | 0.5 | 0.5 |

⟹ **2026-10-03 之前**沟通表达流畅度 **80%** 的声明证据是死的(全仓最糟);
删掉 `speech_ratio` 槽、余下按原比例重归一之后降到 **5/7 ≈ 0.714**
(余下 `fluency_score` **全系统无产出方**、`pause_duration` **卡标定集**)。

同一天还有两处变活:`话语结构特征` 0.3 → **0.0**(删掉 `focus_score` 槽)、
`情境行为稳定性` 0.2 → **0.0**(删掉 `gaze_deviation` 槽)——
两者都是"指向待删列"的槽,处置见 `docs/下一步.md`「丙、17」。
`行为表现活跃度` 保持唯一 100% 活 —— 那是解封 `eye_contact` 的**直接结果**(解封前它也有 0.3 死)。

## 这张表还牵出一个陷阱(登记,但本轮**不改**)

`total_score` 的聚合只按"**有没有分**"归一:

    den = sum(dimension_weights[k] for k, d in ... if d["score"] is not None)

⟹ 一个**只有 2/7 个槽撑着**的维度(`communication_fluency`)与一个**证据全活**的维度,
拿**一样重**的权重。于是在复合分里,声明证据死掉 5/7 的那个维度(0.20)比
完全活的 `confidence_level`(0.15)**权重更大**。

**本轮不改**:`total_score` / `total_level` 明确标注"只为 API 稳定保留、报告层一个字都不渲染"
(`research_mapper.py` 的三处注释 + spec §5.4 :157-158),而**被展示的那一层**是 `coverage`
(报"过门 N/M 个槽 + 逐指标实测区间 + 置信度上限",实测是诚实的)。
改复合权重是**构念决策**,且要重新登记,不该顺手做。详见 `docs/下一步.md`「丙、18」。
"""
from __future__ import annotations

import importlib

import pytest

#: 每个维度的死权重登记(与显示名一起写,免得键名与显示名对不上时读错行)。
#: 键 = `mapping_rules` 的键。值 = (显示名, 该死权重, 为什么)。
#:
#: ⚠️ 这份表**不是免罪符**,是**登记**:死权重变了就说明有人动了封停/生产者/待删名单,
#: 那正是要**显式**看见的事(解封 `eye_contact` 时就该来这里改一行)。
_DEAD_WEIGHT = {
    "logical_thinking": ("话语结构特征", 0.0,
                         "全活 —— 2026-10-03 删掉 focus_score 槽之后"),
    "stress_resilience": ("情境行为稳定性", 0.0,
                          "全活 —— 2026-10-03 删掉 gaze_deviation 槽之后"),
    "communication_fluency": ("沟通表达流畅度", 5 / 7,
                              "fluency_score 无生产者(4/7) + pause_duration 封停(1/7);"
                              "2026-10-03 删掉 speech_ratio 槽后按比例重归一"),
    "confidence_level": ("行为表现活跃度", 0.0, "全活 —— 2026-10-03 解封 eye_contact 的直接结果"),
    "cognitive_efficiency": ("言语流畅特征", 0.5, "au7_freq 封停(0.3) + blink_rate 封停(0.2)"),
}

#: 浮点比较容差(权重都是 0.1 的整数倍)。
_EPS = 1e-9


def _dead_weight_of(spec: dict) -> tuple:
    """返回 `(死权重, [死掉的槽...])`。"""
    from report_frontend.evidence_gate import is_quarantined
    from report_frontend.research_mapper import NO_PRODUCER

    l0 = importlib.import_module("l0_columns")
    doomed = {a["column"] for a in l0.load().get("legacy_allowlist", [])
              if "删列" in str(a.get("planned", ""))}

    dead, why = 0.0, []
    for kw, w, _pos, _name, _imp in spec.get("indicators", []):
        reason = None
        if is_quarantined(kw):
            reason = "封停"
        elif kw in NO_PRODUCER:
            reason = "无生产者"
        elif kw in doomed:
            reason = "指向待删列"
        if reason:
            dead += w
            why.append(f"{kw}({w},{reason})")
    return dead, why


def test_registered_dead_weight_matches_reality():
    """★ 每个维度的死权重必须与登记值一致 —— **两个方向都红**。

    红法(两个方向,**都实测过**):
    - 有人**新**封停一个槽 ⟹ 死权重变大 ⟹ 红(实测:封停 `connective_density`,w=0.4);
    - 有人**解封**一个槽 ⟹ 死权重变小 ⟹ 红(实测:解封 `au7_freq`)。
      ⚠️ **例外(我实测踩到,记在此处)**:槽**同时**「封停」且「指向待删列」时
      (如 `focus_score` / `speech_ratio`),摘掉封停**不会**改变死权重 ——
      下面的判据里**封停优先**,它仍被算作死的。我第一版把"解封 ⟹ 红"写成无条件的,
      拿 `focus_score` 去验**没红**;换成只封停的 `au7_freq` 才红。
      这条判据看的是"它**还是不是**死的",不是"它头上的标签是哪一个"。
    """
    from report_frontend.research_mapper import ResearchCapabilityMapper

    rules = ResearchCapabilityMapper().mapping_rules
    bad = []
    for dim, (display, expect, _why) in _DEAD_WEIGHT.items():
        spec = rules.get(dim)
        if spec is None:
            bad.append(f"{dim}: 维度表里没有这个键了")
            continue
        if spec.get("name") != display:
            bad.append(f"{dim}: 显示名从「{display}」变成了「{spec.get('name')}」"
                       f"(键与显示名对不上时,报告会标错名字)")
        dead, why = _dead_weight_of(spec)
        if abs(dead - expect) > _EPS:
            bad.append(f"{dim}: 死权重登记 {expect},实测 {dead} —— 死的槽: {why or '无'}")
    assert not bad, (
        "维度的死权重与登记不符:\n  " + "\n  ".join(bad)
        + "\n—— 若是有意改动(封停/解封/生产者/待删名单),请一并更新本文件的 `_DEAD_WEIGHT`。")
    assert set(rules) == set(_DEAD_WEIGHT), (
        f"维度表与登记表的键对不上:表里 {sorted(rules)},登记里 {sorted(_DEAD_WEIGHT)}")


def test_every_dimension_declares_full_slot_weight():
    """每个维度的槽权重和必须是 1.0 —— 死权重那个数只有在**权重和=1**时才有可比性。

    红法(实测):把任一维度的某个槽权重改 0.1 ⟹ 红。
    """
    from report_frontend.research_mapper import ResearchCapabilityMapper

    rules = ResearchCapabilityMapper().mapping_rules
    bad = [f"{dim}: 权重和 {round(sum(i[1] for i in s['indicators']), 4)}"
           for dim, s in rules.items()
           if abs(sum(i[1] for i in s["indicators"]) - 1.0) > _EPS]
    assert not bad, "维度内槽权重和不是 1.0:\n  " + "\n  ".join(bad)


@pytest.mark.parametrize("dim", sorted(_DEAD_WEIGHT))
def test_thin_dimensions_are_flagged_not_silently_weighted(dim):
    """★ 死权重超过一半的维度,它的 `dimension_weights` **必须**在登记里被点名。

    这条钉的是那个**陷阱**:`total_score` 的聚合只按"有没有分"归一
    (`den = sum(dimension_weights[k] ... if score is not None)`)⟹
    **只有 1/5 个槽撑着的维度与证据全活的维度拿一样重**,于是复合分里
    `communication_fluency`(0.20,声明证据死掉 80%)比 `confidence_level`(0.15,全活)权重还大。

    本轮**不改**该聚合(它是明确不渲染的遗留字段,而展示层 `coverage` 是诚实的);
    但"哪个维度是薄的"必须**有人看得见**,不能将来哪天有人重新渲染 `total_score` 时
    才发现。详见 `docs/下一步.md`「丙、18」。

    红法(实测):把某个薄维度的死权重登记值改小到一半以下 ⟹ 红(它不再被当"薄"的)。
    """
    from report_frontend.research_mapper import ResearchCapabilityMapper

    _display, dead, why = _DEAD_WEIGHT[dim]
    weights = ResearchCapabilityMapper().dimension_weights
    if dead < 0.5:
        pytest.skip(f"{dim} 死权重 {dead} < 0.5,不属于「薄」的那批")
    assert dim in weights, f"{dim} 不在 dimension_weights 里 —— 那 aggregate 时取不到权重"
    assert weights[dim] > 0, (
        f"{dim} 的维度权重是 0 —— 那它死权重再多也影响不到复合分,"
        f"本登记该相应说明(为什么: {why})")
