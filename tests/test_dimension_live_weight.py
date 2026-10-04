# tests/test_dimension_live_weight.py
"""每个维度**还有多少声明权重是活的** —— 登记 + 守卫。

## 为什么需要

维度表里每个槽都写着权重,但一个槽可能**结构性用不上**:

* **封停**:`evidence_gate.QUARANTINE` 命中 ⟹ G4 永远拦(如 `focus_score`);
* **无生产者**:在 `research_mapper.NO_PRODUCER` 里 ⟹ G1 永远过不了
  (如 `fluency_score`,全系统没有任何代码在算它);
* **门槛结构性不可达(单位错配)**:G3 的门槛拿的是**帧级**默认(10),而这个槽是
  **每段回答一个样本**(题库每场 **8 题**)⟹ 答满全场也过不了(实测 `pitch_variation`);
* **指向待删列**:`l0_columns.json` 白名单里标了「删列」的列(如 `speech_ratio`)。

⭐ **第三条是 2026-10-03 才补进本表的,它的教训值得单独读**:前三条判据都只看
「这个槽有没有**路**」,第三条看的是「这条路**走得通吗**」。它漏掉时的症状
**长得像"数据不够"** —— 报告写「语调变化: 有效样本不足」,读的人会去多答几题,
而多答也不行(题库只有 8 题)。实测:在唯一一场有语音数据的会话(`..._95a8`)上,
`communication_fluency` 唯一那个看起来活的槽 `pitch_variation` 被 G3 永久拦下,
**本表当时登记的死权重 5/7 是错的,真值 7/7**。补上判据 + 修好门槛单位之后,
那个维度才第一次真的出货(0/3「证据不足」→ 1/3,score 51.39,confidence 低)。

于是"这个维度声明的证据"与"它实际能拿到的证据"是**两个数**,而后者此前**没有任何地方记着**。
2026-10-03 实测(四场高质量素材的同一套表):

| 维度(显示名) | 维度权重 | **死权重** | 可达 |
|---|---|---|---|
| 话语结构特征 | 0.25 | **0.0** | 1.0 |
| 情境行为稳定性 | 0.25 | **0.0** | 1.0 |
| **沟通表达流畅度** | **0.20** | **5/7 ≈ 0.714** | **2/7 ≈ 0.286** |
| 行为表现活跃度 | 0.15 | **0.0** | 1.0 |
| 作答过程特征 | 0.15 | **0.0** | 1.0 |

⟹ **2026-10-03 之前**沟通表达流畅度 **80%** 的声明证据是死的(全仓最糟);
删掉 `speech_ratio` 槽、余下按原比例重归一之后降到 **5/7 ≈ 0.714**
(余下 `fluency_score` **全系统无产出方**、`pause_duration` **卡标定集**)。
它现在是**唯一**还薄的维度。

同一天还有三处变活:`话语结构特征` 0.3 → **0.0**(删掉 `focus_score` 槽)、
`情境行为稳定性` 0.2 → **0.0**(删掉 `gaze_deviation` 槽)、
`作答过程特征`(原显示名「言语流畅特征」)0.5 → **0.0**
(删掉 `au7_freq` 0.3 + `blink_rate` 0.2 两个槽)。
三条都是"指向待删列"的槽,处置见 `docs/下一步.md`「丙、17」。
`行为表现活跃度` 保持 100% 活 —— 那是解封 `eye_contact` 的**直接结果**(解封前它也有 0.3 死)。

⚠️ `作答过程特征` 那一维**连显示名一起改了**:原「言语流畅特征」既与它的槽
(回答详尽度 / 反应延迟)对不上,又与 `communication_fluency` 的「沟通表达流畅度」
**撞词** —— 报告与雷达图会把两个名字并排渲染。改名的完整理由在
`research_mapper.py` 那段注释里;这个名字由本文件的 `_DEAD_WEIGHT` 钉着(名字一变就红)。

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
                              "2026-10-03 删掉 speech_ratio 槽后按比例重归一。"
                              "⚠️ 另 2/7 的 pitch_variation **直到同日都是被 G3 结构性挡住的**"
                              "(pitch 族门槛还是帧级 10,而本列每段回答一个样本、题库每场 8 题"
                              "⟹ 实测 n_valid=8 < 10),修好门槛单位后才真的算活 —— 见「甲、2」"),
    "confidence_level": ("行为表现活跃度", 0.0, "全活 —— 2026-10-03 解封 eye_contact 的直接结果"),
    "cognitive_efficiency": ("作答过程特征", 0.0,
                             "全活 —— 2026-10-03 删掉 au7_freq(0.3)与 blink_rate(0.2)"
                             "两个「指向待删列」的槽之后;同日改名(原「言语流畅特征」)"),
}

#: 浮点比较容差(权重都是 0.1 的整数倍)。
_EPS = 1e-9


def _threshold_unreachable(kw: str) -> bool:
    """这个槽的门槛是不是**结构性不可达**(单位错配)。

    判据只用登记数据:`evidence_thresholds.json` 的 `n_valid_units[族] == "answer"`
    (一行 = 一段回答;题库每场 **8 题**)而门槛又 **≥ `_default_n_valid`**(帧级默认 10)
    ⟹ 答满全场也过不了 G3。族名用生产侧的 `threshold_family_for` 取 —— **不许在这里
    再抄一份子串匹配**,那正是这个洞的成因(关键词与真实键不是一个字符串)。

    ⚠️ 为什么这条必须进死权重表:它的**症状长得像"数据不够"**。实测 `pitch_variation`
    在唯一一场有语音数据的会话上被 G3 拦下,报告写的是「语调变化: 有效样本不足」——
    读的人会去多答几题,而多答也不行(题库只有 8 题)。修掉之前,本表把 communication_fluency
    死权重登记成 5/7,而**真值是 7/7**。
    """
    from report_frontend.evidence_gate import load_thresholds, threshold_family_for

    fam = threshold_family_for(kw)
    if fam is None:
        return False
    data = load_thresholds()
    return (data["n_valid_units"].get(fam) == "answer"
            and data["thresholds"][fam] >= data["_default_n_valid"])


def _dead_weight_of(spec: dict) -> tuple:
    """返回 `(死权重, [死掉的槽...])`。

    四个判据,顺序 = 一个"若它真能拿到数据"的槽被拦下的可能性,从最不可能改变到最可能:
    封停 / 无生产者 / **门槛结构性不可达** / 指向待删列。

    ⚠️ **每个槽只计一次**(权重不会重复计入),所以"先判哪个"不影响那个**数**;
    它只影响 `why` 里写的**归因**。归因顺序按"最贴近真实拦截点"排:一个同时被封停
    又门槛不可达的槽(如 `energy`),它剩下的拦路虎是封停(G3 修好后 G4 才是它报的
    那个原因),所以封停排前面。
    """
    from report_frontend.evidence_gate import is_quarantined
    from report_frontend.research_mapper import NO_PRODUCER

    l0 = importlib.import_module("l0_columns")
    # ⚠️ 这里是 `kw in doomed`(**等值**),对"关键词与列名不是一个字符串"的槽会漏
    #    (实例:`blink_rate` vs `blink_rate_per_min`)。本函数只用它算**死权重这个数**;
    #    "槽该不该还在表里"由 `tests/test_report_layer.py` 那条双向子串的守卫管,
    #    两处分工与那次漏判的经过写在本文件 `test_registered_dead_weight_matches_reality`
    #    的 docstring 里。**不许**把这里的漏当成"这个槽没问题"的依据。
    doomed = {a["column"] for a in l0.load().get("legacy_allowlist", [])
              if "删列" in str(a.get("planned", ""))}

    dead, why = 0.0, []
    for kw, w, _pos, _name, _imp in spec.get("indicators", []):
        reason = None
        if is_quarantined(kw):
            reason = "封停"
        elif kw in NO_PRODUCER:
            reason = "无生产者"
        elif _threshold_unreachable(kw):
            reason = "门槛结构性不可达(单位错配)"
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
    - 有人**解封**一个槽 ⟹ 死权重变小 ⟹ 红(实测:解封 `pause_duration`,它在
      `communication_fluency` 里只被**封停**、没指向待删列,权重 1/7 ⟹ 5/7 → 4/7)。
      ⚠️ **例外(实测踩到,记在此处)**:槽**同时**「封停」且「指向待删列」时
      (如 `focus_score` / `speech_ratio`),摘掉封停**不会**改变死权重 ——
      下面的判据里**封停优先**,它仍被算作死的。我第一版把"解封 ⟹ 红"写成无条件的,
      拿 `focus_score` 去验**没红**;换成只封停的槽才红。
      这条判据看的是"它**还是不是**死的",不是"它头上的标签是哪一个"。

    ⚠️ **这条钉子的射程边界(2026-10-03 补记,吃过一次亏)**:它靠的是
    `is_quarantined(kw) / NO_PRODUCER / 待删列` 三条**按关键词**的判据,而
    `_DEAD_WEIGHT` 只登记**权重数字**。所以"关键词与源列名对不上"的槽
    (实例:`blink_rate` → 源列 `blink_rate_per_min`,而这里比的是
    `kw in doomed` **等值**)在这里**也算活着** —— 我当时就是据此写下
    「这两个槽不属于『指向待删列』」这句**错话**,并拿它去给使用者做决策依据。
    补洞在 `tests/test_report_layer.py::test_no_slot_points_at_a_column_marked_for_deletion`
    (双向子串 + 正向对照)。本文件与它的分工:本文件守"**有多少声明权重是死的**"
    (数字),那里守"**这个槽还该不该在表里**"(身份)。
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
