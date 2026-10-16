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

| 维度(显示名) | **死权重** | 可达 |
|---|---|---|
| 话语结构特征 | **0.0** | 1.0 |
| 情境行为稳定性 | **0.0** | 1.0 |
| **沟通表达流畅度** | **5/7 ≈ 0.714** | **2/7 ≈ 0.286** |
| 行为表现活跃度 | **0.0** | 1.0 |
| 作答过程特征 | **0.0** | 1.0 |

⚠️ **原来的「维度权重」那一列已删**(2026-10-04):跨维度权重表 `dimension_weights`
随复合分一起从 payload 摘除了(理由见下面那段"陷阱已摘除")。本表现在只说
**每一维自己**有多少声明证据是活的 —— 那件事与"维度之间怎么加权"无关。

⟹ **2026-10-03 之前**沟通表达流畅度 **80%** 的声明证据是死的(全仓最糟);
删掉 `speech_ratio` 槽、余下按原比例重归一之后降到 **5/7 ≈ 0.714**
(余下 `fluency_score` **全系统无产出方**、`pause_duration` **卡标定集**)。
它现在是**唯一**还薄的维度。

同一天还有三处变活:`话语结构特征` 0.3 → **0.0**(删掉 `focus_score` 槽)、
`情境行为稳定性` 0.2 → **0.0**(删掉 `gaze_deviation` 槽)、
`作答过程特征`(原显示名「言语流畅特征」)0.5 → **0.0**
(删掉 `au7_freq` 0.3 + `blink_rate` 0.2 两个槽)。
三条都是"指向待删列"的槽,处置见 `docs/下一步.md`「丙、17」。
`行为表现活跃度` 从 0.3 死一路降到 **0.0 死**:解封 `eye_contact` 去掉 0.3,
2026-10-04 删掉 `energy` 槽(永久过不了门,见「甲、20」)再去掉 0.2。
⚠️ 那 0.2 是**同一天先被本表正确地数上、随后又被删掉**的 —— 经过见下面那条边界。

## 一条边界(2026-10-04 补):封停这一支判错过一次,而且是**漏**不是误报

本表原来用 `is_quarantined(关键词)` 判封停 —— 那是把**槽关键词**直接当**特征键**喂进去,
而两者的长度关系**两个方向都有**:

* `focus_score` ↔ 封停键 `focus_score`:等值,旧实现抓得到;
* `energy` → 封停键 **`energy_mean`**:封停键**更长**,旧实现**抓不到**
  ⟹ `confidence_level` 的死权重登记成 **0.0**,真值 **0.2**。

这与 `tests/test_report_layer.py` 那条「待删列」守卫是**同一类洞的镜像**(那边是槽关键词
比列名短:`blink_rate` ⊂ `blink_rate_per_min`)。两处现在都是**双向子串**。
⚠️ 这条边界**没被单独做成一件事** —— 它是 `energy` 那一片的必要组成部分(那一维要如实
登记"永久拿不到 0.2 的证据",登记表就不能同时说它 100% 活)。
⚠️ 2026-10-04 当晚又往前一步:**那个槽整个删掉了**(理由见「甲、20」)。
⟹ `_quarantine_key_for` 现在**没有真实用例**了(唯一双向子串才命中的就是 `energy`),
但它与它的**直接单元测试**都留着:那一类洞(槽关键词比封停键短)还会再来,
而留着它的成本是零。

⚠️ `作答过程特征` 那一维**连显示名一起改了**:原「言语流畅特征」既与它的槽
(回答详尽度 / 反应延迟)对不上,又与 `communication_fluency` 的「沟通表达流畅度」
**撞词** —— 报告与雷达图会把两个名字并排渲染。改名的完整理由在
`research_mapper.py` 那段注释里;这个名字由本文件的 `_DEAD_WEIGHT` 钉着(名字一变就红)。

## 这张表还牵出一个陷阱(登记,但本轮**不改**)

## 那个陷阱**已经不存在了**(2026-10-04 摘除,不是修好)

它原来是:`total_score` 的聚合只按"有没有分"归一

    den = sum(dimension_weights[k] for k, d in ... if d["score"] is not None)

⟹ 一个**只有 2/7 个槽撑着**的维度与一个可达 0.8 的维度拿**一样重**的权重,于是复合分里
声明证据死掉 5/7 的那个维度比全活的维度**权重更大**。

**处置是"整层摘除"而不是"改口径"**:`total_score` / `total_level` / 逐维度 `score` / `level`
/ `dimension_weights` 一起从 payload 里删掉了。理由不是"那个聚合算错了",而是**它不该存在**
—— 见 `research_mapper` 返回处那段注释(未标定标尺 + 分母不是常数 + 它是对人的复合判定),
以及 `tests/test_report_layer.py::test_the_payload_carries_no_scoring_keys_at_all` 那条守卫。
实测过的现形:场 `9b2b` 剔掉一个**假过门**之后 `stress_resilience` 反而 59.35 → 94.68。

⟹ 本文件保留的是那张表的**另一件事**:**每一维自己**有多少声明证据是死的。
那件事仍然重要(它决定"这一维能说什么"),只是不再与任何加权有关。
"""
from __future__ import annotations

import importlib
from pathlib import Path

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
    "confidence_level": ("行为表现活跃度", 0.0,
                         "全活 —— 2026-10-04 删掉 `energy` 槽之后的三个槽都在。"
                         "⚠️ `left_hand_spread` 只在手可见的帧上有值(实测 4 帧 ⟹ G3 拦),"
                         "那是**数据限制**不是结构性死,所以不计入死权重"),
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


def _quarantine_key_for(kw: str) -> str | None:
    """槽关键词命中了哪条封停 —— **双向子串**取最长,不是 `is_quarantined(关键词)`。

    为什么不能只查关键词:`is_quarantined(x)` 的语义是「**封停键**是否为 `x` 的子串」,
    而映射函数喂给它的 `x` 是**解析后的特征键**(`face_focus_score_mean` 之类)。
    槽关键词与封停键的**长度关系两个方向都有**:

    * `focus_score` ↔ 封停键 `focus_score` —— 等值,旧实现抓得到;
    * `energy` → 封停键 **`energy_mean`** —— 封停键**更长**,旧实现**抓不到**
      ⟹ 死权重登记成 0.0,而真值是 **0.2**(`confidence_level` 的 `energy` 槽)。

    这与 `tests/test_report_layer.py` 那条「待删列」守卫是**同一类洞的镜像**:
    那边是槽关键词比列名短(`blink_rate` ⊂ `blink_rate_per_min`),
    这边是槽关键词比封停键短。**两处都改成双向子串**,理由同一个。
    (实测:本函数在 15 个槽上跑一遍,命中恰好是 `pause_duration` 与 `energy`,
    零误报;token 交集那种更松的判据实测有 11 条误报,不用。)
    """
    from report_frontend.evidence_gate import QUARANTINE

    hits = [q for q in QUARANTINE if q == kw or q in kw or kw in q]
    return max(hits, key=len) if hits else None


def _dead_weight_of(spec: dict) -> tuple:
    """返回 `(死权重, [死掉的槽...])`。

    四个判据,顺序 = 一个"若它真能拿到数据"的槽被拦下的可能性,从最不可能改变到最可能:
    封停 / 无生产者 / **门槛结构性不可达** / 指向待删列。

    ⚠️ **每个槽只计一次**(权重不会重复计入),所以"先判哪个"不影响那个**数**;
    它只影响 `why` 里写的**归因**。归因顺序按"最贴近真实拦截点"排:一个同时被封停
    又门槛不可达的槽,它剩下的拦路虎是封停(G3 修好后 G4 才是它报的那个原因),
    所以封停排前面。

    ⚠️ 封停这一支用 `_quarantine_key_for`(**双向子串**),不是 `is_quarantined(关键词)`
    —— 后者对 `energy`(封停键 `energy_mean` 更长)会漏,理由见那个函数的 docstring。
    """
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
        if _quarantine_key_for(kw):
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
    """★ 死权重超过一半的维度必须**有人看得见** —— 而且要在文档里被点名。

    ⚠️ 2026-10-04 **这条测试的意义变了**。它原来钉的是那个陷阱(薄维度与全活维度
    在复合分里拿一样重)。复合分连同跨维度权重已**整层摘除**,陷阱随之消失 ⟹
    本测试保留的是它的**另一半**用途:**薄维度不能被静默当成与全活维度同等可信**。

    现在的判据落到**可见性**上,两条都是机器可查的:
      ① 薄维度必须在本文件的 `_DEAD_WEIGHT` 里登记(**并写出为什么**);
      ② 它的**显示名必须出现在 `docs/下一步.md`** —— "有人看得见"这句话要么能被验证,
         要么就只是一句声明。

    红法(实测):把某个薄维度的显示名改掉、不同步文档 ⟹ 第 ② 条红;
    把某个薄维度的死权重登记值改小到一半以下 ⟹ 本条跳过(它不再被当"薄"的),
    而 `test_registered_dead_weight_matches_reality` 会因为新值不匹配而红。
    """
    _display, dead, why = _DEAD_WEIGHT[dim]
    # ⚠️ 2026-10-16:这里原来是 `if dead < 0.5: pytest.skip(...)`。跳过会让下面三条断言
    #   对那几个参数**一行都不执行**,而 `tests/test_assert_coverage.py` 的 never_executed
    #   闸拒绝给从未执行过的 assert 背书 —— 它的判词是对的(没执行过 = 没测)。
    #   改法不是给它开豁免,而是把「不适用」写成**对补集的正面断言**:薄(死权重 ≥ 一半)
    #   ⟹ 这三条都要成立;不薄 ⟹ 本条不适用。判据一字未松,而且现在每个参数都真的走到这三行。
    thin = dead >= 0.5
    assert why.strip() or not thin, f"{dim} 是薄维度却没有登记理由 —— 薄必须有人看得见"
    assert _display.strip() or not thin, f"{dim} 没有显示名(且是薄维度)"

    docs = (Path(__file__).resolve().parents[1] / "docs" / "下一步.md").read_text(encoding="utf-8")
    assert _display in docs or not thin, (
        f"薄维度「{_display}」没有被 `docs/下一步.md` 点名 —— "
        f"「哪个维度是薄的」必须有人看得见,而不是只躺在测试文件里")


def test_the_quarantine_lookup_catches_a_longer_registry_key():
    """★ 正向对照:封停那一支必须抓到「封停键比槽关键词**长**」那一类。

    实测过的漏洞(不是构造的):`energy` → 封停键 **`energy_mean`**。旧实现
    `is_quarantined("energy")` 返回 `None`(`is_quarantined` 的语义是"封停键是否为
    **输入**的子串",而这里输入的是**槽关键词**、不是解析后的特征键)⟹
    `confidence_level` 的死权重登记成 **0.0**,真值 **0.2**。

    红法(实测):把 `_quarantine_key_for` 换回 `is_quarantined(kw)` ⟹ 本条红
    (第一条断言拿到 `None`),且 `test_registered_dead_weight_matches_reality` 也红
    (死权重 0.2 → 0.0)。已登记成变异。

    ⚠️ 第二条断言看着多余,其实是**前提守卫**:哪天有人把封停键改名成 `energy`,
    这条会红并逼着重新论证 —— 否则"双向子串"会退化成"等值也能过",而洞重新打开
    时不会有人知道。
    """
    from report_frontend.evidence_gate import is_quarantined

    assert _quarantine_key_for("energy") in ("energy_mean", "energy_level")
    assert is_quarantined("energy") is None, (
        "`is_quarantined('energy')` 现在能命中了 —— 封停键大概改名了,"
        "本函数的必要性要重新论证(见 docstring)")
    # 反方向(等值那一类)也必须通
    assert _quarantine_key_for("focus_score") == "focus_score"
    # 且不许误报:实测 15 个槽里只有 `pause_duration` 与 `energy` 命中
    for kw in ("n_chars", "reaction_time", "eye_contact", "pitch_variation",
               "au4_frown", "connective_density", "left_wrist_jitter_world"):
        assert _quarantine_key_for(kw) is None, f"{kw} 被误判成封停 —— 判据过松"
