# report_frontend/l1_mapper.py
"""**M4.5:用 L1 指标替换 `research_mapper` 的 14 个槽。**

## 这一层做什么

`research_mapper` 的 14 个槽是从**扁平特征键**(`feature_engine` 产出)里模糊匹配出来的,
匹配靠列名子串,不保证量的是同一个东西 —— 历史上就因此把「皱眉」匹配到
「从 micro_exp 字符串解析出的垃圾」(见 `research_mapper` 里的记录)。

本模块把这 14 个槽改成**指名的 L1 指标**:每个槽声明自己对应哪一条 L1,
L1 那条有公式、有 `n_valid`、有置信度、有 gap。**同名同构念,不是子串撞上的。**

## 规矩(这是本模块存在的全部意义)

1. **只有同构念才换。** 判据是"量的还是不是同一个东西",**不是**"能不能算出个数"。
   构念不同的宁可不换 —— 换过去就是给旧名字配了个新量,比留空更坏。
2. **没有对应量就删槽,不编值。** 删掉的槽进 `evidence_gaps`,写明"L1 没有这个量"。
   绝不拿相邻构念顶替(那正是 `focus_score` / `gaze_deviation` 被删掉的原因)。
3. **权重在幸存槽上重新归一。** 这是本仓删槽时的既有做法(见 `research_mapper`
   里 `focus_score` 删除时的"余下三个槽 ×1/0.7"),不是本模块新发明的。
4. **置信度取最弱的那条。** 维度陈述的强度不可能超过它最弱的证据。
   旧口径是"填了几个槽"(计数),新口径是"最弱证据有多强" —— 两者含义不同,
   所以 `matched_indicators` 仍然照报计数,两种事实都留着。
5. **不给维度分、不给档位。** 与 `research_mapper` 2026-10-04 的决定一致:
   没有标定样本 ⟹ 任何 0–100 都是未标定标尺上的点分。

## 14 槽 → L1 的逐条对账(2026-10-18)

| 旧槽 | 构念 | L1 对应 | 判定 |
|---|---|---|---|
| `connective_density` | 连接词密度(文本) | `discourse_marker_rate` | **换**:见下方 ⚠️ 口径差异 |
| `gaze_direction_x_std` | 视线水平偏移的会话内标准差 | `gaze_stability` | 换:同构念,且 L1 做了极差归一化 |
| `au4_frown` | 皱眉(AU4) | `brow_furrow_ratio_vs_baseline` | 换:同构念。⚠️ L1 用 **AU4 < 本人中位数−3·MAD** 的帧占比 —— 眉内距**越小越皱**,方向与本表旧写法相反 |
| `au23_lip_compression` | 唇部压缩(AU23) | `lip_compression_ratio_vs_baseline` | 换:同构念。⚠️ 同样是**越小越压**(唇红越薄) |
| `left_wrist_jitter_world` | 腕部抖动(m/s) | `gesture_activity_per_sec` | 换:同一个 landmark jitter 量,L1 把腕/肘/肩合起来(旧槽只看左腕) |
| `symmetry_score` | 面部对称性 | —— | **删**:L1 无对称性指标 |
| `fluency_score` | 语音流畅度 | —— | **删**:全系统无产出方(见 `research_mapper.NO_PRODUCER`) |
| `pitch_variation` | 语调变化 | `pitch_variability_per_sec` | 换:同构念 |
| `pause_duration` | 平均停顿时长 | `pause_duration_p50` | 换:同构念;中位数比均值稳健 |
| `left_hand_spread` | 手掌张开度 | —— | **删**:`hand_visible_ratio_l` 是**可见性**,不是张开度 |
| `shrug_level` | 肩部放松度(耸肩÷肩宽) | —— | **删**:`posture_stability` 是**稳定性**,不是耸肩程度 |
| `eye_contact` | 眼神接触比例 | —— | **删**:`gaze_aversion_ratio_vs_baseline` 量的是"偏离占比",与本槽的构念关系**未确认**,不拿它顶替 |
| `n_chars` | 回答详尽度(字数) | —— | **删**:`speech_rate` 是**字/秒**(速率),不是总量 |
| `reaction_time` | 反应延迟 | `response_latency_sec` | 换:同构念 |

**结果:14 槽里 8 条换得掉,6 条删**(2026-10-18 T 族落地后,`connective_density`
由"删"改为"换")。其中 `confidence_level` 维度 3 个槽全删 ⟹
该维度整体消失。这不是"算不出来",是**那个维度量的是三个当前管线都不测的东西**。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .l1_indicators import CONFIDENCE

#: 置信度强弱序:**数值越大越可信**,与 `research_mapper.CONF_ORDER` 逐项相同。
#:
#: ⚠️ 这里踩过一个坑:`l1_indicators.CONFIDENCE = ("高","中","低","无")` 的顺序是
#: "从可信到不可信",若直接 `enumerate` 出来就是 **高=0**(越大越不可信),
#: 与 `research_mapper.CONF_ORDER` 的 **高=3** 恰好相反。两个同名表反序、
#: 都用 `max`,含义正好颠倒 —— 第一次写的时候就把"最弱证据"写成了"最强证据",
#: 被 `test_dimension_confidence_is_the_weakest_matched_evidence` 当场抓住。
#: ⟹ 现在两边都是"大 = 可信":**最弱用 `min`,最强用 `max`**。
#: 由 `test_m4_l1_mapper.py::test_confidence_rank_matches_research_mapper` 守着。
CONF_RANK = {"无": 0, "低": 1, "中": 2, "高": 3}

#: 旧槽 → (L1 指标名 或 None, 理由)。
#: `None` 表示**没有同构念的 L1 指标**,槽必须删;理由要能独立读懂。
SLOT_MAP: dict[str, tuple[str | None, str]] = {
    "connective_density": (
        "discourse_marker_rate",
        # ⚠️ 这条**不是严格同构念**,必须让读的人看见:
        # 旧槽数的是"**几个不同**连接词出现过"(presence-once ÷ 百字),
        # 新指标数的是"连接词**总共说了多少次**"(occurrences ÷ 百字)。
        # 差异只体现在"同一个词反复说"的场次上:说 10 遍"然后"旧口径算 1 次、
        # 新口径算 10 次。判"话语结构"时后者才是密度。
        # 仍然映射,是因为两者量的都是"文本里的连接组织",方向一致;
        # 但**不能**把两边历史数字直接比较。
        "换:同族构念但口径不同 —— 旧=不同标记出现过几次,新=标记出现总次数;"
        "重复用同一个连接词时新口径更高,两边历史数字不可直接比"),
    "gaze_direction_x_std": (
        "gaze_stability", "同构念:视线水平偏移的离散度;L1 以极差归一化"),
    "au4_frown": (
        "brow_furrow_ratio_vs_baseline",
        "同构念:AU4(皱眉)。L1 自比基线(本人中位数−3·MAD),构念从\"绝对有多皱\"变\"比他自己平时更皱多少\" ⟹ 名字随之改"),
    "au23_lip_compression": (
        "lip_compression_ratio_vs_baseline",
        "同构念:AU23(唇压)。同上自比基线 ⟹ 名字随之改"),
    "left_wrist_jitter_world": (
        "gesture_activity_per_sec",
        "同一 landmark jitter 量;L1 把腕/肘/肩三关节合起来,旧槽只看左腕"),
    "symmetry_score": (
        None, "L1 没有面部对称性指标"),
    "fluency_score": (
        None, "全系统没有产出方(见 research_mapper.NO_PRODUCER)"),
    "pitch_variation": (
        "pitch_variability_per_sec", "同构念:音高的变化量"),
    "pause_duration": (
        "pause_duration_p50", "同构念:停顿时长;中位数比均值稳健"),
    "left_hand_spread": (
        None, "hand_visible_ratio_l 量的是**可见性**,不是手掌张开度"),
    "shrug_level": (
        None, "posture_stability 量的是**姿态稳定性**,不是耸肩程度"),
    "eye_contact": (
        None, "gaze_aversion_ratio 量的是偏离占比;与本槽构念的关系未确认,不顶替"),
    "n_chars": (
        None, "speech_rate 是**字/秒**(速率),不是回答字数(总量)"),
    "reaction_time": (
        "response_latency_sec", "同构念:问题结束到首次开口"),
}

#: 5 个维度 → 其旧槽(权重原样照抄 `research_mapper`,以便重归一可复核)。
#: ★ 保留下来、但**数值不可与旧引擎直接比较**的槽。
#:
#: 为什么必须单独说:这 6 个槽的名字与旧报告里的槽名相近(或完全同名),
#: 读者会很自然地拿两份报告的数一减 —— 但两边的**统计量不是同一个**。
#: 这不是"精度差一点",是"量的东西不同",相减得到的差没有任何含义。
#:
#: 判据只有两类,都是硬事实、不是判断:
#:   · 绝对量 → 相对本人基线的偏离(自比基线,见 `l1_thresholds`);
#:   · 统计量换了(均值 → 中位数、原始标准差 → 极差归一化、单列 → 多列合计、
#:     presence-once → 出现次数)。
NOT_COMPARABLE: dict[str, str] = {
    "connective_density": "旧 = **不同**标记出现过几次(presence-once);新 = 标记**出现总次数**",
    "gaze_direction_x_std": "旧 = 原始标准差;新 = 以会话内极差归一化",
    "au4_frown": "旧 = AU4 绝对量;新 = 相对**本人基线**(中位数−3·MAD)的偏离",
    "au23_lip_compression": "旧 = AU23 绝对量;新 = 相对**本人基线**的偏离",
    "left_wrist_jitter_world": "旧 = 仅左腕;新 = 腕/肘/肩**六列合计**",
    "pause_duration": "旧 = 停顿**均值**;新 = 停顿**中位数**",
}

#: 保留下来、且与旧引擎**可以**直接比较的槽。列出来是为了让
#: `test_every_kept_slot_is_classified_for_comparability` 能逐条对账 ——
#: 将来新增一个保留槽却忘了归类,那条测试会红。
COMPARABLE: tuple[str, ...] = (
    "pitch_variation",
    # 同构念同单位(秒)。L1 侧目前因缺 `ask_end` 恒不可出(戊-15)⟹
    # 今天不存在"两边都有数"的情形,也就构不成比较陷阱;但仍如实归类。
    "reaction_time",
)


def comparability_notes() -> list[str]:
    """报告里那句"这两份报告的数不能直接比"的逐条说明。"""
    out: list[str] = []
    for slot, why in NOT_COMPARABLE.items():
        l1, _ = SLOT_MAP.get(slot, (None, ""))
        if l1 is None:
            continue
        out.append(f"{slot} → {l1}：{why}")
    return out


DIMENSIONS: dict[str, dict] = {
    "logical_thinking": {
        "name": "话语结构特征",
        "slots": {"connective_density": 0.5714285714285714,
                  "gaze_direction_x_std": 0.2857142857142857,
                  "au4_frown": 0.14285714285714285},
    },
    "stress_resilience": {
        "name": "情境行为稳定性",
        "slots": {"au23_lip_compression": 0.375,
                  "left_wrist_jitter_world": 0.375,
                  "symmetry_score": 0.25},
    },
    "communication_fluency": {
        "name": "沟通表达流畅度",
        "slots": {"fluency_score": 0.5714285714285714,
                  "pitch_variation": 0.2857142857142857,
                  "pause_duration": 0.14285714285714285},
    },
    "confidence_level": {
        "name": "行为表现活跃度",
        "slots": {"left_hand_spread": 0.375, "shrug_level": 0.25, "eye_contact": 0.375},
    },
    "cognitive_efficiency": {
        "name": "作答过程特征",
        "slots": {"n_chars": 0.4, "reaction_time": 0.6},
    },
}


def _clean(value):
    """把 `NaN` 归一成 `None`。

    ★ 这一步**必须**做:`l1_indicators.unavailable()` 把"算不出来"记作
    `value=math.nan`(不是 `None`),而 `math.nan is not None` 是 **True**。
    第一版只判 `is not None`,于是 35 条里那些"不可出"的指标全被数成"过门",
    `n_passed` 从 0 虚报成 7、`matched_indicators` 也跟着虚高。
    实测抓法:端点测试在**完全隔离**的空数据下断言 `n_passed == 0`,当场红。
    """
    if value is None:
        return None
    try:
        return None if math.isnan(float(value)) else float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class DimensionResult:
    """一个维度在 L1 口径下的证据状态。**没有分,没有档。**"""

    key: str
    display_name: str
    slots: list[dict] = field(default_factory=list)      # 幸存的槽
    dropped: list[dict] = field(default_factory=list)    # 删掉的槽
    confidence: str = "无"
    matched_indicators: str = "0/0"

    @property
    def n_slots(self) -> int:
        return len(self.slots) + len(self.dropped)

    @property
    def n_matched(self) -> int:
        return sum(1 for s in self.slots if s.get("value") is not None)

    def to_dict(self) -> dict:
        return {
            "display_name": self.display_name,
            "evidence_chain": [
                {"indicator": s["l1"], "human_name": s["human_name"],
                 "raw_value": s["value"], "n_valid": s["n_valid"],
                 "n_total": s["n_total"], "unit": s["unit"],
                 "confidence": s["confidence"], "weight": s["weight"],
                 "gap": s["gap"]}
                for s in self.slots],
            "evidence_gaps": [
                {"slot": d["slot"], "reason": d["reason"]} for d in self.dropped]
                + [{"slot": s["slot"], "reason": s["gap"]}
                   for s in self.slots if s["value"] is None],
            "confidence": self.confidence,
            "matched_indicators": self.matched_indicators,
        }


def _renormalize(weights: dict[str, float]) -> dict[str, float]:
    """把幸存槽的权重按原比例放大到和为 1。

    与 `research_mapper` 删槽时的做法一致(原 `×1/0.7` 就是这一步的手算结果)。
    """
    total = sum(weights.values())
    if total <= 0:
        return {k: 0.0 for k in weights}
    return {k: v / total for k, v in weights.items()}


def map_l1_to_dimensions(indicators) -> tuple[dict[str, DimensionResult], list[dict]]:
    """L1 指标列表 → 5 个维度。返回 `(维度, 整体消失的维度)`。

    `indicators` 可以是 `L1Indicator` 对象列表,也可以是 `build_quality_report`
    产出的 dict 列表 —— 两种都收,免得调用方为了传参多绕一层。
    """
    by_name: dict[str, dict] = {}
    for ind in indicators:
        if isinstance(ind, dict):
            by_name[ind["name"]] = {**ind, "value": _clean(ind.get("value"))}
        else:
            by_name[ind.name] = {
                "name": ind.name, "value": _clean(ind.value),
                "n_valid": ind.n_valid,
                "n_total": ind.n_total, "confidence": ind.confidence,
                "unit": getattr(ind, "unit", ""), "gap": ind.gap,
                "human_name": getattr(ind, "formula", ind.name)}

    out: dict[str, DimensionResult] = {}
    vanished: list[dict] = []

    for key, spec in DIMENSIONS.items():
        res = DimensionResult(key=key, display_name=spec["name"])
        survivors: dict[str, float] = {}
        for slot, weight in spec["slots"].items():
            l1, reason = SLOT_MAP.get(slot, (None, "未登记的槽"))
            if l1 is None:
                res.dropped.append({"slot": slot, "weight": weight, "reason": reason})
                continue
            survivors[slot] = weight
            got = by_name.get(l1)
            res.slots.append({
                "slot": slot, "l1": l1, "weight": weight,
                "human_name": (got or {}).get("human_name", l1),
                "value": None if got is None else got.get("value"),
                "n_valid": 0 if got is None else got.get("n_valid", 0),
                "n_total": 0 if got is None else got.get("n_total", 0),
                "unit": "" if got is None else got.get("unit", ""),
                "confidence": "无" if got is None else got.get("confidence", "无"),
                "gap": ("L1 没有产出这条指标" if got is None
                        else (got.get("gap") or "该指标本场无值")),
            })

        if not res.slots:
            # 全部槽都没有同构念的 L1 ⟹ **维度整体消失**,不留在 payload 里充数。
            vanished.append({
                "key": key, "display_name": spec["name"],
                # ★ 槽数**报数**,不要让调用方去数理由串里的分隔符 ——
                #   理由本身就可能含 `;`(如"(自比基线;眉内距越小越皱)"),
                #   按 `;` 拆会数出 15 个槽来(实际 14 个)。
                "n_slots": len(res.dropped),
                "dropped": [{"slot": d["slot"], "reason": d["reason"]}
                            for d in res.dropped],
                "reason": ";".join(f"{d['slot']}:{d['reason']}" for d in res.dropped),
            })
            continue

        norm = _renormalize(survivors)
        for s in res.slots:
            s["weight"] = norm[s["slot"]]

        matched = [s for s in res.slots if s["value"] is not None]
        res.matched_indicators = f"{len(matched)}/{res.n_slots}"
        if not matched:
            res.confidence = "无"
        else:
            # ★ 最弱证据决定维度陈述的强度 —— 不是"填了几个槽"。
            # `CONF_RANK` 里「高」=3 ⟹ **最弱 = min**。
            res.confidence = min((s["confidence"] for s in matched),
                                 key=lambda c: CONF_RANK.get(c, 0))
        out[key] = res

    return out, vanished
