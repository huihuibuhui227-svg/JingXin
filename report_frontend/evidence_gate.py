# report_frontend/evidence_gate.py
"""证据门:任何指标要进打分,必须四关全过。

设计依据:docs/superpowers/specs/2026-09-22-jingxin-report-layer-evidence-gate-design.md §5.1/§5.2

四关:
  G1 有值      非 NaN
  G2 非常量    会话内方差 > 0
  G3 样本够    n_valid >= 登记阈值
  G4 非代理    不在封停名单

本模块不依赖 pandas —— 纯函数,可独立测试。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence

Confidence = Literal["高", "中", "低", "无"]


@dataclass(frozen=True)
class Check:
    ok: bool
    gate_name: str = ""
    reason: str = ""


@dataclass(frozen=True)
class Quarantine:
    reason: str
    unblock: str
    permanent: bool = False
    # ★ E7(2026-09-27 裁定,选项 c):命中了**这些**子串的键不算命中本条。
    #   存在的唯一理由是「一个键名是另一个键名前缀」:`head_tilt` 是 `head_tilt_angle`
    #   的子串,而后者是 maturity=A 的**保留列**(实测 `is_quarantined` 的子串匹配
    #   分不开两者,天真改名会把保留列一起封掉,且**不报错**)。
    #   机制**只**做排除,不放宽匹配 —— 全表仍是「子串 + 最长匹配」一套语义,
    #   豁免就地挂在本条上,不引入全局名单。
    not_substrings: tuple[str, ...] = ()


# 封停名单(spec §5.2)。key 为**子串**,按最长匹配生效。
# 解封条件写在 unblock 里,便于 M3 修好后逐条解封。
#
# ★ 2026-09-27 对账(本名单对三场正式素材的**当前**产出一一量过,三类处置):
#
#   ① **解封 5 条** —— 它们**自己写的那个理由**已不成立(判据是 G4 自己的口径,
#      不是"我们觉得这个量挺有用")。逐条证据写在 `tests/test_report_layer.py`
#      的 `_RELEASED` 上方:`symmetry_score` / `jitter` / `fist_status` /
#      `au4_freq` / `pitch_variation`。
#   ② **改准 5 条**(**不是**删)—— `hand_score` / `shoulder_score` / `arm_score` /
#      `head_score` / `torso_score`:产出它们的列已从 `GestureLogger` 的**落盘契约**里
#      删掉(B3 删列)。⚠️ 2026-09-27 一度按"已无对象可封"把这 5 条**删了**,实测**删不得**,
#      它们有两个**当前就存在**的对象:
#        · **活路径**:`data_loader.get_live_data` 从手势服务的 `/summary` 端点拼出的那一行
#          仍写着 `hand_score` / `left_hand_score` / `right_hand_score` / `shoulder_score` /
#          `left_arm_score` / `right_arm_score`(实测:经 `feature_engine` 变成
#          `gesture_hand_score_mean` 等 30 个键,全部由这几条封停拦下;删掉条目它们当场进分);
#        · **历史日志**:三场正式素材(2026-09-26)的 CSV 里那 7 个 `*_score` 列**还在**。
#      教训:**封停对象是"列",不是"某个产出方"**。删掉一个产出方不等于这条封停没有对象;
#      判「有没有对象」要**逐个产出方数**(落盘 logger、活路径 HTTP、历史文件),
#      只数一个就会把仍在生效的封停当成死条目删掉。
#      ⟹ 保留条目,只把措辞改准(说清它现在管的是活路径与历史日志)。
#   ③ **改写 6 条的措辞**(理由仍成立,但原文描述的是**现在的代码里不存在的机制**,
#      或量错了对象):`tension_score` / `gaze_deviation` / `eye_contact` /
#      `blink_rate` / `eye_closed_sec` / `pause_duration`。
#      ⚠️ 其中 `eye_contact` 是**改了理由但仍须封停**的那一条:它的 unblock
#      (「M3 改眼内相对坐标」)在生产方**已达成**,而**消费方**(`feature_engine` 的
#      `_extract_face_features`)没跟着改 ⟹ 实测 `face_eye_contact_ratio` 恒 0.0。
#      生产方修好不等于这个量能进分 —— 见下面那条理由。
QUARANTINE: dict[str, Quarantine] = {
    # ---- 等 M3 ----
    "focus_score": Quarantine("3 个硬编码取值 {0.3,0.5,0.8},2b11 会话内恒 0.3", "M3"),
    "tension_score": Quarantine(
        "伪合成:硬编码权重(.25/.25/.2/.15/.15)与硬编码情绪加成;5 个分量里 "
        "eye_closure 近乎恒 0(eye_closed_sec 恒 0,2b11/1592 std=0)", "M3"),
    "tension_level": Quarantine("字符串分箱", "M3"),
    "tension_sources_": Quarantine("au4/au23 副本或恒定量", "M3"),
    "gaze_direction_y": Quarantine("恒负,解剖常量", "M3"),
    "gaze_deviation": Quarantine(
        "与 gaze_direction_y 同族冗余(y 分量主导:两列会话均值相对差 "
        "4.5%/8.5%/30.4%,2026-09-27 三场重放实测)", "M3"),
    # ↓ 这 5 条**不是**"无对象可封":`GestureLogger` 的落盘契约里删了这些列(B3),
    #   但活路径(`data_loader.get_live_data` 读手势 `/summary`)与历史日志里都还在产
    #   ⟹ 封停照旧生效,只把措辞改准。理由全文见文件上方 ②。
    "hand_score": Quarantine(
        "可由同 block jitter 精确重构(零额外信息);手势 logger 的落盘契约已删列(B3),"
        "但活路径 `/summary` 与历史日志仍在产 ⟹ 本条仍有效", "改由 jitter 直接测同一构念"),
    "shoulder_score": Quarantine(
        "可由同 block jitter 精确重构(零额外信息);同 hand_score,落盘已删列而活路径仍在产",
        "改由 shrug_level 直接测同一构念"),
    "arm_score": Quarantine(
        "可由同 block jitter 精确重构(零额外信息);同 hand_score,落盘已删列而活路径仍在产",
        "改由 jitter 直接测同一构念"),
    "head_score": Quarantine(
        "可由同 block jitter 精确重构(零额外信息);同 hand_score,落盘已删列而活路径仍在产",
        "改由 jitter 直接测同一构念"),
    "torso_score": Quarantine(
        "可由同 block jitter 精确重构(零额外信息);同 hand_score,落盘已删列而活路径仍在产",
        "改由 jitter 直接测同一构念"),
    # ↓ 这 6 条来自 spec §5.3 的**槽位级**处置,§5.2 的列级表没覆盖它们。
    #   仅靠 §5.2 会让这些指标绕过 G4 直接进打分(实测其中 3 个真的出了分)。
    #   已由 Task 3 的修复轮补入(Task 2 完成时遗漏)。
    #   2026-09-27 对账后 `au4_freq`(解封)与 `fluency_score`(改挂 NO_PRODUCER)
    #   退出本组,余 6 条。
    "gaze_stability": Quarantine("由被封停的 gaze_deviation 派生", "M3 修 gaze_direction_x"),
    "au7_freq": Quarantine("au7 + avg_ear 恒等 1.0,与 au4 互补", "M3"),
    "speech_ratio": Quarantine("自指阈值,87.9% 恰为 1.0", "M3 真 VAD"),
    "eye_contact": Quarantine(
        "**消费方**未随生产方改:iris 四列已改为眼内相对坐标,而 "
        "feature_engine 的 eye_contact 口径仍是「到画面中心 (0.5,0.5) 的距离」"
        "⟹ 实测 face_eye_contact_ratio 恒 0.0(三场),且无 _std 伴随列 "
        "(G2 在此退回 fail-open,挡不住)", "M3 改 feature_engine 的 eye_contact 口径"),
    # 永久封停:unblock 是给人看的显示文本,逻辑判断一律用 permanent 字段。
    # 不要用 unblock == "永不" 反推 —— 字符串是显示层,不是逻辑层。
    # ★ 这两条此前写成 `upper_body_head_tilt` / `shoulder_is_calibrated` —— 那是
    #   **旧代码里的名字**,而报告层拿到的是扁平化后的 `<模态>_<基名>_<统计量>`
    #   (`gesture_head_tilt_mean` / `gesture_is_calibrated_mean`)⟹ 子串匹配恒不命中,
    #   两条永久封停**形同虚设**(E7,2026-09-27 实测复现)。键名已对齐真键。
    #   `head_tilt` 必须带豁免:它是 `head_tilt_angle` 的子串,而后者是
    #   `maturity=A` / `status=implemented` 的保留列(见 `Quarantine.not_substrings`)。
    "head_tilt": Quarantine("参考系错位 180 度,分支命中率 0%", "永不", permanent=True,
                            not_substrings=("head_tilt_angle",)),
    "is_calibrated": Quarantine("纯时长变量,控时长后相关 0.014", "永不", permanent=True),
    "overall_score": Quarantine("属性不存在,getattr 走默认值", "永不", permanent=True),
    "emotion_state": Quarantine("属性不存在,恒 neutral", "永不", permanent=True),
    "emotion_": Quarantine("7 个分量结构性恒 0,单形归一化互竞", "永不", permanent=True),
    "dominant_emotion": Quarantine("argmax,非置信度", "永不", permanent=True),
    "micro_exp_onset_frame": Quarantine("环形缓冲下标,恒在 5~13", "永不", permanent=True),
    "fluency_proxy": Quarantine("与 fluency_score 同自由度,数了两遍", "永不", permanent=True),
    # ---- 等对应里程碑 ----
    "blink_rate": Quarantine(
        "60 秒滚动窗,量的不是 per-sec 眨眼率:三场帧间隔 1.06–1.07 s,"
        "is_blink 只命中 1/271·3/477·11/479 帧 ⟹ 计数由采样格决定,不是被试的眨眼频率",
        "随列处置(删列)或改成按真实帧间隔算的 per-sec 率"),
    "eye_closed_sec": Quarantine(
        "与 is_blink 冗余:判定门 avg_ear < 0.18,而 avg_ear 均值 0.42 ⟹ "
        "2b11/1592 整个会话恒 0、caf0 只有 3 个取值", "随列处置(删列)或并入眨眼时长"),
    "pause_duration": Quarantine(
        "自指能量门限:停顿 = 连续 rms ≤ mean(rms)×0.3 的帧段,门限值无出处 "
        "⟹ 只能来自标定集(M3.1 已裁定跳过)。尾部收尾已由 M3.0 补上",
        "标定集给出能量门限后登记进 evidence_thresholds.json"),
    "energy_mean": Quarantine("设备增益/距离代理,跨会话不可比", "M3 会话内归一"),
    "energy_level": Quarantine("同上", "M3 会话内归一"),
    "is_valid": Quarantine("恒 1 且下游从未使用", "M3 改真掩码"),
}

_THRESHOLD_FILE = Path(__file__).resolve().parent / "evidence_thresholds.json"

# 折算因子的依据类别。`legacy_arbitrary` 是**允许的取值** —— 历史遗留的标尺可以存在,
# 但必须显式声明它没有依据;留空会让"这条有没有依据"变成不可回答的问题(spec §5.1)。
_ALLOWED_BASIS_KINDS = {"definitional", "physical", "legacy_arbitrary"}


def _validate_scale_factors(data: dict) -> None:
    factors = data.get("scale_factors")
    if not isinstance(factors, dict) or not factors:
        raise ValueError(
            "evidence_thresholds.json 必须带非空的 scale_factors —— "
            "折算因子不得写成代码里的裸常量(spec §5.1)"
        )
    default = data.get("_default_scale_factor")
    if not isinstance(default, dict):
        raise ValueError(
            "evidence_thresholds.json 必须带 _default_scale_factor —— "
            "未登记族的默认因子同样是临时值(spec §5.1)"
        )
    for name, spec in list(factors.items()) + [("<default>", default)]:
        if not spec.get("basis"):
            raise ValueError(f"scale_factors[{name}] 未声明 basis(spec §5.1)")
        if spec.get("basis_kind") not in _ALLOWED_BASIS_KINDS:
            raise ValueError(
                f"scale_factors[{name}] 的 basis_kind 非法:{spec.get('basis_kind')!r}"
                f"(应为 {sorted(_ALLOWED_BASIS_KINDS)} 之一;没有依据就写 legacy_arbitrary)"
            )


def load_thresholds(path: Optional[Path] = None) -> dict:
    with (Path(path) if path is not None else _THRESHOLD_FILE).open(encoding="utf-8") as f:
        data = json.load(f)
    if data.get("_provisional") is not True or not data.get("_version"):
        raise ValueError(
            "evidence_thresholds.json 必须带 _provisional: true 与 _version —— "
            "临时阈值必须可被识别为临时(spec §5.1)"
        )
    if "_default_n_valid" not in data:
        raise ValueError(
            "evidence_thresholds.json 必须带 _default_n_valid —— "
            "未登记指标的默认阈值也属于临时值,不得写成代码里的裸常量(spec §5.1)"
        )
    _validate_scale_factors(data)
    return data


def _threshold_for(key: str) -> int:
    """按**最长**子串匹配阈值,与 is_quarantined 用同一套规则。

    不能用"注册表里第一个匹配" —— 阈值表里 "au" 早于 "pause",而 "au" 是
    "pause" 的子串(p-au-se),会让所有 pause_* 键静默拿到 au 的阈值 30
    而不是自己登记的 2。同模块内两套匹配规则是缺陷。
    """
    data = load_thresholds()
    thresholds = data["thresholds"]
    k = key.lower()
    matched = [name for name in thresholds if name in k]
    if not matched:
        # 默认值也来自 JSON —— 不得写成裸常量(spec §5.1)
        return int(data["_default_n_valid"])
    return thresholds[max(matched, key=len)]


def scale_factor_for(key: str) -> dict:
    """按**最长**子串匹配折算因子族(与阈值、封停名单同一套匹配规则)。

    长度相同者按登记顺序取先登记者 —— 只有一个键会撞名:`pause_duration` 同时含
    `ratio` 与 `pause`(实测 `'ratio' in 'pause_duration'` 为真),legacy 的 if 链里
    恒等族排在 pause 之前,故它走恒等族,0.5–3.0s 的分箱从未执行。这一归因由
    `tests/test_report_layer.py::test_pause_duration_is_shadowed_by_the_ratio_family`
    钉住 —— 想改归因就动登记表顺序,届时测试会红。

    原实现在 `research_mapper._dynamic_normalize` 里用 if/elif 顺序决定族,
    且因子是裸常量 —— spec §5.1 明禁。现按族登记在 evidence_thresholds.json,
    本函数只负责取出来;一个数值都不留在代码里。
    """
    data = load_thresholds()
    factors = data["scale_factors"]
    k = key.lower()
    matched = [name for name in factors if name in k]
    if not matched:
        return data["_default_scale_factor"]
    return factors[max(matched, key=len)]


def normalize_value(val: float, key: str) -> float:
    """把原始值按**登记的**折算因子折到 0–1。

    本函数体内不得出现任何可调数值(spec §5.1):满量程、上下界、分箱边界全部来自
    evidence_thresholds.json。这里的 0.0 / 1.0 是 0–1 值域本身的边界,不是标尺参数。
    """
    val = float(val)
    spec = scale_factor_for(key)
    kind = spec["kind"]
    if kind == "identity":
        out = val
    elif kind == "identity_or_percent":
        out = val / spec["percent_divisor"] if val > 1 else val
    elif kind == "full_scale":
        out = val / spec["full_scale"]
    elif kind == "pause_bins":
        if spec["normal_lo"] <= val <= spec["normal_hi"]:
            out = 0.0
        elif val < spec["normal_lo"]:
            out = spec["below_value"]
        else:
            out = (val - spec["normal_hi"]) / spec["above_span"]
    else:
        raise ValueError(
            f"未登记的折算类型:{kind!r} —— 代码与 evidence_thresholds.json 不同步"
        )
    return min(1.0, max(0.0, out))


def is_quarantined(key: str) -> Optional[Quarantine]:
    """按最长子串匹配封停名单;条目可用 `not_substrings` 声明「这些键不算命中本条」。

    匹配**只有一套语义**(子串 + 最长优先,与 `_threshold_for` / `scale_factor_for`
    同一套 —— 同模块内两套匹配规则是本仓已定性的缺陷)。`not_substrings` 是在这套
    语义**之内**的一个排除位,不是第二套规则:被排除的条目从候选里去掉,再取最长的
    那个;候选清空则返回 `None`(放行),不会退化成"随便挑一个"。
    """
    k = key.lower()
    matched = [s for s in QUARANTINE
               if s in k and not any(x in k for x in QUARANTINE[s].not_substrings)]
    if not matched:
        return None
    return QUARANTINE[max(matched, key=len)]


def check_g1_has_value(value) -> Check:
    if value is None:
        return Check(False, "G1", "无值")
    try:
        if math.isnan(float(value)):
            return Check(False, "G1", "NaN")
    except (TypeError, ValueError):
        return Check(False, "G1", f"非数值:{type(value).__name__}")
    return Check(True)


def check_g2_not_constant(values: Optional[Sequence[float]]) -> Check:
    if not values:
        return Check(True)  # 无序列信息时不在本关拦截
    finite = [v for v in values if v is not None and not _isnan(v)]
    if len(finite) < 2:
        return Check(False, "G2", "有效取值不足 2 个,无法排除常量")
    if max(finite) == min(finite):
        return Check(False, "G2", f"会话内恒为 {finite[0]}")
    return Check(True)


def check_g2_from_std(std) -> Check:
    """用伴随的 _std 判定常量性。

    报告层拿到的只有标量(feature_engine 已把序列聚合掉了),
    但 feature_engine 对每个数值列同时产出 _std,足以判定"会话内是否恒定"。
    std 缺失时不在本关拦截。
    """
    if std is None:
        return Check(True)
    try:
        if float(std) == 0.0:
            return Check(False, "G2", "会话内标准差为 0(常量)")
    except (TypeError, ValueError):
        return Check(True)
    return Check(True)


def check_g3_enough_samples(n_valid: int, min_required: int) -> Check:
    if n_valid < min_required:
        return Check(False, "G3", f"有效样本 {n_valid} < 阈值 {min_required}")
    return Check(True)


def check_g4_not_proxy(key: str) -> Check:
    q = is_quarantined(key)
    if q:
        return Check(False, "G4", f"已封停:{q.reason}(解封:{q.unblock})")
    return Check(True)


def gate(key: str, value, n_valid: int,
         values: Optional[Sequence[float]] = None,
         std=None) -> Check:
    """依次跑 G1 -> G4,返回第一个失败;全过返回 Check(ok=True)。

    values 与 std 二选一:有序列用 values,只有标量用 std。
    """
    g2 = check_g2_not_constant(values) if values is not None else check_g2_from_std(std)
    for check in (
        check_g1_has_value(value),
        g2,
        check_g3_enough_samples(n_valid, _threshold_for(key)),
        check_g4_not_proxy(key),
    ):
        if not check.ok:
            return check
    return Check(True)


_USER_MESSAGES: dict[str, str] = {
    "G1": "未采集到对应数据",
    "G2": "本次会话内无变化",
    "G3": "有效样本不足",
    "G4": "该指标本轮停用",
}


def user_message(check: Check) -> str:
    """面向用户的证据缺口说明。

    ⚠️ 报告层只准用这个,**不得直接渲染 `check.reason`** —— reason 面向维护者,
    含封停理由等内部信息(如"可由同 block jitter 精确重构")。spec §5.6 对
    quarantine 文本的豁免只覆盖"留在代码里不进输出";一旦渲染进报告就不再是
    内部文本,而报告必须中性、可辩护。
    """
    return _USER_MESSAGES.get(check.gate_name, "证据不足")


def confidence_from(n_ok: int, n_slots: int) -> Confidence:
    """置信度三值化。'高' 在本轮不可达(spec §5.1)。"""
    if n_ok <= 0:
        return "无"
    if n_slots > 0 and n_ok * 2 < n_slots:
        return "低"
    return "中"


def _isnan(v) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True
