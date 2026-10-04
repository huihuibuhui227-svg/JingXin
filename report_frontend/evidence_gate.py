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
    # ★ 2026-10-03 把 `focus_score` 的理由**改准**。原写「3 个硬编码取值 {0.3,0.5,0.8},
    #   2b11 会话内恒 0.3」—— 现象对,但**归因太浅**:读的人会以为只是"分箱粗"或
    #   "那一场碰巧",于是去调阈值,而那正是本仓禁止的"拿分布凑参数"。
    #   实测四场高质量素材(f98d/2e84/8102/5b10):
    #     · 它可被 (`head_yaw`, `blink_rate_per_min`) **100% 精确重构** ⟹ **零额外信息**,
    #       是 `head_yaw` 的 3 值分箱,不是一次测量;
    #     · **99% 恒 0.3**(f98d 314/314、5b10 343/343、2e84 962/971、8102 6597/6619);
    #     · 根因:`head_yaw` 有 **~+0.35 的常量偏移**(四场中位 +0.287~+0.371),而分箱
    #       阈值是 0.03/0.08 ⟹ **永远**落进"偏头"那档,0.8 那档**结构性不可达**;
    #     · 更深一层:`head_yaw` 自己的登记写着「**本列不实现、不近似**」(真 3D 头姿
    #       需相机内参,而"本项目没有任何内参来源")⟹ 这条**不该靠调阈值救**
    #       (那是拟合分布),也不该靠 2D 近似救(登记明令禁止)。
    "focus_score": Quarantine(
        "`head_yaw` 的 3 值分箱(0.3/0.5/0.8),不是测量 —— 实测四场可被 "
        "(`head_yaw`, `blink_rate_per_min`) 100% 精确重构(零额外信息),且 99% 恒 0.3:"
        "`head_yaw` 有 ~+0.35 常量偏移而阈值是 0.03/0.08,0.8 那档结构性不可达。"
        "而 `head_yaw` 自身登记为「不实现、不近似」(需相机内参,本项目没有)",
        "等 `head_yaw` 有真 3D 头姿(需相机内参)后改用连续 yaw;"
        "或按兄弟条目 `tension_score`/`tension_level` 一并删列(M3.2)"),
    # ★ 2026-10-03 把 `tension_score` 的理由**补实**。原写「伪合成:硬编码权重
    #   (.25/.25/.2/.15/.15)与硬编码情绪加成;5 个分量里 eye_closure 近乎恒 0」——
    #   方向对,但没给**可核的数**,也没说清它为什么**在 L1 层修不了**(读的人会去调权重)。
    #   实测四场高质量素材(f98d/2e84/8102/5b10):
    #     · `tension_engine.py` 共 **10 个无依据参数**:权重 .25/.25/.2/.15/.15、
    #       放大系数 ×1.3/×1.3/×1.5/×0.8、`eye_closed_sec / 2.0` 的那个 2 秒、
    #       `emotional_boost × 0.4`,以及**无 temporal_stats 时的硬编码默认
    #       `expression_instability = 0.3`** ⟹ **全零输入也拿 0.045,不是 0**(有地板);
    #     · **`eye_closure` 是死权重**:它占 **0.2**,而 `eye_closed_sec` 在真实数据里
    #       **97%~99.7% 的帧为 0**(f98d 9/314、2e84 109/971、8102 751/6619、5b10 19/343),
    #       因为它的判定门 `avg_ear < 0.18` 几乎不触发(该列本身也在封停中)
    #       ⟹ `eye_closed_sec = 0` 时可达上限只有 **0.8**,那 0.2 结构性用不上;
    #     · **约 72% 的量来自没落盘的派生状态**:已落盘输入(au4/au23/eye_closed_sec)
    #       只解释中位 **0.117**,而 tension 中位 0.40~0.55、残差中位 **+0.27~+0.43**
    #       —— 那部分来自**滚动窗口** `*_volatility`(CSV 里根本没有 volatility 列)。
    #       重跑管线能逐格复现它,但**读 CSV 的人看不出这个数是怎么来的**。
    #     ⟹ **不可修**:权重没有物理或几何依据,要落地只能靠标定集,而 M5 的两个前置
    #       (许可禁止在 RecruitView 分布上拟合、自评量表要过伦理审查)都卡着
    #       ⟹ 按登记计划 **M3.2 删列**。
    #     ⚠️ 别为了让档位"有变化"去调 0.3/0.6 或那几个权重 —— 那是拿分布凑参数。
    "tension_score": Quarantine(
        "无依据的手工加权合成:`tension_engine.py` 里 **10 个参数全无出处**"
        "(权重 .25/.25/.2/.15/.15、放大 ×1.3/×1.5/×0.8、`eye_closed_sec/2.0`、×0.4、"
        "以及无 temporal_stats 时的硬编码默认 `expression_instability=0.3`);"
        "其中 **`eye_closure` 占 0.2 却是死权重**(`eye_closed_sec` 真实数据里 97%~99.7% 为 0,"
        "`eye_closed_sec=0` 时可达上限只有 0.8);且 **~72% 的量来自没落盘的滚动窗口 "
        "`*_volatility`**(已落盘输入只解释中位 0.117,残差中位 +0.27~+0.43)",
        "权重没有物理依据,只能靠标定集 ⟹ 按登记计划 **M3.2 删列**"),
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
    # ★ 2026-10-03 **更正理由(原话写错了取数路径)**。原理由:「au7 + avg_ear 恒等 1.0,
    #   与 au4 互补」,unblock「M3」。那句话描述的是**源列** `au7_eye_squeeze` 的退化 ——
    #   实测它确实成立(四场逐帧 `au7_eye_squeeze ≡ 1 − avg_ear`,max|diff| = 1.1e-16),
    #   但**不是这个槽的取数路径**:槽的键是 `face_micro_exp_micro_exp_au_name_au7_freq`
    #   (`micro_exp_au_name` 字符串列的 one-hot)。f98d 上实测
    #   `_fuzzy_match("au7_freq", …) == (None, None)` —— 它**根本不走** `au7_eye_squeeze`。
    #   于是"等 M3"是句空话:该 one-hot 键在**全仓 0 产出**。真实现状是源列
    #   `micro_exp_au_name` **51 场 22277 行只有 6 行有值**(场 `20260929_205448_8b75`),
    #   取值只有 `au4_frown` / `au7_eye_squeeze` ⟹ **`au7_freq` 这个标签从没出现过**;
    #   而它原本要量的 AU7 强度那条路又断在退化上(见上)。两条源列都标着 M3.2 删列。
    #   槽已于 2026-10-03 删除(它从不进门 ⟹ 删掉不改任何分数)。
    "au7_freq": Quarantine(
        "全仓 0 产出:取数键来自 `micro_exp_au_name` 的 one-hot,而该列 51 场 22277 行"
        "只有 6 行有值,取值只有 au4_frown / au7_eye_squeeze ——「au7_freq」这个标签"
        "从没出现过;另一条路(AU7 强度列)逐帧 ≡ 1 − avg_ear(四场 max|diff|=1.1e-16)"
        "⟹ 零额外信息。两条源列都已标 M3.2 删列",
        "需要一个真会产出这个标签的来源 —— 现无;若将来 AU7 有独立产出方,连同槽一起重新登记"),
    "speech_ratio": Quarantine("自指阈值,87.9% 恰为 1.0", "M3 真 VAD"),
    # ✅ 2026-10-03 **解封 `eye_contact`**(本组余 5 条)。它自己写的 unblock
    #    (「M3 改 feature_engine 的 eye_contact 口径」)已达成,而且这次是
    #    **连消费方一起验过**的 —— 当年栽的正是"只核了生产方那一半":
    #      · 消费方改用它真正的原点 **0**(iris 四列是**眼内相对坐标**,不是图像坐标);
    #      · 只取水平分量(x 静息位 ≈ 0;y 静息位 −0.15~−0.20 是解剖偏移,无可标定零点);
    #      · 剔除虹膜跑出睑裂的坏帧(物理量程 (1−D/W)/2 ≈ 0.305);
    #      · 补上 G2 要的 `_std` 与 G3 要的 `_sample_size` —— 当年封停理由点名过
    #        「无 `_std` 伴随列,G2 在此退回 fail-open」,那个洞现在堵上了。
    #    逐条证据(四场高质量素材实测)见 `tests/test_report_layer.py` 的 `_RELEASED`。
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
    # ★ 2026-10-03 **更正理由与 unblock(原话把病因写成了单位)**。
    #   原理由说「60 秒滚动窗,量的不是 per-sec 眨眼率 … 改成按真实帧间隔算的 per-sec 率」。
    #   实测已不是这样:`_update_blink_state` 现在**就是**按真实 `timestamp` 算的
    #   60 秒窗(`video_pipeline.py`,`test_video_pipeline_timebase.py` 钉着)——
    #   60 秒窗内的次数**本来就是** per-minute 率,换单位治不了。
    #   真正的病因是**采样率**,而且有两处,方向相反:
    #     · **漏检**:帧间隔实测中位 1.000~1.002 s(四场),而眨眼时长 100~400 ms
    #       ⟹ 绝大多数眨眼根本没被采到;
    #     · **计错单位**:0.3 s 去抖在 1 fps 下**恒不触发**(相邻闭眼帧间隔 ≈1 s > 0.3 s)
    #       ⟹ 每一次"闭眼帧"都记一次 ⟹ `blink_count` 数的是「最近 60 秒里判定闭眼的
    #       **帧数**」,上限就是 60 s 内的帧数(实测 max **57**)。
    #   另有**误检**:实测 duty 4.4%~14.2%(人眼真实闭眼占比约 5% 量级),
    #   `is_blink` 连续段里出现 14/18/19/21/28 帧(≈14~28 秒)的长段。
    #   槽已于 2026-10-03 删除;源列 `blink_rate_per_min` 另已标 M3.2 删列。
    "blink_rate": Quarantine(
        "病因是**采样率**不是单位:帧间隔实测中位 1.000~1.002 s(四场)而眨眼时长 "
        "100~400 ms ⟹ 漏检结构性;0.3 s 去抖在 1 fps 下恒不触发 ⟹ 计的是"
        "「最近 60 秒里判定闭眼的**帧数**」(上限=60 s 内帧数,实测 max 57)而不是眨眼次数。"
        "另有误检:实测 duty 4.4%~14.2%,is_blink 连续段出现 14/18/19/21/28 帧(≈14~28 秒)的长段",
        "提高面部采集率(现约 1 fps)—— 那是采集侧决策,换单位/改列名都治不了;"
        "源列 blink_rate_per_min 另已标 M3.2 删列"),
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
