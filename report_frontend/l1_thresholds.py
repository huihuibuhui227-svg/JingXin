# report_frontend/l1_thresholds.py
"""**τ 从哪来** —— 那 9 条"超阈占比"型 F 指标的阈值解析。

## 为什么需要单独一层(2026-10-18 实测之后)

`l1_face` 原先对这 9 条一律报 `GAP_UNCALIBRATED`,理由是"τ 只能来自标定集(M5)"。
量了全语料 22277 行 / 41 场之后,这个理由**不成立** —— 卡住的不是"没有标定集",
是**跨人基线差**:

```
列                     场内 p10→p90   场间中位数范围      场间/场内
au4_frown              0.290→0.328   0.290–0.337 (1.16×)   同量级
au23_lip_compression   0.038→0.065   0.035–0.063 (1.80×)   同量级
gaze_direction_x      −0.120→0.076   −0.109–+0.922         远大于
head_yaw               0.322→0.562   −0.833–+0.389(符号会翻) 远大于
au12_smile             0.000→0.393   0.000–0.612            真行为差异
```

前四列的**场间基线差 ≥ 场内动态范围** ⟹ 一个统一常量 τ 量到的主要是
"这个人的脸长什么样 / 机位偏哪边",不是"他在做什么"。所以它们的 τ 必须**自比基线**。

`au12_smile` 不同:它的生产方**已经**除掉了本人基线(口宽 ÷ 全程 p10,见
`emotion_engine` 的登记),场间差异是**真的行为差异**(有人爱笑有人不笑)⟹
在上面再加一个**声明常量**才是对的。

## 方向(这里修掉两个真 bug)

`l1_face` 原先三条的公式方向是**反的**:

* `au4_frown` = `dist(lm[107], lm[336]) ÷ 眼距` —— **眉内距 ÷ 眼距,越小越皱**。
  原公式写 `au4 > τ`,量的是"眉展得多开"。`research_mapper` 里早有记录:
  「眉内距**越小**眉越皱(越糟)」。
* `au23_lip_compression` = `唇红厚度 ÷ 面部轮廓高` —— **唇红越薄越压**,方向同上。
  `emotion_engine` 的登记里写着旧公式是 `1 − 当前唇厚/静息唇厚`(带 `1−`),
  换口径时把 `1−` 去掉了,**方向随之翻转,而阈值没重登记** ——
  那条 `au23 > 0.2` 现在是"结构性不可达"(三场 max 0.085)。

⟹ 本模块把**方向**当成阈值的一部分显式声明,不再让公式字符串自己记。

## 规矩

1. **策略随数据性质定,不随方便定**。判据:场间基线差 vs 场内动态范围。
2. **自比基线的指标必须改名**(`*_vs_baseline`)—— 构念从"绝对有多皱"变成
   "比他自己平时更皱多少",名字不改就是偷偷换构念。
3. **声明常量要写清出处**,并且**声明它没被标定**。本模块里的 0.263 是全语料
   41 场的 p75,**不是**标定值。
4. **拿不到数据就不给 τ**(`None`),让上游照旧报 `uncalibrated` —— 不猜。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: 自比基线时 MAD 的倍数。`MAD × 1.4826 ≈ σ`(正态下),所以 `3·MAD ≈ 2σ` ——
#: 这是稳健离群点的常用约定(与 3σ 同量级但不被极端值拖动)。
SELF_K = 3.0

#: `au12_smile` 的声明常量。全语料 41 场 16318 帧的 p75。
#: ⚠️ **这不是标定值**:没有标定集(M5 卡在许可 + 伦理审查)。它只是一个
#: **写明的、可复核的、非拟合的**分界点;换语料就该重算并重写这一行。
AU12_SMILE_TAU = 0.263

STRATEGY_DECLARED = "declared"
STRATEGY_SELF = "self_baseline"


@dataclass(frozen=True)
class Tau:
    """一条列上解析出来的阈值。`None` 值表示解析不出来。"""

    column: str
    value: float | None
    direction: str            # ">" 或 "<" —— **构念方向,不是数值大小方向**
    strategy: str
    transform: str | None     # "abs" 表示先取绝对值
    provenance: str

    def mask(self, values) -> np.ndarray:
        """哪些帧**处于该构念状态**。`NaN` 一律为 `False`(不拿缺失当阴性)。

        入参既可以是 DataFrame 的列(`Series`),也可以是逐帧信号(`ndarray`)——
        手势的"运动段"判据传的是后者。`pd.to_numeric` 对 ndarray 返回 ndarray,
        没有 `.to_numpy()`,所以先包一层 `Series`。
        """
        v = pd.to_numeric(pd.Series(values), errors="coerce")
        if self.transform == "abs":
            v = v.abs()
        arr = v.to_numpy(dtype=float)
        if self.value is None:
            return np.zeros(len(arr), dtype=bool)
        ok = ~np.isnan(arr)
        hit = arr > self.value if self.direction == ">" else arr < self.value
        return ok & hit


#: 列 → 解析规则。**只有登记在这里的列才有 τ**;没登记的返回 `None`。
#: ★ 2026-10-18 新增 5 条可信 AU 的**自比基线**。依据是实测(41 会话 / 22277 行):
#: 这 5 条的**场间中位数极差 ≥ 场内 p90−p10**,即"人跟人之间的常态差"不小于
#: "一个人自己的起伏"⟹ 跨人绝对阈值不成立,只能相对本人。
#:
#:     au1_inner_brow_raise  场间 0.0630  场内 0.0261
#:     au6_cheek_raise       场间 0.0320  场内 0.0128
#:     au10_upper_lip_raise  场间 0.5500  场内 0.0886
#:     au25_mouth_open       场间 0.0620  场内 0.0515
#:     au26_jaw_drop         场间 0.1380  场内 0.0568
#:
#: `au12_smile` **不在**这张表里:它的生产方已经把口宽除以本人全程 p10
#: (自归一化),再加一层基线就是双重归一化 —— 那条仍用声明常量 0.263。
_AU_SELF_PROVENANCE = ("自比基线(本人中位数+3·MAD);场间中位数极差 ≥ 场内 p90−p10,"
                       "见 l1_thresholds.RESOLUTION 上方实测表")

RESOLUTION: dict[str, dict] = {
    "au1_inner_brow_raise": {"strategy": STRATEGY_SELF, "direction": ">",
                             "transform": None, "k": SELF_K,
                             "provenance": _AU_SELF_PROVENANCE},
    "au6_cheek_raise": {"strategy": STRATEGY_SELF, "direction": ">",
                        "transform": None, "k": SELF_K,
                        "provenance": _AU_SELF_PROVENANCE},
    "au10_upper_lip_raise": {"strategy": STRATEGY_SELF, "direction": ">",
                             "transform": None, "k": SELF_K,
                             "provenance": _AU_SELF_PROVENANCE},
    "au25_mouth_open": {"strategy": STRATEGY_SELF, "direction": ">",
                        "transform": None, "k": SELF_K,
                        "provenance": _AU_SELF_PROVENANCE},
    "au26_jaw_drop": {"strategy": STRATEGY_SELF, "direction": ">",
                      "transform": None, "k": SELF_K,
                      "provenance": _AU_SELF_PROVENANCE},
    "au12_smile": dict(
        strategy=STRATEGY_DECLARED, direction=">", constant=AU12_SMILE_TAU,
        transform=None,
        provenance=(f"声明常量 {AU12_SMILE_TAU}(全语料 41 场 p75;**未标定**)。"
                    "生产方已除掉本人基线(口宽÷全程 p10)⟹ 场间差异是真行为差异")),
    "au4_frown": dict(
        strategy=STRATEGY_SELF, direction="<", k=SELF_K, transform=None,
        provenance=("自比基线:中位数 − 3·MAD。眉内距÷眼距,**越小越皱**;"
                    "场间基线差(1.16×)与场内动态范围同量级 ⟹ 绝对阈值量的是脸型")),
    "au23_lip_compression": dict(
        strategy=STRATEGY_SELF, direction="<", k=SELF_K, transform=None,
        provenance=("自比基线:中位数 − 3·MAD。唇红厚÷面高,**越薄越压**;"
                    "场间基线差(1.80×)大于场内动态范围")),
    "gaze_direction_x": dict(
        strategy=STRATEGY_SELF, direction=">", k=SELF_K, transform="abs",
        provenance=("自比基线:|视线偏移| 的中位数 + 3·MAD。场间中位数 −0.109~+0.922 "
                    "(符号 19/22 分裂)⟹ 绝对阈值量的是机位/朝向")),
    "head_yaw": dict(
        strategy=STRATEGY_SELF, direction=">", k=SELF_K, transform="abs",
        provenance=("自比基线:|头部偏转| 的中位数 + 3·MAD。场间中位数 −0.833~+0.389 "
                    "(符号会翻)⟹ 绝对阈值量的是朝向")),
}


def self_tau(values, column: str, *, direction: str = ">",
             transform: str | None = None, k: float = SELF_K,
             provenance: str = "") -> Tau | None:
    """**不查登记表**的自比基线 τ = 本人中位数 ± k·MAD。

    给两类调用方用:
    1. `resolve()` 的自比基线分支(所以逻辑只写一份);
    2. 没有登记、但**构念本身就是"相对本人"**的量 —— 例如手势的运动段判据
       (`gesture_activity` 逐帧信号)。那种量没有"跨人绝对阈值"可言,
       登记成声明常量反而是编数。

    返回 `None` 的两种情形,都是"算不出稳健基线",**不猜**:
    * 有效帧 < 5;
    * `values` 为空 / 全非数值。
    """
    if values is None:
        return None
    # `pd.to_numeric` 对 ndarray 返回 ndarray(没有 `.dropna()`),所以先包成 Series。
    # 调用方既有 DataFrame 的列(Series),也有逐帧信号(ndarray) —— 两种都要吃。
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if transform == "abs":
        v = v.abs()
    if len(v) < 5:
        # 少于 5 帧算不出稳健基线 —— 不猜。
        return None
    med = float(v.median())
    mad = float((v - med).abs().median())
    value = med + k * mad if direction == ">" else med - k * mad
    return Tau(column, value, direction, STRATEGY_SELF, transform, provenance)


def resolve(column: str, values: pd.Series | None) -> Tau | None:
    """解析一条列的 τ。没登记、或数据不够 ⟹ `None`(上游照旧报 `uncalibrated`)。"""
    rule = RESOLUTION.get(column)
    if rule is None:
        return None
    if rule["strategy"] == STRATEGY_DECLARED:
        return Tau(column, float(rule["constant"]), rule["direction"],
                   STRATEGY_DECLARED, rule["transform"], rule["provenance"])

    return self_tau(values, column, direction=rule["direction"],
                    transform=rule["transform"], k=float(rule["k"]),
                    provenance=rule["provenance"])


def resolve_many(columns, frame) -> dict[str, Tau | None]:
    """一次解析多列。`frame` 是本窗口的 DataFrame。"""
    return {c: resolve(c, frame[c] if c in frame.columns else None) for c in columns}
