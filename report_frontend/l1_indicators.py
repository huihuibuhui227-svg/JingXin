# report_frontend/l1_indicators.py
"""**M4.2 L1 指标的数据结构** —— 先有形状,再有数。

## 一条 L1 指标长什么样

键集**固定**(`L1Indicator` 的字段),`to_dict()` 输出的就是报告侧能直接消费的形状。
设计文档 §5.6 给了原型,本文件把它做成**可校验**的:

```json
{
  "name": "blink_rate_per_min",
  "family": "F3_blink",
  "formula": "count_rising_edges(is_blink) / (valid_span_sec / 60)",
  "required_l0": ["is_blink", "timestamp"],
  "window": "answer:0",
  "value": 14.0,
  "unit": "次/分钟",
  "n_valid": 187,
  "n_total": 200,
  "confidence": "中",
  "gap": null
}
```

## 三条不变量(违反即 `ValueError`,不是"警告一下")

1. **算不出来就说算不出来**:`gap != None` ⟹ `value` 必须是 `NaN`、`confidence` 必须是 `"无"`;
2. **`n_valid` 不许超过 `n_total`** —— 那是把证据说多,而"把证据说多"正是本轮要消灭的东西;
3. **`confidence` 只能是那四个中文词**,不接受 `high/medium/...` 之类的近义写法
   (两份词表早晚会不一致)。

## 四档必须都可达

2026-10-04 之前 `confidence_from()` 只有"高/中""中/低"两对可用,
`"高"` 与 `"无"` 都**不可达**,于是零输入也报"中" —— 这是当时那份报告的根因之一。
本文件的 `confidence_for()` 把四档都能取到,并由测试钉住(每一档一个用例)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: 四档。顺序即"从可信到不可信",`CONFIDENCE[0]` 最可信。
CONFIDENCE = ("高", "中", "低", "无")

#: "纯可靠"档要求 `n_valid ≥ 2 × 阈值` 且有效率不低于这个比例。
HIGH_COVERAGE = 0.8

#: `gap` 的取值。**只用这几个** —— 自由文本会退化成"什么都能写",那就没法统计了。
GAP_UNCALIBRATED = "uncalibrated"          # 阈值 τ 只能来自标定集(M5),现在拿不到
GAP_COLUMN_EMPTY = "column_empty"          # 上游根本没采集这一列(实测 0 非空)
GAP_MODALITY_MISSING = "modality_missing"  # 整场没有这个模态
GAP_TIME_UNUSABLE = "time_unusable"        # 时间轴不可重建(spec M4.0)
GAP_WINDOW_NONE = "window_none"            # 没有可用的回答窗口(戊-15)
GAP_NO_SAMPLES = "no_samples"              # 窗口里一条观测都没有
GAP_NOT_STORED = "not_stored"              # 上游只存了聚合量,这一层的粒度产不出来

KINDS = (GAP_UNCALIBRATED, GAP_COLUMN_EMPTY, GAP_MODALITY_MISSING,
         GAP_TIME_UNUSABLE, GAP_WINDOW_NONE, GAP_NO_SAMPLES, GAP_NOT_STORED)


@dataclass
class L1Indicator:
    """一条会话级（或回答窗口级）指标。字段集固定,不要随手加。"""

    name: str
    family: str
    formula: str
    required_l0: tuple[str, ...]
    window: str
    unit: str
    value: float = math.nan
    n_valid: int = 0
    n_total: int = 0
    confidence: str = "无"
    gap: str | None = None

    def __post_init__(self) -> None:
        if self.confidence not in CONFIDENCE:
            raise ValueError(f"confidence 只能是 {CONFIDENCE},收到 {self.confidence!r}")
        if self.gap is not None and self.gap not in KINDS:
            raise ValueError(f"gap 只能是 {KINDS} 或 None,收到 {self.gap!r}")
        if self.n_valid < 0 or self.n_total < 0:
            raise ValueError(f"样本量不能为负:{self.n_valid}/{self.n_total}")
        if self.n_valid > self.n_total:
            raise ValueError(
                f"n_valid({self.n_valid}) > n_total({self.n_total}) —— 把证据说多了")
        # 不变量 1:gap 与"有没有值/能不能信"必须一致
        if self.gap is not None:
            if not _isnan(self.value):
                raise ValueError(f"{self.name}: 登记了 gap={self.gap!r} 就必须 value=NaN")
            if self.confidence != "无":
                raise ValueError(f"{self.name}: 登记了 gap 就必须 confidence='无'")
        elif self.confidence == "无" and not _isnan(self.value):
            raise ValueError(f"{self.name}: confidence='无' 却有值 —— 二者必须一致")

    @property
    def has_value(self) -> bool:
        return not _isnan(self.value)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "family": self.family,
            "formula": self.formula,
            "required_l0": list(self.required_l0),
            "window": self.window,
            "value": None if _isnan(self.value) else float(self.value),
            "unit": self.unit,
            "n_valid": int(self.n_valid),
            "n_total": int(self.n_total),
            "confidence": self.confidence,
            "gap": self.gap,
        }


def _isnan(v) -> bool:
    try:
        return v is None or (isinstance(v, float) and math.isnan(v))
    except TypeError:
        return False


def confidence_for(n_valid: int, n_total: int, threshold: int) -> str:
    """四档判定。**四档都必须能取到**,见模块 docstring。

    * `无`   —— 一条有效观测都没有;
    * `低`   —— 有观测,但不到该族的门槛;
    * `中`   —— 过门槛,但不到"两倍门槛且覆盖率 ≥ 80%";
    * `高`   —— 过门槛,且 `n_valid ≥ 2×threshold`、`n_valid/n_total ≥ 0.8`。
    """
    if n_valid <= 0:
        return "无"
    if n_valid < threshold:
        return "低"
    if n_valid >= 2 * threshold and n_total > 0 and n_valid / n_total >= HIGH_COVERAGE:
        return "高"
    return "中"


def unavailable(name: str, family: str, formula: str, required_l0: tuple[str, ...],
                window: str, unit: str, gap: str, n_total: int = 0) -> L1Indicator:
    """"这个指标这一轮算不出来"的标准构造。**不许**为了凑一个有值而绕开它。"""
    return L1Indicator(name=name, family=family, formula=formula, required_l0=required_l0,
                       window=window, unit=unit, value=math.nan,
                       n_valid=0, n_total=n_total, confidence="无", gap=gap)


def measured(name: str, family: str, formula: str, required_l0: tuple[str, ...],
             window: str, unit: str, value: float, n_valid: int, n_total: int,
             threshold: int) -> L1Indicator:
    """有值的标准构造。`confidence` 由 `confidence_for` 统一决定,调用方不许自己填。"""
    return L1Indicator(name=name, family=family, formula=formula, required_l0=required_l0,
                       window=window, unit=unit, value=float(value),
                       n_valid=int(n_valid), n_total=int(n_total),
                       confidence=confidence_for(n_valid, n_total, threshold), gap=None)


def validate_all(indicators: list[L1Indicator]) -> list[L1Indicator]:
    """整批自检 —— 结构在构造时已校验,这里查**整批层面**的两件事。"""
    seen: set[tuple[str, str]] = set()
    for ind in indicators:
        key = (ind.name, ind.window)
        if key in seen:
            raise ValueError(f"同一窗口里重复的指标:{key}")
        seen.add(key)
    return indicators
