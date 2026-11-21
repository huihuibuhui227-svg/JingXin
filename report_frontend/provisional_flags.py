# patch/provisional_flags.py  →  安装到 report_frontend/provisional_flags.py
"""**未经标定的阈值要自带一张"我不是标定值"的标签。**

## 要解决的是什么

本仓有一批阈值,它们**不是**标定出来的,而是"先拿一个数跑起来"的临时值:

| 阈值 | 值 | 出处 | 依据 |
|---|---|---|---|
| 整场转写低置信线 | **0.75** | `工作脚本/whisper_whole.py:45`(`LOW_CONF`) | 无标定集;实测 135 个逐题转写文件里 **32.6%** 低于它 |
| 面部 `au12_smile` 的门限 | 0.263 | `report_frontend/l1_thresholds.py` | 全语料 41 场 **p75**,代码里已写"未标定" |
| `evidence_thresholds.json` 整份 | — | `_version: "0.3.3-provisional"` / `_provisional: true` | 自认"物理下限,非标定值" |

问题不在"用了临时值"(没有标定集时只能用),问题在**临时值长得跟标定值一样**:
报告里印一个 `0.75`,`eval` 一次就分档,读者无从知道这个数有没有依据。
`evidence_thresholds.json` 的做法是在**文件级**写 `_provisional: true` ——
那是对的,但它只保护了那一份文件里的阈值;散在脚本与代码里的那些没有保护。

## 这个模块给什么

`mark_provisional(threshold_name, value, basis)` 把"一个数"变成"一个声明":

```python
mark_provisional("asr_low_conf", 0.75, "无标定集;135 个逐题转写里 32.6% 低于它")
# {"name": "asr_low_conf", "value": 0.75, "status": "provisional",
#  "basis": "无标定集;135 个逐题转写里 32.6% 低于它"}
```

三条硬规矩(每条都由 `tests/test_gap_explanations.py` 钉):

1. **没有依据就不许标** —— `basis` 为空 ⟹ `ValueError`。"provisional" 若可以
   不带理由,它就会变成一句口头禅,和没标一样;
2. **`status` 只有一个值** —— `"provisional"`。**没有** `"calibrated"`:
   标定到位那天,正确做法是把这条登记**删掉**(并让它进
   `evidence_thresholds.json` 的 `_thresholds_basis`),而不是在本模块里
   把字符串改成另一个 —— 否则同一件事有两个出口,迟早分叉;
3. **`value` 必须是真数** —— 字符串/占位符一律拒绝。标记的是"这个数不可信",
   不是"这里还没有数";后者该走缺值那条路(`gap`),不该混进来。

## 为什么值是**原样**带出去的

`mark_provisional` 不改 `value`:它不是"折算器"。调用方拿到的仍是那个数,
只是多带了 `status` 与 `basis` —— 报告既能照旧取数,又能把标签印出来。
"""

from __future__ import annotations

#: `status` 的唯一取值。见模块 docstring 第 2 条:没有第二个状态。
PROVISIONAL = "provisional"

#: 会被拒绝的"空依据"写法。空字符串之外,这几个占位符同样不算依据 ——
#: `"TODO"` / `"?"` 这类东西进报告,读者会以为是缩写而不是"没写"。
_PLACEHOLDERS = ("", "-", "--", "?", "todo", "tbd", "n/a", "na", "无", "待补")


def mark_provisional(threshold_name: str, value, basis) -> dict:
    """把一个未经标定的阈值登记成**带标签**的字典。

    :param threshold_name: 阈值名(如 `"asr_low_conf"`)。空 ⟹ `ValueError`。
    :param value: 阈值本身。必须是实数(`int` / `float`),`bool` 不算。
    :param basis: **为什么它只是 provisional** —— 一句话,或 `{...}` 明细。
        空 / 占位符(`"TODO"`、`"?"`、`"无"` …)⟹ `ValueError`。
    :return: `{"name":…, "value":…, "status": "provisional", "basis":…}`

    ## 为什么 `name` 也在里面

    标记是要**汇总**的(报告里"以下阈值未经标定"那一段要排队印出来)。
    一个只有 `value` 的字典进了那份清单就分不清是哪一条 —— 0.75 既可能是
    转写置信度,也可能是别的量。名字必须跟着值一起走。

    ## 为什么不接 `None`

    `mark_provisional("x", None, "…")` 会被拒:`None` 是"没有数",该报缺口
    (`gap`),不该报"这个数不可信"。两件事混了之后,报告会拿一条
    `provisional` 的空值去填表,而"空"和"不可信"的补救动作完全不同。
    """
    name = (threshold_name or "").strip() if isinstance(threshold_name, str) else ""
    if not name:
        raise ValueError("阈值名不能为空 —— 无名的 provisional 标记进了报告也分不清是哪一条")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: value 必须是实数,收到 {value!r}")
    if not _has_basis(basis):
        raise ValueError(
            f"{name}: 没有依据就不许标 provisional(收到 {basis!r})—— "
            "不带理由的临时值等于没标,读者照样不知道它有没有依据")
    return {"name": name, "value": value, "status": PROVISIONAL, "basis": basis}


def _has_basis(basis) -> bool:
    """依据是否**真的**写了。字符串比占位符表(只比全等,不拦「无标定集」这种真理由)。

    容器(字典/列表)要求**至少一条**非空 —— `{"why": ""}` 与 `[]` 都算没写。
    """
    if isinstance(basis, dict):
        return any(_has_basis(v) for v in basis.values())
    if isinstance(basis, (list, tuple, set)):
        return any(_has_basis(v) for v in basis)
    if isinstance(basis, str):
        text = basis.strip()
        return bool(text) and text.lower() not in _PLACEHOLDERS
    return basis is not None


def is_provisional(marked) -> bool:
    """这个值是不是一个 provisional 标记 —— 报告层据此决定要不要印"未标定"。"""
    return isinstance(marked, dict) and marked.get("status") == PROVISIONAL


def line(marked) -> str:
    """标记 → 报告里那一行字(`{名字}:{值}(未标定)`),不是标记就返回空串。

    措辞刻意最短:它要跟在同一行的数值后面,长句会把那一行的数挤走。
    """
    if not is_provisional(marked):
        return ""
    return f"{marked['name']}:{marked['value']}(未标定)"
