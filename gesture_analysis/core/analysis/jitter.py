"""抖动(时间归一化率)—— 四个 analyzer **共用**的那一份公式(M3.3 B3,2026-09-27)。

为什么单独一个文件:同一段 `np.std(positions, axis=0).mean()` 此前在 4 个 analyzer 里
**逐字重复了 4 遍**(`hand_analyzer` / `arm_analyzer` / `upper_body_analyzer` 各一份
`_calculate_jitter`,`shoulder_analyzer` 一份 `_calculate_shoulder_jitter`)。M3 要给它们
各加两处分母(÷ 肩宽、÷ 窗内真实秒数)—— 在 4 处各写一遍,就是在 4 处各错一遍的机会。
判据见 `l0_columns.json` 的那 16 行 `*_jitter` / `*_jitter_world`。

**为什么时间戳必须进窗口**(而不是用"帧数 ÷ 假定帧率"):后者等于假设帧率恒定,而帧率
正是被污染的那个量(那些行的 `basis` ② 明写)。窗长 30 **帧**是现行常量
(`gesture_analysis/config.py` 的三处 `history_length`),本批不改成秒 —— 谁能改、要
凭什么改,写在那些行的 `basis` 里。
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

# 现行实现的窗口下限:`min(10, history.maxlen // 3)`。`history_length = 30` ⟹ 10。
_MIN_FRAMES_FLOOR = 10

_MS_PER_SECOND = 1000.0


def window_median(values: Optional[Iterable]) -> Optional[float]:
    """窗内中位数 —— **只数有值那些帧**。一帧都没有 ⟹ `None`(没有标尺)。

    「有值」= 不是 `None`:缺肩的帧在历史里占位(`None`),但对中位数没有发言权。
    口径出处:那 10 行的 `basis` ①「肩宽取**窗内中位数** —— 窗统计量用窗级标尺,
    避免瞬时肩宽噪声进分母」。
    """
    if not values:
        return None
    finite = [float(v) for v in values if v is not None]
    if not finite:
        return None
    return float(np.median(finite))


def windowed_jitter(history: Sequence[tuple],
                    shoulder_widths: Optional[Sequence] = None,
                    *, divide_by_shoulder: bool) -> Optional[float]:
    """一个历史窗 → 抖动**率**。`history` 的每一条是 `(timestamp_ms, x, y)`。

    返回 = `逐轴标准差的均值` ÷ `窗内真实秒数`(画面坐标那批再 ÷ `窗内肩宽中位数`)。

    **两条 `return ... or None` 是本函数的判据**(不是防御性编程):
      · 窗内**没走过时间**(`t_last == t_first`):"每秒抖多少"在零秒上没有定义;
      · `divide_by_shoulder` 而窗内**没有任何肩宽**:没有分母就没有这个率。
    两种情形都交 `None`(= 这一帧没有这个量)⟹ 调用方落**空**、不落 0
    (0 的意思是"完全静止",那是个测量结果,不是"没测到")。
    ⚠️ 交未归一化的分子顶上更不行 —— 那等于把单位从「肩宽/秒」静默换成「归一化图像单位」,
    下游从那一行看不出换了单位(本项目反复禁止的形态)。

    ⚠️ **只有"窗还没攒够"那一条仍交 `0.0`**:那是现行实现的行为(`_MIN_FRAMES_FLOOR`),
    本批不动它(实测 3 场正式素材:每列前 9 行都是 0.0,改前改后一致)。要改成"空"
    是另一次口径决定 —— 得先在 `l0_columns.json` 那几行里立字据。
    """
    min_len = min(_MIN_FRAMES_FLOOR, (history.maxlen or 0) // 3)
    if len(history) < min_len:
        return 0.0

    positions = np.array([(p[1], p[2]) for p in history], dtype=float)
    std = float(np.std(positions, axis=0).mean())

    t_first, t_last = history[0][0], history[-1][0]
    if t_first is None or t_last is None:
        return None                      # 没有时间基(旧调用方没给时间戳)
    span_s = (t_last - t_first) / _MS_PER_SECOND
    if span_s <= 0:
        return None

    if divide_by_shoulder:
        scale = window_median(shoulder_widths)
        if not scale or scale <= 0:
            return None
        std /= scale

    return std / span_s
