# report_frontend/clock_repair.py
"""**墙钟修复** —— 把会倒退的 `time.time()` 映射回单调时间轴。

## 背景(实测,见 `docs/下一步.md` 丙-21)

留存账本的 `received_at_wall` 取自 `time.time()`,会被系统时钟调整/VM 漂移**回拨**;
`declared_ts` 取自 `session_clock.SessionClock`(`time.monotonic`),**物理上不会倒退**。

⟹ 只要同时有这两列,就能把墙钟"修"成单调的:把倒退的那一段按住不动
(`np.maximum.accumulate`),再在 `(修好的墙钟 → declared)` 上插值。

## 为什么只修"间隔",不修"绝对时刻"

墙钟的**绝对值**已经被污染了(某场实测偏 +977 s),修不回来 —— 没有第二个绝对基准。
但**两个墙钟量之差**只受"落在这段区间内的那些跳变"影响,而跳变的位置和大小
都能从账本读出来。所以:

* `reaction_time = 回答开始 − ask_end`:两个墙钟量之差 ⟹ **可修**;
* `ask_end` 本身对不对 ⟹ **不可修**,只能说不确定。

这条边界必须守住,否则就会变成"我修好了时钟"这种站不住的宣称。

## 实测(三场受影响的会话,34 场里有墙钟倒退的只有这 3 场)

```
场次                    题   原反应    修正后     差
20260926_153202_2b11    1    43.03    46.68   +3.65
20260926_153854_1592    0   205.98   218.94  +12.96
20260926_155559_caf0    5    71.66    76.99   +5.33
```

修正量全部是 **~1.8 s 的整数倍**(1.8 / 3.6 / 5.3 / 13.0),与账本里观测到的
单次回拨量一致 —— 这是"方法对上了"的独立佐证,不是我挑出来的。

**其余 31 场修正量为 0**,因为它们的墙钟本来就是单调的。

## 三态,不许混

* `clean` —— 账本在,墙钟单调 ⟹ 无需修正,`delta == 0` 且**可信**;
* `repaired` —— 账本在,有回拨 ⟹ 给出修正量;
* `no_ledger` —— 没有账本 ⟹ **无法判断**,`delta` 记 0 但**必须**带上这一状态。
  把"没法判断"报成 `clean` 就是在说"这个数是准的",而我不知道。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from report_frontend.session_timeline import frames_ledger

#: 三态。`no_ledger` 不是"没问题",是"查不了"。
CLEAN = "clean"
REPAIRED = "repaired"
NO_LEDGER = "no_ledger"
OUT_OF_RANGE = "out_of_range"

#: 判定"墙钟单调"的容差(秒)。小于这个数的倒退当浮点噪声,不算回拨。
BACKSTEP_TOLERANCE = 1e-3

#: 平台歧义小于这个数就当"不歧义"(浮点噪声)。
AMBIGUITY_TOLERANCE = 0.05


@dataclass
class RepairMap:
    """墙钟 → 真实经过时间 的映射。两列都相对本场第一个 frame。"""

    session_id: str
    #: 单调化后的墙钟(相对秒)。平台段(时间"停住"的那一段)保留**首尾两个锚点**,
    #: 于是同一个读数会得到两个可能的真实时刻 —— 这不是缺陷,是回拨本身的信息损失。
    wall_mono: np.ndarray
    declared: np.ndarray
    #: 平台段的**末端**锚点。与上面两列配对使用,给出区间的另一端。
    n_back_steps: int
    #: **相邻帧之间**观测到的最大倒退量(秒)。它是时钟真实调整量的**下界** ——
    #: 帧间隔约 1 s,一次 2 s 的回拨只会表现为 1 s 的帧间倒退。
    #: 不要把它当成"时钟被调了多少"。
    max_back_step: float
    #: 平台段的**末端**锚点(见 `elapsed_range`)。必须排在无默认值字段之后。
    wall_mono_hi: np.ndarray = None
    declared_hi: np.ndarray = None
    #: 回拨造成的"重复墙钟区间" —— 每一次回拨都会让 `[w[i], w[i-1]]` 这一段被走两遍,
    #: 于是落在里面的任何读数都对应**两个**真实时刻。形状 (n, 2) 的 (lo, hi)。
    #:
    #: ⚠️ 不能用"浮点值精确相等"去判重复(第一版就是这么写的,结果恒为 0 个)——
    #: 连续时间戳几乎永远不会精确相等。要按**区间**判。
    ambiguous_ranges: np.ndarray = None
    #: 本场第一帧的绝对墙钟。对外接口收**绝对**墙钟(调用方手上就是那个),
    #: 内部减掉它换成相对量 —— 归一化只在这一处做,免得每处调用各减一次(踩过)。
    wall_base: float = 0.0

    @property
    def status(self) -> str:
        return CLEAN if self.n_back_steps == 0 else REPAIRED

    def elapsed_range(self, t_wall_rel: float) -> tuple[float, float]:
        """墙钟相对时刻 → (最早可能, 最晚可能) 的真实经过秒。

        平台段上同一读数对应一段时间,单一答案是不存在的。调用方要么接受区间,
        要么明确挑一端 —— 不许假装只有一个值。
        """
        rel = float(t_wall_rel) - self.wall_base
        lo = float(np.interp(rel, self.wall_mono, self.declared))
        if self.wall_mono_hi is None:
            return lo, lo
        hi = float(np.interp(rel, self.wall_mono_hi, self.declared_hi))
        return (lo, hi) if lo <= hi else (hi, lo)

    def ambiguous(self, t_wall_rel: float) -> bool:
        """这个墙钟读数是否落在"被走两遍"的区间里(⟹ 对应多个真实时刻)。"""
        return self.ambiguity_bound(t_wall_rel) > 0.0

    def ambiguity_bound(self, t_wall_rel: float) -> float:
        """该读数的不确定度上界(秒)。0 表示不歧义。

        落在重复区间里 ⟹ 事件可能发生在第一遍也可能第二遍,两者相差最多区间宽度。
        """
        if self.ambiguous_ranges is None or self.ambiguous_ranges.size == 0:
            return 0.0
        rel = float(t_wall_rel) - self.wall_base
        lo = self.ambiguous_ranges[:, 0]
        hi = self.ambiguous_ranges[:, 1]
        hit = (rel >= lo) & (rel <= hi)
        if not hit.any():
            return 0.0
        return float((hi - lo)[hit].max())

    def covers(self, t_wall_rel: float) -> bool:
        rel = float(t_wall_rel) - self.wall_base
        return bool(self.wall_mono[0] <= rel <= self.wall_mono[-1])


def build_repair_map(session_id: str, modality: str = "face") -> RepairMap | None:
    """从 frame 账本建映射。账本缺失/为空 ⟹ `None`(调用方须报 `no_ledger`)。"""
    led = frames_ledger(session_id, modality)
    if led.empty:
        return None
    d = led["declared_ts"].to_numpy(dtype=float) / 1000.0
    w_abs = led["received_at_wall"].to_numpy(dtype=float)
    base = float(w_abs[0])
    w = w_abs - base

    diffs = np.diff(w)
    back = diffs[diffs < -BACKSTEP_TOLERANCE]
    wm = np.maximum.accumulate(w)

    # `wall_mono` 必须严格可插值:`np.interp` 要求 xp 递增,平台段允许,但重复值要留
    # 第一个(时间"停住"的那一段取起点)。
    keep = np.concatenate([[True], np.diff(wm) > 0])
    # 平台段末端:下一个值发生变化的位置(最后一个自然保留)
    keep_hi = np.concatenate([np.diff(wm) > 0, [True]])

    # 每一次回拨 ⟹ 区间 [w[i], w[i-1]] 被走两遍。按**区间**收集,不按值相等。
    amb = np.array([[w[i], w[i - 1]] for i in range(1, len(w))
                    if w[i] < w[i - 1] - BACKSTEP_TOLERANCE], dtype=float)
    if amb.size == 0:
        amb = np.empty((0, 2))

    return RepairMap(session_id=session_id, wall_mono=wm[keep], declared=d[keep],
                     wall_mono_hi=wm[keep_hi], declared_hi=d[keep_hi],
                     n_back_steps=int(back.size),
                     max_back_step=float(-back.min()) if back.size else 0.0,
                     ambiguous_ranges=amb, wall_base=base)


@dataclass
class Interval:
    """一段被墙钟计量的间隔。`raw` 是原样相减,`repaired` 是修好之后。"""

    raw: float
    repaired: float
    status: str
    note: str = ""
    #: 端点落在"重复墙钟值"上 ⟹ 这个修正值本身带不确定度(见 `RepairMap.ambiguous`)
    ambiguous: bool = False

    @property
    def delta(self) -> float:
        return self.repaired - self.raw

    def to_dict(self) -> dict:
        return {"raw": round(self.raw, 3), "repaired": round(self.repaired, 3),
                "delta": round(self.delta, 3), "status": self.status,
                "ambiguous": self.ambiguous, "note": self.note}


def repair_interval(rmap: RepairMap | None, t_start_wall_rel: float,
                    t_end_wall_rel: float) -> Interval:
    """修一段间隔。**没有映射时不许假装干净** —— 返回 `no_ledger`。"""
    raw = float(t_end_wall_rel) - float(t_start_wall_rel)
    if rmap is None:
        return Interval(raw=raw, repaired=raw, status=NO_LEDGER,
                        note="没有账本,无法判断墙钟是否倒退 —— 这个数**未经核对**")
    if not (rmap.covers(t_start_wall_rel) and rmap.covers(t_end_wall_rel)):
        return Interval(raw=raw, repaired=raw, status=OUT_OF_RANGE,
                        note="该间隔超出账本覆盖范围,无法核对")
    lo_s, hi_s = rmap.elapsed_range(t_start_wall_rel)
    lo_e, hi_e = rmap.elapsed_range(t_end_wall_rel)
    repaired = ((lo_e + hi_e) - (lo_s + hi_s)) / 2.0
    # 端点的平台歧义直接变成间隔的上下界
    bound = max(abs(hi_e - lo_e), abs(hi_s - lo_s))
    amb = bound > AMBIGUITY_TOLERANCE
    note = "" if rmap.status == CLEAN else (
        f"本场墙钟回拨 {rmap.n_back_steps} 次(帧间最大倒退 {rmap.max_back_step:.2f} s,实际调整量只会更大),"
        f"原值低估 {repaired - raw:+.2f} s")
    if amb:
        extra = (f"端点落在墙钟平台段上,修正值本身还有 ±{bound / 2:.2f} s 不确定度"
                 f"(平台段内同一读数对应一段时间,没有唯一解)")
        note = f"{note};{extra}" if note else extra
    return Interval(raw=raw, repaired=repaired, status=rmap.status, note=note,
                    ambiguous=amb)


def reaction_times(session_id: str, questions: list[dict],
                   voice_rows: list[dict]) -> list[dict]:
    """逐题给出"原反应延迟 / 修正后 / 差"。

    `questions`:`session_meta.read_questions()` 的产出;
    `voice_rows`:语音日志里带 `answer_onset_wall` 的行(转成 dict)。

    ⚠️ 语音日志的 `reaction_time` 是**采集时**算的,本函数**不改它** ——
    只并列给出修正值,让报告可以选择用哪个、并且能说清差在哪。
    改采集侧会让历史日志与新日志不可比。
    """
    rmap = build_repair_map(session_id)
    by_index = {int(q["index"]): q for q in questions if q.get("index") is not None}
    out: list[dict] = []
    for row in voice_rows:
        onset = row.get("answer_onset_wall")
        if onset is None:
            continue
        idx = row.get("index")
        q = by_index.get(int(idx)) if idx is not None else None
        if q is None or q.get("ask_end") is None:
            continue
        ask_end = float(q["ask_end"])
        onset = float(onset)
        iv = repair_interval(rmap, ask_end, onset)
        out.append({"index": int(idx),
                    "raw_reaction_time": round(onset - ask_end, 3),
                    "repaired_reaction_time": round(iv.repaired, 3),
                    "delta": round(iv.delta, 3),
                    "status": iv.status,
                    "ambiguous": iv.ambiguous,
                    "note": iv.note})
    return out
