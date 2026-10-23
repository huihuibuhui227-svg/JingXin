# report_frontend/event_windows.py
"""**M4.1 事件窗口** —— 把一整场切成"每回答一个窗口"。

## 为什么必须先做这件事

2026-10-04 之前,L1 指标的窗口粒度是"一整场",于是:

* `speech_ratio` = 语音时长 / **整场时长** ⟹ 实测 **87.9% 恒为 1.0**;
* `pause_duration` 的单位错配,根子也在"分母是一个会话常量"这个假设里。

这两条都不是"参数没调好",是**粒度没定义**。窗口定不死,G3 的阈值就没有意义 ——
`evidence_thresholds.json` 里 `n_valid_units` 登记的是 `frame` / `answer`,
而当时根本没有 `answer` 这个东西。

## 窗口怎么定

窗口 **k** = `[ask_end_k, ask_end_{k+1})`,最后一个取到会话结束。

**用 `ask_end` 而不是 `ask_start`**:指标问的是"被问完之后他做了什么",
`ask_start → ask_end` 那段是**机器在说话**,不是候选人的行为。
用 `ask_start` 当切点会把提问时长算进"反应延迟"里。

来源是 `session_meta.read_questions()` —— 与 `reaction_time` 用的是**同一份台账**
(`voice_interaction/api/app.py:632` 按 `index` 对齐),不另建一份。

## 两个"不许"

1. **不许按记录顺序硬配。** 题号与音频段不是靠"第几条"对上的,是按时间边界切的。
   一旦靠顺序,一旦有一次重传/补推,后面全体错位而且**静默**。
2. **不许在没有台账时编一个窗口。** 26 场自然对话没有题库台账(戊-15 待裁定),
   它们只有 `granularity="session"` 一个窗口,而且**明写原因**。

## 与时间轴的关系

窗口边界是**墙钟**(`ask_end` 来自 `questions.jsonl`),而 L1 的时间轴以
`declared_ts` 为主(spec §2)。所以本模块接收一个 `origin_wall`,把所有边界
统一换算成"距会话原点的秒" —— 与 `session_timeline.align_*` 的产出同一量纲。

⚠️ **墙钟不可信的场次**(spec §2.2 那 4 场),窗口边界带着同样的误差。
本模块不修它,只把 `wall_reliable` 传下去,让上层决定要不要用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import media_retention
import session_meta

#: 会话级兜底窗口的标签。报表里必须能一眼看出"这一场没切成回答"。
SESSION_GRANULARITY = "session"
ANSWER_GRANULARITY = "answer"


@dataclass
class AnswerWindow:
    """一个回答窗口。时间量都是**距会话原点的秒**,与 `session_timeline` 同量纲。"""

    index: int
    t_start: float
    t_end: float
    qid: str = ""

    def contains(self, t: float) -> bool:
        """左闭右开。`t` 为 NaN 一律不算(缺失不是"落在某个窗口")。"""
        if t is None or (isinstance(t, float) and np.isnan(t)):
            return False
        return self.t_start <= t < self.t_end

    @property
    def duration_sec(self) -> float:
        return self.t_end - self.t_start


@dataclass
class WindowSet:
    session_id: str
    granularity: str
    windows: list[AnswerWindow] = field(default_factory=list)
    n_questions: int = 0
    n_answer_segments: int = 0
    wall_reliable: bool = True
    gap: str | None = None
    #: 模态 → **落在所有窗口之外**的帧数。
    #:
    #: 为什么必须数:有台账时原点 = `min(ask_start)`,而第一个窗口从第一个 `ask_end`
    #: 才开始 ⟹ `[原点, 第一个 ask_end)` 那一段(提问期 + 首题前)不属于任何窗口,
    #: `assign_series` 对它们静默返回 `None`。不数就等于**分母悄悄变小而没人知道** ——
    #: 而 F/G 两族的比率类指标全靠这个分母。
    frames_outside: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.windows)

    def assign(self, t) -> int | None:
        """一个时间点落在哪个窗口。落不进任何一个 ⟹ `None`(不四舍五入到最近)。"""
        for w in self.windows:
            if w.contains(float(t)):
                return w.index
        return None

    def count_outside(self, t) -> int:
        """`t` 里**不属于任何窗口**的帧数。NaN 不算(缺失不是"落在窗口外")。

        会话级(`windows == []`)时恒为 0:那时整场就是一个窗口,没有东西被排除。
        """
        if not self.windows:
            return 0
        s = t if isinstance(t, pd.Series) else pd.Series(t, dtype=float)
        ids = self.assign_series(s)
        return int((ids.isna() & s.notna()).sum())

    def assign_series(self, t: pd.Series) -> pd.Series:
        """逐行分配。返回与 `t` 同 index 的 Series,值为题号或 `None`。

        用 `searchsorted` 而不是逐行 `assign` —— 一场几千帧,逐个 `contains`
        是 O(n·k),而这里 k 是题数(8),实际上也不慢;但**语义**上必须走同一处
        (`_index_of`),两份实现早晚不一致。
        """
        ids = [w.index for w in self.windows]
        starts = np.array([w.t_start for w in self.windows], dtype=float)
        vals = t.to_numpy(dtype=float)
        out: list[int | None] = []
        for v in vals:
            if np.isnan(v):
                out.append(None)
                continue
            k = int(np.searchsorted(starts, v, side="right")) - 1
            if k < 0 or not self.windows[k].contains(v):
                out.append(None)
            else:
                out.append(ids[k])
        return pd.Series(out, index=t.index, dtype=object)


def audio_answer_segments(session_id: str) -> list[dict]:
    """账本里 `/interview/answer_audio` 的 raw 段,按到达顺序。

    只作**诊断**(与题数对不对得上),不参与切窗口 —— 见模块 docstring「两个不许」。
    """
    path = media_retention.resolve_recording_dir(session_id) / (
        f"{media_retention.LEDGER_PREFIX}.audio{media_retention.LEDGER_SUFFIX}")
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("kind") == "raw" and rec.get("source_endpoint") == "/interview/answer_audio":
            out.append(rec)
    return out


def build_windows(session_id: str, origin_wall: float,
                  session_end_sec: float | None = None,
                  wall_reliable: bool = True) -> WindowSet:
    """切出本场的回答窗口。

    `session_end_sec`:最后一个窗口的右边界(距原点秒)。不给 ⟹ 用"最后一条日志的
    时间"由调用方补,或先用 `+inf` 由 `clip_session_end` 收口。**不给也不猜** ——
    这里用 `+inf` 会让最后一个窗口吞掉场后所有数据,所以调用方应当给。
    """
    try:
        questions = session_meta.read_questions(session_id)
    except Exception as exc:
        return WindowSet(session_id, SESSION_GRANULARITY, [], wall_reliable=wall_reliable,
                         gap=f"提问台账读不出:{exc}")

    usable = [q for q in questions
              if q.get("ask_end") is not None and q.get("index") is not None]
    if not usable:
        return WindowSet(session_id, SESSION_GRANULARITY, [], len(questions),
                         wall_reliable=wall_reliable,
                         gap=("本场没有提问台账 —— 无法切成回答窗口"
                              if not questions else
                              "提问台账里没有可用的 ask_end"))

    usable.sort(key=lambda q: (float(q["ask_end"]), int(q["index"])))
    end = float(session_end_sec) if session_end_sec is not None else float("inf")

    windows: list[AnswerWindow] = []
    for i, q in enumerate(usable):
        t_start = float(q["ask_end"]) - origin_wall
        t_end = (float(usable[i + 1]["ask_end"]) - origin_wall) if i + 1 < len(usable) else end
        windows.append(AnswerWindow(index=int(q["index"]), t_start=t_start, t_end=t_end,
                                    qid=str(q.get("qid") or "")))

    # 相邻窗口边界必须严格递增 —— 两题的 ask_end 撞在一起时,前者会退化成空窗口。
    # 空窗口不是错误(题被跳过),但**必须能看见**,所以数出来。
    empty = [w.index for w in windows if w.duration_sec <= 0]

    segments = audio_answer_segments(session_id)
    gap = None
    if segments and len(segments) != len(usable):
        gap = (f"answer_audio 段 {len(segments)} 条 vs 可用题窗口 {len(usable)} 个 —— "
               f"对不上;窗口仍按 ask_end 切(不按顺序硬配),但语音行的归属要复核")
    elif not segments:
        # **没有逐题回答音频 = 一种采集模式,不是缺陷。**
        # 原来这里会走上面的"段 0 条 vs 窗口 N 个 —— 对不上",于是"只有提问标记、
        # 没有每题音频"的场次**每一场都永久带一条假缺陷**,把真缺陷稀释掉。
        # 逐题窗口照常可用(F/G 按题切);但没有 reaction_time ⟹ 反应延迟不可出。
        gap = ("本场没有逐题回答音频(只有提问标记)⟹ 逐题窗口照常可用,"
               "但没有 reaction_time,response_latency_sec 不可出")
    elif empty:
        gap = f"题 {empty} 的窗口时长为 0(相邻两题 ask_end 相同或倒序),该窗口内不会有任何观测"
    if not wall_reliable:
        extra = "本场墙钟不可信(见 timebase),窗口边界带同等量级误差"
        gap = f"{gap};{extra}" if gap else extra

    return WindowSet(session_id, ANSWER_GRANULARITY, windows,
                     n_questions=len(questions),
                     n_answer_segments=len(segments),
                     wall_reliable=wall_reliable, gap=gap)


def clip_session_end(ws: WindowSet, session_end_sec: float) -> WindowSet:
    """把最后一个窗口的右边界收到会话结束 —— 不猜时它会停在 `+inf`。"""
    if not ws.windows:
        return ws
    last = ws.windows[-1]
    if last.t_end > session_end_sec:
        last.t_end = session_end_sec
    return ws
