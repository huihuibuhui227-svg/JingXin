# report_frontend/l1_session.py
"""**M4 的总入口** —— 一场会话 → 一组 L1 指标。

把 M4.0(时间轴)、M4.1(窗口)、M4.2(结构)、M4.3(F 族)串起来,对外只暴露一个函数。

## 它与 `research_mapper` 的关系:**互不干扰**

`research_mapper.map_features_to_scores()` 照旧产出那 5 个维度、14 个槽,
一个字节都不动。本模块**另起一条链**,产出一组**平铺的 L1 指标**。

为什么不直接替换:那些 14 个槽是被证据门 G1–G4 逐个校验过的、报告正在用的东西;
新链要先证明自己能站住,再去动它。替换是 **M4.5**,单独一片、单独验收
(判据:删掉旧槽后,报告里没有一句不成立的陈述)。

## 本轮的真实产出是什么

**大部分指标会报"算不出来",而且这是对的。** 以 `20260926_153854_1592` 为例:

* 该场**没有 face / gesture 日志**(原始媒体在盘上,但日志不在);
* 16 条里 9 条卡标定(τ 只能来自 M5 的标定集);
* 2 条 (`micro_exp_*`) 上游根本没采集。

⟹ 这一轮的价值不在"算出多少数",而在**把"算不出来"变成一个可查询、可统计、
不可能被悄悄填成 0 的记录**。等 10-16 那份摘要说的服务器侧日志齐了,
同一条链会自己给出更多的数 —— 不需要改代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from report_frontend import (event_windows, l1_face, l1_gesture, l1_quality,
                             l1_text, l1_voice, session_timeline, transcript_chars,
                             voice_series)
from report_frontend.l1_indicators import L1Indicator


@dataclass
class L1Result:
    session_id: str
    timebase: session_timeline.SessionTimeline
    windows: event_windows.WindowSet
    indicators: list[L1Indicator] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    #: 每个模态的**实际**可用行占比 —— "日志存在" ≠ "数据存在"。
    coverage: dict = field(default_factory=dict)

    def by_name(self, name: str) -> list[L1Indicator]:
        return [i for i in self.indicators if i.name == name]

    @property
    def n_measured(self) -> int:
        return sum(1 for i in self.indicators if i.has_value)

    def to_dicts(self) -> list[dict]:
        return [i.to_dict() for i in self.indicators]


#: 段间缺口超过这个秒数,就认为发生过**一次录制重启**(而不是丢了几帧)。
#: 依据:13 场 A 档里单段场的最大相邻间隔是 1~2 秒;拼接场的缺口是 26~105 秒,
#: 两者差一个数量级,30 秒这个界不敏感。
SEGMENT_GAP_SEC = 30.0


def capture_segments(t_sec) -> list[tuple[float, float]]:
    """把一条时间轴切成"录制段",返回 `[(起, 止), ...]`(升序、含首尾)。

    为什么要切:一场访谈可能由**多段录制**拼成(客户端每段结束就整文件重传并覆盖
    `media/camera.webm`,盘上只剩最后一段)。不切的话,"这场有多长"会被算成
    首末帧之差,把重启的空档也算进去,而不同族的观察窗口会悄悄不一样长。

    空输入返回 `[]`;单段输入返回 `[(首, 尾)]`。
    """
    import numpy as np
    t = np.sort(np.asarray(t_sec, dtype=float))
    t = t[np.isfinite(t)]
    if t.size == 0:
        return []
    breaks = np.where(np.diff(t) > SEGMENT_GAP_SEC)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [t.size - 1]))
    return [(float(t[a]), float(t[b])) for a, b in zip(starts, ends)]


def segment_gap_note(segs: list[tuple[float, float]],
                     audio_sec: float | None) -> str | None:
    """拼接场/窗口不一致的说明;**没有问题就返回 `None`**。

    抽成纯函数是为了能在**不碰 `data/`** 的前提下测措辞与判断 ——
    `compute_session_l1` 本身要真实日志才能跑。
    """
    if not segs:
        return None
    total = sum(b - a for a, b in segs)
    if len(segs) > 1:
        idle = (segs[-1][1] - segs[0][0]) - total
        msg = (f"本场由 {len(segs)} 段录制拼接而成(段间缺口合计 {idle:.0f} 秒,"
               f"各段合计 {total:.0f} 秒);F/G 两族覆盖全部 {len(segs)} 段")
        if audio_sec and total > 0 and not 0.8 <= total / audio_sec <= 1.2:
            msg += (f",而盘上的 camera.webm 只保留了最后一段({audio_sec:.1f} 秒)"
                    f",V/T 两族因此只覆盖最后 {audio_sec / 60:.1f} 分钟 —— "
                    "同一场里不同族的观察窗口不同,不可互相印证")
        return msg + "。"
    if audio_sec and total > 0 and not 0.8 <= total / audio_sec <= 1.2:
        return (f"F/G 两族覆盖 {total:.0f} 秒,而盘上音频只有 {audio_sec:.1f} 秒"
                f"(比 {total / audio_sec:.2f})—— V/T 两族的观察窗口与 F/G 不同,"
                "不可互相印证")
    return None


def _pick_voice(logs: dict):
    """语音日志在三个键名之间挑一个。

    ⚠️ 不能用 `a or b`:DataFrame 的 `__bool__` 会抛 `ValueError`(踩过)。
    """
    for key in ("voice", "voice_interview", "voice_research"):
        df = logs.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    return None


def frames_outside_note(outside: dict[str, int], totals: dict[str, int],
                        frac: float = 0.05) -> str | None:
    """落在所有回答窗口之外的帧**多到值得说一句**时的说明,否则 None。

    ## 为什么需要它

    有台账时原点 = `min(ask_start)`,而第一个窗口从第一个 `ask_end` 才开始
    ⟹ 提问期(以及首题之前的一切帧)**不属于任何窗口**,`assign_series` 对它们
    静默返回 `None`。这会让 F/G 两族的比率类分母悄悄变小。

    `WindowSet.count_outside` 只负责**数**;要不要为此占一行报告,由本函数决定。
    低于 `frac`(默认 5%)只进载荷,不占篇幅 —— 一两条帧不值得加一行。

    纯函数:不读盘、不看时间轴,所以能直接单测(与 `segment_gap_note` 同一做法)。
    """
    bad = {n: (k, totals[n]) for n, k in outside.items()
           if totals.get(n) and k / totals[n] > frac}
    if not bad:
        return None
    return ("有帧落在所有回答窗口之外(提问期与首题之前不属于任何窗口):"
            + "、".join(f"{n} {k}/{t} 帧({k / t:.0%})"
                        for n, (k, t) in sorted(bad.items())))


def compute_session_l1(session_id: str,
                       logs: dict[str, pd.DataFrame] | None = None,
                       session_end_sec: float | None = None) -> L1Result:
    """一场会话的全部 L1 指标。

    `logs`:"模态 → 日志 DataFrame"。缺哪个模态就按"该模态缺失"处理,
    **不去 `data/` 里自己找** —— 读取口径只有 `LogDataLoader` 一处。
    """
    logs = {k: v for k, v in (logs or {}).items()
            if isinstance(v, pd.DataFrame) and not v.empty}

    # 正式场没有提问台账,但帧账本齐全 ⟹ **显式**允许用首帧墙钟当原点,
    # 否则整场一条指标都出不来(实测 24/24 场如此)。戊-15 的裁定是
    # "默认不建轴",这里是一次**有意的例外**,由调用方明确要求。
    tl = session_timeline.build_timeline(session_id, logs=logs,
                                         allow_frame_origin=True)
    gaps: list[str] = []
    if tl.gap:
        gaps.append(tl.gap)

    if tl.origin_wall is None:
        # 没有时间原点 ⟹ 窗口与指标都无从谈起。仍然把 F 族的 16 条以 `session` 名义
        # 报成"不可出",免得读的人以为这一场什么都没算。
        ws = event_windows.WindowSet(session_id, event_windows.SESSION_GRANULARITY,
                                     [], gap=tl.gap or "无会话原点")
        # ★ T 族**不依赖时间轴** —— 它只读转写文本。所以哪怕没有会话原点
        #   (正式场的常态),这三条照样算得出来。放在早退分支里,否则会白丢。
        inds = (l1_face.compute_face_indicators(None, ws, None)
                + l1_gesture.compute_gesture_indicators(None, ws, None)
                + l1_voice.compute_voice_indicators(_pick_voice(logs))
                + l1_text.compute_text_indicators(
                    transcript_chars.text(session_id)))
        if transcript_chars.text(session_id) is None:
            gaps.append("本场没有 ASR 转写 —— T 族 3 条全部不可出"
                        "(可跑 tools/jx_transcribe_recordings.py 生成)")
        return L1Result(session_id, tl, ws, inds, gaps,
                        coverage=l1_quality.modality_coverage(logs))

    face = logs.get("face")
    face_tb = tl.modalities.get("face")
    if session_end_sec is None:
        # 右边界取本场所有可用模态里最晚的那个观测 —— **不猜**一个固定时长。
        ends = [float(tb.t_sec.max()) for tb in tl.modalities.values()
                if tb.usable and len(tb.t_sec)]
        session_end_sec = max(ends) if ends else 0.0

    wall_ok = all(tb.status is not session_timeline.TimebaseStatus.WALL_UNRELIABLE
                  for tb in tl.modalities.values())
    ws = event_windows.build_windows(session_id, tl.origin_wall,
                                     session_end_sec=session_end_sec,
                                     wall_reliable=wall_ok)
    if ws.gap:
        gaps.append(ws.gap)

    # ★ 落在所有窗口之外的帧数 —— 只有逐题窗口才可能有。
    #
    # 有台账时原点 = `min(ask_start)`,而第一个窗口从第一个 `ask_end` 才开始,
    # 所以提问期(以及首题之前的一切帧)**不属于任何窗口**。`assign_series` 对它们
    # 静默返回 `None`:不数就等于分母悄悄变小而没人知道。这里逐模态数出来,
    # 超过 5% 再明说一句 —— 低于 5% 只进载荷,不占报告篇幅。
    for _name in ("face", "gesture"):
        _tb = tl.modalities.get(_name)
        if _tb is not None and _tb.usable and len(_tb.t_sec):
            ws.frames_outside[_name] = ws.count_outside(_tb.t_sec)
    if ws.frames_outside:
        _note = frames_outside_note(
            ws.frames_outside,
            {n: int(len(tl.modalities[n].t_sec)) for n in ws.frames_outside
             if n in tl.modalities})
        if _note:
            gaps.append(_note)

    inds: list[L1Indicator] = []

    # ── F 族(面部)──────────────────────────────────────────────────────
    if face is None or face_tb is None or not face_tb.usable:
        inds.extend(l1_face.compute_face_indicators(None, ws, None))
        gaps.append("本场没有可用的 face 日志 —— F 族 16 条全部不可出")
    else:
        inds.extend(l1_face.compute_face_indicators(face, ws, face_tb.t_sec))

    # ── G 族(手势)──────────────────────────────────────────────────────
    gest = logs.get("gesture")
    gest_tb = tl.modalities.get("gesture")
    if gest is None or gest_tb is None or not gest_tb.usable:
        inds.extend(l1_gesture.compute_gesture_indicators(None, ws, None))
        gaps.append("本场没有可用的 gesture 日志 —— G 族 7 条全部不可出")
    else:
        inds.extend(l1_gesture.compute_gesture_indicators(gest, ws, gest_tb.t_sec))
        # ★ 拼接场守卫:一场访谈可能是**多段录制**拼起来的。
        #
        # 实测(2026-10-18,13 场 A 档):12 场是单段,但**王语桐一场是 9 段**。
        # 查证过程与结论:
        #   · `retention.camera.jsonl` 有 **9 条**记录,全指向同一个 `media/camera.webm`、
        #     `seq` 都是 1,但 9 个不同的大小与 sha256 ⟹ 每次都是**整文件重传并覆盖**;
        #   · 9 次上传都紧跟在某一段结束后的 26~87 秒;
        #   · 每份文件字节数 ÷ 该段时长恒为 **~315 KB/s** ⟹ 每次上传的就是**该段**的完整录像;
        #   · 于是盘上只剩**最后一段**:camera.webm 实测 434.7 秒(视频 13029 包 / 音频 7245 包)。
        # ⟹ 这一场的访谈真实时长是 9 段合计 6959 秒(124 分钟,段间缺口另 468 秒),
        #   而 V/T 两族(读转写)只覆盖最后 7.2 分钟。**同一场里不同族的观察窗口不同**,
        #   这是采集缺陷,不是计算问题;`session_timeline` 对它报 OK、drift −0.002,不会告警。
        #
        # ⚠️ 注意:两条手势率量的分母**不是**日志跨度,而是 `n_total × 帧间隔`
        #   = 观测到的帧数(本场 6968 秒)—— 段间缺口本来就没算进分母。真正的问题是
        #   "9 段拼接的率"与"其余场单段的率"不可比,以及 V/T 只看得到最后一段。
        _note = segment_gap_note(capture_segments(gest_tb.t_sec),
                                 transcript_chars.audio_seconds(session_id))
        if _note:
            gaps.append(_note)

    # ── V 族(语音)──────────────────────────────────────────────────────
    # 语音日志一行 = 一段回答 ⟹ 不需要时间轴,只需要"有没有这份日志"。
    # `LogDataLoader` 把语音叫 `voice_interview` / `voice_research` —— 三个键都收,
    # 否则会报"没有 voice 日志"而其实有(踩过,12 条全被误判成 modality_missing)。
    voice = _pick_voice(logs)
    # 序列是**派生缓存**:读不到就返回 None,那 8 条退回 `not_stored`(不猜)。
    series = voice_series.load_series(session_id)
    # 语音**日志**没有、但整场录音重算出的序列在 ⟹ voice 模态其实在。
    # 补一条时间轴,否则质量报告的硬不变量会(正确地)拦下"有值但模态不存在"。
    if "voice" not in tl.modalities and series:
        tb_voice = session_timeline.audio_timebase(series)
        if tb_voice is not None:
            tl.modalities["voice"] = tb_voice
            gaps.append("voice 模态来自原始音频重算(日志 0 行),时间轴未与墙钟核对")
    inds.extend(l1_voice.compute_voice_indicators(voice, series=series,
                                                          session_id=session_id))
    if voice is not None and series is None:
        gaps.append("语音日志在,但没有帧级序列缓存 —— V 族 8 条序列指标报 not_stored"
                    "(可跑 report_frontend/voice_series.extract_session 生成)")
    if voice is None:
        gaps.append("本场没有 voice 日志 —— V 族 12 条全部不可出")

    # ── T 族(文本)—— 只依赖转写,与三个模态的日志无关 ──────────────────
    inds.extend(l1_text.compute_text_indicators(
        transcript_chars.text(session_id)))
    if transcript_chars.text(session_id) is None:
        gaps.append("本场没有 ASR 转写 —— T 族 3 条全部不可出"
                    "(可跑 tools/jx_transcribe_recordings.py 生成)")

    return L1Result(session_id, tl, ws, inds, gaps,
                    coverage=l1_quality.modality_coverage(logs))
