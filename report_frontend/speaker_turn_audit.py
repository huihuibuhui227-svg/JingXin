"""问答轮次与受试者声学分离核验（纯 CPU 确定性模块，零黑盒打分）。

解决痛点：
1. 区分本场语音数据是「逐题独立作答录音（已天然剥离提问者声音）」、
   「整场录音按提问台账切分作答窗」，还是「整场连续混音且无提问台账」；
2. 利用 `questions.jsonl` 的 `ask_start` / `ask_end` 统计提问时段与受试者作答时段；
3. 当存在帧级基频序列（`l0_series`）时，用纯 numpy 双簇基频分布检验声学音域分离度，
   给出直白、无术语堆砌的一句话说明。
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

import session_meta
from . import voice_series


def _estimate_f0_bimodality(f0: np.ndarray) -> dict[str, Any] | None:
    """对有效基频序列（Hz）执行确定性一维双簇划分，检验是否存在明显双说话人音域。"""
    valid = f0[np.isfinite(f0) & (f0 > 55.0) & (f0 < 450.0)]
    if valid.size < 40:
        return None
    med = float(np.median(valid))
    low = valid[valid <= med]
    high = valid[valid > med]
    if low.size < 10 or high.size < 10:
        return None
    c1, c2 = float(np.median(low)), float(np.median(high))
    for _ in range(8):
        mid = 0.5 * (c1 + c2)
        g1 = valid[valid <= mid]
        g2 = valid[valid > mid]
        if g1.size < 8 or g2.size < 8:
            break
        c1, c2 = float(np.median(g1)), float(np.median(g2))
    sep_semitones = float(12.0 * np.log2(max(c2, 1e-6) / max(c1, 1e-6)))
    return {
        "low_cluster_hz": round(c1, 1),
        "high_cluster_hz": round(c2, 1),
        "separation_semitones": round(sep_semitones, 2),
        "distinct_registers": sep_semitones >= 4.5,
    }


def audit_speaker_turns(
    session_id: str | None,
    logs: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """返回本场问答轮次与说话人声学隔离状态（100% 确定性输出，不含墙钟耗时浮点）。"""
    logs = logs or {}
    v_int = logs.get("voice_interview")
    v_res = logs.get("voice_research")
    v_any = logs.get("voice")

    n_qa_rows = 0
    for df in (v_int, v_res, v_any):
        if isinstance(df, pd.DataFrame) and not df.empty:
            n_qa_rows = max(n_qa_rows, len(df))

    questions: list[dict] = []
    if session_id:
        try:
            questions = session_meta.read_questions(session_id) or []
        except Exception:
            questions = []

    prompt_sec = 0.0
    answer_Gap_sec = 0.0
    valid_q = [
        q for q in questions
        if q.get("ask_start") is not None and q.get("ask_end") is not None
    ]
    valid_q.sort(key=lambda q: float(q["ask_start"]))
    for idx, q in enumerate(valid_q):
        s = float(q["ask_start"])
        e = float(q["ask_end"])
        if e >= s:
            prompt_sec += (e - s)
        if idx + 1 < len(valid_q):
            nxt_s = float(valid_q[idx + 1]["ask_start"])
            if nxt_s > e:
                answer_Gap_sec += (nxt_s - e)

    series = voice_series.load_series(session_id) if session_id else None
    reg_info = None
    if series is not None and hasattr(series, "f0") and series.f0 is not None:
        try:
            reg_info = _estimate_f0_bimodality(np.asarray(series.f0, dtype=float))
        except Exception:
            reg_info = None

    n_turns = max(n_qa_rows, len(valid_q))
    if n_qa_rows > 0:
        mode = "qa_isolated"
        mode_label = "逐题独立录音"
        summary_plain = (
            f"本场采用逐题独立提交录音，共记录 {n_turns} 轮回答，已直接隔离提问时段，声学指标仅反映受试者本人发声。"
        )
    elif len(valid_q) > 0:
        mode = "window_segmented"
        mode_label = "按提问时间窗切分"
        summary_plain = (
            f"本场已记录 {len(valid_q)} 轮提问时间戳，提问时段合计 {prompt_sec:.0f} 秒、题间作答窗口合计 {answer_Gap_sec:.0f} 秒，视觉与行为窗口已按题目边界对齐。"
        )
    else:
        mode = "mixed_unsegmented"
        mode_label = "整场连续采集"
        if reg_info and reg_info.get("distinct_registers"):
            summary_plain = (
                f"本场为整场连续录音且未记录逐题提问时间戳；基频分布检出高低两个音域簇（中位 {reg_info['low_cluster_hz']} Hz 与 {reg_info['high_cluster_hz']} Hz，跨度 {reg_info['separation_semitones']} 半音），语速与发声占比可能包含提问者话语。"
            )
        else:
            summary_plain = "本场为连续采集且未记录逐题时间戳，全部指标按整场单一时段统计。"

    return {
        "session_id": session_id,
        "isolation_mode": mode,
        "mode_label": mode_label,
        "n_turns": int(n_turns),
        "n_question_windows": int(len(valid_q)),
        "n_answer_audio_rows": int(n_qa_rows),
        "prompt_window_sec": round(prompt_sec, 1),
        "candidate_gap_sec": round(answer_Gap_sec, 1),
        "vocal_register_clusters": reg_info,
        "summary_plain": summary_plain,
    }
