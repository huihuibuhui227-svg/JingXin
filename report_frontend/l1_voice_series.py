# report_frontend/l1_voice_series.py
"""**V 族里需要帧序列的 8 条指标** —— 从 `voice_series` 的逐帧序列算出。

## 每一条的公式(写死在这里,别处不许再算一遍)

| 指标 | 公式 | 单位 |
|---|---|---|
| `voiced_ratio` | 浊音帧数 ÷ 总帧数(**跨回答汇总**) | 无量纲 |
| `pause_duration_p50` | 停顿区间时长的 p50(**跨回答汇总**) | 秒 |
| `pause_duration_p90` | 同上,p90 | 秒 |
| `pitch_median` | 浊音帧 `f0` 的中位数(跨回答汇总) | Hz |
| `pitch_range_semitone` | **每题** `12·log2(f0_首 ÷ f0_末)`,再取跨题中位数 | 半音 |
| `pitch_variability_per_sec` | **每题** `Σ|Δ半音|` ÷ 浊音秒,再取跨题中位数 | 半音/秒 |
| `energy_median` | `rms` 的中位数(跨回答汇总) | 无量纲 |
| `energy_dynamics_per_sec` | **每题** `Σ|Δrms|` ÷ 浊音秒,再取跨题中位数 | 1/秒 |

## 三处必须说清的口径

1. **停顿 = 连续非浊音 ≥ `MIN_PAUSE_SEC`(0.2 s)。** 这是**定义**不是标定;
   更短的间隙是音节过渡。改这个常量会改变全部 `pause_*` 读数,所以它住在
   `voice_series` 里并被显式登记。
2. **`pitch_range_semitone` 用"首末"是规格原文**,但首末各只有一帧 ⟹ 噪声大。
   本模块照规格实现,**不偷偷换成 p10/p90** —— 那是另一个统计量,换了就名不副实。
   若日后要改,应当改规格并改名字,而不是在这里"优化"。
3. **分母用"浊音秒"而不是"整段秒"**:音高与能量只在发声时才有定义;
   拿整段(含静音)当分母会让"说得少"看起来"变化小"。

## 为什么"汇总"与"跨题中位数"混用

`voiced_ratio` / `pause_*` / `pitch_median` / `energy_median` 是**分布量**,
按帧/区间汇总才是无偏的(先算每题再平均会让短答与长答等权)。
`pitch_range_semitone` / `*_per_sec` 是**每题一个的率量**,规格的单位就是"每题一次测量",
所以先每题算、再取中位数(抗单题异常),`n_valid` 也因此是**回答数**。
"""

from __future__ import annotations

import numpy as np

from report_frontend.l1_indicators import (
    GAP_NOT_STORED, L1Indicator, measured,
)

#: 本模块负责的 8 条。其余 4 条由 `l1_voice` 从语音日志的聚合列算出。
SERIES_NAMES = (
    "voiced_ratio",
    "pause_duration_p50",
    "pause_duration_p90",
    "pitch_median",
    "pitch_range_semitone",
    "pitch_variability_per_sec",
    "energy_median",
    "energy_dynamics_per_sec",
)

_SPECS = {
    "voiced_ratio": ("V1_amount", "浊音帧数 ÷ 总帧数", "无量纲", "speech", 2),
    "pause_rate_per_min": ("V2_pause",
                           "停顿区间次数 ÷ 有声分钟(逐帧 pyin;停顿 = 连续非浊音 ≥0.2s,"
                           "含首尾静音)", "次/分", "pause", 2),
    "pause_duration_p50": ("V2_pause", "停顿区间时长 p50", "秒", "pause", 2),
    "pause_duration_p90": ("V2_pause", "停顿区间时长 p90", "秒", "pause", 2),
    "pitch_median": ("V3_pitch", "浊音帧 f0 中位数", "Hz", "pitch", 5),
    "pitch_range_semitone": ("V3_pitch", "12·log2(f0首 ÷ f0末)", "半音", "pitch", 5),
    "pitch_variability_per_sec": ("V3_pitch", "Σ|Δ半音| ÷ 浊音秒", "半音/秒", "pitch", 5),
    "energy_median": ("V4_energy", "rms 中位数", "无量纲", "energy", 5),
    "energy_dynamics_per_sec": ("V4_energy", "Σ|Δrms| ÷ 浊音秒", "1/秒", "energy", 5),
}


def pause_runs(voiced: np.ndarray, frame_sec: float,
               min_pause_sec: float) -> list[float]:
    """连续非浊音段的时长(秒),只保留 ≥ `min_pause_sec` 的。

    **首尾的静音也算停顿** —— 与 `prosody_extractor` 里那段"尾部静音整个丢掉"
    的修正同一判据(那里是 2026-09-27 修过的坑)。开头那段前导静音同样保留:
    它是"开口前的沉默",属于反应延迟的一部分,不该在这里被吃掉。
    """
    if voiced.size == 0 or frame_sec <= 0:
        return []
    out: list[float] = []
    run = 0
    for v in voiced:
        if v:
            if run:
                d = run * frame_sec
                if d >= min_pause_sec:
                    out.append(d)
                run = 0
        else:
            run += 1
    if run:
        d = run * frame_sec
        if d >= min_pause_sec:
            out.append(d)
    return out


def _semitone(f0: np.ndarray) -> np.ndarray:
    return 12.0 * np.log2(f0)


def compute_from_series(series, thresholds: dict[str, int] | None = None,
                        n_total: int = 0) -> dict[str, L1Indicator]:
    """返回 `{name: L1Indicator}`。没有序列 ⟹ 全部 `not_stored`。"""
    thr = thresholds or {}
    if not series:
        return {n: _un(n, GAP_NOT_STORED, n_total) for n in SERIES_NAMES}

    from report_frontend.voice_series import MIN_PAUSE_SEC

    usable = [s for s in series if s.f0.size > 0]
    if not usable:
        return {n: _un(n, GAP_NOT_STORED, n_total) for n in SERIES_NAMES}

    # ★ 样本单位随素材走,不能一律当"题"。
    #
    # · **逐题音频**(有 `questions.jsonl` 的场次):一条序列 = 一段回答,
    #   汇总量的样本单位就是"题",与 `evidence_thresholds.json` 的
    #   `n_valid_units=answer` 一致。
    # · **整场录音**(正式场没有题目边界,见 `voice_series.whole_session_audio`):
    #   根本没有"题"这个单位。此时汇总量的样本是**帧**,`n_valid` 就报帧数 ——
    #   报成 1 会把"几千帧的分布"说成"一次观测",那是**少报**证据;
    #   而按"题"算的率量(首末差、每秒变化)确实只有 1 次测量,如实报 1。
    whole = all(s.index is None for s in usable)
    unit_suffix = "帧" if whole else "题"

    out: dict[str, L1Indicator] = {}

    voiced_total = sum(int(s.voiced_flag.sum()) for s in usable)
    frames_total = sum(int(s.f0.size) for s in usable)
    if frames_total > 0:
        out["voiced_ratio"] = measured(
            "voiced_ratio", *_SPECS["voiced_ratio"][:2], ("f0", "voiced_flag"),
            "session", _SPECS["voiced_ratio"][2],
            value=voiced_total / frames_total,
            n_valid=frames_total if whole else len(usable),
            n_total=frames_total if whole else (n_total or len(usable)),
            threshold=thr.get("speech", 2))

    # ── 停顿分位数 ────────────────────────────────────────────────────────
    runs: list[float] = []
    for s in usable:
        runs.extend(pause_runs(s.voiced_flag, s.frame_sec, MIN_PAUSE_SEC))
    voiced_sec_total = sum(float(s.voiced_sec) for s in usable)

    # ── 停顿率 ────────────────────────────────────────────────────────────
    #
    # ⚠️ **`pause_ratio` 不做**(仍报不可出)。实测 13 场 A 档:
    #
    #     pause_ratio  vs  1/voiced_ratio − 1        r = 0.9990
    #
    #   按"全部非语音"算,pause_ratio 就是 `1/voiced_ratio − 1`,**恒等变换**;
    #   按 ≥0.2s 的停顿区间算,与那个恒等式的平均差也只有 0.0585,r 仍 0.999。
    #   两条路都是 `voiced_ratio` 的单调变换 ⟹ 对读者零额外信息。
    #   本仓对"零额外信息"的东西一向是**标出来**而不是留着(见 `evidence_gate`
    #   里 `speech_ratio` 那几条),所以这里不产出它。
    #
    #   `pause_rate_per_min` 不同量:它是"每分钟开口多少次",与最近的既有指标
    #   `pause_duration_p50` 的 r 只有 0.549(与 voiced_ratio 是 −0.866,
    #   相关但不恒等)。⟹ 保留,但把相关性记在这里,免得下一个人以为它独立。
    # ★ `pause_rate_per_min` **故意不在 `SERIES_NAMES` 里**:它是一条**两边都能算**的
    #   指标 —— 有逐题语音日志时走日志侧(`pause_frequency` 列),没有日志时走这里。
    #   放进 `SERIES_NAMES` 会让日志侧把它排除掉(`l1_voice` 按
    #   `n not in SERIES_NAMES` 建日志侧那几条),于是**有日志的场次反而丢掉它**。
    #   `compute_from_series` 的调用方是 `l1_voice` 的无日志分支,返回多一个键无妨。
    if voiced_sec_total > 0:
        out["pause_rate_per_min"] = measured(
            "pause_rate_per_min", *_SPECS["pause_rate_per_min"][:2],
            ("f0", "frame_sec"), "session", "次/分",
            value=len(runs) / (voiced_sec_total / 60.0),
            # 率量按"题"计:整场录音没有题目边界 ⟹ 只有 1 次测量,如实报 1。
            n_valid=len(usable), n_total=n_total or len(usable),
            threshold=thr.get("pause", 2))
    if runs:
        # ★ `n_valid` 报**停顿区间数**,不是帧数。
        #
        # 这两个分位数是从 `runs`(停顿区间)算的 —— 一条区间就是一个样本。
        # 报帧数是**把证据说多**:实测场 `20260928_180131_8102` 报 `n_valid=13585`
        # (帧数),而 p50 只建立在 **142 个停顿区间**上,两者差两个数量级。
        # 读者看到 13585 会以为这条分位数有上万次观测支撑。
        #
        # 与 `hand_visible_ratio`(387 帧里 1 帧有值却报 n_valid=387)同源,
        # 也与本仓对率量的既有裁定一致(见 `l1_gesture`:率的证据单位是"段"不是"帧")。
        # `n_total` 仍报帧数 —— 分母是"看了多少帧",这一点没错。
        n_pause = len(runs)
        out["pause_duration_p50"] = measured(
            "pause_duration_p50", *_SPECS["pause_duration_p50"][:2],
            ("f0", "frame_sec"), "session", "秒",
            value=float(np.percentile(runs, 50)),
            n_valid=n_pause, n_total=frames_total,
            threshold=thr.get("pause", 2))
        out["pause_duration_p90"] = measured(
            "pause_duration_p90", *_SPECS["pause_duration_p90"][:2],
            ("f0", "frame_sec"), "session", "秒",
            value=float(np.percentile(runs, 90)),
            n_valid=n_pause, n_total=frames_total,
            threshold=thr.get("pause", 2))

    # ── 音高 ──────────────────────────────────────────────────────────────
    pooled = np.concatenate([s.f0[s.voiced_flag] for s in usable
                             if s.voiced_flag.any()]) if voiced_total else np.array([])
    if pooled.size:
        out["pitch_median"] = measured(
            "pitch_median", *_SPECS["pitch_median"][:2], ("f0",), "session", "Hz",
            value=float(np.median(pooled)),
            n_valid=frames_total if whole else len(usable),
            n_total=frames_total if whole else (n_total or len(usable)),
            threshold=thr.get("pitch", 5))

    # 每题一个的率量:先每题算,再取中位数
    per_answer_range: list[float] = []
    per_answer_pvar: list[float] = []
    per_answer_edyn: list[float] = []
    for s in usable:
        f = s.f0[s.voiced_flag]
        if f.size >= 2:
            ratio = f[0] / f[-1] if f[-1] > 0 else np.nan
            if np.isfinite(ratio) and ratio > 0:
                per_answer_range.append(float(12.0 * np.log2(ratio)))
            semi = _semitone(f)
            sec = s.voiced_sec
            if sec > 0:
                per_answer_pvar.append(float(np.abs(np.diff(semi)).sum() / sec))
        r = s.rms[np.isfinite(s.rms)]
        if r.size >= 2 and s.voiced_sec > 0:
            per_answer_edyn.append(float(np.abs(np.diff(r)).sum() / s.voiced_sec))

    if per_answer_range:
        out["pitch_range_semitone"] = measured(
            "pitch_range_semitone", *_SPECS["pitch_range_semitone"][:2], ("f0",),
            "session", "半音", value=float(np.median(per_answer_range)),
            n_valid=len(per_answer_range), n_total=n_total or len(usable),
            threshold=thr.get("pitch", 5))
    if per_answer_pvar:
        out["pitch_variability_per_sec"] = measured(
            "pitch_variability_per_sec", *_SPECS["pitch_variability_per_sec"][:2],
            ("f0", "frame_sec"), "session", "半音/秒",
            value=float(np.median(per_answer_pvar)),
            n_valid=len(per_answer_pvar), n_total=n_total or len(usable),
            threshold=thr.get("pitch", 5))

    # ── 能量 ──────────────────────────────────────────────────────────────
    all_rms = np.concatenate([s.rms[np.isfinite(s.rms)] for s in usable])
    if all_rms.size:
        out["energy_median"] = measured(
            "energy_median", *_SPECS["energy_median"][:2], ("rms",), "session",
            "无量纲", value=float(np.median(all_rms)),
            n_valid=frames_total if whole else len(usable),
            n_total=frames_total if whole else (n_total or len(usable)),
            threshold=thr.get("energy", 5))
    if per_answer_edyn:
        out["energy_dynamics_per_sec"] = measured(
            "energy_dynamics_per_sec", *_SPECS["energy_dynamics_per_sec"][:2],
            ("rms", "frame_sec"), "session", "1/秒",
            value=float(np.median(per_answer_edyn)),
            n_valid=len(per_answer_edyn), n_total=n_total or len(usable),
            threshold=thr.get("energy", 5))

    for n in SERIES_NAMES:
        if n not in out:
            out[n] = _un(n, GAP_NOT_STORED, n_total)
    return out


def _un(name: str, gap: str, n_total: int) -> L1Indicator:
    from report_frontend.l1_indicators import unavailable
    family, formula, unit, _fam, _thr = _SPECS[name]
    return unavailable(name, family, formula, (), "session", unit, gap, n_total)
