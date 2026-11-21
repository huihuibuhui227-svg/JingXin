# report_frontend/l1_voice.py
"""**V 族(语音)11 个 L1 指标** —— 其中 3 条能按规格从日志原样产出(2026-11-22 起)。

> ★ **2026-11-22:退掉 `pause_ratio`,V 族由 12 条变 11 条。**
> 它与 `voiced_ratio` 是**代数恒等变换**,本仓 2026-10-24 已裁定"不重复产出"
> (`not_produced`)。复核(2026-11-22,10 场有 `duration_sec`/`speech_duration_sec`
> 的语音日志):
>
> ```
> pause_ratio − (1/voiced_ratio − 1)        最大绝对差 4.441e-16   ← 机器精度
> Spearman ρ(pause_ratio, voiced_ratio)     −1.000000 (n=10, p=6.7e-64)
> ```
>
> 也就是说 `pause_ratio` 不是"另一条测量",而是 `(speech_duration_sec, duration_sec)`
> 这两列的一个换算。留着它占"38 条"里的一格,读者会以为多了一条独立证据。
>
> **那条"为什么不产出"的诚实解释去哪了** —— 三处,逐处交代:
> 1. `gap_explanations.SPECIAL["pause_ratio"]["not_produced"]`(B2 写的专属句)
>    **原样留在那个文件里**。本文件不动它(那是并发改动中的文件),它从此**命中不到
>    任何指标** —— 由交付说明登记为"待清理的死条目",不在这里偷偷改另一份登记表。
> 2. 解释的**内容**搬到了本模块的 `RETIRED["pause_ratio"]`(下面),代码里仍可复核;
> 3. 报告"这套采集不测"那一节**装不下它**:那一节逐行来自**14 个槽**的删槽记录
>    (`DROP_REASON`),而 `pause_ratio` 从来不是 14 槽之一 —— 硬塞进去等于凭空
>    发明一个槽(本仓明令禁止)。所以它的去向是"随指标一起退",审计线索留在代码里。


## 为什么只有 4 条:粒度错配,不是标定问题

设计文档 §5.4 的 V 族 12 条,**大部分定义在音频帧上**("真 VAD 判为语音的帧占比"、
"首末差换成半音"、`pitch_variability_per_sec`、`energy_dynamics_per_sec`)。
而语音日志 `interview_emotion_log_*.csv` 的粒度是 **一段回答一行、只存聚合量**:

```
pitch_mean / pitch_variation(=std) / pitch_p10 / pitch_p90 / pitch_trend
energy_mean / energy_variation / energy_p10 / energy_p90
pause_duration_mean / pause_duration_max / pause_frequency
speech_duration_sec / duration_sec / chars_per_sec / reaction_time
```

⟹ 帧级序列**没有落盘**。所以:

| # | 指标 | 能否产出 | 依据 |
|---|---|---|---|
| 1 | `speech_rate` | ✅ 精确 | `chars_per_sec` 就是 `n_chars ÷ 有声秒`(`replay_contract.py:35`) |
| 2 | `voiced_ratio` | ✅ 序列 | `voiced_flag` 的帧占比(M4.9 从原始音频重算) |
| 3 | `pause_rate_per_min` | ✅ 精确 | `pause_frequency` 的分母**已经是语音时长**(`prosody_extractor.py:283`) |
| 4 | `pause_duration_p50` | ✅ 序列 | 停顿区间的分位数(原提取器内部有列表,只是没存) |
| 5 | `pause_duration_p90` | ✅ 序列 | 同上 |
| ~~6~~ | ~~`pause_ratio`~~ | ⛔ **已退役**(2026-11-22) | 与 `voiced_ratio` 代数恒等(残差 4.4e-16、ρ = −1.000000)⟹ 重复证据,不再产出;见模块 docstring 与 `RETIRED` |
| 7 | `response_latency_sec` | ✅ 精确 | `reaction_time`(并可经 `clock_repair` 修正) |
| 8 | `pitch_median` | ✅ 序列 | `f0` 浊音帧的中位数 |
| 9 | `pitch_range_semitone` | ✅ 序列 | 浊音帧**首末**之差换成半音(规格原文) |
| 10 | `pitch_variability_per_sec` | ✅ 序列 | 逐帧半音差的绝对值和 ÷ 浊音秒 |
| 11 | `energy_median` | ✅ 序列 | `rms` 的中位数 |
| 12 | `energy_dynamics_per_sec` | ✅ 序列 | 逐帧 rms 差的绝对值和 ÷ 浊音秒 |

**这 8 条缺的是"序列没落盘"**,不是"标定没到位" —— 所以 `gap` 分开登记
(`not_stored` vs `uncalibrated`)。

2026-10-18 的处置:**不重写提取器,而是从原始音频重算序列**(`voice_series.py`)。
素材在盘上(`media/audio/*_converted.wav`,或整场 `camera.webm`),
`librosa.pyin` 本来就产出逐帧 `f0/voiced_prob`,`rms` 也是逐帧的 ——
只是原来算完就扔。重算一遍就能拿到序列,于是这 8 条**全部可产**。

没有序列时(没素材、或没跑过提取),这 8 条仍报 `not_stored` —— **不许**
用聚合量硬凑同名指标(名字一样、含义不同,比缺着更坏)。

## 聚合单位:会话级,不是逐回答

V 族的 `n_valid` 单位是 **answer**(`evidence_thresholds.json` 的 `n_valid_units`),
门槛按"整场有几段带值的回答"设(pitch=5 / energy=5 / speech=2 / pause=2 / reaction_time=5)。
所以本模块把整场的回答**聚合**成一条,`window="session"`、`n_valid=带值的回答数`。

逐回答的值当然存在,但一条回答只有 1 个样本,永远过不了门槛(1 < 2),
报出来只会是清一色"有效样本不足" —— 那是把门槛错配成噪声。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from report_frontend.l1_indicators import (
    GAP_MODALITY_MISSING, GAP_NO_SAMPLES, GAP_NOT_EXTRACTED,
    GAP_NOT_STORED, GAP_WINDOW_NONE, L1Indicator, measured, unavailable,
)

FAMILY = "V"

#: 规格承诺的 11 条(2026-11-22 起;原 12 条,退掉 `pause_ratio`),顺序即输出顺序。
_SPECS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "speech_rate": ("V1_amount", "字数 ÷ 语音秒数", ("n_chars", "speech_duration_sec")),
    "voiced_ratio": ("V1_amount", "真 VAD 判为语音的帧占比", ("voiced_flag",)),
    "pause_rate_per_min": ("V2_pause", "停顿次数 ÷ 语音分钟", ("pause_frequency",)),
    "pause_duration_p50": ("V2_pause", "停顿时长中位数", ("pause_duration",)),
    "pause_duration_p90": ("V2_pause", "停顿时长 p90", ("pause_duration",)),
    "response_latency_sec": ("V2_pause", "问题结束 → 首次开口", ("reaction_time",)),
    "pitch_median": ("V3_pitch", "音高中位数", ("pitch",)),
    "pitch_range_semitone": ("V3_pitch", "首末音高差(半音)", ("pitch",)),
    "pitch_variability_per_sec": ("V3_pitch", "音高每秒变化量", ("pitch_series",)),
    "energy_median": ("V4_energy", "能量中位数", ("energy",)),
    "energy_dynamics_per_sec": ("V4_energy", "能量每秒变化量", ("energy_series",)),
}

ORDER = tuple(_SPECS)

#: 能精确产出的 3 条(2026-11-22 起;`pause_ratio` 已退役)。
#: 其余一律 `not_stored`,理由见模块 docstring 的表。
_MEASURABLE = ("speech_rate", "pause_rate_per_min", "response_latency_sec")

#: **已退役指标** → 为什么退 + 那条"为什么不产出"的解释现在在哪。
#:
#: 为什么留这张表而不是把说明删干净:退掉一条指标时,读者最该保住的是
#: **"为什么它不在"这句话本身** —— 它被删掉之后,下一个人只会看到一个空缺,
#: 然后重新把它加回来(同一条恒等变换会被再犯一次)。
#: `tests/test_three_indicators_replaced.py` 拿它钉住"退役也是可解释的"。
RETIRED: dict[str, str] = {
    "pause_ratio": (
        "与发声占比是同一个数换了个算法:`pause_ratio ≡ 1/voiced_ratio − 1`"
        "(实测最大残差 4.4e-16,Spearman ρ = −1.000000,n=10)⟹ 重复证据,不再产出。"
        "专属解释原在 `gap_explanations.SPECIAL['pause_ratio']`(B2),"
        "该条目现已命中不到任何指标;内容留在这里可复核。"),
}

#: **序列缓存救不了**的日志侧指标 —— 各自的真实缺法不同,所以码也不同。
#:
#: * `response_latency_sec`:要「提问结束」时刻,而整场录音**没有题目边界**
#:   ⟹ 与「切不出回答窗口」是同一件事 ⟹ `window_none`。
#:
#: ★ 为什么必须分开(2026-10-24 实测踩到):把 31 场序列补齐之后,这两条**仍然**
#:   报 `not_extracted`,而前端的提示是「跑一次提取即可,不用重录」。提取已经跑过了
#:   —— 再跑一百次它们也出不来。判词比事实更狠的第二种形态:
#:   上一版说「查采集」(方向反了),这一版说「再跑一次提取」(永远跑不完)。
#:
#: ⚠️ **`pause_ratio` 的表项已随指标一起删掉**(2026-11-22):它曾是这里唯一的
#:   `not_produced` 用户;指标退役后留着这条登记只会让读者以为它还在产出口里。
#:   缺口码 `not_produced` 本身**保留在 `l1_indicators.KINDS`**(9 类不许增减)。
_NOT_EXTRACTABLE_GAP = {
    "response_latency_sec": GAP_WINDOW_NONE,
}


# ── 反应延迟报缺口时的**诊断事实**(只给措辞层选句,不改任何数、不改码)──────
#:
#: 2026-11-21 实测踩到:这条指标报 `window_none` 的 31 场里,病因**不是一种**:
#:
#: | 场次 | 题目台账 | 语音日志 | 真实病因 |
#: |---|---|---|---|
#: | 25 场(如 `20260928_180131_8102`) | 无 | 0 行 | 没走题库作答流程(只当录像机用) |
#: | 6 场(`a971`/`0dc7`/`253a`/`4579`/`90e7`/`87c9`) | **有** | **0 行** | 采集端没把逐题应答时刻落盘 |
#:
#: 而当时的文案对 31 场一律说「多数场次没存题目时刻表;补上题目时刻就能算」——
#: 对有台账的那 6 场是**当众说错话**:台账就在 `questions.jsonl` 里,补台账补不出
#: 一个"当时没落盘的应答时刻"。所以缺口照报、码照旧,但**要把实测事实一起带出去**。
#:
#: 三个键的含义(`None` = 这一层拿不到,不等于"否"):
#:
#: * `has_ledger`   —— 录像目录里 `questions.jsonl` 在不在;
#: * `voice_rows`   —— 实际加载到的语音日志行数(`len(voice)`);
#: * `rt_col_present` —— 语音日志里有没有 `reaction_time` 这一列。
#:
#: ⚠️ 读侧的硬事实(决定了文案能承诺什么):`compute_voice_indicators` **只读**
#: 日志里的 `reaction_time` 列,全仓没有任何报告路径用 `questions.jsonl` 重算它
#: (`clock_repair.reaction_times()` 存在但无人调用)。⟹ 台账在 ≠ 能算出:
#: 应答时刻是写侧(`voice_interaction/api/app.py`)当场落盘的,补不回来。
def _ledger_present(session_id: str | None) -> bool | None:
    """盘上有没有这一场的题目台账(`questions.jsonl`)。拿不到给 `None`。

    ## 为什么按 id 走 `media_retention.resolve_recording_dir`

    录像目录名有两种形态(纯 id 与 `<标签>__<短id>`),短 id 才是 `session_id`。
    自己拼 `root()/sid` 会在存过标注的场次上判错 —— 实测(`coverage_survey.py`
    的注释里记着)那会让 29/45 场取不到日志、覆盖率算成 9.2%。所以解析只认
    那一处实现(与 `session_meta` 同一套),这里不另写一份。

    ## 为什么 `None` 不并进 `False`

    "没有台账"是一个**实测结论**,要拿来印进正文;"问不到"(没有 `session_id`、
    id 形状不合法、读盘出错)是**没证据**。把后者说成前者,就是本仓一直在消灭的
    "判词比事实更狠"。措辞层对 `None` 退回不带因果判断的说法。
    """
    if not isinstance(session_id, str) or not session_id:
        return None
    try:
        import media_retention
        d = media_retention.resolve_recording_dir(session_id)
        return (d / "questions.jsonl").is_file()
    except Exception:  # noqa: BLE001 - 守卫/读盘出错按"拿不到"处理,别让报告炸掉
        return None


def _rt_gap_detail(voice: pd.DataFrame | None, session_id: str | None) -> dict:
    """反应延迟的诊断事实。**只记录实测到的**,`voice` 为 `None` 时行数记 0。"""
    n_rows = 0 if voice is None else int(len(voice))
    return {
        "has_ledger": _ledger_present(session_id),
        "voice_rows": n_rows,
        "rt_col_present": bool(voice is not None and "reaction_time" in voice.columns),
    }


def _col(df: pd.DataFrame, name: str) -> pd.Series | None:
    if name not in df.columns:
        return None
    return pd.to_numeric(df[name], errors="coerce")


#: `speech_rate` 走转写时的公式串。**必须**把"全场混音"写进去 ——
#: 音频没有说话人分离,字数是面试官 + 考生一起数的,量的是"这一场说话有多密",
#: 不是"考生语速"。不写清楚,读者会当考生语速用,那是无声地改构念。
_TRANSCRIPT_FORMULA = (
    "Σ字数(批量 ASR 转写,`data/transcripts/<sid>.json` 的 n_chars) "
    "÷ Σ有声秒(逐帧 pyin 的浊音帧 × 帧长);"
    "⚠️ 音频为全场混音(含面试官),无说话人分离 ⟹ 量的是全场语音字数率,不是考生语速"
)


def _speech_rate_from_transcript(session_id: str | None, series: list | None,
                                 thresholds: dict[str, int] | None) -> L1Indicator:
    """用转写字数 + 序列有声秒补出 `speech_rate`。

    三条前置任一不满足就如实报缺,不猜:

    * 没有转写 ⟹ `not_stored`(不是 0 字);
      (没有 `session_id` 时 `transcript_chars.n_chars(None)` 本来就返回 `None`,
      所以这里**不另设守卫** —— 我加过一条 `if session_id is None`,变异验证时
      删掉它测试**照样绿**,说明它是死代码,已删。)
    * 
    * 没有序列 / 有声秒为 0 ⟹ `no_samples`。

    置信度走 `confidence_for`:整场只有**一个**合并值 ⟹ `n_valid=1`,
    低于 `speech` 族的门槛 2 ⟹ 恒为 **低**。这是如实反映,不是保守。
    """
    from report_frontend import transcript_chars
    from report_frontend.voice_series import MIN_PAUSE_SEC  # noqa: F401  (保持导入面一致)

    name = "speech_rate"
    fam, formula, req = _SPECS[name]
    chars = transcript_chars.n_chars(session_id)
    if chars is None:
        return _un(name, GAP_NOT_STORED)
    if not series:
        return _un(name, GAP_NO_SAMPLES)
    voiced = sum(float(s.voiced_sec) for s in series if getattr(s, "f0", None) is not None)
    if not voiced > 0:
        return _un(name, GAP_NO_SAMPLES)
    thr = (thresholds or {}).get("speech", 2)
    return measured(name, fam, _TRANSCRIPT_FORMULA, req, "session", "字/秒",
                    value=chars / voiced, n_valid=1, n_total=1, threshold=thr)


def _source_on_disk(session_id: str | None) -> bool:
    """本场**有没有可用于重算序列的音频**。没有 `session_id` ⟹ False。

    只回答"素材在不在",**不提取** —— 报告是读侧,不许有副作用。
    判据交给 `voice_series.audio_source`(它认得逐题音频与整场 `camera.webm`),
    不在这里另写一份"文件存不存在"的逻辑:本仓为此栽过太多次
    (`jx_check_session` 曾自带一份 ffprobe 量具)。
    """
    if not session_id:
        return False
    try:
        from report_frontend import voice_series
        return voice_series.audio_source(session_id)[0] != "none"
    except Exception:  # noqa: BLE001 - 读盘出错按"没有"处理,别让报告炸掉
        return False


def compute_voice_indicators(voice: pd.DataFrame | None,
                             thresholds: dict[str, int] | None = None,
                             series: list | None = None,
                             session_id: str | None = None) -> list[L1Indicator]:
    """整场语音的 V 族 12 条。`voice` 一行 = 一段回答。

    `series`:`voice_series.VoiceSeries` 列表(逐题)。给了它,那 8 条
    "需要帧序列"的指标才会出值;没给就报 `not_stored`。

    `session_id`:给了它,**日志侧 4 条里能救的**才有机会救 —— 目前只有
    `speech_rate`:它的 `n_chars` 不在语音日志里(正式场 0 行),而在
    `data/transcripts/<sid>.json`(批量 ASR 的产物)。见
    `report_frontend/transcript_chars.py` 与下面 `_speech_rate_from_transcript`。
    """
    # ★ 语音**日志**没有,不代表 V 族全无:那 8 条序列指标**根本不读日志**,
    #   它们只要帧序列。正式场正是这种情况 —— 24/24 场语音日志 0 行,
    #   但整场录音(`camera.webm`)在,序列照样算得出来。
    #   所以这里**先**把日志侧的 4 条判死,**再**把序列侧合进来。
    from report_frontend import l1_voice_series
    # 诊断事实只依赖"这一场盘上有什么、日志里有什么",与走哪条分支无关 ⟹ 先算一次。
    # 它不进任何数值,只交给措辞层(见 `_rt_gap_detail`)。
    rt_detail = _rt_gap_detail(voice, session_id)
    if voice is None and not series:
        # 既没有日志、也没有序列 —— 但**这还不足以说"模态缺失"**。
        # ★ 2026-10-24 实测:场 `20260929_205448_8b75` 的 `camera.webm` 在盘上,
        #   34 秒就能算出 V 族 10/12,而载荷写着「V 族 12 条全部不可出」+ `modality_missing`
        #   —— 前端对 `modality_missing` 的提示是「**补救:查采集**」。采集没坏,
        #   坏的是"还没跑提取"。判词比事实更狠,人会去查一个没坏的东西。
        #   所以先问一句"素材在不在":在 ⟹ `not_extracted`(跑一次就有,不用重录);
        #   不在 ⟹ 才是真的 `modality_missing`。
        gap = GAP_NOT_EXTRACTED if _source_on_disk(session_id) else GAP_MODALITY_MISSING
        return [_un(n, gap) for n in ORDER]
    if voice is None or voice.empty:
        # ⚠️ `voice is None` 时(本场没有日志文件)也走 `not_extracted` 的判据 ——
        #    否则 24 场正式场(日志 0 行、录音都在)会在这里又被判成"模态缺失"。
        gap = (GAP_NOT_EXTRACTED if _source_on_disk(session_id)
               else GAP_MODALITY_MISSING) if voice is None else GAP_NO_SAMPLES
        got0: dict[str, L1Indicator] = {
            n: _un(n, gap) for n in ORDER if n not in l1_voice_series.SERIES_NAMES}
        got0.update(l1_voice_series.compute_from_series(series, thresholds, n_total=0))
        # ★ `speech_rate` 还能救:它的两个输入都不是语音日志 ——
        #   字数在批量转写里,有声秒在帧序列里。实测 13/13 场 A 档都能算。
        got0["speech_rate"] = _speech_rate_from_transcript(session_id, series, thresholds)
        # ★ 但**序列已经在了**的时候,这两条不许沿用"跑一次提取就有"的判词:
        #   2026-10-24 实测(31 场序列补齐之后)`pause_ratio` / `response_latency_sec`
        #   仍报 `not_extracted`,而提取早就跑完了 —— 再跑一百次它们也出不来。
        #   判词比事实更狠的第二种形态:上一版说「查采集」(方向反了),
        #   这一版说「再跑一次提取」(永远跑不完)。见 `_NOT_FIXABLE_BY_EXTRACTION`。
        if series:
            for _n, _g in _NOT_EXTRACTABLE_GAP.items():
                # `response_latency_sec` 额外交出诊断事实(见 `_rt_gap_detail`)——
                # 它的 `window_none` 在 31 场里有两种病因,措辞层要据此选句。
                # 码、值、键集一个都没动:只是多带一份"这一场实测到了什么"。
                got0[_n] = _un(_n, _g, gap_detail=(rt_detail
                                                   if _n == "response_latency_sec" else None))
        return [got0[n] for n in ORDER]

    thr = thresholds or {}
    n_rows = len(voice)
    got: dict[str, L1Indicator] = {}

    speech = _col(voice, "speech_duration_sec")
    n_chars = _col(voice, "n_chars")
    # ⚠️ 原来是 `dur = _col(voice, "duration_sec")`,只为 `pause_ratio` 存在;
    #    该指标 2026-11-22 退役后这一行变成死变量,已删。

    # ── 1. speech_rate = Σ字数 ÷ Σ有声秒 ────────────────────────────────
    # 用**加权**合并(Σn_chars ÷ Σspeech),不是"各段比值的均值" ——
    # 后者会让一段 3 秒的回答与一段 30 秒的回答等权,把短答的噪声放大。
    if speech is not None and n_chars is not None:
        m = speech.gt(0) & n_chars.notna()
        if m.any():
            got["speech_rate"] = measured(
                "speech_rate", _SPECS["speech_rate"][0], _SPECS["speech_rate"][1],
                _SPECS["speech_rate"][2], "session", "字/秒",
                value=float(n_chars[m].sum()) / float(speech[m].sum()),
                n_valid=int(m.sum()), n_total=n_rows,
                threshold=thr.get("speech", 2))

    # ── 3. pause_rate_per_min = Σ停顿次数 ÷ Σ有声分钟 ───────────────────
    # `pause_frequency` 已是"每分钟"量 ⟹ 次数 = freq × 有声秒 ÷ 60;
    # 合并后 = Σ(freq_i × 秒_i) ÷ Σ秒_i,即按有声时长加权。
    freq = _col(voice, "pause_frequency")
    if freq is not None and speech is not None:
        m = freq.notna() & speech.gt(0)
        if m.any():
            got["pause_rate_per_min"] = measured(
                "pause_rate_per_min", _SPECS["pause_rate_per_min"][0],
                _SPECS["pause_rate_per_min"][1], _SPECS["pause_rate_per_min"][2],
                "session", "次/分钟",
                value=float((freq[m] * speech[m]).sum() / speech[m].sum()),
                n_valid=int(m.sum()), n_total=n_rows,
                threshold=thr.get("pause", 2))

    # ── 6. `pause_ratio` **已退役**(2026-11-22)─────────────────────────
    #
    # 这里原来是 `Σ(整段 − 有声) ÷ Σ有声`。它与 `voiced_ratio` 是代数恒等变换
    # (`pause_ratio ≡ 1/voiced_ratio − 1`,实测残差 4.4e-16、ρ = −1.000000),
    # 报出来只是把同一份证据数两遍 ⟹ 连同 `_SPECS` 一起退掉(见 `RETIRED`)。
    # ⚠️ `duration_sec` 这一列**没有从采集里删**:本函数只是不再用它算指标。

    # ── 7. response_latency_sec = 逐题反应延迟的中位数 ──────────────────
    rt = _col(voice, "reaction_time")
    if rt is not None:
        m = rt.notna()
        if m.any():
            got["response_latency_sec"] = measured(
                "response_latency_sec", _SPECS["response_latency_sec"][0],
                _SPECS["response_latency_sec"][1], _SPECS["response_latency_sec"][2],
                "session", "秒",
                value=float(np.median(rt[m])),
                n_valid=int(m.sum()), n_total=n_rows,
                threshold=thr.get("reaction_time", 5))

    # ── 需要帧序列的 8 条 ──────────────────────────────────────────────
    from report_frontend import l1_voice_series
    for name, ind in l1_voice_series.compute_from_series(series, thr, n_total=n_rows).items():
        got[name] = ind

    # ── 仍未出值的:粒度不够,不许硬凑 ────────────────────────────────
    for name in ORDER:
        if name not in got:
            # `response_latency_sec` 落到这里的情形是"日志有行、但没有应答时刻"
            # (`reaction_time` 列缺失或全空)⟹ 同样把诊断事实交出去,让措辞层
            # 说的是"应答时刻没落盘",而不是笼统的"粒度太粗"。
            got[name] = _un(name, GAP_NOT_STORED, n_total=n_rows,
                            gap_detail=(rt_detail if name == "response_latency_sec" else None))

    return [got[n] for n in ORDER]


def _un(name: str, gap: str, n_total: int = 0,
        gap_detail: dict | None = None) -> L1Indicator:
    family, formula, cols = _SPECS[name]
    return unavailable(name, family, formula, tuple(cols), "session", "", gap, n_total,
                       gap_detail=gap_detail)


def indicator_names() -> tuple[str, ...]:
    """V 族对外承诺的 12 个名字(其中 8 条本轮固定 `not_stored`)。"""
    return ORDER
