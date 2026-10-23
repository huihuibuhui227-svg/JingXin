# report_frontend/l1_quality.py
"""**M4.4 对齐质量报告** —— 每个指标说清"它靠哪几个模态、用了几个观测、为什么不可信"。

## 为什么要有这一层

L1 指标本身只回答"这个读数是多少"。报告要能站住,还必须回答三件事:

1. **它是从哪个模态来的?** —— 一个 `blink_rate_per_min` 只可能来自 face;
2. **它用了几个观测?** —— `n_valid` 必须能追到具体帧数,不是"落盘了多少行";
3. **它为什么不可信?** —— `gap` 要能分类统计(卡标定 / 没采集 / 模态缺失 / 时间轴坏了)。

第 3 条是这一层的重点:**"算不出来"有四种完全不同的原因,补救动作也完全不同**。
把它们混成一句"证据不足",等于把"去查采集"和"去等标定集"说成同一件事。

## 一条硬不变量

**任一模态缺失 ⟹ 依赖它的指标必须"无证据"。**

这条不是靠自觉:本模块的 `QualityReport.validate()` 会真的去查 ——
如果一个指标自称有值、而它依赖的模态在时间轴上根本不存在,直接 `ValueError`。
测试里用变异钉着(把这条检查删掉 ⟹ 红)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from report_frontend import session_timeline
from report_frontend.l1_indicators import (
    CONFIDENCE, GAP_COLUMN_EMPTY, GAP_MODALITY_MISSING, GAP_NO_SAMPLES,
    GAP_NOT_STORED, GAP_TIME_UNUSABLE, GAP_UNCALIBRATED, GAP_WINDOW_NONE,
    KINDS, L1Indicator,
)

#: 每个指标族依赖哪些模态。**这是唯一的登记处** —— 别处不许再抄一份。
#: 目前只有 F 族(面部);V / G / T 族落地时在这里加行。
FAMILY_MODALITY = {
    "F1_smile": "face",
    "F2_gaze": "face",
    "F3_blink": "face",
    "F4_head": "face",
    "F5_tension": "face",
    "F6_expressivity": "face",
    "F7_micro": "face",
    "G1_activity": "gesture",
    "G2_posture": "gesture",
    "G3_visibility": "gesture",
    "V1_amount": "voice",
    "V2_pause": "voice",
    "V3_pitch": "voice",
    "V4_energy": "voice",
}

#: 人话解释 —— 报告里给读者看的那一句。**不许直接渲染 `gap` 的英文常量。**
GAP_EXPLANATION = {
    GAP_UNCALIBRATED: "该指标需要一个标定阈值,而标定集还没到位(属 M5)",
    GAP_COLUMN_EMPTY: "上游没有采集这一列,不是样本量不够",
    GAP_MODALITY_MISSING: "本场没有这个模态的日志",
    GAP_TIME_UNUSABLE: "时间轴无法重建,不能按窗口切分",
    GAP_WINDOW_NONE: "本场没有可用的回答窗口",
    GAP_NO_SAMPLES: "这个窗口里没有有效观测",
    GAP_NOT_STORED: "上游只存了聚合量,这一层需要的粒度没落盘",
}


#: 模态 → 判定"这一行有没有数据"的列。
#:
#: **不取全部列**:日志里有一批结构性常量的列(见 `evidence_gate.QUARANTINE`),
#: 拿它们判"有数据"会把整行空日志判成"有数据"。这里只取**该模态真正要用的**那几列。
COVERAGE_COLUMNS: dict[str, tuple[str, ...]] = {
    "face": ("au12_smile", "au4_frown", "avg_ear", "head_pitch"),
    "gesture": ("torso_jitter", "shoulder_width"),
    "voice": ("n_chars", "duration_sec", "speech_duration_sec"),
}


def modality_coverage(logs: dict) -> dict[str, dict]:
    """每个模态:日志有几行、其中**真正带数据**的几行。

    ## 为什么需要这个

    "日志存在"与"数据存在"是两件事,而报告层此前只看前者。实测(49 场):

    ```
    面部日志 21861 行,其中 25.5% 整行为空(没检出脸)
    13 场会话的面部日志**每一行**都是空的
    6 场会话脸 ~100% 但手臂可见度 0.10(低于阈值 ⟹ 角度被扣留)
    研究模式 11 个日志文件,全部 0 行
    ```

    ⟹ 一场会话可以有 `face` 日志、时间轴成立、16 条 F 指标**全部** `n_valid=0`,
    而质量报告只说"face 已提供"。读者会把"提供"读成"有数据"。

    本函数把这件事变成可查询的数:`usable_ratio == 0` 就是"这个模态在本场
    **名义存在、实际为零**"。
    """
    out: dict[str, dict] = {}
    for modality, cols in COVERAGE_COLUMNS.items():
        df = _pick_log(logs, modality)
        if df is None:
            out[modality] = {"rows": 0, "usable_rows": 0, "usable_ratio": 0.0,
                             "provided": False}
            continue
        n = len(df)
        present = [c for c in cols if c in df.columns]
        if n == 0 or not present:
            out[modality] = {"rows": n, "usable_rows": 0, "usable_ratio": 0.0,
                             "provided": True}
            continue
        ok = df[present].notna().any(axis=1)
        k = int(ok.sum())
        out[modality] = {"rows": n, "usable_rows": k,
                         "usable_ratio": (k / n) if n else 0.0, "provided": True}
    return out


def _pick_log(logs: dict, modality: str):
    """模态 → 日志 DataFrame。语音在三个键名之间挑(见 `l1_session._pick_voice`)。"""
    import pandas as pd

    keys = ("voice", "voice_interview", "voice_research") if modality == "voice" \
        else (modality,)
    for key in keys:
        df = logs.get(key)
        if isinstance(df, pd.DataFrame):
            return df
    return None


@dataclass
class QualityReport:
    session_id: str
    timebase: dict
    windows: dict
    by_confidence: dict[str, int] = field(default_factory=dict)
    by_gap: dict[str, int] = field(default_factory=dict)
    families: dict[str, dict] = field(default_factory=dict)
    indicators: list[dict] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    #: 每个模态的**实际**可用帧/行占比(见 `modality_coverage`)。
    coverage: dict[str, dict] = field(default_factory=dict)
    #: 本场是否有 `voice_series`(`data/l0_series/<sid>.npz`)。
    #:
    #: ★ `timebase` 与 `coverage` **都只描述日志**。而 voice 还有第二个数据源:
    #: 整场录音逐帧重算出的 series。没有 voice 日志的场次,V 族仍可能有值
    #: (实测 12 条里 9 条能重算)—— 不把这件事告诉 `validate()`,
    #: 它会把"日志没有、series 有"判成"自称有值",然后抛错。
    #: 这是**校验的观测对象少了一个**,不是指标算错了。
    series_available: bool = False

    @property
    def n_total(self) -> int:
        return len(self.indicators)

    @property
    def n_measured(self) -> int:
        return self.by_confidence.get("高", 0) + self.by_confidence.get("中", 0) \
            + self.by_confidence.get("低", 0)

    def validate(self) -> None:
        """★ 硬不变量:模态缺失 ⟹ 依赖它的指标必须无证据。

        另外三条自检(都与"把证据说多"有关):
        · 每个指标必须带 `gap` 或 `value`,不能两者皆无;
        · `n_valid <= n_total`;
        · 有值的指标,其族依赖的模态必须在时间轴上存在。
        """
        missing = {m for m, tb in self.timebase.items()
                   if not tb.get("usable", False)}
        # ★ 名义存在、实际为零的模态:日志在,但**没有一行带数据**。
        #   它比"模态缺失"更危险 —— 时间轴成立、`timebase.usable` 为真,
        #   于是 `missing` 抓不到它,报告会照常说"face 已提供"。
        empty = {m for m, cov in self.coverage.items()
                 if cov.get("provided") and cov.get("usable_ratio", 0) == 0}
        # ★ **另一个数据源**:`voice` 模态有一条不经日志的来路 ——
        #   `voice_series`(`data/l0_series/<sid>.npz`,整场录音逐帧 f0/rms)。
        #   实测:没有 voice 日志的场次,V 族 12 条里 9 条能由 series 重算。
        #   这两条来路各自独立,`timebase.voice` 与 `coverage.voice` **都**只描述日志,
        #   于是"日志没有、series 有"会被上面的 `missing` 判成"自称有值"。
        #   这不是缺陷,是校验的**观测对象少了一个**。
        #
        #   ⚠️ 这个豁免是**窄的**:只对 `voice` 生效,且只在 series 真的存在时生效。
        #   其余模态没有任何"日志之外"的来路,不该借这条口子。
        series_backed = bool(self.series_available)
        if series_backed:
            missing.discard("voice")
            empty.discard("voice")
        for row in self.indicators:
            fam = row["family"]
            dep = FAMILY_MODALITY.get(fam)
            if dep and dep in missing and row["value"] is not None:
                raise ValueError(
                    f"{row['name']}@{row['window']} 自称有值({row['value']}),"
                    f"但它依赖的模态 {dep!r} 在本场不存在/不可用")
            if dep and dep in empty and row["value"] is not None:
                raise ValueError(
                    f"{row['name']}@{row['window']} 自称有值({row['value']}),"
                    f"但它依赖的模态 {dep!r} 在本场**一行数据都没有**"
                    f"(日志存在但 usable_ratio=0)")
            if row["value"] is None and row["gap"] is None:
                raise ValueError(
                    f"{row['name']}@{row['window']}: 既没有值也没有 gap —— "
                    f"读者会以为它是 0")
            if row["n_valid"] > row["n_total"]:
                raise ValueError(
                    f"{row['name']}@{row['window']}: n_valid({row['n_valid']}) > "
                    f"n_total({row['n_total']}) —— 把证据说多了")
            if row["gap"] is not None and row["confidence"] != "无":
                raise ValueError(
                    f"{row['name']}@{row['window']}: 有 gap 却报置信度 {row['confidence']}")

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "timebase": self.timebase,
            "windows": self.windows,
            "summary": {
                "n_indicators": self.n_total,
                "n_measured": self.n_measured,
                "by_confidence": dict(self.by_confidence),
                "by_gap": dict(self.by_gap),
            },
            "families": self.families,
            "indicators": self.indicators,
            "gaps": list(self.gaps),
        }


def build_quality_report(result) -> QualityReport:
    """从 `l1_session.L1Result` 汇总。**不重算任何指标**,只整理与自检。"""
    tl = result.timebase
    timebase: dict[str, dict] = {}
    for m, tb in tl.modalities.items():
        # `status` 是 `TimebaseStatus`(str 混入枚举);测试替身可能直接给字符串。
        # 两种都收,免得"为了测试而改生产代码的形状"。
        timebase[m] = {
            "status": getattr(tb.status, "value", tb.status),
            "provided": True,
            "usable": tb.usable,
            "n_frames": tb.n_frames,
            "wall_back_steps": tb.wall_back_steps,
            "drift_sec": tb.drift_sec,
            # 覆盖到的时间范围 —— 判断"两族看的是不是同一段时间"只能比这个。
            "span_sec": tb.span_sec,
        }
    # 没被传进来的模态也要出现在表里,但状态是 `not_provided` ——
    # **不能**写成"缺失":那会让人以为这一场根本没录,而实际只是没加载。
    # 这两种情况的补救动作完全不同(去补日志 vs 去补加载)。
    for m in ("face", "gesture", "voice"):
        timebase.setdefault(m, {"status": "not_provided", "provided": False,
                                "usable": False, "n_frames": 0,
                                "wall_back_steps": 0, "drift_sec": None,
                                "span_sec": None})

    ws = result.windows
    windows = {
        "granularity": ws.granularity,
        "n_windows": len(ws),
        "n_questions": ws.n_questions,
        "n_answer_segments": ws.n_answer_segments,
        "wall_reliable": ws.wall_reliable,
        # 落在所有窗口之外的帧数(模态 → 计数)。有台账时首个 ask_end 之前的帧
        # 不属于任何窗口,这个数是唯一能看出"分母被排除了多少"的地方。
        "frames_outside": dict(ws.frames_outside),
    }

    by_conf = {c: 0 for c in CONFIDENCE}
    by_gap = {k: 0 for k in KINDS}
    families: dict[str, dict] = {}
    rows: list[dict] = []

    for ind in result.indicators:
        by_conf[ind.confidence] = by_conf.get(ind.confidence, 0) + 1
        if ind.gap:
            by_gap[ind.gap] = by_gap.get(ind.gap, 0) + 1
        fam = families.setdefault(ind.family, {
            "modality": FAMILY_MODALITY.get(ind.family),
            "n": 0, "n_measured": 0,
        })
        fam["n"] += 1
        if ind.has_value:
            fam["n_measured"] += 1
        rows.append(ind.to_dict())

    report = QualityReport(
        session_id=result.session_id,
        timebase=timebase,
        windows=windows,
        # 从结果本身读,不在校验处再猜一次 —— `compute_session_l1` 是唯一
        # 知道"series 到底有没有被用上"的地方。
        series_available=bool(getattr(result, "series_available", False)),
        by_confidence=by_conf,
        by_gap=by_gap,
        families=families,
        indicators=rows,
        gaps=list(result.gaps),
        # "日志存在" ≠ "数据存在":把每个模态**实际**可用行占比带进报告。
        coverage=dict(getattr(result, "coverage", None) or {}),
    )
    report.validate()
    return report


#: 置信「低」的**原因**,按指标名。置信度只说"有多信",不说"为什么只有这么信" ——
#: 后者才是读者要的那句话。**没有登记的指标不会瞎猜**,返回 `None`。
LOW_CONFIDENCE_REASON: dict[str, str] = {
    # ★ 2026-10-18 补两条。它们一直是「低」但**没有登记原因** ——
    #   报告上只写「置信 低」,不写为什么低。置信度只说"有多信",
    #   不回答"为什么只有这么信",后者才是读者要的那句话。
    # ★ 两条手势率量的证据单位是"段"不是"帧"。13 场实测段数 1~62,
    #   所以它们**本来就该是「低」或「中」** —— 一条只有两三次事件的率,
    #   不该因为"看了 470 帧"就被判成证据充分。
    "gesture_burst_rate_per_min_vs_baseline": (
        "证据单位是**段数**(观察到几次运动段),不是帧数。本场段数很少 ⟹ 率的"
        "不确定度大。这不是算错:整场只出现了这么几次超本人基线的连续段。"),
    "postural_shift_rate_per_min_vs_baseline": (
        "同上:证据单位是**段数**。本场段数很少 ⟹ 率的不确定度大。"
        "⚠️ 本族与 `gesture_burst_rate_per_min_vs_baseline` 高度相关"
        "(13 场里 9 场取到相同的整数段数)。"),
    "speech_rate": (
        "整场只有**一次**测量(转写字数 ÷ 有声秒数),n_valid=1 ⟹ 报不出跨样本一致性。"
        "素材上没有任何题目边界可用,这是素材决定的,不是算错了。"
        "⚠️ 分子是**全场混音**的字数(含面试官),不是考生一人的语速。"),
    "pause_rate_per_min": (
        "整场只有**一次**测量(停顿段数 ÷ 有声分钟数),n_valid=1。"
        "⚠️ 同样基于全场混音,停顿里包含面试官与考生之间的换手间隙。"),
    "pitch_range_semitone": (
        "整场音频没有题目边界,本指标在整场只测量**一次**(首末 f0 之差),"
        "n_valid=1 ⟹ 报不出跨样本一致性。这是素材决定的,不是算错了。"),
    "pitch_variability_per_sec": (
        "同上:分母是「这一次测量」的时长,整场只有一次测量 ⟹ n_valid=1。"),
    "energy_dynamics_per_sec": (
        "同上:整场只有一次测量 ⟹ n_valid=1。"),
}

#: 这几条指标为什么**不**改成分窗多次测量 —— 这是使用者的裁定,写在这里免得后人反复问。
LOW_CONFIDENCE_DECISION = (
    "**裁定(2026-10-18):接受单次测量,如实报「低」,不改口径。**"
    "备选做法是按固定 30 秒切段、段间取中位数,以获得多次测量;"
    "但那把统计量从「按题/按场」换成了「按 30 秒窗」—— 名字、单位、可比性全变。"
    "本仓的规矩是不许**偷偷**换统计量(见 M4.9 关于 pitch_range_semitone 的约定),"
    "所以要么原样报低,要么另起一个新名字(如 *_per_30s)并同时改规格 —— 后者未做。")


def explain_confidence(report: QualityReport) -> list[str]:
    """把"置信低"翻译成一句人话。只对**有值但置信低**的指标说。"""
    out: list[str] = []
    for row in report.indicators:
        if row.get("confidence") != "低" or row.get("value") is None:
            continue
        reason = LOW_CONFIDENCE_REASON.get(row.get("name", ""))
        if reason:
            out.append(f"{row['name']}:{reason}")
    return out


def explain_gaps(report: QualityReport) -> list[str]:
    """把 `gap` 统计翻成**报告里能直接印的中文句子**。

    只印非零项,且按 `KINDS` 的顺序 —— 顺序固定,免得两次生成的报告读起来不一样。
    """
    out: list[str] = []
    for kind in KINDS:
        n = report.by_gap.get(kind, 0)
        if n:
            out.append(f"{GAP_EXPLANATION[kind]}:{n} 条")
    return out


def modality_line(report: QualityReport) -> str:
    """一行说清本场哪些模态可用 —— 报告的"材料"段用这一句。

    三档分开说:**可用** / **提供了但不可用** / **没提供**。
    把后两者混成"缺失"会让人去补一个其实已经在盘上的日志。
    """
    usable = [m for m, tb in report.timebase.items() if tb.get("usable")]
    broken = [m for m, tb in report.timebase.items()
              if tb.get("provided") and not tb.get("usable")]
    missing = [m for m, tb in report.timebase.items() if not tb.get("provided")]
    parts = ["可用模态:" + "、".join(usable) if usable else "本场没有任何可用模态"]
    if broken:
        parts.append("提供了但不可用:" + "、".join(broken))
    if missing:
        parts.append("未提供:" + "、".join(missing))
    return ";".join(parts)
