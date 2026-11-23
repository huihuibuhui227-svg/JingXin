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

from report_frontend import gap_explanations, instrument_provenance, session_timeline
from report_frontend.l1_indicators import (
    CONFIDENCE, GAP_COLUMN_EMPTY, GAP_MODALITY_MISSING, GAP_NO_SAMPLES,
    GAP_NOT_EXTRACTED, GAP_NOT_PRODUCED, GAP_NOT_STORED, GAP_TIME_UNUSABLE,
    GAP_UNCALIBRATED, GAP_WINDOW_NONE, KINDS, L1Indicator,
)

#: 每个指标族依赖哪些模态。**这是唯一的登记处** —— 别处不许再抄一份。
FAMILY_MODALITY = {
    "F1_smile": "face",
    "F2_gaze": "face",
    "F3_blink": "face",
    "F4_head": "face",
    "F5_brow_lip": "face",
    "F6_expressivity": "face",
    # ★ 2026-11-22:退掉 `F7_micro`(它的两条指标 `micro_exp_*` 从没出过值,
    #   已随指标一起退),补进 `F8_symmetry`(`facial_asymmetry_p90`)。
    "F8_symmetry": "face",
    "G1_activity": "gesture",
    "G2_posture": "gesture",
    "G3_visibility": "gesture",
    "G4_hand": "gesture",
    "V1_amount": "voice",
    "V2_pause": "voice",
    "V3_pitch": "voice",
    "V4_energy": "voice",
    # ★ 2026-11-23:补齐 T 族 3 个文本族,使 `validate()` 与 `modality_summary()` 覆盖全部 41 条。
    "T1_lexical": "text",
    "T2_discourse": "text",
    "T3_repair": "text",
}

#: 指标族代码 → 中文显示名（面向前端与审计展示，不含任何心理/情绪词）
FAMILY_LABEL: dict[str, str] = {
    "F1_smile": "面部 · 微笑动作",
    "F2_gaze": "面部 · 视线注视",
    "F3_blink": "面部 · 眨眼节律",
    "F4_head": "面部 · 头部运动（2D 代理）",
    "F5_brow_lip": "面部 · 眉唇局部形变",
    "F6_expressivity": "面部 · 动作活跃度",
    "F8_symmetry": "面部 · 左右不对称度",
    "G1_activity": "手势 · 上肢活动",
    "G2_posture": "姿态 · 躯干与肩部",
    "G3_visibility": "姿态 · 可见度协变量",
    "G4_hand": "手势 · 手掌张开",
    "V1_amount": "语音 · 发声与语速",
    "V2_pause": "语音 · 停顿结构",
    "V3_pitch": "语音 · 基频韵律",
    "V4_energy": "语音 · 声学能量",
    "T1_lexical": "文本 · 词汇丰富度",
    "T2_discourse": "文本 · 话语连接词",
    "T3_repair": "文本 · 自我修正",
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
    GAP_NOT_EXTRACTED: "素材在盘上,派生序列还没生成(跑一次提取就有,不用重录)",
    GAP_NOT_PRODUCED: "这一条故意不算 —— 它与另一条指标是同一个数换了个算法,多报一遍只是把同一份证据数两遍",
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
        # 文本族由转写文件而非帧时间轴提供；仅当 coverage["text"] 显式声明不可用时触发拦截
        text_cov = self.coverage.get("text")
        if isinstance(text_cov, dict) and (
            not text_cov.get("provided", False) or text_cov.get("usable_ratio", 0) == 0
        ):
            empty.add("text")
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
        row = ind.to_dict()
        # 新增键(**不动任何既有键**):这条指标为什么没有值/为什么不可信。
        # ★ 2026-11-21:选句改成"按实测诊断事实"。反应延迟同一个码有两种病因
        #   (没走题库流程 / 台账在但语音日志空),拿不到事实才退回通用特例句。
        #   `gap_detail` 只**递给解释层**、不进 `row` —— 明细行的键集是既有契约
        #   (`tests/test_gap_explanations.py` 那条"新增键只允许仪器来源两键"钉着它)。
        row["gap_explanation"] = gap_explanations.explain(
            {**row, "gap_detail": getattr(ind, "gap_detail", None)})
        rows.append(row)

    # ★ A5 仪器来源披露(2026-11-21):**装配点就在这里**。
    #
    # ## 为什么是这一行
    #
    # 38 条指标的形状在这里第一次定型(`L1Indicator` → `dict`),此后所有出口
    # (`/api/report/structured` 的 `l1.indicators`、HTML 报告的页脚汇总、
    # 前端的逐条标签)读的都是**这一份 rows**。在装配置标一次,三个出口就是同一份口径。
    #
    # ## 为什么不在别处标
    #
    # * **不在 `l1_session`(计算层)**:那一层只回答"这个读数是多少"。一条读数由
    #   哪一套仪器产出,取决于它的 `required_l0`(列名)落在哪一路登记表里 —— 那是
    #   披露,不是计算。计算层不该知道报告要不要披露。
    # * **不在渲染层(HTML / 前端)**:两处各算一次就一定会分叉,而这里最禁不起分叉的
    #   恰恰是 `speech_rate`(转写字数 ÷ 有声时长 = `text` + `audio` 两路)——
    #   一边写成 audio、另一边写成 text,报告就自相矛盾。载荷给一次,两个出口读同一份。
    #
    # ## 它只做加法
    #
    # `annotate_indicators` 返回副本,**只新增** `instrument`(单仪器)或
    # `instruments`(多仪器);上面刚写的 `gap_explanation` 与 `to_dict()` 的原字段
    # 一个不动。钉子:`tests/test_report_instrument_rendering.py` 里那条"既有键值零变化"。
    rows = instrument_provenance.annotate_indicators(rows)

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


#: L1 指标名 → **中文名**。交付物与界面印这一列,不印左边的键。
#:
#: ## 为什么需要这张表
#:
#: `explain_confidence()` 拼的是 `{指标名}:{理由}`,而指标名是**审计键**。
#: 2026-10-25 深夜实测:交付物正文里印出 `speech_rate:` / `pause_rate_per_min:` /
#: `pitch_range_semitone:` 等 5 个键,而同一份报告的维度表里,同一个指标印的是
#: 中文槽名(「语调变化」「平均停顿时长」)⟹ 读者会以为是两个东西。
#:
#: ## 2026-10-26:从 7 条扩到全部 38 条
#:
#: 原来只登记「会以低置信出现」的那 7 条,理由是「没在报告里出现过的指标,
#: 读者没有'见过的那一版',现在编一个只会将来打架」。
#:
#: 这条理由在**逐题离散度**进报告之后失效了:那张表一印就是 20 条指标,
#: 读者会在同一份报告里同时看到它们 —— 没有中文名,那一整块就是英文键的表格。
#: ⟹ 现在全部登记。与 `l1_mapper.SLOT_LABEL` 重叠的 8 条**逐字照抄槽名**
#: (同一个量在两个出口必须同名),由
#: `tests/test_indicator_names_agree_with_slot_names.py` 对账。
INDICATOR_LABEL: dict[str, str] = {
    # F 族(面部)
    "smile_ratio": "微笑帧占比",
    "smile_rate_per_min": "微笑次数",
    "smile_duration_p50": "微笑持续时长",
    "gaze_aversion_ratio_vs_baseline": "视线偏离占比",
    "gaze_aversion_rate_per_min_vs_baseline": "视线偏离次数",
    "gaze_stability": "视线稳定性",
    # ★ 2026-11-22 新增(第 14 个槽 `eye_contact` 的时长面)。
    #   ⚠️ 这一条与 `l1_mapper.SLOT_LABEL["eye_contact"]` =「眼神接触比例」**故意不同名**:
    #   槽数的是**比例**(无量纲),这条数的是**秒**(注视带内连续段长的中位)。
    #   同名会让读者把 12.5 读成"接触比例 12.5"(比例不可能 >1),含义整个错位。
    #   与下面「面部不对称度」属于**两类不同的例外**,登记在
    #   `tests/test_indicator_names_agree_with_slot_names.py`:
    #   · `symmetry_score` = 方向相反(`_DIRECTION_FLIP`);
    #   · `eye_contact` = **统计面不同:比例 → 秒**(`_DIFFERENT_STATISTIC`)。
    "gaze_dwell_p50": "眼神接触时长",
    "blink_rate_per_min": "眨眼频率",
    "blink_duration_mean": "眨眼时长",
    "head_motion_per_sec": "头部运动量",
    "head_away_ratio_vs_baseline": "头部偏离占比",
    "brow_furrow_ratio_vs_baseline": "皱眉",
    "lip_compression_ratio_vs_baseline": "唇部压缩",
    "au_activation_any_ratio": "面部动作活跃帧占比",
    "au_variability_per_sec": "面部动作变化速度",
    # ★ 2026-11-22 换人:`micro_exp_rate_per_min`("微表情次数")/
    #   `micro_exp_duration_mean_sec`("微表情时长")**退役**(全语料 0 场出值),
    #   由 `facial_asymmetry_p90` 承接 F 族这一格。
    #   ⚠️ 这一条与 `l1_mapper.SLOT_LABEL["symmetry_score"]` =「面部对称性」**故意不同名**:
    #   指标数的是 `1 − symmetry_score`,**方向相反**(值越大越不对称)。
    #   同名会让读者把 0.062 读成"对称度 0.062"(极不对称),恰好说反。
    #   这是本仓唯一登记在案的"同一构念、两个出口不同名"的情形,由
    #   `tests/test_indicator_names_agree_with_slot_names.py` 的 `_DIRECTION_FLIP` 钉着。
    "facial_asymmetry_p90": "面部不对称度",
    # G 族(手势)
    "gesture_activity_per_sec": "上肢活动抖动度",
    "gesture_burst_rate_per_min_vs_baseline": "手势爆发频率",
    "posture_stability": "躯干肩部抖动度",
    "postural_shift_rate_per_min_vs_baseline": "姿态变换频率",
    # ⚠️ 这两条**与槽名逐字相同**(「耸肩幅度」/「手掌张开度」):构念没变,
    #   变的只是统计量(均值 → p90 / 中位)。
    "shrug_elevation_p90": "耸肩幅度",
    "hand_openness_p50": "手掌张开度",
    "hand_visible_ratio_l": "左手可见占比",
    "hand_visible_ratio_r": "右手可见占比",
    "pose_valid_ratio": "姿态可读帧占比",
    # V 族(语音)
    "speech_rate": "语音字数率",
    "voiced_ratio": "发声占比",
    "pause_rate_per_min": "停顿频率",
    "pause_duration_p50": "停顿中位时长",
    "pause_duration_p90": "长停顿时长",
    # ★ 2026-11-22:`pause_ratio`("停顿占比")退役 —— 与 `voiced_ratio` 代数恒等
    #   (`1/voiced_ratio − 1`,残差 4.4e-16)。V 族由 12 条变 11 条。
    "response_latency_sec": "反应延迟",
    "pitch_median": "音高中位数",
    "pitch_range_semitone": "音高跨度",
    "pitch_variability_per_sec": "语调变化",
    "energy_median": "音量中位数",
    "energy_dynamics_per_sec": "能量变化速度",
    # T 族(转写文本)
    "lexical_diversity": "用词丰富度",
    "discourse_marker_rate": "连接词密度",
    "self_repair_rate": "自我修正频率",
    # ★ 2026-11-23 正名：`pause_structure_entropy` 实为停顿时长分布熵（见 _DIFFERENT_STATISTIC）
    "pause_structure_entropy": "停顿结构均匀度",
    "answer_detail_chars_p50": "回答详尽度",
}

#: 41 条指标的统一元数据表（单位 / 方向极性 / 0 值语义分类 / 所属族），供载荷、HTML 与前端共用。
#: zero_kind 五类枚举：
#:   upper_threshold — 上尾超阈事件（如微笑、视线偏离、手势爆发）
#:   lower_threshold — 下尾低于阈值事件（如皱眉、唇部压缩，越小越触发）
#:   event_count     — 计数/占比事件（如眨眼、停顿、连接词、自我修正、可见占比）
#:   zero_magnitude  — 差值/幅度/分位/波动类（0 表示实测变化幅度为零）
#:   zero_duration   — 时长/延迟/速率类（0 表示实测时长或延迟为零）
_INDICATOR_SPEC_TABLE: dict[str, tuple[str, str, str, str, str]] = {
    # name: (family, unit, polarity, polarity_note, zero_kind)
    "smile_ratio": ("F1_smile", "占比", "neutral", "微笑帧占有效观测帧的比例", "upper_threshold"),
    "smile_rate_per_min": ("F1_smile", "次/分钟", "neutral", "每分钟超过微笑阈值的次数", "upper_threshold"),
    "smile_duration_p50": ("F1_smile", "秒", "neutral", "单次微笑持续段长的中位数", "zero_duration"),
    "gaze_aversion_ratio_vs_baseline": ("F2_gaze", "占比", "lower_is_steadier", "相对本人中位基线偏离注视带的帧占比，越小越稳定", "upper_threshold"),
    "gaze_aversion_rate_per_min_vs_baseline": ("F2_gaze", "次/分钟", "lower_is_steadier", "每分钟视线偏离并回正的次数", "upper_threshold"),
    "gaze_stability": ("F2_gaze", "无量纲", "higher_is_steadier", "相对本场极差归一化的视线稳定度，越大越稳", "zero_magnitude"),
    "gaze_dwell_p50": ("F2_gaze", "秒", "higher_is_steadier", "单次注视带内连续停留时长中位数（秒）", "zero_duration"),
    "blink_rate_per_min": ("F3_blink", "次/分钟", "neutral", "每分钟检测到的眨眼上升沿次数", "event_count"),
    "blink_duration_mean": ("F3_blink", "秒", "neutral", "单次连续闭眼段平均时长", "zero_duration"),
    "head_motion_per_sec": ("F4_head", "归一化位移/秒", "lower_is_steadier", "基于 2D 像素比代理的每秒头部累计位移，越小越平稳", "zero_magnitude"),
    "head_away_ratio_vs_baseline": ("F4_head", "占比", "lower_is_steadier", "基于 2D 偏航代理超出本人基线的帧占比", "upper_threshold"),
    "brow_furrow_ratio_vs_baseline": ("F5_brow_lip", "占比", "neutral", "双眉内侧距离低于本人中位基线（皱眉）的帧占比", "lower_threshold"),
    "lip_compression_ratio_vs_baseline": ("F5_brow_lip", "占比", "neutral", "唇红厚度低于本人中位基线（抿唇）的帧占比", "lower_threshold"),
    "au_activation_any_ratio": ("F6_expressivity", "占比", "neutral", "至少一条可信面部动作单元超阈的帧占比", "upper_threshold"),
    "au_variability_per_sec": ("F6_expressivity", "1/秒", "neutral", "可信面部动作单元每秒累计变化幅度", "zero_magnitude"),
    "facial_asymmetry_p90": ("F8_symmetry", "无量纲", "lower_is_steadier", "面部左右不对称度 90 分位，数值越大越不对称（含头姿滚转分量）", "zero_magnitude"),
    "gesture_activity_per_sec": ("G1_activity", "1/秒", "neutral", "腕、肘、肩六关节按肩宽归一化的每秒累计抖动幅度", "zero_magnitude"),
    "gesture_burst_rate_per_min_vs_baseline": ("G1_activity", "次/分钟", "neutral", "超出本人常态基线的连续上肢动作段频率", "upper_threshold"),
    "posture_stability": ("G2_posture", "1/秒", "lower_is_steadier", "躯干与双肩归一化抖动幅度，数值越小表示躯干姿态越平稳", "zero_magnitude"),
    "postural_shift_rate_per_min_vs_baseline": ("G2_posture", "次/分钟", "lower_is_steadier", "超出本人基线的连续躯干姿态变换段频率", "upper_threshold"),
    "shrug_elevation_p90": ("G2_posture", "无量纲", "neutral", "相对本人中位肩高的 90 分位耸肩抬升幅度，0 表示与平时持平", "zero_magnitude"),
    "hand_openness_p50": ("G4_hand", "无量纲", "neutral", "双侧指尖距腕比掌长的逐帧均值中位数", "zero_magnitude"),
    "hand_visible_ratio_l": ("G3_visibility", "占比", "neutral", "左手在画面中可见的帧占比下界（协变量）", "event_count"),
    "hand_visible_ratio_r": ("G3_visibility", "占比", "neutral", "右手在画面中可见的帧占比下界（协变量）", "event_count"),
    "pose_valid_ratio": ("G3_visibility", "占比", "neutral", "躯干姿态关键点可读帧占比（协变量）", "event_count"),
    "speech_rate": ("V1_amount", "字/秒", "neutral", "全场混音转写字数除以有声时长（含面试官话语）", "zero_duration"),
    "voiced_ratio": ("V1_amount", "占比", "neutral", "有效基频检出（发声）帧占总帧比例", "event_count"),
    "pause_rate_per_min": ("V2_pause", "次/分钟", "neutral", "每有声分钟内检测到的有效停顿次数", "event_count"),
    "pause_duration_p50": ("V2_pause", "秒", "neutral", "有效停顿持续时长的中位数", "zero_duration"),
    "pause_duration_p90": ("V2_pause", "秒", "neutral", "较长停顿（90 分位）的持续时长", "zero_duration"),
    "pause_structure_entropy": ("V2_pause", "无量纲", "neutral", "停顿时长分布的归一化香农熵（0–1，越接近 1 表示各次停顿长短越均匀）", "zero_magnitude"),
    "response_latency_sec": ("V1_amount", "秒", "neutral", "提问结束到首次开口响应的延迟时间中位数", "zero_duration"),
    "answer_detail_chars_p50": ("V1_amount", "字", "neutral", "逐段转写非空白字符数的中位数", "event_count"),
    "pitch_median": ("V3_pitch", "Hz", "neutral", "有声段基频中位数", "zero_magnitude"),
    "pitch_range_semitone": ("V3_pitch", "半音", "neutral", "基频 95 与 5 分位之间的半音跨度", "zero_magnitude"),
    "pitch_variability_per_sec": ("V3_pitch", "半音/秒", "neutral", "基频每秒累计半音变化率", "zero_magnitude"),
    "energy_median": ("V4_energy", "RMS", "neutral", "录音均方根音量中位数（受麦克风增益影响，仅本场参考）", "zero_magnitude"),
    "energy_dynamics_per_sec": ("V4_energy", "dB/秒", "neutral", "音量包络每秒累计分贝变化率", "zero_magnitude"),
    "lexical_diversity": ("T1_lexical", "无量纲", "neutral", "100 字滑动窗口移动平均词汇丰富度（MATTR）", "zero_magnitude"),
    "discourse_marker_rate": ("T2_discourse", "次/百字", "neutral", "每百字中逻辑与衔接词出现的频次", "event_count"),
    "self_repair_rate": ("T3_repair", "次/百字", "neutral", "每百字中填充修正标记与紧邻重复出现的频次", "event_count"),
}


def build_indicator_meta() -> dict[str, dict[str, str]]:
    """生成全部 41 条指标的结构化中文元数据字典，供载荷、HTML 与前端共享。"""
    out: dict[str, dict[str, str]] = {}
    for name, label in INDICATOR_LABEL.items():
        fam, unit, polarity, polarity_note, zero_kind = _INDICATOR_SPEC_TABLE.get(
            name, ("", "", "neutral", "", "zero_magnitude")
        )
        out[name] = {
            "label": label,
            "unit": unit,
            "family": fam,
            "family_label": FAMILY_LABEL.get(fam, fam),
            "polarity": polarity,
            "polarity_note": polarity_note,
            "zero_kind": zero_kind,
        }
    return out


def explain_zero(name: str, value, n_valid, n_total=None) -> str:
    """为实测为 0（且已观测到有效样本）的指标生成精确人话说明，按 5 类语义区分。

    非 0 或未有效观测返回空串。
    """
    if value is None:
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if not (v == v) or abs(v) > 1e-9:
        return ""
    try:
        n = int(n_valid) if n_valid is not None else 0
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        return ""

    spec = _INDICATOR_SPEC_TABLE.get(name)
    zero_kind = spec[4] if spec else "upper_threshold"

    # 手势爆发/姿态变换的 n_valid 记的是段数（0 段时占位 1），展示帧数避免出现“看过 1 个有效观测”
    if name in ("gesture_burst_rate_per_min_vs_baseline", "postural_shift_rate_per_min_vs_baseline"):
        try:
            nt = int(n_total) if n_total is not None else 0
        except (TypeError, ValueError):
            nt = 0
        if nt > 1:
            return f"整场观测了 {nt} 帧，一次也没超过触发阈值（并非没测到）"
        return "整场观测期间未出现超过基线的连续动作段（并非没测到）"

    if zero_kind == "upper_threshold":
        return f"看过 {n} 个有效观测，一次也没超过阈值"
    if zero_kind == "lower_threshold":
        return f"看过 {n} 个有效观测，一次也没低于触发阈值"
    if zero_kind == "event_count":
        return f"在 {n} 个有效观测中未出现该事件（并非没测到）"
    if zero_kind == "zero_duration":
        return f"基于 {n} 个有效观测实测为 0（并非没测到）"
    return f"测得 {n} 个有效观测，实测变化幅度为 0（并非没测到）"


def indicator_label(name: str) -> str:
    """指标名 → 中文名。**未登记的原样返回键名** —— 宁可露英文,也不许吞掉。"""
    return INDICATOR_LABEL.get(name, name)


#: 置信「低」的**原因**,按指标名。置信度只说"有多信",不说"为什么只有这么信" ——
#: 后者才是读者要的那句话。**没有登记的指标不会瞎猜**,返回 `None`。
#:
#: ## 措辞规矩(2026-10-26)
#:
#: 使用者原话:「语音晦涩难懂,这怎么看」。实测正文里印着
#: 「证据单位是**段数**」「n_valid=1」—— 前者是 markdown 星号(交付物不渲染),
#: 后者是代码里的字段名。这一整段**直接进交付物正文**,所以:
#:
#: * 不写 `**`、反引号(交付物不渲染 markdown);
#: * 不写 `n_valid` / `n_total` 这类字段名 —— 换成「只测了一次」;
#: * 不点内部指标键 —— 换成中文名;
#: * 不用 `⟹` / `→` 这类记号。
#: 由 `tests/test_report_says_something_to_a_person.py` 扫正文。
LOW_CONFIDENCE_REASON: dict[str, str] = {
    # ★ 2026-10-18 补两条。它们一直是「低」但**没有登记原因** ——
    #   报告上只写「置信 低」,不写为什么低。置信度只说"有多信",
    #   不回答"为什么只有这么信",后者才是读者要的那句话。
    # ★ 两条手势率量的证据单位是"段"不是"帧"。13 场实测段数 1~62,
    #   所以它们**本来就该是「低」或「中」** —— 一条只有两三次事件的率,
    #   不该因为"看了 470 帧"就被判成证据充分。
    "gesture_burst_rate_per_min_vs_baseline": (
        "这一条数的是「动了几段」,不是「看了几帧」。整场只出现了两三次"
        "超过本人基线的连续动作段,样本太少,频率的不确定度就大。"
        "这不是算错:整场确实只动了这么几次。"),
    # ★ 2026-11-22 补三条(新指标)。三条的「低」都是**帧数不够**,不是"这一场没测到":
    #   它们的门槛是 30 帧(与同族一致),而取景会决定帧数 —— 手没入画、肩被裁掉、
    #   脸被裁掉,都会让帧数停在十几帧。读者的下一步动作因此是"下次把取景拍全",
    #   不是"再等一场"。措辞按 `LOW_CONFIDENCE_REASON` 的规矩:不写字段名、不点内部键。
    "shrug_elevation_p90": (
        "这一条要看到足够多帧的肩膀才算得稳,本场带肩膀读数的帧太少。"
        "这不是没耸肩,而是看得太短;另外它是相对本人平时的高度,不跨场比。"),
    "hand_openness_p50": (
        "本场手出现在画面里的帧太少,中位数只建立在很短的一段上。"
        "手没入画不等于手掌没张开,所以这条只当参考。"),
    "facial_asymmetry_p90": (
        "本场带面部左右差读数的帧太少,上分位估计不稳。"
        "另外这一条里混着头歪的分量,只在本场内比较,不跨场比。"),
    "postural_shift_rate_per_min_vs_baseline": (
        "同上:这一条数的是「动了几段」,整场只出现两三次,所以置信低。"
        "另外它与「手势爆发频率」高度相关 —— 13 场里有 9 场两者取到相同的段数,"
        "读的时候可以当成同一个信号。"),
    "speech_rate": (
        "整场只测了一次(总字数除以有声时长),没有第二个样本可以互相印证。"
        "素材里没有任何题目边界可用,这是素材决定的,不是算错了。"
        "另外:分子是全场混音的字数(含面试官),不是考生一个人的语速。"),
    "pause_rate_per_min": (
        "整场只测了一次(停顿段数除以有声分钟数),没有第二个样本可以互相印证。"
        "另外:同样基于全场混音,停顿里包含面试官与考生之间的换手间隙。"),
    "pitch_range_semitone": (
        "整场音频没有题目边界,这一条在整场只测了一次(开头与结尾的音高之差),"
        "没有第二个样本可以互相印证。这是素材决定的,不是算错了。"),
    "pitch_variability_per_sec": (
        "同上:整场只测了一次,分母就是这一次测量的时长,没有第二个样本可以互相印证。"),
    "energy_dynamics_per_sec": (
        "同上:整场只测了一次,没有第二个样本可以互相印证。"),
}

#: 这几条指标为什么**不**改成分窗多次测量 —— 这是使用者的裁定,写在这里免得后人反复问。
LOW_CONFIDENCE_DECISION = (
    "**裁定(2026-10-18):接受单次测量,如实报「低」,不改口径。**"
    "备选做法是按固定 30 秒切段、段间取中位数,以获得多次测量;"
    "但那把统计量从「按题/按场」换成了「按 30 秒窗」—— 名字、单位、可比性全变。"
    "本仓的规矩是不许**偷偷**换统计量(见 M4.9 关于 pitch_range_semitone 的约定),"
    "所以要么原样报低,要么另起一个新名字(如 *_per_30s)并同时改规格 —— 后者未做。")


def modality_summary(report: QualityReport) -> dict[str, dict]:
    """每个模态**到底算出了几条**指标 —— 报告头要说这句话,而它自己数不出来。

    ## 为什么必须有这个

    报告头看的是 `LogDataLoader.selected_sessions`,那**只描述日志文件**。
    而语音有第二个数据源(`voice_series`,整场录音逐帧重算),不经日志。
    实测场 `20260928_180131_8102`:语音日志 0 行,报告头写

        语音（面试） · 未读到数据（…若本场确实答过题，那才是文件损坏。）

    而同一份报告的正文里 V 族 **12 条有 10 条有值**。读者先看头,会以为这一场废了
    —— 它其实是本场最完整的模态。两个出口对同一个模态说了两句相反的话,
    这正是本仓第 15 条要消灭的形态。

    ## 为什么不在这里重算

    `build_quality_report` 已经把每条指标按族归好了(`families`),
    这里只做**汇总与措辞**,不碰任何原始数据。

    `source_note` 只在"值不是从日志来的"时非空 —— 那句话的作用是回答
    "日志都没有,这些数哪来的"。有日志时不写,免得读者以为日志没用上。
    """
    out: dict[str, dict] = {}
    for fam, info in report.families.items():
        modality = info.get("modality")
        if not modality:
            continue
        acc = out.setdefault(modality, {"n_total": 0, "n_measured": 0,
                                        "source_note": ""})
        acc["n_total"] += info["n"]
        acc["n_measured"] += info["n_measured"]

    # 语音的第二个数据源。`validate()` 早就为它开过豁免(`series_available`),
    # 这里把同一件事说给读者听。
    #
    # ⚠️ 只在**日志真的没有行**时才说这句。有日志时说「由整场录音重算得出」
    #    会反过来骗人(那些值其实是日志给的)。判据用 `coverage.voice.rows` ——
    #    那是"日志有几行",不是"有几行可用"。
    voice = out.get("voice")
    log_rows = (report.coverage.get("voice") or {}).get("rows", 0)
    if voice and voice["n_measured"] and report.series_available and not log_rows:
        voice["source_note"] = "由整场录音重算得出"
    return out


def explain_confidence(report: QualityReport, plain: bool = False) -> list[str]:
    """把"置信低"翻译成一句人话。只对**有值但置信低**的指标说。

    ## `plain`(2026-10-25 补):印名字还是印键

    * `plain=False`(默认)—— **审计版**,`{指标名}:{理由}`。核对口径的人查它;
    * `plain=True`          —— **人话版**,`{中文名}:{理由}`,交付物与界面印这一版。

    两版**条数、顺序、理由逐字相同**,差别只在名字(由
    `test_the_plain_version_keeps_every_entry_and_only_swaps_the_name` 守着)。

    ⚠️ 未登记中文名时**原样露出键名**,不许吞掉这一条 ——
    宁可露英文,也不能让读者以为这条指标没有限制。
    """
    out: list[str] = []
    for row in report.indicators:
        if row.get("confidence") != "低" or row.get("value") is None:
            continue
        name = row.get("name", "")
        reason = LOW_CONFIDENCE_REASON.get(name)
        if not reason:
            # 通用兜底：其余 31 条若以「低」出现，根据实际观测样本量动态给出人话说明（不含字段名）
            nv = row.get("n_valid")
            nt = row.get("n_total")
            if isinstance(nv, int) and isinstance(nt, int) and nt > 0:
                cov_pct = int(round(100 * nv / nt))
                reason = (
                    f"本场该指标有效观测仅 {nv} 项（占总记录约 {cov_pct}%），"
                    "低于稳健估计建议样本量，读数仅供本场参考。"
                )
        if reason:
            label = INDICATOR_LABEL.get(name, name) if plain else name
            out.append(f"{label}:{reason}")
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
