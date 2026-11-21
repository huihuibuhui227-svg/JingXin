# report_frontend/l1_face.py
"""**M4.3 F 族(面部)15 个 L1 指标**。

选 F 族先做,理由只有一条:**它不需要跨模态对齐**。全部由 face 日志 + 回答窗口得出,
所以不必等 M4.0 的墙钟问题定案就能独立验收。

## 15 条的产出状态:5 条相对量 + 9 条 τ 型 + 1 条分位量

| 类型 | 条数 | 出值条件 | τ 从哪来 |
|---|---|---|---|
| 相对量 / 率 | **5** | 只用"相对量/率",不需要任何外部阈值 | 不需要 τ |
| τ 型 · 声明常量 | **3** | `au12_smile` 超过声明常量 `AU12_SMILE_TAU`(0.263) | `l1_thresholds.RESOLUTION` 的 `declared` 分支 |
| τ 型 · 自比基线 | **5** | 阈值 = **本人中位数 ± `SELF_K`(3.0)·MAD**;名字带 `_vs_baseline` | `l1_thresholds.self_tau`(`self_baseline` 分支) |
| τ 型 · 两种并用 | **1** | 可信 AU 里**至少一条**过阈即算该帧激活 | 上面两种并用于不同列 |
| 分位量(F8) | **1** | `p90(1 − symmetry_score)`;不需要 τ(与 G 族的量同一取法) | 不需要 τ |

5 + 3 + 5 + 1 + 1 = 15,与 `_ALL_SPECS` / `indicator_names()` 逐条一致。

## 2026-11-22:退掉 2 条从未出过值的 `micro_exp_*`,补进 1 条 `facial_asymmetry_p90`

**退役的是这两条**:`micro_exp_rate_per_min` / `micro_exp_duration_mean_sec`。
它们**从来没有代码去算**,只是把"上游没采集"记成一行 `column_empty`。复核(2026-11-22,
55 份非 `.bak` face 日志 / 23355 行):

* `micro_exp_*` 四列在 55/55 份日志里都有**列名**,但非空单元合计 **23/23355 = 0.098%**
  (全部集中在极少数行);**212 个 (会话×列)** 是"列在、整份 0 非空";
* 两条指标在全语料里**出值 0 场**(任何一场都走不到 `measured()`)。

留下的 `column_empty` 缺口语义**没有消失**:它现在由 `facial_asymmetry_p90` 在
"`symmetry_score` 整份为空"时照实报出(实测全语料 10/53 场是这种)。至于"微表情没采集"
这件事,报告层仍有话说 —— 见 `l1_quality.GAP_EXPLANATION[column_empty]` 与
`assets` 里的采集清单;本文件不再用一条恒为空的指标行承载它(一条永不产出的指标
占着"38 条"里的一格,读者会把它读成一条测量)。

### 逐条对应(名字 → 策略)

* **相对量 / 率(5)**:`gaze_stability`、`blink_rate_per_min`、`blink_duration_mean`、
  `head_motion_per_sec`、`au_variability_per_sec`。这些是"视频内标准化过的比值"或
  "每分钟/每秒次数",阈值只进 `confidence` 的档位(`l1_indicators.confidence_for`)。
* **声明常量(3)**:`smile_ratio`、`smile_rate_per_min`、`smile_duration_p50` ——
  都吃 `au12_smile`,分界点 `0.263`。**这不是标定值**:它是全语料 41 场 16318 帧的
  p75,写死在 `l1_thresholds.AU12_SMILE_TAU`、可复核、非拟合;换语料就该重算重写。
  能用常量的理由:这一列的生产方**已经**除掉了本人基线(口宽 ÷ 全程 p10),
  场间差异是**真行为差异**(有人爱笑有人不笑),再叠一层本人基线就是双重归一化。
* **自比基线(5)**:`gaze_aversion_ratio_vs_baseline`、
  `gaze_aversion_rate_per_min_vs_baseline`、`head_away_ratio_vs_baseline`、
  `brow_furrow_ratio_vs_baseline`、`lip_compression_ratio_vs_baseline`。
  `gaze_direction_x` / `head_yaw` 先取绝对值再算(符号会翻,见 `RESOLUTION`);
  `au4_frown` / `au23_lip_compression` 取**低于**中位数一侧(越小越皱眉/越压唇)。
  为什么只能自比:这几列的**场间基线差 ≥ 场内动态范围**,一个统一常量量到的主要是
  "这个人的脸长什么样 / 机位偏哪边",不是"他在做什么"(实测表在 `l1_thresholds`)。
  ⚠️ **构念随之改变** ⟹ 名字必须带 `_vs_baseline`,不改名就是偷偷换构念。
* **两种并用(1)**:`au_activation_any_ratio` —— 6 条可信 AU 各自解析 τ
  (其中 5 条自比基线 + `au12_smile` 声明常量),**至少一条**过阈即算该帧有表情活动。

### `uncalibrated` 仍然可达 —— 但只在"τ 解析不出来"时

`_tau_block` 对每一列调 `l1_thresholds.resolve()`;只有下面三种情形才报
`gap="uncalibrated"`、`value=NaN`,**绝不猜一个数**:

1. 列**没登记**在 `l1_thresholds.RESOLUTION` 里(`resolve` 返回 `None`;
   现有 9 条全都登记了,这条留给将来的新列);
2. **自比基线算不出来** —— 有效帧 **< 5**,稳健位置/尺度不可信;
3. `au_activation_any_ratio` 可解析的 AU **< 2** 条(只有一条时,拿它算"至少一条
   激活"等于把该指标改名叫 `smile_ratio`)。

实测本仓 `data/logs/face_au_log_*.csv`(**55 份**非 `.bak` 日志,会话级单窗口):

* **5 条自比基线型**:39 份出值 / 15 份 `uncalibrated`(有效帧 < 5)/ 1 份 `no_samples`;
* **3 条声明常量型**:44 份出值,其余全部 `no_samples`(声明常量不需要估计,永不 `uncalibrated`)。

⚠️ 注意 `uncalibrated` 出现的原因**已经换了**:不再是"没有标定集",而是上面第 2 条。

## 自比基线**不是**"视频内分位顶替" —— 这条区别必须说清

原稿这里的禁令针对的是「拿被测量自身的**分位**当基准」:按构造,"超过本场 p90 的
占比"恒等于 10%,对每个人都 ≈ 常数、零信息量(设计文档 §5.1 规矩 1 的原话,
与当年 `speech_ratio` 退化的错误同类)。

现在的取法不同,两处都不同:

* **阈值**是**中位数 ± k·MAD**(稳健位置 + 尺度,不是分位);
* **指标**是"过阈帧占比",它**不恒等于任何固定值** —— 实测上面那 55 份日志的
  会话级结果:`brow_furrow_ratio_vs_baseline` 0.0000~0.2295、
  `lip_compression_ratio_vs_baseline` 0.0000~0.1359、
  `gaze_aversion_ratio_vs_baseline` 0.0000~0.2500、
  `head_away_ratio_vs_baseline` 0.0000~0.2744、
  `au_activation_any_ratio` 0.1256~0.8750。

有区分度,不是常数。判据始终是**"跨人基线差 vs 场内动态范围"**,不是"哪种便宜用哪种";
规矩与实测表写在 `l1_thresholds` 的模块 docstring 里,本文件只是消费者。

## 两个已实测的坑,本文件按坑写

1. **`blink_rate_per_min` 那一列不能直接用。** face 日志里已有一列同名,但它是
   **滑动 60 秒窗口**量(`video_pipeline.py:288-289`:`sum(t for t in blink_times if t > now-60)`),
   不是"全窗口的眨眼速率"。直接取均值会低估。本文件**由 `is_blink` 的上升沿重数**。
2. **`tension` 不出。** 它是个加权合成别名(F5-3),不是独立测量。
   出了它就等于把"复合分"从后门放回来 —— 与甲-22「折算层整段删除」相抵触。
   设计文档 §5.7 自己也注了"`tension` 是兼容用的合成别名,不计入实际测量指标"。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from report_frontend.l1_indicators import (
    GAP_COLUMN_EMPTY, GAP_MODALITY_MISSING, GAP_NO_SAMPLES, GAP_TIME_UNUSABLE,
    GAP_UNCALIBRATED, L1Indicator, measured, unavailable,
)
from report_frontend.event_windows import WindowSet

#: F6 `au_activation_any_ratio` 只在这些 AU 上算 —— 设计文档 §5.2 F6 点名的"可信 AU"。
#: 别的 AU 列要么贴地板、要么与另一列承载同一自由度(见 `l0_columns.json`)。
TRUSTED_AUS = ("au1_inner_brow_raise", "au6_cheek_raise", "au10_upper_lip_raise",
               "au12_smile", "au25_mouth_open", "au26_jaw_drop")

# ★ 2026-11-22 **删除 `MICRO_EXP_COLUMNS` 与 F7_micro 那两条指标**。
#
#   它原本是 F 族的"2 条不可出":`micro_exp_rate_per_min` /
#   `micro_exp_duration_mean_sec`。删它的理由不是"换条数据就能出",而是
#   **这两条从来没有任何代码算过** —— `_family_for_window` 里那段只是把
#   上游的空白翻译成一个缺口码,值永远是 NaN。
#
#   复核(2026-11-22,55 份非 `.bak` face 日志 / 23355 行,脚本见交付说明):
#     · 四列在 55/55 份日志里都有列名,非空单元合计 23 / 23355 = **0.098%**;
#     · "列在、整份 0 非空"的 (会话×列) 计数 **212**;
#     · 两条指标在全语料里**出值 0 场**。
#
#   为什么敢删:一条**永不产出**的指标占着"38 条"里的一格,读者会把这一格
#   读成"一条测量"。它承载的那句真话(微表情这一路没采集)**换了个载体**:
#   缺口码 `column_empty` 仍在(`facial_asymmetry_p90` 会用),而"微表情没采集"
#   的总说明在 `l1_quality.GAP_EXPLANATION` 与采集清单里,不靠一条空指标行。
#   替代它的是 F8 的 `facial_asymmetry_p90` —— 一条**真会出值**的测量。

# ★ 2026-11-21 **删除 `_TAU_DEPENDENT`**(死代码)。三条理由,逐条可复核:
#
#   1. **全仓无引用。** `grep -rn _TAU_DEPENDENT /home/zgy/JingXin/jingxin` 只命中
#      它自己的定义处(1 处),`tests/` 与 `tools/` 里没有;它是模块私有名(`_` 前缀),
#      仓外也不该有人据它编程。
#   2. **与 `_ALL_SPECS` 重复登记同一批 9 条**(名字 → 族 / 公式 / 依赖列),两份台账
#      必然漂移 —— 删前的现场就漂了:那段注释断言这 9 条"本轮不会出值"(`uncalibrated`),
#      而 `_tau_block` 从 2026-10-18 起就按 `l1_thresholds.RESOLUTION` 给它们**真出值**。
#      留着它就是留一句与实现相反的断言在代码里。
#   3. 输出顺序与"条数恒定 15"由 `_ALL_SPECS` / `_ORDER` 决定(`indicator_names()`
#      由 `_ORDER` 派生),删它**不改变任何输出** —— 由
#      `verify/verify_l1_face.py` 用两份真实日志逐条对比钉住。
#
#   台账(名字 → 族/公式/列)现在**唯一来源**是 `_ALL_SPECS`;新增指标只加那一处。

_FAMILY = "F"


# ── 小工具 ────────────────────────────────────────────────────────────────────

def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """列不存在 ⟹ 空 Series(调用点会因此拿不到有效观测,这是**对的**)。"""
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(dtype=float)


def _arr_or_none(df: pd.DataFrame, col: str) -> np.ndarray | None:
    """列不存在 ⟹ `None`。与 `_num` 的区别:空数组会被当成"有一列但全是空",
    在按列拼接时会同长度不符而炸掉(踩过)。"""
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _span_sec(t: np.ndarray) -> float:
    """有效观测的时间跨度(秒)。窗口内只有 1 个观测 ⟹ 0,速率类指标因此不可算。"""
    if len(t) < 2:
        return 0.0
    return float(np.nanmax(t) - np.nanmin(t))


def _rising_edges(flag: np.ndarray) -> int:
    """0→1 的上升沿数。**第一个元素为真不算上升沿**(没有"之前"可比)。

    `_update_blink_state` 里 `is_blink = ear < EAR_THRESHOLD` 是逐帧电平,
    不是事件;数"眨眼次数"必须数沿,不能数 `sum(is_blink)` —— 后者数的是帧数。
    """
    if flag.size == 0:
        return 0
    # 首帧的「之前」取它自己,否则首帧为真会被算成一次上升沿(踩过)。
    prev = np.concatenate([[flag[0]], flag[:-1].astype(np.int8)])
    return int(np.sum((flag.astype(np.int8) == 1) & (prev == 0)))


def _run_lengths(flag: np.ndarray, dt: np.ndarray) -> list[float]:
    """`flag==1` 的连续段时长(秒)。`dt[i]` 是第 i 帧到第 i-1 帧的间隔。"""
    out: list[float] = []
    start = None
    for i, on in enumerate(flag):
        if on and start is None:
            start = i
        elif not on and start is not None:
            out.append(float(np.nansum(dt[start + 1:i + 1])))
            start = None
    if start is not None:
        out.append(float(np.nansum(dt[start + 1:])))
    return out


# ── 主入口 ────────────────────────────────────────────────────────────────────

def compute_face_indicators(face: pd.DataFrame | None,
                            windows: WindowSet,
                            t_sec: pd.Series | None,
                            thresholds: dict[str, int] | None = None) -> list[L1Indicator]:
    """逐窗口算 F 族 15 条。

    `face`:face 日志(与 `t_sec` 同 index);`t_sec`:M4.0 给的时间轴(距会话原点秒);
    `windows`:M4.1 的回答窗口;`thresholds`:各族 G3 门槛,不给就用
    `evidence_gate._threshold_for` 的默认值。
    """
    # 三种"整族不可出"必须报成**不同**的原因 —— 它们指向完全不同的补救动作:
    #   · 没有这个模态        ⟹ 去查采集/回填(该不该有日志)
    #   · 有这个模态但没有时间轴 ⟹ 去查 M4.0(账本/行数对不上)
    #   · 有数据但这一窗口空    ⟹ 正常,该窗口本来就没观测
    if face is None:
        return _all_unavailable(windows, GAP_MODALITY_MISSING)
    if face.empty:
        return _all_unavailable(windows, GAP_NO_SAMPLES)
    if t_sec is None:
        return _all_unavailable(windows, GAP_TIME_UNUSABLE)
    if len(face) != len(t_sec):
        # 行数对不上 ⟹ 时间轴不可信 ⟹ 整族不可出。不按序硬配。
        return _all_unavailable(windows, GAP_TIME_UNUSABLE)

    thr = thresholds or {}
    out: list[L1Indicator] = []
    if not windows.windows:
        # ★ 没有任何窗口 ⟹ 按**会话级**算一遍,而不是"一条都不出"。
        #
        # 判据:零窗口有两种完全不同的原因,不该都退化成"无证据":
        #   · **没有提问台账**(正式场就是这样,实测 24/24 场没有 `questions.jsonl`)
        #     ⟹ 会话级统计仍然成立,15 条照样能算;
        #   · 有台账但 `ask_end` 全缺 ⟹ 同上,退到会话级仍然诚实。
        # 戊-15 待裁定的是**窗口粒度**(自然对话怎么切),不是"能不能算会话级"。
        # 粒度这件事由 `windows.gap` 如实写进报告,不靠"一条都不出"来表达。
        t = t_sec.to_numpy(dtype=float)
        return _family_for_window(face, t, "session", thr)

    for w in windows.windows:
        t = t_sec.to_numpy(dtype=float)
        mask = (~np.isnan(t)) & (t >= w.t_start) & (t < w.t_end)
        sub = face.loc[mask]
        tsub = t[mask]
        out.extend(_family_for_window(sub, tsub, f"answer:{w.index}", thr))

    return out


def _family_for_window(sub: pd.DataFrame, tsub: np.ndarray, window: str,
                       thr: dict[str, int]) -> list[L1Indicator]:
    n_total = len(sub)
    if n_total == 0:
        return [_un(name, meta, window, GAP_NO_SAMPLES) for name, meta in _ALL_SPECS.items()]

    got: dict[str, L1Indicator] = {}

    # ── 5 条可出数 ────────────────────────────────────────────────────────
    gaze = _num(sub, "gaze_direction_x").to_numpy(dtype=float)
    gaze_ok = ~np.isnan(gaze)
    if gaze_ok.sum() >= 2:
        span = float(np.nanmax(gaze[gaze_ok]) - np.nanmin(gaze[gaze_ok]))
        if span > 0:
            got["gaze_stability"] = measured(
                "gaze_stability", "F2_gaze",
                "1 − std(gaze_direction_x) / 视频内极差",
                ("gaze_direction_x",), window, "无量纲",
                value=1.0 - float(np.nanstd(gaze[gaze_ok])) / span,
                n_valid=int(gaze_ok.sum()), n_total=n_total,
                threshold=thr.get("gaze", 30))
        else:
            # 极差为 0 ⟹ 分母为 0。这是"场内心完全不动",不是"稳定度 1.0"。
            got["gaze_stability"] = unavailable(
                "gaze_stability", "F2_gaze", "1 − std/视频内极差",
                ("gaze_direction_x",), window, "无量纲", GAP_NO_SAMPLES, n_total)

    blink = _num(sub, "is_blink").to_numpy(dtype=float)
    blink_ok = ~np.isnan(blink)
    good_t = tsub[blink_ok]
    span = _span_sec(good_t)
    if blink_ok.sum() >= 2 and span > 0:
        flag = (blink[blink_ok] > 0).astype(np.int8)
        n_edges = _rising_edges(flag)
        got["blink_rate_per_min"] = measured(
            "blink_rate_per_min", "F3_blink",
            "is_blink 的 0→1 上升沿数 ÷ 有效分钟",
            ("is_blink", "timestamp"), window, "次/分钟",
            value=n_edges / (span / 60.0),
            n_valid=int(blink_ok.sum()), n_total=n_total,
            threshold=thr.get("blink", 20))

        dt = np.concatenate([[np.nan], np.diff(good_t)])
        runs = _run_lengths(flag, dt)
        if runs:
            got["blink_duration_mean"] = measured(
                "blink_duration_mean", "F3_blink", "连续 is_blink=1 段时长均值",
                ("is_blink", "timestamp"), window, "秒",
                value=float(np.mean(runs)),
                n_valid=int(blink_ok.sum()), n_total=n_total,
                threshold=thr.get("blink", 20))

    yaw = _arr_or_none(sub, "head_yaw")
    pitch = _arr_or_none(sub, "head_pitch")
    roll = _arr_or_none(sub, "head_roll")
    parts = [a for a in (yaw, pitch, roll) if a is not None]
    # ⚠️ `n_ok` 必须在**出值之前**算出来。踩过的坑:列存在但整列为 NaN 时,
    # `np.nansum` 给 0.0、除以跨度得 0.0 ⟹ 报出"头动 0 弧度/秒"且 n_valid=0。
    # 那是把"没观测"说成"没动",正是本仓反复消灭的那类错误。
    # 现在由 `L1Indicator` 的不变量在构造时直接拦下(实跑 20260927_195323_35e1 抓到的)。
    n_ok = max((int((~np.isnan(a)).sum()) for a in parts), default=0)
    if n_ok >= 2 and _span_sec(tsub) > 0:
        motion = np.zeros(len(sub))
        for a in parts:
            motion = motion + np.nan_to_num(np.abs(np.diff(a, prepend=np.nan)))
        got["head_motion_per_sec"] = measured(
            "head_motion_per_sec", "F4_head", "Σ(|Δyaw|+|Δpitch|+|Δroll|) ÷ 秒",
            ("head_yaw", "head_pitch"), window, "弧度/秒",
            value=float(np.nansum(motion)) / _span_sec(tsub),
            n_valid=n_ok, n_total=n_total,
            threshold=thr.get("head_pose", 30))

    au_cols = [c for c in TRUSTED_AUS if c in sub.columns]
    if au_cols:
        arr = np.column_stack([_num(sub, c).to_numpy(dtype=float) for c in au_cols])
        valid_rows = ~np.isnan(arr).any(axis=1)
        if valid_rows.sum() >= 2 and _span_sec(tsub[valid_rows]) > 0:
            d = np.abs(np.diff(arr[valid_rows], axis=0))
            got["au_variability_per_sec"] = measured(
                "au_variability_per_sec", "F6_expressivity",
                "Σ|Δau| over 可信 AU ÷ 秒",
                tuple(tuple(au_cols)), window, "1/秒",
                value=float(np.nansum(d)) / _span_sec(tsub[valid_rows]),
                n_valid=int(valid_rows.sum()), n_total=n_total,
                threshold=thr.get("au", 30))

    # ── τ 型:超阈占比 / 越阈次数 / 超阈段时长 ─────────────────────────────
    #
    # ★ 这一块原先一律报 `GAP_UNCALIBRATED`("τ 只能来自标定集")。2026-10-18 量了
    #   全语料 22277 行 / 41 场之后改了:卡住的不是"没有标定集",是**跨人基线差**
    #   (见 `l1_thresholds` 的实测表)。现在按列的性质取 τ:
    #   · 场间基线差 ≥ 场内动态范围的 → **自比基线**(中位数 ± 3·MAD),且指标改名;
    #   · 生产方已除掉本人基线的(`au12_smile`)→ **声明常量**,并写明"未标定"。
    _tau_block(sub, tsub, window, thr, got, n_total)

    # ── F8 面部不对称(2026-11-22 补,替换掉恒为空的 2 条 `micro_exp_*`)──────
    _asymmetry_block(sub, window, thr, got, n_total)

    # 缺的(列不存在等)也补成不可出,保证**条数恒定 15**
    for name, meta in _ALL_SPECS.items():
        if name not in got:
            got[name] = _un(name, meta, window, GAP_NO_SAMPLES)

    return [got[name] for name in _ORDER]


def _asymmetry_block(sub: pd.DataFrame, window: str, thr: dict[str, int],
                     got: dict[str, L1Indicator], n_total: int) -> None:
    """F8 `facial_asymmetry_p90` —— `p90(1 − symmetry_score)`,窗口内所有非空帧。

    ## 为什么不自比基线、也不要 τ

    这条是**分位量**,与 G 族的 `shrug_elevation_p90` / `hand_openness_p50`
    同一取法:量的就是"本人这一场里最不对称的那一小截有多不对称"。
    不需要任何外部阈值,也不需要本人基线 ⇒ `uncalibrated` 在这条上不可达。

    ## 值域很窄,所以**只能报上分位**,不能当绝对判据

    实测(2026-11-22)帧级 p10/p50/p90 = **0.935 / 0.966 / 0.988**(对称度),
    跨场中位 0.933~0.988 ⇒ 换算成不对称度是 0.012~0.067。窄值域 + 无人群参照
    ⇒ 绝不能把 0.034(=1−0.966)读成"对称度良好"。公式串里写清了这一点。

    ## 与头滚转不可分(构念边界,必须写在公式串里)

    会话级 `r(对称度中位, 手势 head_tilt_angle 中位) = −0.555`(n=29)⇒
    `1 − symmetry_score` 里混着头姿滚转的分量。这不是缺陷,是口径:
    它量的是"这一场脸在画面里有多歪",不是"面部肌肉有多不对称"。

    ## 挡掉 `video_pipeline` 的伪完美对称

    `video_pipeline.py:136` 在 zero-AU 支路上把 `symmetry_score` 写成 **1.0**,
    同时把 8 条 AU 全写成 **0.0**、`head_yaw`/`head_pitch` 写成 0。那是一帧
    "什么都检测不到"的占位帧,不是"完美对称的脸"。
    判据取**组合签名**(`symmetry_score == 1.0` **且** 全部 AU 恰为 0.0)——
    只判"等于 1.0"会把真的完美对称帧也扔了(虽然实测 1.0 的帧占比为 0,
    但那条支路存在,不能靠"今天没碰上"来保证)。
    实测(55 份日志 / 23355 行):被这条挡掉的帧 **0**。
    """
    name = "facial_asymmetry_p90"
    family, formula, cols = _ALL_SPECS[name]

    sym = _arr_or_none(sub, "symmetry_score")
    if sym is None:
        # 旧格式日志里没这一列 —— 与"手没入画"同一种处境:不是这一场没测到,
        # 是这一份日志的格式里没有它。
        got[name] = unavailable(name, family, formula, cols, window, "",
                                GAP_COLUMN_EMPTY, n_total)
        return

    ok = ~np.isnan(sym)
    # 伪完美对称的排除(见 docstring 末段)。只在条目列都真的存在时才有判据。
    au_present = [c for c in TRUSTED_AUS if c in sub.columns]
    if au_present:
        au_arr = np.column_stack([_num(sub, c).to_numpy(dtype=float) for c in au_present])
        zero_au_frame = (sym == 1.0) & (np.nan_to_num(au_arr, nan=0.0) == 0.0).all(axis=1)
        ok = ok & ~zero_au_frame

    n_ok = int(ok.sum())
    if not ok.any():
        # 列在、但这一窗口里一条非空都没有(实测全语料 10/53 场是整份空)。
        got[name] = unavailable(name, family, formula, cols, window, "",
                                GAP_COLUMN_EMPTY, n_total)
        return
    if n_ok < 2:
        got[name] = unavailable(name, family, formula, cols, window, "",
                                GAP_NO_SAMPLES, n_total)
        return

    asym = 1.0 - sym[ok]
    got[name] = measured(
        name, family, formula, cols, window, "无量纲(越大越不对称)",
        value=float(np.percentile(asym, 90)),
        n_valid=n_ok, n_total=n_total,
        threshold=thr.get("au", 30))


def _tau_block(sub: pd.DataFrame, tsub: np.ndarray, window: str,
               thr: dict[str, int], got: dict[str, L1Indicator], n_total: int) -> None:
    """9 条 τ 型指标。τ 解析不出来就照旧报 `uncalibrated` —— 不猜。"""
    from . import l1_thresholds

    def _series(col):
        return _num(sub, col) if col in sub.columns else None

    def _fill(name, col, kind):
        """`kind`:`ratio` 占比 / `rate` 越阈次数每分钟 / `duration` 超阈段中位时长。"""
        family, formula, cols = _ALL_SPECS[name]
        vals = _series(col)
        if vals is None:
            got[name] = unavailable(name, family, formula, tuple(cols), window, "",
                                    GAP_COLUMN_EMPTY, n_total)
            return
        tau = l1_thresholds.resolve(col, vals)
        if tau is None or tau.value is None:
            got[name] = unavailable(name, family, formula, tuple(cols), window, "",
                                    GAP_UNCALIBRATED, n_total)
            return
        m = tau.mask(vals)
        n_ok = int((~np.isnan(pd.to_numeric(vals, errors="coerce").to_numpy(dtype=float))).sum())
        if n_ok < 2:
            got[name] = unavailable(name, family, formula, tuple(cols), window, "",
                                    GAP_NO_SAMPLES, n_total)
            return
        if kind == "ratio":
            got[name] = measured(name, family, formula, tuple(cols), window, "占比",
                                 value=float(m.sum()) / n_ok,
                                 n_valid=n_ok, n_total=n_total,
                                 threshold=thr.get("au", 30))
        elif kind == "rate":
            span = _span_sec(tsub)
            if span <= 0:
                got[name] = unavailable(name, family, formula, tuple(cols), window, "",
                                        GAP_TIME_UNUSABLE, n_total)
                return
            got[name] = measured(name, family, formula, tuple(cols), window, "次/分钟",
                                 value=_rising_edges(m.astype(np.int8)) / (span / 60.0),
                                 n_valid=n_ok, n_total=n_total,
                                 threshold=thr.get("au", 30))
        else:
            dt = np.concatenate([[np.nan], np.diff(tsub)])
            runs = _run_lengths(m.astype(np.int8), dt)
            if not runs:
                got[name] = unavailable(name, family, formula, tuple(cols), window, "",
                                        GAP_NO_SAMPLES, n_total)
                return
            got[name] = measured(name, family, formula, tuple(cols), window, "秒",
                                 value=float(np.median(runs)),
                                 n_valid=n_ok, n_total=n_total,
                                 threshold=thr.get("au", 30))

    _fill("smile_ratio", "au12_smile", "ratio")
    _fill("smile_rate_per_min", "au12_smile", "rate")
    _fill("smile_duration_p50", "au12_smile", "duration")
    _fill("gaze_aversion_ratio_vs_baseline", "gaze_direction_x", "ratio")
    _fill("gaze_aversion_rate_per_min_vs_baseline", "gaze_direction_x", "rate")
    _fill("head_away_ratio_vs_baseline", "head_yaw", "ratio")
    _fill("brow_furrow_ratio_vs_baseline", "au4_frown", "ratio")
    _fill("lip_compression_ratio_vs_baseline", "au23_lip_compression", "ratio")

    # ── `au_activation_any_ratio`(原 `au_activity_index`,2026-10-18 换构念)──
    #
    # ★ 为什么换:**原构念(六条 AU 的联合 AND)实测退化**。三场 A 档实测:
    #
    #     单条 AU 激活占比  au1 .053~.092  au6 .019~.074  au10 .023~.446
    #                       au12 .000~.090  au25 .016~.414  au26 .016~.117
    #     联合 AND          0.000000 / 0.000000 / 0.000000   ← 恒为 0
    #     至少一个          0.325 / 0.514 / 0.474           ← 有区分度
    #
    #   六条各 2%~45% 的事件取交集,乘积趋 0 ⟹ 报出来会是一个**看着像测量、
    #   其实恒为 0** 的数。改成"至少一个"才有区分度。
    #   ⚠️ 构念变了 ⟹ **名字必须换**(与 `_vs_baseline` 同一条规矩):
    #   留着旧名字会让下一个人以为"联合占比"还能用。
    name = "au_activation_any_ratio"
    family, formula, cols = _ALL_SPECS[name]
    taus: dict[str, object] = {}
    for c in TRUSTED_AUS:
        if c not in sub.columns:
            continue
        t = l1_thresholds.resolve(c, sub[c])
        if t is None:
            continue
        # ⚠️ 这里曾有一条"τ 退化(MAD=0 ⟹ τ=中位数)就跳过该 AU"的检查。
        #   变异验证时**删掉它测试照样绿** ⟹ 它是死代码,已删。原因:`Tau.mask`
        #   用的是**严格** `>`,MAD=0 时"等于中位数的帧"并不算激活,所以不会出现
        #   "一半帧都算激活"那种退化。留着只会让人以为存在一个不存在的问题。
        taus[c] = t
    if len(taus) < 2:
        got[name] = unavailable(
            name, family, formula, tuple(cols), window, "", GAP_UNCALIBRATED, n_total)
        return
    masks = [taus[c].mask(sub[c]) for c in taus]
    any_active = np.logical_or.reduce(masks)
    arr = np.column_stack([_num(sub, c).to_numpy(dtype=float) for c in taus])
    # 分母 = "**至少有一条** AU 观测到"的帧。
    #
    # ⚠️ 这里原来写的是 `~np.isnan(arr).any(axis=1)` —— 那个表达式数的是
    #   "**全部** AU 都非空"的帧(any 在 nan 上取,再取反),与公式串说的
    #   "至少有一条观测到"**相反**。后果:分子(任一条激活,含部分观测帧)
    #   会大于分母,占比能 >1 —— 实测造出过 value=2.0。
    #   真数据上没触发(13 场 0.13~0.55),属潜伏;换采集或回填策略就会炸。
    n_ok = int((~np.isnan(arr)).any(axis=1).sum())
    if n_ok < 2:
        got[name] = unavailable(name, family, formula, tuple(cols), window, "",
                                GAP_NO_SAMPLES, n_total)
        return
    got[name] = measured(name, family, formula, tuple(cols), window, "占比",
                         value=float(any_active.sum()) / n_ok, n_valid=n_ok,
                         n_total=n_total, threshold=thr.get("au", 30))


# ── 规格表 ────────────────────────────────────────────────────────────────────

#: 名字 → (族, 公式, 依赖列)。**顺序即输出顺序**,由 `_ORDER` 固定。
_ALL_SPECS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "smile_ratio": ("F1_smile", "au12 > 0.263(声明常量,非标定) 帧占比", ("au12_smile",)),
    "smile_rate_per_min": ("F1_smile", "au12 穿越 0.263(声明常量,非标定) 的上升沿数 ÷ 分钟", ("au12_smile",)),
    "smile_duration_p50": ("F1_smile", "连续超 0.263(声明常量,非标定) 段时长的中位数", ("au12_smile",)),
    "gaze_aversion_ratio_vs_baseline": ("F2_gaze", "|gaze| > 本人中位数+3·MAD 的帧占比(自比基线)",
                            ("gaze_direction_x",)),
    "gaze_aversion_rate_per_min_vs_baseline": ("F2_gaze", "偏离→回正次数 ÷ 分钟(阈值 = 本人|视线|中位数+3·MAD)", ("gaze_direction_x",)),
    "gaze_stability": ("F2_gaze", "1 − std(gaze_direction_x) / 视频内极差",
                       ("gaze_direction_x",)),
    "blink_rate_per_min": ("F3_blink", "is_blink 的 0→1 上升沿数 ÷ 有效分钟",
                           ("is_blink", "timestamp")),
    "blink_duration_mean": ("F3_blink", "连续 is_blink=1 段时长均值", ("is_blink", "timestamp")),
    "head_motion_per_sec": ("F4_head", "Σ(|Δyaw|+|Δpitch|+|Δroll|) ÷ 秒",
                            ("head_yaw", "head_pitch")),
    "head_away_ratio_vs_baseline": ("F4_head", "|head_yaw| > 本人中位数+3·MAD 的帧占比(自比基线)", ("head_yaw",)),
    "brow_furrow_ratio_vs_baseline": ("F5_tension", "au4 < 本人中位数−3·MAD 的帧占比(自比基线;眉内距越小越皱)", ("au4_frown",)),
    "lip_compression_ratio_vs_baseline": ("F5_tension", "au23 < 本人中位数−3·MAD 的帧占比(自比基线;唇红越薄越压)", ("au23_lip_compression",)),
    "au_activation_any_ratio": (
        "F6_expressivity",
        "**至少一条**可信 AU 超本人 τ 的帧占比(τ:5 条自比基线 + au12_smile 声明常量 0.263)", TRUSTED_AUS),
    "au_variability_per_sec": ("F6_expressivity", "Σ|Δau| over 可信 AU ÷ 秒", TRUSTED_AUS),
    # ★ 2026-11-22 新增(替换掉 `micro_exp_rate_per_min` / `micro_exp_duration_mean_sec`)。
    #   统计量与 G 族两条新指标同一取法:分位量 + 帧级最小样本 30 帧 + 无 τ。
    "facial_asymmetry_p90": (
        "F8_symmetry",
        "p90(1 − symmetry_score);symmetry_score = 1 − (|Δ眼中心y| + |Δ嘴角y|) ÷ 面部轮廓高"
        "(尺度不变;**含头姿滚转分量**,不与 head_tilt 独立);值域窄 ⇒ 只作相对量,"
        "不得当作「对称度良好」的绝对判据", ("symmetry_score",)),
}

_ORDER = tuple(_ALL_SPECS)


def _un(name: str, meta: tuple[str, str, tuple[str, ...]], window: str, gap: str) -> L1Indicator:
    family, formula, cols = meta
    return unavailable(name, family, formula, tuple(cols), window, "", gap)


def _all_unavailable(windows: WindowSet, gap: str) -> list[L1Indicator]:
    out: list[L1Indicator] = []
    for w in windows.windows:
        out.extend(_un(n, meta, f"answer:{w.index}", gap) for n, meta in _ALL_SPECS.items())
    if not windows.windows:
        # 没有任何窗口:仍然把 15 条以 `session` 名义报出来 —— 不列出来等于让读的人
        # 以为 F 族不存在。
        out.extend(_un(n, meta, "session", gap) for n, meta in _ALL_SPECS.items())
    return out


def indicator_names() -> tuple[str, ...]:
    """F 族对外承诺的 15 个名字。测试用它钉"条数恒定"。

    2026-11-22 由 16 变 15:退 2 条恒空的 `micro_exp_*`、补 1 条 `facial_asymmetry_p90`。
    """
    return _ORDER
