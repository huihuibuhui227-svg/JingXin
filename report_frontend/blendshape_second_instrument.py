# report_frontend/blendshape_second_instrument.py
"""**第二套仪器**的数值 —— 用 blendshape 列把少数面部构念独立算一遍。

## 为什么要有这个文件

`l1_face.py` 的 16 条 F 族指标**全部**只读几何列(`gaze_*` / `head_*` / `is_blink` /
`au_*`)。日志里明明躺着 52 列 `bs_*`(FaceLandmarker 的模型回归输出,见
`face_expression/models/blendshapes.py`),却一条都没用上 ⟹ 报告里说"本系统同时存在
两套面部测量"这句话在代码上**是空的**。本文件把能一一对应的那几个构念用 `bs_*`
再算一遍,让"第二套仪器"有数值可引用。

## 三件本文件**不做**的事

1. **不下"两套一致 / 可互换"的结论。** 一致性数字来自离线实测(32 场 / 389 条,
   `07_…/两套面部仪器一致性_v2.md`):12 个构念里仅 1 个达 ρ ≥ 0.7
   (`au7_eye_squeeze` ρ=0.752);`au12_smile` 0.551、`au4_frown` 0.210,
   `au10` / `au15` 贴地板率 52% / 50%。**那些数不在这里重算、也不在这里改写。**
2. **不把这些值放进主表。** 见 `patch/06_应用.md`:只有 1/12 达 ρ ≥ 0.7 ⟹
   第二套数只进"附录 / 交叉核对"区。
3. **不替几何侧改构念。** 两侧的构念名、阈值来源、定义方向都各写各的;
   口径不同就在 `threshold_basis` 里说清(见下"四个构念的口径对照")。

## 四个构念的口径对照(读表前必须先看这张)

| 构念 | 几何侧(`l1_face`) | 本文件(blendshape 侧) | 能否直接对比 |
|---|---|---|---|
| `smile_ratio` | `au12_smile > 0.263` 帧占比 | `(bs_mouthSmileL+R)/2 > 0.5` 帧占比 | **口径相近**(都是绝对阈值),可并列读 |
| `brow_furrow_ratio` | `brow_furrow_ratio_vs_baseline`(**自比基线**:比他本人平时更皱) | 绝对阈值 `> 0.5` 的帧占比 | **不能**。一个是"相对他自己",一个是"绝对有没有",构念不同 |
| `blink_rate_per_min` | `is_blink` 上升沿 ÷ 有效分钟(阈值来自 EAR) | `(bs_eyeBlinkL+R)/2 > 0.5` 上升沿 ÷ 有效分钟 | 构念相同,阈值来源不同 |
| `au_activation_any_ratio` | 6 条"可信 AU"各自的**自比基线** τ | 51 列各取**本文件 p90**,任一列超阈 | **不能**。AU 集合与 τ 来源都不同,只能做量级对照 |

⚠️ 还有一条容易误读的:**离线那份 ρ 是"逐帧序列"的秩一致性**(`au4_frown` 帧序列 vs
`bs_browDown` 帧序列),**不是**本文件这两个"占比"之间的一致性。所以
"`au4_frown` 的 ρ=0.210"不能拿来当 `brow_furrow_ratio` 这一列数值的验证结论 ——
它验证的是**序列**,这里报的是**汇总占比**。附录里必须两个都写清。

## 阈值从哪来(每个都写明依据,都**未标定**)

`DECLARED_THRESHOLDS` 里三个 0.5 是**声明常量**,出处在各条 `THRESHOLD_BASIS`:

* 为什么不是 0.5 的替代品 —— 中位数:实测 `bs_mouthSmileLeft` 全语料**中位数 = 0.000**
  (65.4% 的帧**恒为 0**),中位数当 τ 等于把"几乎全部帧"判成微笑;`bs_browDownLeft`
  中位数 0.004 同理。**零膨胀列的中位数不是一个可用分界点。**
* 为什么不是"本文件中位数":那是**自比基线**,构念会从"绝对有没有在笑"悄悄变成
  "比他自己的中位更笑" —— 而构念变了名字没变,正是 `l1_thresholds` 规矩 2 要禁的事。
* `au_activation_any_ratio` 用的是**逐列 p90(本文件内)**——那是题目指定的口径,
  但它按构造就让占比接近 1(51 列里"任一列超自身 p90"),**它是分辨率自检量,
  不是行为量**。见该条 `THRESHOLD_BASIS` 与 `06_应用.md` 的建议。

⚠️ 这些 0.5 与 `au12_smile` 的 0.263 一样:**不是标定值**(没有标定集,M5 卡在许可 +
伦理审查),只是写明的、可复核的、非拟合的分界点。换语料就该重算并重写这几行。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

# ── 构念 ──────────────────────────────────────────────────────────────────────

SMILE = "smile_ratio"
BROW = "brow_furrow_ratio"
BLINK = "blink_rate_per_min"
ANY_AU = "au_activation_any_ratio"

#: 实现并验证过的四个构念。**只有这四个** —— 别的构念在两侧之间没有一一对应
#: (例如 `gaze_stability` 在 blendshape 里没有构念,`au10_upper_lip_raise` 几何侧
#: 贴地板 52%,再用 bs 算一遍只会多一个同样贴地板的数)。
CONSTRUCTS: tuple[str, ...] = (SMILE, BROW, BLINK, ANY_AU)

#: 构念 → 直接依赖的 blendshape 列(除 `ANY_AU` 之外)。
CONSTRUCT_COLUMNS: dict[str, tuple[str, ...]] = {
    SMILE: ("bs_mouthSmileLeft", "bs_mouthSmileRight"),
    BROW: ("bs_browDownLeft", "bs_browDownRight"),
    BLINK: ("bs_eyeBlinkLeft", "bs_eyeBlinkRight"),
}

#: 模型的"中性"类 —— `blendshapes.py` 说得很清楚:它**不是肌肉动作**。
#: 把它算进"任一 bs_ 激活"等于把"没有表情"也算成"有表情" ⟹ 逐列激活构念里排除它。
#: (名字带双下划线是因为 `blend_column("_neutral")` = `bs_` + `_neutral`。)
NEUTRAL_COLUMN = "bs__neutral"

#: 声明常量(见模块 docstring "阈值从哪来")。
DECLARED_THRESHOLDS: dict[str, float] = {
    SMILE: 0.5,
    BROW: 0.5,
    BLINK: 0.5,
}

#: `ANY_AU` 的逐列分位点。
ANY_AU_QUANTILE: float = 0.90

#: key = 构念(三个声明常量构念);`ANY_AU` 的阈值不是一个数值,见 `THRESHOLD_BASIS`。
#: 这些字符串会原样进报告的附录 —— 写给人看,不写给人猜。
THRESHOLD_BASIS: dict[str, str] = {
    SMILE: (
        "声明常量 0.5,未标定。依据:①blendshape 是 FaceLandmarker 输出的 0–1 激活分,"
        "0.5 是「系数过半即判定该动作出现」的模型系数中点,与 au12_smile 用的 0.263 同类"
        "(都是写明的非拟合分界点);②不取中位数:全语料 44 场 / 16872 非空帧实测 "
        "bs_mouthSmileLeft 中位数 0.000,p75 0.001,65.4% 的帧恒为 0(右列 66.4%)——"
        "零膨胀列的中位数落在 0 上,拿它当 τ 会把几乎全部帧判成微笑;"
        "③不取本文件中位数:那是自比基线,构念会变成「比他平时更笑」,与本构念问的"
        "「绝对有没有在笑」不是一个东西(l1_thresholds 规矩 2);"
        "④该常量下的实测发生率只有 1.65%(左)/ 1.19%(右),p90=0.020、p95=0.112 "
        "⟹ 本构念只能读成「明确微笑的帧占比」,**不能**当微笑强度。"
    ),
    BROW: (
        "声明常量 0.5,未标定(理由同 smile_ratio 的 ①②③)。"
        "★ 必须同时读这个数:全语料实测 bs_browDownLeft 仅 0.11% 的帧 > 0.5"
        "(右列 0.03%),p90 = 0.040 —— 该通道在本语料上**几乎从不激活**。"
        "所以 brow_furrow_ratio 接近 0 是「这一路的分辨力不足以判皱眉」的证据,"
        "**不能**读成「此人从不皱眉」。离线实测几何侧 au4_frown 与 bs_browDown 的逐帧 "
        "ρ = 0.210,与这条分辨率结论方向一致(但 ρ 说的是序列,不是本文件的占比)。"
    ),
    BLINK: (
        "声明常量 0.5,未标定:双侧 (L+R)/2 > 0.5 记该帧闭眼,数 0→1 上升沿 ÷ 有效分钟。"
        "依据:①0.5 同上(模型系数中点);②该通道有分辨力(不像 mouthSmile 贴 0):"
        "全语料 > 0.5 的帧占 13.05%(左)/ 10.73%(右),中位数 0.186 / 0.142,"
        "无零膨胀;③上升沿定义与 l1_face._rising_edges **逐字对齐**(首帧为真不算上升沿),"
        "这样两侧数的是同一件事;④与几何侧的差别只在阈值来源:几何侧来自 EAR "
        "(avg_ear < 阈值),这里来自模型系数,两路不是同一个测量原理。"
    ),
    ANY_AU: (
        "Q = 0.90,在本文件内**逐列**取该列 p90(51 列各一个,"
        "排除 bs__neutral —— 模型的「中性」类不是肌肉动作),"
        "任一侧均值式无关:逐列各自超自身 p90 即记该帧「有激活」。"
        "⚠️ 因为 τ 来自本文件分布,单列超阈概率≈10%,51 列取「任一」⟹ 占比按构造趋近 1。"
        "**它是分辨率自检量(这一场的 blendshape 通道还活着吗),不是行为量**,"
        "不能读成「表情丰富度」,也不能与几何侧 au_activation_any_ratio 的数值直接比。"
    ),
}


# ── 小工具 ────────────────────────────────────────────────────────────────────

def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """列不存在 ⟹ 空 Series(调用点因此报"算不出来",而不是拿缺列当 0)。"""
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _paired_mean(df: pd.DataFrame, cols: tuple[str, ...]) -> tuple[np.ndarray, int]:
    """双侧均值,**只取两侧都有观测的帧**。

    为什么要配对:构念是"双侧嘴角上翘";只有一侧有值时按单侧算,等于在
    分母里混进"半个观测",而且左右列在采集里的缺失并不保证同步。
    实测 44 场里左右两侧的非空帧数恰好相等,所以配对在本语料上不丢帧 ——
    但这是**数据碰巧**,不是可以依赖的性质。
    """
    vals = [_num(df, c).to_numpy(dtype=float) for c in cols]
    if any(v.size == 0 for v in vals):
        return np.array([], dtype=float), 0
    arr = np.column_stack(vals)
    ok = ~np.isnan(arr).any(axis=1)
    if not ok.any():
        return np.array([], dtype=float), 0
    return arr[ok].mean(axis=1), int(ok.sum())


def _rising_edges(flag: np.ndarray) -> int:
    """0→1 的上升沿数。**首帧为真不算上升沿**(没有"之前"可比)。

    与 `l1_face._rising_edges` 同一个定义 —— 两侧数出来的"眨眼次数"必须是同一件事,
    否则两个速率天生差一个边缘帧,读的人会以为是仪器分歧。
    """
    if flag.size == 0:
        return 0
    prev = np.concatenate([[flag[0]], flag[:-1].astype(np.int8)])
    return int(np.sum((flag.astype(np.int8) == 1) & (prev == 0)))


def _result(construct: str, value: float, n_frames: int,
            threshold_used: float | None, threshold_basis: str) -> dict:
    """统一的返回形状。五个键,**不多不少**。"""
    return {
        "construct": construct,
        "blendshape_value": float(value),
        "n_frames": int(n_frames),
        "threshold_used": None if threshold_used is None else float(threshold_used),
        "threshold_basis": threshold_basis,
    }


def _unavailable(construct: str, why: str) -> dict:
    """算不出来就报 `NaN` + 原因 —— 不拿 0 冒充"没有这个行为"。"""
    return _result(construct, math.nan, 0, None, f"算不出来:{why}")


# ── 主入口 ────────────────────────────────────────────────────────────────────

def compute_second_instrument(csv_path: Path, construct: str) -> dict:
    """在**一份** `data/logs/face_au_log_*.csv` 上算一个构念的 blendshape 侧数值。

    返回五个键:`construct` / `blendshape_value` / `n_frames` / `threshold_used` /
    `threshold_basis`。语义:

    * `blendshape_value`:比率类构造给占比(0–1),`blink_rate_per_min` 给次/分钟;
    * `n_frames`:**参与计算的配对有效帧数**(不是文件行数)。分母就是这个数,
      读的人可以自己乘回去复核;
    * `threshold_used`:三个声明常量构念给**数值阈值**;`au_activation_any_ratio`
      给**分位水平 0.90**(它逐列取值,"有没有一个数值"这一栏只能记分位水平),
      见其 `threshold_basis`;
    * 数据不够(缺列 / 全空 / 跨越时间为 0)⟹ `blendshape_value = NaN` 且
      `threshold_basis` 以"算不出来:"开头。**不抛异常** —— 采集缺列是常态,
      抛异常会让整份验证脚本停在第 1 个文件。

    `construct` 不在 `CONSTRUCTS` 里 ⟹ `ValueError`(那是调用方的编程错误,
    与数据无关)。
    """
    if construct not in CONSTRUCTS:
        raise ValueError(f"construct 只能是 {CONSTRUCTS},收到 {construct!r}")

    df = _read_csv(Path(csv_path))

    if construct == ANY_AU:
        return _any_au(df, construct)

    cols = CONSTRUCT_COLUMNS[construct]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return _unavailable(construct, f"缺列 {missing}")
    series, n_frames = _paired_mean(df, cols)
    if n_frames == 0:
        return _unavailable(construct, f"{list(cols)} 全为空 —— 本场没有 blendshape 观测")

    tau = DECLARED_THRESHOLDS[construct]

    if construct in (SMILE, BROW):
        flag = series > tau
        return _result(construct, float(flag.sum()) / n_frames, n_frames,
                       tau, THRESHOLD_BASIS[construct])

    # BLINK:上升沿 ÷ 有效分钟。
    # ⚠️ 这里**重新**取掩码,不复用上面那个 `series`:分母必须是"双侧系数**和**时刻
    # 都拿得到"的那批帧。复用会让 `flag`(长度 n_frames)与掩码(长度 = 行数)对不上
    # —— 踩过一次 IndexError;更糟的写法是"对得上但错位",那会静默算出错的速率。
    # 掩码用 `.all(...)`(双侧配对),与 smile/brow 的口径一致:单侧有值的帧不算
    # "闭眼",也不算进分母 —— 不拿"半个观测"当一整帧。
    if "timestamp" not in df.columns:
        return _unavailable(construct, "缺 timestamp 列,算不出分钟数")
    t = _num(df, "timestamp").to_numpy(dtype=float)
    arr = np.column_stack([_num(df, c).to_numpy(dtype=float) for c in cols])
    ok = (~np.isnan(arr)).all(axis=1) & ~np.isnan(t)
    n_ok = int(ok.sum())
    if n_ok < 2:
        return _unavailable(construct, f"双侧系数与时间戳都有效的帧只有 {n_ok} 帧,不足以算速率")
    span = float(t[ok].max() - t[ok].min())
    if span <= 0:
        return _unavailable(construct, "所有帧落在同一时刻,速率无定义")
    closed = arr[ok].mean(axis=1) > tau
    n_edges = _rising_edges(closed.astype(np.int8))
    return _result(construct, n_edges / (span / 60.0), n_ok,
                   tau, THRESHOLD_BASIS[construct])


def _any_au(df: pd.DataFrame, construct: str) -> dict:
    """任一 `bs_*`(除中性类)超过**该列本文件 p90** 的帧占比。"""
    cols = [c for c in df.columns
            if str(c).startswith("bs_") and c != NEUTRAL_COLUMN]
    if not cols:
        return _unavailable(construct, "这份日志没有任何 bs_* 列")

    arr = np.column_stack([_num(df, c).to_numpy(dtype=float) for c in cols])
    # 分母 = "**至少有一列**观测到"的帧。
    #
    # ⚠️ 这里不能写 `~np.isnan(arr).any(axis=1)` —— 那个表达式数的是"**全部**列
    #   都非空"的帧。实测 `bs_*` 是按块缺失的(同一场的 51 列一起空/一起有),
    #   所以在本语料上两种写法碰巧同值;但换采集或换回填策略,前者会让分母缩到 0
    #   而**不报错**。`l1_face.au_activation_any_ratio` 上真出过这个 bug(占比 >1),
    #   这里按修好后的写法对齐。
    ok = (~np.isnan(arr)).any(axis=1)
    n_ok = int(ok.sum())
    if n_ok == 0:
        return _unavailable(construct, "所有 bs_* 列全为空 —— 本场没有 blendshape 观测")

    sub = arr[ok]
    # 逐列 p90。常数列(如恒 0)的 p90 就是该常数,严格 `>` 因此不会把恒 0 的帧判成激活
    # —— 与 `l1_thresholds.Tau.mask` 用严格 `>` 的理由相同(见 l1_face 里删掉的那段
    # "τ 退化就跳过该列"的死代码)。
    #
    # 逐列自己算,不用 `np.nanquantile(..., axis=0)`:整列为空的列会让它吐
    # `All-NaN slice encountered` 的 RuntimeWarning(本仓 55 份日志里 11 份整块 bs
    # 为空,这是常态而不是异常)。整列为空 ⟹ q = NaN ⟹ 比较恒为 False ⟹ 该列既不
    # 贡献激活,也不污染分母。
    q = np.full(sub.shape[1], np.nan, dtype=float)
    for j in range(sub.shape[1]):
        col = sub[:, j]
        ok_j = ~np.isnan(col)
        if ok_j.any():
            q[j] = float(np.quantile(col[ok_j], ANY_AU_QUANTILE))
    hit = (sub > q).any(axis=1)
    return _result(construct, float(hit.sum()) / n_ok, n_ok,
                   ANY_AU_QUANTILE, THRESHOLD_BASIS[construct])


def _read_csv(path: Path) -> pd.DataFrame:
    """读一份 face 日志。空文件 / 只有表头 ⟹ 空 DataFrame(调用方报"算不出来")。"""
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def compute_all(csv_path: Path) -> dict[str, dict]:
    """一份 CSV 上把四个构念都算一遍。`verify` 与附录生成脚本用它。"""
    return {c: compute_second_instrument(csv_path, c) for c in CONSTRUCTS}
