# report_frontend/l1_gesture.py
"""**G 族(手势)4 个指标 + 3 个协变量**。

## 与 F 族的结构差别

手势日志是**帧级**的(一行一帧),所以 G 族和 F 族同构:按回答窗口切、逐帧算。
但手势日志的**有效列少得多**。实测(49 场)的列可用性:

```
可用(236/236 行):  left/right_wrist_jitter、left/right_elbow_jitter、
                    left/right_shoulder_jitter、torso_jitter、head_jitter、
                    shoulder_width、shrug_level
全空(0 行):        left/right_elbow_angle、left/right_shoulder_angle、torso_angle
几乎全空:          left_hand_*(4/236)、right_hand_*(2/236)、hand_visible_*(同上)
```

⟹ 手部指标(`fist_status` / `spread` / 指节角度)基本没有样本,不可用;
躯干与肩可用。这与"取景只拍到上半身、手经常在画面外"是一致的。

## 两个坑(都已实测)

**坑 1:`is_valid` 恒为 False。** 49 场 50 个日志文件,**每一行**都是 `is_valid=False`,
`used_features="none"`,而 `torso_jitter` 等列是满的。也就是说这个标志位**不代表
"有没有姿态数据"**,它代表"手势情绪推断有没有拿到有效输入"—— 而后者结构性失败
(`emotion_state` 恒 `中性`、`feedback` 恒同一句、`used_features` 恒 `none`,
三者跨 49 场**零变化**)。

⟹ **`pose_valid_ratio` 绝不能取 `is_valid` 的均值**,那会恒为 0,
把"有满帧的躯干数据"说成"一帧有效姿态都没有"。
本模块改取**真正被用的那几列的非空占比**,含义明确且可复核。

**坑 2:`hand_visible_*` 的空格不是 0。** 写入方 `_hand_visible_cells` 明确写:
"没有依据时写**空串**,永不写 `0`" —— `0` 意为"确定没有这只手",空串意为"不知道"。
⟹ `hand_visible_ratio_l/r` 只能算作**下界**(`==1` 的帧 ÷ 总帧),不能算作"可见率"。
本模块在 `formula` 里写明"下界"。

## jitter 的量纲

`left_wrist_jitter` 等**已经是"每秒、且除以肩宽中位数"**的量
(`shoulder_analyzer.py` 的 `_calculate_jitter(..., divide_by_shoulder=True)`)。
所以 `gesture_activity_per_sec` **不再乘除任何东西**,直接逐帧求和取均值 ——
再除一次肩宽就是重复归一化(会把取景远近的影响平方掉)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from report_frontend import l1_thresholds
from report_frontend.event_windows import WindowSet
from report_frontend.l1_indicators import (
    GAP_MODALITY_MISSING, GAP_NO_SAMPLES, GAP_TIME_UNUSABLE, GAP_UNCALIBRATED,
    L1Indicator, measured, unavailable,
)

#: 参与"运动活跃度"的 jitter 列(腕 / 肘 / 肩,按规格 §5.3 G1)。
ACTIVITY_COLS = ("left_wrist_jitter", "right_wrist_jitter",
                 "left_elbow_jitter", "right_elbow_jitter",
                 "left_shoulder_jitter", "right_shoulder_jitter")

#: 参与"姿态稳定"的列(躯干 + 肩,按规格 §5.3 G2)。
POSTURE_COLS = ("torso_jitter", "left_shoulder_jitter", "right_shoulder_jitter")

#: 一个"运动段 / 位移段"至少要持续多久。**按秒定义,不按帧** ——
#: 实测 13 场 A 档的手势日志全是 **1.000 Hz**(p10=p90=1000 ms),
#: 所以 2 秒 = 2 帧;将来若换成 15 fps 采集,同一条定义会自动变成 30 帧。
#:
#: 为什么是 2 秒:1 Hz 下 1 帧就是 1 秒,**单帧超基线无法与一次检测抖动区分**
#: (相邻帧之间没有中间态可看)。要求连续 ≥2 帧(≥2 s)才叫一段。
#: 这不是标定值,是**采样率的下限推论**。
MIN_EVENT_SEC = 2.0

#: `pose_valid_ratio` 的判据列 —— **不是** `is_valid`(见模块 docstring 坑 1)。
POSE_COLS = ("torso_jitter", "left_shoulder_jitter", "right_shoulder_jitter",
             "shoulder_width")

_SPECS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "gesture_activity_per_sec": ("G1_activity", "Σ(腕/肘/肩 jitter) 逐帧均值",
                                 ACTIVITY_COLS),
    "gesture_burst_rate_per_min_vs_baseline": (
        "G1_activity",
        "超本人基线(腕/肘/肩 jitter 逐帧和的中位数+3·MAD)的连续段(≥2 s)次数 ÷ 分钟",
        ACTIVITY_COLS),
    "posture_stability": ("G2_posture", "Σ(躯干/肩 jitter) 逐帧均值(越小越稳)",
                          POSTURE_COLS),
    "postural_shift_rate_per_min_vs_baseline": (
        "G2_posture",
        "torso_jitter 超本人基线(中位数+3·MAD)的连续段(≥2 s)次数 ÷ 分钟",
        ("torso_jitter",)),
    "hand_visible_ratio_l": ("G3_visibility", "hand_visible_left==1 的帧占比(**下界**)",
                             ("hand_visible_left",)),
    "hand_visible_ratio_r": ("G3_visibility", "hand_visible_right==1 的帧占比(**下界**)",
                             ("hand_visible_right",)),
    "pose_valid_ratio": ("G3_visibility", "躯干/肩/肩宽 四列同时非空的帧占比",
                         POSE_COLS),
}

ORDER = tuple(_SPECS)


def compute_gesture_indicators(gesture: pd.DataFrame | None,
                               windows: WindowSet,
                               t_sec: pd.Series | None,
                               thresholds: dict[str, int] | None = None) -> list[L1Indicator]:
    """逐窗口算 G 族 7 条(4 指标 + 3 协变量)。"""
    if gesture is None:
        return _all_unavailable(windows, GAP_MODALITY_MISSING)
    if gesture.empty:
        return _all_unavailable(windows, GAP_NO_SAMPLES)
    if t_sec is None or len(gesture) != len(t_sec):
        return _all_unavailable(windows, GAP_TIME_UNUSABLE)

    thr = thresholds or {}
    t = t_sec.to_numpy(dtype=float)
    out: list[L1Indicator] = []
    for w in windows.windows:
        mask = (~np.isnan(t)) & (t >= w.t_start) & (t < w.t_end)
        out.extend(_for_window(gesture.loc[mask], f"answer:{w.index}", thr))
    if not windows.windows:
        out.extend(_for_window(gesture, "session", thr))
    return out


def _num(df: pd.DataFrame, col: str) -> np.ndarray | None:
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _frame_sec(sub: pd.DataFrame) -> float | None:
    """从 `timestamp` 列取帧间隔(秒)。取不到 ⟹ `None`(不猜成 1.0)。

    实测 13 场 A 档全是 1.000 s;但**不写死** —— 采集帧率是可以变的,
    写死会让将来换 15 fps 时"2 秒段"悄悄变成"2 帧段"。
    """
    if "timestamp" not in sub.columns:
        return None
    t = pd.to_numeric(sub["timestamp"], errors="coerce").to_numpy(dtype=float)
    t = t[np.isfinite(t)]
    if t.size < 2:
        return None
    # ★ 先排序再差分。踩过的坑:直接差分后丢掉负值,一处乱序会让
    #   `[0,1,3,2,4]` 得到 2.0(应 1.0)⟹ `min_frames` 由 2 变 1、时长翻倍、率减半。
    d = np.diff(np.sort(t))
    d = d[d > 0]
    return float(np.median(d)) if d.size else None


def _runs(mask: np.ndarray, min_frames: int) -> int:
    """连续为真的段里,长度 ≥ `min_frames` 的有几段。"""
    out = 0
    run = 0
    for v in mask:
        if v:
            run += 1
        else:
            if run >= min_frames:
                out += 1
            run = 0
    if run >= min_frames:
        out += 1
    return out


def _for_window(sub: pd.DataFrame, window: str, thr: dict[str, int]) -> list[L1Indicator]:
    n_total = len(sub)
    if n_total == 0:
        return [_un(n, GAP_NO_SAMPLES, window) for n in ORDER]

    got: dict[str, L1Indicator] = {}

    # ── G1 运动活跃度 ────────────────────────────────────────────────────
    cols = [a for a in (_num(sub, c) for c in ACTIVITY_COLS) if a is not None]
    if cols:
        # 逐帧求和(缺的列当 0 加),再对"至少有一列有值"的帧取均值。
        #
        # ⚠️ 变量名带 `act_` 前缀是**必须的**。这里原来叫 `any_val`/`total`,
        #   而下面 G2 段又用**同样的名字**装姿态列的和 —— 于是 G2 把 G1 的
        #   结果覆盖掉了,后面算"运动段"时拿到的是**姿态信号**,
        #   而 `required_l0` 对外仍报活动六列。实测 9/13 场的段数是错的
        #   (最多差 3 倍)。这类覆盖不会报错、不会告警,只会悄悄换掉分子。
        act_any = np.zeros(n_total, dtype=bool)
        act_tot = np.zeros(n_total)
        for a in cols:
            ok = ~np.isnan(a)
            act_any |= ok
            act_tot = act_tot + np.nan_to_num(a)
        n_ok = int(act_any.sum())
        if n_ok >= 2:
            got["gesture_activity_per_sec"] = measured(
                "gesture_activity_per_sec", "G1_activity",
                _SPECS["gesture_activity_per_sec"][1], ACTIVITY_COLS,
                window, "1/秒(已除肩宽)",
                value=float(act_tot[act_any].mean()),
                n_valid=n_ok, n_total=n_total,
                threshold=thr.get("head_pose", 30))

    # ── G2 姿态稳定 ──────────────────────────────────────────────────────
    pcols = [a for a in (_num(sub, c) for c in POSTURE_COLS) if a is not None]
    if pcols:
        pos_any = np.zeros(n_total, dtype=bool)
        pos_tot = np.zeros(n_total)
        for a in pcols:
            ok = ~np.isnan(a)
            pos_any |= ok
            pos_tot = pos_tot + np.nan_to_num(a)
        n_ok = int(pos_any.sum())
        if n_ok >= 2:
            got["posture_stability"] = measured(
                "posture_stability", "G2_posture", _SPECS["posture_stability"][1],
                POSTURE_COLS, window, "1/秒(已除肩宽,越小越稳)",
                value=float(pos_tot[pos_any].mean()),
                n_valid=n_ok, n_total=n_total,
                threshold=thr.get("head_pose", 30))

    # ── 两条率量:τ 用**自比基线**,不是视频内分位 ──────────────────────
    #
    # ⚠️ 这里原来写着"用视频内分位当 τ 是明令禁止的自指基准 ⟹ 报 uncalibrated"。
    #   那条禁令针对的是"超过本场 p90 的占比按构造恒等于 10%"那类**自证数字**;
    #   而 2026-10-18 的 τ 决定用的是**中位数 + 3·MAD**(稳健位置+尺度),不是分位,
    #   它给出的是一个"比本人平时明显更活跃"的绝对水平线,不会自证。
    #   两条指标因此**改名**加 `_vs_baseline` —— 构念从"运动起始"变成
    #   "超出本人常态的连续段",名字必须跟着变。
    frame_sec = _frame_sec(sub)
    min_frames = (max(1, int(np.ceil(MIN_EVENT_SEC / frame_sec)))
                  if frame_sec else None)
    duration_min = (n_total * frame_sec / 60.0) if frame_sec else None

    if frame_sec is None or not duration_min:
        for name in ("gesture_burst_rate_per_min_vs_baseline",
                     "postural_shift_rate_per_min_vs_baseline"):
            got[name] = _un(name, GAP_NO_SAMPLES, window)
    else:
        act_sig = None
        if cols:
            act_sig = np.where(act_any, act_tot, np.nan)
        for name, sig, col_name in (
                ("gesture_burst_rate_per_min_vs_baseline", act_sig, "gesture_activity"),
                ("postural_shift_rate_per_min_vs_baseline",
                 _num(sub, "torso_jitter"), "torso_jitter")):
            tau = l1_thresholds.self_tau(
                sig, col_name, direction=">",
                provenance="自比基线(本人中位数+3·MAD);1 Hz 采样 ⟹ 最短段 2 s")
            if tau is None:
                got[name] = _un(name, GAP_UNCALIBRATED, window)
                continue
            episodes = _runs(tau.mask(sig), min_frames)
            # ★ `n_valid` 报**段数**,不报帧数。
            #
            # 一度报帧数(理由是"绕开 0 段时 n_valid=0 违反不变量"),但那是**倒果为因**:
            # 帧数会让 163 帧 / 2 段的一场拿到「高」置信 —— 一条只有 2 次事件的率
            # 被判成"证据充分"。率的证据单位就是"观察到几次",不是"看了多少帧"。
            #
            # 0 段取 `max(episodes, 1)`:"整场一次都没超基线"是一次**完成的观测**
            # (看完了,结论是没有),不是"没观测" ⟹ 置信「低」而不是「无」。
            #
            # ⚠️ 于是 `n_valid`(段)与 `n_total`(帧)**单位不同**,
            #   `n_valid/n_total` 这个比值对率量**没有意义**;置信实际只由段数决定
            #   (「高」要求覆盖率 ≥0.8,本族永远不满足)。这是有意的:
            #   与其编一个"可能有多少段"的分母,不如把这一点写明。
            got[name] = measured(
                name, _SPECS[name][0], _SPECS[name][1], _SPECS[name][2],
                window, "次/分",
                value=episodes / duration_min,
                n_valid=max(episodes, 1), n_total=n_total,
                threshold=thr.get("head_pose", 30))

    # ── G3 协变量 ────────────────────────────────────────────────────────
    for side, col in (("l", "hand_visible_left"), ("r", "hand_visible_right")):
        name = f"hand_visible_ratio_{side}"
        if col in sub.columns:
            # ⚠️ 空串 ≠ 0(见模块 docstring 坑 2)⟹ 只能算下界。
            #
            # ★ `n_valid` 必须报**真的带了观测的帧数**,不是 `n_total`。
            #
            # 2026-10-24 实测(生产 500):场 `20260928_202928_3d8f` 的手势日志
            # 387 行里,`torso_jitter` / `shoulder_width` **0 个非空**、
            # `hand_visible_left` **只有 1 行有值**。旧写法把这个 1/387 的读数
            # 报成 `n_valid=387`(等于宣称 387 帧都看过),于是它活过了
            # `l1_quality.validate()` 的守卫「模态 `usable_ratio=0` ⟹ 不许有值」,
            # 整个 `/api/report/structured` 抛 `ValueError`。
            #
            # 病根与 `pose_valid_ratio` 那条一样:**把证据说多**。
            # 空串的意思是"不知道"(坑 2),不是一次观测 —— 它只能进分母,
            # 不能进 `n_valid`。所以下界的值照旧按 `n_total` 算(那才是下界),
            # 但"这条结论建立在几次观测上"要如实报成非空帧数。
            obs = pd.to_numeric(sub[col], errors="coerce").notna()
            n_obs = int(obs.sum())
            if n_obs < 2:
                # 与同函数其余指标同一个门槛:两条观测才算"算过一次"。
                # 1 行有值 ⟹ 这一列在本窗口内**基本没被写过**,报缺失而不是报下界。
                got[name] = _un(name, GAP_NO_SAMPLES, window)
            else:
                got[name] = measured(
                    name, "G3_visibility", _SPECS[name][1], (col,), window,
                    "无量纲(下界)",
                    value=float(pd.to_numeric(sub[col], errors="coerce")
                                .fillna(0).astype(float).gt(0).sum()) / n_total,
                    n_valid=n_obs, n_total=n_total,
                    threshold=thr.get("head_pose", 30))
        else:
            got[name] = _un(name, GAP_NO_SAMPLES, window)

    present = [_num(sub, c) is not None for c in POSE_COLS]
    if all(present):
        ok = np.ones(n_total, dtype=bool)
        for c in POSE_COLS:
            ok &= ~np.isnan(_num(sub, c))
        n_ok = int(ok.sum())
        if n_ok >= 2:
            got["pose_valid_ratio"] = measured(
                "pose_valid_ratio", "G3_visibility", _SPECS["pose_valid_ratio"][1],
                POSE_COLS, window, "无量纲",
                value=n_ok / n_total, n_valid=n_ok, n_total=n_total,
                threshold=thr.get("head_pose", 30))
        else:
            got["pose_valid_ratio"] = _un("pose_valid_ratio", GAP_NO_SAMPLES, window)
    else:
        got["pose_valid_ratio"] = _un("pose_valid_ratio", GAP_NO_SAMPLES, window)

    for name in ORDER:
        if name not in got:
            got[name] = _un(name, GAP_NO_SAMPLES, window)
    return [got[n] for n in ORDER]


#: 没有窗口时的窗口标签。与 `l1_face` 用的是同一个字面量(`"session"`),
#: 也是 `event_windows.SESSION_GRANULARITY` 的值 —— 三处必须一致。
_SESSION = "session"


def _un(name: str, gap: str, window: str = _SESSION) -> L1Indicator:
    """一条"不可出"的 G 族指标。

    ⚠️ `window` **必须由调用方给**。这里原先写死 `"answer:0"` —— 后果是:
    整场口径(没有台账,只有一个 `session` 窗口)下,G 族 7 条会带着
    `answer:0` 的标签进报告,而同一份报告里 F 族 16 条带的是 `session`。
    读者看到同一场里两个族"窗口不同",会以为逐题窗口生效了 ——
    实际那是没算出来的占位值,连一题都没对上。

    实测 `20260926_153202_2b11`:`{'answer:-1': 16, 'answer:0': 7, 'session': 15}`,
    G 族 7 条全在 `answer:0`。这正是"标签比数先撒谎"。
    """
    family, formula, cols = _SPECS[name]
    return unavailable(name, family, formula, tuple(cols), window, "", gap)


def _all_unavailable(windows: WindowSet, gap: str) -> list[L1Indicator]:
    """整族不可出(没日志/没样本/时间轴不可信)。

    窗口标签按**本场的实际口径**给:有窗口就逐题,没窗口就是会话级。
    """
    labels = [f"answer:{w.index}" for w in windows.windows] or [_SESSION]
    return [_un(n, gap, lab) for lab in labels for n in ORDER]


def indicator_names() -> tuple[str, ...]:
    """G 族 4 个指标 + 3 个协变量。"""
    return ORDER
