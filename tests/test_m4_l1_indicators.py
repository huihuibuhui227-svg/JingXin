# tests/test_m4_l1_indicators.py
"""**M4.2 指标结构** + **M4.3 F 族**的守卫。

## 钉的是什么

1. **四档 `confidence` 都必须可达。** 2026-10-04 之前只有"高/中""中/低"两对可用,
   `"高"` 与 `"无"` 都不可达 ⟹ 零输入也报"中"。那是当时那份报告失真的根因之一。
2. **`gap` 与"有没有值"必须一致。** 登记了"算不出来"却留着一个数,等于继续骗人。
3. **`n_valid` 不许超过 `n_total`** —— 把证据说多正是本轮要消灭的。
4. **`blink_rate_per_min` 必须由上升沿重数**,不能拿日志里同名列(`blink_rate_per_min`)
   求均值 —— 那一列是**滑动 60 秒窗口**量,不是全窗口速率。
5. **卡标定的一律 `uncalibrated`**,不许用视频内分位顶替。

## 不依赖 `data/`

全部合成 df,直接跑真函数。
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from report_frontend import l1_face, l1_indicators as li
from report_frontend.event_windows import AnswerWindow, WindowSet


def _windows(*bounds: tuple[float, float]) -> WindowSet:
    return WindowSet(session_id="s", granularity="answer",
                     windows=[AnswerWindow(index=i, t_start=a, t_end=b)
                              for i, (a, b) in enumerate(bounds)],
                     n_questions=len(bounds))


# ── 1. 四档可达 ──────────────────────────────────────────────────────────

def test_all_four_confidence_levels_are_reachable():
    """★ 一档一个用例。红法:把 `confidence_for` 的高档条件删掉 ⟹ "高" 不可达,本条红。"""
    thr = 20
    assert li.confidence_for(0, 100, thr) == "无"
    assert li.confidence_for(10, 100, thr) == "低", "不到门槛"
    assert li.confidence_for(25, 100, thr) == "中", "过门槛但覆盖率低"
    assert li.confidence_for(90, 100, thr) == "高", "两倍门槛且覆盖率 ≥ 80%"
    # 边界:过两倍门槛但覆盖率不足 80% ⟹ 仍是"中",不是"高"
    assert li.confidence_for(40, 100, thr) == "中"


# ── 2. gap 与值的一致性 ──────────────────────────────────────────────────

def test_gap_with_a_value_is_rejected():
    """红法:把 `__post_init__` 里那段一致性检查删掉 ⟹ 本条红。"""
    with pytest.raises(ValueError, match="必须 value=NaN"):
        li.L1Indicator(name="x", family="F", formula="f", required_l0=("a",),
                       window="answer:0", unit="", value=1.0,
                       n_valid=0, n_total=10, confidence="无",
                       gap=li.GAP_UNCALIBRATED)


def test_value_with_confidence_none_is_rejected():
    with pytest.raises(ValueError, match="二者必须一致"):
        li.L1Indicator(name="x", family="F", formula="f", required_l0=("a",),
                       window="answer:0", unit="", value=1.0,
                       n_valid=10, n_total=10, confidence="无", gap=None)


def test_n_valid_above_n_total_is_rejected():
    """把证据说多 —— 这正是 2026-10-04 那批 16 倍高估的形态。"""
    with pytest.raises(ValueError, match="把证据说多了"):
        li.L1Indicator(name="x", family="F", formula="f", required_l0=("a",),
                       window="answer:0", unit="", value=1.0,
                       n_valid=200, n_total=100, confidence="高", gap=None)


def test_unknown_gap_kind_is_rejected():
    with pytest.raises(ValueError, match="gap 只能是"):
        li.L1Indicator(name="x", family="F", formula="f", required_l0=("a",),
                       window="answer:0", unit="", n_valid=0, n_total=1,
                       confidence="无", gap="随便写点什么")


# ── 3. blink 必须靠上升沿,不能靠同名滑动列 ───────────────────────────────

def test_blink_rate_counts_rising_edges_not_the_sliding_column():
    """★ 日志里的 `blink_rate_per_min` 是"最近 60 秒内的次数",不是全窗口速率。

    构造:120 秒、2 次眨眼(各 1 帧)。上升沿法 = 2 / (120/60) = **1.0 次/分**。
    而滑动列在每个眨眼瞬间 = 1、其余为 0 ⟹ 求均值 ≈ 0.017,**差 60 倍**。
    红法:把上升沿法换成 `df["blink_rate_per_min"].mean()` ⟹ 本条红。
    """
    n = 121
    t = np.arange(n, dtype=float)                 # 每秒一帧,共 120 s
    is_blink = np.zeros(n)
    is_blink[30] = 1.0
    is_blink[90] = 1.0
    sliding = np.zeros(n)
    sliding[30] = 1.0                             # 滑动窗口在那一帧看到 1 次
    sliding[90] = 1.0

    df = pd.DataFrame({"is_blink": is_blink,
                       "blink_rate_per_min": sliding,
                       "gaze_direction_x": np.linspace(0, 1, n),
                       "head_yaw": np.zeros(n), "head_pitch": np.zeros(n)})
    inds = l1_face.compute_face_indicators(
        df, _windows((-1.0, 200.0)), pd.Series(t), thresholds={"blink": 20})
    got = {i.name: i for i in inds if i.window == "answer:0"}

    assert got["blink_rate_per_min"].n_valid == n - 0
    assert abs(got["blink_rate_per_min"].value - 1.0) < 1e-6, (
        f"上升沿法应得 1.0 次/分,实得 {got['blink_rate_per_min'].value}")
    # 若误用同名滑动列,均值会是 2/121 ≈ 0.0165 —— 与 1.0 差 60 倍
    assert got["blink_rate_per_min"].value > 10 * sliding.mean()


def test_first_frame_true_is_not_a_rising_edge():
    """首帧就是眨眼 ⟹ 它前面没有"0",不算一次眨眼。"""
    is_blink = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    t = np.arange(6, dtype=float)
    df = pd.DataFrame({"is_blink": is_blink,
                       "gaze_direction_x": np.linspace(0, 1, 6),
                       "head_yaw": np.zeros(6), "head_pitch": np.zeros(6)})
    inds = l1_face.compute_face_indicators(
        df, _windows((-1.0, 99.0)), pd.Series(t), thresholds={"blink": 2})
    rate = next(i for i in inds if i.name == "blink_rate_per_min").value
    # 只有 0→1(第 4 帧)一次;跨度 5 s ⟹ 1 / (5/60) = 12 次/分
    assert abs(rate - 12.0) < 1e-6, rate


# ── 4. 卡标定的一律不可出 ────────────────────────────────────────────────

def test_tau_indicators_compute_but_only_where_a_threshold_is_declared():
    """★ τ 型指标**现在会出值**,但阈值来源必须写在公式里,且 `au_activity_index` 仍不可出。

    ## 这条测试改写过(2026-10-18),原因要说清

    原版断言"9 条一律 `uncalibrated` + NaN",理由是「不许用视频内分位顶着出数」——
    那个关切**仍然有效**:一个"超过本场 p90 的占比"按构造恒等于 10%,是自证的数字。

    但 2026-10-18 量了全语料 22277 行 / 41 场之后发现,卡住的不是"没有标定集",
    是**跨人基线差**(场间基线差 ≥ 场内动态范围,见 `l1_thresholds`)。所以改成:
    τ 取**本人中位数 ± 3·MAD**(稳健位置+尺度,不是分位),并把指标**改名**
    为 `*_vs_baseline` —— 名字如实说了它量的是"比他自己平时偏离多少"。

    ⟹ 本条改成守住三件事:
      ① 出值的那些,公式串里必须写明阈值来源;
      ② 自比基线的必须带 `_vs_baseline` 后缀(构念变了,名字就得变);
      ③ **只有一条 AU 可解析时,聚合型指标仍不可出** —— 本 fixture 里可信 AU 只有
         `au12_smile` 一条,拿一条算"至少一条激活"等于把它改名叫 `smile_ratio`,
         那是骗人。新构念叫 `au_activation_any_ratio`(旧名 `au_activity_index`
         已因构念退化而废弃,见 `l1_face` 里那段实测)。

    红法:把 `_vs_baseline` 后缀去掉 ⟹ ② 红;把 `len(taus) < 2` 判断删掉 ⟹ ③ 红。
    """
    n = 200
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "au12_smile": rng.rand(n),
        "au4_frown": rng.rand(n),
        "au23_lip_compression": rng.rand(n),
        "gaze_direction_x": rng.rand(n),
        "head_yaw": rng.rand(n),
        "head_pitch": rng.rand(n),
        "is_blink": np.zeros(n),
    })
    inds = l1_face.compute_face_indicators(
        df, _windows((-1.0, 999.0)), pd.Series(np.arange(n, dtype=float)))
    by = {i.name: i for i in inds if i.window == "answer:0"}

    # ① 出值的必须写明阈值来源
    computed = [n_ for n_, i in by.items()
                if i.value is not None and not math.isnan(i.value)]
    assert computed, "一条 τ 型指标都没出值"
    for name in computed:
        if name.startswith(("smile", "brow", "lip", "gaze_aversion", "head_away")):
            formula = by[name].formula
            assert ("3·MAD" in formula or "声明常量" in formula), (name, formula)

    # ② 自比基线的必须带后缀
    for name in ("brow_furrow_ratio_vs_baseline",
                 "lip_compression_ratio_vs_baseline",
                 "gaze_aversion_ratio_vs_baseline",
                 "head_away_ratio_vs_baseline"):
        assert name in by, name

    # ③ 聚合型指标不许拿一条 AU 冒充(本 fixture 只有 au12 一条可信 AU)
    assert "au_activity_index" not in by, "旧名还在输出里 —— 构念变了,名字必须换"
    agg = by["au_activation_any_ratio"]
    assert agg.gap == li.GAP_UNCALIBRATED
    assert agg.value is None or math.isnan(agg.value)


def test_any_activation_is_measurable_where_the_joint_and_was_degenerate():
    """★ 换构念的**理由**本身要被钉住:联合 AND 恒为 0 时,"至少一个"仍有值。

    构造:六条可信 AU 各在一段**互不重叠**的帧上激活(各占 1/6),
    基线是 0.10~0.20 的均匀噪声(所以 MAD>0、τ 有意义),尖峰抬到 1.0。

    * 联合 AND = **0**(没有任何一帧六条同时激活);
    * 至少一个 = **1.0**(每帧都恰好有一条激活)。

    这正是原构念退化的现场:六个各 17% 的事件取交集,乘积趋 0。

    红法:把 `np.logical_or.reduce` 换回 `np.logical_and.reduce` ⟹ 得 0.0,本条红。
    """
    n = 600
    rng = np.random.RandomState(7)
    df = pd.DataFrame({"is_blink": np.zeros(n)})
    for k, col in enumerate(l1_face.TRUSTED_AUS):
        v = rng.uniform(0.10, 0.20, n)
        v[np.arange(n) % 6 == k] = 1.0
        df[col] = v
    inds = l1_face.compute_face_indicators(
        df, _windows((-1.0, 999.0)), pd.Series(np.arange(n, dtype=float)))
    by = {i.name: i for i in inds if i.window == "answer:0"}
    ind = by["au_activation_any_ratio"]
    assert ind.has_value, (ind.gap, ind.n_valid)
    assert ind.value == pytest.approx(1.0, abs=1e-9)



# ── 5. 条数恒定 ──────────────────────────────────────────────────────────

def test_family_always_reports_sixteen_indicators_per_window():
    """★ 不管有没有数据,条数恒定 16 —— 不列出来等于让人以为 F 族不存在。

    红法:在 `_family_for_window` 里把"补成不可出"那段删掉 ⟹ 缺列的窗口条数变少,本条红。
    """
    n = 30
    full = pd.DataFrame({
        "is_blink": np.zeros(n), "gaze_direction_x": np.linspace(0, 1, n),
        "head_yaw": np.zeros(n), "head_pitch": np.zeros(n),
        **{c: np.zeros(n) for c in l1_face.TRUSTED_AUS},
    })
    inds_full = l1_face.compute_face_indicators(
        full, _windows((-1.0, 999.0)), pd.Series(np.arange(n, dtype=float)))
    assert len([i for i in inds_full if i.window == "answer:0"]) == 16

    empty = pd.DataFrame({"is_blink": [np.nan] * n})
    inds_empty = l1_face.compute_face_indicators(
        empty, _windows((-1.0, 999.0)), pd.Series(np.arange(n, dtype=float)))
    got = [i for i in inds_empty if i.window == "answer:0"]
    assert len(got) == 16
    assert all(not i.has_value for i in got), "没数据时不许有一条出值"


def test_no_windows_still_reports_the_family_under_session():
    """戊-15 那 26 场:没有回答窗口,但 16 条仍要以 `session` 名义出现。"""
    ws = WindowSet(session_id="s", granularity="session", windows=[], gap="没有台账")
    inds = l1_face.compute_face_indicators(None, ws, None)
    assert len(inds) == 16
    assert all(i.window == "session" for i in inds)


# ── 6. 行数对不上就不出数 ────────────────────────────────────────────────

def test_row_mismatch_between_log_and_timeline_blocks_the_family():
    """时间轴与日志行数不符 ⟹ 整族不可出,不按序硬配。"""
    df = pd.DataFrame({"is_blink": np.zeros(10), "gaze_direction_x": np.zeros(10)})
    inds = l1_face.compute_face_indicators(
        df, _windows((-1.0, 999.0)), pd.Series(np.arange(9, dtype=float)))
    assert all(not i.has_value for i in inds)


# ── 7. gaze_stability 的极差为 0 分支 ────────────────────────────────────

def test_zero_range_gaze_is_no_samples_not_perfect_stability():
    """极差 0 ⟹ 分母 0。那是"场内心完全不动",不是"稳定度 1.0"。"""
    n = 20
    df = pd.DataFrame({"gaze_direction_x": np.zeros(n), "is_blink": np.zeros(n),
                       "head_yaw": np.zeros(n), "head_pitch": np.zeros(n)})
    inds = l1_face.compute_face_indicators(
        df, _windows((-1.0, 999.0)), pd.Series(np.arange(n, dtype=float)))
    gs = next(i for i in inds if i.name == "gaze_stability")
    assert gs.gap == li.GAP_NO_SAMPLES
    assert not gs.has_value


# ── 8. 整列为 NaN 时不许报 0 ─────────────────────────────────────────────

def test_all_nan_head_columns_report_no_samples_not_zero_motion():
    """★ 实跑 `20260927_195323_35e1` 时被不变量抓到的真 bug。

    列存在但整列 NaN ⟹ `np.nansum` 给 0、除以跨度得 0.0 ⟹ 报"头动 0 弧度/秒"。
    那是把"没观测"说成"没动"。红法:把 `n_ok >= 2` 这个前置条件去掉 ⟹ 本条红
    (`L1Indicator` 会以 "confidence='无' 却有值" 抛错)。
    """
    n = 50
    df = pd.DataFrame({
        "head_yaw": [np.nan] * n,
        "head_pitch": [np.nan] * n,
        "is_blink": np.zeros(n),
        "gaze_direction_x": np.linspace(0, 1, n),
    })
    inds = l1_face.compute_face_indicators(
        df, _windows((-1.0, 999.0)), pd.Series(np.arange(n, dtype=float)))
    hm = next(i for i in inds if i.name == "head_motion_per_sec")
    assert hm.gap == li.GAP_NO_SAMPLES, hm.gap
    assert not hm.has_value, f"整列 NaN 却报出 {hm.value}"


# ── 5. 指标清单本身必须被钉住 ────────────────────────────────────────────

def test_indicator_inventory_matches_the_spec():
    """★ 四族条数合计 = 38(F 16 / G 7 / V 12 / T 3),且名字不许跨族重名。

    ## 为什么需要这一条

    此前**没有任何测试**钉住"一共多少条指标"。规格里写着 38、报告上印着 38,
    但代码里增删一条指标不会让任何测试变红 —— 于是规格与实现可以悄悄漂开
    (本轮就发现规格里还留着 36、F 还写着 17、还有一个代码里根本不存在的
    `tension` 和一个只有 `p50` 却被写成 `p50/p90` 的指标)。

    红法:给任一族加一条指标(例如往 `l1_text.ORDER` 里塞一个名字)⟹ 本条红。
    """
    from report_frontend import l1_gesture, l1_text, l1_voice

    fams = {
        "F": l1_face.indicator_names(),
        "G": l1_gesture.ORDER,
        "V": l1_voice.ORDER,
        "T": l1_text.ORDER,
    }
    counts = {k: len(v) for k, v in fams.items()}
    assert counts == {"F": 16, "G": 7, "V": 12, "T": 3}, counts
    assert sum(counts.values()) == 38, counts

    # 跨族重名会让报告里两行同名、`rowKey` 撞车 —— 前端表格的 key 就是名字。
    allnames = [n for v in fams.values() for n in v]
    assert len(allnames) == len(set(allnames)), (
        "跨族重名:", [n for n in set(allnames) if allnames.count(n) > 1])

    # 名字里不许还留着已废弃的构念名(换构念必须换名,见 l1_face 的实测记录)。
    assert "au_activity_index" not in allnames


def test_any_activation_ratio_cannot_exceed_one_when_observation_is_partial():
    """★ 部分观测时,占比仍必须 ≤1。

    ## 这条测试的由来(一级 bug,潜伏)

    分子数"任一条 AU 激活"的帧(**含**部分观测帧),分母却是
    `~np.isnan(arr).any(axis=1)` —— 那个表达式数的是"**全部** AU 都非空"的帧,
    与公式串说的"至少有一条观测到"**相反**。于是分子可以大于分母,占比 >1。

    构造:au1 全程观测、前 40 帧激活;au6 只在 80~99 帧观测(其中 80~84 激活)。
    * 正确:分子 45(40 + 5)、分母 100(至少一条观测到的帧)⟹ **0.45**;
    * 错的时候:分母只剩"两条都观测到"的 20 帧 ⟹ **2.25**。

    红法:把 `(~np.isnan(arr)).any(axis=1)` 改回 `~np.isnan(arr).any(axis=1)` ⟹ 本条红。
    """
    n = 100
    df = pd.DataFrame({"is_blink": np.zeros(n)})
    au1 = np.full(n, 0.15)
    au1[:40] = 1.0
    df["au1_inner_brow_raise"] = au1
    # au6 只在 80~99 帧有观测:80~84 激活,85~99 是它的基线(τ 由这 20 帧定)。
    au6 = np.full(n, np.nan)
    au6[80:85] = 1.0
    au6[85:100] = 0.15
    df["au6_cheek_raise"] = au6
    for c in l1_face.TRUSTED_AUS:
        if c not in df.columns:
            df[c] = np.full(n, np.nan)

    inds = l1_face.compute_face_indicators(
        df, _windows((-1.0, 999.0)), pd.Series(np.arange(n, dtype=float)))
    ind = {i.name: i for i in inds if i.window == "answer:0"}["au_activation_any_ratio"]
    assert ind.has_value, (ind.gap, ind.n_valid)
    assert ind.value <= 1.0, f"占比 >1:{ind.value}(分子分母口径不一致)"
    assert ind.value == pytest.approx(0.45)
    assert ind.n_valid == 100
