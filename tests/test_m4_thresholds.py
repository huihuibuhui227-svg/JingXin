# tests/test_m4_thresholds.py
"""**τ 从哪来**的守卫。

## 钉的是什么

1. **自比基线 = 中位数 ± 3·MAD**,**不是**分位。这条最要紧:分位阈值按构造恒给出
   固定占比(超本场 p90 的占比永远是 10%),那是自证的数字;中位数+3·MAD 不是分位,
   占比随分布尾巴厚薄变化。
2. **方向是阈值的一部分** —— `au4_frown` / `au23_lip_compression` 是**越小越皱/越压**,
   写成 `>` 就量反了(这是修掉的真 bug,不是口味问题)。
3. **没登记 / 数据太少 ⟹ 不给 τ**,上游照旧报 `uncalibrated` —— 不猜。
4. **声明常量必须写明"未标定"**,不许让人以为它是标定值。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from report_frontend import l1_thresholds as lt


# ── 1. 自比基线不是分位 ─────────────────────────────────────────────────

def test_self_baseline_is_median_plus_three_mad_not_a_quantile():
    """★ τ 必须**恰好等于** 中位数 + 3·MAD。

    红法:把 `resolve` 里的 `med + k * mad` 换成 `v.quantile(0.9)` ⟹ 本条红。
    两者的区别是本质的:分位阈值让"超阈占比"恒为 10%,而中位数+3·MAD 的占比
    取决于分布尾巴 —— 前者是自证,后者是测量。
    """
    v = pd.Series([1.0] * 18 + [5.0, 9.0])          # 中位数 1.0,MAD 0.0
    tau = lt.resolve("gaze_direction_x", v)
    assert tau is not None and tau.value == pytest.approx(1.0)

    v2 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    med, mad = 5.5, 2.5
    tau2 = lt.resolve("gaze_direction_x", v2)
    assert tau2.value == pytest.approx(med + 3 * mad)   # |x| 的中位数 + 3·MAD


def test_self_baseline_ratio_is_not_fixed_by_construction():
    """不同尾巴 ⟹ 不同的超阈占比。若换成 p90,两个分布都会给出 0.10。

    红法:改成分位阈值 ⟹ 两条占比都变 0.1,本条红。
    """
    rng = np.random.RandomState(0)
    thin = pd.Series(rng.normal(0, 1, 2000))                      # 尾巴薄
    fat = pd.Series(rng.standard_t(2, 2000))                     # 尾巴厚
    r_thin = lt.resolve("head_yaw", thin).mask(thin).mean()
    r_fat = lt.resolve("head_yaw", fat).mask(fat).mean()
    assert abs(r_thin - r_fat) > 0.01, (r_thin, r_fat)


# ── 2. 方向是阈值的一部分 ───────────────────────────────────────────────

def test_frown_and_lip_compression_point_downward():
    """★ `au4_frown` / `au23_lip_compression` 的方向是 `<`(越小越皱/越压)。

    实测出处:
      · `au4_frown` = `dist(lm[107], lm[336]) ÷ 眼距` —— **眉内距 ÷ 眼距,越小越皱**;
      · `au23_lip_compression` = 唇红厚 ÷ 面高,旧公式带 `1−`(换口径时方向翻转,
        阈值没重登记,`au23 > 0.2` 现在是"结构性不可达")。

    红法:把方向改回 `>` ⟹ 本条红,且报告会开始说"眉展得越开越紧张"。
    """
    for col in ("au4_frown", "au23_lip_compression"):
        tau = lt.resolve(col, pd.Series([0.1, 0.2, 0.3, 0.4, 0.5]))
        assert tau.direction == "<", col
    # 语义自检:低于阈值的帧才被判为"处于构念状态"。
    # 夹具用**真实形态**(大量帧聚在基线附近、少数帧明显偏离)—— 实测
    # `au4_frown` 的 p10→p90 只有 0.038 宽,而中位数跨场只差 0.047。
    # 注意:均匀分布下 `中位数−3·MAD` 会落到值域之外(均匀分布的 MAD 约 0.25×极差),
    # 那时占比恒为 0 —— 那是这条规则的正确行为,不是 bug。
    v = pd.Series([0.30] * 50 + [0.20, 0.18, 0.15])
    tau = lt.resolve("au4_frown", v)
    assert tau.value == pytest.approx(0.30)          # MAD = 0 ⟹ τ = 中位数
    assert tau.mask(v).sum() == 3                    # 只有那 3 帧被判为"更皱"


def test_smile_and_aversion_point_upward():
    """`au12`(越大越笑)与 `|视线|` / `|头部偏转|`(越大越偏)方向都是 `>`。"""
    for col in ("au12_smile", "gaze_direction_x", "head_yaw"):
        assert lt.resolve(col, pd.Series([1.0] * 10)).direction == ">", col


def test_abs_transform_applies_before_thresholding():
    """`gaze_direction_x` / `head_yaw` 先取绝对值 —— 它们的符号跨场会翻。

    实测:视线中位数 −0.109~+0.922(19 正 / 22 负),头部 −0.833~+0.389。
    不取绝对值的话,"偏左"和"偏右"会互相抵消。
    """
    tau = lt.resolve("gaze_direction_x", pd.Series([-0.5, 0.5, -0.5, 0.5, -0.5]))
    assert tau.transform == "abs"
    assert tau.value == pytest.approx(0.5)          # |x| 全为 0.5,MAD=0


# ── 3. 拿不到就不给 ─────────────────────────────────────────────────────

def test_unregistered_column_gets_no_threshold():
    """没登记的列返回 `None` ⟹ 上游报 `uncalibrated`,不猜一个数出来。"""
    assert lt.resolve("au9_nose_wrinkle", pd.Series([1.0] * 10)) is None


def test_too_few_frames_gets_no_threshold():
    """少于 5 帧算不出稳健基线 ⟹ `None`。红法:把阈值改成 2 ⟹ 本条红。"""
    assert lt.resolve("au4_frown", pd.Series([0.1, 0.2, 0.3])) is None
    assert lt.resolve("au4_frown", None) is None


# ── 4. 声明常量必须写明未标定 ───────────────────────────────────────────

def test_declared_constant_states_it_is_not_calibrated():
    """★ 声明常量 0.263 的出处必须写清,且**必须**说明"未标定"。

    红法:把 provenance 里的"未标定"删掉 ⟹ 本条红。本仓没有标定集(M5 卡许可+伦理),
    把 p75 说成标定值就是伪造出处。
    """
    tau = lt.resolve("au12_smile", None)
    assert tau.strategy == lt.STRATEGY_DECLARED
    assert tau.value == pytest.approx(lt.AU12_SMILE_TAU)
    assert "未标定" in tau.provenance
    assert "p75" in tau.provenance


def test_self_baseline_provenance_states_the_measured_reason():
    """自比基线的出处必须写明实测理由(场间基线差),不能只说"经验值"。"""
    tau = lt.resolve("au4_frown", pd.Series([0.1, 0.2, 0.3, 0.4, 0.5]))
    assert tau.strategy == lt.STRATEGY_SELF
    assert "MAD" in tau.provenance and "越小越皱" in tau.provenance


# ── 5. 缺失不当阴性 ─────────────────────────────────────────────────────

def test_nan_frames_are_not_counted_as_negative():
    """`NaN` 帧 `mask` 为 `False`,但**也不该**被算进分母 —— 由调用方用非空数当分母。

    这里只钉住 `mask` 本身:NaN 不许被当成"处于构念状态"。
    """
    tau = lt.resolve("au4_frown", pd.Series([0.30] * 50 + [0.15] * 3))
    assert tau.value == pytest.approx(0.30)
    v = pd.Series([0.10, np.nan, 0.25, 0.35])
    assert tau.mask(v).tolist() == [True, False, True, False]


def test_mask_applies_the_abs_transform_too():
    """★ 绝对值必须在 **mask** 里生效,不只是在解析 τ 时生效。

    红法:把 `Tau.mask` 里的 `v = v.abs()` 删掉 ⟹ 本条红。
    实测背景:`gaze_direction_x` 的场间中位数从 −0.109 到 +0.922(19 正 / 22 负),
    `head_yaw` 从 −0.833 到 +0.389(符号会翻)。不取绝对值,"偏左"与"偏右"
    在阈值比较里会落到不同侧,同一个偏离量被判成两种状态。
    """
    base = pd.Series([0.10] * 50 + [0.50] * 3)       # |x| 的基线 0.10,离群 0.50
    tau = lt.resolve("gaze_direction_x", base)
    assert tau.transform == "abs" and tau.value == pytest.approx(0.10)

    # 一正一负、绝对值相同 ⟹ 必须**同样**被判为偏离
    v = pd.Series([0.05, -0.05, 0.80, -0.80])
    assert tau.mask(v).tolist() == [False, False, True, True]
