# tests/test_l1_sample_size.py
"""**L1 样本量**:每个统计量必须带自己的有效样本量(G3 与报告都读它)。

## 为什么需要

`evidence_gate` 的 G3 判"样本够不够",而**报告里显示的 `n_valid`** 也是同一个数。它的
来源是 `research_mapper._indicator_sample_size` ⟹ `feature_engine` 的
`<基名>_sample_size`。取不到时,调用方**退回模态行数**(`<模态>__n_rows`)——
而模态行数是"落盘了多少行",**不是**"这个统计量用了几个观测"。那是个**反保守**的兜底:
它会把证据说多。

实测的活标本(2026-10-04,修之前):

* 场 `20260927_203322_6d57`:`gaze_direction_x` 只有 **12/200** 行非空,而报告印的是
  **`n_valid=200`** ⟹ `logical_thinking` 拿到 **99.05 分、「卓越」**;
* 全语料 53 场里,修完是 **3 场判定变化**(全部是"假过门被拿下")、
  **33 场只是 `n_valid` 被修正**(修正幅度可达 4 倍,如 `253a` 的
  `au23_lip_compression` **240→61**)。

两处成因都修了:
1. `_extract_numeric_stats` 以前**只在短序列**那一支产出 `_sample_size`(长序列只出
   `_trend`)⟹ 长序列的指标没有自己的样本量;现在**一律产出**。
2. 「3. Gaze 方向统计」那块(`face_<col>_mean`/`_std`)以及另外三处直接赋值
   **完全不产出**;现在都产出。
3. `_indicator_sample_size` 以前只认 `<基名>_mean` 的兄弟键;槽消费的是 `_std` 时
   (如 `gaze_direction_x_std`)查不到同列已登记的样本量 ⟹ 现在**剥掉统计量后缀再查一次**。

## 本文件怎么测(不依赖 `data/`)

用**合成 df**:把某个 gaze 列刻意做成"100 行里只有 5 行非空",直接跑真引擎,断言
样本量 = **5**(而不是模态行数 100)。红法:把上面三处任一处改回去 ⟹ 对应断言红。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from report_frontend.feature_engine import PsychologicalFeatureEngine
from report_frontend.research_mapper import _indicator_sample_size

#: 合成场的行数;稀疏列只在前 5 行有值。
_N = 100
_SPARSE = 5


def _face_df() -> pd.DataFrame:
    """最小复刻真 face 日志的**形状**:一个命中关键词表的列 + 一个稀疏 gaze 列。

    ⚠️ `focus_score` 必须带:少了它 `_extract_face_features` 会走进末尾的
    「全量数值扫描」兜底分支 ⟹ 测的是另一条路(与 `test_evidence_gaps_tell_the_truth`
    里那条同样的理由)。
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "focus_score": [0.3] * _N,
        "au4_frown": rng.uniform(0.2, 0.4, _N),              # 稠密:100 行全有值
    })
    for col in ("gaze_direction_x", "left_iris_x", "right_iris_x",
                "left_iris_y", "right_iris_y", "gaze_deviation"):
        s = pd.Series([np.nan] * _N)
        s.iloc[:_SPARSE] = rng.uniform(-0.05, 0.05, _SPARSE)
        df[col] = s
    return df


@pytest.fixture()
def feats() -> dict:
    """跑**真引擎**(不是复刻它的算法)。"""
    out = PsychologicalFeatureEngine({"face": _face_df()}).extract_all_features()
    return out["face"]


def _flat(feats: dict) -> dict:
    return {k if k.startswith("face") else f"face_{k}": v for k, v in feats.items()}


def test_the_sparse_gaze_column_carries_its_own_sample_size(feats):
    """★ 回归:稀疏 gaze 列必须带**自己的**样本量 = 真非空数(5),不是模态行数(100)。

    红法(实测):把「3. Gaze 方向统计」里的 `features[f"face_{col}_sample_size"] = ...`
    删掉 ⟹ 本条红(`KeyError`),且下一条(`_indicator_sample_size`)也红。
    """
    assert feats["face_gaze_direction_x_sample_size"] == _SPARSE, (
        f"gaze 列的样本量是 {feats.get('face_gaze_direction_x_sample_size')},"
        f"应为真实的非空行数 {_SPARSE}")
    assert feats["_n_rows"] == _N, (
        "模态行数本身没问题 —— 问题在于**拿它当指标的样本量**")


def test_indicator_sample_size_finds_it_through_a_statistic_suffix(feats):
    """★ 槽消费的是 `_std` 时,也要能查到**同列**已登记的样本量。

    活标本:槽「视线稳定性」取的键就是 `face_gaze_direction_x_std`。修之前
    `_indicator_sample_size` 只认 `<基名>_mean` 的兄弟键 ⟹ 查到 `None` ⟹
    G3 与报告双双退到模态行数 ⟹ 报告印出 `n_valid=200`(真值 12)。
    """
    flat = _flat(feats)
    assert _indicator_sample_size("face_gaze_direction_x_std", flat) == _SPARSE
    assert _indicator_sample_size("face_gaze_direction_x_mean", flat) == _SPARSE


def test_long_series_also_carries_a_sample_size(feats):
    """★ 长序列(≥20)也必须带样本量 —— 以前只有短序列带(长序列只出 `_trend`)。

    红法(实测):把 `_extract_numeric_stats` 里那行 `_sample_size` 挪回 `else` 分支
    ⟹ 本条红(`n_chars`/`au4_frown` 这类稠密列查不到)。
    """
    flat = _flat(feats)
    assert _indicator_sample_size("face_au4_frown_mean", flat) == _N


@pytest.mark.parametrize("suffix", ["_mean", "_std", "_min", "_max", "_sum", "_trend"])
def test_every_statistic_suffix_resolves_to_the_same_sample_size(suffix):
    """一个源列的**全部**统计量共用一个样本量 —— 它们出自同一个 `dropna()` 序列。

    这条钉的是查法的**彻底性**:只补 `_mean`/`_std` 两种后缀,将来某个槽消费
    `_min`/`_trend` 就又静默退回模态行数了。
    """
    flat = {"face_x_mean": 1.0, "face_x_std": 0.1, "face_x_min": 0.0,
            "face_x_max": 2.0, "face_x_sum": 10.0, "face_x_trend": 0.2,
            "face_x_sample_size": 7}
    assert _indicator_sample_size(f"face_x{suffix}", flat) == 7


def test_the_fallback_would_have_been_wrong_here(feats):
    """把"退回模态行数"的后果**写进断言**,免得将来有人以为那个兜底无害。

    稀疏列的模态行数是 100,真样本量是 5 —— 差 **20 倍**,而门槛(pitch/au/gaze 族)
    是 10~30 ⟹ 20 倍的差别**正好横跨门槛**,能把一个样本不足的指标送进报告。
    """
    flat = _flat(feats)
    own = _indicator_sample_size("face_gaze_direction_x_std", flat)
    modality_rows = flat["face__n_rows"]
    assert own == _SPARSE and modality_rows == _N
    assert modality_rows >= 10 > own, (
        "本测试的前提是「模态行数跨过门槛而真样本量没有」—— 合成数据的形状变了?")
